"""Does mask geometry predict malignancy outside ThyroidXL?

    python scripts/run_tn3k.py

This is the external test of the project's headline claim. On ThyroidXL, seven
shape descriptors taken from the segmentation contour reach 0.9602 patient-level
AUC with no pixel read. The standing objection is that this could be an artefact
of one annotator team: if ThyroidXL's annotators outlined nodules they knew to
be malignant differently, the contour leaks the label and the result says
nothing about thyroid contours in general.

DDTI could not settle that, because it ships no malignancy endpoint, so only
contour-to-TI-RADS was testable there and it did not replicate (QWK 0.13
against 0.60). That left the objection open.

TN3K settles it directly: pixel-level masks and benign/malignant labels from a
different centre and different machines. 614 test images, 38.4% malignant.

Two caveats that constrain the reading, both reported rather than smoothed over:

  No patient identifiers. TN3K is distributed per image, so images from one
  patient cannot be kept in the same fold. If the split contains several images
  per patient, cross-validated AUC here is optimistic in a way the ThyroidXL
  figure is not, since that one was grouped on patient.

  Label provenance. TN3K's labels are described as attributed by experienced
  radiologists rather than confirmed by pathology. If they are readings rather
  than outcomes, this measures whether contour encodes a radiologist's
  impression, which is a weaker claim than predicting malignancy.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import controls as C
from src import data as D

NAMES = C.MaskGeometry.names()


def load_labels(path: Path) -> pd.DataFrame:
    # No header row in the release; first line is already data.
    return pd.read_csv(path, header=None, names=["image", "label"])


def collect(mask_dir: Path, labels: pd.DataFrame) -> pd.DataFrame:
    from PIL import Image

    rows, missing = [], 0
    for _, row in labels.iterrows():
        path = mask_dir / str(row["image"])
        if not path.exists():
            missing += 1
            continue
        with Image.open(path) as im:
            mask = np.array(im.convert("L"))
        geom = C.mask_geometry(mask)
        rows.append({
            "image": row["image"],
            "label": int(row["label"]),
            **dict(zip(NAMES, geom.as_array())),
        })
    out = pd.DataFrame(rows)
    out.attrs["missing"] = missing
    return out


def cv_auc(X: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    oof = np.full(len(y), np.nan)
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000)
        ).fit(X[tr], y[tr])
        oof[va] = model.predict_proba(X[va])[:, 1]
    return oof


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tn3k", default="data/tn3k_raw")
    ap.add_argument("--out", default="outputs/tn3k")
    args = ap.parse_args()

    root = Path(args.tn3k)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import roc_auc_score

    from src.metrics import bootstrap_auc_ci

    labels = load_labels(root / "label4test.csv")
    geom = collect(root / "test-mask", labels)
    y = geom["label"].to_numpy(int)
    print(f"  {len(geom)} masks read, {geom.attrs['missing']} missing")
    print(f"  prevalence {y.mean():.4f}  ({int(y.sum())} malignant)")

    print("\n=== geometry -> malignancy on TN3K ===")
    oof = cv_auc(geom[NAMES].to_numpy(float), y)
    auc = float(roc_auc_score(y, oof))
    _, lo, hi = bootstrap_auc_ci(y, oof, n_boot=2000, seed=0)
    print(f"  full geometry   AUC {auc:.4f} [{lo:.4f}, {hi:.4f}]")

    area_oof = cv_auc(geom[["area"]].to_numpy(float), y)
    area_auc = float(roc_auc_score(y, area_oof))
    print(f"  area alone      AUC {area_auc:.4f}")

    # Only the scale-invariant descriptors, which are the ones that survive a
    # change of image framing and dominated the ThyroidXL fit.
    shape_only = ["circularity", "bbox_aspect", "extent"]
    shape_oof = cv_auc(geom[shape_only].to_numpy(float), y)
    shape_auc = float(roc_auc_score(y, shape_oof))
    print(f"  shape only      AUC {shape_auc:.4f}   {shape_only}")

    thyroidxl = 0.9602
    print(f"\n  ThyroidXL (patient level, grouped) AUC {thyroidxl:.4f}")
    print(f"  TN3K      (image level, ungrouped) AUC {auc:.4f}")
    if auc >= 0.80:
        verdict = ("REPLICATES: contour geometry predicts malignancy in an "
                   "independent cohort, so the ThyroidXL result is not an "
                   "artefact of one annotator team.")
    elif auc >= 0.65:
        verdict = ("PARTIALLY REPLICATES: contour carries real but weaker "
                   "signal here; the ThyroidXL figure is inflated by something "
                   "specific to that dataset.")
    else:
        verdict = ("DOES NOT REPLICATE: the ThyroidXL result looks specific to "
                   "its annotation process rather than to thyroid contours.")
    print(f"  {verdict}")

    report = {
        "n": int(len(geom)),
        "prevalence": round(float(y.mean()), 4),
        "auc_full_geometry": round(auc, 4),
        "auc_ci": [round(lo, 4), round(hi, 4)],
        "auc_area_only": round(area_auc, 4),
        "auc_shape_only": round(shape_auc, 4),
        "shape_only_features": shape_only,
        "thyroidxl_reference_auc": thyroidxl,
        "verdict": verdict,
        "caveats": [
            "TN3K ships no patient identifiers, so folds are not grouped on "
            "patient and this AUC is image level; the ThyroidXL figure is "
            "patient level and grouped, so TN3K is the more optimistic of the "
            "two protocols.",
            "TN3K labels are described as radiologist-attributed rather than "
            "pathology-confirmed, so this may measure agreement with a "
            "radiologist's impression rather than with an outcome.",
        ],
    }
    D.save_report(report, out_dir / "tn3k.json")
    print(f"\n  wrote {(out_dir / 'tn3k.json').resolve()}")


if __name__ == "__main__":
    main()
