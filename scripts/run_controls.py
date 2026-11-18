"""Controls A and B: does the segmentation mask leak the label?

    python scripts/run_controls.py --masks data/thyroidxl_masks --data data/thyroidxl

Why this matters. The best published result on ThyroidXL (patient-level AUC
0.9939) feeds a *segmentation-masked* image as its second branch, and RCAF
names "mask-related shortcut analysis" as a gap it set out to address. If mask
shape alone separates benign from malignant, then mask-consuming models are
partly reading an annotation artefact rather than tissue.

Evaluation is patient-level and grouped on patient, like everything else here.
A mask is one per image, so per-image geometry is aggregated to the patient by
taking the largest nodule, which is the one that drives clinical management.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import controls as C
from src import data as D
from src import schema as S
from src.baselines import to_patient_level


def load_mask(path: Path) -> np.ndarray | None:
    from PIL import Image

    try:
        with Image.open(path) as im:
            return np.array(im.convert("L"))
    except Exception:
        return None


def collect_geometry(df: pd.DataFrame, mask_root: Path, split: str,
                     limit: int = 0) -> pd.DataFrame:
    """Per-image mask geometry, joined to patient id and label."""
    rows = []
    missing = 0
    frame = df if not limit else df.head(limit)
    total = len(frame)
    for i, (_, row) in enumerate(frame.iterrows()):
        if i % 2000 == 0 and i:
            print(f"    {i}/{total} masks read", flush=True)
        path = mask_root / split / "masks" / str(row[S.IMAGE_ID])
        mask = load_mask(path)
        if mask is None:
            missing += 1
            continue
        geom = C.mask_geometry(mask)
        rows.append({
            S.PATIENT_ID: row[S.PATIENT_ID],
            S.IMAGE_ID: row[S.IMAGE_ID],
            S.LABEL: int(row[S.LABEL]),
            **dict(zip(C.MaskGeometry.names(), geom.as_array())),
        })
    out = pd.DataFrame(rows)
    out.attrs["missing"] = missing
    return out


def to_patient_geometry(geom: pd.DataFrame) -> pd.DataFrame:
    """Aggregate images to patients by taking the largest-area mask.

    Averaging would blend a suspicious nodule into incidental views; the
    largest lesion is what management follows.
    """
    idx = geom.groupby(S.PATIENT_ID)["area"].idxmax()
    return geom.loc[idx].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--masks", default="data/thyroidxl_masks")
    ap.add_argument("--data", default="data/thyroidxl")
    ap.add_argument("--out", default="outputs/controls")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--limit", type=int, default=0,
                    help="read only N masks per split (smoke test)")
    args = ap.parse_args()

    mask_root = Path(args.masks)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {"masks": str(mask_root.resolve()), "splits": {}}
    per_split: dict[str, tuple] = {}

    for split in args.splits:
        print(f"\n=== {split} ===")
        df = D.load_thyroidxl(Path(args.data), split)
        geom = collect_geometry(df, mask_root, split, limit=args.limit)
        if geom.empty:
            print(f"  no masks read under {mask_root / split / 'masks'}")
            continue
        print(f"  {len(geom)} masks read, {geom.attrs['missing']} missing")

        pts = to_patient_geometry(geom)
        y = pts[S.LABEL].to_numpy()
        groups = pts[S.PATIENT_ID].to_numpy()
        print(f"  {len(pts)} patients, prevalence {y.mean():.4f}")

        probe = C.geometry_probe(pts, y, groups)
        print(f"\n  CONTROL B -- mask geometry alone: AUC {probe['auc']:.4f}")
        print(f"  {probe['verdict']}")
        print("  coefficients (standardised):")
        for name, coef in sorted(probe["coefficients"].items(),
                                 key=lambda kv: -abs(kv[1])):
            print(f"    {name:14s} {coef:+.4f}")

        # Area alone, as the simplest possible shortcut: nodule size.
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedGroupKFold

        area = pts[["area"]].to_numpy()
        oof = np.full(len(y), np.nan)
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(
            area, y, groups
        ):
            model = LogisticRegression(max_iter=1000).fit(area[tr], y[tr])
            oof[va] = model.predict_proba(area[va])[:, 1]
        area_auc = float(roc_auc_score(y, oof))
        print(f"\n  area alone: AUC {area_auc:.4f}")

        results["splits"][split] = {
            "n_masks": int(len(geom)),
            "n_patients": int(len(pts)),
            "prevalence": round(float(y.mean()), 4),
            "control_B_geometry": probe,
            "area_only_auc": round(area_auc, 4),
        }
        per_split[split] = (pts, df)

    if "train" in per_split and "test" in per_split:
        print("\n=== transfer: fit on train patients, evaluate on test ===")
        tr_pts, _ = per_split["train"]
        te_pts, te_df = per_split["test"]
        tx = transfer(tr_pts, te_pts, te_df)
        results["transfer"] = tx
        g = tx["mask_geometry"]
        print(f"  mask geometry   AUC {g['auc']:.4f}  {g.get('auc_ci')}")
        if "tirads_auc_same_patients" in tx:
            print(f"  TI-RADS         AUC {tx['tirads_auc_same_patients']:.4f}"
                  f"  (n={tx['n_with_tirads']})")
            d = tx["delong_geometry_vs_tirads"]
            print(f"  DeLong geometry vs TI-RADS: p = {d.get('p'):.3g}")

    D.save_report(results, out_dir / "controls.json")
    print(f"\n  wrote {(out_dir / 'controls.json').resolve()}")


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------- transfer
# The CV-within-test number above is honest but in-split. The number that
# belongs beside a published result is fit-on-train / evaluate-on-test, which
# is the protocol every ThyroidXL paper uses, and it lets mask geometry be
# compared against TI-RADS on the same patients with a paired test.

def transfer(train_pts: pd.DataFrame, test_pts: pd.DataFrame,
             test_df: pd.DataFrame) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from src.metrics import delong_test, evaluate, threshold_at_sensitivity

    names = C.MaskGeometry.names()
    Xtr, ytr = train_pts[names].to_numpy(float), train_pts[S.LABEL].to_numpy(int)
    Xte, yte = test_pts[names].to_numpy(float), test_pts[S.LABEL].to_numpy(int)

    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000)
    ).fit(Xtr, ytr)
    geom_prob = model.predict_proba(Xte)[:, 1]
    # Threshold chosen on train, never on test: picking it on test would
    # flatter every operating-point metric below.
    tr_prob = model.predict_proba(Xtr)[:, 1]
    threshold = threshold_at_sensitivity(ytr, tr_prob, 0.95)
    geom_report = evaluate(yte, geom_prob, threshold=threshold)

    out = {
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "train_prevalence": round(float(ytr.mean()), 4),
        "test_prevalence": round(float(yte.mean()), 4),
        "threshold_from_train_at_95_sens": round(float(threshold), 4),
        "mask_geometry": geom_report.as_dict(),
    }

    # Head-to-head against TI-RADS on exactly these patients. TI-RADS is the
    # radiologist's reading of the same contour, so this asks whether an
    # automated shape read matches the human category.
    tirads = to_patient_level(test_df).set_index(S.PATIENT_ID)[S.TIRADS]
    aligned = tirads.reindex(test_pts[S.PATIENT_ID]).to_numpy(dtype=float)
    usable = ~np.isnan(aligned)
    if usable.sum() > 50:
        from sklearn.metrics import roc_auc_score
        out["tirads_auc_same_patients"] = round(
            float(roc_auc_score(yte[usable], aligned[usable])), 4
        )
        out["n_with_tirads"] = int(usable.sum())
        out["delong_geometry_vs_tirads"] = delong_test(
            yte[usable], geom_prob[usable], aligned[usable]
        )
    return out
