"""Control C: is burned-in annotation distributed differently by class?

    python scripts/run_control_c.py --images data/thyroidxl_images

Sonographers mark nodules they measure with calipers, crosshairs and text, and
whether a nodule got measured is not independent of how suspicious it looked.
If those marks are more common on malignant scans, a model can score well by
reading the sonographer's concern rather than the tissue.

This became load-bearing rather than decorative. M3, image only, reached 0.9421
on the official test split -- above TI-RADS alone and above the published
image-only 0.921 -- so whether that is tissue or overlay is now an open
question about our own best image model, not just about other people's.

It also tests the same hypothesis as the mask work from a different direction.
TN3K showed the ThyroidXL contour-shape signal does not survive a change of
centre, which points to something specific in how ThyroidXL was prepared. If
burned-in marks also differ by class, that is a second, independent annotation
channel carrying the label.

The detector is deliberately crude: near-saturated pixels forming straight runs
at least `min_run` long, which is what overlay graphics look like and tissue
does not. It will also fire on genuinely bright tissue, so the comparison that
matters is benign against malignant within the same cohort, not the absolute
level.
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
from src import schema as S
from src.baselines import to_patient_level


def score_split(df: pd.DataFrame, image_root: Path, split: str,
                limit: int = 0) -> pd.DataFrame:
    from PIL import Image

    frame = df if not limit else df.head(limit)
    total = len(frame)
    rows, missing = [], 0
    for i, (_, row) in enumerate(frame.iterrows()):
        if i and i % 2000 == 0:
            print(f"    {i}/{total} images scored", flush=True)
        path = image_root / split / "images" / str(row[S.IMAGE_ID])
        if not path.exists():
            missing += 1
            continue
        try:
            with Image.open(path) as im:
                arr = np.array(im.convert("L"))
        except Exception:
            missing += 1
            continue
        score = C.caliper_score(arr)
        rows.append({
            S.PATIENT_ID: row[S.PATIENT_ID],
            S.IMAGE_ID: row[S.IMAGE_ID],
            S.LABEL: int(row[S.LABEL]),
            **score,
        })
    out = pd.DataFrame(rows)
    out.attrs["missing"] = missing
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/thyroidxl_images")
    ap.add_argument("--data", default="data/thyroidxl")
    ap.add_argument("--out", default="outputs/control_c")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    image_root = Path(args.images)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"images": str(image_root.resolve()), "splits": {}}

    for split in args.splits:
        print(f"\n=== {split} ===")
        df = D.load_thyroidxl(Path(args.data), split)
        scores = score_split(df, image_root, split, limit=args.limit)
        if scores.empty:
            print(f"  no images found under {image_root / split / 'images'}")
            continue
        print(f"  {len(scores)} images scored, {scores.attrs['missing']} missing")

        # Patient level, mean over that patient's images: the question is
        # whether a patient's scans carry more overlay, not whether one frame
        # happens to.
        pts = (
            scores.groupby([S.PATIENT_ID, S.LABEL], as_index=False)
            [["saturated_fraction", "run_mass"]].mean()
        )
        audit = C.caliper_audit(pts, pts[S.LABEL].to_numpy(int))
        print(f"  {len(pts)} patients")
        for field in ("saturated_fraction", "run_mass"):
            res = audit.get(field, {})
            if "p" not in res:
                continue
            print(f"  {field:20s} benign {res['benign_mean']:.6f}  "
                  f"malignant {res['malignant_mean']:.6f}  p {res['p']:.3g}"
                  f"{'  DIFFERS' if res['differs'] else ''}")
        print(f"  {audit['verdict']}")

        # How much of the label can the overlay signal alone recover?
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X = pts[["saturated_fraction", "run_mass"]].to_numpy(float)
        y = pts[S.LABEL].to_numpy(int)
        groups = pts[S.PATIENT_ID].to_numpy()

        # A partial download, or --limit, can leave too few patients to split.
        # Say so and move on rather than dying after the audit already ran.
        counts = pd.Series(y).value_counts()
        if len(y) < 25 or len(counts) < 2 or counts.min() < 5:
            print(f"  annotation signal alone: skipped, only {len(y)} patients "
                  f"with class counts {counts.to_dict()}")
            report["splits"][split] = {
                "n_images": int(len(scores)),
                "n_patients": int(len(pts)),
                "missing": int(scores.attrs["missing"]),
                "audit": audit,
                "annotation_only_auc": None,
            }
            scores.to_csv(out_dir / f"caliper_{split}.csv", index=False)
            continue

        oof = np.full(len(y), np.nan)
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(
            X, y, groups
        ):
            model = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=5000)
            ).fit(X[tr], y[tr])
            oof[va] = model.predict_proba(X[va])[:, 1]
        auc = float(roc_auc_score(y, oof))
        print(f"  annotation signal alone: AUC {auc:.4f}")

        report["splits"][split] = {
            "n_images": int(len(scores)),
            "n_patients": int(len(pts)),
            "missing": int(scores.attrs["missing"]),
            "audit": audit,
            "annotation_only_auc": round(auc, 4),
        }
        scores.to_csv(out_dir / f"caliper_{split}.csv", index=False)

    D.save_report(report, out_dir / "control_c.json")
    print(f"\n  wrote {(out_dir / 'control_c.json').resolve()}")


if __name__ == "__main__":
    main()
