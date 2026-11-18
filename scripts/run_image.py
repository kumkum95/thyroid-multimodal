"""Train and evaluate M3-M6, the image and fusion configurations.

    python scripts/run_image.py --data data/thyroidxl --configs M3 M4 M6 --seeds 0 1 2

Each (config, seed) trains independently so the headline table is not confounded
by shared state. The internal test split is scored once per run; external
cohorts are scored with the same validation-derived temperature and threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D
from src import metrics as M
from src import schema as S
from src import splits as SP
from src.train import TrainConfig, evaluate_cohort, subgroups, train_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/thyroidxl_synthetic")
    ap.add_argument("--out", default="outputs/image")
    ap.add_argument("--configs", nargs="+", default=["M3", "M4", "M6"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--backbone", default="efficientnet_b4")
    ap.add_argument("--image-size", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-images", type=int, default=6)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="keep only N patients per split; for smoke tests only")
    ap.add_argument("--allow-missing-images", action="store_true",
                    help="substitute deterministic noise for absent image files; "
                         "for exercising the loop only, never for results")
    args = ap.parse_args()

    root = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic = (root / "SYNTHETIC_DO_NOT_PUBLISH.txt").exists()
    tag = "SYNTHETIC" if synthetic else "REAL"
    if args.allow_missing_images:
        tag += "+PLACEHOLDER_IMAGES"

    full_train = D.load_thyroidxl(root, "train")
    test = D.load_thyroidxl(root, "test")
    SP.assert_no_patient_overlap(full_train, test)

    # Development split, grouped on patient. The official test split stays
    # untouched until the final scoring pass.
    train_df, val_df = SP.holdout(full_train, val_frac=args.val_frac)
    SP.assert_no_patient_overlap(train_df, val_df, test)

    if args.limit:
        tag += f"+LIMIT{args.limit}"

        def head_patients(frame, n):
            """Keep n patients, balanced across both classes.

            A plain head() of the patient list draws them in file order,
            and on ThyroidXL that gives 40 benign patients and no
            malignant ones, so every AUC comes back nan and the smoke
            test proves nothing. Half from each class fixes that.
            """
            per_patient = frame[[S.PATIENT_ID, S.LABEL]].drop_duplicates(
                subset=S.PATIENT_ID
            )
            keep = []
            for label, want in ((0, n - n // 2), (1, n // 2)):
                ids = per_patient.loc[
                    per_patient[S.LABEL] == label, S.PATIENT_ID
                ]
                keep.extend(ids.head(want).tolist())
            return frame[frame[S.PATIENT_ID].isin(set(keep))].copy()

        train_df = head_patients(train_df, args.limit)
        val_df = head_patients(val_df, max(args.limit // 4, 20))
        test = head_patients(test, max(args.limit // 4, 20))

    # Model selection, temperature scaling and the sensitivity threshold
    # all need both classes in validation. Failing here names the cause;
    # letting training run produces a nan AUC and selects nothing.
    for frame, name in ((val_df, 'val'), (test, 'test')):
        present = set(frame[S.LABEL].unique())
        if present != {0, 1}:
            raise SystemExit(
                f'{name} split contains only class(es) {sorted(present)}. '
                'Model selection, temperature scaling and the sensitivity '
                'threshold all need both classes, so training would produce '
                'a nan AUC and select nothing. Raise --limit or drop it.'
            )

    print(f"\n=== {tag} ===")
    for frame, name in ((train_df, "train"), (val_df, "val"), (test, "test")):
        print("  ", SP.describe(frame, name))

    results = {"tag": tag, "data_dir": str(root.resolve()), "runs": []}

    for config in args.configs:
        for seed in args.seeds:
            print(f"\n--- {config} seed {seed} ---")
            cfg = TrainConfig(
                config=config,
                backbone=args.backbone,
                image_size=args.image_size,
                max_images=args.max_images,
                batch_size=args.batch_size,
                epochs=args.epochs,
                num_workers=args.num_workers,
                seed=seed,
                allow_missing_images=args.allow_missing_images,
            )
            fitted = train_model(train_df, val_df, cfg, out_dir,
                                 resume=not args.no_resume)
            scored = evaluate_cohort(fitted, test, cfg, "thyroidxl_test")
            scores = scored.pop("_scores")

            run = {
                "config": config,
                "seed": seed,
                "best_val_auc": fitted["best_val_auc"],
                "temperature": fitted["temperature"],
                "threshold": fitted["threshold"],
                "test": scored,
                "subgroups": subgroups(test, scores, fitted["threshold"]),
                "history": fitted["history"],
            }
            results["runs"].append(run)

            np.save(out_dir / f"{config}_seed{seed}_test_prob.npy",
                    np.asarray(scores["prob"]))
            (out_dir / f"{config}_seed{seed}_test_ids.json").write_text(
                json.dumps(scores["patient_id"]), encoding="utf-8"
            )

            print(f"  TEST AUC {scored['auc']:.4f} "
                  f"[{scored['auc_lo']:.4f}-{scored['auc_hi']:.4f}]  "
                  f"ECE {scored['ece']:.4f}")
            if "mean_gates" in scored:
                print(f"  mean modality gates: {scored['mean_gates']}")

    D.save_report(results, out_dir / "results.json")

    print("\n=== summary ===")
    for run in results["runs"]:
        t = run["test"]
        print(f"  {run['config']} seed {run['seed']}: "
              f"AUC {t['auc']:.4f}  AUPRC {t['auprc']:.4f}  ECE {t['ece']:.4f}")
    print(f"\n  wrote {(out_dir / 'results.json').resolve()}")


if __name__ == "__main__":
    main()
