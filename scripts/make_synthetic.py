"""Generate a synthetic ThyroidXL-shaped cohort so the pipeline can be run today.

WHY THIS EXISTS: the real ThyroidXL release is gated on Hugging Face, so the
pipeline needs something to execute against while access is pending. The
marginals below are taken from the published dataset card; the *associations*
between variables are assumptions written by hand.

CONSEQUENCE: numbers produced from this file validate the code, not the science.
Nothing from a synthetic run belongs in the paper. The runner stamps every
output with cohort="SYNTHETIC" so a synthetic number can never be mistaken for
a real one.

Published marginals reproduced here:
  total            4,093 patients / 11,545 images
  train            3,275 patients /  9,541 images
  test               739 patients /  2,094 images
  sex                89.2% female
  test prevalence    47.8% malignant (52.2% benign)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# --- published marginals -------------------------------------------------
N_TRAIN_PATIENTS, N_TRAIN_IMAGES = 3275, 9541
N_TEST_PATIENTS, N_TEST_IMAGES = 739, 2094
FRAC_FEMALE = 0.892
PREVALENCE = 0.478

# --- hand-chosen associations (ASSUMPTIONS, not data) --------------------
# TI-RADS distribution conditional on pathology. Tuned so TI-RADS alone gives
# an AUC near 0.85, which is the mid-range of what the literature reports for
# TI-RADS as a standalone predictor. Change these and M0 changes: that is the
# point of keeping them visible.
TIRADS_GIVEN_BENIGN = [0.08, 0.30, 0.34, 0.22, 0.06]   # categories 1..5
TIRADS_GIVEN_MALIGNANT = [0.01, 0.05, 0.14, 0.40, 0.40]

AGE_MEAN_BENIGN, AGE_MEAN_MALIGNANT = 47.0, 43.0       # malignancy skews younger
AGE_SD = 13.0
FEMALE_FRAC_BENIGN, FEMALE_FRAC_MALIGNANT = 0.91, 0.87


def _draw_split(rng, n_patients, n_images, start_id):
    label = (rng.random(n_patients) < PREVALENCE).astype(int)

    age = np.where(
        label == 1,
        rng.normal(AGE_MEAN_MALIGNANT, AGE_SD, n_patients),
        rng.normal(AGE_MEAN_BENIGN, AGE_SD, n_patients),
    ).clip(18, 88).round().astype(int)

    p_female = np.where(label == 1, FEMALE_FRAC_MALIGNANT, FEMALE_FRAC_BENIGN)
    is_female = (rng.random(n_patients) < p_female).astype(int)
    # Dataset card codes Male=1, Female=2; emit the raw encoding so the loader's
    # normalisation is exercised rather than bypassed.
    sex_raw = np.where(is_female == 1, 2, 1)

    tirads = np.empty(n_patients, dtype=int)
    for i, lab in enumerate(label):
        probs = TIRADS_GIVEN_MALIGNANT if lab == 1 else TIRADS_GIVEN_BENIGN
        tirads[i] = rng.choice([1, 2, 3, 4, 5], p=probs)

    # Spread images over patients, preserving the published image total.
    counts = rng.integers(1, 6, n_patients)
    counts = np.maximum(1, np.round(counts * n_images / counts.sum()).astype(int))
    drift = n_images - counts.sum()
    while drift != 0:
        j = rng.integers(0, n_patients)
        if drift > 0:
            counts[j] += 1
            drift -= 1
        elif counts[j] > 1:
            counts[j] -= 1
            drift += 1

    rows = []
    for i in range(n_patients):
        pid = f"P{start_id + i:05d}"
        for k in range(counts[i]):
            rows.append(
                {
                    "patient_id": pid,
                    "image_id": f"{pid}_{k}",
                    "image_path": f"images/{pid}_{k}.png",
                    "age": age[i],
                    "sex": int(sex_raw[i]),
                    "tirads": int(tirads[i]),
                    "diagnosis": int(label[i]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/thyroidxl_synthetic")
    ap.add_argument("--seed", type=int, default=20261117)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = _draw_split(rng, N_TRAIN_PATIENTS, N_TRAIN_IMAGES, 1)
    test = _draw_split(rng, N_TEST_PATIENTS, N_TEST_IMAGES, 90001)

    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    (out / "SYNTHETIC_DO_NOT_PUBLISH.txt").write_text(
        "Synthetic data. Validates the pipeline only. Not a source of findings.\n",
        encoding="utf-8",
    )

    for name, df in (("train", train), ("test", test)):
        pts = df.groupby("patient_id").first()
        print(
            f"{name}: {df.patient_id.nunique()} patients, {len(df)} images, "
            f"prevalence {pts.diagnosis.mean():.3f}, "
            f"female {(pts.sex == 2).mean():.3f}"
        )
    print(f"written to {out.resolve()}")


if __name__ == "__main__":
    main()
