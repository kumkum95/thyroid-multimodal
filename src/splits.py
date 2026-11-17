"""Patient-level splitting.

With ~2.8 images per patient, an image-level split puts the same nodule on both
sides of the boundary and inflates AUC by several points. Every split in this
project groups on patient_id. `assert_no_patient_overlap` is called after every
split so the mistake cannot survive a run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .schema import LABEL, PATIENT_ID


def patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the image table to one row per patient.

    Labels are per-patient in ThyroidXL, so `first` is exact, not an
    approximation. `schema.validate` has already proved the label is unique
    within each patient.
    """
    agg = {LABEL: "first", "n_images": (PATIENT_ID, "size")}
    out = (
        df.groupby(PATIENT_ID)
        .agg(label=(LABEL, "first"), n_images=(PATIENT_ID, "size"))
        .reset_index()
    )
    return out


def assert_no_patient_overlap(*frames: pd.DataFrame) -> None:
    seen: dict[object, int] = {}
    for i, f in enumerate(frames):
        for pid in f[PATIENT_ID].unique():
            if pid in seen:
                raise AssertionError(
                    f"patient {pid!r} appears in split {seen[pid]} and split {i}"
                )
            seen[pid] = i


def grouped_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 0):
    """Yield (train_df, val_df) image-level frames with disjoint patients.

    Stratified on the patient-level label so each fold keeps the benign/malignant
    ratio, grouped on patient so no patient straddles a fold.
    """
    pts = patient_table(df)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(
        splitter.split(pts, y=pts[LABEL], groups=pts[PATIENT_ID])
    ):
        tr_ids = set(pts.iloc[tr_idx][PATIENT_ID])
        va_ids = set(pts.iloc[va_idx][PATIENT_ID])
        tr = df[df[PATIENT_ID].isin(tr_ids)].copy()
        va = df[df[PATIENT_ID].isin(va_ids)].copy()
        assert_no_patient_overlap(tr, va)
        yield fold, tr, va


def holdout(df: pd.DataFrame, val_frac: float = 0.15, seed: int = 0):
    """Single stratified, grouped dev/val split.

    Implemented as one fold of StratifiedGroupKFold rather than a random patient
    shuffle, so the class balance is preserved exactly the same way as in CV.
    """
    n_splits = max(2, int(round(1.0 / val_frac)))
    for _, tr, va in grouped_folds(df, n_splits=n_splits, seed=seed):
        return tr, va
    raise RuntimeError("no folds produced")


def describe(df: pd.DataFrame, name: str) -> dict:
    pts = patient_table(df)
    return {
        "split": name,
        "patients": int(len(pts)),
        "images": int(len(df)),
        "images_per_patient": round(float(len(df) / max(len(pts), 1)), 2),
        "malignant_patients": int(pts[LABEL].sum()),
        "prevalence": round(float(pts[LABEL].mean()), 4),
    }
