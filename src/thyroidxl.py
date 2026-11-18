"""Loader for the real ThyroidXL release (VNPT-AI/ThyroidXL on Hugging Face).

Repository layout:
    train/{images,labels,labels_orig,masks}/  train/train_annotations.json
    test/{images,labels,labels_orig,masks}/   test/test_annotations.json
    stats/id2info_eng.json                    patient metadata, 4093 entries
    stats/{train,test}_patients.txt           the official patient-level splits

Metadata schema, per patient id:
    age, gender (1=male, 2=female), conclusion, images[], nodule_1{...}

`conclusion` is the diagnosis and how it was reached:
    1  Malignant (Histopathology)   708
    2  Malignant (FNAC)             522
    3  Benign (FNAC)               2769
    4  Benign (Histopathology)       94
so {1,2} is malignant and {3,4} benign.

Three facts about this release that change how results must be read:

1. The official split is NOT stratified by outcome. Train is 26.2% malignant,
   test is 47.8% -- a 21.6-point shift. A model trained and calibrated on the
   train distribution is therefore miscalibrated on test by construction, and
   any calibration figure that ignores this is measuring the shift rather than
   the model.

2. FNAC and Histopathology are the label, not features. FNAC agrees with the
   derived label for 95.3% of patients, and one of its own categories is
   literally "6 (papillary thyroid carcinoma)". Using either as an input would
   be circular, so `clinical_fields` never exposes them.

3. `age` contains year-of-birth values (1948, 1950, 1953...) and other
   impossible entries: 36 patients at >=1000 and 3 between 100 and 999. These
   are set to NaN rather than repaired, because the exam date needed to convert
   a birth year is not in the release. The plausible subset has mean 48.3,
   median 48, consistent with the literature; leaving the raw values in would
   have put the mean at 64.9.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import schema as S

REPO_ID = "VNPT-AI/ThyroidXL"

MALIGNANT_CONCLUSIONS = {1, 2}
BENIGN_CONCLUSIONS = {3, 4}

CONCLUSION_MEANING = {
    1: "Malignant (Histopathology)",
    2: "Malignant (FNAC)",
    3: "Benign (FNAC)",
    4: "Benign (Histopathology)",
}

# Outcome-derived fields. Never inputs.
LEAKY_FIELDS = ("FNAC", "Histopathology", "Conclusion")

AGE_MIN, AGE_MAX = 1, 99

METADATA_FILES = [
    "stats/id2info_eng.json",
    "stats/train_patients.txt",
    "stats/test_patients.txt",
]


def download_metadata(dest: str | Path = "data/thyroidxl_meta") -> Path:
    """Fetch just the metadata (about 2.4 MB), not the 6.87 GB of images."""
    from huggingface_hub import hf_hub_download

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in METADATA_FILES:
        hf_hub_download(REPO_ID, name, repo_type="dataset", local_dir=str(dest))
    return dest


def clean_age(values: pd.Series) -> tuple[pd.Series, dict]:
    """Null out impossible ages and report what was dropped.

    Repairing a year-of-birth would need the exam date, which the release does
    not contain, so guessing it would invent data.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    bad = numeric.notna() & ((numeric < AGE_MIN) | (numeric > AGE_MAX))
    report = {
        "n_total": int(numeric.notna().sum()),
        "n_nulled": int(bad.sum()),
        "looks_like_birth_year": int((numeric >= 1000).sum()),
        "examples": sorted({int(v) for v in numeric[bad].dropna().unique()})[:8],
    }
    return numeric.mask(bad), report


def load_metadata(meta_dir: str | Path = "data/thyroidxl_meta") -> dict:
    meta_dir = Path(meta_dir)
    info_path = meta_dir / "stats" / "id2info_eng.json"
    if not info_path.exists():
        raise FileNotFoundError(
            f"{info_path} missing. Run: python -c "
            f"\"from src.thyroidxl import download_metadata; download_metadata()\""
        )
    return json.loads(info_path.read_text(encoding="utf-8"))


def load_split_ids(meta_dir: str | Path, split: str) -> list[str]:
    path = Path(meta_dir) / "stats" / f"{split}_patients.txt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    return [line.strip() for line in path.read_text().split() if line.strip()]


def build_table(
    split: str,
    meta_dir: str | Path = "data/thyroidxl_meta",
    image_root: str | Path | None = None,
    *,
    extra_fields: bool = False,
) -> pd.DataFrame:
    """One row per image, in the canonical schema.

    `extra_fields` adds nodule width, height and position. They are legitimate
    inputs -- a sonographer records them from the same scan, so they are no more
    endogenous than TI-RADS -- but they are off by default because the external
    cohorts do not all carry them, and the clinical vector must stay the
    intersection across cohorts for transport results to mean anything.
    """
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    info = load_metadata(meta_dir)
    ids = load_split_ids(meta_dir, split)
    root = Path(image_root) if image_root else Path(split) / "images"

    rows = []
    missing_ids = 0
    for pid in ids:
        entry = info.get(pid)
        if entry is None:
            missing_ids += 1
            continue
        conclusion = entry.get("conclusion")
        if conclusion in MALIGNANT_CONCLUSIONS:
            label = 1
        elif conclusion in BENIGN_CONCLUSIONS:
            label = 0
        else:
            missing_ids += 1
            continue

        nodule = entry.get("nodule_1") or {}
        for filename in entry.get("images") or []:
            row = {
                S.PATIENT_ID: pid,
                S.IMAGE_ID: filename,
                S.IMAGE_PATH: str(root / filename),
                S.AGE: entry.get("age"),
                S.SEX: entry.get("gender"),
                S.TIRADS: nodule.get("TIRADS"),
                S.LABEL: label,
                S.COHORT: f"thyroidxl_{split}",
            }
            if extra_fields:
                row["nodule_width"] = nodule.get("Width")
                row["nodule_height"] = nodule.get("Height")
                row["position"] = nodule.get("Position")
            rows.append(row)

    if not rows:
        raise ValueError(f"no rows built for split {split!r}")

    df = pd.DataFrame(rows)
    df[S.AGE], age_report = clean_age(df[S.AGE])
    df[S.SEX] = S.normalise_sex(df[S.SEX])      # gender 1/2 -> 0=male, 1=female
    df[S.TIRADS] = S.normalise_tirads(df[S.TIRADS])

    columns = S.CANONICAL_COLUMNS + (
        ["nodule_width", "nodule_height", "position"] if extra_fields else []
    )
    df = df[columns]
    S.validate(df)
    df.attrs["age_cleaning"] = age_report
    df.attrs["patients_skipped"] = missing_ids
    return df


def split_summary(df: pd.DataFrame) -> dict:
    from .baselines import to_patient_level

    pts = to_patient_level(df)
    return {
        "cohort": str(df[S.COHORT].iloc[0]),
        "patients": int(len(pts)),
        "images": int(len(df)),
        "images_per_patient": round(float(len(df) / len(pts)), 2),
        "malignant_patients": int(pts[S.LABEL].sum()),
        "prevalence": round(float(pts[S.LABEL].mean()), 4),
        "female_fraction": round(float(pts[S.SEX].mean()), 4),
        "age_missing_after_cleaning": int(pts[S.AGE].isna().sum()),
        "tirads_distribution": {
            int(k): int(v) for k, v in pts[S.TIRADS].value_counts().sort_index().items()
        },
        "age_cleaning": df.attrs.get("age_cleaning"),
    }


def prevalence_shift(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Quantify the train/test outcome shift built into the official split."""
    from .baselines import to_patient_level

    a = float(to_patient_level(train_df)[S.LABEL].mean())
    b = float(to_patient_level(test_df)[S.LABEL].mean())
    return {
        "train_prevalence": round(a, 4),
        "test_prevalence": round(b, 4),
        "shift": round(b - a, 4),
        "note": (
            "The official split is not stratified by outcome. A model "
            "calibrated on the training distribution is miscalibrated on test "
            "by construction, so calibration must be reported before and after "
            "recalibration, and any uncorrected figure measures the shift "
            "rather than the model."
        ),
    }
