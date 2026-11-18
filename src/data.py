"""Cohort loaders.

The exact column names in the ThyroidXL release have not been inspected yet
(the Hugging Face repo is gated), so `load_csv_cohort` takes an explicit
column-mapping and `sniff_columns` suggests one. Nothing here guesses silently:
an unmapped required field raises.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import schema as S

# Candidate source names for each canonical field, lowercased.
ALIASES: dict[str, tuple[str, ...]] = {
    S.PATIENT_ID: ("patient_id", "patientid", "patient", "pid", "case_id", "case", "subject_id"),
    S.IMAGE_ID: ("image_id", "imageid", "image", "img_id", "filename", "file_name", "file"),
    S.IMAGE_PATH: ("image_path", "path", "filepath", "file_path", "relative_path"),
    S.AGE: ("age", "patient_age", "age_years"),
    S.SEX: ("sex", "gender", "patient_sex"),
    S.TIRADS: ("tirads", "ti_rads", "ti-rads", "tirads_category", "acr_tirads",
               "final_tirads", "tirads_final", "tr", "risk_category"),
    S.LABEL: ("label", "diagnosis", "malignant", "pathology", "target", "class", "benign_malignant"),
}


def sniff_columns(df: pd.DataFrame) -> dict[str, str | None]:
    """Propose a canonical -> source column mapping for a raw cohort table.

    Run this once on a new release, eyeball the result, then hard-code the
    mapping. Auto-detection is a convenience for inspection, not a dependency.
    """
    lowered = {c.lower().strip(): c for c in df.columns}
    out: dict[str, str | None] = {}
    for canonical, candidates in ALIASES.items():
        hit = next((lowered[c] for c in candidates if c in lowered), None)
        if hit is None:  # fall back to substring match
            hit = next(
                (orig for low, orig in lowered.items()
                 if any(c in low for c in candidates)),
                None,
            )
        out[canonical] = hit
    return out


def load_csv_cohort(
    csv_path: str | Path,
    cohort: str,
    *,
    mapping: dict[str, str] | None = None,
    image_root: str | Path | None = None,
    label_positive: object = 1,
) -> pd.DataFrame:
    """Load any cohort CSV into the canonical schema.

    `mapping` maps canonical name -> source column. Omitted entries are sniffed.
    `label_positive` is the raw value meaning malignant.
    """
    csv_path = Path(csv_path)
    raw = pd.read_csv(csv_path)
    guess = sniff_columns(raw)
    mapping = {**{k: v for k, v in guess.items() if v}, **(mapping or {})}

    required = [S.PATIENT_ID, S.AGE, S.SEX, S.TIRADS, S.LABEL]
    unmapped = [f for f in required if f not in mapping]
    if unmapped:
        raise ValueError(
            f"{csv_path.name}: could not map required fields {unmapped}.\n"
            f"Columns present: {list(raw.columns)}\n"
            f"Pass mapping={{'{unmapped[0]}': '<source column>', ...}}"
        )

    df = pd.DataFrame()
    df[S.PATIENT_ID] = raw[mapping[S.PATIENT_ID]].astype(str)

    if S.IMAGE_ID in mapping:
        df[S.IMAGE_ID] = raw[mapping[S.IMAGE_ID]].astype(str)
    else:  # one row per patient
        df[S.IMAGE_ID] = df[S.PATIENT_ID]

    if S.IMAGE_PATH in mapping:
        paths = raw[mapping[S.IMAGE_PATH]].astype(str)
    else:
        paths = df[S.IMAGE_ID]
    # A CSV written on Windows carries backslash separators. On Linux those
    # are ordinary filename characters, so joining them yields one long
    # bogus filename and every image silently goes missing -- which an
    # --allow-missing-images run would then paper over with noise.
    paths = paths.str.replace('\\', '/', regex=False)
    df[S.IMAGE_PATH] = (
        paths.map(lambda p: str(Path(image_root) / p)) if image_root else paths
    )

    df[S.AGE] = pd.to_numeric(raw[mapping[S.AGE]], errors="coerce")
    df[S.SEX] = S.normalise_sex(raw[mapping[S.SEX]])
    df[S.TIRADS] = S.normalise_tirads(raw[mapping[S.TIRADS]])

    lab = raw[mapping[S.LABEL]]
    if lab.dtype == object:
        lab_norm = lab.astype(str).str.strip().str.lower()
        pos = str(label_positive).strip().lower()
        df[S.LABEL] = (lab_norm == pos).astype(int)
        if df[S.LABEL].sum() == 0:
            df[S.LABEL] = lab_norm.isin({"malignant", "1", "true", "yes", "m"}).astype(int)
    else:
        df[S.LABEL] = (lab == label_positive).astype(int)

    df[S.COHORT] = cohort
    df = df[S.CANONICAL_COLUMNS]
    return S.validate(df)


def load_thyroidxl(root: str | Path, split: str = "train") -> pd.DataFrame:
    """Load a ThyroidXL split.

    Looks for `<root>/<split>.csv` or `<root>/metadata_<split>.csv`. If the
    release encodes sex as Male=1 / Female=2 (as the dataset card states), the
    schema normaliser handles it via SEX_ALIASES.
    """
    root = Path(root)
    for name in (f"{split}.csv", f"metadata_{split}.csv", f"{split}/metadata.csv"):
        candidate = root / name
        if candidate.exists():
            return load_csv_cohort(
                candidate, cohort=f"thyroidxl_{split}", image_root=root
            )
    raise FileNotFoundError(
        f"No ThyroidXL metadata CSV under {root} for split {split!r}. "
        f"Looked for: {split}.csv, metadata_{split}.csv, {split}/metadata.csv"
    )


def coverage(df: pd.DataFrame) -> dict:
    """How complete is each clinical field? Decides what fusion is even possible."""
    fields = [S.AGE, S.SEX, S.TIRADS]
    return {
        "cohort": str(df[S.COHORT].iloc[0]) if len(df) else "empty",
        "rows": int(len(df)),
        "patients": int(df[S.PATIENT_ID].nunique()),
        **{f"{f}_present": round(float(df[f].notna().mean()), 4) for f in fields},
        "complete_clinical": round(
            float(df[fields].notna().all(axis=1).mean()), 4
        ),
    }


def save_report(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path
