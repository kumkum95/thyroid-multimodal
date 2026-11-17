"""Canonical schema shared by every cohort.

The clinical modality must be the *intersection* of fields available in the
training cohort and in every external cohort, otherwise external validation is
impossible. That intersection is {age, sex, tirads}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Canonical column names. Every loader must emit exactly these.
PATIENT_ID = "patient_id"
IMAGE_ID = "image_id"
IMAGE_PATH = "image_path"
AGE = "age"
SEX = "sex"          # 0 = male, 1 = female
TIRADS = "tirads"    # ordinal 1..5, ACR/C-TIRADS category
LABEL = "label"      # 0 = benign, 1 = malignant (pathology confirmed)
COHORT = "cohort"

CANONICAL_COLUMNS = [
    PATIENT_ID, IMAGE_ID, IMAGE_PATH, AGE, SEX, TIRADS, LABEL, COHORT,
]

# Exogenous = independent of the pixels. Endogenous = read off the pixels by a
# radiologist. Keeping these apart is the point of the whole study.
EXOGENOUS_FIELDS = [AGE, SEX]
ENDOGENOUS_FIELDS = [TIRADS]

TIRADS_MIN, TIRADS_MAX = 1, 5

SEX_STRING_ALIASES = {
    "m": 0, "male": 0, "man": 0,
    "f": 1, "female": 1, "woman": 1,
}


def normalise_sex(series: pd.Series) -> pd.Series:
    """Map any common sex encoding onto 0 = male, 1 = female.

    Numeric encodings are ambiguous per-value -- a lone `1` means female under
    {0,1} but male under ThyroidXL's {1,2} (Male=1, Female=2, per its dataset
    card). So the encoding is inferred from the whole column's value set, not
    value by value. An unexpected value set raises rather than guessing: silently
    inverting sex would quietly corrupt every subgroup result.
    """
    if series.dtype == object:
        text = series.astype("string").str.strip().str.lower()
        if text.dropna().isin(SEX_STRING_ALIASES).all():
            return text.map(SEX_STRING_ALIASES).astype("Float64")
        # Fall through: numeric values stored as strings.
        series = pd.to_numeric(series, errors="coerce")

    numeric = pd.to_numeric(series, errors="coerce")
    present = set(numeric.dropna().unique())

    if present <= {0.0, 1.0}:
        mapping = {0.0: 0, 1.0: 1}          # 0 = male, 1 = female
    elif present <= {1.0, 2.0}:
        mapping = {1.0: 0, 2.0: 1}          # ThyroidXL: Male=1, Female=2
    else:
        raise ValueError(
            f"cannot infer sex encoding from value set {sorted(present)}; "
            "expected {0,1}, {1,2}, or strings like 'M'/'F'"
        )
    return numeric.map(mapping).astype("Float64")


def normalise_tirads(series: pd.Series) -> pd.Series:
    """Coerce TI-RADS onto the integer scale 1..5.

    Cohorts write this field inconsistently ("TR4", "4a", "4A", 4.0, "IV").
    Sub-categories 4a/4b/4c all collapse to 4: that is the coarsest common
    resolution across ThyroidXL, DDTI and Stanford, so it is the only level at
    which the three cohorts can be compared.
    """
    roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}

    def _one(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            iv = int(round(float(v)))
        else:
            s = str(v).strip().lower().replace("tr", "").replace("tirads", "")
            s = s.strip(" -_:")
            if s in roman:
                iv = roman[s]
            else:
                digits = "".join(ch for ch in s if ch.isdigit())
                if not digits:
                    raise ValueError(f"unrecognised TI-RADS value: {v!r}")
                iv = int(digits[0])
        if not (TIRADS_MIN <= iv <= TIRADS_MAX):
            raise ValueError(f"TI-RADS out of range after parsing {v!r} -> {iv}")
        return iv

    return series.map(_one).astype("Float64")


def validate(df: pd.DataFrame, *, require_paths: bool = False) -> pd.DataFrame:
    """Fail loudly on a malformed cohort table rather than silently later."""
    missing = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"cohort table missing canonical columns: {missing}")

    if df[PATIENT_ID].isna().any():
        raise ValueError("null patient_id: patient-level splitting would break")
    if df[IMAGE_ID].duplicated().any():
        dupes = df.loc[df[IMAGE_ID].duplicated(), IMAGE_ID].head().tolist()
        raise ValueError(f"duplicate image_id values, e.g. {dupes}")

    bad_label = ~df[LABEL].isin([0, 1])
    if bad_label.any():
        raise ValueError(f"{int(bad_label.sum())} rows have a label outside {{0,1}}")

    # One pathology outcome per patient. A patient with both labels means the
    # join is wrong, and would corrupt every patient-level metric downstream.
    per_patient = df.groupby(PATIENT_ID)[LABEL].nunique()
    if (per_patient > 1).any():
        offenders = per_patient[per_patient > 1].index[:5].tolist()
        raise ValueError(f"patients with inconsistent labels, e.g. {offenders}")

    if require_paths and df[IMAGE_PATH].isna().any():
        raise ValueError("null image_path with require_paths=True")

    return df
