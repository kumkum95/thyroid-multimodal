"""External cohort loaders: DDTI and Stanford AIMI Cine-clip.

These exist to answer gap 4: does a *fusion* model transport across centres, or
does adding clinical data make generalisation worse? Both published
transportability studies on ThyroidXL are image-only, so the question is open.

The clinical vector is fixed at {age, sex, TI-RADS} because that is the
intersection of fields available in training and in every external cohort.
Reaching for anything richer breaks external validation, which is why
`harmonise` drops extra fields rather than letting them in.

Cohort-specific notes that affect how results must be read:

  DDTI       Colombian, different vendor and era. Its XML carries richer
             descriptors (composition, echogenicity, margins, calcification)
             and a Bethesda cytology result. Counts differ between the paper
             (299 cases / 347 images) and the project page (99 cases / 134
             images), so whatever is downloaded must be counted, not assumed.

  Stanford   167 patients, 192 nodules, but only 17 malignant. At ~9%
             prevalence, accuracy is meaningless; AUC, AUPRC and calibration
             are the only honest metrics, and the intercept will need
             recalibrating for the prevalence shift from ThyroidXL's ~48%.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from . import schema as S

# DDTI XML tag names, lowercased, mapped onto canonical fields.
DDTI_TAGS = {
    S.AGE: ("age",),
    S.SEX: ("sex", "gender"),
    S.TIRADS: ("tirads", "ti-rads"),
    "composition": ("composition",),
    "echogenicity": ("echogenicity",),
    "margins": ("margins",),
    "calcifications": ("calcifications", "calcification"),
    "bethesda": ("bethesda", "cytology"),
}

MALIGNANT_BETHESDA = {"5", "6", "v", "vi"}


def _first_text(root: ET.Element, names: tuple[str, ...]) -> str | None:
    for element in root.iter():
        if element.tag.lower() in names:
            text = (element.text or "").strip()
            if text:
                return text
    return None


def load_ddti(root: str | Path, *, keep_extra: bool = False) -> pd.DataFrame:
    """Parse DDTI's per-case XML files into the canonical schema.

    Malignancy is taken from the Bethesda cytology category (V/VI malignant),
    which is the only pathology-linked field DDTI ships. Cases without one are
    dropped rather than guessed: an invented label would silently corrupt the
    external comparison this cohort exists to provide.
    """
    root = Path(root)
    xml_files = sorted(root.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"no DDTI XML files under {root}")

    rows, skipped = [], {"no_bethesda": 0, "unparsable": 0}
    for path in xml_files:
        try:
            tree = ET.parse(path)
        except ET.ParseError:
            skipped["unparsable"] += 1
            continue
        node = tree.getroot()

        bethesda = _first_text(node, DDTI_TAGS["bethesda"])
        if not bethesda:
            skipped["no_bethesda"] += 1
            continue
        key = bethesda.strip().lower().lstrip("category ").strip()
        label = 1 if key in MALIGNANT_BETHESDA else 0

        case_id = path.stem
        images = sorted(
            p for p in path.parent.glob(f"{case_id}*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ) or [path.with_suffix(".jpg")]

        for i, image in enumerate(images):
            row = {
                S.PATIENT_ID: f"ddti_{case_id}",
                S.IMAGE_ID: f"ddti_{case_id}_{i}",
                S.IMAGE_PATH: str(image),
                S.AGE: _first_text(node, DDTI_TAGS[S.AGE]),
                S.SEX: _first_text(node, DDTI_TAGS[S.SEX]),
                S.TIRADS: _first_text(node, DDTI_TAGS[S.TIRADS]),
                S.LABEL: label,
                S.COHORT: "ddti",
            }
            if keep_extra:
                for extra in ("composition", "echogenicity", "margins",
                              "calcifications"):
                    row[extra] = _first_text(node, DDTI_TAGS[extra])
            rows.append(row)

    if not rows:
        raise ValueError(
            f"parsed {len(xml_files)} DDTI XML files but none carried a "
            f"Bethesda result (skipped: {skipped})"
        )

    df = pd.DataFrame(rows)
    df[S.AGE] = pd.to_numeric(df[S.AGE], errors="coerce")
    df[S.SEX] = S.normalise_sex(df[S.SEX])
    df[S.TIRADS] = S.normalise_tirads(df[S.TIRADS])
    df.attrs["skipped"] = skipped
    df.attrs["xml_files"] = len(xml_files)
    return S.validate(df[S.CANONICAL_COLUMNS + (
        ["composition", "echogenicity", "margins", "calcifications"]
        if keep_extra else []
    )])


def load_stanford(csv_path: str | Path, image_root: str | Path | None = None,
                  *, frames_per_nodule: int = 1) -> pd.DataFrame:
    """Load the Stanford AIMI Cine-clip annotations.

    One representative frame per nodule by default. Pooling all 17,412 cine
    frames would treat near-identical consecutive frames as independent
    observations, which is the same leak as an image-level split.
    """
    from .data import load_csv_cohort

    df = load_csv_cohort(csv_path, cohort="stanford", image_root=image_root)
    if frames_per_nodule > 0:
        df = (
            df.groupby(S.PATIENT_ID, group_keys=False)
            .head(frames_per_nodule)
            .reset_index(drop=True)
        )
    return S.validate(df)


def harmonise(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce any cohort to the shared schema, dropping extras.

    External validation is only meaningful if every cohort presents the same
    inputs, so richer fields are removed here rather than silently widening the
    clinical vector for one cohort.
    """
    return S.validate(df[S.CANONICAL_COLUMNS].copy())


def coverage_report(df: pd.DataFrame) -> dict:
    """What fraction of each required field survived parsing.

    A cohort with poor TI-RADS coverage cannot support M5 or M6 and should be
    reported as such instead of being quietly imputed to the median.
    """
    from .baselines import to_patient_level

    pts = to_patient_level(df)
    fields = [S.AGE, S.SEX, S.TIRADS]
    out = {
        "cohort": str(df[S.COHORT].iloc[0]) if len(df) else "empty",
        "images": int(len(df)),
        "patients": int(len(pts)),
        "malignant_patients": int(pts[S.LABEL].sum()),
        "prevalence": round(float(pts[S.LABEL].mean()), 4) if len(pts) else None,
    }
    for field in fields:
        out[f"{field}_coverage"] = round(float(pts[field].notna().mean()), 4)
    out["complete_clinical"] = round(
        float(pts[fields].notna().all(axis=1).mean()), 4
    )
    usable = []
    if out[f"{S.AGE}_coverage"] > 0.8 and out[f"{S.SEX}_coverage"] > 0.8:
        usable.append("M4 (image+age+sex)")
    if out[f"{S.TIRADS}_coverage"] > 0.8:
        usable += ["M5 (image+TI-RADS)", "M6 (full)"]
    out["supports"] = usable or ["M3 only (image)"]
    return out
