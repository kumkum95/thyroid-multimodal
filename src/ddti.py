"""DDTI loader, built from what the release actually contains.

The copy distributed as `dasmehdixtr/ddti-thyroid-ultrasound-images` holds 390
cases and 480 marked images from IDIME, Bogota. Measured field coverage:

    number        390 / 390
    age           304 / 390
    sex           305 / 390
    tirads        298 / 390      2, 3, 4a, 4b, 4c, 5
    svg contour   474 / 480
    reportbacaf     0 / 390      the FNA biopsy field, empty in every case
    reporteco      21 / 390      free text, one mentioning a Bethesda result

So **this release carries no malignancy ground truth**. `src/external.py`
assumed a Bethesda category was present and dropped cases without one, which
would discard all 390. Papers reporting DDTI "malignancy" accuracy therefore
owe an account of where their labels came from; deriving them by thresholding
TI-RADS would make the task circular, since TI-RADS is the input.

What the release does support is the mechanism question. ThyroidXL showed mask
contour recovers the radiologist's TI-RADS category at QWK 0.65. DDTI is a
different country, vendor and era, carries contours and TI-RADS, and grades at
finer resolution (4a/4b/4c rather than a collapsed 4), so it can test whether
that holds elsewhere.

Two properties of the format constrain how it is used:

  Dual panel. An image is 560x360 holding two side-by-side views, so a single
  nodule appears as two freehand regions. Normalising area, perimeter and
  centroid by the full width makes them incomparable with ThyroidXL's
  single-panel images. Circularity, bbox_aspect and extent are unaffected --
  and those are the three features that dominated the ThyroidXL result. Models
  are therefore refit within each cohort rather than transferred, which absorbs
  the difference; with no labels to transfer, that is the only option anyway.

  Largest region. The largest contour per image is taken, matching the
  largest-nodule rule used for ThyroidXL, so the two are aggregated alike.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from . import controls as C
from . import schema as S

IMAGE_SIZE = (560, 360)   # width, height; uniform across the release


def _text(node: ET.Element, tag: str) -> str | None:
    found = node.find(tag)
    if found is None:
        return None
    value = (found.text or "").strip()
    return value or None


def polygon_mask(points: list[dict], width: int, height: int) -> np.ndarray:
    """Rasterise one freehand region onto a binary mask."""
    from PIL import Image, ImageDraw

    xy = [(float(p["x"]), float(p["y"])) for p in points]
    canvas = Image.new("L", (width, height), 0)
    if len(xy) >= 3:
        ImageDraw.Draw(canvas).polygon(xy, fill=255)
    return np.array(canvas)


def parse_case(path: Path) -> list[dict]:
    """One row per marked image, with geometry of the largest region."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []

    number = _text(root, "number") or path.stem
    age, sex, tirads = (_text(root, t) for t in ("age", "sex", "tirads"))

    rows = []
    for mark in root.iter("mark"):
        index = _text(mark, "image") or "1"
        raw = _text(mark, "svg")
        if not raw:
            continue
        try:
            regions = json.loads(raw)
        except json.JSONDecodeError:
            continue

        image_path = path.parent / f"{number}_{index}.jpg"
        width, height = IMAGE_SIZE
        if image_path.exists():
            from PIL import Image

            with Image.open(image_path) as im:
                width, height = im.size

        best = None
        for region in regions:
            points = region.get("points") or []
            if len(points) < 3:
                continue
            geom = C.mask_geometry(polygon_mask(points, width, height))
            if best is None or geom.area > best.area:
                best = geom
        if best is None:
            continue

        rows.append({
            S.PATIENT_ID: f"ddti_{number}",
            S.IMAGE_ID: f"ddti_{number}_{index}",
            S.IMAGE_PATH: str(image_path),
            S.AGE: age,
            S.SEX: sex,
            S.TIRADS: tirads,
            S.COHORT: "ddti",
            **dict(zip(C.MaskGeometry.names(), best.as_array())),
        })
    return rows


def build_table(root: str | Path = "data/ddti_raw") -> pd.DataFrame:
    """Per-image table with geometry. No label column: the release has none."""
    root = Path(root)
    files = sorted(root.glob("*.xml"))
    if not files:
        raise FileNotFoundError(f"no DDTI XML files under {root}")

    rows: list[dict] = []
    for path in files:
        rows.extend(parse_case(path))
    if not rows:
        raise ValueError(f"parsed {len(files)} DDTI XML files but found no contours")

    df = pd.DataFrame(rows)
    df[S.AGE] = pd.to_numeric(df[S.AGE], errors="coerce")
    df[S.SEX] = S.normalise_sex(df[S.SEX])
    # Keep the sub-category before collapsing, since DDTI grades finer than
    # ThyroidXL and that extra resolution is worth reporting on its own.
    df["tirads_raw"] = df[S.TIRADS]
    df[S.TIRADS] = S.normalise_tirads(df[S.TIRADS])
    df.attrs["xml_files"] = len(files)
    return df


def to_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Largest nodule per case, matching the ThyroidXL aggregation."""
    idx = df.groupby(S.PATIENT_ID)["area"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def coverage(df: pd.DataFrame) -> dict:
    pts = to_patient(df)
    return {
        "cohort": "ddti",
        "xml_files": int(df.attrs.get("xml_files", 0)),
        "images_with_contour": int(len(df)),
        "patients": int(len(pts)),
        "age_coverage": round(float(pts[S.AGE].notna().mean()), 4),
        "sex_coverage": round(float(pts[S.SEX].notna().mean()), 4),
        "tirads_coverage": round(float(pts[S.TIRADS].notna().mean()), 4),
        "tirads_raw_distribution": {
            str(k): int(v)
            for k, v in pts["tirads_raw"].value_counts().sort_index().items()
        },
        "has_malignancy_label": False,
        "note": (
            "reportbacaf, the FNA biopsy field, is empty in all 390 cases, so "
            "this release supports no malignancy endpoint."
        ),
    }
