"""Gap 5, as far as the available data honestly allows.

    python scripts/run_external.py

The intended test was whether a *fusion* model transports across centres. It
cannot be run on DDTI: the release carries no malignancy endpoint at all, since
its FNA biopsy field is empty in all 390 cases (see src/ddti.py). Thresholding
TI-RADS to manufacture labels would make the task circular, because TI-RADS is
the input, so no label is invented here.

What DDTI can answer is the mechanism claim, which is the load-bearing one.
On ThyroidXL, mask contour recovered the radiologist's TI-RADS category at
QWK 0.6545 and Spearman +0.672. If that is a property of thyroid contours
rather than of ThyroidXL's annotators, it should reappear in a different
country, vendor and era. DDTI is all three, and grades at finer resolution.

Comparison is between within-cohort refits, cross-validated and grouped on
patient. A transferred model would be confounded: DDTI images are 560x360
holding two side-by-side panels, so area, perimeter and centroid normalised by
full width are not comparable with ThyroidXL's single-panel images, while
circularity, bbox_aspect and extent are. Refitting absorbs that.
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
from src import ddti as DD
from src import schema as S

NAMES = C.MaskGeometry.names()


def recover_tirads(geom: np.ndarray, tirads: np.ndarray, groups: np.ndarray,
                   seed: int = 0) -> dict:
    """Cross-validated geometry -> TI-RADS category, grouped on patient."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, cohen_kappa_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from scipy import stats

    y = tirads.astype(int)
    pred = np.zeros(len(y), dtype=int)
    # Categories with too few members cannot be stratified across 5 folds.
    counts = pd.Series(y).value_counts()
    keep = np.isin(y, counts[counts >= 5].index)
    geom, y, groups, pred = geom[keep], y[keep], groups[keep], pred[keep]

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, va in splitter.split(geom, y, groups=groups):
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000)
        ).fit(geom[tr], y[tr])
        pred[va] = model.predict(geom[va])

    major = int(pd.Series(y).mode().iloc[0])
    rho, rho_p = stats.spearmanr(pred, y)
    return {
        "n": int(len(y)),
        "categories": sorted(int(c) for c in set(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "majority_floor": round(float((y == major).mean()), 4),
        "qwk": round(float(cohen_kappa_score(y, pred, weights="quadratic")), 4),
        "spearman": round(float(rho), 4),
        "spearman_p": float(rho_p),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddti", default="data/ddti_raw")
    ap.add_argument("--geometry", default="outputs/controls")
    ap.add_argument("--out", default="outputs/external")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    print("=== DDTI coverage ===")
    ddti_df = DD.build_table(args.ddti)
    cov = DD.coverage(ddti_df)
    results["ddti_coverage"] = cov
    for key in ("patients", "images_with_contour", "age_coverage",
                "sex_coverage", "tirads_coverage", "has_malignancy_label"):
        print(f"  {key:22s} {cov[key]}")
    print(f"  tirads grades          {cov['tirads_raw_distribution']}")

    pts = DD.to_patient(ddti_df).dropna(subset=[S.TIRADS]).reset_index(drop=True)
    print(f"\n  {len(pts)} patients with both a contour and a TI-RADS grade")

    print("\n=== mechanism: does contour recover TI-RADS outside ThyroidXL? ===")
    # On DDTI's native scale, not the collapsed one. normalise_tirads folds
    # 4a/4b/4c into a single 4, which turns DDTI into 72% class 4 -- accuracy
    # and the majority floor then coincide and a QWK near zero measures the
    # collapse, not the contour. DDTI grades finer than ThyroidXL, so that
    # resolution is the fair target here.
    fine = {"2": 1, "3": 2, "4a": 3, "4b": 4, "4c": 5, "5": 6}
    pts_fine = pts[pts["tirads_raw"].astype(str).isin(fine)].reset_index(drop=True)
    ddti_res = recover_tirads(
        pts_fine[NAMES].to_numpy(float),
        pts_fine["tirads_raw"].astype(str).map(fine).to_numpy(float),
        pts_fine[S.PATIENT_ID].to_numpy(),
    )
    ddti_res["scale"] = "native 6-grade (2,3,4a,4b,4c,5)"
    results["ddti_tirads_recovery"] = ddti_res

    # The collapsed scale as well, to record why it cannot be used here.
    ddti_collapsed = recover_tirads(
        pts[NAMES].to_numpy(float),
        pts[S.TIRADS].to_numpy(float),
        pts[S.PATIENT_ID].to_numpy(),
    )
    ddti_collapsed["scale"] = "collapsed 4-grade, degenerate on DDTI"
    results["ddti_tirads_recovery_collapsed"] = ddti_collapsed
    print(f"  [collapsed scale on DDTI: QWK {ddti_collapsed['qwk']:.4f}, "
          f"accuracy {ddti_collapsed['accuracy']:.4f} vs floor "
          f"{ddti_collapsed['majority_floor']:.4f} -- degenerate, 72% in one class]")

    # The same measurement on ThyroidXL, from the cached geometry, so the two
    # are computed by identical code rather than quoted from an earlier run.
    thy = pd.concat(
        [pd.read_csv(Path(args.geometry) / f"geometry_{s}.csv")
         for s in ("train", "test")],
        ignore_index=True,
    ).dropna(subset=[S.TIRADS]).reset_index(drop=True)
    thy_res = recover_tirads(
        thy[NAMES].to_numpy(float),
        thy[S.TIRADS].to_numpy(float),
        thy[S.PATIENT_ID].to_numpy(),
    )
    results["thyroidxl_tirads_recovery"] = thy_res

    print(f"  {'cohort':12s} {'n':>5s} {'QWK':>8s} {'spearman':>9s} "
          f"{'acc':>7s} {'floor':>7s}")
    for name, res in (("ThyroidXL", thy_res), ("DDTI", ddti_res)):
        print(f"  {name:12s} {res['n']:5d} {res['qwk']:8.4f} "
              f"{res['spearman']:+9.4f} {res['accuracy']:7.4f} "
              f"{res['majority_floor']:7.4f}")

    gap = ddti_res["qwk"] - thy_res["qwk"]
    # Not the same task: DDTI is graded on 6 categories and ThyroidXL on 4,
    # and a longer ordinal scale is harder, so the two QWKs are indicative
    # rather than interchangeable. Each is scored against its own floor.
    verdict = (
        "REPLICATES: contour carries the radiologist's grade in both "
        "cohorts, so this is a property of thyroid contours rather than of "
        "one annotator team -- though on scales of different length."
        if ddti_res["qwk"] >= 0.30 else
        "DOES NOT REPLICATE on DDTI, which leaves the ThyroidXL result "
        "possibly specific to its annotators or its image format."
    )
    print(f"\n  QWK difference DDTI - ThyroidXL: {gap:+.4f}")
    print(f"  {verdict}")
    results["qwk_difference"] = round(float(gap), 4)
    results["verdict"] = verdict

    results["gap5_status"] = (
        "Malignancy transport is untestable on DDTI: the release has no "
        "pathology endpoint. Only the mechanism claim is tested here. Closing "
        "gap 5 for malignancy needs a labelled external cohort such as "
        "Stanford AIMI, which requires an access request."
    )
    D.save_report(results, out_dir / "external.json")
    print(f"\n  wrote {(out_dir / 'external.json').resolve()}")


if __name__ == "__main__":
    main()
