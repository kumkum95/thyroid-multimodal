"""Is the segmentation mask carrying the radiologist's own assessment?

    python scripts/run_redundancy.py

Control B established that mask shape alone reaches 0.9602 on the ThyroidXL
test split, beating TI-RADS at 0.9287 (DeLong p = 0.0022). That says the mask
is informative. It does not say *why*, and the answer decides what can be
claimed.

Two readings fit the Control B result equally well:

  artefact  annotators outlined malignant nodules differently, so the contour
            is a labelling tell and the signal is spurious.

  endogenous the contour genuinely carries shape features of malignancy --
            taller-than-wide, lobulation, irregular margin -- which are the
            same features a radiologist reads to assign TI-RADS. The mask is
            then a second recording of the human's judgement, and fusing it
            with TI-RADS is close to circular.

These make different, testable predictions, which is what this script checks:

  1. Redundancy. Under the endogenous reading, adding TI-RADS to geometry
     should buy little, because both encode the same shape assessment. Under
     the artefact reading they are unrelated and should combine additively.

  2. Recoverability. Under the endogenous reading, geometry should predict the
     TI-RADS *category itself* well above chance -- the strong version of the
     claim, since it needs no outcome label at all.

Reported with quadratic weighted kappa alongside accuracy, because TI-RADS is
ordinal: confusing 4 with 5 is not the same error as confusing 2 with 5, and
plain accuracy treats them identically.
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
from src import schema as S
from src.metrics import delong_test

NAMES = C.MaskGeometry.names()


def thermometer(tirads: np.ndarray) -> np.ndarray:
    """Ordinal encoding: column k is 1 when TI-RADS >= k+2.

    One-hot would throw away the ordering, and a single integer would impose
    equal spacing between categories. Neither is right for TI-RADS.
    """
    return np.column_stack([(tirads >= k).astype(float) for k in (2, 3, 4, 5)])


def load(split: str, geom_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(geom_dir / f"geometry_{split}.csv")
    return df.dropna(subset=[S.TIRADS]).reset_index(drop=True)


def fit_auc(Xtr, ytr, Xte) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000)
    ).fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="outputs/controls")
    ap.add_argument("--out", default="outputs/redundancy")
    args = ap.parse_args()

    geom_dir = Path(args.geometry)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, te = load("train", geom_dir), load("test", geom_dir)
    ytr, yte = tr[S.LABEL].to_numpy(int), te[S.LABEL].to_numpy(int)
    ttr, tte = tr[S.TIRADS].to_numpy(float), te[S.TIRADS].to_numpy(float)
    Gtr, Gte = tr[NAMES].to_numpy(float), te[NAMES].to_numpy(float)

    print(f"  train {len(tr)} patients, test {len(te)} patients "
          f"(both with TI-RADS present)")

    from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                                 roc_auc_score)
    from scipy import stats

    results: dict = {"n_train": int(len(tr)), "n_test": int(len(te))}

    # ------------------------------------------------- 1. redundancy
    print("\n=== 1. does TI-RADS add anything to mask geometry? ===")
    prob_g = fit_auc(Gtr, ytr, Gte)
    prob_t = fit_auc(thermometer(ttr), ytr, thermometer(tte))
    prob_both = fit_auc(
        np.hstack([Gtr, thermometer(ttr)]), ytr,
        np.hstack([Gte, thermometer(tte)]),
    )
    auc_g, auc_t, auc_b = (float(roc_auc_score(yte, p))
                           for p in (prob_g, prob_t, prob_both))
    _, _, z, p_val = delong_test(yte, prob_both, prob_g)
    print(f"  geometry only          {auc_g:.4f}")
    print(f"  TI-RADS only           {auc_t:.4f}")
    print(f"  geometry + TI-RADS     {auc_b:.4f}")
    print(f"  gain from adding TI-RADS: {auc_b - auc_g:+.4f}  "
          f"(DeLong z = {z:+.3f}, p = {p_val:.3g})")
    results["redundancy"] = {
        "auc_geometry": round(auc_g, 4),
        "auc_tirads": round(auc_t, 4),
        "auc_combined": round(auc_b, 4),
        "gain": round(auc_b - auc_g, 4),
        "delong_z": round(float(z), 4),
        "delong_p": p_val,
    }

    # ------------------------------------------------- 2. recoverability
    print("\n=== 2. can mask geometry recover the TI-RADS category? ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000)
    ).fit(Gtr, ttr.astype(int))
    pred = clf.predict(Gte)

    # Majority class is the honest floor: TI-RADS is badly imbalanced here, so
    # raw accuracy looks impressive without meaning anything.
    major = int(pd.Series(ttr.astype(int)).mode().iloc[0])
    base = float(accuracy_score(tte.astype(int),
                                np.full(len(tte), major)))
    acc = float(accuracy_score(tte.astype(int), pred))
    qwk = float(cohen_kappa_score(tte.astype(int), pred, weights="quadratic"))
    rho, rho_p = stats.spearmanr(pred, tte)
    print(f"  accuracy    {acc:.4f}   (majority-class floor {base:.4f})")
    print(f"  QWK         {qwk:.4f}")
    print(f"  spearman    {float(rho):+.4f}  (p = {float(rho_p):.3g})")
    print("  distribution of true vs predicted TI-RADS:")
    comp = pd.DataFrame({
        "true": pd.Series(tte.astype(int)).value_counts().sort_index(),
        "pred": pd.Series(pred).value_counts().sort_index(),
    }).fillna(0).astype(int)
    for cat, row in comp.iterrows():
        print(f"    TI-RADS {cat}:  true {row['true']:4d}   pred {row['pred']:4d}")
    results["recoverability"] = {
        "accuracy": round(acc, 4),
        "majority_floor": round(base, 4),
        "qwk": round(qwk, 4),
        "spearman": round(float(rho), 4),
        "spearman_p": float(rho_p),
    }

    # How strongly does the geometry malignancy score track the human category?
    rho2, rho2_p = stats.spearmanr(prob_g, tte)
    print(f"\n  geometry malignancy score vs TI-RADS category: "
          f"spearman {float(rho2):+.4f} (p = {float(rho2_p):.3g})")
    results["score_vs_tirads_spearman"] = round(float(rho2), 4)
    results["score_vs_tirads_p"] = float(rho2_p)

    D.save_report(results, out_dir / "redundancy.json")
    print(f"\n  wrote {(out_dir / 'redundancy.json').resolve()}")


if __name__ == "__main__":
    main()
