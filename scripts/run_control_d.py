"""Control D: label permutation against our own pipeline.

    python scripts/run_control_d.py

Every number this project reports is an accusation that someone else's result
is inflated, so the protocol producing them has to answer the same question.
If the labels are shuffled, each model must collapse to chance. A pipeline that
still scores above 0.5 on permuted labels is leaking through its own splitting,
scaling or feature construction, and none of its real figures can be trusted.

The permuted fit runs the *same* cross-validated, patient-grouped protocol as
the reported results, not a simplified stand-in, so what is being checked is
the protocol actually used.

On reading the output: a single permutation proves nothing. AUC is rank based
and scale invariant, so after shuffling the coefficients are near zero but the
ordering of predictions is still fixed by their signs, and one shuffle draws
from a wide distribution -- measured here between 0.255 and 0.783 in earlier
work. Only the mean over many permutations is pinned to 0.5, and it is compared
against its own standard error rather than a fixed window.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import controls as C
from src import data as D
from src import schema as S
from src.baselines import to_patient_level

GEOM = C.MaskGeometry.names()


def grouped_oof(groups: np.ndarray):
    """fit_predict that runs the real protocol: grouped 5-fold, out of fold.

    permutation_test calls fit_predict(X, y_shuffled, X). Returning in-sample
    predictions would understate leakage, because a model that memorises noise
    scores well on its own training rows; out-of-fold predictions are what the
    reported numbers are built from, so they are what gets permuted.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def fit_predict(X_train, y_train, X_test):
        oof = np.full(len(y_train), np.nan)
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        for tr, va in splitter.split(X_train, y_train, groups=groups):
            model = make_pipeline(
                StandardScaler(), LogisticRegression(max_iter=5000)
            ).fit(X_train[tr], y_train[tr])
            oof[va] = model.predict_proba(X_train[va])[:, 1]
        return oof

    return fit_predict


def thermometer(tirads: np.ndarray) -> np.ndarray:
    return np.column_stack([(tirads >= k).astype(float) for k in (2, 3, 4, 5)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/thyroidxl")
    ap.add_argument("--geometry", default="outputs/controls")
    ap.add_argument("--out", default="outputs/control_d")
    ap.add_argument("--n-perm", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Clinical features, patient level, from the training split.
    df = D.load_thyroidxl(Path(args.data), "train")
    pts = to_patient_level(df).dropna(subset=[S.TIRADS]).reset_index(drop=True)
    tirads = pts[S.TIRADS].to_numpy(float)
    age = pts[S.AGE].fillna(pts[S.AGE].median()).to_numpy(float)
    sex = pts[S.SEX].fillna(pts[S.SEX].median()).to_numpy(float)

    # Mask geometry, from the cached table, aligned on patient.
    geom_tbl = pd.read_csv(Path(args.geometry) / "geometry_train.csv")

    blocks = {
        "M0 (TI-RADS)": (thermometer(tirads), pts[S.LABEL].to_numpy(int),
                         pts[S.PATIENT_ID].to_numpy()),
        "M2 (age+sex+TI-RADS)": (
            np.column_stack([age, sex, thermometer(tirads)]),
            pts[S.LABEL].to_numpy(int), pts[S.PATIENT_ID].to_numpy()),
        "mask geometry": (geom_tbl[GEOM].to_numpy(float),
                          geom_tbl[S.LABEL].to_numpy(int),
                          geom_tbl[S.PATIENT_ID].to_numpy()),
    }

    print(f"  {args.n_perm} permutations per model, grouped 5-fold each\n")
    print(f"  {'model':24s} {'mean':>7s} {'sd':>7s} {'min':>7s} {'max':>7s} "
          f"{'tol':>6s}  verdict")
    report: dict = {"n_permutations": args.n_perm, "models": {}}
    for name, (X, y, groups) in blocks.items():
        res = C.permutation_test(grouped_oof(groups), X, y, groups,
                                 n_perm=args.n_perm, seed=0)
        flag = "PASS" if res["passes"] else "FAIL -- LEAK"
        print(f"  {name:24s} {res['mean_auc']:7.4f} {res['std']:7.4f} "
              f"{res['min']:7.4f} {res['max']:7.4f} {res['tolerance']:6.4f}  {flag}")
        report["models"][name] = res

    all_pass = all(r["passes"] for r in report["models"].values())
    report["all_pass"] = all_pass
    report["verdict"] = (
        "Every model collapses to chance on permuted labels, so the reported "
        "figures are not products of the splitting, scaling or feature code."
        if all_pass else
        "At least one model scores above chance on permuted labels. The "
        "pipeline leaks and its reported figures cannot be trusted."
    )
    print(f"\n  {report['verdict']}")
    D.save_report(report, out_dir / "control_d.json")
    print(f"\n  wrote {(out_dir / 'control_d.json').resolve()}")


if __name__ == "__main__":
    main()
