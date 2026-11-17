"""Prove the protocol guarantees hold. Run after any change to src/.

These are the mistakes that inflate results in this literature. Each one should
be impossible to make silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import baselines as B
from src import data as D
from src import metrics as M
from src import schema as S
from src import splits as SP

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    results.append((PASS if condition else FAIL, name, detail))


def main() -> None:
    root = Path("data/thyroidxl_synthetic")
    train = D.load_thyroidxl(root, "train")
    test = D.load_thyroidxl(root, "test")

    # 1. Sex encoding: {1,2} must become {0,1} with female = 1.
    #    The synthetic cohort is ~89% female, so the mean must be ~0.89 not ~0.11.
    female_frac = float(train.groupby(S.PATIENT_ID)[S.SEX].first().mean())
    check("sex encoding not inverted", 0.80 < female_frac < 0.95,
          f"female fraction = {female_frac:.3f}")

    # 2. Ambiguous numeric sex encodings resolve by value set, not per value.
    check("sex {0,1} -> male=0",
          list(S.normalise_sex(pd.Series([0, 1, 1]))) == [0, 1, 1])
    check("sex {1,2} -> male=0",
          list(S.normalise_sex(pd.Series([1, 2, 2]))) == [0, 1, 1])
    try:
        S.normalise_sex(pd.Series([3, 7]))
        check("sex garbage rejected", False, "accepted {3,7}")
    except ValueError:
        check("sex garbage rejected", True)

    # 3. TI-RADS harmonisation across the spellings the cohorts actually use.
    got = list(S.normalise_tirads(pd.Series(["TR4", "4a", "4A", 3.0, "II", "5"])))
    check("TI-RADS harmonised", got == [4, 4, 4, 3, 2, 5], f"got {got}")

    # 4. No patient straddles train/test.
    try:
        SP.assert_no_patient_overlap(train, test)
        check("train/test patient-disjoint", True)
    except AssertionError as exc:
        check("train/test patient-disjoint", False, str(exc))

    # 5. The overlap guard must actually fire when overlap exists.
    leaked = pd.concat([test, train.head(50)])
    try:
        SP.assert_no_patient_overlap(train, leaked)
        check("overlap guard fires", False, "guard stayed silent on real overlap")
    except AssertionError:
        check("overlap guard fires", True)

    # 6. Every CV fold is patient-disjoint.
    disjoint = True
    for _, tr, va in SP.grouped_folds(train, n_splits=5, seed=0):
        if set(tr[S.PATIENT_ID]) & set(va[S.PATIENT_ID]):
            disjoint = False
            break
    check("CV folds patient-disjoint", disjoint)

    # 7. A patient carrying two different labels must be rejected: it means the
    #    pathology join is wrong and every patient-level metric is corrupted.
    bad = train.head(4).copy()
    bad[S.LABEL] = [0, 1, 0, 1]
    bad[S.PATIENT_ID] = "SAME"
    bad[S.IMAGE_ID] = ["a", "b", "c", "d"]
    try:
        S.validate(bad)
        check("inconsistent patient label rejected", False, "validate() let it pass")
    except ValueError:
        check("inconsistent patient label rejected", True)

    # 8. Label permutation must destroy performance (control D).
    #
    #    This must average over many permutations, not test one. AUC is
    #    rank-based and scale-invariant: after permutation the fitted
    #    coefficients are near zero, but the *ordering* of predictions is still
    #    fixed by their signs, so a single permutation draws from a wide
    #    distribution around 0.5 (we have seen 0.38). Only the mean over
    #    permutations is pinned to 0.5, and that is the quantity worth asserting.
    train_pts = B.to_patient_level(train)
    test_pts = B.to_patient_level(test)
    from sklearn.metrics import roc_auc_score

    n_perm = 25
    perm_aucs = []
    for i in range(n_perm):
        rng = np.random.default_rng(100 + i)
        shuffled = train_pts.copy()
        shuffled[S.LABEL] = rng.permutation(shuffled[S.LABEL].to_numpy())
        fitted = B.fit_baseline(shuffled, "M2", kind="logreg", seed=i)
        perm_aucs.append(
            float(roc_auc_score(test_pts[S.LABEL], fitted.predict_proba(test_pts)))
        )
    mean_auc = float(np.mean(perm_aucs))
    check("permuted labels -> mean AUC ~0.5", 0.45 < mean_auc < 0.55,
          f"mean {mean_auc:.4f} over {n_perm} perms "
          f"(range {min(perm_aucs):.3f}-{max(perm_aucs):.3f})")

    # 9. A model may only see the fields its config allows.
    m1 = B.fit_baseline(train_pts, "M1", kind="logreg", seed=0)
    check("M1 cannot see TI-RADS",
          all("tirads" not in n for n in m1.spec.names), f"{m1.spec.names}")
    m0 = B.fit_baseline(train_pts, "M0", kind="logreg", seed=0)
    check("M0 cannot see age or sex",
          all(n.startswith("tirads") for n in m0.spec.names), f"{m0.spec.names}")

    # 10. DeLong on two identical score vectors: zero difference, p = 1.
    y = test_pts[S.LABEL].to_numpy()
    s = m0.predict_proba(test_pts)
    a, b, z, p = M.delong_test(y, s, s)
    check("DeLong identical -> p=1", abs(a - b) < 1e-12 and p > 0.99,
          f"delta={a - b:.2e}, p={p:.4f}")

    # 11. Bootstrap CI must bracket the point estimate.
    point, lo, hi = M.bootstrap_auc_ci(y, s, n_boot=300, seed=0)
    check("bootstrap CI brackets point", lo < point < hi,
          f"{lo:.4f} < {point:.4f} < {hi:.4f}")

    # 12. A perfectly calibrated predictor has near-zero ECE.
    probs = np.linspace(0.01, 0.99, 4000)
    truth = (np.random.default_rng(1).random(4000) < probs).astype(int)
    ece = M.expected_calibration_error(truth, probs)
    check("ECE ~0 when calibrated", ece < 0.05, f"ECE = {ece:.4f}")

    width = max(len(n) for _, n, _ in results)
    print()
    for status, name, detail in results:
        suffix = f"   {detail}" if detail else ""
        print(f"  [{status}] {name:<{width}}{suffix}")
    failed = sum(1 for s, _, _ in results if s == FAIL)
    print(f"\n  {len(results) - failed}/{len(results)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
