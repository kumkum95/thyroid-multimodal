"""Week 1: the clinical-only baselines the literature skipped.

Answers three questions before a single image is loaded:
  M0  TI-RADS alone            -- how much does the radiologist's own score give?
  M1  age + sex alone          -- do bare demographics carry signal?
  M2  age + sex + TI-RADS      -- the clinical-only baseline nobody reported

If M0 is already ~0.90, the "multimodal beats image-only" claims in this
literature are mostly TI-RADS, and the project's framing is confirmed.

Protocol notes:
  * Every split groups on patient_id. Never on image.
  * The operating threshold is chosen from out-of-fold training predictions and
    applied unchanged to the test split. The test split is scored once.
  * Confidence intervals bootstrap over patients.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import baselines as B
from src import data as D
from src import metrics as M
from src import schema as S
from src import splits as SP

CONFIG_ORDER = ["M0", "M1", "M2"]
KINDS = ["logreg", "gbdt"]
DESC = {"M0": "TI-RADS", "M1": "age+sex", "M2": "age+sex+TI-RADS"}


def cross_validate(train_pts, config, kind, n_splits=5, seed=0):
    """Out-of-fold predictions over the training patients."""
    oof = np.full(len(train_pts), np.nan)
    fold_aucs = []

    # Reuse the image-level splitter on a one-row-per-patient frame: grouping is
    # then trivially satisfied and the stratification logic stays identical.
    frame = train_pts.assign(
        **{S.COHORT: "cv", S.IMAGE_ID: train_pts[S.PATIENT_ID], S.IMAGE_PATH: ""}
    )
    pos = {pid: i for i, pid in enumerate(train_pts[S.PATIENT_ID])}

    for fold, tr, va in SP.grouped_folds(frame, n_splits=n_splits, seed=seed):
        tr_pts = train_pts[train_pts[S.PATIENT_ID].isin(set(tr[S.PATIENT_ID]))]
        va_pts = train_pts[train_pts[S.PATIENT_ID].isin(set(va[S.PATIENT_ID]))]
        fitted = B.fit_baseline(tr_pts, config, kind=kind, seed=seed + fold)
        probs = fitted.predict_proba(va_pts)
        for pid, prob in zip(va_pts[S.PATIENT_ID], probs):
            oof[pos[pid]] = prob
        fold_aucs.append(float(roc_auc_score(va_pts[S.LABEL], probs)))

    if np.isnan(oof).any():
        raise RuntimeError("some training patients received no out-of-fold prediction")
    return oof, fold_aucs


def subgroup_table(test_pts, prob, threshold):
    """Per-sex and per-age-band performance.

    ThyroidXL is ~89% female, so the male subgroup is small by construction.
    Rows below 20 patients, or with only one class present, are reported as
    underpowered rather than given a meaningless AUC.
    """
    rows = []

    for value, name in {0: "male", 1: "female"}.items():
        mask = (pd.to_numeric(test_pts[S.SEX], errors="coerce") == value).to_numpy()
        label = test_pts.loc[mask, S.LABEL]
        if mask.sum() < 20 or label.nunique() < 2:
            rows.append(
                {"subgroup": f"sex={name}", "n": int(mask.sum()),
                 "auc": None, "note": "underpowered"}
            )
            continue
        rep = M.evaluate(label, prob[mask], threshold=threshold, n_boot=1000, seed=1)
        rows.append({"subgroup": f"sex={name}", **rep.as_dict()})

    age = pd.to_numeric(test_pts[S.AGE], errors="coerce").to_numpy()
    for lo, hi in [(0, 40), (40, 55), (55, 200)]:
        mask = (age >= lo) & (age < hi)
        name = f"age={lo}-{hi}" if hi < 200 else f"age={lo}+"
        label = test_pts.loc[mask, S.LABEL]
        if mask.sum() < 20 or label.nunique() < 2:
            rows.append({"subgroup": name, "n": int(mask.sum()),
                         "auc": None, "note": "underpowered"})
            continue
        rep = M.evaluate(label, prob[mask], threshold=threshold, n_boot=1000, seed=2)
        rows.append({"subgroup": name, **rep.as_dict()})

    return rows


def write_markdown(results, out_dir, root, synthetic, tag):
    lines = [
        f"# Week 1 - clinical-only baselines ({tag})",
        "",
        f"Data: `{root}`",
        "",
    ]
    if synthetic:
        lines += [
            "> **SYNTHETIC DATA.** These numbers validate the pipeline only.",
            "> They are not findings and must not appear in the paper.",
            "",
        ]
    lines += [
        "| Model | Inputs | CV AUC | Test AUC [95% CI] | AUPRC | ECE | Sens | Spec |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for kind in KINDS:
        for config in CONFIG_ORDER:
            key = f"{config}_{kind}"
            cv, t = results["cv"][key], results["test"][key]
            lines.append(
                f"| {config} ({kind}) | {DESC[config]} "
                f"| {cv['mean']:.4f}+-{cv['std']:.4f} "
                f"| {t['auc']:.4f} [{t['auc_lo']:.4f}-{t['auc_hi']:.4f}] "
                f"| {t['auprc']:.4f} | {t['ece']:.4f} "
                f"| {t['sensitivity']:.3f} | {t['specificity']:.3f} |"
            )

    lines += ["", "## DeLong tests", "", "| Comparison | Delta AUC | p |", "|---|---|---|"]
    for name, d in results["delong"].items():
        p = d["p"]
        shown = "<1e-300" if p == 0 else f"{p:.3g}"
        lines.append(f"| {name} | {d['delta']:+.4f} | {shown} |")

    lines += ["", "## Subgroups (M2, logreg)", "", "| Subgroup | n | AUC [95% CI] |", "|---|---|---|"]
    for row in results["subgroups"]["M2_logreg"]:
        if row.get("auc") is None:
            lines.append(f"| {row['subgroup']} | {row['n']} | underpowered |")
        else:
            lines.append(
                f"| {row['subgroup']} | {row['n']} "
                f"| {row['auc']:.4f} [{row['auc_lo']:.4f}-{row['auc_hi']:.4f}] |"
            )

    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/thyroidxl_synthetic",
                    help="directory holding train.csv and test.csv")
    ap.add_argument("--out", default="outputs/week1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-sensitivity", type=float, default=0.95)
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="bootstrap resamples for AUC CIs; lower for a quick pass")
    args = ap.parse_args()

    root = Path(args.data)
    synthetic = (root / "SYNTHETIC_DO_NOT_PUBLISH.txt").exists()
    tag = "SYNTHETIC" if synthetic else "REAL"

    train = D.load_thyroidxl(root, "train")
    test = D.load_thyroidxl(root, "test")
    SP.assert_no_patient_overlap(train, test)

    train_pts = B.to_patient_level(train)
    test_pts = B.to_patient_level(test)

    results = {
        "cohort_tag": tag,
        "data_dir": str(root.resolve()),
        "coverage": [D.coverage(train), D.coverage(test)],
        "splits": [SP.describe(train, "train"), SP.describe(test, "test")],
        "cv": {},
        "test": {},
        "delong": {},
        "coefficients": {},
        "subgroups": {},
    }

    print(f"\n=== {tag} cohort: {root} ===")
    for d in results["splits"]:
        print("  ", d)
    for c in results["coverage"]:
        print("  ", c)
    print()

    test_store = {}

    for kind in KINDS:
        for config in CONFIG_ORDER:
            key = f"{config}_{kind}"
            oof, fold_aucs = cross_validate(train_pts, config, kind, seed=args.seed)
            results["cv"][key] = {
                "fold_aucs": [round(a, 4) for a in fold_aucs],
                "mean": round(float(np.mean(fold_aucs)), 4),
                "std": round(float(np.std(fold_aucs)), 4),
            }

            # Threshold from out-of-fold training predictions only.
            thr = M.threshold_at_sensitivity(
                train_pts[S.LABEL], oof, target=args.target_sensitivity
            )
            fitted = B.fit_baseline(train_pts, config, kind=kind, seed=args.seed)
            prob = fitted.predict_proba(test_pts)
            test_store[key] = prob

            rep = M.evaluate(test_pts[S.LABEL], prob, threshold=thr,
                             n_boot=args.n_boot, seed=args.seed)
            results["test"][key] = rep.as_dict()

            coefs = fitted.coefficients()
            if coefs:
                results["coefficients"][key] = {k: round(v, 4) for k, v in coefs.items()}

            print(
                f"  {key:12s} CV {np.mean(fold_aucs):.4f}+-{np.std(fold_aucs):.4f}"
                f"   TEST AUC {rep.auc:.4f} [{rep.auc_lo:.4f}-{rep.auc_hi:.4f}]"
                f"  AUPRC {rep.auprc:.4f}  ECE {rep.ece:.4f}"
            )

    print()
    for kind in KINDS:
        for a, b in (("M2", "M0"), ("M2", "M1"), ("M0", "M1")):
            ka, kb = f"{a}_{kind}", f"{b}_{kind}"
            auc_a, auc_b, z, p = M.delong_test(
                test_pts[S.LABEL], test_store[ka], test_store[kb]
            )
            results["delong"][f"{ka}_vs_{kb}"] = {
                "auc_a": round(auc_a, 4),
                "auc_b": round(auc_b, 4),
                "delta": round(auc_a - auc_b, 4),
                "z": None if np.isnan(z) else round(z, 3),
                # Not rounded: p can be ~1e-26 and round(p, 6) would store 0.0.
                "p": p,
            }
            print(f"  DeLong {ka} vs {kb}: delta {auc_a - auc_b:+.4f}  p={p:.4g}")

    best = "M2_logreg"
    thr_best = results["test"][best]["threshold"]
    results["subgroups"][best] = subgroup_table(test_pts, test_store[best], thr_best)
    print(f"\n  subgroups for {best}:")
    for row in results["subgroups"][best]:
        auc = row.get("auc")
        shown = f"{auc:.4f}" if isinstance(auc, float) else "n/a (underpowered)"
        print(f"    {row['subgroup']:16s} n={row['n']:5d}  AUC {shown}")

    out_dir = Path(args.out)
    D.save_report(results, out_dir / "results.json")
    write_markdown(results, out_dir, root, synthetic, tag)
    print(f"\n  wrote {(out_dir / 'results.json').resolve()}")
    print(f"  wrote {(out_dir / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
