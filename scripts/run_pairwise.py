"""Paired DeLong across M3-M6, using the saved per-patient predictions.

    python scripts/run_pairwise.py --runs outputs/kaggle/outputs/image

Gap 3 turns on M4 against M5: whether exogenous demographics and the
image-derived TI-RADS contribute differently once the image is already in the
model. An unpaired comparison would throw away the fact that every
configuration scores the same 739 patients, so DeLong is used throughout.

Seeds are handled two ways, because they answer different questions. Per-seed
tests say whether a difference survives in each individual training run. The
seed-averaged ensemble says whether it survives once optimisation noise is
averaged out, which is the comparison the headline table should carry -- a
single seed of M4 ranged from 0.5811 to 0.9513 in this run, so a per-seed
verdict alone would be close to arbitrary.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data as D
from src import schema as S
from src.metrics import delong_test

CONFIGS = ["M3", "M4", "M5", "M6"]


def load(runs: Path, config: str, seed: int) -> tuple[list[str], np.ndarray]:
    ids = json.loads((runs / f"{config}_seed{seed}_test_ids.json").read_text())
    prob = np.load(runs / f"{config}_seed{seed}_test_prob.npy")
    return ids, prob


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/kaggle/outputs/image")
    ap.add_argument("--data", default="data/thyroidxl")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="outputs/pairwise")
    args = ap.parse_args()

    runs, out_dir = Path(args.runs), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import roc_auc_score

    from src.baselines import to_patient_level

    # Labels, looked up by patient id so alignment is never assumed.
    test_df = D.load_thyroidxl(Path(args.data), "test")
    labels = to_patient_level(test_df).set_index(S.PATIENT_ID)[S.LABEL]

    ids, _ = load(runs, CONFIGS[0], args.seeds[0])
    y = labels.reindex(ids).to_numpy(dtype=int)
    print(f"  {len(y)} patients, prevalence {y.mean():.4f}\n")

    probs: dict[str, np.ndarray] = {}
    per_seed: dict[str, list[np.ndarray]] = {}
    for config in CONFIGS:
        collected = []
        for seed in args.seeds:
            seed_ids, prob = load(runs, config, seed)
            if seed_ids != ids:
                raise SystemExit(f"{config} seed {seed} has a different id order")
            collected.append(prob)
        per_seed[config] = collected
        probs[config] = np.mean(collected, axis=0)

    print("  per-seed AUC, and the seed-averaged ensemble:")
    print(f"  {'cfg':4s} " + " ".join(f"{'seed'+str(s):>8s}" for s in args.seeds)
          + f" {'mean':>8s} {'sd':>7s} {'ensemble':>9s}")
    summary = {}
    for config in CONFIGS:
        aucs = [float(roc_auc_score(y, p)) for p in per_seed[config]]
        ens = float(roc_auc_score(y, probs[config]))
        print(f"  {config:4s} " + " ".join(f"{a:8.4f}" for a in aucs)
              + f" {np.mean(aucs):8.4f} {np.std(aucs, ddof=1):7.4f} {ens:9.4f}")
        summary[config] = {
            "per_seed_auc": [round(a, 4) for a in aucs],
            "mean_auc": round(float(np.mean(aucs)), 4),
            "sd_auc": round(float(np.std(aucs, ddof=1)), 4),
            "ensemble_auc": round(ens, 4),
        }

    print("\n  paired DeLong on the seed-averaged ensembles:")
    pairs = {}
    for a, b in combinations(CONFIGS, 2):
        auc_a, auc_b, z, p = delong_test(y, probs[a], probs[b])
        flag = "" if p >= 0.05 else "  *"
        print(f"  {a} vs {b}:  {auc_a:.4f} vs {auc_b:.4f}  "
              f"diff {auc_a - auc_b:+.4f}  z {z:+.3f}  p {p:.3g}{flag}")
        pairs[f"{a}_vs_{b}"] = {
            "auc_a": round(auc_a, 4), "auc_b": round(auc_b, 4),
            "diff": round(auc_a - auc_b, 4),
            "z": round(float(z), 4), "p": p,
            "significant_at_05": bool(p < 0.05),
        }

    report = {
        "n": int(len(y)),
        "prevalence": round(float(y.mean()), 4),
        "seeds": args.seeds,
        "per_config": summary,
        "pairwise_ensemble_delong": pairs,
    }
    D.save_report(report, out_dir / "pairwise.json")
    print(f"\n  wrote {(out_dir / 'pairwise.json').resolve()}")


if __name__ == "__main__":
    main()
