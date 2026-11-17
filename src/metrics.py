"""Evaluation metrics.

Everything here is patient-level by construction. Bootstraps resample
*patients*, never images: resampling images treats 2.8 correlated views as
independent observations and produces confidence intervals that are too narrow.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


# ---------------------------------------------------------------- DeLong test

def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _fast_delong(preds_pos_first: np.ndarray, n_pos: int):
    """Sun & Xu (2014) fast DeLong covariance.

    `preds_pos_first` is (k, n) with the positive cases in the first n_pos
    columns. Returns (aucs, covariance).
    """
    m = n_pos
    n = preds_pos_first.shape[1] - m
    if m < 2 or n < 2:
        raise ValueError("DeLong needs at least 2 positives and 2 negatives")
    k = preds_pos_first.shape[0]

    pos = preds_pos_first[:, :m]
    neg = preds_pos_first[:, m:]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(preds_pos_first[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01).reshape(k, k)
    sy = np.cov(v10).reshape(k, k)
    cov = sx / m + sy / n
    return aucs, cov


def delong_test(y_true, score_a, score_b):
    """Two-sided DeLong test for two correlated ROC curves on the same cases.

    Use this for M4-vs-M3 and M6-vs-M2: both models score the same patients, so
    an unpaired test would throw away the pairing and lose power.
    """
    y_true = np.asarray(y_true).astype(int)
    score_a = np.asarray(score_a, dtype=float)
    score_b = np.asarray(score_b, dtype=float)

    order = np.argsort(-y_true, kind="mergesort")  # positives first, stable
    n_pos = int(y_true.sum())
    preds = np.vstack((score_a, score_b))[:, order]

    aucs, cov = _fast_delong(preds, n_pos)
    contrast = np.array([[1.0, -1.0]])
    # The product is (1, 1); NumPy 2.x refuses float() on a 2-D array.
    var = float((contrast @ cov @ contrast.T).item())
    if var <= 0:
        return float(aucs[0]), float(aucs[1]), np.nan, 1.0
    z = float((aucs[0] - aucs[1]) / np.sqrt(var))
    p = float(2 * stats.norm.sf(abs(z)))
    return float(aucs[0]), float(aucs[1]), z, p


# ---------------------------------------------------------------- calibration

def expected_calibration_error(y_true, prob, n_bins: int = 10) -> float:
    """Equal-width binned ECE."""
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - prob[mask].mean())
    return float(ece)


def reliability_curve(y_true, prob, n_bins: int = 10):
    """Bin centres, observed frequency and mean predicted probability."""
    y_true = np.asarray(y_true, dtype=float)
    prob = np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append((float(prob[mask].mean()), float(y_true[mask].mean()), int(mask.sum())))
    return rows


# ---------------------------------------------------------------- bootstrap

def bootstrap_auc_ci(y_true, score, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Percentile bootstrap CI for AUC, resampling patients with replacement."""
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = roc_auc_score(y_true, score)
    draws = np.empty(n_boot)
    filled = 0
    attempts = 0
    while filled < n_boot and attempts < n_boot * 10:
        attempts += 1
        pick = rng.integers(0, n, n)
        yb = y_true[pick]
        if yb.min() == yb.max():      # degenerate resample, no both classes
            continue
        draws[filled] = roc_auc_score(yb, score[pick])
        filled += 1
    draws = draws[:filled]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


# ---------------------------------------------------------------- summary

@dataclass
class Report:
    n: int
    n_pos: int
    prevalence: float
    auc: float
    auc_lo: float
    auc_hi: float
    auprc: float
    brier: float
    ece: float
    sensitivity: float
    specificity: float
    threshold: float

    def as_dict(self):
        return asdict(self)


def evaluate(y_true, prob, *, threshold: float, n_boot: int = 2000, seed: int = 0) -> Report:
    """Full patient-level report at a threshold fixed on the dev split.

    The threshold is an argument on purpose: choosing it from the test set is the
    single most common way these papers overstate sensitivity.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)

    auc, lo, hi = bootstrap_auc_ci(y_true, prob, n_boot=n_boot, seed=seed)
    pred = (prob >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())

    return Report(
        n=len(y_true),
        n_pos=int(y_true.sum()),
        prevalence=float(y_true.mean()),
        auc=auc,
        auc_lo=lo,
        auc_hi=hi,
        auprc=float(average_precision_score(y_true, prob)),
        brier=float(brier_score_loss(y_true, prob)),
        ece=expected_calibration_error(y_true, prob),
        sensitivity=tp / (tp + fn) if (tp + fn) else float("nan"),
        specificity=tn / (tn + fp) if (tn + fp) else float("nan"),
        threshold=float(threshold),
    )


def threshold_at_sensitivity(y_true, prob, target: float = 0.95) -> float:
    """Lowest threshold that still reaches `target` sensitivity on this split.

    Call this on dev only; pass the result into `evaluate` for the test split.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob, dtype=float)
    for thr in np.unique(np.sort(prob))[::-1]:
        sens = ((prob >= thr) & (y_true == 1)).sum() / max(y_true.sum(), 1)
        if sens >= target:
            return float(thr)
    return 0.0
