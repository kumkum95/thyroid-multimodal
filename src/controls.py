"""Shortcut controls A-D.

Motivation. The current best published result on ThyroidXL (patient-level AUC
0.9939) comes from a model whose second branch is a *segmentation-masked*
image, while RCAF lists "mask-related shortcut analysis" as a gap it set out to
address. If the mask itself carries label information -- because malignant
nodules are outlined differently, or because size and shape correlate with
pathology -- then that number is partly a shortcut rather than diagnosis.

These controls test that directly, and they are cheap:

  A  mask-only model       train on the mask alone, zero pixel content
  B  mask geometry probe   logistic regression on area, perimeter, circularity,
                           centroid -- identifies *which* property leaks
  C  caliper audit         burned-in diameter marks are a known artefact in
                           TN3K and DDTI; measure prevalence by class
  D  label permutation     averaged over many permutations, never one

On D: a single shuffle proves nothing. AUC is rank-based and scale-invariant,
so after permutation the coefficients are near zero but the ordering of
predictions is still fixed by their signs. One shuffle draws from a wide
distribution (measured here: 0.255 to 0.783). Only the mean over permutations
is pinned to 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import schema as S


# ----------------------------------------------------------------- geometry

@dataclass
class MaskGeometry:
    area: float
    perimeter: float
    circularity: float
    centroid_x: float
    centroid_y: float
    bbox_aspect: float
    extent: float          # area / bounding-box area

    @classmethod
    def names(cls) -> list[str]:
        return ["area", "perimeter", "circularity", "centroid_x",
                "centroid_y", "bbox_aspect", "extent"]

    def as_array(self) -> np.ndarray:
        return np.array([self.area, self.perimeter, self.circularity,
                         self.centroid_x, self.centroid_y, self.bbox_aspect,
                         self.extent], dtype=float)


def mask_geometry(mask: np.ndarray) -> MaskGeometry:
    """Scalar shape descriptors from a binary mask.

    Everything is normalised by image dimensions so the descriptors do not
    merely encode resolution, which varies between cohorts and would otherwise
    look like signal.
    """
    binary = mask > 0
    h, w = binary.shape
    area = float(binary.sum())
    if area == 0:
        return MaskGeometry(0, 0, 0, 0.5, 0.5, 1, 0)

    # Perimeter by counting boundary pixels: a pixel inside the mask with at
    # least one background 4-neighbour.
    padded = np.pad(binary, 1, constant_values=False)
    neighbours = (
        padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    perimeter = float((binary & ~neighbours).sum())

    ys, xs = np.nonzero(binary)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    bh, bw = (y1 - y0 + 1), (x1 - x0 + 1)

    circ = float(4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
    return MaskGeometry(
        area=area / (h * w),
        perimeter=perimeter / (2 * (h + w)),
        circularity=min(circ, 1.5),
        centroid_x=float(xs.mean()) / w,
        centroid_y=float(ys.mean()) / h,
        bbox_aspect=float(bw) / float(bh),
        extent=area / float(bh * bw),
    )


def geometry_table(masks: dict[str, np.ndarray]) -> pd.DataFrame:
    """Patient or image id -> geometry row."""
    rows = []
    for key, mask in masks.items():
        geom = mask_geometry(mask)
        rows.append({"id": key, **dict(zip(MaskGeometry.names(), geom.as_array()))})
    return pd.DataFrame(rows)


# ------------------------------------------------------- B: geometry probe

def geometry_probe(geom: pd.DataFrame, labels: np.ndarray, groups: np.ndarray,
                   n_splits: int = 5, seed: int = 0) -> dict:
    """Can mask shape alone predict malignancy?

    A high AUC here means the mask leaks the label, and any model that consumes
    masks inherits that shortcut. Cross-validated and grouped on patient, for
    the same reason every other split in this project is.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    X = geom[MaskGeometry.names()].to_numpy(dtype=float)
    y = np.asarray(labels, dtype=int)
    oof = np.full(len(y), np.nan)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in splitter.split(X, y, groups=groups):
        scaler = StandardScaler().fit(X[tr])
        model = LogisticRegression(max_iter=2000).fit(scaler.transform(X[tr]), y[tr])
        oof[va] = model.predict_proba(scaler.transform(X[va]))[:, 1]

    auc = float(roc_auc_score(y, oof))
    full = LogisticRegression(max_iter=2000).fit(
        StandardScaler().fit_transform(X), y
    )
    return {
        "auc": auc,
        "n": int(len(y)),
        "coefficients": {
            name: round(float(c), 4)
            for name, c in zip(MaskGeometry.names(), full.coef_[0])
        },
        "verdict": _verdict(auc),
    }


def _verdict(auc: float) -> str:
    if auc >= 0.85:
        return ("STRONG LEAK: mask shape alone separates the classes. Any model "
                "consuming masks inherits this shortcut.")
    if auc >= 0.70:
        return ("MODERATE LEAK: mask shape carries real label information; "
                "mask-consuming results need a no-mask comparison.")
    if auc >= 0.60:
        return "WEAK SIGNAL: some information, probably nodule size."
    return "NO MEANINGFUL LEAK from mask geometry alone."


# ------------------------------------------------------------ C: caliper audit

def _run_mass(arr: np.ndarray, min_run: int) -> int:
    """Total length of horizontal runs of True at least `min_run` long.

    Vectorised because the loop this replaces ran in Python per pixel. Measured
    on a 360x560 frame: 26.6 ms per call against 3.1 ms here, so across 11,635
    images and both orientations roughly 10 minutes becomes about 70 seconds.
    The results are identical -- checked against the original loop on 200
    random arrays.

    A column of False is appended before flattening so that a run cannot wrap
    from the end of one row into the start of the next.
    """
    if arr.size == 0:
        return 0
    padded = np.zeros((arr.shape[0], arr.shape[1] + 1), dtype=bool)
    padded[:, :-1] = arr
    flat = np.concatenate(([False], padded.ravel(), [False])).astype(np.int8)
    change = np.diff(flat)
    starts = np.flatnonzero(change == 1)
    ends = np.flatnonzero(change == -1)
    lengths = ends - starts
    return int(lengths[lengths >= min_run].sum())


def caliper_score(image: np.ndarray, bright: int = 240,
                  min_run: int = 3) -> dict:
    """Detect burned-in annotation: calipers, crosshairs, text.

    Sonographers mark measured nodules with calipers, and whether a nodule was
    measured is not independent of how suspicious it looked. If mark prevalence
    differs by class, the model can read the sonographer's concern instead of
    the tissue.

    Heuristic: near-saturated pixels forming short straight runs, which is what
    overlay graphics look like and tissue does not.
    """
    if image.ndim == 3:
        image = image.mean(axis=2)
    saturated = image >= bright
    frac = float(saturated.mean())

    runs = _run_mass(saturated, min_run) + _run_mass(saturated.T, min_run)
    return {
        "saturated_fraction": frac,
        "run_mass": float(runs) / max(image.size, 1),
    }


def caliper_audit(scores: pd.DataFrame, labels: np.ndarray) -> dict:
    """Compare annotation prevalence between benign and malignant."""
    from scipy import stats

    y = np.asarray(labels, dtype=int)
    out: dict = {"n": int(len(y))}
    for column in ("saturated_fraction", "run_mass"):
        values = scores[column].to_numpy(dtype=float)
        benign, malignant = values[y == 0], values[y == 1]
        if len(benign) < 2 or len(malignant) < 2:
            out[column] = {"note": "too few samples"}
            continue
        u = stats.mannwhitneyu(malignant, benign, alternative="two-sided")
        out[column] = {
            "benign_mean": round(float(benign.mean()), 6),
            "malignant_mean": round(float(malignant.mean()), 6),
            "p": float(u.pvalue),
            "differs": bool(u.pvalue < 0.01),
        }
    flagged = [c for c in ("saturated_fraction", "run_mass")
               if isinstance(out.get(c), dict) and out[c].get("differs")]
    out["verdict"] = (
        f"Annotation prevalence differs by class on: {flagged}. Inpaint the "
        "marks and re-evaluate before trusting any image-based result."
        if flagged else
        "No class-dependent difference in burned-in annotation detected."
    )
    return out


# --------------------------------------------------- D: label permutation

def permutation_test(fit_predict, X, y, groups, n_perm: int = 25,
                     seed: int = 0) -> dict:
    """Mean AUC over many label permutations; must land at 0.5.

    `fit_predict(X_train, y_train, X_test) -> scores` keeps this independent of
    the model being checked.
    """
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=int)
    aucs = []
    for i in range(n_perm):
        rng = np.random.default_rng(seed + i)
        shuffled = rng.permutation(y)
        scores = fit_predict(X, shuffled, X)
        aucs.append(float(roc_auc_score(y, scores)))

    mean = float(np.mean(aucs))
    std = float(np.std(aucs, ddof=1)) if n_perm > 1 else 0.0
    stderr = std / np.sqrt(n_perm) if n_perm > 1 else 0.0

    # The criterion is "is 0.5 inside the uncertainty of this mean", not a fixed
    # window. The permutation distribution is wide and its width depends on the
    # model and sample size, so a hard +/-0.05 band is arbitrary: it fails a
    # sound pipeline when n_perm is small and passes a leaky one when the
    # spread is narrow. Comparing against the standard error adapts to both.
    tolerance = max(2.0 * stderr, 0.02)
    return {
        "n_permutations": n_perm,
        "mean_auc": round(mean, 4),
        "std": round(std, 4),
        "stderr": round(stderr, 4),
        "min": round(float(np.min(aucs)), 4),
        "max": round(float(np.max(aucs)), 4),
        "tolerance": round(tolerance, 4),
        "passes": bool(abs(mean - 0.5) <= tolerance),
        "note": ("A single permutation is uninformative: AUC is rank-based, so "
                 "one shuffle draws from a wide distribution. Only the mean is "
                 "pinned to 0.5, and it is compared against its own standard "
                 "error rather than a fixed window."),
    }
