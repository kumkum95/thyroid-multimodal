"""Clinical-only baselines M0, M1, M2.

These are the models the published literature skipped. M2 (age + sex + TI-RADS)
is the headline baseline: if it already scores ~0.90, then the reported gains
from "multimodal" fusion in this field are largely TI-RADS doing the work.

All feature statistics are fitted on the training split only and reused
unchanged everywhere else, so external cohorts get no information from their own
distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from . import schema as S

# Which clinical fields each configuration is allowed to see.
CONFIGS: dict[str, list[str]] = {
    "M0": [S.TIRADS],                      # endogenous only
    "M1": [S.AGE, S.SEX],                  # exogenous only
    "M2": [S.AGE, S.SEX, S.TIRADS],        # full clinical
}


def to_patient_level(df: pd.DataFrame, *, tirads_agg: str = "max") -> pd.DataFrame:
    """One row per patient with clinical fields and the label.

    `tirads_agg="max"` takes the most suspicious category across a patient's
    images. That matches clinical practice (management follows the worst nodule)
    and is the only choice that does not leak: a mean would blend in information
    about how many images were taken, which correlates with workup intensity and
    therefore with the outcome.
    """
    if tirads_agg not in {"max", "first", "mean"}:
        raise ValueError(f"unsupported tirads_agg: {tirads_agg}")
    out = (
        df.groupby(S.PATIENT_ID)
        .agg(
            age=(S.AGE, "first"),
            sex=(S.SEX, "first"),
            tirads=(S.TIRADS, tirads_agg),
            label=(S.LABEL, "first"),
            n_images=(S.PATIENT_ID, "size"),
        )
        .reset_index()
    )
    if tirads_agg == "mean":
        out[S.TIRADS] = out[S.TIRADS].round()
    return out


@dataclass
class FeatureSpec:
    """Fitted feature transform. Fit on train, apply everywhere."""

    fields: list[str]
    age_mean: float = 0.0
    age_std: float = 1.0
    tirads_fill: float = 3.0
    age_fill: float = 50.0
    sex_fill: float = 1.0
    names: list[str] = field(default_factory=list)

    @classmethod
    def fit(cls, pts: pd.DataFrame, fields: list[str]) -> "FeatureSpec":
        spec = cls(fields=list(fields))
        if S.AGE in fields:
            age = pd.to_numeric(pts[S.AGE], errors="coerce")
            spec.age_mean = float(age.mean())
            spec.age_std = float(age.std() or 1.0)
            spec.age_fill = float(age.median())
        if S.TIRADS in fields:
            spec.tirads_fill = float(pd.to_numeric(pts[S.TIRADS], errors="coerce").median())
        if S.SEX in fields:
            spec.sex_fill = float(pd.to_numeric(pts[S.SEX], errors="coerce").median())
        spec.names = spec._names()
        return spec

    def _names(self) -> list[str]:
        names: list[str] = []
        if S.AGE in self.fields:
            names += ["age_z", "age_z_sq"]
        if S.SEX in self.fields:
            names += ["is_female"]
        if S.TIRADS in self.fields:
            # Thermometer encoding: respects the ordering of TI-RADS, unlike
            # one-hot, and stays linear in risk, unlike a raw scalar.
            names += [f"tirads_ge_{k}" for k in range(2, S.TIRADS_MAX + 1)]
        return names

    def transform(self, pts: pd.DataFrame) -> np.ndarray:
        cols: list[np.ndarray] = []
        if S.AGE in self.fields:
            age = pd.to_numeric(pts[S.AGE], errors="coerce").fillna(self.age_fill)
            z = ((age - self.age_mean) / self.age_std).to_numpy(dtype=float)
            cols += [z, z ** 2]          # malignancy risk is U-shaped in age
        if S.SEX in self.fields:
            sex = pd.to_numeric(pts[S.SEX], errors="coerce").fillna(self.sex_fill)
            cols.append(sex.to_numpy(dtype=float))
        if S.TIRADS in self.fields:
            tr = pd.to_numeric(pts[S.TIRADS], errors="coerce").fillna(self.tirads_fill)
            tr = tr.clip(S.TIRADS_MIN, S.TIRADS_MAX).to_numpy(dtype=float)
            for k in range(2, S.TIRADS_MAX + 1):
                cols.append((tr >= k).astype(float))
        return np.column_stack(cols)


def make_model(kind: str, seed: int = 0):
    if kind == "logreg":
        return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
    if kind == "gbdt":
        return HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=400,
            l2_regularization=1.0, early_stopping=True,
            validation_fraction=0.15, random_state=seed,
        )
    raise ValueError(f"unknown model kind: {kind}")


@dataclass
class FittedBaseline:
    config: str
    kind: str
    spec: FeatureSpec
    model: object

    def predict_proba(self, pts: pd.DataFrame) -> np.ndarray:
        X = self.spec.transform(pts)
        return self.model.predict_proba(X)[:, 1]

    def coefficients(self) -> dict[str, float] | None:
        if not hasattr(self.model, "coef_"):
            return None
        return dict(zip(self.spec.names, map(float, self.model.coef_[0])))


def fit_baseline(train_pts: pd.DataFrame, config: str, kind: str = "logreg", seed: int = 0):
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; expected one of {list(CONFIGS)}")
    fields = CONFIGS[config]
    spec = FeatureSpec.fit(train_pts, fields)
    model = make_model(kind, seed=seed)
    X = spec.transform(train_pts)
    y = train_pts[S.LABEL].to_numpy(dtype=int)
    model.fit(X, y)
    return FittedBaseline(config=config, kind=kind, spec=spec, model=model)
