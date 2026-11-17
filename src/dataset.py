"""Patient-level dataset: one item is one patient, carrying K images.

The label is the patient's pathology, so the bag -- not the image -- is the
unit of learning and of evaluation. Batching bags of unequal size needs padding
plus a mask, and the mask must reach the attention pool, or padded slots take
attention mass away from real images.

Clinical features are built with the same FeatureSpec used by the M0-M2 tabular
baselines. That is deliberate: if M2 and M4 encoded age or TI-RADS differently,
the comparison between them would measure the encoding rather than the image.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import schema as S
from .baselines import FeatureSpec

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DataConfig:
    image_size: int = 384
    max_images: int = 6       # bounds VRAM; patients above this are sampled
    augment: bool = False
    allow_missing_images: bool = False


def build_transforms(cfg: DataConfig, train: bool):
    from torchvision import transforms as T

    if train and cfg.augment:
        # No aggressive random cropping: a crop can remove the nodule entirely
        # while the patient-level label still says malignant, which teaches the
        # model to guess from background texture.
        return T.Compose([
            T.Resize((cfg.image_size, cfg.image_size)),
            T.RandomHorizontalFlip(),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize((cfg.image_size, cfg.image_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _placeholder_image(image_id: str, size: int) -> torch.Tensor:
    """Deterministic noise standing in for an absent image file.

    Only reachable with allow_missing_images=True, which exists so the training
    loop can be exercised before the gated cohort is downloaded. It carries no
    signal, so a model trained on it must score ~0.5; anything higher means the
    label is leaking through a non-image path.
    """
    seed = int(hashlib.sha256(image_id.encode()).hexdigest()[:8], 16)
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(3, size, size, generator=gen) * 0.5


class PatientBags(Dataset):
    """One item per patient: (images, mask, exo, endo, label, patient_id)."""

    def __init__(
        self,
        df: pd.DataFrame,
        spec_exo: FeatureSpec | None,
        spec_endo: FeatureSpec | None,
        cfg: DataConfig,
        train: bool = False,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.train = train
        self.tf = build_transforms(cfg, train)
        self.rng = np.random.default_rng(seed)

        self.patients: list[str] = []
        self.paths: dict[str, list[str]] = {}
        self.ids: dict[str, list[str]] = {}
        for pid, group in df.groupby(S.PATIENT_ID, sort=True):
            self.patients.append(pid)
            self.paths[pid] = group[S.IMAGE_PATH].astype(str).tolist()
            self.ids[pid] = group[S.IMAGE_ID].astype(str).tolist()

        # Patient-level clinical table, encoded with the shared FeatureSpec.
        from .baselines import to_patient_level

        pts = to_patient_level(df).set_index(S.PATIENT_ID).loc[self.patients]
        self.labels = pts[S.LABEL].to_numpy(dtype=np.float32)
        self.exo = spec_exo.transform(pts).astype(np.float32) if spec_exo else None
        self.endo = spec_endo.transform(pts).astype(np.float32) if spec_endo else None

    def __len__(self) -> int:
        return len(self.patients)

    def _load(self, path: str, image_id: str) -> torch.Tensor:
        p = Path(path)
        if not p.exists():
            if not self.cfg.allow_missing_images:
                raise FileNotFoundError(
                    f"image not found: {p}\n"
                    "Pass allow_missing_images=True only to exercise the loop "
                    "without real data."
                )
            return _placeholder_image(image_id, self.cfg.image_size)
        from PIL import Image

        return self.tf(Image.open(p).convert("RGB"))

    def __getitem__(self, idx: int):
        pid = self.patients[idx]
        paths, ids = self.paths[pid], self.ids[pid]

        order = list(range(len(paths)))
        if len(order) > self.cfg.max_images:
            # Sample during training for variety; take the first k at eval so a
            # patient's score does not change between runs.
            if self.train:
                order = list(self.rng.choice(order, self.cfg.max_images, replace=False))
            else:
                order = order[: self.cfg.max_images]

        images = torch.stack([self._load(paths[i], ids[i]) for i in order])
        return {
            "images": images,
            "exo": torch.from_numpy(self.exo[idx]) if self.exo is not None else None,
            "endo": torch.from_numpy(self.endo[idx]) if self.endo is not None else None,
            "label": torch.tensor(self.labels[idx]),
            "patient_id": pid,
        }


def collate_bags(batch: list[dict]) -> dict:
    """Pad bags to the batch maximum and emit the mask the pool needs."""
    k_max = max(item["images"].shape[0] for item in batch)
    b = len(batch)
    c, h, w = batch[0]["images"].shape[1:]

    images = torch.zeros(b, k_max, c, h, w)
    mask = torch.zeros(b, k_max, dtype=torch.bool)
    for i, item in enumerate(batch):
        k = item["images"].shape[0]
        images[i, :k] = item["images"]
        mask[i, :k] = True

    out = {
        "images": images,
        "mask": mask,
        "label": torch.stack([item["label"] for item in batch]),
        "patient_id": [item["patient_id"] for item in batch],
    }
    for key in ("exo", "endo"):
        if batch[0][key] is not None:
            out[key] = torch.stack([item[key] for item in batch])
    return out


def fit_specs(train_df: pd.DataFrame) -> tuple[FeatureSpec, FeatureSpec]:
    """Fit exogenous and endogenous encoders on the training split only.

    Returned unchanged for every other split and every external cohort, so no
    evaluation set contributes information about its own distribution.
    """
    from .baselines import to_patient_level

    pts = to_patient_level(train_df)
    return (
        FeatureSpec.fit(pts, [S.AGE, S.SEX]),
        FeatureSpec.fit(pts, [S.TIRADS]),
    )
