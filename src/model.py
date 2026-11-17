"""DEX-Thyroid model: image branch + two separate clinical branches.

The architecture is deliberately plain. The contribution of this project is a
measurement, not a network, so every configuration must differ from its
neighbour by exactly one input. Anything clever here would become a confound:
the first question a reviewer asks is "is the gain from fusion, or from a
fancier backbone?", and the design has to make that question unanswerable.

Two clinical branches, never sharing weights:

    E (exogenous)   age, sex     -- independent of the pixels
    N (endogenous)  TI-RADS      -- a radiologist's reading OF the pixels

Keeping them apart is the point. Fusing TI-RADS is close to circular, which is
why the published gains from "multimodal" fusion are small; fusing age and sex
is a claim nobody has tested on this benchmark.

Patient-level by construction: a patient contributes K images, a gated-attention
MIL pool reduces them to one embedding, and the label is the patient's
pathology. The head stays linear so a prediction can be decomposed per modality.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Which inputs each configuration may see. The image branch is present iff
# "image" appears here; the clinical branches iff their fields do.
CONFIGS: dict[str, list[str]] = {
    "M3": ["image"],
    "M4": ["image", "exo"],           # the uncontaminated fusion claim
    "M5": ["image", "endo"],          # replicates the TI-RADS fusion preprint
    "M6": ["image", "exo", "endo"],   # full model
}


@dataclass
class ModelConfig:
    config: str = "M6"
    backbone: str = "efficientnet_b4"
    embed_dim: int = 512
    clinical_dim: int = 64
    attn_dim: int = 256
    dropout: float = 0.25
    pretrained: bool = True
    n_tirads: int = 5

    @property
    def uses_image(self) -> bool:
        return "image" in CONFIGS[self.config]

    @property
    def uses_exo(self) -> bool:
        return "exo" in CONFIGS[self.config]

    @property
    def uses_endo(self) -> bool:
        return "endo" in CONFIGS[self.config]


class GatedAttentionMIL(nn.Module):
    """Gated attention pooling (Ilse et al., 2018).

    Reduces a patient's K image embeddings to one, weighting each by a learned
    relevance. Mean pooling would dilute a single suspicious nodule among
    several benign views, which is the opposite of how the patient-level label
    is actually determined.

    The gate (sigmoid path) lets the network suppress uninformative frames
    rather than merely ranking them, which matters because image count varies
    (1 to ~6) and says nothing about pathology.
    """

    def __init__(self, in_dim: int, attn_dim: int = 256):
        super().__init__()
        self.tanh = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Tanh())
        self.sigm = nn.Sequential(nn.Linear(in_dim, attn_dim), nn.Sigmoid())
        self.weight = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """x: (B, K, D); mask: (B, K) with True for real images.

        Returns (pooled (B, D), attention (B, K)).
        """
        scores = self.weight(self.tanh(x) * self.sigm(x)).squeeze(-1)   # (B, K)
        if mask is not None:
            # Padded slots must not receive attention mass. Using -inf before
            # softmax is what keeps a 1-image patient from being diluted by
            # padding in a batch sized for a 6-image patient.
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bk,bkd->bd", attn, x)
        return pooled, attn


class ClinicalEncoder(nn.Module):
    """Tiny MLP for a handful of tabular variables.

    Kept deliberately small: with two or three inputs, anything larger
    memorises the training split instead of learning a risk relationship.
    """

    def __init__(self, in_dim: int, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Dropout(dropout),
            nn.Linear(32, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return (feature extractor, feature dim).

    torchvision is used rather than timm so the kernel has no extra dependency
    to install at run time.
    """
    import torchvision.models as tvm

    factories = {
        "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights),
        "efficientnet_b4": (tvm.efficientnet_b4, tvm.EfficientNet_B4_Weights),
        "convnext_tiny": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights),
        "resnet50": (tvm.resnet50, tvm.ResNet50_Weights),
    }
    if name not in factories:
        raise ValueError(f"unknown backbone {name!r}; expected {list(factories)}")

    factory, weights_enum = factories[name]
    model = factory(weights=weights_enum.DEFAULT if pretrained else None)

    if name.startswith("efficientnet"):
        feat_dim = model.classifier[1].in_features
        model.classifier = nn.Identity()
    elif name.startswith("convnext"):
        feat_dim = model.classifier[2].in_features
        model.classifier = nn.Sequential(
            model.classifier[0], model.classifier[1], nn.Identity()
        )
    else:
        feat_dim = model.fc.in_features
        model.fc = nn.Identity()
    return model, feat_dim


class DexThyroid(nn.Module):
    """Image + exogenous + endogenous fusion with readable modality gates.

    Fusion is gated-additive rather than cross-attention on purpose: the gates
    are directly interpretable as each modality's contribution, which is the
    research question itself. A linear head then allows per-modality
    decomposition of an individual prediction.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.embed_dim

        if cfg.uses_image:
            self.backbone, feat_dim = build_backbone(cfg.backbone, cfg.pretrained)
            self.project = nn.Sequential(
                nn.Linear(feat_dim, d), nn.GELU(), nn.LayerNorm(d)
            )
            self.pool = GatedAttentionMIL(d, cfg.attn_dim)

        # age_z, age_z^2, is_female -- the squared term because malignancy risk
        # is not monotone in age.
        if cfg.uses_exo:
            self.exo = ClinicalEncoder(3, cfg.clinical_dim)
            self.exo_proj = nn.Linear(cfg.clinical_dim, d)

        # Thermometer encoding of TI-RADS: preserves the ordering that one-hot
        # discards, without forcing linearity in risk the way a raw scalar does.
        if cfg.uses_endo:
            self.endo = ClinicalEncoder(cfg.n_tirads - 1, cfg.clinical_dim)
            self.endo_proj = nn.Linear(cfg.clinical_dim, d)

        self.n_modalities = sum(
            [cfg.uses_image, cfg.uses_exo, cfg.uses_endo]
        )
        if self.n_modalities == 0:
            raise ValueError(f"config {cfg.config!r} selects no inputs")

        if self.n_modalities > 1:
            self.gate = nn.Linear(d * self.n_modalities, self.n_modalities)

        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(d, 1)       # linear, for per-modality attribution

    def forward(
        self,
        images: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        exo: torch.Tensor | None = None,
        endo: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        parts: list[torch.Tensor] = []
        out: dict[str, torch.Tensor] = {}

        if self.cfg.uses_image:
            if images is None:
                raise ValueError(f"{self.cfg.config} requires images")
            b, k = images.shape[:2]
            flat = images.flatten(0, 1)                    # (B*K, C, H, W)
            feats = self.project(self.backbone(flat))      # (B*K, D)
            feats = feats.view(b, k, -1)
            pooled, attn = self.pool(feats, mask)
            out["attention"] = attn
            parts.append(pooled)

        if self.cfg.uses_exo:
            if exo is None:
                raise ValueError(f"{self.cfg.config} requires exogenous features")
            parts.append(self.exo_proj(self.exo(exo)))

        if self.cfg.uses_endo:
            if endo is None:
                raise ValueError(f"{self.cfg.config} requires endogenous features")
            parts.append(self.endo_proj(self.endo(endo)))

        if len(parts) == 1:
            fused = parts[0]
        else:
            gates = torch.softmax(self.gate(torch.cat(parts, dim=-1)), dim=-1)
            fused = sum(gates[:, i : i + 1] * p for i, p in enumerate(parts))
            # Reported per cohort: if the TI-RADS gate dominates internally but
            # collapses externally, that is itself a result.
            out["gates"] = gates

        out["logit"] = self.head(self.dropout(fused)).squeeze(-1)
        return out

    @torch.no_grad()
    def modality_names(self) -> list[str]:
        names = []
        if self.cfg.uses_image:
            names.append("image")
        if self.cfg.uses_exo:
            names.append("exogenous")
        if self.cfg.uses_endo:
            names.append("endogenous")
        return names


def build_model(config: str, **kwargs) -> DexThyroid:
    if config not in CONFIGS:
        raise ValueError(f"unknown config {config!r}; expected {list(CONFIGS)}")
    return DexThyroid(ModelConfig(config=config, **kwargs))
