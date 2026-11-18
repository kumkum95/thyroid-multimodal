"""Training and evaluation for M3-M6.

Protocol commitments enforced here:
  * The unit is the patient. Bags in, one score out, metrics over patients.
  * The operating threshold comes from the validation split and is applied
    unchanged to test. Picking it on test is the most common way these papers
    overstate sensitivity.
  * Temperature scaling is fitted on validation only, after training.
  * Checkpoints are written every epoch. A Kaggle session is capped at 12 h, so
    a run that cannot resume is a run that can be lost whole.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import metrics as M
from . import schema as S
from .dataset import DataConfig, PatientBags, collate_bags, fit_specs
from .model import CONFIGS, build_model


@dataclass
class TrainConfig:
    config: str = "M6"
    backbone: str = "efficientnet_b4"
    image_size: int = 384
    max_images: int = 6
    batch_size: int = 8
    epochs: int = 30
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 7
    num_workers: int = 2
    amp: bool = True
    seed: int = 0
    target_sensitivity: float = 0.95
    allow_missing_images: bool = False


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(df, specs, cfg: TrainConfig, train: bool) -> DataLoader:
    spec_exo, spec_endo = specs
    model_cfg = CONFIGS[cfg.config]
    data_cfg = DataConfig(
        image_size=cfg.image_size,
        max_images=cfg.max_images,
        augment=train,
        allow_missing_images=cfg.allow_missing_images,
    )
    ds = PatientBags(
        df,
        spec_exo if "exo" in model_cfg else None,
        spec_endo if "endo" in model_cfg else None,
        data_cfg,
        train=train,
        seed=cfg.seed,
    )
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=cfg.num_workers,
        collate_fn=collate_bags,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def forward_batch(model, batch, device):
    kwargs = {}
    if model.cfg.uses_image:
        kwargs["images"] = batch["images"].to(device, non_blocking=True)
        kwargs["mask"] = batch["mask"].to(device, non_blocking=True)
    if model.cfg.uses_exo:
        kwargs["exo"] = batch["exo"].to(device, non_blocking=True)
    if model.cfg.uses_endo:
        kwargs["endo"] = batch["endo"].to(device, non_blocking=True)
    return model(**kwargs)


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    """Patient-level logits, labels, ids and mean modality gates."""
    model.eval()
    logits, labels, ids, gates = [], [], [], []
    for batch in loader:
        out = forward_batch(model, batch, device)
        logits.append(out["logit"].float().cpu().numpy())
        labels.append(batch["label"].numpy())
        ids.extend(batch["patient_id"])
        if "gates" in out:
            gates.append(out["gates"].float().cpu().numpy())
    return (
        np.concatenate(logits),
        np.concatenate(labels),
        ids,
        np.concatenate(gates) if gates else None,
    )


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling fitted on validation logits only.

    Calibration is reported per cohort because a model that holds calibration
    across a prevalence shift is the interesting result, and one that does not
    is still a result.
    """
    t = torch.ones(1, requires_grad=True)
    z = torch.from_numpy(logits).float()
    y = torch.from_numpy(labels).float()
    opt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(z / t.clamp(min=1e-2), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.detach().clamp(min=1e-2).item())


def save_checkpoint(path: Path, model, optimiser, scaler, epoch: int,
                    best: float, cfg: TrainConfig, history: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "best_auc": best,
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "config": asdict(cfg),
            "history": history,
        },
        tmp,
    )
    # Atomic replace: a session killed mid-write must not leave a checkpoint
    # that loads as valid but is truncated.
    tmp.replace(path)


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: TrainConfig,
                out_dir: Path, resume: bool = True) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{cfg.config}_seed{cfg.seed}.pt"

    specs = fit_specs(train_df)
    train_loader = make_loader(train_df, specs, cfg, train=True)
    val_loader = make_loader(val_df, specs, cfg, train=False)

    model = build_model(cfg.config, backbone=cfg.backbone).to(device)

    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        (backbone_params if name.startswith("backbone") else head_params).append(param)
    optimiser = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": cfg.lr_backbone},
            {"params": head_params, "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda") if (cfg.amp and device.type == "cuda") else None
    loss_fn = nn.BCEWithLogitsLoss()

    start_epoch, best_auc, history, stale = 0, -1.0, [], 0
    if resume and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_auc = state["best_auc"]
        history = state.get("history", [])
        print(f"  resumed from epoch {start_epoch} (best val AUC {best_auc:.4f})")

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        started, total, seen = time.time(), 0.0, 0
        for batch in train_loader:
            optimiser.zero_grad(set_to_none=True)
            y = batch["label"].to(device)
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = loss_fn(forward_batch(model, batch, device)["logit"], y)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                loss = loss_fn(forward_batch(model, batch, device)["logit"], y)
                loss.backward()
                optimiser.step()
            total += loss.detach().item() * len(y)
            seen += len(y)
        scheduler.step()

        logits, labels, _, _ = predict(model, val_loader, device)
        from sklearn.metrics import roc_auc_score

        if len(set(labels.tolist())) < 2:
            # One class in validation: AUC is undefined, so every
            # comparison below is False and nothing is ever selected.
            raise RuntimeError(
                'validation split has a single class, so val AUC is '
                'undefined and no model can be selected'
            )
        val_auc = float(roc_auc_score(labels, logits))
        history.append(
            {"epoch": epoch, "train_loss": total / max(seen, 1), "val_auc": val_auc}
        )
        print(
            f"  epoch {epoch:3d}  loss {total / max(seen, 1):.4f}  "
            f"val AUC {val_auc:.4f}  ({time.time() - started:.0f}s)"
        )

        improved = val_auc > best_auc
        if improved:
            best_auc, stale = val_auc, 0
            torch.save(model.state_dict(), out_dir / f"{cfg.config}_seed{cfg.seed}_best.pt")
        else:
            stale += 1
        save_checkpoint(ckpt_path, model, optimiser, scaler, epoch, best_auc, cfg, history)
        if stale >= cfg.patience:
            print(f"  early stop at epoch {epoch} (no gain for {stale} epochs)")
            break

    # A run whose val AUC never improved on the initial -1.0 sentinel --
    # a nan AUC, or a single epoch that crashed -- writes no best file, and
    # loading it unconditionally turns that into a FileNotFoundError that
    # hides the real cause.
    best_path = out_dir / f'{cfg.config}_seed{cfg.seed}_best.pt'
    if not best_path.exists():
        raise RuntimeError(
            f'no best checkpoint at {best_path}: validation AUC never '
            f'improved over {cfg.epochs} epoch(s), so no weights were '
            'selected. Check the validation split and the loss.'
        )
    model.load_state_dict(torch.load(best_path, map_location=device))
    val_logits, val_labels, _, _ = predict(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    val_prob = 1 / (1 + np.exp(-val_logits / temperature))
    threshold = M.threshold_at_sensitivity(
        val_labels, val_prob, target=cfg.target_sensitivity
    )

    return {
        "model": model,
        "specs": specs,
        "temperature": temperature,
        "threshold": threshold,
        "best_val_auc": best_auc,
        "history": history,
        "device": device,
    }


def evaluate_cohort(fitted: dict, df: pd.DataFrame, cfg: TrainConfig,
                    name: str) -> dict:
    """Score one cohort with the validation-derived temperature and threshold."""
    loader = make_loader(df, fitted["specs"], cfg, train=False)
    logits, labels, ids, gates = predict(fitted["model"], loader, fitted["device"])
    prob = 1 / (1 + np.exp(-logits / fitted["temperature"]))
    report = M.evaluate(labels, prob, threshold=fitted["threshold"])

    out = {"cohort": name, **report.as_dict()}
    if gates is not None:
        out["mean_gates"] = {
            n: round(float(v), 4)
            for n, v in zip(fitted["model"].modality_names(), gates.mean(axis=0))
        }
    out["_scores"] = {"patient_id": ids, "prob": prob.tolist(),
                      "label": labels.tolist()}
    return out


def subgroups(df: pd.DataFrame, scores: dict, threshold: float) -> list[dict]:
    """Per-sex and per-age-band performance on an already-scored cohort."""
    from .baselines import to_patient_level

    pts = to_patient_level(df).set_index(S.PATIENT_ID)
    order = scores["patient_id"]
    prob = np.asarray(scores["prob"])
    label = np.asarray(scores["label"])
    sex = pd.to_numeric(pts.loc[order, S.SEX], errors="coerce").to_numpy()
    age = pd.to_numeric(pts.loc[order, S.AGE], errors="coerce").to_numpy()

    rows = []
    groups = [("sex=male", sex == 0), ("sex=female", sex == 1),
              ("age<40", age < 40), ("age40-55", (age >= 40) & (age < 55)),
              ("age>=55", age >= 55)]
    for gname, mask in groups:
        if mask.sum() < 20 or len(np.unique(label[mask])) < 2:
            rows.append({"subgroup": gname, "n": int(mask.sum()),
                         "auc": None, "note": "underpowered"})
            continue
        rep = M.evaluate(label[mask], prob[mask], threshold=threshold, n_boot=1000)
        rows.append({"subgroup": gname, **rep.as_dict()})
    return rows
