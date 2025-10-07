from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

_CE_WEIGHT_CLAMP = 10.0
_CE_WEIGHT_MIN = 0.1
_POS_WEIGHT_CLAMP = 10.0


def _compute_ce_weights(labels: Tensor, num_classes: int) -> Tensor:
    """Compute class-balanced weights for cross-entropy loss."""
    device = labels.device
    try:
        counts = torch.bincount(labels, minlength=num_classes)
    except RuntimeError:
        counts = torch.bincount(labels.cpu(), minlength=num_classes).to(device)
    counts = counts.float().clamp_min(1.0)
    weights = counts.sum() / (counts * num_classes)
    return weights.clamp(min=_CE_WEIGHT_MIN, max=_CE_WEIGHT_CLAMP)


def _compute_pos_weights(targets: Tensor) -> Tensor:
    """Compute positive class weights for multi-label BCE loss."""
    flat = targets.reshape(-1, targets.shape[-1])
    pos = flat.sum(dim=0)
    total = flat.shape[0]
    neg = total - pos
    pos_weight = neg / pos.clamp_min(1.0)
    return pos_weight.clamp(min=1.0, max=_POS_WEIGHT_CLAMP).to(targets.device)


def compute_loss_components(
    pred: Mapping[str, Tensor],
    target_info: Mapping[str, Any],
    *,
    label_smoothing: float,
    use_moe: bool,
    moe_aux_loss_weight: float,
) -> Dict[str, Tensor]:
    """Compute total loss and its components for the controller model.

    Returns a dictionary containing individual components and the summed total, all
    as tensors kept on the prediction device so they participate in autograd.
    """
    logits_main = pred["main_stick"]
    if logits_main.ndim != 3:
        raise ValueError("'main_stick' logits must have shape [B, L, K].")
    B, L, _ = logits_main.shape

    main_targets = target_info["main_idx"].reshape(B * L)
    main_logits = logits_main.reshape(B * L, -1)
    main_weights = _compute_ce_weights(main_targets, int(target_info["main_K"]))
    loss_main = F.cross_entropy(
        main_logits,
        main_targets,
        reduction="mean",
        label_smoothing=label_smoothing,
        weight=main_weights,
    )

    logits_c = pred["c_stick"]
    c_targets = target_info["c_idx"].reshape(B * L)
    c_logits = logits_c.reshape(B * L, -1)
    c_weights = _compute_ce_weights(c_targets, int(target_info["c_K"]))
    loss_c = F.cross_entropy(
        c_logits,
        c_targets,
        reduction="mean",
        label_smoothing=label_smoothing,
        weight=c_weights,
    )

    logits_btn = pred["buttons"]
    target_btn = target_info["buttons"]
    pos_weight = _compute_pos_weights(target_btn)
    loss_buttons = F.binary_cross_entropy_with_logits(
        logits_btn,
        target_btn,
        reduction="mean",
        pos_weight=pos_weight,
    )

    loss_shoulder = torch.zeros((), device=logits_main.device)
    shoulder_logits = pred.get("shoulder")
    shoulder_idx = target_info.get("shoulder_idx")
    shoulder_K = int(target_info.get("shoulder_K", 0))
    if (
        shoulder_logits is not None
        and shoulder_idx is not None
        and shoulder_K > 0
    ):
        shoulder_logits_flat = shoulder_logits.reshape(B * L, -1)
        shoulder_targets_flat = shoulder_idx.reshape(B * L)
        loss_shoulder = F.cross_entropy(
            shoulder_logits_flat,
            shoulder_targets_flat,
            reduction="mean",
            label_smoothing=label_smoothing,
        )

    loss_aux = torch.zeros((), device=logits_main.device)
    if use_moe and "moe_aux_loss" in pred:
        loss_aux = pred["moe_aux_loss"].mean() * moe_aux_loss_weight

    total_loss = loss_main + loss_c + loss_buttons + loss_shoulder + loss_aux
    return {
        "total": total_loss,
        "main": loss_main,
        "c": loss_c,
        "buttons": loss_buttons,
        "shoulder": loss_shoulder,
        "moe_aux": loss_aux,
    }


__all__ = [
    "compute_loss_components",
    "_compute_ce_weights",
    "_compute_pos_weights",
]
