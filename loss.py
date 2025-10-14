from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

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
        sample_weights: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    """Compute total loss and its components for the controller model.

    Returns a dictionary containing individual components and the summed total, all
    as tensors kept on the prediction device so they participate in autograd.
    """
    logits_main = pred["main_stick"]
    if logits_main.ndim != 3:
        raise ValueError("'main_stick' logits must have shape [B, L, K].")
    B, L, _ = logits_main.shape
    weights_flat = sample_weights.view(B * L) if sample_weights is not None else None

    main_targets = target_info["main_idx"].reshape(B * L)
    main_logits = logits_main.reshape(B * L, -1)
    main_weights = _compute_ce_weights(main_targets, int(target_info["main_K"]))
    loss_main = F.cross_entropy(
        main_logits,
        main_targets,
        reduction='none',  # Change reduction
        label_smoothing=label_smoothing,
        weight=main_weights,
    )
    if weights_flat is not None:
        loss_main = (loss_main * weights_flat).mean()
    else:
        loss_main = loss_main.mean()

    logits_c = pred["c_stick"]
    c_targets = target_info["c_idx"].reshape(B * L)
    c_logits = logits_c.reshape(B * L, -1)
    c_weights = _compute_ce_weights(c_targets, int(target_info["c_K"]))
    loss_c = F.cross_entropy(
        c_logits,
        c_targets,
        reduction="none",
        label_smoothing=label_smoothing,
        weight=c_weights,
    )
    if weights_flat is not None:
        loss_c = (loss_c * weights_flat).mean()
    else:
        loss_c = loss_c.mean()

    logits_btn = pred["buttons"]
    target_btn = target_info["buttons"]
    pos_weight = _compute_pos_weights(target_btn)
    loss_buttons = F.binary_cross_entropy_with_logits(
        logits_btn,
        target_btn,
        reduction='none',
        pos_weight=pos_weight,
    ).mean(dim=-1)

    if sample_weights is not None:
        loss_buttons = (loss_buttons * sample_weights).mean()
    else:
        loss_buttons = loss_buttons.mean()

    # TODO: It is safe to assume we always have the shoulder, we don't need to be defensive here
    # TODO: Apply the sample weights here as well
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
