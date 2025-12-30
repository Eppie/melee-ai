from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING, Union

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor


if TYPE_CHECKING:
    from config import LossConfig


def focal_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    weight: Optional[Tensor] = None,
    label_smoothing: float = 0.0,
    reduction: str = "none",
) -> Tensor:
    """Focal loss for multi-class classification.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    Args:
        logits: [N, C] unnormalized logits
        targets: [N] class indices
        gamma: focusing parameter (0 = standard CE)
        weight: [C] per-class weights
        label_smoothing: label smoothing factor
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Focal loss values
    """
    # Compute softmax probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    # Get probability of correct class
    targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()

    # Apply label smoothing if requested
    if label_smoothing > 0:
        targets_one_hot = targets_one_hot * (1 - label_smoothing) + label_smoothing / logits.size(-1)

    # p_t = probability assigned to true class
    p_t = (probs * targets_one_hot).sum(dim=-1)

    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma

    # Standard cross-entropy: -log(p_t)
    ce_loss = -(log_probs * targets_one_hot).sum(dim=-1)

    # Apply focal weighting
    loss = focal_weight * ce_loss

    # Apply class weights if provided
    if weight is not None:
        class_weights = weight[targets]
        loss = loss * class_weights

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def focal_binary_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "none",
) -> Tensor:
    """Focal loss for binary/multi-label classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        logits: [...] unnormalized logits
        targets: [...] binary targets (0 or 1)
        gamma: focusing parameter (0 = standard BCE)
        alpha: weight for positive class (1-alpha for negative)
        reduction: 'none', 'mean', or 'sum'

    Returns:
        Focal loss values
    """
    # Compute probabilities
    probs = torch.sigmoid(logits)

    # p_t = p if y=1, else 1-p
    p_t = probs * targets + (1 - probs) * (1 - targets)

    # alpha_t = alpha if y=1, else 1-alpha
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)

    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma

    # Standard BCE
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # Apply focal weighting
    loss = alpha_t * focal_weight * bce_loss

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    return loss


def _compute_ce_weights(
    labels: Tensor, num_classes: int, loss_config: "LossConfig"
) -> Optional[Tensor]:
    """Compute class-balanced weights for cross-entropy loss."""
    if not loss_config.enable_class_balancing:
        return None
    device = labels.device
    try:
        counts = torch.bincount(labels, minlength=num_classes)
    except RuntimeError:
        counts = torch.bincount(labels.cpu(), minlength=num_classes).to(device)
    counts = counts.float().clamp_min(1.0)
    weights = counts.sum() / (counts * num_classes)
    return weights.clamp(min=loss_config.ce_weight_min, max=loss_config.ce_weight_max)


def _compute_pos_weights(
    targets: Tensor, loss_config: "LossConfig"
) -> Optional[Tensor]:
    """Compute positive class weights for multi-label BCE loss."""
    if not loss_config.enable_pos_weighting:
        return None

    flat = targets.reshape(-1, targets.shape[-1])
    pos = flat.sum(dim=0)
    total = flat.shape[0]
    neg = total - pos
    pos_weight = neg / pos.clamp_min(1.0)
    return pos_weight.clamp(min=1.0, max=loss_config.pos_weight_max).to(targets.device)


def _mean_with_weights(x: Tensor, w: Tensor, loss_config: "LossConfig") -> Tensor:
    if not loss_config.use_weighted_component_means:
        return x.mean()
    # Match dims to broadcast, then true weighted mean:
    w = w.to(x.dtype)
    num = (x * w).sum()
    den = w.sum().clamp_min(1e-12)
    return num / den


def _blend_weights(weights: Optional[Tensor], scale: float) -> Optional[Tensor]:
    assert scale > 0, f"scale {scale} is invalid"
    if weights is None or scale >= 1:
        return weights
    return torch.ones_like(weights) + (weights - 1.0) * scale


def compute_loss_components(
    pred: Mapping[str, Tensor],
    target_info: Mapping[str, Any],
    *,
    label_smoothing: float,
    sample_weights: Optional[Union[Tensor, Mapping[str, Tensor]]] = None,
    loss_config: "LossConfig",
    ce_weight_scale: float = 1.0,
    pos_weight_scale: float = 1.0,
) -> Dict[str, Tensor]:
    """
    Accepts either:
      - sample_weights: [B, L] (legacy)
      - sample_weights: {"main":[B,L], "c":[B,L], "shoulder":[B,L], "buttons":[B,L,K]}
    """
    logits_main = pred["main_stick"]  # [B, L, K_main]
    logits_c = pred["c_stick"]  # [B, L, K_c]
    logits_btn = pred["buttons"]  # [B, L, K_btn]
    shoulder_logits = pred.get("shoulder")  # [B, L, K_sh]

    B, L, _ = logits_main.shape

    # Pull per-component weights (or None)
    def _get_w(name: str, expect_ndim: int) -> Optional[Tensor]:
        if sample_weights is None:
            return None
        if isinstance(sample_weights, Tensor):
            if expect_ndim == 3:
                if sample_weights.ndim == 2:
                    return sample_weights.unsqueeze(-1)
                return sample_weights
            assert expect_ndim in (1, 2)  # only [B,L] valid for non-buttons
            return sample_weights  # [B,L]
        w = sample_weights.get(name)
        if w is None:
            return None
        assert (
            w.ndim == expect_ndim
        ), f"{name} weights must have ndim={expect_ndim}, got {w.shape}"
        return w

    w_main = _get_w("main", 2)  # [B, L]
    w_c = _get_w("c", 2)  # [B, L]
    w_buttons = _get_w("buttons", 3)  # [B, L, K_btn]
    w_shoulder = _get_w("shoulder", 2)  # [B, L]

    # --- MAIN ---
    main_targets = target_info["main_idx"].reshape(B * L)
    main_logits = logits_main.reshape(B * L, -1)
    main_weights = _compute_ce_weights(
        main_targets, int(target_info["main_K"]), loss_config
    )
    main_weights = _blend_weights(main_weights, ce_weight_scale)

    if loss_config.use_focal_loss:
        loss_main_vec = focal_cross_entropy(
            main_logits,
            main_targets,
            gamma=loss_config.focal_gamma,
            weight=main_weights,
            label_smoothing=label_smoothing,
            reduction="none",
        ).reshape(B, L)
    else:
        loss_main_vec = F.cross_entropy(
            main_logits,
            main_targets,
            reduction="none",
            label_smoothing=label_smoothing,
            weight=main_weights,
        ).reshape(B, L)
    loss_main = _mean_with_weights(loss_main_vec, w_main, loss_config)

    # --- C-STICK ---
    c_targets = target_info["c_idx"].reshape(B * L)
    c_logits = logits_c.reshape(B * L, -1)
    c_weights = _compute_ce_weights(c_targets, int(target_info["c_K"]), loss_config)
    c_weights = _blend_weights(c_weights, ce_weight_scale)

    if loss_config.use_focal_loss:
        loss_c_vec = focal_cross_entropy(
            c_logits,
            c_targets,
            gamma=loss_config.focal_gamma,
            weight=c_weights,
            label_smoothing=label_smoothing,
            reduction="none",
        ).reshape(B, L)
    else:
        loss_c_vec = F.cross_entropy(
            c_logits,
            c_targets,
            reduction="none",
            label_smoothing=label_smoothing,
            weight=c_weights,
        ).reshape(B, L)
    loss_c = _mean_with_weights(loss_c_vec, w_c, loss_config)

    # --- BUTTONS (per-label weighting)
    target_btn = target_info["buttons"]

    if loss_config.use_focal_loss:
        loss_btn_all = focal_binary_cross_entropy(
            logits_btn,
            target_btn,
            gamma=loss_config.focal_gamma,
            alpha=loss_config.focal_alpha,
            reduction="none",
        )  # [B, L, K_btn]
    else:
        pos_weight = _compute_pos_weights(target_btn, loss_config)  # [K_btn] or None
        pos_weight = _blend_weights(pos_weight, pos_weight_scale)
        loss_btn_all = F.binary_cross_entropy_with_logits(
            logits_btn, target_btn, reduction="none", pos_weight=pos_weight
        )  # [B, L, K_btn]

    if w_buttons is not None:
        # true weighted mean over all dims
        loss_buttons = _mean_with_weights(loss_btn_all, w_buttons, loss_config)
    else:
        # original behavior (mean over label dim, then batch/time)
        loss_buttons = loss_btn_all.mean()

    # --- SHOULDER ---
    shoulder_idx = target_info.get("shoulder_idx")
    if loss_config.use_focal_loss:
        sh_vec = focal_cross_entropy(
            shoulder_logits.reshape(B * L, -1),
            shoulder_idx.reshape(B * L),
            gamma=loss_config.focal_gamma,
            label_smoothing=label_smoothing,
            reduction="none",
        ).reshape(B, L)
    else:
        sh_vec = F.cross_entropy(
            shoulder_logits.reshape(B * L, -1),
            shoulder_idx.reshape(B * L),
            reduction="none",
            label_smoothing=label_smoothing,
        ).reshape(B, L)
    loss_shoulder = _mean_with_weights(sh_vec, w_shoulder, loss_config)

    total_loss = loss_main + loss_c + loss_buttons + loss_shoulder
    return {
        "total": total_loss,
        "main": loss_main,
        "c": loss_c,
        "buttons": loss_buttons,
        "shoulder": loss_shoulder,
    }
