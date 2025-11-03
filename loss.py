from __future__ import annotations

from typing import Any, Mapping, Union
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor


# TODO: Make using this configurable via config.py
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


# TODO: Make using this configurable via config.py
def _compute_pos_weights(targets: Tensor) -> Tensor:
    """Compute positive class weights for multi-label BCE loss."""
    flat = targets.reshape(-1, targets.shape[-1])
    pos = flat.sum(dim=0)
    total = flat.shape[0]
    neg = total - pos
    pos_weight = neg / pos.clamp_min(1.0)
    return pos_weight.clamp(min=1.0, max=_POS_WEIGHT_CLAMP).to(targets.device)


# TODO: Make using these configurable via config.py
_CE_WEIGHT_CLAMP = 10.0
_CE_WEIGHT_MIN = 0.1
_POS_WEIGHT_CLAMP = 10.0


# TODO: Make using this configurable via config.py
def _mean_with_weights(x: Tensor, w: Optional[Tensor]) -> Tensor:
    if w is None:
        return x.mean()
    # Match dims to broadcast, then true weighted mean:
    w = w.to(x.dtype)
    num = (x * w).sum()
    den = w.sum().clamp_min(1e-12)
    return num / den


def compute_loss_components(
    pred: Mapping[str, Tensor],
    target_info: Mapping[str, Any],
    *,
    label_smoothing: float,
    sample_weights: Optional[Union[Tensor, Mapping[str, Tensor]]] = None,
) -> Dict[str, Tensor]:
    """
    Accepts either:
      - sample_weights: [B, L] (legacy)
      - sample_weights: {"main":[B,L], "c":[B,L], "shoulder":[B,L], "buttons":[B,L,K]}
    """
    logits_main = pred["main_stick"]  # [B, L, K_main]
    logits_c = pred["c_stick"]  # [B, L, K_c]
    logits_btn = pred["buttons"]  # [B, L, K_btn]
    shoulder_logits = pred.get("shoulder")  # [B, L, K_sh]? optional

    B, L, _ = logits_main.shape

    # Pull per-component weights (or None)
    def _get_w(name: str, expect_ndim: int) -> Optional[Tensor]:
        if sample_weights is None:
            return None
        if isinstance(sample_weights, Tensor):
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
    main_weights = _compute_ce_weights(main_targets, int(target_info["main_K"]))
    loss_main_vec = F.cross_entropy(
        main_logits,
        main_targets,
        reduction="none",
        label_smoothing=label_smoothing,
        weight=main_weights,
    ).reshape(B, L)
    loss_main = _mean_with_weights(loss_main_vec, w_main)

    # --- C-STICK ---
    c_targets = target_info["c_idx"].reshape(B * L)
    c_logits = logits_c.reshape(B * L, -1)
    c_weights = _compute_ce_weights(c_targets, int(target_info["c_K"]))
    loss_c_vec = F.cross_entropy(
        c_logits,
        c_targets,
        reduction="none",
        label_smoothing=label_smoothing,
        weight=c_weights,
    ).reshape(B, L)
    loss_c = _mean_with_weights(loss_c_vec, w_c)

    # --- BUTTONS (per-label weighting)
    target_btn = target_info["buttons"]
    pos_weight = _compute_pos_weights(target_btn)  # [K_btn]
    loss_btn_all = F.binary_cross_entropy_with_logits(
        logits_btn, target_btn, reduction="none", pos_weight=pos_weight
    )  # [B, L, K_btn]

    if w_buttons is not None:
        # true weighted mean over all dims
        loss_buttons = _mean_with_weights(loss_btn_all, w_buttons)
    else:
        # original behavior (mean over label dim, then batch/time)
        loss_buttons = loss_btn_all.mean()

    shoulder_idx = target_info.get("shoulder_idx")
    sh_vec = F.cross_entropy(
        shoulder_logits.reshape(B * L, -1),
        shoulder_idx.reshape(B * L),
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape(B, L)
    loss_shoulder = _mean_with_weights(sh_vec, w_shoulder)

    total_loss = loss_main + loss_c + loss_buttons + loss_shoulder
    return {
        "total": total_loss,
        "main": loss_main,
        "c": loss_c,
        "buttons": loss_buttons,
        "shoulder": loss_shoulder,
    }


class PolicyLossComputer:
    """Computes weighted imitation learning loss"""

    def __init__(self, config):
        self.config = config
        # We'll reuse the existing compute_loss_components function
        self.label_smoothing = config.train.label_smoothing

    def compute_loss(
        self,
        outputs: TensorDict,
        targets: Dict,
        weights: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            outputs: model outputs with keys (buttons, main_stick, c_stick, shoulder)
            targets: target_info dict from quantize_controller_targets
            weights: (B, L) sample weights from imitation strategy
            mask: (B, L) optional mask for valid timesteps

        Returns:
            loss: scalar loss
            metrics: dict of per-component losses
        """
        from loss import compute_loss_components

        # Convert weights to dict format expected by compute_loss_components
        # The existing function expects component-specific weights
        # We'll use the same weight for all components
        B, L = weights.shape
        K_btn = outputs["buttons"].shape[-1]

        sample_weights = {
            "main": weights,  # [B, L]
            "c": weights,  # [B, L]
            "shoulder": weights,  # [B, L]
            "buttons": weights.unsqueeze(-1).expand(B, L, K_btn),  # [B, L, K_btn]
        }

        loss_components = compute_loss_components(
            outputs,
            targets,
            label_smoothing=self.label_smoothing,
            sample_weights=sample_weights,
        )

        return loss_components["total"], {
            "main": loss_components["main"].item(),
            "c": loss_components["c"].item(),
            "buttons": loss_components["buttons"].item(),
            "shoulder": loss_components["shoulder"].item(),
        }

