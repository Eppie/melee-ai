from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING, Union

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor


if TYPE_CHECKING:
    from config import LossConfig


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

    shoulder_idx = target_info.get("shoulder_idx")
    sh_vec = F.cross_entropy(
        shoulder_logits.reshape(B * L, -1),
        shoulder_idx.reshape(B * L),
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape(B, L)
    loss_shoulder = _mean_with_weights(sh_vec, w_shoulder, loss_config)

    # --- FUTURE POSITION ---
    # Only compute if future position targets are present
    loss_future_x = torch.tensor(0.0, device=logits_main.device, dtype=logits_main.dtype)
    loss_future_y = torch.tensor(0.0, device=logits_main.device, dtype=logits_main.dtype)

    if "future_x_idx" in target_info and "future_y_idx" in target_info:
        future_x_logits = pred.get("future_x")  # [B, L, K_future_x]
        future_y_logits = pred.get("future_y")  # [B, L, K_future_y]

        if future_x_logits is not None and future_y_logits is not None:
            future_x_targets = target_info["future_x_idx"].reshape(B * L)
            future_y_targets = target_info["future_y_idx"].reshape(B * L)
            future_valid = target_info["future_valid"].reshape(B * L)  # [B*L]

            # Cross-entropy loss with validity masking
            loss_future_x_vec = F.cross_entropy(
                future_x_logits.reshape(B * L, -1),
                future_x_targets,
                reduction="none",
                label_smoothing=label_smoothing,
            )  # [B*L]
            loss_future_y_vec = F.cross_entropy(
                future_y_logits.reshape(B * L, -1),
                future_y_targets,
                reduction="none",
                label_smoothing=label_smoothing,
            )  # [B*L]

            # Mask out invalid positions (near episode end)
            loss_future_x_vec = loss_future_x_vec * future_valid
            loss_future_y_vec = loss_future_y_vec * future_valid

            # Mean over valid samples only
            num_valid = future_valid.sum().clamp_min(1.0)
            loss_future_x = (loss_future_x_vec.sum() / num_valid) * loss_config.future_x_weight
            loss_future_y = (loss_future_y_vec.sum() / num_valid) * loss_config.future_y_weight

    total_loss = loss_main + loss_c + loss_buttons + loss_shoulder + loss_future_x + loss_future_y
    return {
        "total": total_loss,
        "main": loss_main,
        "c": loss_c,
        "buttons": loss_buttons,
        "shoulder": loss_shoulder,
        "future_x": loss_future_x,
        "future_y": loss_future_y,
    }


class PolicyLossComputer:
    """Computes weighted imitation learning loss"""

    def __init__(self, config):
        self.config = config
        # We'll reuse the existing compute_loss_components function
        self.label_smoothing = config.train.label_smoothing
        self.loss_config = config.loss_weights

    def compute_loss(
        self,
        outputs: TensorDict,
        targets: Dict,
        weights: torch.Tensor,
    ) -> tuple:
        """
        Args:
            outputs: model outputs with keys (buttons, main_stick, c_stick, shoulder)
            targets: target_info dict from quantize_controller_targets
            weights: (B, L) sample weights from imitation strategy

        Returns:
            loss: scalar loss
            metrics: dict of per-component losses
        """

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
            loss_config=self.loss_config,
        )

        # Batch .item() calls into single GPU->CPU transfer
        component_values = (
            torch.stack(
                [
                    loss_components["main"],
                    loss_components["c"],
                    loss_components["buttons"],
                    loss_components["shoulder"],
                ]
            )
            .cpu()
            .tolist()
        )

        return loss_components["total"], {
            "main": component_values[0],
            "c": component_values[1],
            "buttons": component_values[2],
            "shoulder": component_values[3],
        }
