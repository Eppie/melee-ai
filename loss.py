from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING, Union

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor


if TYPE_CHECKING:
    from config import LossConfig


def _mean_with_weights(x: Tensor, w: Tensor, loss_config: "LossConfig") -> Tensor:
    if w is None or not loss_config.use_weighted_component_means:
        return x.mean()
    # Match dims to broadcast, then true weighted mean:
    w = w.to(x.dtype)
    num = (x * w).sum()
    den = w.sum().clamp_min(1e-12)
    return num / den


def focal_loss_ce(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
    label_smoothing: float = 0.0,
) -> Tensor:
    """
    Focal loss for multiclass classification (cross-entropy variant).

    Focal loss down-weights easy examples and focuses on hard examples by applying
    a modulating factor (1 - p_t)^gamma to the cross-entropy loss.

    Args:
        logits: [N, K] unnormalized logits
        targets: [N] class indices
        gamma: Focusing parameter. Higher values increase focus on hard examples.
               gamma=0 reduces to standard cross-entropy.
        alpha: Optional class balancing weight for the correct class.
        label_smoothing: Label smoothing factor (applied before focal weighting)

    Returns:
        focal_loss: [N] per-sample focal loss
    """
    # Get probabilities for computing focal weight
    probs = F.softmax(logits, dim=-1)

    # Get probability of correct class [N]
    p_t = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # Compute focal weight: (1 - p_t)^gamma
    # This down-weights easy examples (high p_t) and up-weights hard examples (low p_t)
    focal_weight = (1 - p_t) ** gamma

    # Standard cross-entropy loss (per-sample)
    ce_loss = F.cross_entropy(
        logits, targets, reduction="none", label_smoothing=label_smoothing
    )

    # Apply focal weighting
    focal_loss = focal_weight * ce_loss

    # Optional: Apply class balancing factor
    if alpha is not None:
        focal_loss = alpha * focal_loss

    return focal_loss


def focal_loss_bce(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    alpha: float | None = None,
) -> Tensor:
    """
    Focal loss for multi-label binary classification.

    Applies focal loss independently to each binary label.

    Args:
        logits: [B, L, K] unnormalized logits for K binary labels
        targets: [B, L, K] binary targets (0 or 1)
        gamma: Focusing parameter
        alpha: Optional balancing weight for positive class

    Returns:
        focal_loss: [B, L, K] per-sample, per-label focal loss
    """
    # Get probabilities
    probs = torch.sigmoid(logits)

    # Standard BCE loss (per-sample, per-label)
    bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # Compute p_t: probability of the correct class
    # If target=1, p_t=p, if target=0, p_t=1-p
    p_t = torch.where(targets > 0.5, probs, 1 - probs)

    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma

    # Apply focal weighting
    focal_loss = focal_weight * bce_loss

    # Optional: Apply class balancing
    if alpha is not None:
        # Alpha typically applies to positive class (target=1)
        alpha_t = torch.where(targets > 0.5, alpha, 1 - alpha)
        focal_loss = alpha_t * focal_loss

    return focal_loss


def compute_loss_components(
    pred: Mapping[str, Tensor],
    target_info: Mapping[str, Any],
    *,
    label_smoothing: float,
    sample_weights: Optional[Union[Tensor, Mapping[str, Tensor]]] = None,
    loss_config: "LossConfig",
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

    if loss_config.use_focal_loss:
        loss_main_vec = focal_loss_ce(
            main_logits,
            main_targets,
            gamma=loss_config.focal_gamma,
            alpha=loss_config.focal_alpha,
            label_smoothing=label_smoothing,
        ).reshape(B, L)
    else:
        loss_main_vec = F.cross_entropy(
            main_logits,
            main_targets,
            reduction="none",
            label_smoothing=label_smoothing,
        ).reshape(B, L)

    loss_main = _mean_with_weights(loss_main_vec, w_main, loss_config)

    # --- C-STICK ---
    c_targets = target_info["c_idx"].reshape(B * L)
    c_logits = logits_c.reshape(B * L, -1)

    if loss_config.use_focal_loss:
        loss_c_vec = focal_loss_ce(
            c_logits,
            c_targets,
            gamma=loss_config.focal_gamma,
            alpha=loss_config.focal_alpha,
            label_smoothing=label_smoothing,
        ).reshape(B, L)
    else:
        loss_c_vec = F.cross_entropy(
            c_logits,
            c_targets,
            reduction="none",
            label_smoothing=label_smoothing,
        ).reshape(B, L)

    loss_c = _mean_with_weights(loss_c_vec, w_c, loss_config)

    # --- BUTTONS (per-label weighting)
    target_btn = target_info["buttons"]

    if loss_config.use_focal_loss:
        loss_btn_all = focal_loss_bce(
            logits_btn,
            target_btn,
            gamma=loss_config.focal_gamma,
            alpha=loss_config.focal_alpha,
        )  # [B, L, K_btn]
    else:
        loss_btn_all = F.binary_cross_entropy_with_logits(
            logits_btn, target_btn, reduction="none"
        )  # [B, L, K_btn]

    if w_buttons is not None:
        # true weighted mean over all dims
        loss_buttons = _mean_with_weights(loss_btn_all, w_buttons, loss_config)
    else:
        # original behavior (mean over label dim, then batch/time)
        loss_buttons = loss_btn_all.mean()

    shoulder_idx = target_info.get("shoulder_idx")

    if loss_config.use_focal_loss:
        sh_vec = focal_loss_ce(
            shoulder_logits.reshape(B * L, -1),
            shoulder_idx.reshape(B * L),
            gamma=loss_config.focal_gamma,
            alpha=loss_config.focal_alpha,
            label_smoothing=label_smoothing,
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
