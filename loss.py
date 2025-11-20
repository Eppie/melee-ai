from __future__ import annotations

from typing import Any, Dict, Mapping, TYPE_CHECKING

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor


if TYPE_CHECKING:
    from config import LossConfig


def sigmoid_focal_loss(
    logits: Tensor, targets: Tensor, gamma: float, alpha: float
) -> Tensor:
    """Multi-label focal loss for sigmoid outputs."""
    if logits.shape != targets.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} must match targets {tuple(targets.shape)}"
        )
    prob = torch.sigmoid(logits)
    pt = torch.where(targets.bool(), prob, 1.0 - prob).clamp_min(1e-8)
    alpha_t = torch.where(targets.bool(), alpha, 1.0 - alpha)
    focal_factor = (1.0 - pt).pow(gamma)
    return -alpha_t * focal_factor * pt.log()


def softmax_focal_loss(
    logits: Tensor, targets: Tensor, gamma: float, alpha: float, label_smoothing: float
) -> Tensor:
    """Multi-class focal loss with optional label smoothing."""
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} incompatible with targets {tuple(targets.shape)}"
        )
    B, L, K = logits.shape
    log_probs = F.log_softmax(logits, dim=-1)
    log_pt = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    pt = log_pt.exp()
    focal_factor = (1.0 - pt).pow(gamma)
    ce = F.cross_entropy(
        logits.reshape(B * L, K),
        targets.reshape(B * L),
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape(B, L)
    return alpha * focal_factor * ce


def compute_loss_components(
    pred: Mapping[str, Tensor],
    target_info: Mapping[str, Any],
    *,
    label_smoothing: float,
    sample_weights: Tensor,
    loss_config: "LossConfig",
) -> Dict[str, Tensor]:
    """
    Compute focal losses for each controller component using shared sample weights.
    """
    logits_main = pred["main_stick"]  # [B, L, K_main]
    logits_c = pred["c_stick"]  # [B, L, K_c]
    logits_btn = pred["buttons"]  # [B, L, K_btn]
    shoulder_logits = pred.get("shoulder")  # [B, L, K_sh]

    if shoulder_logits is None:
        raise ValueError("Shoulder logits missing from model predictions.")

    B, L, _ = logits_main.shape
    if sample_weights.shape[:2] != (B, L):
        raise ValueError(
            f"sample_weights must have leading shape {(B, L)}, got {tuple(sample_weights.shape)}"
        )
    if not torch.isfinite(sample_weights).all():
        raise ValueError("sample_weights contains non-finite values.")
    weights = sample_weights.to(logits_main.dtype)

    main_targets = target_info["main_idx"].reshape(B, L)
    loss_main_vec = softmax_focal_loss(
        logits_main,
        main_targets,
        gamma=loss_config.focal_gamma,
        alpha=loss_config.focal_alpha,
        label_smoothing=label_smoothing,
    )
    loss_main = (loss_main_vec * weights).mean()

    c_targets = target_info["c_idx"].reshape(B, L)
    loss_c_vec = softmax_focal_loss(
        logits_c,
        c_targets,
        gamma=loss_config.focal_gamma,
        alpha=loss_config.focal_alpha,
        label_smoothing=label_smoothing,
    )
    loss_c = (loss_c_vec * weights).mean()

    target_btn = target_info["buttons"]
    if target_btn.shape[:2] != (B, L):
        raise ValueError(
            f"buttons target leading shape {tuple(target_btn.shape[:2])} does not match {(B, L)}"
        )
    loss_btn_all = sigmoid_focal_loss(
        logits_btn,
        target_btn,
        gamma=loss_config.focal_gamma,
        alpha=loss_config.focal_alpha,
    )  # [B, L, K_btn]
    loss_buttons = (loss_btn_all * weights.unsqueeze(-1)).mean()

    shoulder_idx = target_info.get("shoulder_idx")
    if shoulder_idx is None:
        raise ValueError("Shoulder targets missing from target_info.")
    sh_vec = softmax_focal_loss(
        shoulder_logits,
        shoulder_idx.reshape(B, L),
        gamma=loss_config.focal_gamma,
        alpha=loss_config.focal_alpha,
        label_smoothing=label_smoothing,
    )
    loss_shoulder = (sh_vec * weights).mean()

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
            targets: target_info dict from quantization
            weights: (B, L) sample weights from imitation strategy

        Returns:
            loss: scalar loss
            metrics: dict of per-component losses
        """

        loss_components = compute_loss_components(
            outputs,
            targets,
            label_smoothing=self.label_smoothing,
            sample_weights=weights,
            loss_config=self.loss_config,
        )

        return loss_components["total"], {
            "main": loss_components["main"].item(),
            "c": loss_components["c"].item(),
            "buttons": loss_components["buttons"].item(),
            "shoulder": loss_components["shoulder"].item(),
        }
