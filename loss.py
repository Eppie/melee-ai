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

    # TODO: Shoulder is not optional, we will always have it!
    # --- SHOULDER (optional) ---
    loss_shoulder = torch.zeros((), device=logits_main.device)
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


class AuxLossComputer:
    """Computes auxiliary task losses"""

    def __init__(self, config):
        self.config = config
        self.aux_config = config.aux_tasks

    def compute_loss(
        self,
        aux_outputs: Dict[str, torch.Tensor],
        aux_targets: Dict[str, torch.Tensor],
        mask: torch.Tensor = None,
    ) -> tuple:
        """
        Args:
            aux_outputs: dict with keys (opponent_action, damage_diff, action_effectiveness)
            aux_targets: dict with corresponding target tensors
            mask: (B, L) optional mask for valid timesteps

        Returns:
            loss: scalar loss
            metrics: dict of per-task losses
        """
        # Convert TensorDict to regular dict if needed
        if hasattr(aux_outputs, "keys"):
            aux_outputs_keys = list(aux_outputs.keys())
        else:
            aux_outputs_keys = []

        if hasattr(aux_targets, "keys"):
            aux_targets_keys = list(aux_targets.keys())
        else:
            aux_targets_keys = []

        # Infer B, L from any available output/target
        device = None
        B, L = None, None

        if aux_outputs_keys:
            first_key = aux_outputs_keys[0]
            B, L = aux_outputs[first_key].shape[:2]
            device = aux_outputs[first_key].device
        elif aux_targets_keys:
            first_key = aux_targets_keys[0]
            B, L = aux_targets[first_key].shape[:2]
            device = aux_targets[first_key].device
        else:
            # No outputs or targets, return zero loss
            return torch.tensor(0.0), {}

        if mask is None:
            mask = torch.ones((B, L), device=device)

        losses = {}

        # 1. Opponent action prediction (cross-entropy)
        if (
            "opponent_action" in aux_outputs_keys
            and "opponent_action" in aux_targets_keys
        ):
            logits = aux_outputs["opponent_action"]  # (B, L, num_actions)
            targets = aux_targets["opponent_action"]  # (B, L)

            loss_vec = F.cross_entropy(
                logits.reshape(B * L, -1), targets.reshape(B * L), reduction="none"
            ).reshape(B, L)

            losses["opponent_action"] = (loss_vec * mask).sum() / mask.sum().clamp_min(
                1e-8
            )

        # 2. Damage differential (MSE)
        if "damage_diff" in aux_outputs_keys and "damage_diff" in aux_targets_keys:
            pred = aux_outputs["damage_diff"]  # (B, L, 1)
            target = aux_targets["damage_diff"]  # (B, L, 1)

            loss_vec = F.mse_loss(pred, target, reduction="none").squeeze(-1)  # (B, L)
            losses["damage_diff"] = (loss_vec * mask).sum() / mask.sum().clamp_min(1e-8)

        # 3. Action effectiveness (BCE with pos_weight)
        if (
            "action_effectiveness" in aux_outputs_keys
            and "action_effectiveness" in aux_targets_keys
        ):
            logits = aux_outputs["action_effectiveness"]  # (B, L, 1)
            targets = aux_targets["action_effectiveness"]  # (B, L, 1)

            # Compute positive weight for class imbalance
            pos_rate = targets.mean().clamp(min=0.01, max=0.99)
            pos_weight = (1 - pos_rate) / pos_rate
            pos_weight = pos_weight.clamp(
                max=self.aux_config.effectiveness_pos_weight_max
            )

            loss_vec = F.binary_cross_entropy_with_logits(
                logits, targets, reduction="none", pos_weight=pos_weight
            ).squeeze(
                -1
            )  # (B, L)

            losses["action_effectiveness"] = (
                loss_vec * mask
            ).sum() / mask.sum().clamp_min(1e-8)

        # Weighted sum
        total = torch.zeros((), device=device)
        for task, loss in losses.items():
            weight = getattr(self.aux_config, f"{task}_weight", 1.0)
            total += weight * loss

        metrics = {k: v.item() for k, v in losses.items()}

        return total, metrics


class CompositeLossComputer:
    """Combines policy and auxiliary losses"""

    def __init__(self, config):
        self.config = config
        self.policy_computer = PolicyLossComputer(config)
        self.aux_computer = AuxLossComputer(config)

    def compute_loss(
        self,
        policy_outputs: TensorDict,
        policy_targets: Dict,
        aux_outputs: Dict[str, torch.Tensor],
        aux_targets: Dict[str, torch.Tensor],
        imitation_weights: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> tuple:
        """
        Returns:
            total_loss: scalar
            metrics: dict with all loss components
        """
        # Policy loss
        policy_loss, policy_metrics = self.policy_computer.compute_loss(
            policy_outputs, policy_targets, imitation_weights, mask
        )

        # Auxiliary loss
        aux_loss, aux_metrics = self.aux_computer.compute_loss(
            aux_outputs, aux_targets, mask
        )

        # Combine
        total = policy_loss + self.config.aux_tasks.aux_loss_weight * aux_loss

        metrics = {
            "total": total.item(),
            "policy": policy_loss.item(),
            "aux_total": aux_loss.item(),
            **{f"policy/{k}": v for k, v in policy_metrics.items()},
            **{f"aux/{k}": v for k, v in aux_metrics.items()},
        }

        return total, metrics
