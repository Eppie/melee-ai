"""Core forward/backward training step helpers."""

from __future__ import annotations

from typing import Dict

import torch
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_

from constants import CONTROLLER_KEY_GROUPS
from loss import compute_loss_components
from train.batch_utils import (
    build_model_inputs,
    compute_component_sample_weights,
    quantize_controller_targets,
)
from train.components import ForwardPassResult, TrainingComponents
from train.gradients import collect_gradient_diagnostics
from train.value_head import compute_value_targets


def perform_forward_pass(
    components: TrainingComponents,
    batch_tensors: Dict[str, torch.Tensor],
    *,
    progress: float,
    in_warmup: bool,
) -> ForwardPassResult:
    """Runs the model forward pass and computes losses/targets."""
    X = batch_tensors["X"]
    Y = batch_tensors["Y"]
    config = components.config
    amp = components.amp

    with autocast(
        device_type=amp.device_type,
        dtype=amp.dtype,
        enabled=amp.enabled,
    ):
        inputs_td = build_model_inputs(X, components.column_map)
        target_info = quantize_controller_targets(
            Y, components.column_map, input_domain="unit01"
        )
        pred = components.model(inputs_td)

        base_smoothing = config.train.label_smoothing
        final_smoothing = 0.5 * base_smoothing
        if in_warmup:
            label_smoothing = base_smoothing
        else:
            label_smoothing = (
                base_smoothing + (final_smoothing - base_smoothing) * progress
            )
        label_smoothing = float(max(label_smoothing, 0.0))

        final_change_scale = 0.5
        if in_warmup:
            imbalance_scale = 1.0
        else:
            imbalance_scale = 1.0 - (1.0 - final_change_scale) * progress
        imbalance_scale = float(max(min(imbalance_scale, 1.0), final_change_scale))

        weights = compute_component_sample_weights(
            target_info,
            components.device,
            ratios=components.ratios,
            button_names=CONTROLLER_KEY_GROUPS["buttons"],
            change_scale=imbalance_scale,
        )

        policy_loss_components = compute_loss_components(
            pred,
            target_info,
            label_smoothing=label_smoothing,
            sample_weights=weights,
            loss_config=config.loss_weights,
            ce_weight_scale=imbalance_scale,
            pos_weight_scale=imbalance_scale,
        )
        loss = policy_loss_components["total"]
        loss_components = dict(policy_loss_components)

        value_pred = pred["value"]
        value_target = compute_value_targets(
            X,
            components.column_map,
            gamma=config.rl.gamma,
            reward_idx=components.value_idx,
        )
        value_loss_raw = torch.nn.functional.mse_loss(
            value_pred, value_target, reduction="none"
        ).squeeze(-1)
        value_w = weights.get("global", weights["main"])
        loss_value = (value_loss_raw * value_w).sum() / value_w.sum().clamp_min(1e-12)
        loss = loss + config.rl.value_loss_coef * loss_value
        loss_components["value"] = loss_value

    batch_targets = {
        "main": target_info["main_idx"],
        "c": target_info["c_idx"],
        "buttons": target_info["buttons"],
        "shoulder_idx": target_info.get("shoulder_idx"),
    }
    batch_inputs = {"X": X}

    return ForwardPassResult(
        pred=pred,
        target_info=target_info,
        weights=weights,
        loss=loss,
        loss_components=loss_components,
        value_pred=value_pred,
        value_target=value_target,
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        label_smoothing=label_smoothing,
        change_scale=imbalance_scale,
    )

# TODO: We are probably failing to call clip_grad_norm if collect_grad_stats is False
def perform_backward_pass(
    components: TrainingComponents,
    loss: torch.Tensor,
    *,
    collect_grad_stats: bool,
) -> Dict[str, float]:
    """Backpropagates the loss, steps the optimizer, and optionally collects grad stats."""
    optimizer = components.optimizer
    scaler = components.scaler

    loss_detached = loss.detach()
    if not torch.isfinite(loss_detached).all():
        loss_value = float(loss_detached.float().cpu().item())
        print(f"Warning: Non-finite loss ({loss_value}); skipping backward step")
        return {}

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()

    if scaler.is_enabled():
        scaler.unscale_(optimizer)

    grad_clip = components.config.train.grad_clip
    pre_clip_norm = float(clip_grad_norm_(components.model.parameters(), grad_clip))

    grad_stats: Dict[str, float] = {}
    if collect_grad_stats:
        grad_stats = collect_gradient_diagnostics(components.model)
        grad_stats["total_norm_pre_clip"] = pre_clip_norm
        grad_stats["total_norm_post_clip"] = min(pre_clip_norm, grad_clip)

    scaler.step(optimizer)
    scaler.update()
    return grad_stats
