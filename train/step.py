"""Core forward/backward training step helpers."""

from __future__ import annotations

from typing import Dict

import torch
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_

from constants import CONTROLLER_KEY_GROUPS
from controller_quantization import quantize_targets
from loss import compute_loss_components
from train.batch_utils import (
    build_model_inputs,
    compute_component_sample_weights,
)
from train.components import ForwardPassResult, TrainingComponents
from train.gradients import collect_gradient_diagnostics
from train.imitation_weights import compute_imitation_weights
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
        target_info = quantize_targets(Y, components.column_map, input_domain="unit01")
        pred = components.model(inputs_td)
        # Clone to prevent CUDA graph overwriting when using torch.compile()
        pred = pred.clone()

        base_smoothing = config.train.label_smoothing
        final_smoothing = 0.5 * base_smoothing
        if in_warmup:
            label_smoothing = base_smoothing
        else:
            label_smoothing = (
                base_smoothing + (final_smoothing - base_smoothing) * progress
            )
        label_smoothing = float(max(label_smoothing, 0.0))

        # Imbalance scale scheduling: three-phase approach
        # Phase 1: Keep at initial value for first initial_fraction (learn action space)
        # Phase 2: Linear decay from initial to final over middle portion (transition)
        # Phase 3: Keep at final value for last final_fraction (learn timing)
        initial_scale = config.train.imbalance_scale_initial
        final_scale = config.train.imbalance_scale_final
        initial_fraction = config.train.imbalance_scale_initial_fraction
        final_fraction = config.train.imbalance_scale_final_fraction

        if progress < initial_fraction:
            # Phase 1: Keep at initial value
            imbalance_scale = initial_scale
        elif progress >= (1.0 - final_fraction):
            # Phase 3: Keep at final value
            imbalance_scale = final_scale
        else:
            # Phase 2: Linear ramp from initial to final over middle portion
            ramp_start = initial_fraction
            ramp_end = 1.0 - final_fraction
            ramp_progress = (progress - ramp_start) / (ramp_end - ramp_start)
            imbalance_scale = (
                initial_scale + (final_scale - initial_scale) * ramp_progress
            )

        imbalance_scale = float(
            max(
                min(imbalance_scale, max(initial_scale, final_scale)),
                min(initial_scale, final_scale),
            )
        )

        # Compute change-based weights (focus on action changes)
        change_weights = compute_component_sample_weights(
            target_info,
            components.device,
            ratios=components.ratios,
            button_names=CONTROLLER_KEY_GROUPS["buttons"],
            change_scale=imbalance_scale,
        )

        # Compute value-based weights (focus on high-value states)
        # This weights frames by their value_target to emphasize learning from winning play
        imitation_weights_tensor = compute_imitation_weights(
            X, components.value_idx, config.imitation
        )  # [B, L]

        # Combine change-based and value-based weights
        # Apply value weights to all components
        combined_weights = {}
        for key, change_w in change_weights.items():
            # Multiply change weights by value weights
            combined_weights[key] = change_w * imitation_weights_tensor

        policy_loss_components = compute_loss_components(
            pred,
            target_info,
            label_smoothing=label_smoothing,
            sample_weights=combined_weights,
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
        value_w = combined_weights.get("global", combined_weights["main"])
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

    # Use gradient scaling only if scaler is enabled (float16)
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        # bfloat16 or full precision - no scaling needed
        loss.backward()

    grad_clip = components.config.train.grad_clip
    pre_clip_norm = float(clip_grad_norm_(components.model.parameters(), grad_clip))

    grad_stats: Dict[str, float] = {}
    if collect_grad_stats:
        grad_stats = collect_gradient_diagnostics(components.model)
        grad_stats["total_norm_pre_clip"] = pre_clip_norm
        grad_stats["total_norm_post_clip"] = min(pre_clip_norm, grad_clip)

    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return grad_stats
