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
    """Runs the model forward pass and computes losses/targets.

    Args:
        components: Training components (model, optimizer, etc.)
        batch_tensors: Input batch
        progress: Training progress [0, 1]
        in_warmup: Whether in warmup phase
    """
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
        if (
            components.column_map.y_main_idx is None
            or components.column_map.y_c_idx is None
            or components.column_map.y_shoulder_idx is None
            or not components.column_map.y_buttons
        ):
            raise RuntimeError(
                "Pre-quantized targets required but target indices missing; regenerate dataset with preprocessing."
            )
        head_dims = components.config.model.target_shapes_by_head

        target_info = {
            "main_idx": Y[..., components.column_map.y_main_idx].to(torch.long),
            "c_idx": Y[..., components.column_map.y_c_idx].to(torch.long),
            "shoulder_idx": Y[..., components.column_map.y_shoulder_idx].to(torch.long),
            "buttons": Y[..., components.column_map.y_buttons].to(torch.float32),
            "main_K": int(head_dims["main_stick"]),  # should align with palette size
            "c_K": int(head_dims["c_stick"]),
            "buttons_K": len(components.column_map.y_buttons),
            "shoulder_K": int(head_dims["shoulder"]),
        }
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

        # Get value predictions and targets FIRST (needed for model-dependent advantages)
        value_pred = pred["value"]
        value_target = compute_value_targets(
            X,
            components.column_map,
            reward_cfg=config.reward,
            reward_idx=components.value_idx,
        )

        # Compute change-based weights (focus on action changes)
        change_weights = compute_component_sample_weights(
            target_info,
            components.device,
            ratios=components.ratios,
            button_names=CONTROLLER_KEY_GROUPS["buttons"],
            change_scale=imbalance_scale,
        )

        # Compute value-based weights using MODEL-DEPENDENT advantages
        # Advantage = how much better the actual outcome was vs model prediction
        # This focuses learning on frames where expert did better than model expected
        advantages_tensor = None
        if config.imitation.strategy == "value_advantage":
            from train.imitation_weights import compute_model_advantage_weights

            # Compute model-dependent advantages: ground_truth - prediction
            # Positive = actual outcome better than predicted (learn from expert!)
            # Negative = actual outcome worse than predicted (expert mistake or model overestimate)
            (
                imitation_weights_tensor,
                advantages_tensor,
            ) = compute_model_advantage_weights(
                value_pred=value_pred.squeeze(-1),  # [B, L, 1] -> [B, L]
                value_target=value_target.squeeze(-1),  # [B, L, 1] -> [B, L]
                alpha=config.imitation.advantage_alpha,
                return_advantages=True,
            )
        else:
            imitation_weights_tensor = compute_imitation_weights(
                X, components.value_idx, config.imitation
            )

        # Combine change-based and value-based weights
        # Apply value weights to all components
        combined_weights = {}
        for key, change_w in change_weights.items():
            # Multiply change weights by value weights
            # Handle broadcasting: change_w might be [B, L] or [B, L, num_classes]
            # imitation_weights is [B, L], so reshape to [B, L, 1] for broadcasting if needed
            if change_w.ndim == 3:
                # change_w is [B, L, C], so broadcast imitation weights to [B, L, 1]
                combined_weights[key] = change_w * imitation_weights_tensor.unsqueeze(
                    -1
                )
            else:
                # change_w is [B, L], direct multiplication
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
        value_loss_raw = torch.nn.functional.mse_loss(
            value_pred, value_target, reduction="none"
        ).squeeze(-1)

        # CRITICAL: Value head trains on FULL distribution (not filtered)
        # Policy heads use combined_weights (filtered for high-value states)
        # This prevents value head from learning biased estimator while
        # policy heads still benefit from focusing on winning play
        loss_value = value_loss_raw.mean()  # Uniform weighting

        scaled_value_loss = config.rl.value_loss_coef * loss_value
        loss = loss + scaled_value_loss
        loss_components[
            "value"
        ] = scaled_value_loss  # Log scaled version for accurate reporting

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
        weights=combined_weights,
        loss=loss,
        loss_components=loss_components,
        value_pred=value_pred,
        value_target=value_target,
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        label_smoothing=label_smoothing,
        change_scale=imbalance_scale,
        imitation_weights=imitation_weights_tensor,
        advantages=advantages_tensor,
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

    # Clip value head gradients separately BEFORE global clipping
    # This prevents value head from corrupting transformer even if it has large errors
    grad_clip = components.config.train.grad_clip
    grad_stats: Dict[str, float] = {}

    if collect_grad_stats:
        # Slow path: Compute gradient norms for logging (causes CPU-GPU sync)
        value_head_grad_norm = float(
            clip_grad_norm_(components.model.value_head.parameters(), max_norm=1.0)
        )
        pre_clip_norm = float(clip_grad_norm_(components.model.parameters(), grad_clip))

        grad_stats = collect_gradient_diagnostics(components.model)
        grad_stats["total_norm_pre_clip"] = pre_clip_norm
        grad_stats["total_norm_post_clip"] = min(pre_clip_norm, grad_clip)
        grad_stats["value_head_norm_pre_clip"] = value_head_grad_norm
        grad_stats["value_head_norm_post_clip"] = min(value_head_grad_norm, 1.0)
    else:
        # Fast path: Just clip without computing norms (no sync!)
        clip_grad_norm_(components.model.value_head.parameters(), max_norm=1.0)
        clip_grad_norm_(components.model.parameters(), grad_clip)

    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    return grad_stats
