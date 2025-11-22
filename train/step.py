"""Core forward/backward training step helpers."""

from __future__ import annotations

from typing import Dict
import time

import torch
from torch.amp import autocast
from torch.nn.utils import clip_grad_norm_

from loss import compute_loss_components
from train.batch_utils import (
    build_model_inputs,
    compute_value_based_weights,
)
from train.components import ForwardPassResult, TrainingComponents
from train.gradients import collect_gradient_diagnostics
from train.value_head import compute_value_targets
from controller_quantization import quantize_targets


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
    t0 = time.perf_counter()
    value_stats: Dict[str, float] = {}

    with autocast(
        device_type=amp.device_type,
        dtype=amp.dtype,
        enabled=amp.enabled,
    ):
        t_inputs_start = time.perf_counter()
        inputs_td = build_model_inputs(X, components.column_map)
        t_inputs = time.perf_counter()
        target_info = quantize_targets(Y, components.column_map, input_domain="unit01")
        t_quant = time.perf_counter()
        pred = components.model(inputs_td)
        t_model = time.perf_counter()

        value_target = compute_value_targets(
            X,
            components.column_map,
            gamma=config.rl.gamma,
            reward_idx=components.value_idx,
        )
        t_value_target = time.perf_counter()

        value_pred = pred.get("value")
        if value_pred is None:
            raise ValueError("Model predictions missing value head output.")
        if not torch.isfinite(value_pred).all():
            raise ValueError("Value predictions contain non-finite values.")

        value_weights = compute_value_based_weights(value_target, config.loss_weights)
        assert torch.isfinite(value_weights).all(), "Non-finite value-based weights"
        t_value_weights = time.perf_counter()

        # Collect stats so we can later replace with fixed scalars if desired.
        vt = value_target.detach().float().reshape(-1)
        vw = value_weights.detach().float().reshape(-1)
        min_clip_cfg = 0.1
        max_clip_cfg = float(config.loss_weights.value_weight_clip)
        # Compute all stats on device, sync once.
        vt_mean = vt.mean()
        vt_std = vt.std(unbiased=False)
        vt_min = vt.min()
        vt_max = vt.max()
        vw_mean = vw.mean()
        vw_std = vw.std(unbiased=False)
        vw_min = vw.min()
        vw_max = vw.max()
        vw_clip_lo_pct = (vw <= min_clip_cfg + 1e-6).float().mean() * 100.0
        vw_clip_hi_pct = (vw >= max_clip_cfg - 1e-6).float().mean() * 100.0
        stats_tensor = torch.stack(
            [
                vt_mean,
                vt_std,
                vt_min,
                vt_max,
                vw_mean,
                vw_std,
                vw_min,
                vw_max,
                vw_clip_lo_pct,
                vw_clip_hi_pct,
            ]
        )
        (
            vt_mean_v,
            vt_std_v,
            vt_min_v,
            vt_max_v,
            vw_mean_v,
            vw_std_v,
            vw_min_v,
            vw_max_v,
            vw_clip_lo_pct_v,
            vw_clip_hi_pct_v,
        ) = stats_tensor.cpu().tolist()
        value_stats = {
            "vt_mean": vt_mean_v,
            "vt_std": vt_std_v,
            "vt_min": vt_min_v,
            "vt_max": vt_max_v,
            "vw_mean": vw_mean_v,
            "vw_std": vw_std_v,
            "vw_min": vw_min_v,
            "vw_max": vw_max_v,
            "vw_clip_lo_pct": vw_clip_lo_pct_v,
            "vw_clip_hi_pct": vw_clip_hi_pct_v,
            "vw_scale_cfg": float(config.loss_weights.value_weight_scale),
            "vw_clip_min_cfg": min_clip_cfg,
            "vw_clip_max_cfg": max_clip_cfg,
        }

        base_smoothing = config.train.label_smoothing
        final_smoothing = 0.5 * base_smoothing
        if in_warmup:
            label_smoothing = base_smoothing
        else:
            label_smoothing = (
                base_smoothing + (final_smoothing - base_smoothing) * progress
            )
        label_smoothing = float(max(label_smoothing, 0.0))

        policy_loss_components = compute_loss_components(
            pred,
            target_info,
            label_smoothing=label_smoothing,
            sample_weights=value_weights,
            loss_config=config.loss_weights,
        )
        loss = policy_loss_components["total"]
        loss_components = dict(policy_loss_components)
        t_loss_policy = time.perf_counter()

        value_loss_raw = torch.nn.functional.mse_loss(
            value_pred, value_target, reduction="none"
        ).squeeze(-1)
        loss_value = (value_loss_raw * value_weights).mean()
        loss = loss + config.rl.value_loss_coef * loss_value
        loss_components["value"] = loss_value
        t_loss_value = time.perf_counter()

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
        weights=value_weights,
        loss=loss,
        loss_components=loss_components,
        value_pred=value_pred,
        value_target=value_target,
        batch_inputs=batch_inputs,
        batch_targets=batch_targets,
        label_smoothing=label_smoothing,
        timing_ms={
            "inputs_ms": 1000.0 * (t_inputs - t_inputs_start),
            "quantize_ms": 1000.0 * (t_quant - t_inputs),
            "model_ms": 1000.0 * (t_model - t_quant),
            "value_target_ms": 1000.0 * (t_value_target - t_model),
            "value_weights_ms": 1000.0 * (t_value_weights - t_value_target),
            "loss_policy_ms": 1000.0 * (t_loss_policy - t_value_weights),
            "loss_value_ms": 1000.0 * (t_loss_value - t_loss_policy),
            "forward_total_ms": 1000.0 * (t_loss_value - t0),
        },
        value_stats=value_stats,
    )

# TODO: We are probably failing to call clip_grad_norm if collect_grad_stats is False
def perform_backward_pass(
    components: TrainingComponents,
    loss: torch.Tensor,
    *,
    collect_grad_stats: bool,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Backpropagates the loss, steps the optimizer, and optionally collects grad stats."""
    optimizer = components.optimizer
    scaler = components.scaler
    t0 = time.perf_counter()

    loss_detached = loss.detach()
    if not torch.isfinite(loss_detached).all():
        loss_value = float(loss_detached.float().cpu().item())
        print(f"Warning: Non-finite loss ({loss_value}); skipping backward step")
        return {}, {}

    optimizer.zero_grad(set_to_none=True)
    t_zero = time.perf_counter()
    scaler.scale(loss).backward()
    t_backward = time.perf_counter()

    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    t_unscale = time.perf_counter()

    # Gradient clipping temporarily disabled for performance profiling.
    pre_clip_norm = float("nan")
    t_clip = time.perf_counter()

    grad_stats: Dict[str, float] = {}
    if collect_grad_stats:
        grad_stats = collect_gradient_diagnostics(components.model)

    scaler.step(optimizer)
    scaler.update()
    t_step = time.perf_counter()

    timing_ms = {
        "zero_grad_ms": 1000.0 * (t_zero - t0),
        "backward_ms": 1000.0 * (t_backward - t_zero),
        "unscale_ms": 1000.0 * (t_unscale - t_backward),
        "clip_ms": 0.0,
        "optimizer_step_ms": 1000.0 * (t_step - t_clip),
        "backward_total_ms": 1000.0 * (t_step - t0),
    }
    return grad_stats, timing_ms
