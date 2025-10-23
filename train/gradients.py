"""Gradient utilities for training."""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer


def _move_optimizer_state_to_device(optimizer: Optimizer, device: torch.device) -> None:
    """Move optimizer state tensors to specified device."""
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def collect_gradient_diagnostics(model: torch.nn.Module, eps: float = 1e-12) -> Dict[str, float]:
    """Aggregate gradient statistics for monitoring numerical stability."""
    total_sq = 0.0
    total_abs = 0.0
    total_sum = 0.0
    grad_elems = 0
    zero_elems = 0
    nan_elems = 0
    inf_elems = 0
    max_grad_abs = 0.0
    params_with_grad = 0

    total_param_sq = 0.0
    max_param_abs = 0.0
    ratio_sum = 0.0
    ratio_max = 0.0
    ratio_min = float("inf")
    ratio_count = 0

    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        params_with_grad += 1

        grad_data = grad.detach()
        grad_float = grad_data.float()

        sum_sq = grad_float.pow(2).sum().item()
        total_sq += sum_sq
        abs_sum = grad_float.abs().sum().item()
        total_abs += abs_sum
        total_sum += grad_float.sum().item()

        numel = grad_float.numel()
        grad_elems += numel
        zero_elems += int((grad_float == 0).sum().item())
        nan_elems += int(torch.isnan(grad_float).sum().item())
        inf_elems += int(torch.isinf(grad_float).sum().item())

        if numel:
            max_grad_abs = max(max_grad_abs, float(grad_float.abs().max().item()))

        param_data = param.detach().float()
        total_param_sq += param_data.pow(2).sum().item()
        if param_data.numel():
            max_param_abs = max(max_param_abs, float(param_data.abs().max().item()))

        param_abs_mean = float(param_data.abs().mean().item()) if param_data.numel() else 0.0
        grad_abs_mean = float(grad_float.abs().mean().item()) if numel else 0.0
        if param_abs_mean > eps and numel:
            ratio = grad_abs_mean / max(param_abs_mean, eps)
            ratio_sum += ratio
            ratio_count += 1
            ratio_max = max(ratio_max, ratio)
            ratio_min = min(ratio_min, ratio)

    total_norm = math.sqrt(total_sq) if total_sq > 0 else 0.0
    mean_abs = total_abs / max(1, grad_elems)
    mean_val = total_sum / max(1, grad_elems)
    mean_sq = total_sq / max(1, grad_elems)
    variance = max(mean_sq - mean_val ** 2, 0.0)
    std_val = math.sqrt(variance)
    zero_fraction = zero_elems / max(1, grad_elems)

    param_total_norm = math.sqrt(total_param_sq) if total_param_sq > 0 else 0.0
    ratio_avg = ratio_sum / ratio_count if ratio_count else 0.0
    ratio_min = ratio_min if ratio_count else 0.0
    grad_to_param_ratio = total_norm / max(param_total_norm, eps)

    return {
        "total_norm": float(total_norm),
        "mean_abs": float(mean_abs),
        "mean": float(mean_val),
        "std": float(std_val),
        "max_abs": float(max_grad_abs),
        "zero_fraction": float(zero_fraction),
        "num_elements": float(grad_elems),
        "zero_count": float(zero_elems),
        "nan_count": float(nan_elems),
        "inf_count": float(inf_elems),
        "nonfinite_count": float(nan_elems + inf_elems),
        "params_with_grad": float(params_with_grad),
        "param_total_norm": float(param_total_norm),
        "param_max_abs": float(max_param_abs),
        "grad_param_ratio_mean": float(ratio_avg),
        "grad_param_ratio_max": float(ratio_max if ratio_count else 0.0),
        "grad_param_ratio_min": float(ratio_min),
        "grad_to_param_norm_ratio": float(grad_to_param_ratio),
    }


def clip_gradients_with_diagnostics(
        model: torch.nn.Module,
        max_norm: float,
        scaler: Optional[GradScaler] = None
) -> Dict[str, float]:
    """Clip gradients and return diagnostics.
    
    Args:
        model: Model to clip gradients for
        max_norm: Maximum gradient norm
        scaler: Optional GradScaler for AMP training
        
    Returns:
        Dictionary with gradient statistics including clipping info
    """
    stats: Dict[str, float] = {}
    
    # Unscale if using AMP
    if scaler is not None and scaler.is_enabled():
        scaler.unscale_(model.parameters().__iter__().__next__().grad.device)
    
    # Collect stats before clipping
    pre_clip_stats = collect_gradient_diagnostics(model)
    pre_clip_norm = pre_clip_stats["total_norm"]
    
    # Clip gradients
    clip_grad_norm_(model.parameters(), max_norm)
    
    # Compute post-clip norm (actual will be min of pre_clip and max_norm)
    post_clip_norm = min(pre_clip_norm, max_norm)
    was_clipped = pre_clip_norm > max_norm
    clip_coef = max_norm / max(pre_clip_norm, 1e-12) if was_clipped else 1.0
    
    stats.update(pre_clip_stats)
    stats["total_norm_pre_clip"] = pre_clip_norm
    stats["total_norm_post_clip"] = post_clip_norm
    stats["was_clipped"] = float(was_clipped)
    stats["clip_coef"] = clip_coef
    
    return stats
