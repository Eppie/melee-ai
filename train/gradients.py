"""Gradient utilities for training."""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer


def _move_optimizer_state_to_device(optimizer: Optimizer, device: torch.device) -> None:
    """Relocate every tensor in an optimizer's state dictionary onto ``device``.

    Example:
        Consider an ``Adam`` optimizer tracking ``exp_avg`` and ``exp_avg_sq`` on CPU. Calling
        ``_move_optimizer_state_to_device(optimizer, torch.device("cuda"))`` iterates through every
        parameter's state, finds the tensors, and replaces them with GPU copies produced by
        ``tensor.to(device)``. After the function completes, ``optimizer.state[p]["exp_avg"].device``
        reports ``cuda:0`` for each parameter, showing the step-by-step migration.

    Args:
        optimizer: Optimizer whose internal state tensors should be moved.
        device: Destination device.
    """
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def collect_gradient_diagnostics(
    model: torch.nn.Module, eps: float = 1e-12
) -> Dict[str, float]:
    """Aggregate gradient statistics for monitoring numerical stability with an explicit example.

    Example:
        Suppose ``model`` has two parameters with gradients ``tensor([1.0, -2.0])`` and
        ``tensor([0.0, float('nan')])``. ``collect_gradient_diagnostics`` traverses each gradient,
        accumulating sums (``total_sq = 1^2 + (-2)^2 = 5``), counting zeros (one element equals
        ``0.0``), and tracking NaNs (one element). The returned dictionary therefore includes
        ``{"total_norm": sqrt(5), "zero_count": 1, "nan_count": 1}`` among many other statistics.
        This walkthrough mirrors the exact sequence of reductions performed by the function.

    Args:
        model: Module whose gradients will be inspected.
        eps: Small constant used when computing gradient-to-parameter ratios.

    Returns:
        Dictionary mapping metric names to floating-point summaries of the gradients.
    """
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

        param_abs_mean = (
            float(param_data.abs().mean().item()) if param_data.numel() else 0.0
        )
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
    variance = max(mean_sq - mean_val**2, 0.0)
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
