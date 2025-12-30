"""Gradient utilities for training."""

from __future__ import annotations

import math
from typing import Dict

import torch
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
    """Aggregate gradient statistics for monitoring numerical stability.

    Optimized to batch GPU operations and minimize CPU transfers.

    Args:
        model: Module whose gradients will be inspected.
        eps: Small constant used when computing gradient-to-parameter ratios.

    Returns:
        Dictionary mapping metric names to floating-point summaries of the gradients.
    """
    # Collect all gradients and parameters with gradients
    grads = []
    params = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.detach().float().reshape(-1))
            params.append(param.detach().float().reshape(-1))

    if not grads:
        return {
            "total_norm": 0.0,
            "mean_abs": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "max_abs": 0.0,
            "zero_fraction": 0.0,
            "nan_count": 0.0,
            "inf_count": 0.0,
            "param_total_norm": 0.0,
            "param_max_abs": 0.0,
            "grad_to_param_norm_ratio": 0.0,
        }

    # Concatenate all gradients and params for batched operations
    all_grads = torch.cat(grads)
    all_params = torch.cat(params)
    grad_elems = all_grads.numel()

    # Compute all gradient statistics on GPU
    grad_sq = all_grads.pow(2)
    grad_abs = all_grads.abs()
    param_sq = all_params.pow(2)
    param_abs = all_params.abs()

    # Batch all reductions into a single tensor for one GPU->CPU transfer
    stats = torch.stack(
        [
            grad_sq.sum(),  # 0: total_sq
            grad_abs.sum(),  # 1: total_abs
            all_grads.sum(),  # 2: total_sum
            (all_grads == 0).sum().float(),  # 3: zero_elems
            torch.isnan(all_grads).sum().float(),  # 4: nan_elems
            torch.isinf(all_grads).sum().float(),  # 5: inf_elems
            grad_abs.max(),  # 6: max_grad_abs
            param_sq.sum(),  # 7: total_param_sq
            param_abs.max(),  # 8: max_param_abs
        ]
    )

    # Single GPU->CPU transfer
    stats_cpu = stats.cpu().tolist()

    total_sq = stats_cpu[0]
    total_abs = stats_cpu[1]
    total_sum = stats_cpu[2]
    zero_elems = int(stats_cpu[3])
    nan_elems = int(stats_cpu[4])
    inf_elems = int(stats_cpu[5])
    max_grad_abs = stats_cpu[6]
    total_param_sq = stats_cpu[7]
    max_param_abs = stats_cpu[8]

    # Compute derived statistics
    total_norm = math.sqrt(total_sq) if total_sq > 0 else 0.0
    mean_abs = total_abs / max(1, grad_elems)
    mean_val = total_sum / max(1, grad_elems)
    mean_sq = total_sq / max(1, grad_elems)
    variance = max(mean_sq - mean_val**2, 0.0)
    std_val = math.sqrt(variance)
    zero_fraction = zero_elems / max(1, grad_elems)

    param_total_norm = math.sqrt(total_param_sq) if total_param_sq > 0 else 0.0
    grad_to_param_ratio = total_norm / max(param_total_norm, eps)

    return {
        "total_norm": float(total_norm),
        "mean_abs": float(mean_abs),
        "mean": float(mean_val),
        "std": float(std_val),
        "max_abs": float(max_grad_abs),
        "zero_fraction": float(zero_fraction),
        "nan_count": float(nan_elems),
        "inf_count": float(inf_elems),
        "param_total_norm": float(param_total_norm),
        "param_max_abs": float(max_param_abs),
        "grad_to_param_norm_ratio": float(grad_to_param_ratio),
    }
