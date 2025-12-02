from __future__ import annotations

import math
import time

import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def match_state_dict_keys(
    state_dict: Dict[str, Any], model: nn.Module
) -> Dict[str, Any]:
    """Adjust state dict keys to match model's compilation state.

    Handles mismatches between compiled/non-compiled models and checkpoints:
    - If model is compiled but checkpoint isn't -> add _orig_mod. prefix
    - If checkpoint is compiled but model isn't -> strip _orig_mod. prefix
    - If both match -> return unchanged

    Args:
        state_dict: State dictionary from checkpoint
        model: Model to load the state dict into

    Returns:
        State dictionary with keys adjusted to match the model
    """
    PREFIX = "_orig_mod."

    if not state_dict:
        return state_dict

    # Check if checkpoint has compiled keys
    ckpt_has_prefix = all(k.startswith(PREFIX) for k in state_dict.keys())

    # Check if model has compiled keys by examining its state dict
    model_keys = list(model.state_dict().keys())
    if not model_keys:
        return state_dict
    model_has_prefix = all(k.startswith(PREFIX) for k in model_keys)

    # Case 1: Model is compiled but checkpoint isn't -> add prefix
    if model_has_prefix and not ckpt_has_prefix:
        return {f"{PREFIX}{k}": v for k, v in state_dict.items()}

    # Case 2: Checkpoint is compiled but model isn't -> strip prefix
    if ckpt_has_prefix and not model_has_prefix:
        return {k[len(PREFIX) :]: v for k, v in state_dict.items()}

    # Case 3: Both match -> return unchanged
    return state_dict


def _resolve_device(preferred: Optional[str] = None) -> torch.device:
    if preferred is None or preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


def _collect_param_stats(
    module: nn.Module,
    cache: Dict[int, Tuple[int, int]],
) -> Tuple[int, int]:
    module_id = id(module)
    if module_id in cache:
        return cache[module_id]

    direct_total = 0
    direct_trainable = 0
    for param in module.parameters(recurse=False):
        count = param.numel()
        direct_total += count
        if param.requires_grad:
            direct_trainable += count

    total = direct_total
    trainable = direct_trainable
    for child in module.children():
        child_total, child_trainable = _collect_param_stats(child, cache)
        total += child_total
        trainable += child_trainable

    cache[module_id] = (total, trainable)
    return total, trainable


def _module_label(module: nn.Module, cache: Dict[int, Tuple[int, int]]) -> str:
    base = module.__class__.__name__
    extras: List[str] = []

    if isinstance(module, nn.Linear):
        extras.append(f"{module.in_features}→{module.out_features}")
        extras.append("bias" if module.bias is not None else "no-bias")
    elif isinstance(module, nn.Embedding):
        extras.append(f"{module.num_embeddings}×{module.embedding_dim}")
    elif isinstance(module, nn.LayerNorm):
        extras.append(f"shape={tuple(module.normalized_shape)}")
    elif module.__class__.__name__ == "RMSNorm" and hasattr(module, "weight"):
        weight = getattr(module, "weight")
        if isinstance(weight, torch.Tensor):
            extras.append(f"dim={weight.shape[0]}")
    elif isinstance(module, nn.Dropout):
        extras.append(f"p={module.p}")
    elif isinstance(module, (nn.ModuleList, nn.Sequential)):
        extras.append(f"len={len(module)}")
    elif isinstance(module, nn.ModuleDict):
        extras.append(f"keys={len(module)}")
    elif module.__class__.__name__ == "ActivationFFN":
        mult = getattr(module, "mult", None)
        act = getattr(module, "activation", None)
        if mult is not None:
            extras.append(f"mult={mult}")
        if act is not None:
            extras.append(f"act={act}")
    elif module.__class__.__name__ == "MLP":
        extras.append("dense")

    total_params, trainable_params = _collect_param_stats(module, cache)
    if total_params:
        if trainable_params == total_params:
            extras.append(f"params={total_params:,}")
        else:
            extras.append(f"params={trainable_params:,}/{total_params:,}")

    if not extras:
        return base
    return f"{base} ({', '.join(extras)})"


def print_model_diagram(
    model: nn.Module,
) -> str:
    """Print an ASCII diagram of ``model`` and return the rendered text.

    Parameters
    ----------
    model:
        The module to render.
    """

    param_cache: Dict[int, Tuple[int, int]] = {}
    visited: set[int] = {id(model)}
    lines: list[str] = [_module_label(model, param_cache)]

    def _render(module: nn.Module, indent: str, depth: int) -> None:
        children = list(module.named_children())
        for idx, (child_name, child) in enumerate(children):
            is_last = idx == len(children) - 1
            branch = "└── " if is_last else "├── "
            label = _module_label(child, param_cache)

            child_id = id(child)
            if child_id in visited:
                lines.append(f"{indent}{branch}{child_name}: {label} [shared]")
                continue

            lines.append(f"{indent}{branch}{child_name}: {label}")
            visited.add(child_id)

            grand_children = list(child.named_children())
            if not grand_children:
                continue

            new_indent = indent + ("    " if is_last else "│   ")
            _render(child, new_indent, depth + 1)

    _render(model, "", 0)

    diagram = "\n".join(lines)
    print(diagram, file=sys.stdout)
    return diagram


class Profiler:
    """
    Lightweight wall-clock profiler.

    Stats ignore the burn-in iterations once the burn-in period has completed
    """

    def __init__(self, burnin: int = 1, ema_alpha: float = 0.1) -> None:
        # Core counters
        self.cumtime: float = 0.0
        self.cumtime_sq: float = 0.0  # for variance/stddev
        self.num_calls: int = 0

        # Burn-in handling
        self.burnin: int = burnin
        self.needs_reset: bool = False

        # Extra stats
        self.min_time: float = math.inf
        self.max_time: float = 0.0
        self.last_time: float = 0.0

        # Exponential moving average of time
        self.ema_alpha: float = ema_alpha
        self.ema_time: Optional[float] = None

        # Internal timing
        self._enter_time: float = 0.0

    def _reset_stats(self) -> None:
        self.cumtime = 0.0
        self.cumtime_sq = 0.0
        self.num_calls = 0
        self.min_time = math.inf
        self.max_time = 0.0
        self.last_time = 0.0
        self.ema_time = None

    def __enter__(self) -> "Profiler":
        if self.needs_reset:
            self._reset_stats()
            self.needs_reset = False

        self._enter_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        duration = time.perf_counter() - self._enter_time

        # Update stats
        self.num_calls += 1
        self.cumtime += duration
        self.cumtime_sq += duration * duration
        self.last_time = duration

        if duration < self.min_time:
            self.min_time = duration
        if duration > self.max_time:
            self.max_time = duration

        if self.ema_alpha > 0.0:
            if self.ema_time is None:
                self.ema_time = duration
            else:
                self.ema_time += self.ema_alpha * (duration - self.ema_time)

        if self.burnin > 0:
            self.burnin -= 1
            if self.burnin == 0:
                self.needs_reset = True

    # --- Public accessors ---

    def mean_time(self) -> float:
        """Arithmetic mean of durations (seconds)."""
        if self.num_calls == 0:
            return 0.0
        return self.cumtime / self.num_calls

    def std_time(self) -> float:
        """Sample standard deviation of durations (seconds)."""
        if self.num_calls < 2:
            return 0.0
        n = self.num_calls
        mean = self.cumtime / n
        # variance = E[x^2] - (E[x])^2
        var = max(self.cumtime_sq / n - mean * mean, 0.0)
        return math.sqrt(var)

    def total_time(self) -> float:
        """Total accumulated time (seconds)."""
        return self.cumtime

    def min_duration(self) -> float:
        """Minimum observed duration (seconds)."""
        if self.num_calls == 0 or self.min_time is math.inf:
            return 0.0
        return self.min_time

    def max_duration(self) -> float:
        """Maximum observed duration (seconds)."""
        return self.max_time if self.num_calls > 0 else 0.0

    def last_duration(self) -> float:
        """Duration of the most recent call (seconds)."""
        return self.last_time

    def ema_duration(self) -> float:
        """
        Exponential moving average of durations (seconds).

        Smoother than raw mean when there is drift; 0.0 if no data yet.
        """
        return self.ema_time if self.ema_time is not None else 0.0

    def calls_per_second(self) -> float:
        """Throughput based on accumulated time."""
        if self.cumtime <= 0.0:
            return 0.0
        return self.num_calls / self.cumtime

    def summary(self) -> Dict[str, float]:
        """Convenient snapshot of all stats."""
        return {
            "num_calls": float(self.num_calls),
            "total_time": self.total_time(),
            "mean_time": self.mean_time(),
            "std_time": self.std_time(),
            "min_time": self.min_duration(),
            "max_time": self.max_duration(),
            "last_time": self.last_duration(),
            "ema_time": self.ema_duration(),
            "calls_per_second": self.calls_per_second(),
        }
