from __future__ import annotations

import sys
from contextlib import nullcontext
from typing import Dict, List, Optional, TextIO, Tuple

import torch
import torch.nn as nn


def _resolve_device(preferred: Optional[str]) -> torch.device:
    if preferred is None or preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


class AmpFP16:
    """
    Unified AMP helper:
      - CUDA: FP16 autocast + GradScaler (when training)
      - MPS:  FP16 autocast (no scaler due to current limitations)
      - CPU:  BF16 autocast (FP16 not supported on CPU)
    Use as a context manager. Exposes .scaler (Optional[torch.amp.GradScaler]) and
    convenience .backward(loss) / .step(optim) methods.
    """

    def __init__(self, device: torch.device, training: bool = True) -> None:
        self.device = device
        self.training = training
        self._ctx: object
        self.scaler: Optional[torch.amp.GradScaler] = None

    def __enter__(self) -> "AmpFP16":
        dt = self.device.type
        if dt == "cuda":
            # Standard CUDA AMP: FP16 autocast + GradScaler
            self._ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            if self.training:
                self.scaler = torch.amp.GradScaler(device="cuda")
        elif dt == "mps":
            # MPS AMP currently supports FP16 autocast only
            # (requires PyTorch >= 2.5). No scaler due to open issues.
            self._ctx = torch.amp.autocast(device_type="mps", dtype=torch.float16)
        elif dt == "cpu":
            # CPU autocast only supports BF16 (not FP16)
            self._ctx = torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
        else:
            self._ctx = nullcontext()
        # activate inner context
        assert hasattr(self._ctx, "__enter__") and hasattr(self._ctx, "__exit__")
        self._ctx.__enter__()  # type: ignore[call-arg]
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self._ctx.__exit__(exc_type, exc, tb)  # type: ignore[attr-defined]

    # Convenience wrappers so your training loop doesn't branch on scaler
    def backward(self, loss: torch.Tensor) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()


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
    elif module.__class__.__name__ == "MoEMLP":
        extras.append("moe" if getattr(module, "is_moe", False) else "dense")

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
        *,
        max_depth: Optional[int] = None,
        stream: Optional[TextIO] = None,
) -> str:
    """Print an ASCII diagram of ``model`` and return the rendered text.

    Parameters
    ----------
    model:
        The module to render.
    max_depth:
        Optional limit on recursion depth (root is depth 0). ``None`` prints
        the full module tree.
    stream:
        Optional stream to print to. Defaults to ``sys.stdout``.
    """

    param_cache: Dict[int, Tuple[int, int]] = {}
    visited: set[int] = {id(model)}
    lines: List[str] = []

    lines.append(_module_label(model, param_cache))

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

            if max_depth is not None and depth + 1 >= max_depth:
                continuation_indent = indent + ("    " if is_last else "│   ")
                lines.append(
                    f"{continuation_indent}└── … ({len(grand_children)} submodules)"
                )
                continue

            new_indent = indent + ("    " if is_last else "│   ")
            _render(child, new_indent, depth + 1)

    _render(model, "", 0)

    diagram = "\n".join(lines)
    destination = stream if stream is not None else sys.stdout
    print(diagram, file=destination)
    return diagram
