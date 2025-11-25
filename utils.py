from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def strip_compiled_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Strip _orig_mod. prefix from state dict keys if present.

    This handles the case where a model was saved with torch.compile()
    (which adds _orig_mod. prefix) but is being loaded into a non-compiled model.

    Args:
        state_dict: State dictionary potentially with _orig_mod. prefix

    Returns:
        State dictionary with _orig_mod. prefix stripped if all keys have it
    """
    PREFIX = "_orig_mod."

    # Check if all keys start with the prefix
    all_have_prefix = all(k.startswith(PREFIX) for k in state_dict.keys())

    if all_have_prefix:
        return {k[len(PREFIX):]: v for k, v in state_dict.items()}

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
