"""Utilities for optionally wrapping models with torch.compile."""

from __future__ import annotations

from typing import Optional

import torch


def maybe_torch_compile(
    module: torch.nn.Module,
    *,
    label: str = "model",
    enable: bool = True,
    mode: Optional[str] = "reduce-overhead",
    fullgraph: bool = False,
    **compile_kwargs,
) -> torch.nn.Module:
    """Compile ``module`` with torch.compile when available.

    Args:
        module: The nn.Module to (optionally) compile.
        label: Friendly name used in log messages.
        enable: Master switch; when False the module is returned unchanged.
        mode: torch.compile ``mode`` argument (defaults to "reduce-overhead").
        fullgraph: Whether to require a single full graph during compilation.
        **compile_kwargs: Additional keyword args forwarded to torch.compile.

    Returns:
        The compiled module when compilation succeeds, otherwise the original
        module.
    """

    compile_fn = getattr(torch, "compile", None)
    if not enable or compile_fn is None:
        return module

    if getattr(module, "_is_compiled_with_torch_compile", False):
        return module

    kwargs = {"mode": mode, "fullgraph": fullgraph}
    kwargs.update(compile_kwargs)

    try:
        compiled = compile_fn(module, **kwargs)
        setattr(compiled, "_is_compiled_with_torch_compile", True)
        print(
            f"[torch.compile] Compiled {label} (mode={kwargs.get('mode')}, fullgraph={kwargs.get('fullgraph')})"
        )
        return compiled
    except Exception as exc:  # pragma: no cover - defensive logging
        print(
            f"[torch.compile] Failed to compile {label}: {exc}. Proceeding without compilation."
        )
        return module


__all__ = ["maybe_torch_compile"]
