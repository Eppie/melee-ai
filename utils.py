from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch


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
