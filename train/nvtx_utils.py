"""NVTX profiling utilities for training instrumentation.

Provides a lightweight wrapper around torch.cuda.nvtx.range
for conditional NVTX range annotation during training.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch


class NVTXContext:
    """Context manager for NVTX range profiling.

    Wraps torch.cuda.nvtx.range to emit NVTX ranges that can be
    viewed in profiling tools like Nsight Systems.

    Usage:
        nvtx_ctx = NVTXContext(enabled=True)

        with nvtx_ctx("initialization"):
            # initialization code

        with nvtx_ctx("training_step"):
            # training step code
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize NVTX context.

        Args:
            enabled: Whether to emit NVTX ranges. If False, uses nullcontext
                    for zero overhead.
        """
        self.enabled = enabled

    def __call__(self, name: str):
        """Create a context manager for the given NVTX range name.

        Args:
            name: Name of the NVTX range to emit

        Returns:
            Context manager that emits an NVTX range when enabled
        """
        if self.enabled:
            return torch.cuda.nvtx.range(name)
        return nullcontext()


@contextmanager
def nvtx_range(name: str, enabled: bool = True) -> Iterator[None]:
    """Standalone context manager for NVTX ranges.

    Args:
        name: Name of the NVTX range to emit
        enabled: Whether to emit the NVTX range

    Yields:
        None

    Example:
        with nvtx_range("data_loading"):
            batch = next(dataloader)
    """
    if enabled:
        with torch.cuda.nvtx.range(name):
            yield
    else:
        yield
