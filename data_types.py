from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch

RawNumpyArray: TypeAlias = np.ndarray
ProcessedNumpyArray: TypeAlias = np.ndarray
RawTorchTensor: TypeAlias = torch.Tensor
ProcessedTorchTensor: TypeAlias = torch.Tensor

__all__ = [
    "RawNumpyArray",
    "ProcessedNumpyArray",
    "RawTorchTensor",
    "ProcessedTorchTensor",
]
