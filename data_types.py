from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypeAlias

import numpy as np  # type: ignore[import]
import torch  # type: ignore[import]

RawNumpyArray: TypeAlias = np.ndarray
ProcessedNumpyArray: TypeAlias = np.ndarray
RawTorchTensor: TypeAlias = torch.Tensor
ProcessedTorchTensor: TypeAlias = torch.Tensor


@dataclass(frozen=True)
class RawNumpyBatch:
    """Container for raw observation data plus lightweight provenance."""

    data: RawNumpyArray
    zarr_path: Optional[str] = None


@dataclass(frozen=True)
class RawTorchBatch:
    """Container for raw torch tensors plus lightweight provenance."""

    data: RawTorchTensor
    zarr_path: Optional[str] = None


@dataclass(frozen=True)
class ProcessedNumpyBatch:
    """Container for transformed data alongside its provenance."""

    data: ProcessedNumpyArray
    zarr_path: Optional[str] = None
    raw_source: Optional[str] = None
    transform_id: Optional[str] = None


@dataclass(frozen=True)
class ProcessedTorchBatch:
    """Container for transformed torch tensors alongside their provenance."""

    data: ProcessedTorchTensor
    zarr_path: Optional[str] = None
    raw_source: Optional[str] = None
    transform_id: Optional[str] = None


__all__ = [
    "RawNumpyArray",
    "ProcessedNumpyArray",
    "RawTorchTensor",
    "ProcessedTorchTensor",
    "RawNumpyBatch",
    "RawTorchBatch",
    "ProcessedNumpyBatch",
    "ProcessedTorchBatch",
]
