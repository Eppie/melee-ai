from __future__ import annotations

from typing import TypeAlias

import numpy as np
import torch

RawNumpyArray: TypeAlias = np.ndarray
ProcessedNumpyArray: TypeAlias = np.ndarray
RawTorchTensor: TypeAlias = torch.Tensor
ProcessedTorchTensor: TypeAlias = torch.Tensor

RawFeatureArray: TypeAlias = RawNumpyArray
TransformedFeatureArray: TypeAlias = ProcessedNumpyArray
RawTargetArray: TypeAlias = RawNumpyArray

FeatureBlockArray: TypeAlias = np.ndarray
TargetBlockArray: TypeAlias = np.ndarray

RawFeatureTensor: TypeAlias = RawTorchTensor
TransformedFeatureTensor: TypeAlias = ProcessedTorchTensor
RawTargetTensor: TypeAlias = RawTorchTensor
TransformedTargetTensor: TypeAlias = ProcessedTorchTensor

__all__ = [
    "RawNumpyArray",
    "ProcessedNumpyArray",
    "RawTorchTensor",
    "ProcessedTorchTensor",
    "RawFeatureArray",
    "TransformedFeatureArray",
    "RawTargetArray",
    "FeatureBlockArray",
    "TargetBlockArray",
    "RawFeatureTensor",
    "TransformedFeatureTensor",
    "RawTargetTensor",
    "TransformedTargetTensor",
]
