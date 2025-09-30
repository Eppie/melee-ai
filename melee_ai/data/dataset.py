"""
Dataset implementations for Melee AI.

This module provides dataset abstractions and implementations for
training data loading and sampling.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Protocol, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from melee_ai.config import Settings
from .schema import Row


class DatasetProvider(Protocol):
    """Protocol for dataset providers."""

    def get_dataset(self, settings: Settings) -> Dataset:
        """Create dataset from configuration."""
        ...

    def get_dataloader(self, settings: Settings) -> DataLoader:
        """Create dataloader from configuration."""
        ...


class WindowDataset(Dataset):
    """Dataset that provides sliding windows of sequential data."""

    def __init__(self, rows: List[Row], window_size: int, settings: Settings):
        """
        Initialize window dataset.

        Args:
            rows: List of Row objects
            window_size: Size of sliding window
            settings: Configuration settings
        """
        self.rows = rows
        self.window_size = window_size
        self.settings = settings

        # Pre-compute valid window indices
        self.valid_indices = []
        for i in range(len(rows) - window_size + 1):
            self.valid_indices.append(i)

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get window of data at index."""
        start_idx = self.valid_indices[idx]
        window_rows = self.rows[start_idx:start_idx + self.window_size]

        # Convert to tensors
        X = self._rows_to_features(window_rows[:-1])  # Input features (all but last)
        Y = self._rows_to_targets(window_rows[1:])    # Target features (all but first)

        return X, Y

    def _rows_to_features(self, rows: List[Row]) -> torch.Tensor:
        """Convert rows to feature tensor."""
        # Extract features for each row
        features = []
        for row in rows:
            row_features = [
                row.frame,
                row.stage,
                row.distance,
                # Add more features as needed
            ]
            features.append(row_features)

        return torch.tensor(features, dtype=torch.float32)

    def _rows_to_targets(self, rows: List[Row]) -> torch.Tensor:
        """Convert rows to target tensor."""
        # Extract targets for each row
        targets = []
        for row in rows:
            row_targets = [
                row.p1_main_stick_x, row.p1_main_stick_y,
                row.p1_c_stick_x, row.p1_c_stick_y,
                row.p1_button_a, row.p1_button_b, row.p1_button_xy,
                row.p1_button_z, row.p1_button_lr, row.p1_shoulder_analog,
            ]
            targets.append(row_targets)

        return torch.tensor(targets, dtype=torch.float32)


class ParquetDatasetProvider:
    """Dataset provider that loads from parquet files."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.storage_manager = None  # Would be injected

    def get_dataset(self, settings: Settings) -> Dataset:
        """Create dataset from parquet files."""
        # Load rows from parquet
        # For now, return empty dataset
        return WindowDataset([], settings.model.block_size, settings)

    def get_dataloader(self, settings: Settings) -> DataLoader:
        """Create dataloader from configuration."""
        dataset = self.get_dataset(settings)

        return DataLoader(
            dataset,
            batch_size=settings.training.batch_size,
            shuffle=True,
            num_workers=settings.training.num_workers,
            pin_memory=settings.training.pin_memory,
        )


class ZarrDatasetProvider:
    """Dataset provider that loads from zarr stores."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def get_dataset(self, settings: Settings) -> Dataset:
        """Create dataset from zarr store."""
        # Load rows from zarr
        return WindowDataset([], settings.model.block_size, settings)

    def get_dataloader(self, settings: Settings) -> DataLoader:
        """Create dataloader from configuration."""
        dataset = self.get_dataset(settings)

        return DataLoader(
            dataset,
            batch_size=settings.training.batch_size,
            shuffle=True,
            num_workers=settings.training.num_workers,
            pin_memory=settings.training.pin_memory,
        )
