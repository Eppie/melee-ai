"""
Data management module for Melee AI.

This module handles data storage, dataset creation, and data loading
for training. It provides abstractions for different storage formats
and sampling strategies.
"""

from .schema import Schema, Row
from .storage import StorageManager
from .dataset import DatasetProvider, WindowDataset, ParquetDatasetProvider
from .samplers import EpisodeLinearSampler, RandomWindowSampler

__all__ = [
    "Schema",
    "Row",
    "StorageManager",
    "DatasetProvider",
    "WindowDataset",
    "ParquetDatasetProvider",
    "EpisodeLinearSampler",
    "RandomWindowSampler",
]
