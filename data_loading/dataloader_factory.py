"""DataLoaderFactory: Single source of truth for DataLoader creation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

if TYPE_CHECKING:
    from config.config import Config


def worker_init_fn(worker_id: int) -> None:
    """Seed NumPy and PyTorch for worker_id.

    Args:
        worker_id: Worker process ID assigned by PyTorch
    """
    import numpy as np

    # Same recipe as PyTorch DistributedSampler docs
    base_seed = torch.initial_seed() % 2**31
    np.random.seed(base_seed + worker_id)


class DataLoaderFactory:
    """Factory for creating PyTorch DataLoaders with consistent configuration."""

    @staticmethod
    def create(
        dataset: Dataset,
        sampler: Sampler,
        config: "Config",
    ) -> DataLoader:
        """Create a DataLoader with configuration from config.

        Args:
            dataset: PyTorch Dataset to load from
            sampler: PyTorch Sampler for sampling indices
            config: Training configuration

        Returns:
            Configured PyTorch DataLoader
        """
        mp_ctx = DataLoaderFactory._get_multiprocessing_context(config)
        pin_memory = config.train.pin_memory
        prefetch_factor = DataLoaderFactory._calculate_prefetch_factor(config)

        return DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            sampler=sampler,
            num_workers=config.train.num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=config.train.persistent_workers,
            worker_init_fn=worker_init_fn,
            drop_last=False,
            multiprocessing_context=mp_ctx,
        )

    @staticmethod
    def _get_multiprocessing_context(
        config: "Config",
    ) -> Optional[torch.multiprocessing.SpawnContext]:
        """Get multiprocessing context based on worker_start_method."""
        start_method = config.train.worker_start_method

        if not config.train.num_workers or config.train.num_workers <= 0:
            return None

        if not start_method:
            return None

        try:
            return torch.multiprocessing.get_context(start_method)
        except RuntimeError as exc:
            print(
                f"[dataloader] Requested start method '{start_method}' unavailable "
                f"({exc}); falling back to PyTorch default."
            )
            return None

    @staticmethod
    def _calculate_prefetch_factor(config: "Config") -> Optional[int]:
        """Calculate prefetch_factor (None if num_workers == 0)."""
        num_workers = config.train.num_workers

        if not num_workers or num_workers <= 0:
            return None

        return config.train.prefetch_factor
