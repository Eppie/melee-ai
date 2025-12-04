"""DataLoaderFactory: Single source of truth for DataLoader creation.

This module eliminates duplication between make_dataloader() and _build_loader_for_chunk()
by providing a centralized factory for creating PyTorch DataLoaders with consistent
configuration.
"""

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
    """Factory for creating PyTorch DataLoaders with consistent configuration.

    This class centralizes all DataLoader creation logic to eliminate duplication
    and ensure consistent behavior across different usage contexts (initial creation,
    chunked mode, etc.).
    """

    @staticmethod
    def create(
        dataset: Dataset,
        sampler: Sampler,
        config: "Config",
        is_chunked_mode: bool = False,
    ) -> DataLoader:
        """Create a DataLoader with configuration from config.

        Args:
            dataset: PyTorch Dataset to load from
            sampler: PyTorch Sampler for sampling indices
            config: Training configuration
            is_chunked_mode: If True, adjusts settings for chunked loading

        Returns:
            Configured PyTorch DataLoader

        Note:
            In chunked mode, persistent_workers is disabled because workers
            are recreated for each chunk to avoid stale state.
        """
        # Determine multiprocessing context
        mp_ctx = DataLoaderFactory._get_multiprocessing_context(config)

        # Determine pin_memory setting
        pin_memory = DataLoaderFactory._should_pin_memory(config)

        # Calculate prefetch_factor with memory budget constraints
        prefetch_factor = DataLoaderFactory._calculate_prefetch_factor(
            dataset=dataset, config=config
        )

        # Determine persistent_workers setting
        # In chunked mode, we disable persistent_workers to avoid stale state
        persistent_workers = config.train.persistent_workers and not is_chunked_mode

        # Create DataLoader
        loader = DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            sampler=sampler,
            num_workers=config.train.num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            worker_init_fn=worker_init_fn,
            drop_last=False,
            multiprocessing_context=mp_ctx,
        )

        return loader

    @staticmethod
    def _get_multiprocessing_context(
        config: "Config",
    ) -> Optional[torch.multiprocessing.SpawnContext]:
        """Get multiprocessing context based on worker_start_method.

        Args:
            config: Training configuration

        Returns:
            Multiprocessing context or None for default
        """
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
    def _should_pin_memory(config: "Config") -> bool:
        """Determine if pin_memory should be enabled.

        Args:
            config: Training configuration

        Returns:
            True if pin_memory should be enabled
        """
        return config.train.pin_memory

    @staticmethod
    def _calculate_prefetch_factor(
        dataset: Dataset,
        config: "Config",
    ) -> Optional[int]:
        """Calculate prefetch_factor respecting memory budget.

        Args:
            dataset: Dataset to estimate batch size for
            config: Training configuration

        Returns:
            Prefetch factor (None if num_workers == 0)
        """
        num_workers = config.train.num_workers

        if not num_workers or num_workers <= 0:
            return None

        prefetch_factor = config.train.prefetch_factor
        max_prefetch_mb = config.train.max_loader_prefetch_mb

        if not max_prefetch_mb:
            return prefetch_factor

        # Estimate batch size in bytes
        batch_bytes = DataLoaderFactory._estimate_batch_bytes(dataset, config)

        # Calculate budget
        max_prefetch_bytes = max_prefetch_mb * 1024 * 1024
        total_batches_budget = max_prefetch_bytes // max(1, batch_bytes)

        # Handle edge cases
        budget_saturated = False
        if total_batches_budget == 0:
            total_batches_budget = 1
            budget_saturated = True

        allowed_per_worker = total_batches_budget // num_workers
        if allowed_per_worker == 0:
            allowed_per_worker = 1
            budget_saturated = True

        # Reduce prefetch_factor if it exceeds budget
        if allowed_per_worker < prefetch_factor:
            approx_batch_mb = batch_bytes / (1024**2)
            print(
                "[dataloader] Reducing prefetch_factor from "
                f"{prefetch_factor} to {allowed_per_worker} to honor "
                f"{max_prefetch_mb} MiB prefetch budget (batch ≈ "
                f"{approx_batch_mb:.2f} MiB)."
            )
            if budget_saturated:
                print(
                    "[dataloader] Consider lowering train.num_workers or "
                    "batch_size, or increase train.max_loader_prefetch_mb "
                    "if you need more throughput."
                )
            prefetch_factor = allowed_per_worker

        return prefetch_factor

    @staticmethod
    def _estimate_batch_bytes(dataset: Dataset, config: "Config") -> int:
        """Estimate memory usage of a single batch.

        Args:
            dataset: Dataset to estimate from
            config: Training configuration

        Returns:
            Estimated bytes per batch
        """
        # The dataset is always a WindowDataset, which implements estimate_batch_bytes.
        return max(1, dataset.estimate_batch_bytes(config.train.batch_size))
