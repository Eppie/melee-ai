"""Memory profiling and optimization utilities for dynamic chunk sizing."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from window_dataset import ZarrCorpusIndex


def estimate_episode_memory(index: ZarrCorpusIndex, sample_size: int = 100) -> int:
    """Estimate average memory usage per episode by sampling.

    Args:
        index: ZarrCorpusIndex containing episode metadata
        sample_size: Number of episodes to sample for estimation

    Returns:
        Average bytes per episode (features + targets)
    """
    if len(index.episodes) == 0:
        return 0

    # Sample a subset of episodes to estimate memory usage
    actual_sample_size = min(sample_size, len(index.episodes))
    sampled_episodes = random.sample(index.episodes, actual_sample_size)

    total_bytes = 0
    for ep in sampled_episodes:
        feature_array, target_array = index.open_episode_arrays(ep)
        total_bytes += feature_array.nbytes + target_array.nbytes

    avg_bytes = total_bytes // actual_sample_size
    return avg_bytes


def calculate_optimal_chunk_size(
    total_episodes: int,
    bytes_per_episode: int,
    num_overlapping: int,
    ram_budget_mb: int,
    safety_margin: float = 0.8,
) -> int:
    """Calculate the largest chunk size that fits in available RAM.

    With multi-chunk overlap, we load `num_overlapping` chunks simultaneously.
    This function calculates how large each chunk can be given the RAM budget.

    Args:
        total_episodes: Total number of episodes in dataset
        bytes_per_episode: Average memory usage per episode
        num_overlapping: Number of chunks loaded simultaneously
        ram_budget_mb: Available RAM in megabytes
        safety_margin: Use this fraction of available RAM (default 0.8 = 80%)

    Returns:
        Chunk size in episodes (minimum 1, maximum total_episodes)

    Example:
        With 8192 MB available, 2 overlapping chunks, 4 MB/episode:
        - Available bytes: 8192 * 1024 * 1024 * 0.8 = 6,871,947,673 bytes
        - Bytes per chunk: 6,871,947,673 / 2 = 3,435,973,836 bytes
        - Chunk size: 3,435,973,836 / (4 * 1024 * 1024) ≈ 819 episodes
    """
    if bytes_per_episode <= 0:
        return total_episodes

    # Apply safety margin to avoid OOM
    available_bytes = int(ram_budget_mb * 1024 * 1024 * safety_margin)

    # Divide available memory by number of overlapping chunks
    bytes_per_chunk = available_bytes // max(1, num_overlapping)

    # Calculate chunk size
    chunk_size = int(bytes_per_chunk / bytes_per_episode)

    # Clamp to valid range
    chunk_size = max(1, min(chunk_size, total_episodes))

    return chunk_size


def get_available_ram_mb() -> int:
    """Get available system RAM in megabytes.

    Returns:
        Available RAM in MB, or 0 if psutil unavailable
    """
    try:
        import psutil

        return int(psutil.virtual_memory().available / (1024**2))
    except ImportError:
        print(
            "[memory_utils] WARNING: psutil not installed, cannot detect available RAM"
        )
        return 0
