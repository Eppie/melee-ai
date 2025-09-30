"""
Sampling strategies for dataset creation.

This module provides different sampling strategies for creating training
datasets from replay data.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import Iterator, List, Optional

import torch

from melee_ai.config import Settings
from melee_ai.utils import guard_clause


def _generate_episode_permutation(
    num_episodes: int,
    epoch: int,
    generator: Optional[torch.Generator] = None
) -> List[int]:
    """Generate a shuffled permutation of episode indices."""
    guard_clause(num_episodes > 0, "Number of episodes must be positive")

    g = generator or torch.Generator()
    seed = (epoch * 0x9E3779B97F4A7C15 + 4242) % (2**63 - 1)
    g.manual_seed(seed)

    return torch.randperm(num_episodes, generator=g).tolist()


def _yield_window_indices(
    episodes: List[int],
    episode_lengths: List[int],
    block_size: int,
    max_windows: Optional[int] = None
) -> Iterator[int]:
    """Yield window indices from episodes."""
    total_windows = 0

    for episode_idx in episodes:
        episode_length = episode_lengths[episode_idx]
        max_start = max(0, episode_length - block_size)

        if max_start <= 0:
            continue

        # Yield all windows in this episode
        for start_idx in range(max_start + 1):
            if max_windows and total_windows >= max_windows:
                return
            yield start_idx
            total_windows += 1


class Sampler(ABC):
    """Base class for sampling strategies."""

    def __init__(self, settings: Settings):
        self.settings = settings

    @abstractmethod
    def sample_indices(self, total_frames: int) -> List[int]:
        """Sample frame indices for training."""
        pass


class EpisodeLinearSampler(Sampler):
    """Sampler that processes episodes sequentially."""

    def __init__(self, settings: Settings, episodes_per_epoch: Optional[int] = None):
        super().__init__(settings)
        self.episodes_per_epoch = episodes_per_epoch or settings.training.episodes_per_epoch

    def sample_indices(self, total_frames: int) -> List[int]:
        """Sample indices using episode-linear strategy."""
        # For now, use a simple approach
        # In practice, this would identify episode boundaries
        if self.episodes_per_epoch:
            # Sample specific number of episodes
            num_samples = min(self.episodes_per_epoch, total_frames)
            return list(range(num_samples))
        else:
            # Sample all frames
            return list(range(total_frames))


class RandomWindowSampler(Sampler):
    """Sampler that creates random windows of data."""

    def __init__(self, settings: Settings, replacement: bool = False, num_samples: Optional[int] = None):
        super().__init__(settings)
        self.replacement = replacement or settings.training.replacement
        self.num_samples = num_samples or settings.training.num_samples

    def sample_indices(self, total_frames: int) -> List[int]:
        """Sample indices using random window strategy."""
        guard_clause(total_frames > 0, "Total frames must be positive")

        if self.num_samples is None:
            # Default to reasonable number based on block size
            block_size = self.settings.model.block_size
            self.num_samples = min(total_frames // block_size, 10000)

        max_start = total_frames - self.settings.model.block_size

        if self.replacement:
            # Sample with replacement
            return [random.randint(0, max_start) for _ in range(self.num_samples)]
        else:
            # Sample without replacement
            guard_clause(max_start >= 0, "Not enough frames for window sampling")

            population = list(range(max_start + 1))
            return random.sample(population, min(len(population), self.num_samples))


def create_sampler(settings: Settings) -> Sampler:
    """Factory function to create appropriate sampler."""
    guard_clause(settings.training.mode in ["episode_linear", "random_windows"],
                 f"Unknown sampling mode: {settings.training.mode}")

    if settings.training.mode == "episode_linear":
        return EpisodeLinearSampler(settings)
    else:  # random_windows
        return RandomWindowSampler(settings)
