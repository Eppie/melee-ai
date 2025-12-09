"""PyTorch Dataset for loading preprocessed .npz data."""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional

from loguru import logger

from ..constants import (
    NPZ_FEATURES_KEY,
    NPZ_FUTURE_DELTAS_KEY,
    NPZ_VALID_MASK_KEY,
    CONTEXT_LENGTH,
    HORIZONS, # Added HORIZONS for max_horizon calculation
)
from .build_dataset import load_index # Added load_index


class NPZDataset(Dataset):
    """PyTorch Dataset for future position prediction.

    Loads preprocessed .npz files containing features and ground truth deltas.
    """

    def __init__(
        self,
        data_dir: Path,
        context_length: int = CONTEXT_LENGTH,
        cache_episodes: int = 100,
        max_episodes: Optional[int] = None,
    ):
        """Initialize dataset.

        Args:
            data_dir: Directory containing .npz episode files and index.json
            context_length: Number of past frames to use as context
            cache_episodes: Number of hot episodes to cache in memory
            max_episodes: If set, limit to the first N episodes from index.json

        Notes:
            - Loads index.json to enumerate all episodes
            - Pre-computes valid sample indices (frame, episode) tuples
            - Caches frequently accessed episodes
        """
        self.data_dir: Path = data_dir
        self.context_length: int = context_length
        self.cache_episodes: int = cache_episodes

        self.episodes: List[dict] = load_index(data_dir / 'index.json')
        if max_episodes is not None:
            original_count = len(self.episodes)
            self.episodes = self.episodes[:max_episodes]
            logger.info(f"Limiting dataset to first {len(self.episodes)} episodes (of {original_count} available).")
        self.samples: List[Tuple[int, int]] = []  # (episode_idx, frame_idx)
        self.cache: dict = {}

        max_horizon = max(HORIZONS)

        for ep_idx, ep_meta in enumerate(self.episodes):
            num_frames = ep_meta['num_frames']
            for frame_idx in range(context_length, num_frames - max_horizon):
                self.samples.append((ep_idx, frame_idx))
        
        # Build cache for hot episodes
        self._build_cache(cache_episodes)


    def __len__(self) -> int:
        """Return number of valid samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single training sample.

        Args:
            idx: Sample index

        Returns:
            context: [context_length, feature_dim] tensor
            targets: [n_horizons, 4] position deltas
            valid: [n_horizons] bool mask

        Notes:
            - Loads episode from cache or disk
            - Extracts context window and target deltas
            - Returns as torch tensors
        """
        ep_idx, frame_idx = self.samples[idx]
        episode_data = self._load_episode(ep_idx)

        context = episode_data[NPZ_FEATURES_KEY][frame_idx - self.context_length : frame_idx]
        targets = episode_data[NPZ_FUTURE_DELTAS_KEY][frame_idx]
        valid = episode_data[NPZ_VALID_MASK_KEY][frame_idx]

        return (
            torch.from_numpy(context).float(),
            torch.from_numpy(targets).float(),
            torch.from_numpy(valid).bool(),
        )

    def _load_episode(self, episode_idx: int) -> dict:
        """Load episode from cache or disk.

        Args:
            episode_idx: Index of episode in self.episodes

        Returns:
            Dict with keys: features, future_deltas, valid_mask
        """
        if episode_idx in self.cache:
            return self.cache[episode_idx]

        ep_meta = self.episodes[episode_idx]
        npz_path = Path(ep_meta['path'])
        
        # Load the NPZ file
        loaded_data = np.load(npz_path)
        episode_data = {
            NPZ_FEATURES_KEY: loaded_data[NPZ_FEATURES_KEY],
            NPZ_FUTURE_DELTAS_KEY: loaded_data[NPZ_FUTURE_DELTAS_KEY],
            NPZ_VALID_MASK_KEY: loaded_data[NPZ_VALID_MASK_KEY],
        }
        loaded_data.close() # Close the NPZ file handle

        # Only cache if cache_episodes is set and not too many episodes are cached
        if self.cache_episodes > 0 and len(self.cache) < self.cache_episodes:
            self.cache[episode_idx] = episode_data
            
        return episode_data

    def _build_cache(self, top_n: int) -> None:
        """Pre-load top N most-sampled episodes into cache.

        Args:
            top_n: Number of episodes to cache
        """
        if top_n <= 0:
            return

        # Count samples per episode
        episode_sample_counts = {}
        for ep_idx, _ in self.samples:
            episode_sample_counts[ep_idx] = episode_sample_counts.get(ep_idx, 0) + 1

        # Sort episodes by sample count (descending)
        sorted_episodes = sorted(episode_sample_counts.items(), key=lambda item: item[1], reverse=True)

        # Load top N episodes into cache
        for ep_idx, _ in sorted_episodes[:top_n]:
            self._load_episode(ep_idx)  # Calling _load_episode will automatically cache it
