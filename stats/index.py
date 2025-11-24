"""Dataset index for accessing validation data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class EpisodeInfo:
    """Information about a single episode in the dataset."""

    episode_id: int
    shard_id: int
    num_frames: int


class ValidationDatasetIndex:
    """Index for accessing episodes in the validation dataset.

    Provides efficient access to episode data stored in Zarr format.
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize the dataset index.

        Args:
            data_dir: Path to the validation dataset root directory.
        """
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "meta.json"
        index_path = self.data_dir / "index.jsonl"

        if not meta_path.exists() or not index_path.exists():
            raise FileNotFoundError(f"Expected meta.json and index.jsonl in {data_dir}")

        with meta_path.open("r") as f:
            meta = json.load(f)

        schema = meta.get("schema", {})
        self.feature_names: List[str] = list(schema.get("features", []))
        self.target_names: List[str] = list(schema.get("targets", []))

        # Build feature name to index mapping
        self.feature_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.feature_names)
        }

        episodes: List[EpisodeInfo] = []
        with index_path.open("r") as f:
            for line in f:
                row = json.loads(line)
                episodes.append(
                    EpisodeInfo(
                        episode_id=int(row["episode_id"]),
                        shard_id=int(row["shard_id"]),
                        num_frames=int(row.get("frames", 0)),
                    )
                )
        self.episodes = episodes

        # Cache for opened shard groups
        self._shard_cache: Dict[int, object] = {}

    @property
    def num_episodes(self) -> int:
        """Total number of episodes in the dataset."""
        return len(self.episodes)

    @property
    def num_features(self) -> int:
        """Number of features per frame."""
        return len(self.feature_names)

    @property
    def total_frames(self) -> int:
        """Total number of frames across all episodes."""
        return sum(ep.num_frames for ep in self.episodes)

    def get_feature_idx(self, name: str) -> int:
        """Get the column index for a feature name.

        Args:
            name: Feature name.

        Returns:
            Column index.

        Raises:
            KeyError: If feature name not found.
        """
        return self.feature_to_idx[name]

    def open_episode_arrays(
        self, episode: EpisodeInfo
    ) -> Tuple["np.ndarray", Optional["np.ndarray"]]:
        """Open the X and Y arrays for an episode.

        Args:
            episode: Episode info.

        Returns:
            Tuple of (X array, Y array or None).
        """
        import zarr

        shard_group = self._shard_cache.get(episode.shard_id)
        if shard_group is None:
            shard_path = self.data_dir / f"shard_{episode.shard_id:05d}.zarr"
            shard_group = zarr.open_group(str(shard_path), mode="r")
            self._shard_cache[episode.shard_id] = shard_group

        ep_group = shard_group[f"ep_{episode.episode_id:06d}"]
        X = ep_group["X"]
        Y = ep_group.get("Y")
        return X, Y

    def load_episode_data(self, episode: EpisodeInfo) -> np.ndarray:
        """Load full episode data as numpy array.

        Args:
            episode: Episode info.

        Returns:
            Numpy array of shape (num_frames, num_features).
        """
        X, _ = self.open_episode_arrays(episode)
        return np.asarray(X[:])

    def iter_episodes(self, shard_id: Optional[int] = None):
        """Iterate over episodes, optionally filtered by shard.

        Args:
            shard_id: If provided, only yield episodes from this shard.

        Yields:
            EpisodeInfo objects.
        """
        for episode in self.episodes:
            if shard_id is None or episode.shard_id == shard_id:
                yield episode

    def get_unique_shards(self) -> List[int]:
        """Get list of unique shard IDs in the dataset."""
        return sorted(set(ep.shard_id for ep in self.episodes))

    def summary(self) -> Dict[str, object]:
        """Get summary statistics about the dataset.

        Returns:
            Dictionary with summary information.
        """
        return {
            "num_episodes": self.num_episodes,
            "num_features": self.num_features,
            "total_frames": self.total_frames,
            "num_shards": len(self.get_unique_shards()),
            "feature_names": self.feature_names,
            "target_names": self.target_names,
        }
