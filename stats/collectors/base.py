"""Base class for statistics collectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence

import numpy as np

from stats.index import EpisodeInfo


class StatsCollector(ABC):
    """Abstract base class for statistics collectors.

    Collectors process episode data and accumulate statistics that can be
    merged across parallel workers and finalized into output.
    """

    # Human-readable name for this collector
    name: str = "base"

    # Whether this collector requires the full episode data at once
    # (vs being able to process in chunks)
    requires_full_episode: bool = False

    def __init__(self, feature_names: Sequence[str]) -> None:
        """Initialize the collector.

        Args:
            feature_names: List of feature names in column order.
        """
        self.feature_names = list(feature_names)
        self.feature_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(feature_names)
        }
        self._episodes_processed = 0
        self._frames_processed = 0

    def get_feature_idx(self, name: str) -> int:
        """Get column index for a feature name."""
        return self.feature_to_idx[name]

    def get_feature_indices(self, names: Sequence[str]) -> List[int]:
        """Get column indices for multiple feature names."""
        return [self.feature_to_idx[name] for name in names]

    @abstractmethod
    def process_episode(self, data: np.ndarray, episode: EpisodeInfo) -> None:
        """Process data from a single episode.

        Args:
            data: Episode data array of shape (num_frames, num_features).
            episode: Episode metadata.
        """
        pass

    @abstractmethod
    def merge(self, other: "StatsCollector") -> None:
        """Merge statistics from another collector instance.

        Used for combining results from parallel workers.

        Args:
            other: Another collector of the same type.
        """
        pass

    @abstractmethod
    def finalize(self) -> Dict[str, Any]:
        """Finalize and return statistics.

        Called after all episodes have been processed and merged.

        Returns:
            Dictionary of computed statistics.
        """
        pass

    @property
    def episodes_processed(self) -> int:
        """Number of episodes processed by this collector."""
        return self._episodes_processed

    @property
    def frames_processed(self) -> int:
        """Number of frames processed by this collector."""
        return self._frames_processed

    def _record_episode(self, num_frames: int) -> None:
        """Record that an episode was processed.

        Args:
            num_frames: Number of frames in the episode.
        """
        self._episodes_processed += 1
        self._frames_processed += num_frames


class CompositeCollector(StatsCollector):
    """A collector that combines multiple sub-collectors.

    Useful for running multiple collectors in a single pass over the data.
    """

    name = "composite"

    def __init__(
        self,
        feature_names: Sequence[str],
        collectors: Sequence[StatsCollector],
    ) -> None:
        """Initialize the composite collector.

        Args:
            feature_names: List of feature names.
            collectors: Sub-collectors to run.
        """
        super().__init__(feature_names)
        self.collectors = list(collectors)

    def process_episode(self, data: np.ndarray, episode: EpisodeInfo) -> None:
        """Process episode with all sub-collectors."""
        for collector in self.collectors:
            collector.process_episode(data, episode)
        self._record_episode(data.shape[0])

    def merge(self, other: "CompositeCollector") -> None:
        """Merge all sub-collectors."""
        for my_collector, other_collector in zip(self.collectors, other.collectors):
            my_collector.merge(other_collector)
        self._episodes_processed += other._episodes_processed
        self._frames_processed += other._frames_processed

    def finalize(self) -> Dict[str, Any]:
        """Finalize all sub-collectors."""
        results = {}
        for collector in self.collectors:
            results[collector.name] = collector.finalize()
        return results
