"""Cross-feature correlation and analysis collector."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from stats.collectors.base import StatsCollector
from stats.index import EpisodeInfo


class CrossFeatureCollector(StatsCollector):
    """Collector for cross-feature correlations and joint distributions."""

    name = "cross_features"

    # Features to compute correlations for
    CORRELATION_FEATURES = [
        "p1_position_x",
        "p1_position_y",
        "p1_percent",
        "p2_position_x",
        "p2_position_y",
        "p2_percent",
    ]

    def __init__(self, feature_names: Sequence[str]) -> None:
        super().__init__(feature_names)

        # For streaming correlation computation
        self._n = 0
        self._means: Dict[str, float] = {}
        self._m2: Dict[str, float] = {}  # For variance
        self._cross_m2: Dict[Tuple[str, str], float] = {}  # For covariance

        # Position joint distribution (2D histogram)
        self._position_hist = np.zeros(
            (40, 40), dtype=np.int64
        )  # x: -200 to 200, y: -100 to 300

        # Action by position (which positions for attacks, movement, etc.)
        self._p1_attack_positions: List[Tuple[float, float]] = []
        self._p2_attack_positions: List[Tuple[float, float]] = []

    def process_episode(self, data: np.ndarray, episode: EpisodeInfo) -> None:
        num_frames = data.shape[0]

        # Get indices for correlation features
        indices = {}
        for name in self.CORRELATION_FEATURES:
            try:
                indices[name] = self.get_feature_idx(name)
            except KeyError:
                pass

        # Update correlation statistics using Welford's algorithm
        for frame_idx in range(num_frames):
            self._n += 1
            for name, col_idx in indices.items():
                val = float(data[frame_idx, col_idx])

                if name not in self._means:
                    self._means[name] = 0.0
                    self._m2[name] = 0.0

                delta = val - self._means[name]
                self._means[name] += delta / self._n
                delta2 = val - self._means[name]
                self._m2[name] += delta * delta2

        # Update cross terms (sample covariance) - simplified
        # For efficiency, we update covariance less frequently
        if num_frames > 0 and len(indices) >= 2:
            feature_data = {name: data[:, idx] for name, idx in indices.items()}
            for i, name1 in enumerate(indices.keys()):
                for name2 in list(indices.keys())[i + 1 :]:
                    key = (name1, name2)
                    if key not in self._cross_m2:
                        self._cross_m2[key] = 0.0

                    # Batch update for covariance
                    vals1 = feature_data[name1]
                    vals2 = feature_data[name2]
                    cov_contribution = np.sum(
                        (vals1 - self._means.get(name1, 0))
                        * (vals2 - self._means.get(name2, 0))
                    )
                    self._cross_m2[key] += cov_contribution

        # Position histogram
        try:
            p1_x_idx = self.get_feature_idx("p1_position_x")
            p1_y_idx = self.get_feature_idx("p1_position_y")

            p1_x = data[:, p1_x_idx]
            p1_y = data[:, p1_y_idx]

            # Bin positions: x from -200 to 200, y from -100 to 300
            x_bins = np.clip(((p1_x + 200) / 10).astype(int), 0, 39)
            y_bins = np.clip(((p1_y + 100) / 10).astype(int), 0, 39)

            for x, y in zip(x_bins, y_bins):
                self._position_hist[y, x] += 1
        except KeyError:
            pass

        self._record_episode(num_frames)

    def merge(self, other: "CrossFeatureCollector") -> None:
        # Merge Welford statistics (approximate)
        if other._n > 0:
            total_n = self._n + other._n
            for name in set(self._means.keys()) | set(other._means.keys()):
                m1 = self._means.get(name, 0)
                m2 = other._means.get(name, 0)
                n1 = self._n
                n2 = other._n

                delta = m2 - m1
                new_mean = (n1 * m1 + n2 * m2) / total_n if total_n > 0 else 0

                v1 = self._m2.get(name, 0)
                v2 = other._m2.get(name, 0)
                new_m2 = v1 + v2 + delta**2 * n1 * n2 / total_n if total_n > 0 else 0

                self._means[name] = new_mean
                self._m2[name] = new_m2

            for key in set(self._cross_m2.keys()) | set(other._cross_m2.keys()):
                self._cross_m2[key] = self._cross_m2.get(key, 0) + other._cross_m2.get(
                    key, 0
                )

            self._n = total_n

        self._position_hist += other._position_hist

        self._p1_attack_positions.extend(other._p1_attack_positions)
        self._p2_attack_positions.extend(other._p2_attack_positions)

        self._episodes_processed += other._episodes_processed
        self._frames_processed += other._frames_processed

    def finalize(self) -> Dict[str, Any]:
        # Compute correlation matrix
        correlations = {}
        if self._n > 1:
            for (name1, name2), cross_m2 in self._cross_m2.items():
                var1 = self._m2.get(name1, 0) / (self._n - 1) if self._n > 1 else 0
                var2 = self._m2.get(name2, 0) / (self._n - 1) if self._n > 1 else 0
                cov = cross_m2 / (self._n - 1) if self._n > 1 else 0

                if var1 > 0 and var2 > 0:
                    corr = cov / (np.sqrt(var1) * np.sqrt(var2))
                    correlations[f"{name1}_vs_{name2}"] = float(np.clip(corr, -1, 1))

        return {
            "correlations": correlations,
            "position_heatmap": self._position_hist.tolist(),
            "position_heatmap_info": {
                "x_range": [-200, 200],
                "y_range": [-100, 300],
                "bin_size": 10,
            },
            "total_samples": self._n,
        }
