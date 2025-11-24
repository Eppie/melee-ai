"""Data quality validation collector."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence, Set

import numpy as np

from stats.collectors.base import StatsCollector
from stats.index import EpisodeInfo
from stats.utils.melee_constants import GAME_LIMITS


class DataQualityCollector(StatsCollector):
    """Collector for data quality metrics and validation."""

    name = "data_quality"

    def __init__(self, feature_names: Sequence[str]) -> None:
        super().__init__(feature_names)

        # Out-of-range value counts
        self._out_of_range: Dict[str, int] = {}
        self._out_of_range_examples: Dict[str, List[float]] = {}

        # NaN/Inf counts
        self._nan_counts: Dict[str, int] = {}
        self._inf_counts: Dict[str, int] = {}

        # Consistency checks
        self._stock_inconsistencies = 0  # Stock increased unexpectedly
        self._percent_anomalies = 0  # Percent decreased without stock change

        # Feature coverage (which features have non-default values)
        self._non_default_counts: Dict[str, int] = {}

        # Episode-level issues
        self._empty_episodes = 0
        self._very_short_episodes = 0  # < 60 frames
        self._very_long_episodes = 0  # > 28800 frames (8 minutes)

        # Suspicious patterns
        self._constant_position_frames = 0
        self._teleport_frames = 0  # Position changed > 50 units in one frame

    def process_episode(self, data: np.ndarray, episode: EpisodeInfo) -> None:
        num_frames = data.shape[0]

        # Check episode length
        if num_frames == 0:
            self._empty_episodes += 1
            self._record_episode(num_frames)
            return
        if num_frames < 60:
            self._very_short_episodes += 1
        if num_frames > 28800:
            self._very_long_episodes += 1

        # Check each feature
        for col_name in self.feature_names:
            col_idx = self.get_feature_idx(col_name)
            col_data = data[:, col_idx]

            # Initialize counters
            if col_name not in self._out_of_range:
                self._out_of_range[col_name] = 0
                self._out_of_range_examples[col_name] = []
                self._nan_counts[col_name] = 0
                self._inf_counts[col_name] = 0
                self._non_default_counts[col_name] = 0

            # Check for NaN/Inf
            nan_mask = np.isnan(col_data)
            inf_mask = np.isinf(col_data)
            self._nan_counts[col_name] += int(nan_mask.sum())
            self._inf_counts[col_name] += int(inf_mask.sum())

            # Check range limits
            base_name = col_name.replace("p1_", "").replace("p2_", "")
            if base_name in GAME_LIMITS:
                limits = GAME_LIMITS[base_name]
                valid_data = col_data[~nan_mask & ~inf_mask]
                out_of_range = (valid_data < limits["min"]) | (
                    valid_data > limits["max"]
                )
                count = int(out_of_range.sum())
                self._out_of_range[col_name] += count

                # Store examples
                if count > 0 and len(self._out_of_range_examples[col_name]) < 10:
                    examples = valid_data[out_of_range][:5].tolist()
                    self._out_of_range_examples[col_name].extend(examples)

            # Track non-default values
            if "button" in col_name or "facing" in col_name:
                # Boolean-ish: count non-zero
                self._non_default_counts[col_name] += int((col_data != 0).sum())
            elif "stick" in col_name:
                # Sticks: count non-centered (not 0.5)
                self._non_default_counts[col_name] += int(
                    (np.abs(col_data - 0.5) > 0.1).sum()
                )
            else:
                # Other: just count non-zero
                self._non_default_counts[col_name] += int((col_data != 0).sum())

        # Consistency checks
        self._check_consistency(data)

        # Suspicious patterns
        self._check_suspicious_patterns(data)

        self._record_episode(num_frames)

    def _check_consistency(self, data: np.ndarray) -> None:
        """Check for logical inconsistencies in the data."""
        try:
            p1_stock_idx = self.get_feature_idx("p1_stock")
            p2_stock_idx = self.get_feature_idx("p2_stock")
            p1_percent_idx = self.get_feature_idx("p1_percent")
            p2_percent_idx = self.get_feature_idx("p2_percent")
        except KeyError:
            return

        p1_stock = data[:, p1_stock_idx]
        p2_stock = data[:, p2_stock_idx]
        p1_percent = data[:, p1_percent_idx]
        p2_percent = data[:, p2_percent_idx]

        # Stock should never increase mid-game
        for i in range(1, len(p1_stock)):
            if p1_stock[i] > p1_stock[i - 1]:
                self._stock_inconsistencies += 1
            if p2_stock[i] > p2_stock[i - 1]:
                self._stock_inconsistencies += 1

        # Percent decreasing without stock change is suspicious
        # (can happen legitimately with healing items, but rare in competitive)
        for i in range(1, len(p1_percent)):
            if p1_percent[i] < p1_percent[i - 1] - 1 and p1_stock[i] == p1_stock[i - 1]:
                self._percent_anomalies += 1
            if p2_percent[i] < p2_percent[i - 1] - 1 and p2_stock[i] == p2_stock[i - 1]:
                self._percent_anomalies += 1

    def _check_suspicious_patterns(self, data: np.ndarray) -> None:
        """Check for suspicious patterns in position data."""
        try:
            p1_x_idx = self.get_feature_idx("p1_position_x")
            p1_y_idx = self.get_feature_idx("p1_position_y")
        except KeyError:
            return

        p1_x = data[:, p1_x_idx]
        p1_y = data[:, p1_y_idx]

        # Check for constant position
        x_diff = np.diff(p1_x)
        y_diff = np.diff(p1_y)
        constant = (x_diff == 0) & (y_diff == 0)
        self._constant_position_frames += int(constant.sum())

        # Check for teleports (position changed > 50 units)
        distance = np.sqrt(x_diff**2 + y_diff**2)
        teleports = distance > 50
        self._teleport_frames += int(teleports.sum())

    def merge(self, other: "DataQualityCollector") -> None:
        for col_name in self.feature_names:
            self._out_of_range[col_name] = self._out_of_range.get(
                col_name, 0
            ) + other._out_of_range.get(col_name, 0)
            self._nan_counts[col_name] = self._nan_counts.get(
                col_name, 0
            ) + other._nan_counts.get(col_name, 0)
            self._inf_counts[col_name] = self._inf_counts.get(
                col_name, 0
            ) + other._inf_counts.get(col_name, 0)
            self._non_default_counts[col_name] = self._non_default_counts.get(
                col_name, 0
            ) + other._non_default_counts.get(col_name, 0)

            if col_name not in self._out_of_range_examples:
                self._out_of_range_examples[col_name] = []
            self._out_of_range_examples[col_name].extend(
                other._out_of_range_examples.get(col_name, [])[:10]
            )
            self._out_of_range_examples[col_name] = self._out_of_range_examples[
                col_name
            ][:10]

        self._empty_episodes += other._empty_episodes
        self._very_short_episodes += other._very_short_episodes
        self._very_long_episodes += other._very_long_episodes
        self._stock_inconsistencies += other._stock_inconsistencies
        self._percent_anomalies += other._percent_anomalies
        self._constant_position_frames += other._constant_position_frames
        self._teleport_frames += other._teleport_frames

        self._episodes_processed += other._episodes_processed
        self._frames_processed += other._frames_processed

    def finalize(self) -> Dict[str, Any]:
        total_frames = self._frames_processed

        # Range violations
        range_violations = {}
        for col_name in self.feature_names:
            count = self._out_of_range.get(col_name, 0)
            if count > 0:
                range_violations[col_name] = {
                    "count": count,
                    "percent": count / total_frames * 100 if total_frames else 0,
                    "examples": self._out_of_range_examples.get(col_name, []),
                }

        # NaN/Inf issues
        nan_inf_issues = {}
        for col_name in self.feature_names:
            nan_count = self._nan_counts.get(col_name, 0)
            inf_count = self._inf_counts.get(col_name, 0)
            if nan_count > 0 or inf_count > 0:
                nan_inf_issues[col_name] = {
                    "nan_count": nan_count,
                    "inf_count": inf_count,
                    "nan_percent": nan_count / total_frames * 100
                    if total_frames
                    else 0,
                }

        # Feature utilization
        feature_utilization = {}
        for col_name in self.feature_names:
            non_default = self._non_default_counts.get(col_name, 0)
            feature_utilization[col_name] = {
                "non_default_count": non_default,
                "non_default_percent": non_default / total_frames * 100
                if total_frames
                else 0,
            }

        return {
            "summary": {
                "total_episodes": self._episodes_processed,
                "total_frames": total_frames,
                "empty_episodes": self._empty_episodes,
                "very_short_episodes": self._very_short_episodes,
                "very_long_episodes": self._very_long_episodes,
                "stock_inconsistencies": self._stock_inconsistencies,
                "percent_anomalies": self._percent_anomalies,
                "constant_position_frames": self._constant_position_frames,
                "teleport_frames": self._teleport_frames,
            },
            "range_violations": range_violations,
            "nan_inf_issues": nan_inf_issues,
            "feature_utilization": feature_utilization,
            "data_quality_score": self._compute_quality_score(total_frames),
        }

    def _compute_quality_score(self, total_frames: int) -> float:
        """Compute an overall data quality score (0-100)."""
        if total_frames == 0:
            return 0.0

        score = 100.0

        # Penalize range violations
        total_violations = sum(self._out_of_range.values())
        violation_rate = total_violations / (total_frames * len(self.feature_names))
        score -= min(20, violation_rate * 1000)

        # Penalize NaN/Inf
        total_nan_inf = sum(self._nan_counts.values()) + sum(self._inf_counts.values())
        nan_inf_rate = total_nan_inf / (total_frames * len(self.feature_names))
        score -= min(20, nan_inf_rate * 1000)

        # Penalize inconsistencies
        inconsistency_rate = (
            self._stock_inconsistencies + self._percent_anomalies
        ) / total_frames
        score -= min(20, inconsistency_rate * 100)

        # Penalize empty/short episodes
        episode_issue_rate = (self._empty_episodes + self._very_short_episodes) / max(
            1, self._episodes_processed
        )
        score -= min(20, episode_issue_rate * 100)

        # Penalize suspicious patterns
        suspicious_rate = self._teleport_frames / total_frames
        score -= min(20, suspicious_rate * 100)

        return max(0.0, score)
