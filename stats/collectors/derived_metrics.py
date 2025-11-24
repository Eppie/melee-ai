"""Derived metrics collector (distance, combos, advantage states)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence

import numpy as np

from stats.collectors.base import StatsCollector
from stats.index import EpisodeInfo
from stats.utils.melee_constants import FRAMES_PER_SECOND


class DerivedMetricsCollector(StatsCollector):
    """Collector for derived metrics like distance, combos, and advantage."""

    name = "derived_metrics"

    def __init__(self, feature_names: Sequence[str]) -> None:
        super().__init__(feature_names)

        # Distance statistics
        self._distances: List[float] = []
        self._distance_sum = 0.0
        self._distance_count = 0

        # Combo detection (consecutive hitstun frames)
        self._p1_combo_lengths: List[int] = []
        self._p2_combo_lengths: List[int] = []

        # Advantage state tracking (who has more stocks, lower percent)
        self._p1_advantage_frames = 0
        self._p2_advantage_frames = 0
        self._neutral_frames = 0

        # Off-stage time
        self._p1_offstage_frames = 0
        self._p2_offstage_frames = 0

        # Edgeguard situations (opponent offstage while you're on)
        self._p1_edgeguard_frames = 0
        self._p2_edgeguard_frames = 0

        # Kill percent tracking
        self._p1_kill_percents: List[float] = []
        self._p2_kill_percents: List[float] = []

    def process_episode(self, data: np.ndarray, episode: EpisodeInfo) -> None:
        num_frames = data.shape[0]

        try:
            p1_x_idx = self.get_feature_idx("p1_position_x")
            p1_y_idx = self.get_feature_idx("p1_position_y")
            p2_x_idx = self.get_feature_idx("p2_position_x")
            p2_y_idx = self.get_feature_idx("p2_position_y")
            p1_percent_idx = self.get_feature_idx("p1_percent")
            p2_percent_idx = self.get_feature_idx("p2_percent")
            p1_stock_idx = self.get_feature_idx("p1_stock")
            p2_stock_idx = self.get_feature_idx("p2_stock")
            p1_hitstun_idx = self.get_feature_idx("p1_is_in_hitstun")
            p2_hitstun_idx = self.get_feature_idx("p2_is_in_hitstun")
            p1_offstage_idx = self.get_feature_idx("p1_off_stage")
            p2_offstage_idx = self.get_feature_idx("p2_off_stage")
        except KeyError:
            self._record_episode(num_frames)
            return

        # Extract data
        p1_x = data[:, p1_x_idx]
        p1_y = data[:, p1_y_idx]
        p2_x = data[:, p2_x_idx]
        p2_y = data[:, p2_y_idx]
        p1_percent = data[:, p1_percent_idx]
        p2_percent = data[:, p2_percent_idx]
        p1_stock = data[:, p1_stock_idx]
        p2_stock = data[:, p2_stock_idx]
        p1_hitstun = data[:, p1_hitstun_idx] > 0.5
        p2_hitstun = data[:, p2_hitstun_idx] > 0.5
        p1_offstage = data[:, p1_offstage_idx] > 0.5
        p2_offstage = data[:, p2_offstage_idx] > 0.5

        # Distance calculation
        distances = np.sqrt((p1_x - p2_x)**2 + (p1_y - p2_y)**2)
        self._distance_sum += float(distances.sum())
        self._distance_count += len(distances)

        # Sample distances for distribution
        sample_idx = np.linspace(0, num_frames - 1, min(100, num_frames)).astype(int)
        self._distances.extend(distances[sample_idx].tolist())

        # Combo detection (consecutive hitstun frames)
        self._detect_combos(p1_hitstun, self._p1_combo_lengths)
        self._detect_combos(p2_hitstun, self._p2_combo_lengths)

        # Advantage tracking
        for i in range(num_frames):
            p1_adv_score = (p1_stock[i] - p2_stock[i]) * 100 + (p2_percent[i] - p1_percent[i])
            if p1_adv_score > 50:
                self._p1_advantage_frames += 1
            elif p1_adv_score < -50:
                self._p2_advantage_frames += 1
            else:
                self._neutral_frames += 1

        # Offstage tracking
        self._p1_offstage_frames += int(p1_offstage.sum())
        self._p2_offstage_frames += int(p2_offstage.sum())

        # Edgeguard situations
        p1_on_p2_off = (~p1_offstage) & p2_offstage
        p2_on_p1_off = (~p2_offstage) & p1_offstage
        self._p1_edgeguard_frames += int(p1_on_p2_off.sum())
        self._p2_edgeguard_frames += int(p2_on_p1_off.sum())

        # Kill percent tracking (percent when stock decreases)
        for i in range(1, num_frames):
            if p1_stock[i] < p1_stock[i-1]:
                self._p1_kill_percents.append(float(p1_percent[i-1]))
            if p2_stock[i] < p2_stock[i-1]:
                self._p2_kill_percents.append(float(p2_percent[i-1]))

        self._record_episode(num_frames)

    def _detect_combos(self, hitstun: np.ndarray, combo_lengths: List[int]) -> None:
        """Detect combo sequences from hitstun data."""
        in_combo = False
        combo_length = 0

        for in_hitstun in hitstun:
            if in_hitstun and not in_combo:
                in_combo = True
                combo_length = 1
            elif in_hitstun and in_combo:
                combo_length += 1
            elif not in_hitstun and in_combo:
                if combo_length >= FRAMES_PER_SECOND // 4:  # At least 0.25s
                    combo_lengths.append(combo_length)
                in_combo = False
                combo_length = 0

        if in_combo and combo_length >= FRAMES_PER_SECOND // 4:
            combo_lengths.append(combo_length)

    def merge(self, other: "DerivedMetricsCollector") -> None:
        self._distances.extend(other._distances)
        self._distance_sum += other._distance_sum
        self._distance_count += other._distance_count

        self._p1_combo_lengths.extend(other._p1_combo_lengths)
        self._p2_combo_lengths.extend(other._p2_combo_lengths)

        self._p1_advantage_frames += other._p1_advantage_frames
        self._p2_advantage_frames += other._p2_advantage_frames
        self._neutral_frames += other._neutral_frames

        self._p1_offstage_frames += other._p1_offstage_frames
        self._p2_offstage_frames += other._p2_offstage_frames
        self._p1_edgeguard_frames += other._p1_edgeguard_frames
        self._p2_edgeguard_frames += other._p2_edgeguard_frames

        self._p1_kill_percents.extend(other._p1_kill_percents)
        self._p2_kill_percents.extend(other._p2_kill_percents)

        self._episodes_processed += other._episodes_processed
        self._frames_processed += other._frames_processed

    def finalize(self) -> Dict[str, Any]:
        total_frames = self._p1_advantage_frames + self._p2_advantage_frames + self._neutral_frames

        # Distance stats
        dist_stats = {}
        if self._distances:
            arr = np.array(self._distances)
            dist_stats = {
                "mean": self._distance_sum / self._distance_count if self._distance_count else 0,
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "percentiles": {
                    str(p): float(np.percentile(arr, p))
                    for p in [10, 25, 50, 75, 90]
                },
            }

        # Combo stats
        def combo_stats(lengths: List[int]) -> Dict:
            if not lengths:
                return {"count": 0}
            arr = np.array(lengths)
            return {
                "count": len(lengths),
                "total_frames": int(arr.sum()),
                "mean_length_frames": float(arr.mean()),
                "mean_length_seconds": float(arr.mean() / FRAMES_PER_SECOND),
                "max_length_frames": int(arr.max()),
                "max_length_seconds": float(arr.max() / FRAMES_PER_SECOND),
            }

        # Kill percent stats
        def kill_stats(percents: List[float]) -> Dict:
            if not percents:
                return {"count": 0}
            arr = np.array(percents)
            return {
                "count": len(percents),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "median": float(np.median(arr)),
            }

        return {
            "distance": dist_stats,
            "combos": {
                "p1_received": combo_stats(self._p1_combo_lengths),
                "p2_received": combo_stats(self._p2_combo_lengths),
            },
            "advantage": {
                "p1_advantage_frames": self._p1_advantage_frames,
                "p2_advantage_frames": self._p2_advantage_frames,
                "neutral_frames": self._neutral_frames,
                "p1_advantage_percent": self._p1_advantage_frames / total_frames * 100 if total_frames else 0,
                "p2_advantage_percent": self._p2_advantage_frames / total_frames * 100 if total_frames else 0,
            },
            "offstage": {
                "p1_frames": self._p1_offstage_frames,
                "p2_frames": self._p2_offstage_frames,
                "p1_percent": self._p1_offstage_frames / self._frames_processed * 100 if self._frames_processed else 0,
                "p2_percent": self._p2_offstage_frames / self._frames_processed * 100 if self._frames_processed else 0,
            },
            "edgeguards": {
                "p1_frames": self._p1_edgeguard_frames,
                "p2_frames": self._p2_edgeguard_frames,
            },
            "kill_percents": {
                "p1": kill_stats(self._p1_kill_percents),
                "p2": kill_stats(self._p2_kill_percents),
            },
        }
