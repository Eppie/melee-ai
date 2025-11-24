"""Episode-level statistics collector."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence

import numpy as np

from stats.collectors.base import StatsCollector
from stats.index import EpisodeInfo
from stats.utils.melee_constants import STAGE_NAMES

# Death action states (0-10 are various death animations)
# Game ends when a player enters these states, before stock decrements
DEATH_ACTION_STATES = frozenset(range(0, 11))


class EpisodeStatsCollector(StatsCollector):
    """Collector for episode-level statistics."""

    name = "episodes"

    def __init__(self, feature_names: Sequence[str]) -> None:
        super().__init__(feature_names)

        # Episode length tracking
        self._length_counts: Counter = Counter()
        self._total_length: int = 0

        # Stage distribution
        self._stage_counts: Counter = Counter()

        # Stock outcomes
        self._p1_stocks_lost: Counter = Counter()
        self._p2_stocks_lost: Counter = Counter()

        # Damage statistics
        self._p1_max_percent: List[float] = []
        self._p2_max_percent: List[float] = []
        self._p1_total_damage_received: List[float] = []
        self._p2_total_damage_received: List[float] = []

        # Winner tracking (based on final stock counts)
        self._p1_wins = 0
        self._p2_wins = 0
        self._draws = 0

    def process_episode(self, data: np.ndarray, episode: EpisodeInfo) -> None:
        num_frames = data.shape[0]

        # Track length
        self._length_counts[num_frames] += 1
        self._total_length += num_frames

        # Get column indices
        try:
            stage_idx = self.get_feature_idx("stage")
            p1_stock_idx = self.get_feature_idx("p1_stock")
            p2_stock_idx = self.get_feature_idx("p2_stock")
            p1_percent_idx = self.get_feature_idx("p1_percent")
            p2_percent_idx = self.get_feature_idx("p2_percent")
            p1_action_idx = self.get_feature_idx("p1_action")
            p2_action_idx = self.get_feature_idx("p2_action")
            p1_offscreen_idx = self.get_feature_idx("p1_is_offscreen")
            p2_offscreen_idx = self.get_feature_idx("p2_is_offscreen")
        except KeyError:
            self._record_episode(num_frames)
            return

        # Stage (use first frame)
        stage = int(data[0, stage_idx])
        self._stage_counts[stage] += 1

        # Stock analysis
        p1_stocks = data[:, p1_stock_idx]
        p2_stocks = data[:, p2_stock_idx]

        p1_start_stock = int(p1_stocks[0])
        p2_start_stock = int(p2_stocks[0])
        p1_end_stock = int(p1_stocks[-1])
        p2_end_stock = int(p2_stocks[-1])

        p1_lost = p1_start_stock - p1_end_stock
        p2_lost = p2_start_stock - p2_end_stock

        self._p1_stocks_lost[p1_lost] += 1
        self._p2_stocks_lost[p2_lost] += 1

        # Winner determination
        # Check action states and offscreen status - game ends when a player
        # enters death animation or goes offscreen, before stock decrements
        p1_end_action = int(data[-1, p1_action_idx])
        p2_end_action = int(data[-1, p2_action_idx])
        p1_offscreen = data[-1, p1_offscreen_idx] > 0.5
        p2_offscreen = data[-1, p2_offscreen_idx] > 0.5
        p1_dying = p1_end_action in DEATH_ACTION_STATES or p1_offscreen
        p2_dying = p2_end_action in DEATH_ACTION_STATES or p2_offscreen

        if p2_dying and not p1_dying:
            # P2 is dying/offscreen, P1 wins
            self._p1_wins += 1
        elif p1_dying and not p2_dying:
            # P1 is dying/offscreen, P2 wins
            self._p2_wins += 1
        elif p1_end_stock > p2_end_stock:
            self._p1_wins += 1
        elif p2_end_stock > p1_end_stock:
            self._p2_wins += 1
        else:
            # Both dying or equal stocks with no death - true draw/timeout
            self._draws += 1

        # Damage analysis
        p1_percent = data[:, p1_percent_idx]
        p2_percent = data[:, p2_percent_idx]

        self._p1_max_percent.append(float(p1_percent.max()))
        self._p2_max_percent.append(float(p2_percent.max()))

        # Approximate total damage received (sum of all damage increases)
        p1_damage_diff = np.diff(p1_percent)
        p2_damage_diff = np.diff(p2_percent)
        self._p1_total_damage_received.append(float(p1_damage_diff[p1_damage_diff > 0].sum()))
        self._p2_total_damage_received.append(float(p2_damage_diff[p2_damage_diff > 0].sum()))

        self._record_episode(num_frames)

    def merge(self, other: "EpisodeStatsCollector") -> None:
        self._length_counts.update(other._length_counts)
        self._total_length += other._total_length
        self._stage_counts.update(other._stage_counts)
        self._p1_stocks_lost.update(other._p1_stocks_lost)
        self._p2_stocks_lost.update(other._p2_stocks_lost)
        self._p1_max_percent.extend(other._p1_max_percent)
        self._p2_max_percent.extend(other._p2_max_percent)
        self._p1_total_damage_received.extend(other._p1_total_damage_received)
        self._p2_total_damage_received.extend(other._p2_total_damage_received)
        self._p1_wins += other._p1_wins
        self._p2_wins += other._p2_wins
        self._draws += other._draws
        self._episodes_processed += other._episodes_processed
        self._frames_processed += other._frames_processed

    def finalize(self) -> Dict[str, Any]:
        num_episodes = self._episodes_processed

        # Length statistics
        if self._length_counts:
            lengths = sorted(self._length_counts.items())
            length_stats = {
                "min": lengths[0][0],
                "max": lengths[-1][0],
                "mean": self._total_length / num_episodes if num_episodes else 0,
                "total": self._total_length,
            }
            # Percentiles
            length_arr = []
            for length, count in lengths:
                length_arr.extend([length] * count)
            if length_arr:
                arr = np.array(length_arr)
                for p in [10, 25, 50, 75, 90]:
                    length_stats[f"p{p}"] = float(np.percentile(arr, p))
        else:
            length_stats = {}

        # Stage distribution
        stages = {}
        for stage_id, count in sorted(self._stage_counts.items()):
            stage_name = STAGE_NAMES.get(stage_id, f"Unknown({stage_id})")
            stages[stage_name] = {
                "count": count,
                "percent": count / num_episodes * 100 if num_episodes else 0,
            }

        # Stock outcomes
        p1_stocks = {k: v for k, v in sorted(self._p1_stocks_lost.items())}
        p2_stocks = {k: v for k, v in sorted(self._p2_stocks_lost.items())}

        # Win rates
        total_decisive = self._p1_wins + self._p2_wins + self._draws
        outcomes = {
            "p1_wins": self._p1_wins,
            "p2_wins": self._p2_wins,
            "draws": self._draws,
            "p1_win_rate": self._p1_wins / total_decisive * 100 if total_decisive else 0,
            "p2_win_rate": self._p2_wins / total_decisive * 100 if total_decisive else 0,
        }

        # Damage statistics
        damage_stats = {}
        if self._p1_max_percent:
            p1_max = np.array(self._p1_max_percent)
            p2_max = np.array(self._p2_max_percent)
            p1_total = np.array(self._p1_total_damage_received)
            p2_total = np.array(self._p2_total_damage_received)

            damage_stats = {
                "p1_max_percent": {
                    "mean": float(p1_max.mean()),
                    "std": float(p1_max.std()),
                    "max": float(p1_max.max()),
                },
                "p2_max_percent": {
                    "mean": float(p2_max.mean()),
                    "std": float(p2_max.std()),
                    "max": float(p2_max.max()),
                },
                "p1_total_damage": {
                    "mean": float(p1_total.mean()),
                    "std": float(p1_total.std()),
                },
                "p2_total_damage": {
                    "mean": float(p2_total.mean()),
                    "std": float(p2_total.std()),
                },
            }

        return {
            "total_episodes": num_episodes,
            "total_frames": self._frames_processed,
            "length": length_stats,
            "stages": stages,
            "stocks_lost": {
                "p1": p1_stocks,
                "p2": p2_stocks,
            },
            "outcomes": outcomes,
            "damage": damage_stats,
        }
