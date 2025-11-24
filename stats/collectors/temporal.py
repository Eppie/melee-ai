"""Temporal analysis collector (transitions, change rates)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from stats.collectors.base import StatsCollector
from stats.index import EpisodeInfo


class TemporalCollector(StatsCollector):
    """Collector for temporal patterns and transitions."""

    name = "temporal"

    # Features to track transitions for
    TRANSITION_FEATURES = [
        "p1_on_ground",
        "p2_on_ground",
        "p1_facing",
        "p2_facing",
        "p1_is_shield_active",
        "p2_is_shield_active",
    ]

    # Features to track change rates for
    CHANGE_RATE_FEATURES = [
        "p1_position_x",
        "p1_position_y",
        "p2_position_x",
        "p2_position_y",
        "p1_percent",
        "p2_percent",
    ]

    def __init__(self, feature_names: Sequence[str]) -> None:
        super().__init__(feature_names)

        # State transition counts (0->0, 0->1, 1->0, 1->1)
        self._transitions: Dict[str, Counter] = {
            f: Counter() for f in self.TRANSITION_FEATURES
        }

        # Frame-to-frame change magnitudes
        self._change_magnitudes: Dict[str, List[float]] = {
            f: [] for f in self.CHANGE_RATE_FEATURES
        }

        # Time between state changes
        self._time_between_changes: Dict[str, List[int]] = {}

        # Button timing relative to action changes
        self._button_to_action_timing: Dict[str, List[int]] = {}

        # Input sequence patterns (last N inputs)
        self._input_sequences: Counter = Counter()

    def process_episode(self, data: np.ndarray, episode: EpisodeInfo) -> None:
        num_frames = data.shape[0]
        if num_frames < 2:
            self._record_episode(num_frames)
            return

        # State transitions
        for feature in self.TRANSITION_FEATURES:
            try:
                idx = self.get_feature_idx(feature)
                values = (data[:, idx] > 0.5).astype(int)

                for i in range(1, len(values)):
                    transition = (values[i - 1], values[i])
                    self._transitions[feature][transition] += 1

                    # Track time between changes
                    if feature not in self._time_between_changes:
                        self._time_between_changes[feature] = []
            except KeyError:
                pass

        # Change rates for continuous features
        for feature in self.CHANGE_RATE_FEATURES:
            try:
                idx = self.get_feature_idx(feature)
                values = data[:, idx]
                changes = np.abs(np.diff(values))

                # Sample changes to avoid memory issues
                if len(changes) > 100:
                    sample_idx = np.random.choice(len(changes), 100, replace=False)
                    changes = changes[sample_idx]

                self._change_magnitudes[feature].extend(changes.tolist())
            except KeyError:
                pass

        # Track time between state changes for boolean features
        for feature in ["p1_facing", "p2_facing", "p1_on_ground", "p2_on_ground"]:
            try:
                idx = self.get_feature_idx(feature)
                values = (data[:, idx] > 0.5).astype(int)

                if feature not in self._time_between_changes:
                    self._time_between_changes[feature] = []

                last_change = 0
                for i in range(1, len(values)):
                    if values[i] != values[i - 1]:
                        time_since = i - last_change
                        self._time_between_changes[feature].append(time_since)
                        last_change = i
            except KeyError:
                pass

        # Input sequence analysis (simplified)
        self._analyze_input_sequences(data)

        self._record_episode(num_frames)

    def _analyze_input_sequences(self, data: np.ndarray) -> None:
        """Analyze common input sequences."""
        try:
            # Get button states
            button_cols = []
            for button in [
                "button_a",
                "button_b",
                "button_xy",
                "button_z",
                "button_lr",
            ]:
                try:
                    idx = self.get_feature_idx(f"p1_{button}")
                    button_cols.append(data[:, idx] > 0.5)
                except KeyError:
                    pass

            if not button_cols:
                return

            button_matrix = np.column_stack(button_cols)

            # Look at 3-frame windows
            window_size = 3
            for i in range(len(button_matrix) - window_size):
                window = button_matrix[i : i + window_size]
                # Create a hashable key for the sequence
                key = tuple(tuple(row.astype(int)) for row in window)
                self._input_sequences[key] += 1

        except Exception:
            pass

    def merge(self, other: "TemporalCollector") -> None:
        for feature in self.TRANSITION_FEATURES:
            self._transitions[feature].update(other._transitions[feature])

        for feature in self.CHANGE_RATE_FEATURES:
            self._change_magnitudes[feature].extend(other._change_magnitudes[feature])

        for feature, times in other._time_between_changes.items():
            if feature not in self._time_between_changes:
                self._time_between_changes[feature] = []
            self._time_between_changes[feature].extend(times)

        self._input_sequences.update(other._input_sequences)

        self._episodes_processed += other._episodes_processed
        self._frames_processed += other._frames_processed

    def finalize(self) -> Dict[str, Any]:
        # State transitions
        transitions = {}
        for feature, counts in self._transitions.items():
            total = sum(counts.values())
            transitions[feature] = {
                "stay_off": counts[(0, 0)],
                "turn_on": counts[(0, 1)],
                "turn_off": counts[(1, 0)],
                "stay_on": counts[(1, 1)],
                "total": total,
            }
            if total > 0:
                transitions[feature]["turn_on_rate"] = (
                    counts[(0, 1)] / (counts[(0, 0)] + counts[(0, 1)])
                    if (counts[(0, 0)] + counts[(0, 1)]) > 0
                    else 0
                )
                transitions[feature]["turn_off_rate"] = (
                    counts[(1, 0)] / (counts[(1, 0)] + counts[(1, 1)])
                    if (counts[(1, 0)] + counts[(1, 1)]) > 0
                    else 0
                )

        # Change rates
        change_rates = {}
        for feature, magnitudes in self._change_magnitudes.items():
            if magnitudes:
                arr = np.array(magnitudes)
                change_rates[feature] = {
                    "mean_change": float(arr.mean()),
                    "std_change": float(arr.std()),
                    "max_change": float(arr.max()),
                    "zero_change_percent": float((arr == 0).sum() / len(arr) * 100),
                }

        # Time between changes
        time_between = {}
        for feature, times in self._time_between_changes.items():
            if times:
                arr = np.array(times)
                time_between[feature] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "median": float(np.median(arr)),
                    "min": int(arr.min()),
                    "max": int(arr.max()),
                }

        # Top input sequences
        top_sequences = []
        for seq, count in self._input_sequences.most_common(20):
            # Convert back to readable format
            button_names = ["A", "B", "XY", "Z", "LR"]
            frames = []
            for frame in seq:
                pressed = [
                    button_names[i]
                    for i, v in enumerate(frame)
                    if v and i < len(button_names)
                ]
                frames.append(pressed if pressed else ["none"])
            top_sequences.append(
                {
                    "sequence": frames,
                    "count": count,
                }
            )

        return {
            "state_transitions": transitions,
            "change_rates": change_rates,
            "time_between_changes": time_between,
            "top_input_sequences": top_sequences,
        }
