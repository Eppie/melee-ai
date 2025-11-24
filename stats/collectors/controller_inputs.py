"""Controller input analysis collector."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from stats.collectors.base import StatsCollector
from stats.index import EpisodeInfo
from stats.utils.melee_constants import BUTTON_NAMES, get_stick_region


class ControllerInputCollector(StatsCollector):
    """Collector for controller input analysis."""

    name = "controller_inputs"

    def __init__(self, feature_names: Sequence[str]) -> None:
        super().__init__(feature_names)

        # Button press counts
        self._p1_button_counts: Dict[str, int] = {b: 0 for b in BUTTON_NAMES}
        self._p2_button_counts: Dict[str, int] = {b: 0 for b in BUTTON_NAMES}

        # Button press durations (frames held)
        self._p1_button_durations: Dict[str, List[int]] = {b: [] for b in BUTTON_NAMES}
        self._p2_button_durations: Dict[str, List[int]] = {b: [] for b in BUTTON_NAMES}

        # Button co-occurrence (which buttons pressed together)
        self._p1_button_combos: Counter = Counter()
        self._p2_button_combos: Counter = Counter()

        # Stick region counts
        self._p1_main_stick_regions: Counter = Counter()
        self._p2_main_stick_regions: Counter = Counter()
        self._p1_c_stick_regions: Counter = Counter()
        self._p2_c_stick_regions: Counter = Counter()

        # Stick position histograms (binned)
        self._p1_main_stick_hist = np.zeros((20, 20), dtype=np.int64)
        self._p2_main_stick_hist = np.zeros((20, 20), dtype=np.int64)
        self._p1_c_stick_hist = np.zeros((20, 20), dtype=np.int64)
        self._p2_c_stick_hist = np.zeros((20, 20), dtype=np.int64)

        # Shoulder analog distribution
        self._p1_shoulder_hist = np.zeros(10, dtype=np.int64)
        self._p2_shoulder_hist = np.zeros(10, dtype=np.int64)

        # Total frames for rate calculations
        self._total_frames = 0

    def process_episode(self, data: np.ndarray, episode: EpisodeInfo) -> None:
        num_frames = data.shape[0]
        self._total_frames += num_frames

        # Process buttons for each player
        for player, button_counts, button_durations, button_combos in [
            (
                "p1",
                self._p1_button_counts,
                self._p1_button_durations,
                self._p1_button_combos,
            ),
            (
                "p2",
                self._p2_button_counts,
                self._p2_button_durations,
                self._p2_button_combos,
            ),
        ]:
            self._process_buttons(
                data, player, button_counts, button_durations, button_combos
            )

        # Process sticks
        self._process_sticks(data)

        self._record_episode(num_frames)

    def _process_buttons(
        self,
        data: np.ndarray,
        player: str,
        button_counts: Dict[str, int],
        button_durations: Dict[str, List[int]],
        button_combos: Counter,
    ) -> None:
        button_data = {}
        for button in BUTTON_NAMES:
            col_name = f"{player}_{button}"
            try:
                idx = self.get_feature_idx(col_name)
                button_data[button] = data[:, idx] > 0.5
            except KeyError:
                continue

        if not button_data:
            return

        # Count presses and durations
        for button, pressed in button_data.items():
            # Count frames where button is pressed
            button_counts[button] += int(pressed.sum())

            # Track press durations (transitions from 0 to 1 start a new press)
            in_press = False
            press_length = 0
            for val in pressed:
                if val and not in_press:
                    in_press = True
                    press_length = 1
                elif val and in_press:
                    press_length += 1
                elif not val and in_press:
                    button_durations[button].append(press_length)
                    in_press = False
                    press_length = 0
            if in_press:
                button_durations[button].append(press_length)

        # Track button combinations
        button_matrix = np.column_stack(list(button_data.values()))
        for frame in range(len(data)):
            combo = tuple(
                sorted(
                    button
                    for i, button in enumerate(button_data.keys())
                    if button_matrix[frame, i]
                )
            )
            if combo:
                button_combos[combo] += 1

    def _process_sticks(self, data: np.ndarray) -> None:
        for player, main_hist, c_hist, main_regions, c_regions, shoulder_hist in [
            (
                "p1",
                self._p1_main_stick_hist,
                self._p1_c_stick_hist,
                self._p1_main_stick_regions,
                self._p1_c_stick_regions,
                self._p1_shoulder_hist,
            ),
            (
                "p2",
                self._p2_main_stick_hist,
                self._p2_c_stick_hist,
                self._p2_main_stick_regions,
                self._p2_c_stick_regions,
                self._p2_shoulder_hist,
            ),
        ]:
            try:
                main_x_idx = self.get_feature_idx(f"{player}_main_stick_x")
                main_y_idx = self.get_feature_idx(f"{player}_main_stick_y")
                c_x_idx = self.get_feature_idx(f"{player}_c_stick_x")
                c_y_idx = self.get_feature_idx(f"{player}_c_stick_y")
                shoulder_idx = self.get_feature_idx(f"{player}_shoulder_analog")
            except KeyError:
                continue

            main_x = data[:, main_x_idx]
            main_y = data[:, main_y_idx]
            c_x = data[:, c_x_idx]
            c_y = data[:, c_y_idx]
            shoulder = data[:, shoulder_idx]

            # Update histograms
            main_x_bins = np.clip((main_x * 20).astype(int), 0, 19)
            main_y_bins = np.clip((main_y * 20).astype(int), 0, 19)
            for x, y in zip(main_x_bins, main_y_bins):
                main_hist[y, x] += 1

            c_x_bins = np.clip((c_x * 20).astype(int), 0, 19)
            c_y_bins = np.clip((c_y * 20).astype(int), 0, 19)
            for x, y in zip(c_x_bins, c_y_bins):
                c_hist[y, x] += 1

            shoulder_bins = np.clip((shoulder * 10).astype(int), 0, 9)
            for b in shoulder_bins:
                shoulder_hist[b] += 1

            # Update region counts
            for x, y in zip(main_x, main_y):
                main_regions[get_stick_region(x, y)] += 1
            for x, y in zip(c_x, c_y):
                c_regions[get_stick_region(x, y)] += 1

    def merge(self, other: "ControllerInputCollector") -> None:
        for button in BUTTON_NAMES:
            self._p1_button_counts[button] += other._p1_button_counts[button]
            self._p2_button_counts[button] += other._p2_button_counts[button]
            self._p1_button_durations[button].extend(other._p1_button_durations[button])
            self._p2_button_durations[button].extend(other._p2_button_durations[button])

        self._p1_button_combos.update(other._p1_button_combos)
        self._p2_button_combos.update(other._p2_button_combos)

        self._p1_main_stick_regions.update(other._p1_main_stick_regions)
        self._p2_main_stick_regions.update(other._p2_main_stick_regions)
        self._p1_c_stick_regions.update(other._p1_c_stick_regions)
        self._p2_c_stick_regions.update(other._p2_c_stick_regions)

        self._p1_main_stick_hist += other._p1_main_stick_hist
        self._p2_main_stick_hist += other._p2_main_stick_hist
        self._p1_c_stick_hist += other._p1_c_stick_hist
        self._p2_c_stick_hist += other._p2_c_stick_hist
        self._p1_shoulder_hist += other._p1_shoulder_hist
        self._p2_shoulder_hist += other._p2_shoulder_hist

        self._total_frames += other._total_frames
        self._episodes_processed += other._episodes_processed
        self._frames_processed += other._frames_processed

    def finalize(self) -> Dict[str, Any]:
        def format_button_stats(
            counts: Dict, durations: Dict, combos: Counter, total: int
        ) -> Dict:
            buttons = {}
            for button in BUTTON_NAMES:
                count = counts[button]
                durs = durations[button]
                dur_arr = np.array(durs) if durs else np.array([0])
                buttons[button] = {
                    "total_frames_pressed": count,
                    "percent_frames": count / total * 100 if total else 0,
                    "num_presses": len(durs),
                    "presses_per_minute": len(durs) / (total / 3600) if total else 0,
                    "avg_hold_frames": float(dur_arr.mean()) if len(durs) > 0 else 0,
                    "max_hold_frames": int(dur_arr.max()) if len(durs) > 0 else 0,
                }

            # Top button combos
            top_combos = []
            for combo, count in combos.most_common(20):
                top_combos.append(
                    {
                        "buttons": list(combo),
                        "count": count,
                        "percent": count / total * 100 if total else 0,
                    }
                )

            return {
                "buttons": buttons,
                "top_combinations": top_combos,
            }

        def format_stick_stats(regions: Counter, hist: np.ndarray, total: int) -> Dict:
            region_pcts = {
                region: {
                    "count": count,
                    "percent": count / total * 100 if total else 0,
                }
                for region, count in regions.most_common()
            }
            return {
                "regions": region_pcts,
                "histogram": hist.tolist(),
            }

        def format_shoulder_stats(hist: np.ndarray, total: int) -> Dict:
            bins = []
            for i, count in enumerate(hist):
                bins.append(
                    {
                        "range": f"{i/10:.1f}-{(i+1)/10:.1f}",
                        "count": int(count),
                        "percent": count / total * 100 if total else 0,
                    }
                )
            return {"distribution": bins}

        return {
            "p1": {
                **format_button_stats(
                    self._p1_button_counts,
                    self._p1_button_durations,
                    self._p1_button_combos,
                    self._total_frames,
                ),
                "main_stick": format_stick_stats(
                    self._p1_main_stick_regions,
                    self._p1_main_stick_hist,
                    self._total_frames,
                ),
                "c_stick": format_stick_stats(
                    self._p1_c_stick_regions,
                    self._p1_c_stick_hist,
                    self._total_frames,
                ),
                "shoulder": format_shoulder_stats(
                    self._p1_shoulder_hist, self._total_frames
                ),
            },
            "p2": {
                **format_button_stats(
                    self._p2_button_counts,
                    self._p2_button_durations,
                    self._p2_button_combos,
                    self._total_frames,
                ),
                "main_stick": format_stick_stats(
                    self._p2_main_stick_regions,
                    self._p2_main_stick_hist,
                    self._total_frames,
                ),
                "c_stick": format_stick_stats(
                    self._p2_c_stick_regions,
                    self._p2_c_stick_hist,
                    self._total_frames,
                ),
                "shoulder": format_shoulder_stats(
                    self._p2_shoulder_hist, self._total_frames
                ),
            },
            "total_frames": self._total_frames,
        }
