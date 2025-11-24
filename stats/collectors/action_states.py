"""Action state analysis collector."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from stats.collectors.base import StatsCollector
from stats.index import EpisodeInfo
from stats.utils.melee_constants import (
    ACTION_STATE_CATEGORIES,
    ACTION_STATE_NAMES,
    get_action_category,
    get_action_group,
)


class ActionStateCollector(StatsCollector):
    """Collector for action state analysis."""

    name = "action_states"

    def __init__(self, feature_names: Sequence[str]) -> None:
        super().__init__(feature_names)

        # Per-player action state counts
        self._p1_action_counts: Counter = Counter()
        self._p2_action_counts: Counter = Counter()

        # Per-player category counts
        self._p1_category_counts: Counter = Counter()
        self._p2_category_counts: Counter = Counter()

        # Per-player group counts
        self._p1_group_counts: Counter = Counter()
        self._p2_group_counts: Counter = Counter()

        # Transition counts (from_action -> to_action)
        self._p1_transitions: Dict[int, Counter] = defaultdict(Counter)
        self._p2_transitions: Dict[int, Counter] = defaultdict(Counter)

        # Category transitions
        self._p1_category_transitions: Dict[str, Counter] = defaultdict(Counter)
        self._p2_category_transitions: Dict[str, Counter] = defaultdict(Counter)

        # Run length statistics per action
        self._p1_action_runs: Dict[int, List[int]] = defaultdict(list)
        self._p2_action_runs: Dict[int, List[int]] = defaultdict(list)

    def process_episode(self, data: np.ndarray, episode: EpisodeInfo) -> None:
        num_frames = data.shape[0]

        try:
            p1_action_idx = self.get_feature_idx("p1_action")
            p2_action_idx = self.get_feature_idx("p2_action")
        except KeyError:
            self._record_episode(num_frames)
            return

        p1_actions = data[:, p1_action_idx].astype(np.int32)
        p2_actions = data[:, p2_action_idx].astype(np.int32)

        # Process each player
        self._process_player_actions(
            p1_actions,
            self._p1_action_counts,
            self._p1_category_counts,
            self._p1_group_counts,
            self._p1_transitions,
            self._p1_category_transitions,
            self._p1_action_runs,
        )
        self._process_player_actions(
            p2_actions,
            self._p2_action_counts,
            self._p2_category_counts,
            self._p2_group_counts,
            self._p2_transitions,
            self._p2_category_transitions,
            self._p2_action_runs,
        )

        self._record_episode(num_frames)

    def _process_player_actions(
        self,
        actions: np.ndarray,
        action_counts: Counter,
        category_counts: Counter,
        group_counts: Counter,
        transitions: Dict[int, Counter],
        category_transitions: Dict[str, Counter],
        action_runs: Dict[int, List[int]],
    ) -> None:
        if len(actions) == 0:
            return

        # Count actions
        unique, counts = np.unique(actions, return_counts=True)
        for action, count in zip(unique, counts):
            action = int(action)
            action_counts[action] += int(count)
            category = get_action_category(action)
            category_counts[category] += int(count)
            group = get_action_group(action)
            group_counts[group] += int(count)

        # Count transitions and runs
        prev_action = int(actions[0])
        prev_category = get_action_category(prev_action)
        run_length = 1

        for i in range(1, len(actions)):
            curr_action = int(actions[i])
            curr_category = get_action_category(curr_action)

            if curr_action != prev_action:
                # Record transition
                transitions[prev_action][curr_action] += 1
                # Record run
                action_runs[prev_action].append(run_length)
                run_length = 1
            else:
                run_length += 1

            if curr_category != prev_category:
                category_transitions[prev_category][curr_category] += 1

            prev_action = curr_action
            prev_category = curr_category

        # Record final run
        action_runs[prev_action].append(run_length)

    def merge(self, other: "ActionStateCollector") -> None:
        self._p1_action_counts.update(other._p1_action_counts)
        self._p2_action_counts.update(other._p2_action_counts)
        self._p1_category_counts.update(other._p1_category_counts)
        self._p2_category_counts.update(other._p2_category_counts)
        self._p1_group_counts.update(other._p1_group_counts)
        self._p2_group_counts.update(other._p2_group_counts)

        for action, counter in other._p1_transitions.items():
            self._p1_transitions[action].update(counter)
        for action, counter in other._p2_transitions.items():
            self._p2_transitions[action].update(counter)

        for cat, counter in other._p1_category_transitions.items():
            self._p1_category_transitions[cat].update(counter)
        for cat, counter in other._p2_category_transitions.items():
            self._p2_category_transitions[cat].update(counter)

        for action, runs in other._p1_action_runs.items():
            self._p1_action_runs[action].extend(runs)
        for action, runs in other._p2_action_runs.items():
            self._p2_action_runs[action].extend(runs)

        self._episodes_processed += other._episodes_processed
        self._frames_processed += other._frames_processed

    def finalize(self) -> Dict[str, Any]:
        def format_action_distribution(counts: Counter, total: int) -> List[Dict]:
            result = []
            for action, count in counts.most_common():
                name = ACTION_STATE_NAMES.get(action, f"Action_{action}")
                category = get_action_category(action)
                result.append({
                    "action_id": action,
                    "name": name,
                    "category": category,
                    "count": count,
                    "percent": count / total * 100 if total else 0,
                })
            return result

        def format_category_distribution(counts: Counter, total: int) -> Dict:
            return {
                cat: {
                    "count": count,
                    "percent": count / total * 100 if total else 0,
                }
                for cat, count in counts.most_common()
            }

        def format_transitions(trans: Dict[int, Counter], limit: int = 10) -> Dict:
            result = {}
            for from_action, to_counts in trans.items():
                from_name = ACTION_STATE_NAMES.get(from_action, f"Action_{from_action}")
                top_transitions = []
                total = sum(to_counts.values())
                for to_action, count in to_counts.most_common(limit):
                    to_name = ACTION_STATE_NAMES.get(to_action, f"Action_{to_action}")
                    top_transitions.append({
                        "to_action": to_action,
                        "to_name": to_name,
                        "count": count,
                        "percent": count / total * 100 if total else 0,
                    })
                result[from_name] = {
                    "from_action": from_action,
                    "total_transitions": total,
                    "top_transitions": top_transitions,
                }
            return result

        def format_run_stats(runs: Dict[int, List[int]]) -> Dict:
            result = {}
            for action, run_lengths in runs.items():
                if not run_lengths:
                    continue
                name = ACTION_STATE_NAMES.get(action, f"Action_{action}")
                arr = np.array(run_lengths)
                result[name] = {
                    "action_id": action,
                    "total_runs": len(run_lengths),
                    "min": int(arr.min()),
                    "max": int(arr.max()),
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                }
            return result

        p1_total = sum(self._p1_action_counts.values())
        p2_total = sum(self._p2_action_counts.values())

        return {
            "p1": {
                "total_frames": p1_total,
                "action_distribution": format_action_distribution(self._p1_action_counts, p1_total),
                "category_distribution": format_category_distribution(self._p1_category_counts, p1_total),
                "group_distribution": format_category_distribution(self._p1_group_counts, p1_total),
                "run_lengths": format_run_stats(self._p1_action_runs),
            },
            "p2": {
                "total_frames": p2_total,
                "action_distribution": format_action_distribution(self._p2_action_counts, p2_total),
                "category_distribution": format_category_distribution(self._p2_category_counts, p2_total),
                "group_distribution": format_category_distribution(self._p2_group_counts, p2_total),
                "run_lengths": format_run_stats(self._p2_action_runs),
            },
            "category_transitions": {
                "p1": {k: dict(v) for k, v in self._p1_category_transitions.items()},
                "p2": {k: dict(v) for k, v in self._p2_category_transitions.items()},
            },
        }
