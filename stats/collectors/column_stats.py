"""Per-column statistics collector."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from stats.collectors.base import StatsCollector
from stats.index import EpisodeInfo

DEFAULT_PERCENTILES: List[float] = [
    0.5,
    1.0,
    5.0,
    10.0,
    25.0,
    50.0,
    75.0,
    90.0,
    95.0,
    99.0,
    99.5,
]
CATEGORICAL_THRESHOLD = 50  # Max unique values to treat as categorical


@dataclass
class RunLengthStats:
    """Statistics about run lengths (consecutive identical values)."""

    counts: Counter = field(default_factory=Counter)
    total_length: int = 0
    total_runs: int = 0

    def add(self, length: int) -> None:
        if length <= 0:
            return
        self.counts[length] += 1
        self.total_length += length
        self.total_runs += 1

    def merge(self, other: "RunLengthStats") -> None:
        self.counts.update(other.counts)
        self.total_length += other.total_length
        self.total_runs += other.total_runs

    def summary(self) -> Dict[str, Optional[float]]:
        if self.total_runs == 0:
            return {"min": None, "max": None, "mean": None, "median": None}
        lengths = sorted(self.counts.items())
        return {
            "min": lengths[0][0],
            "max": lengths[-1][0],
            "mean": self.total_length / self.total_runs,
            "median": self._compute_median(lengths),
        }

    def _compute_median(self, lengths: List[Tuple[int, int]]) -> float:
        mid1 = (self.total_runs + 1) // 2
        mid2 = (self.total_runs + 2) // 2
        acc, first, second = 0, None, None
        for length, count in lengths:
            acc += count
            if first is None and acc >= mid1:
                first = length
            if acc >= mid2:
                second = length
                break
        return (first + second) / 2.0 if first and second else 0.0

    def percentiles(self, pcts: Sequence[float]) -> Dict[str, Optional[float]]:
        if self.total_runs == 0:
            return {str(p): None for p in pcts}
        sorted_counts = sorted(self.counts.items())
        results = {}
        for p in pcts:
            idx = (self.total_runs - 1) * (p / 100.0)
            results[str(p)] = self._value_at_index(sorted_counts, idx)
        return results

    def _value_at_index(
        self, sorted_counts: List[Tuple[int, int]], idx: float
    ) -> float:
        if idx <= 0:
            return float(sorted_counts[0][0])
        if idx >= self.total_runs - 1:
            return float(sorted_counts[-1][0])

        lower_idx = int(math.floor(idx))
        upper_idx = int(math.ceil(idx))

        # Build cumulative counts
        cum = 0
        lower_val = upper_val = sorted_counts[0][0]
        for length, count in sorted_counts:
            if cum <= lower_idx < cum + count:
                lower_val = length
            if cum <= upper_idx < cum + count:
                upper_val = length
                break
            cum += count

        fraction = idx - lower_idx
        return lower_val + fraction * (upper_val - lower_val)


@dataclass
class ValueStats:
    """Statistics for a single value in categorical columns."""

    count: int = 0
    run_stats: RunLengthStats = field(default_factory=RunLengthStats)

    def merge(self, other: "ValueStats") -> None:
        self.count += other.count
        self.run_stats.merge(other.run_stats)


@dataclass
class ContinuousStats:
    """Statistics for continuous columns."""

    total: int = 0
    sum_: float = 0.0
    sum_sq: float = 0.0
    min_: float = float("inf")
    max_: float = float("-inf")
    run_stats: RunLengthStats = field(default_factory=RunLengthStats)
    # For exact percentiles we accumulate samples (up to a limit)
    samples: List[float] = field(default_factory=list)
    max_samples: int = 100000

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        flat = values.astype(np.float64).ravel()
        self.total += flat.size
        self.sum_ += float(flat.sum())
        self.sum_sq += float((flat * flat).sum())
        self.min_ = min(self.min_, float(flat.min()))
        self.max_ = max(self.max_, float(flat.max()))

        # Reservoir sampling for percentiles
        if len(self.samples) < self.max_samples:
            to_add = min(self.max_samples - len(self.samples), len(flat))
            self.samples.extend(flat[:to_add].tolist())

    def merge(self, other: "ContinuousStats") -> None:
        if other.total == 0:
            return
        self.total += other.total
        self.sum_ += other.sum_
        self.sum_sq += other.sum_sq
        if other.min_ < self.min_:
            self.min_ = other.min_
        if other.max_ > self.max_:
            self.max_ = other.max_
        self.run_stats.merge(other.run_stats)
        # Merge samples with simple truncation
        remaining = self.max_samples - len(self.samples)
        if remaining > 0:
            self.samples.extend(other.samples[:remaining])

    def finalize(self, percentiles: Sequence[float]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": "continuous",
            "total_count": self.total,
        }
        if self.total == 0:
            return result

        mean = self.sum_ / self.total
        variance = max(0.0, (self.sum_sq / self.total) - (mean * mean))
        result.update(
            {
                "min": self.min_,
                "max": self.max_,
                "mean": mean,
                "std": math.sqrt(variance),
                "run_lengths": self.run_stats.summary(),
                "run_percentiles": self.run_stats.percentiles(percentiles),
            }
        )

        # Compute percentiles from samples
        if self.samples:
            arr = np.array(self.samples)
            pct_values = np.percentile(arr, percentiles)
            result["percentiles"] = {
                str(p): float(v) for p, v in zip(percentiles, pct_values)
            }

        return result


@dataclass
class CategoricalStats:
    """Statistics for categorical columns."""

    total: int = 0
    value_stats: Dict[Any, ValueStats] = field(default_factory=dict)

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        self.total += values.size
        unique, counts = np.unique(values, return_counts=True)
        for val, cnt in zip(unique, counts):
            key = self._normalize_key(val)
            if key not in self.value_stats:
                self.value_stats[key] = ValueStats()
            self.value_stats[key].count += int(cnt)

    def update_runs(self, runs: Iterable[Tuple[Any, int]]) -> None:
        for value, length in runs:
            if value is None:
                continue
            key = self._normalize_key(value)
            if key not in self.value_stats:
                self.value_stats[key] = ValueStats()
            self.value_stats[key].run_stats.add(length)

    def merge(self, other: "CategoricalStats") -> None:
        self.total += other.total
        for key, stats in other.value_stats.items():
            if key not in self.value_stats:
                self.value_stats[key] = ValueStats()
            self.value_stats[key].merge(stats)

    def finalize(
        self, percentiles: Sequence[float], col_type: str = "categorical"
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "type": col_type,
            "total_count": self.total,
            "cardinality": len(self.value_stats),
        }

        values = []
        for key in sorted(self.value_stats.keys(), key=lambda x: (str(type(x)), x)):
            stats = self.value_stats[key]
            percent = (stats.count / self.total * 100.0) if self.total else 0.0
            values.append(
                {
                    "value": key,
                    "count": stats.count,
                    "percent": percent,
                    "run_lengths": stats.run_stats.summary(),
                    "run_percentiles": stats.run_stats.percentiles(percentiles),
                }
            )
        result["values"] = values
        return result

    @staticmethod
    def _normalize_key(value: Any) -> Any:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            if math.isnan(value):
                return None
            rounded = round(value)
            if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-8):
                return int(rounded)
        if isinstance(value, (np.integer, int)):
            return int(value)
        return value


def iter_runs(values: np.ndarray) -> Iterable[Tuple[Any, int]]:
    """Iterate over runs of consecutive identical values."""
    if values.size == 0:
        return
    prev = CategoricalStats._normalize_key(values[0])
    run = 1
    for raw_val in values[1:]:
        current = CategoricalStats._normalize_key(raw_val)
        if current == prev or (prev is None and current is None):
            run += 1
        else:
            yield prev, run
            prev = current
            run = 1
    yield prev, run


class ColumnStatsCollector(StatsCollector):
    """Collector for per-column statistics."""

    name = "columns"

    def __init__(
        self,
        feature_names: Sequence[str],
        percentiles: Sequence[float] = DEFAULT_PERCENTILES,
        categorical_threshold: int = CATEGORICAL_THRESHOLD,
    ) -> None:
        super().__init__(feature_names)
        self.percentiles = list(percentiles)
        self.categorical_threshold = categorical_threshold

        # Column type classification
        self._column_types: Dict[
            str, str
        ] = {}  # "boolean", "categorical", "continuous"
        self._continuous_stats: Dict[str, ContinuousStats] = {}
        self._categorical_stats: Dict[str, CategoricalStats] = {}

        # Track unique values per column during classification
        self._unique_values: Dict[str, set] = {name: set() for name in feature_names}
        self._is_boolean_candidate: Dict[str, bool] = {
            name: True for name in feature_names
        }
        self._exceeded_threshold: Dict[str, bool] = {
            name: False for name in feature_names
        }
        self._classification_done = False

    def process_episode(self, data: np.ndarray, episode: EpisodeInfo) -> None:
        num_frames = data.shape[0]

        # Always update classification tracking
        self._classify_columns(data)

        # Collect BOTH continuous and categorical stats for all columns
        # We'll choose which to report at finalize time based on final classification
        for col_name in self.feature_names:
            col_idx = self.get_feature_idx(col_name)
            col_data = data[:, col_idx]

            # Always collect continuous stats
            if col_name not in self._continuous_stats:
                self._continuous_stats[col_name] = ContinuousStats()
            self._continuous_stats[col_name].update(col_data)
            for _, length in iter_runs(col_data):
                self._continuous_stats[col_name].run_stats.add(length)

            # Also collect categorical stats if cardinality hasn't exceeded threshold
            if not self._exceeded_threshold[col_name]:
                if col_name not in self._categorical_stats:
                    self._categorical_stats[col_name] = CategoricalStats()
                self._categorical_stats[col_name].update(col_data)
                self._categorical_stats[col_name].update_runs(iter_runs(col_data))

        self._record_episode(num_frames)

    def _classify_columns(self, data: np.ndarray) -> None:
        """Classify columns based on their values."""
        for col_name in self.feature_names:
            col_idx = self.get_feature_idx(col_name)
            col_data = data[:, col_idx]

            # Check if still boolean candidate
            if self._is_boolean_candidate[col_name]:
                if not np.all(np.isin(col_data, [0, 1])):
                    self._is_boolean_candidate[col_name] = False

            # Track unique values
            unique = np.unique(col_data)
            for val in unique:
                self._unique_values[col_name].add(CategoricalStats._normalize_key(val))

            # Stop tracking if over threshold
            if len(self._unique_values[col_name]) > self.categorical_threshold:
                self._unique_values[col_name].clear()
                self._exceeded_threshold[col_name] = True

    def _finalize_classification(self) -> None:
        """Finalize column type classification."""
        if self._classification_done:
            return

        for col_name in self.feature_names:
            unique = self._unique_values[col_name]
            non_null = {v for v in unique if v is not None}

            if self._is_boolean_candidate[col_name] and non_null.issubset({0, 1}):
                self._column_types[col_name] = "boolean"
            elif not self._exceeded_threshold[col_name] and len(unique) > 0:
                self._column_types[col_name] = "categorical"
            else:
                self._column_types[col_name] = "continuous"

        self._classification_done = True

    def merge(self, other: "ColumnStatsCollector") -> None:
        # Merge unique value tracking
        for col_name in self.feature_names:
            self._unique_values[col_name].update(other._unique_values[col_name])
            if len(self._unique_values[col_name]) > self.categorical_threshold:
                self._unique_values[col_name].clear()
                self._exceeded_threshold[col_name] = True
            self._is_boolean_candidate[col_name] = (
                self._is_boolean_candidate[col_name]
                and other._is_boolean_candidate[col_name]
            )
            self._exceeded_threshold[col_name] = (
                self._exceeded_threshold[col_name]
                or other._exceeded_threshold[col_name]
            )

        # Merge statistics
        for col_name, stats in other._continuous_stats.items():
            if col_name not in self._continuous_stats:
                self._continuous_stats[col_name] = ContinuousStats()
            self._continuous_stats[col_name].merge(stats)

        for col_name, stats in other._categorical_stats.items():
            if col_name not in self._categorical_stats:
                self._categorical_stats[col_name] = CategoricalStats()
            self._categorical_stats[col_name].merge(stats)

        self._episodes_processed += other._episodes_processed
        self._frames_processed += other._frames_processed

    def finalize(self) -> Dict[str, Any]:
        self._finalize_classification()

        results: Dict[str, Any] = {}
        for col_name in tqdm(
            self.feature_names,
            desc="  Finalizing column stats",
            unit="col",
            leave=False,
        ):
            col_type = self._column_types.get(col_name, "continuous")

            if col_type == "continuous":
                stats = self._continuous_stats.get(col_name, ContinuousStats())
                results[col_name] = stats.finalize(self.percentiles)
            else:
                stats = self._categorical_stats.get(col_name, CategoricalStats())
                results[col_name] = stats.finalize(self.percentiles, col_type)

        return results
