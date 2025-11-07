#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import multiprocessing

import numpy as np

from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)
from tqdm import tqdm

# TODO: Review this entire file for duplication -- seems like it re-implements some functions

DEFAULT_PERCENTILES: List[float] = [0.5, 1.0, 10.0, 25.0, 50.0, 90.0, 99.0, 99.5]

MAIN_STICK_PALETTE = np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32)
MAIN_STICK_PALETTE_NORM = np.sum(MAIN_STICK_PALETTE**2, axis=1, keepdims=True)
C_STICK_PALETTE = np.asarray(C_STICK_QUANTIZED, dtype=np.float32)
C_STICK_PALETTE_NORM = np.sum(C_STICK_PALETTE**2, axis=1, keepdims=True)

STICK_QUANTIZATION_CONFIG = [
    (
        "p1_main_stick_quantized",
        "p1_main_stick_x",
        "p1_main_stick_y",
        MAIN_STICK_PALETTE,
        MAIN_STICK_PALETTE_NORM,
    ),
    (
        "p2_main_stick_quantized",
        "p2_main_stick_x",
        "p2_main_stick_y",
        MAIN_STICK_PALETTE,
        MAIN_STICK_PALETTE_NORM,
    ),
    (
        "p1_c_stick_quantized",
        "p1_c_stick_x",
        "p1_c_stick_y",
        C_STICK_PALETTE,
        C_STICK_PALETTE_NORM,
    ),
    (
        "p2_c_stick_quantized",
        "p2_c_stick_x",
        "p2_c_stick_y",
        C_STICK_PALETTE,
        C_STICK_PALETTE_NORM,
    ),
]

SHOULDER_QUANTIZATION_CONFIG = {
    "p1_shoulder_analog": "p1_shoulder_quantized",
    "p2_shoulder_analog": "p2_shoulder_quantized",
}

SHOULDER_PALETTE = np.asarray(SHOULDER_QUANTIZED, dtype=np.float32)

_ARRAY_LAYOUT_CACHE: Dict[Path, Tuple[object, object]] = {}


def _get_array_dataset_names(
    data_root: Path, meta: Optional[Dict[str, object]] = None
) -> Tuple[str, str]:
    """Return the (feature, target) dataset names for ``data_root``."""
    root = data_root.resolve()
    cached = _ARRAY_LAYOUT_CACHE.get(root)
    if cached is not None and meta is None:
        return cached
    if meta is None:
        meta_path = root / "meta.json"
        with meta_path.open("r") as f:
            meta = json.load(f)
    layout = meta.get("array_layout", {}) if meta else {}
    feature_key = layout.get("features", {}).get("transformed", "X")
    target_key = layout.get("targets", {}).get("transformed", "Y")
    result = (feature_key, target_key)
    _ARRAY_LAYOUT_CACHE[root] = result
    return result


@dataclass(frozen=True)
class ShoulderQuantizationSpec:
    name: str
    source_idx: int

    @property
    def num_categories(self) -> int:
        return int(len(SHOULDER_QUANTIZED))


@dataclass(frozen=True)
class StickQuantizationSpec:
    name: str
    x_idx: int
    y_idx: int
    palette: np.ndarray
    palette_norm: np.ndarray

    @property
    def num_categories(self) -> int:
        return int(self.palette.shape[0])


@dataclass(frozen=True)
class EpisodeInfo:
    episode_id: int
    shard_id: int
    num_frames: int


class ValidationDatasetIndex:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir).resolve()
        meta_path = self.data_dir / "meta.json"
        index_path = self.data_dir / "index.jsonl"
        if not meta_path.exists() or not index_path.exists():
            raise FileNotFoundError(
                "Expected meta.json and index.jsonl in validation dataset root"
            )

        with meta_path.open("r") as f:
            meta = json.load(f)

        schema = meta.get("schema", {})
        self.feature_names: List[str] = list(schema.get("features", []))
        self.target_names: List[str] = list(schema.get("targets", []))
        (
            self._feature_dataset_name,
            self._target_dataset_name,
        ) = _get_array_dataset_names(self.data_dir, meta)

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

        self._shard_cache: Dict[int, object] = {}

    def open_episode_arrays(
        self, episode: EpisodeInfo
    ) -> Tuple["zarr.Array", Optional["zarr.Array"]]:
        import zarr

        shard_group = self._shard_cache.get(episode.shard_id)
        if shard_group is None:
            shard_path = self.data_dir / f"shard_{episode.shard_id:05d}.zarr"
            shard_group = zarr.open_group(str(shard_path), mode="r")
            self._shard_cache[episode.shard_id] = shard_group
        ep_group = shard_group[f"ep_{episode.episode_id:06d}"]
        feature_key = self._feature_dataset_name or "X"
        X = ep_group.get(feature_key)
        if X is None:
            X = ep_group["X"]
        return X, None


def _split_into_chunks(
    items: Sequence[EpisodeInfo], max_chunks: int
) -> List[List[EpisodeInfo]]:
    if not items:
        return []
    max_chunks = max(1, min(max_chunks, len(items)))
    chunk_size = (len(items) + max_chunks - 1) // max_chunks
    chunks: List[List[EpisodeInfo]] = []
    for start in range(0, len(items), chunk_size):
        end = min(len(items), start + chunk_size)
        chunks.append(list(items[start:end]))
    return chunks


def _resolve_worker_count(requested: Optional[int]) -> int:
    if requested is None or requested <= 0:
        cpu_count = multiprocessing.cpu_count() or 1
        return max(1, cpu_count)
    return requested


def _open_episode_array(
    data_root: Path, episode: EpisodeInfo, shard_cache: Dict[int, object]
) -> "zarr.Array":
    import zarr

    group = shard_cache.get(episode.shard_id)
    if group is None:
        shard_path = data_root / f"shard_{episode.shard_id:05d}.zarr"
        group = zarr.open_group(str(shard_path), mode="r")
        shard_cache[episode.shard_id] = group
    ep_name = f"ep_{episode.episode_id:06d}"
    feature_key, _ = _get_array_dataset_names(data_root)
    ep_group = group[ep_name]
    array = ep_group.get(feature_key)
    if array is None and feature_key != "X":
        array = ep_group.get("X")
    if array is None:
        raise KeyError(f"Feature array '{feature_key}' missing in {ep_name}")
    return array


def _sticks01_to_unit11_np(xy01: np.ndarray) -> np.ndarray:
    xy01_clipped = np.clip(xy01, 0.0, 1.0)
    xy11 = xy01_clipped * 2.0 - 1.0
    norms = np.linalg.norm(xy11, axis=1, keepdims=True)
    mask = norms > 1.0
    if np.any(mask):
        xy11[mask] /= norms[mask]
    return xy11


def _quantize_stick_indices(
    xy_values: np.ndarray,
    *,
    palette: np.ndarray,
    palette_norm: np.ndarray,
) -> np.ndarray:
    if xy_values.shape[1] != 2:
        raise ValueError("Stick quantization expects two columns (x, y).")
    values = xy_values.astype(np.float32, copy=False)
    if np.any(values < 0.0) or np.any(values > 1.0):
        xy11 = np.clip(values, -1.0, 1.0)
        norms = np.linalg.norm(xy11, axis=1, keepdims=True)
        mask = norms > 1.0
        if np.any(mask):
            xy11[mask] /= norms[mask]
    else:
        xy01 = np.clip(values, 0.0, 1.0)
        xy11 = _sticks01_to_unit11_np(xy01.copy())
    dot = xy11 @ palette.T
    norm = np.sum(xy11**2, axis=1, keepdims=True)
    d2 = norm - 2.0 * dot + palette_norm.T
    idx = np.argmin(d2, axis=1)
    return idx.astype(np.int32, copy=False)


def _quantize_shoulder_indices(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        values = values.reshape(-1)
    arr = values.astype(np.float32, copy=False)
    palette = SHOULDER_PALETTE
    idx = np.searchsorted(palette, arr, side="right") - 1
    idx = np.clip(idx, 0, palette.shape[0] - 1)
    return idx.astype(np.int32, copy=False)


def _canonical_value(value: object) -> object:
    """Convert numpy scalars to builtin Python types and normalize integers."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if math.isnan(value):
            return None
        rounded = round(value)
        if math.isclose(value, rounded, rel_tol=0.0, abs_tol=1e-8):
            return int(rounded)
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _values_equal(a: object, b: object) -> bool:
    if a is None and b is None:
        return True
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
    return a == b


@dataclass
class RunStats:
    counts: Counter
    total_length: int
    total_runs: int

    def __init__(self) -> None:
        self.counts = Counter()
        self.total_length = 0
        self.total_runs = 0

    def add(self, length: int) -> None:
        if length <= 0:
            return
        self.counts[length] += 1
        self.total_length += length
        self.total_runs += 1

    def merge(self, other: RunStats) -> None:
        if other.total_runs == 0:
            return
        self.counts.update(other.counts)
        self.total_length += other.total_length
        self.total_runs += other.total_runs

    def summary(self) -> Dict[str, Optional[float]]:
        if self.total_runs == 0:
            return {"min": None, "max": None, "mean": None, "median": None}
        lengths = sorted(self.counts.items())
        min_len = lengths[0][0]
        max_len = lengths[-1][0]
        mean_len = self.total_length / self.total_runs
        median_len = _median_from_counts(lengths, self.total_runs)
        return {
            "min": int(min_len),
            "max": int(max_len),
            "mean": mean_len,
            "median": median_len,
        }

    def percentile_values(
        self, percentiles: Sequence[float]
    ) -> Dict[str, Optional[float]]:
        if self.total_runs == 0:
            return {str(p): None for p in percentiles}
        if self.total_runs == 1:
            only_length = next(iter(self.counts))
            return {str(p): float(only_length) for p in percentiles}
        sorted_counts = sorted(self.counts.items())
        cumulative_counts = []
        total = 0
        for length, count in sorted_counts:
            total += count
            cumulative_counts.append((length, total))

        def value_at_index(idx: float) -> float:
            if idx <= 0:
                return float(sorted_counts[0][0])
            if idx >= self.total_runs - 1:
                return float(sorted_counts[-1][0])
            lower_idx = int(math.floor(idx))
            upper_idx = int(math.ceil(idx))
            lower_val = _value_at_rank(sorted_counts, cumulative_counts, lower_idx)
            upper_val = _value_at_rank(sorted_counts, cumulative_counts, upper_idx)
            fraction = idx - lower_idx
            return lower_val + fraction * (upper_val - lower_val)

        results: Dict[str, Optional[float]] = {}
        for p in percentiles:
            h = (self.total_runs - 1) * (p / 100.0)
            results[str(p)] = value_at_index(h)
        return results


def _median_from_counts(
    length_counts: Sequence[Tuple[int, int]], total_runs: int
) -> float:
    if total_runs == 0:
        return math.nan
    midpoint1 = (total_runs + 1) // 2
    midpoint2 = (total_runs + 2) // 2

    acc = 0
    first = None
    second = None
    for length, count in length_counts:
        acc += count
        if first is None and acc >= midpoint1:
            first = length
        if acc >= midpoint2:
            second = length
            break
    assert first is not None and second is not None
    return (first + second) / 2.0


def _value_at_rank(
    length_counts: Sequence[Tuple[int, int]],
    cumulative_counts: Sequence[Tuple[int, int]],
    rank: int,
) -> float:
    if rank <= 0:
        return float(length_counts[0][0])
    total_runs = cumulative_counts[-1][1]
    if rank >= total_runs - 1:
        return float(length_counts[-1][0])
    for length, cum_count in cumulative_counts:
        if rank < cum_count:
            return float(length)
    return float(length_counts[-1][0])


# --- Unified Value Stats -----------------------------------------------------


@dataclass
class ValueStats:
    """Statistics for a single value in categorical/boolean columns."""

    count: int
    run_stats: RunStats

    def __init__(self) -> None:
        self.count = 0
        self.run_stats = RunStats()

    def merge(self, other: ValueStats) -> None:
        self.count += other.count
        self.run_stats.merge(other.run_stats)


# --- Column Stats Base -------------------------------------------------------


class ColumnStats(ABC):
    """Base class for column statistics."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.total = 0

    @abstractmethod
    def update(self, values: np.ndarray) -> None:
        """Update stats with values from a chunk."""
        pass

    @abstractmethod
    def update_runs(self, runs: Iterable[Tuple[object, int]]) -> None:
        """Update run-length statistics."""
        pass

    @abstractmethod
    def merge(self, other: ColumnStats) -> None:
        """Merge stats from another collector."""
        pass

    @abstractmethod
    def finalize(self, percentiles: Sequence[float]) -> Dict[str, object]:
        """Return final statistics as a dictionary."""
        pass


class DiscreteColumnStats(ColumnStats):
    """Base for boolean and categorical columns."""

    def __init__(self, name: str, known_values: Sequence[object]) -> None:
        super().__init__(name)
        self.value_stats: Dict[object, ValueStats] = {
            val: ValueStats() for val in known_values
        }

    def _ensure_value(self, value: object) -> ValueStats:
        """Ensure a value exists in stats, creating if needed."""
        if value not in self.value_stats:
            self.value_stats[value] = ValueStats()
        return self.value_stats[value]

    def update_runs(self, runs: Iterable[Tuple[object, int]]) -> None:
        for value, length in runs:
            if value is None:
                continue
            canonical = _canonical_value(value)
            self._ensure_value(canonical).run_stats.add(length)

    def merge(self, other: DiscreteColumnStats) -> None:
        self.total += other.total
        for value, stats in other.value_stats.items():
            self._ensure_value(value).merge(stats)

    def _format_value_results(
        self, percentiles: Sequence[float]
    ) -> List[Dict[str, object]]:
        """Format value statistics for output."""
        if self.total == 0:
            return []

        results = []
        for value in sorted(self.value_stats.keys(), key=lambda v: (str(type(v)), v)):
            data = self.value_stats[value]
            percent = (data.count / self.total * 100.0) if self.total else 0.0
            results.append(
                {
                    "value": value,
                    "count": data.count,
                    "percent": percent,
                    "run_lengths": data.run_stats.summary(),
                    "run_percentiles": data.run_stats.percentile_values(percentiles),
                }
            )
        return results


class BooleanColumnStats(DiscreteColumnStats):
    def __init__(self, name: str) -> None:
        super().__init__(name, [0, 1])

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        rounded = np.rint(values)
        mask = ~np.isnan(rounded)
        ints = rounded[mask].astype(np.int8, copy=False)
        zeros = int((ints == 0).sum())
        ones = int((ints == 1).sum())
        self.value_stats[0].count += zeros
        self.value_stats[1].count += ones
        self.total += ints.size

    def finalize(self, percentiles: Sequence[float]) -> Dict[str, object]:
        return {
            "type": "boolean",
            "total_count": self.total,
            "values": self._format_value_results(percentiles),
        }


class CategoricalColumnStats(DiscreteColumnStats):
    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        self.total += values.size
        for raw_val, cnt in zip(*np.unique(values, return_counts=True)):
            val = _canonical_value(raw_val)
            self._ensure_value(val).count += int(cnt)

    def finalize(self, percentiles: Sequence[float]) -> Dict[str, object]:
        return {
            "type": "categorical",
            "total_count": self.total,
            "cardinality": len(self.value_stats),
            "values": self._format_value_results(percentiles),
        }


class ContinuousColumnStats(ColumnStats):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.sum_ = 0.0
        self.sum_sq = 0.0
        self.min_ = float("inf")
        self.max_ = float("-inf")
        self.run_stats = RunStats()
        self.percentiles: Dict[str, float] = {}

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        flat = values.astype(np.float64)
        self.total += flat.size
        self.sum_ += float(flat.sum())
        self.sum_sq += float((flat * flat).sum())
        current_min = float(flat.min())
        current_max = float(flat.max())
        if current_min < self.min_:
            self.min_ = current_min
        if current_max > self.max_:
            self.max_ = current_max

    def update_runs(self, runs: Iterable[Tuple[object, int]]) -> None:
        for _value, length in runs:
            self.run_stats.add(length)

    def merge(self, other: ContinuousColumnStats) -> None:
        if other.total == 0:
            self.run_stats.merge(other.run_stats)
            return
        if self.total == 0:
            self.min_ = other.min_
            self.max_ = other.max_
        else:
            if other.min_ < self.min_:
                self.min_ = other.min_
            if other.max_ > self.max_:
                self.max_ = other.max_
        self.total += other.total
        self.sum_ += other.sum_
        self.sum_sq += other.sum_sq
        self.run_stats.merge(other.run_stats)

    def set_percentiles(self, percentile_map: Dict[str, float]) -> None:
        self.percentiles = percentile_map

    def finalize(self, percentiles: Sequence[float]) -> Dict[str, object]:
        results: Dict[str, object] = {
            "type": "continuous",
            "total_count": self.total,
            "run_lengths": self.run_stats.summary(),
        }
        if self.total == 0:
            return results
        mean = self.sum_ / self.total
        variance = (self.sum_sq / self.total) - (mean * mean)
        variance = max(variance, 0.0)
        std = math.sqrt(variance)
        results.update(
            {
                "min": self.min_,
                "max": self.max_,
                "mean": mean,
                "std": std,
                "percentiles": self.percentiles,
                "run_percentiles": self.run_stats.percentile_values(percentiles),
            }
        )
        return results


# --- main computation --------------------------------------------------------


def _iter_episode_chunks(
    array: np.ndarray, chunk_size: int = 8192
) -> Iterable[np.ndarray]:
    total = array.shape[0]
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        yield np.asarray(array[start:end])


def _iter_runs(values: np.ndarray) -> Iterable[Tuple[object, int]]:
    if values.size == 0:
        return
    prev = _canonical_value(values[0])
    run = 1
    for raw_val in values[1:]:
        current = _canonical_value(raw_val)
        if _values_equal(current, prev):
            run += 1
        else:
            yield prev, run
            prev = current
            run = 1
    yield prev, run


@dataclass
class ColumnSignature:
    unique_values: set
    saw_more_than_limit: bool
    is_boolean_candidate: bool

    def __init__(self) -> None:
        self.unique_values = set()
        self.saw_more_than_limit = False
        self.is_boolean_candidate = True

    def update(self, values: np.ndarray, limit: int) -> None:
        if values.size == 0:
            return
        if self.is_boolean_candidate:
            if not np.isin(values, [0, 1]).all():
                self.is_boolean_candidate = False
        if not self.saw_more_than_limit:
            uniques = np.unique(values)
            for raw in uniques:
                self.unique_values.add(_canonical_value(raw))
            if len(self.unique_values) > limit:
                self.saw_more_than_limit = True

    def merge(self, other: ColumnSignature) -> None:
        self.unique_values.update(other.unique_values)
        self.saw_more_than_limit = self.saw_more_than_limit or other.saw_more_than_limit
        self.is_boolean_candidate = (
            self.is_boolean_candidate and other.is_boolean_candidate
        )


def _process_episodes(
    data_root: Path,
    episodes: List[EpisodeInfo],
    processor: callable,
) -> object:
    """Generic episode processor for workers."""
    shard_cache: Dict[int, object] = {}
    for episode in episodes:
        X = _open_episode_array(data_root, episode, shard_cache)
        processor(X, episode)
    return processor.get_result()


class ClassifyProcessor:
    """Processor for column classification."""

    def __init__(self, num_features: int, limit: int) -> None:
        self.signatures = [ColumnSignature() for _ in range(num_features)]
        self.num_features = num_features
        self.limit = limit

    def __call__(self, X: "zarr.Array", episode: EpisodeInfo) -> None:
        for chunk in _iter_episode_chunks(X):
            for col_idx in range(self.num_features):
                self.signatures[col_idx].update(chunk[:, col_idx], self.limit)

    def get_result(self) -> List[ColumnSignature]:
        return self.signatures


class StatsProcessor:
    """Processor for computing statistics."""

    def __init__(
        self,
        feature_names: Sequence[str],
        column_types: Sequence[str],
        unique_values: Sequence[set],
        stick_specs: Sequence[StickQuantizationSpec],
        shoulder_specs: Sequence[ShoulderQuantizationSpec],
    ) -> None:
        self.collectors = _create_collectors(feature_names, column_types, unique_values)
        self.num_features = len(feature_names)
        self.stick_specs = list(stick_specs)
        self.shoulder_specs = list(shoulder_specs)

        self._stick_offset = self.num_features
        for spec in self.stick_specs:
            known_values = list(range(spec.num_categories))
            self.collectors.append(CategoricalColumnStats(spec.name, known_values))

        self._shoulder_offset = self._stick_offset + len(self.stick_specs)
        shoulder_categories = list(range(len(SHOULDER_QUANTIZED)))
        for spec in self.shoulder_specs:
            self.collectors.append(
                CategoricalColumnStats(spec.name, shoulder_categories)
            )

    def __call__(self, X: "zarr.Array", episode: EpisodeInfo) -> None:
        data = np.asarray(X[:])
        for col_idx in range(self.num_features):
            values = data[:, col_idx]
            collector = self.collectors[col_idx]
            collector.update(values)
            collector.update_runs(_iter_runs(values))
        for offset, spec in enumerate(self.stick_specs):
            collector = self.collectors[self._stick_offset + offset]
            xy = data[:, [spec.x_idx, spec.y_idx]]
            quantized = _quantize_stick_indices(
                xy, palette=spec.palette, palette_norm=spec.palette_norm
            )
            collector.update(quantized)
            collector.update_runs(_iter_runs(quantized))
        for offset, spec in enumerate(self.shoulder_specs):
            collector = self.collectors[self._shoulder_offset + offset]
            analog = data[:, spec.source_idx]
            quantized = _quantize_shoulder_indices(analog)
            collector.update(quantized)
            collector.update_runs(_iter_runs(quantized))

    def get_result(self) -> List[ColumnStats]:
        return self.collectors


def _parallel_worker(args: Tuple[str, List[EpisodeInfo], str, object]) -> object:
    """Generic parallel worker."""
    data_root_str, episodes, processor_type, processor_args = args
    data_root = Path(data_root_str)

    if processor_type == "classify":
        num_features, limit = processor_args
        processor = ClassifyProcessor(num_features, limit)
    elif processor_type == "stats":
        (
            feature_names,
            column_types,
            unique_values,
            stick_specs,
            shoulder_specs,
        ) = processor_args
        processor = StatsProcessor(
            feature_names, column_types, unique_values, stick_specs, shoulder_specs
        )
    else:
        raise ValueError(f"Unknown processor type: {processor_type}")

    return _process_episodes(data_root, episodes, processor)


@dataclass
class DatasetProfile:
    column_types: List[str]
    unique_values: List[set]


def classify_columns(
    index: ValidationDatasetIndex, *, categorical_threshold: int = 20, workers: int = 1
) -> DatasetProfile:
    workers = _resolve_worker_count(workers)
    num_features = len(index.feature_names)
    signatures = [ColumnSignature() for _ in range(num_features)]

    total_eps = len(index.episodes)
    print(f"Classifying columns across {total_eps} episodes...", flush=True)
    effective_workers = max(1, min(workers, len(index.episodes)))

    if effective_workers == 1:
        with tqdm(
            total=total_eps, desc="classify", unit="episode", leave=False
        ) as pbar:
            processor = ClassifyProcessor(num_features, categorical_threshold)
            for episode in index.episodes:
                X, _ = index.open_episode_arrays(episode)
                processor(X, episode)
                pbar.update(1)
            signatures = processor.get_result()
    else:
        chunks = _split_into_chunks(index.episodes, effective_workers * 4)
        with tqdm(
            total=total_eps, desc="classify", unit="episode", leave=False
        ) as pbar:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                future_to_size: Dict[object, int] = {}
                for chunk in chunks:
                    future = executor.submit(
                        _parallel_worker,
                        (
                            str(index.data_dir),
                            chunk,
                            "classify",
                            (num_features, categorical_threshold),
                        ),
                    )
                    future_to_size[future] = len(chunk)
                for future in as_completed(future_to_size):
                    partial = future.result()
                    for idx in range(num_features):
                        signatures[idx].merge(partial[idx])
                    pbar.update(future_to_size[future])

    column_types: List[str] = []
    unique_values: List[set] = []
    for sig in signatures:
        values = sig.unique_values
        unique_values.append(set(values))
        non_nan_values = {v for v in values if v is not None}
        if (
            sig.is_boolean_candidate
            and non_nan_values.issubset({0, 1})
            and len(non_nan_values) <= 2
        ):
            column_types.append("boolean")
        elif not sig.saw_more_than_limit:
            column_types.append("categorical")
        else:
            column_types.append("continuous")

    return DatasetProfile(column_types, unique_values)


def _create_collectors(
    feature_names: Sequence[str],
    column_types: Sequence[str],
    unique_values: Sequence[set],
) -> List[ColumnStats]:
    collectors: List[ColumnStats] = []
    for idx, column_type in enumerate(column_types):
        name = feature_names[idx]
        if column_type == "boolean":
            collectors.append(BooleanColumnStats(name))
        elif column_type == "categorical":
            known_values = sorted(unique_values[idx])
            collectors.append(CategoricalColumnStats(name, known_values))
        else:
            collectors.append(ContinuousColumnStats(name))
    return collectors


def _percentile_worker(
    args: Tuple[str, List[EpisodeInfo], List[int], Sequence[float]],
) -> Dict[int, Dict[str, float]]:
    data_root_str, episodes, column_indices, percentiles = args
    data_root = Path(data_root_str)
    shard_cache: Dict[int, object] = {}
    results: Dict[int, Dict[str, float]] = {}
    for col_idx in column_indices:
        arrays: List[np.ndarray] = []
        for episode in episodes:
            X = _open_episode_array(data_root, episode, shard_cache)
            column_values = np.asarray(X[:, col_idx], dtype=np.float32)
            arrays.append(column_values)
        if arrays:
            combined = np.concatenate(arrays)
            percentile_values = np.percentile(combined, percentiles, method="linear")
            results[col_idx] = {
                str(p): float(v) for p, v in zip(percentiles, percentile_values)
            }
        else:
            results[col_idx] = {}
    return results


def compute_continuous_percentiles(
    index: ValidationDatasetIndex,
    column_indices: Sequence[int],
    percentiles: Sequence[float],
    *,
    workers: int,
) -> Dict[int, Dict[str, float]]:
    if not column_indices:
        return {}
    total_cols = len(column_indices)
    print(
        f"Computing exact percentiles for {total_cols} continuous columns...",
        flush=True,
    )
    effective_workers = max(1, min(_resolve_worker_count(workers), total_cols))
    columns = list(column_indices)

    if effective_workers == 1:
        with tqdm(
            total=total_cols, desc="percentiles", unit="column", leave=False
        ) as pbar:
            results = _percentile_worker(
                (str(index.data_dir), index.episodes, columns, percentiles)
            )
            pbar.update(total_cols)
        return results

    step = (total_cols + effective_workers - 1) // effective_workers
    chunks = [columns[start : start + step] for start in range(0, total_cols, step)]

    results: Dict[int, Dict[str, float]] = {}
    with tqdm(total=total_cols, desc="percentiles", unit="column", leave=False) as pbar:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_to_size: Dict[object, int] = {}
            for chunk in chunks:
                future = executor.submit(
                    _percentile_worker,
                    (str(index.data_dir), index.episodes, chunk, percentiles),
                )
                future_to_size[future] = len(chunk)
            for future in as_completed(future_to_size):
                results.update(future.result())
                pbar.update(future_to_size[future])
    return results


def compute_statistics(
    index: ValidationDatasetIndex,
    *,
    percentiles: Sequence[float],
    output_path: Path,
    workers: int = 1,
) -> None:
    worker_count = _resolve_worker_count(workers)
    profile = classify_columns(index, workers=worker_count)
    feature_names = index.feature_names

    feature_idx = {name: idx for idx, name in enumerate(feature_names)}
    stick_specs: List[StickQuantizationSpec] = []
    for column_name, x_name, y_name, palette, palette_norm in STICK_QUANTIZATION_CONFIG:
        try:
            x_idx = feature_idx[x_name]
            y_idx = feature_idx[y_name]
        except KeyError:
            continue
        spec = StickQuantizationSpec(column_name, x_idx, y_idx, palette, palette_norm)
        stick_specs.append(spec)
    shoulder_specs: List[ShoulderQuantizationSpec] = []
    for src_name, column_name in SHOULDER_QUANTIZATION_CONFIG.items():
        idx = feature_idx.get(src_name)
        if idx is None:
            continue
        shoulder_specs.append(ShoulderQuantizationSpec(column_name, idx))

    collector_names: List[str] = list(feature_names)
    collector_names.extend(spec.name for spec in stick_specs)
    collector_names.extend(spec.name for spec in shoulder_specs)

    master_processor = StatsProcessor(
        feature_names,
        profile.column_types,
        profile.unique_values,
        stick_specs,
        shoulder_specs,
    )
    collectors = master_processor.get_result()

    total_eps = len(index.episodes)
    print(f"Computing column statistics across {total_eps} episodes...", flush=True)
    effective_workers = max(1, min(worker_count, len(index.episodes)))

    if effective_workers == 1:
        with tqdm(total=total_eps, desc="stats", unit="episode", leave=False) as pbar:
            for episode in index.episodes:
                X, _ = index.open_episode_arrays(episode)
                master_processor(X, episode)
                pbar.update(1)
    else:
        chunks = _split_into_chunks(index.episodes, effective_workers * 4)
        with tqdm(total=total_eps, desc="stats", unit="episode", leave=False) as pbar:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                future_to_size: Dict[object, int] = {}
                for chunk in chunks:
                    future = executor.submit(
                        _parallel_worker,
                        (
                            str(index.data_dir),
                            chunk,
                            "stats",
                            (
                                feature_names,
                                profile.column_types,
                                profile.unique_values,
                                stick_specs,
                                shoulder_specs,
                            ),
                        ),
                    )
                    future_to_size[future] = len(chunk)
                for future in as_completed(future_to_size):
                    partial_collectors = future.result()
                    for idx in range(len(collectors)):
                        collectors[idx].merge(partial_collectors[idx])
                    pbar.update(future_to_size[future])

    # Compute percentiles for continuous columns
    continuous_indices = [
        idx for idx, typ in enumerate(profile.column_types) if typ == "continuous"
    ]
    percentile_maps = compute_continuous_percentiles(
        index,
        continuous_indices,
        percentiles,
        workers=worker_count,
    )
    for idx in continuous_indices:
        collectors[idx].set_percentiles(percentile_maps.get(idx, {}))

    results: Dict[str, object] = {"columns": {}}
    for name, collector in zip(collector_names, collectors):
        results["columns"][name] = collector.finalize(percentiles)

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[output] Wrote statistics to {output_path.resolve()}", flush=True)
    print("[output] Final statistics:", flush=True)
    print(_format_results_for_terminal(results), flush=True)


# --- presentation ------------------------------------------------------------


def _format_number(
    value: Optional[float], *, precision: int = 4, is_int: bool = False
) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if is_int or isinstance(value, (int, np.integer)):
        return f"{int(round(value)):,}"
    return f"{value:.{precision}f}"


def _format_percent(value: Optional[float], precision: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}%"


def _percentile_label(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    s = f"{value}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            if len(cell) > widths[idx]:
                widths[idx] = len(cell)

    def _fmt_row(row: Sequence[str]) -> str:
        return "  " + "  ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    parts = [_fmt_row(headers), _fmt_row(["-" * w for w in widths])]
    for row in rows:
        parts.append(_fmt_row(row))
    return "\n".join(parts)


def _format_run_summary(run_summary: Dict[str, Optional[float]]) -> List[str]:
    return [
        _format_number(run_summary.get("min"), is_int=True),
        _format_number(run_summary.get("max"), is_int=True),
        _format_number(run_summary.get("mean")),
        _format_number(run_summary.get("median")),
    ]


def _format_run_table(
    run_summary: Dict[str, Optional[float]], run_percentiles: Dict[str, Optional[float]]
) -> str:
    rows = [
        ["min", _format_number(run_summary.get("min"), is_int=True)],
        ["max", _format_number(run_summary.get("max"), is_int=True)],
        ["mean", _format_number(run_summary.get("mean"))],
        ["median", _format_number(run_summary.get("median"))],
    ]
    percentile_items = sorted(
        ((float(k), v) for k, v in (run_percentiles or {}).items()),
        key=lambda item: item[0],
    )
    for p, value in percentile_items:
        rows.append(
            [
                f"p{_percentile_label(p)}%",
                _format_number(value),
            ]
        )
    return _format_table(["metric", "value"], rows)


def _format_discrete_column(name: str, data: Dict[str, object], col_type: str) -> str:
    """Format boolean or categorical column output."""
    header_parts = [f"Column: {name} ({col_type})"]
    if col_type == "categorical":
        header_parts.append(f"cardinality={data.get('cardinality')}")
    header = ", ".join(header_parts)

    values = data.get("values", [])
    percentile_keys = sorted(
        {
            key
            for entry in values
            for key in (entry.get("run_percentiles", {}) or {}).keys()
        },
        key=lambda k: float(k),
    )

    rows = []
    for entry in values:
        run_summary = _format_run_summary(entry.get("run_lengths", {}))
        run_percentiles = entry.get("run_percentiles", {}) or {}
        percentile_values = [
            _format_number(run_percentiles.get(key)) for key in percentile_keys
        ]
        rows.append(
            [
                str(entry.get("value")),
                _format_number(entry.get("count"), is_int=True),
                _format_percent(entry.get("percent")),
                *run_summary,
                *percentile_values,
            ]
        )

    percentile_headers = [
        f"run_p{_percentile_label(float(key))}%" for key in percentile_keys
    ]
    table = _format_table(
        [
            "value",
            "count",
            "percent",
            "run_min",
            "run_max",
            "run_mean",
            "run_median",
            *percentile_headers,
        ],
        rows,
    )
    return f"{header}\n{table}"


def _format_continuous_column(name: str, data: Dict[str, object]) -> str:
    lines = [f"Column: {name} (continuous)"]
    stats_line = (
        "  stats: "
        f"min={_format_number(data.get('min'))}  max={_format_number(data.get('max'))}  "
        f"mean={_format_number(data.get('mean'))}  std={_format_number(data.get('std'))}"
    )
    lines.append(stats_line)
    percentile_map = data.get("percentiles", {}) or {}
    if percentile_map:
        ordered = sorted(
            ((float(k), v) for k, v in percentile_map.items()), key=lambda t: t[0]
        )
        pct_rows = [[f"{p:.1f}%", _format_number(val)] for p, val in ordered]
        lines.append("  percentiles:")
        lines.append(_format_table(["percentile", "value"], pct_rows))
    run_summary = data.get("run_lengths", {})
    run_percentiles = data.get("run_percentiles", {}) or {}
    lines.append("  run lengths:")
    lines.append(_format_run_table(run_summary, run_percentiles))
    return "\n".join(lines)


def _format_results_for_terminal(results: Dict[str, object]) -> str:
    lines: List[str] = []
    columns: Dict[str, object] = results.get("columns", {})
    for name in sorted(columns.keys()):
        data = columns[name]
        col_type = data.get("type")
        if col_type == "boolean":
            lines.append(_format_discrete_column(name, data, "boolean"))
        elif col_type == "categorical":
            lines.append(_format_discrete_column(name, data, "categorical"))
        elif col_type == "continuous":
            lines.append(_format_continuous_column(name, data))
        else:
            lines.append(f"Column: {name} (unknown type {col_type})")
        lines.append("")
    return "\n".join(lines).rstrip()


# --- CLI ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute descriptive statistics for validation dataset features."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Path to validation dataset root (default: validation_set)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation_statistics.json"),
        help="Output JSON file path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    data_root = args.data_root or (project_root / "validation_set")
    index = ValidationDatasetIndex(data_root)
    percentiles = list(DEFAULT_PERCENTILES)
    compute_statistics(
        index, percentiles=percentiles, output_path=args.output, workers=args.workers
    )


if __name__ == "__main__":
    main()
