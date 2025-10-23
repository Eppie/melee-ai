#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import multiprocessing

import numpy as np

from tqdm import tqdm


DEFAULT_PERCENTILES: List[float] = [0.5, 1.0, 10.0, 25.0, 50.0, 90.0, 99.0, 99.5]


@dataclass(frozen=True)
class EpisodeInfo:
    episode_id: int
    shard_id: int
    num_frames: int


class ValidationDatasetIndex:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "meta.json"
        index_path = self.data_dir / "index.jsonl"
        if not meta_path.exists() or not index_path.exists():
            raise FileNotFoundError("Expected meta.json and index.jsonl in validation dataset root")

        with meta_path.open("r") as f:
            meta = json.load(f)

        schema = meta.get("schema", {})
        self.feature_names: List[str] = list(schema.get("features", []))
        self.target_names: List[str] = list(schema.get("targets", []))

        episodes: List[EpisodeInfo] = []
        with index_path.open("r") as f:
            for line in f:
                row = json.loads(line)
                episodes.append(EpisodeInfo(
                    episode_id=int(row["episode_id"]),
                    shard_id=int(row["shard_id"]),
                    num_frames=int(row.get("frames", 0)),
                ))
        self.episodes = episodes

        self._shard_cache: Dict[int, object] = {}

    def open_episode_arrays(self, episode: EpisodeInfo) -> Tuple["zarr.Array", Optional["zarr.Array"]]:
        import zarr

        shard_group = self._shard_cache.get(episode.shard_id)
        if shard_group is None:
            shard_path = self.data_dir / f"shard_{episode.shard_id:05d}.zarr"
            shard_group = zarr.open_group(str(shard_path), mode="r")
            self._shard_cache[episode.shard_id] = shard_group
        ep_group = shard_group[f"ep_{episode.episode_id:06d}"]
        X = ep_group["X"]
        Y = ep_group.get("Y")
        return X, Y


# --- helpers -----------------------------------------------------------------


def _split_into_chunks(items: Sequence[EpisodeInfo], max_chunks: int) -> List[List[EpisodeInfo]]:
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


def _open_episode_array(data_root: Path, episode: EpisodeInfo, shard_cache: Dict[int, object]) -> "zarr.Array":
    import zarr  # lazy import for worker compatibility

    group = shard_cache.get(episode.shard_id)
    if group is None:
        shard_path = data_root / f"shard_{episode.shard_id:05d}.zarr"
        group = zarr.open_group(str(shard_path), mode="r")
        shard_cache[episode.shard_id] = group
    ep_name = f"ep_{episode.episode_id:06d}"
    return group[ep_name]["X"]


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

    def percentile_values(self, percentiles: Sequence[float]) -> Dict[str, Optional[float]]:
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


def _median_from_counts(length_counts: Sequence[Tuple[int, int]], total_runs: int) -> float:
    """Compute an exact median from (length, count) pairs."""
    if total_runs == 0:
        return math.nan
    midpoint1 = (total_runs + 1) // 2
    midpoint2 = (total_runs + 2) // 2  # same as midpoint1 for odd totals

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
    # rank is zero-based index into the expanded run-length list.
    if rank <= 0:
        return float(length_counts[0][0])
    total_runs = cumulative_counts[-1][1]
    if rank >= total_runs - 1:
        return float(length_counts[-1][0])
    for length, cum_count in cumulative_counts:
        if rank < cum_count:
            return float(length)
    return float(length_counts[-1][0])


@dataclass
class BooleanValueStats:
    count: int
    run_stats: RunStats

    def __init__(self) -> None:
        self.count = 0
        self.run_stats = RunStats()

    def merge(self, other: BooleanValueStats) -> None:
        self.count += other.count
        self.run_stats.merge(other.run_stats)


@dataclass
class BooleanColumnStats:
    name: str
    value_stats: Dict[int, BooleanValueStats]
    total: int

    def __init__(self, name: str) -> None:
        self.name = name
        self.value_stats = {0: BooleanValueStats(), 1: BooleanValueStats()}
        self.total = 0

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

    def update_runs(self, runs: Iterable[Tuple[object, int]]) -> None:
        for value, length in runs:
            if value is None:
                continue
            key = int(value)
            if key in self.value_stats:
                self.value_stats[key].run_stats.add(length)

    def merge(self, other: "BooleanColumnStats") -> None:
        self.total += other.total
        for key in (0, 1):
            self.value_stats[key].merge(other.value_stats[key])

    def finalize(self, percentiles: Sequence[float]) -> Dict[str, object]:
        results: Dict[str, object] = {
            "type": "boolean",
            "total_count": self.total,
            "values": [],
        }
        if self.total == 0:
            return results
        for key in (0, 1):
            data = self.value_stats[key]
            percent = (data.count / self.total * 100.0) if self.total else 0.0
            results["values"].append({
                "value": key,
                "count": data.count,
                "percent": percent,
                "run_lengths": data.run_stats.summary(),
                "run_percentiles": data.run_stats.percentile_values(percentiles),
            })
        return results


@dataclass
class CategoricalValueStats:
    count: int
    run_stats: RunStats

    def __init__(self) -> None:
        self.count = 0
        self.run_stats = RunStats()

    def merge(self, other: "CategoricalValueStats") -> None:
        self.count += other.count
        self.run_stats.merge(other.run_stats)


@dataclass
class CategoricalColumnStats:
    name: str
    values: Dict[object, CategoricalValueStats]
    total: int

    def __init__(self, name: str, known_values: Sequence[object]) -> None:
        self.name = name
        self.values = {val: CategoricalValueStats() for val in known_values}
        self.total = 0

    def update(self, values: np.ndarray) -> None:
        if values.size == 0:
            return
        self.total += values.size
        for raw_val, cnt in zip(*np.unique(values, return_counts=True)):
            val = _canonical_value(raw_val)
            if val not in self.values:
                self.values[val] = CategoricalValueStats()
            self.values[val].count += int(cnt)

    def update_runs(self, runs: Iterable[Tuple[object, int]]) -> None:
        for value, length in runs:
            val = _canonical_value(value)
            if val not in self.values:
                self.values[val] = CategoricalValueStats()
            self.values[val].run_stats.add(length)

    def merge(self, other: "CategoricalColumnStats") -> None:
        self.total += other.total
        for value, stats in other.values.items():
            if value not in self.values:
                self.values[value] = CategoricalValueStats()
            self.values[value].merge(stats)

    def finalize(self, percentiles: Sequence[float]) -> Dict[str, object]:
        results: Dict[str, object] = {
            "type": "categorical",
            "total_count": self.total,
            "cardinality": len(self.values),
            "values": [],
        }
        if self.total == 0:
            return results
        for value in sorted(self.values.keys(), key=lambda v: (str(type(v)), v)):
            data = self.values[value]
            percent = (data.count / self.total * 100.0) if self.total else 0.0
            results["values"].append({
                "value": value,
                "count": data.count,
                "percent": percent,
                "run_lengths": data.run_stats.summary(),
                "run_percentiles": data.run_stats.percentile_values(percentiles),
            })
        return results


@dataclass
class ContinuousColumnStats:
    name: str
    total: int
    sum_: float
    sum_sq: float
    min_: float
    max_: float
    run_stats: RunStats
    percentiles: Dict[str, float]

    def __init__(self, name: str) -> None:
        self.name = name
        self.total = 0
        self.sum_ = 0.0
        self.sum_sq = 0.0
        self.min_ = float("inf")
        self.max_ = float("-inf")
        self.run_stats = RunStats()
        self.percentiles = {}

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

    def merge(self, other: "ContinuousColumnStats") -> None:
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
        results.update({
            "min": self.min_,
            "max": self.max_,
            "mean": mean,
            "std": std,
            "percentiles": self.percentiles,
            "run_percentiles": self.run_stats.percentile_values(percentiles),
        })
        return results


# --- main computation --------------------------------------------------------


def _iter_episode_chunks(array: np.ndarray, chunk_size: int = 8192) -> Iterable[np.ndarray]:
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

    def merge(self, other: "ColumnSignature") -> None:
        self.unique_values.update(other.unique_values)
        self.saw_more_than_limit = self.saw_more_than_limit or other.saw_more_than_limit
        self.is_boolean_candidate = self.is_boolean_candidate and other.is_boolean_candidate


def _classify_worker(args: Tuple[str, List[EpisodeInfo], int, int]) -> List[ColumnSignature]:
    data_root_str, episodes, num_features, limit = args
    data_root = Path(data_root_str)
    shard_cache: Dict[int, object] = {}
    signatures = [ColumnSignature() for _ in range(num_features)]
    for episode in episodes:
        X = _open_episode_array(data_root, episode, shard_cache)
        for chunk in _iter_episode_chunks(X):
            for col_idx in range(num_features):
                signatures[col_idx].update(chunk[:, col_idx], limit)
    return signatures


@dataclass
class DatasetProfile:
    column_types: List[str]
    unique_values: List[set]


def classify_columns(index: ValidationDatasetIndex, *, categorical_threshold: int = 20, workers: int = 1) -> DatasetProfile:
    workers = _resolve_worker_count(workers)
    num_features = len(index.feature_names)
    signatures = [ColumnSignature() for _ in range(num_features)]

    total_eps = len(index.episodes)
    print(f"Classifying columns across {total_eps} episodes...", flush=True)
    effective_workers = max(1, min(workers, len(index.episodes)))
    if effective_workers == 1:
        with tqdm(total=total_eps, desc="classify", unit="episode", leave=False) as pbar:
            for episode in index.episodes:
                X, _ = index.open_episode_arrays(episode)
                for chunk in _iter_episode_chunks(X):
                    for col_idx in range(num_features):
                        signatures[col_idx].update(chunk[:, col_idx], categorical_threshold)
                pbar.update(1)
    else:
        chunks = _split_into_chunks(index.episodes, effective_workers * 4)
        with tqdm(total=total_eps, desc="classify", unit="episode", leave=False) as pbar:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                future_to_size: Dict[object, int] = {}
                for chunk in chunks:
                    future = executor.submit(
                        _classify_worker,
                        (str(index.data_dir), chunk, num_features, categorical_threshold),
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
        if sig.is_boolean_candidate and non_nan_values.issubset({0, 1}) and len(non_nan_values) <= 2:
            column_types.append("boolean")
        elif not sig.saw_more_than_limit:
            column_types.append("categorical")
        else:
            column_types.append("continuous")

    return DatasetProfile(column_types, unique_values)


def _create_collectors(feature_names: Sequence[str], column_types: Sequence[str], unique_values: Sequence[set]) -> List[object]:
    collectors: List[object] = []
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


def _stats_worker(args: Tuple[str, List[EpisodeInfo], Sequence[str], Sequence[str], Sequence[set]]) -> List[object]:
    data_root_str, episodes, feature_names, column_types, unique_values = args
    data_root = Path(data_root_str)
    collectors = _create_collectors(feature_names, column_types, unique_values)
    num_features = len(feature_names)
    shard_cache: Dict[int, object] = {}
    for episode in episodes:
        X = _open_episode_array(data_root, episode, shard_cache)
        data = np.asarray(X[:])
        for col_idx in range(num_features):
            values = data[:, col_idx]
            collector = collectors[col_idx]
            collector.update(values)
            collector.update_runs(_iter_runs(values))
    return collectors


def _percentile_worker(args: Tuple[str, List[EpisodeInfo], List[int], Sequence[float]]) -> Dict[int, Dict[str, float]]:
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
            results[col_idx] = {str(p): float(v) for p, v in zip(percentiles, percentile_values)}
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
    print(f"Computing exact percentiles for {total_cols} continuous columns...", flush=True)
    effective_workers = max(1, min(_resolve_worker_count(workers), total_cols))
    columns = list(column_indices)

    if effective_workers == 1:
        with tqdm(total=total_cols, desc="percentiles", unit="column", leave=False) as pbar:
            results = _percentile_worker((str(index.data_dir), index.episodes, columns, percentiles))
            pbar.update(total_cols)
        return results

    step = (total_cols + effective_workers - 1) // effective_workers
    chunks = [columns[start:start + step] for start in range(0, total_cols, step)]

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


def compute_statistics(index: ValidationDatasetIndex, *, percentiles: Sequence[float], output_path: Path, workers: int = 1) -> None:
    worker_count = _resolve_worker_count(workers)
    profile = classify_columns(index, workers=worker_count)
    num_features = len(index.feature_names)

    collectors = _create_collectors(index.feature_names, profile.column_types, profile.unique_values)

    total_eps = len(index.episodes)
    print(f"Computing column statistics across {total_eps} episodes...", flush=True)
    effective_workers = max(1, min(worker_count, len(index.episodes)))
    if effective_workers == 1:
        with tqdm(total=total_eps, desc="stats", unit="episode", leave=False) as pbar:
            for episode in index.episodes:
                X, _ = index.open_episode_arrays(episode)
                data = np.asarray(X[:])
                for col_idx in range(num_features):
                    values = data[:, col_idx]
                    collector = collectors[col_idx]
                    collector.update(values)
                    collector.update_runs(_iter_runs(values))
                pbar.update(1)
    else:
        chunks = _split_into_chunks(index.episodes, effective_workers * 4)
        with tqdm(total=total_eps, desc="stats", unit="episode", leave=False) as pbar:
            with ProcessPoolExecutor(max_workers=effective_workers) as executor:
                future_to_size: Dict[object, int] = {}
                for chunk in chunks:
                    future = executor.submit(
                        _stats_worker,
                        (str(index.data_dir), chunk, index.feature_names, profile.column_types, profile.unique_values),
                    )
                    future_to_size[future] = len(chunk)
                for future in as_completed(future_to_size):
                    partial_collectors = future.result()
                    for idx in range(num_features):
                        collectors[idx].merge(partial_collectors[idx])
                    pbar.update(future_to_size[future])

    # Compute percentiles for continuous columns (serial or parallel per column)
    continuous_indices = [idx for idx, typ in enumerate(profile.column_types) if typ == "continuous"]
    percentile_maps = compute_continuous_percentiles(
        index,
        continuous_indices,
        percentiles,
        workers=worker_count,
    )
    for idx in continuous_indices:
        collectors[idx].set_percentiles(percentile_maps.get(idx, {}))

    results: Dict[str, object] = {"columns": {}}
    for idx, collector in enumerate(collectors):
        name = index.feature_names[idx]
        if isinstance(collector, BooleanColumnStats):
            results["columns"][name] = collector.finalize(percentiles)
        elif isinstance(collector, CategoricalColumnStats):
            results["columns"][name] = collector.finalize(percentiles)
        elif isinstance(collector, ContinuousColumnStats):
            results["columns"][name] = collector.finalize(percentiles)
        else:
            raise TypeError(f"Unknown collector type for column {name}")

    with output_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"[output] Wrote statistics to {output_path.resolve()}", flush=True)
    print("[output] Final statistics:", flush=True)
    print(_format_results_for_terminal(results), flush=True)


# --- presentation ------------------------------------------------------------


def _format_number(value: Optional[float], *, precision: int = 4, is_int: bool = False) -> str:
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


def _format_run_summary(run_summary: Dict[str, Optional[float]]) -> List[str]:
    return [
        _format_number(run_summary.get("min"), is_int=True),
        _format_number(run_summary.get("max"), is_int=True),
        _format_number(run_summary.get("mean")),
        _format_number(run_summary.get("median")),
    ]


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


def _format_run_table(run_summary: Dict[str, Optional[float]], run_percentiles: Dict[str, Optional[float]]) -> str:
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
        rows.append([
            f"p{_percentile_label(p)}%",
            _format_number(value),
        ])
    return _format_table(["metric", "value"], rows)


def _format_boolean_column(name: str, data: Dict[str, object]) -> str:
    header = f"Column: {name} (boolean)"
    values = data.get("values", [])
    percentile_keys = sorted({
        key
        for entry in values
        for key in (entry.get("run_percentiles", {}) or {}).keys()
    }, key=lambda k: float(k))

    rows = []
    for entry in values:
        run_summary = _format_run_summary(entry.get("run_lengths", {}))
        run_percentiles = entry.get("run_percentiles", {}) or {}
        percentile_values = [
            _format_number(run_percentiles.get(key))
            for key in percentile_keys
        ]
        rows.append([
            str(entry.get("value")),
            _format_number(entry.get("count"), is_int=True),
            _format_percent(entry.get("percent")),
            *run_summary,
            *percentile_values,
        ])
    percentile_headers = [
        f"run_p{_percentile_label(float(key))}%"
        for key in percentile_keys
    ]
    table = _format_table([
        "value",
        "count",
        "percent",
        "run_min",
        "run_max",
        "run_mean",
        "run_median",
        *percentile_headers,
    ], rows)
    return f"{header}\n{table}"


def _format_categorical_column(name: str, data: Dict[str, object]) -> str:
    cardinality = data.get("cardinality")
    header = f"Column: {name} (categorical, cardinality={cardinality})"
    values = data.get("values", [])
    percentile_keys = sorted({
        key
        for entry in values
        for key in (entry.get("run_percentiles", {}) or {}).keys()
    }, key=lambda k: float(k))

    rows = []
    for entry in values:
        run_summary = _format_run_summary(entry.get("run_lengths", {}))
        run_percentiles = entry.get("run_percentiles", {}) or {}
        percentile_values = [
            _format_number(run_percentiles.get(key))
            for key in percentile_keys
        ]
        rows.append([
            str(entry.get("value")),
            _format_number(entry.get("count"), is_int=True),
            _format_percent(entry.get("percent")),
            *run_summary,
            *percentile_values,
        ])
    percentile_headers = [
        f"run_p{_percentile_label(float(key))}%"
        for key in percentile_keys
    ]
    table = _format_table([
        "value",
        "count",
        "percent",
        "run_min",
        "run_max",
        "run_mean",
        "run_median",
        *percentile_headers,
    ], rows)
    return f"{header}\n{table}"


def _format_continuous_column(name: str, data: Dict[str, object]) -> str:
    lines = [
        f"Column: {name} (continuous)"
    ]
    stats_line = (
        "  stats: "
        f"min={_format_number(data.get('min'))}  max={_format_number(data.get('max'))}  "
        f"mean={_format_number(data.get('mean'))}  std={_format_number(data.get('std'))}"
    )
    lines.append(stats_line)
    percentile_map = data.get("percentiles", {}) or {}
    if percentile_map:
        ordered = sorted(((float(k), v) for k, v in percentile_map.items()), key=lambda t: t[0])
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
            lines.append(_format_boolean_column(name, data))
        elif col_type == "categorical":
            lines.append(_format_categorical_column(name, data))
        elif col_type == "continuous":
            lines.append(_format_continuous_column(name, data))
        else:
            lines.append(f"Column: {name} (unknown type {col_type})")
        lines.append("")
    return "\n".join(lines).rstrip()


# --- CLI ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute descriptive statistics for validation dataset features.")
    parser.add_argument("--data-root", type=Path, default=None, help="Path to validation dataset root (default: validation_set)")
    parser.add_argument("--output", type=Path, default=Path("validation_statistics.json"), help="Output JSON file path")
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU count)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    data_root = args.data_root or (project_root / "validation_set")
    index = ValidationDatasetIndex(data_root)
    percentiles = list(DEFAULT_PERCENTILES)
    compute_statistics(index, percentiles=percentiles, output_path=args.output, workers=args.workers)


if __name__ == "__main__":
    main()
