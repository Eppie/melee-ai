"""Parallel processing utilities for statistics computation."""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

from tqdm import tqdm

T = TypeVar("T")
R = TypeVar("R")


def resolve_worker_count(requested: Optional[int]) -> int:
    """Resolve the number of workers to use.

    Args:
        requested: Requested number of workers. None or <= 0 means use CPU count.

    Returns:
        Number of workers to use.
    """
    if requested is None or requested <= 0:
        cpu_count = multiprocessing.cpu_count() or 1
        return max(1, cpu_count)
    return requested


def split_into_chunks(items: Sequence[T], max_chunks: int) -> List[List[T]]:
    """Split a sequence into roughly equal chunks.

    Args:
        items: Sequence to split.
        max_chunks: Maximum number of chunks.

    Returns:
        List of chunks.
    """
    if not items:
        return []
    max_chunks = max(1, min(max_chunks, len(items)))
    chunk_size = (len(items) + max_chunks - 1) // max_chunks
    chunks: List[List[T]] = []
    for start in range(0, len(items), chunk_size):
        end = min(len(items), start + chunk_size)
        chunks.append(list(items[start:end]))
    return chunks


def parallel_process_episodes(
    episodes: Sequence[T],
    worker_fn: Callable[[Sequence[T]], R],
    merge_fn: Callable[[R, R], R],
    initial_result: R,
    *,
    workers: int = 1,
    desc: str = "processing",
    chunk_multiplier: int = 4,
) -> R:
    """Process episodes in parallel with progress bar.

    Args:
        episodes: Sequence of episodes to process.
        worker_fn: Function to process a chunk of episodes. Takes sequence, returns result.
        merge_fn: Function to merge two results.
        initial_result: Initial result to merge into.
        workers: Number of worker processes.
        desc: Description for progress bar.
        chunk_multiplier: Multiplier for number of chunks per worker.

    Returns:
        Merged result from all workers.
    """
    effective_workers = max(1, min(resolve_worker_count(workers), len(episodes)))
    total = len(episodes)

    if effective_workers == 1:
        # Single-threaded processing
        with tqdm(total=total, desc=desc, unit="episode", leave=False) as pbar:
            result = worker_fn(episodes)
            pbar.update(total)
        return merge_fn(initial_result, result)

    # Multi-process
    chunks = split_into_chunks(episodes, effective_workers * chunk_multiplier)
    result = initial_result

    with tqdm(total=total, desc=desc, unit="episode", leave=False) as pbar:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_to_size: Dict[Any, int] = {}
            for chunk in chunks:
                future = executor.submit(worker_fn, chunk)
                future_to_size[future] = len(chunk)

            for future in as_completed(future_to_size):
                chunk_result = future.result()
                result = merge_fn(result, chunk_result)
                pbar.update(future_to_size[future])

    return result


def parallel_map(
    items: Sequence[T],
    worker_fn: Callable[[T], R],
    *,
    workers: int = 1,
    desc: str = "processing",
) -> List[R]:
    """Map a function over items in parallel.

    Args:
        items: Items to process.
        worker_fn: Function to apply to each item.
        workers: Number of worker processes.
        desc: Description for progress bar.

    Returns:
        List of results in same order as items.
    """
    effective_workers = max(1, min(resolve_worker_count(workers), len(items)))

    if effective_workers == 1:
        results = []
        with tqdm(total=len(items), desc=desc, unit="item", leave=False) as pbar:
            for item in items:
                results.append(worker_fn(item))
                pbar.update(1)
        return results

    results = [None] * len(items)
    with tqdm(total=len(items), desc=desc, unit="item", leave=False) as pbar:
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            future_to_idx: Dict[Any, int] = {}
            for idx, item in enumerate(items):
                future = executor.submit(worker_fn, item)
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()
                pbar.update(1)

    return results
