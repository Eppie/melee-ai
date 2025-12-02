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
