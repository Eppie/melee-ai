"""Instrumentation for data loading performance metrics."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

import numpy as np


@dataclass
class DataLoadingMetrics:
    """Track data loading performance metrics to identify bottlenecks.

    Metrics tracked:
    - Chunk load time: Time to load episodes into shared memory
    - Batch preparation time: Time from Dataset.__getitem__ to GPU
    - GPU idle time: Time between batch arrivals (estimated)
    - Iteration timing: Overall iteration speed

    Usage:
        metrics = DataLoadingMetrics()

        # Record chunk loading
        start = time.time()
        load_chunks([0, 1])
        metrics.record_chunk_load(time.time() - start, iteration=0)

        # Record batch timing
        metrics.record_batch_start()
        batch = prepare_batch(...)
        metrics.record_batch_prep(time.time() - batch_start)

        # Get summary for logging
        summary = metrics.get_summary()
    """

    # Chunk loading metrics
    chunk_load_times: List[float] = field(default_factory=list)
    chunk_load_iterations: List[int] = field(default_factory=list)

    # Batch preparation metrics (rolling window)
    batch_prep_times: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    # GPU idle time estimation (time between batches)
    gpu_idle_times: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    last_batch_end_time: float = 0.0

    # Iteration timing
    iteration_start_times: Deque[float] = field(
        default_factory=lambda: deque(maxlen=100)
    )
    iteration_durations: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def record_chunk_load(self, duration_s: float, iteration: int) -> None:
        """Record time taken to load a chunk.

        Args:
            duration_s: Time in seconds to load the chunk
            iteration: Iteration number (for tracking when chunks are loaded)
        """
        self.chunk_load_times.append(duration_s)
        self.chunk_load_iterations.append(iteration)

    def record_batch_start(self) -> None:
        """Record the start of batch processing.

        Call this before fetching the next batch from DataLoader.
        This helps estimate GPU idle time by measuring gaps between batches.
        """
        now = time.time()

        # Calculate idle time since last batch
        if self.last_batch_end_time > 0:
            idle_time = now - self.last_batch_end_time
            # Only record if positive (sanity check)
            if idle_time > 0:
                self.gpu_idle_times.append(idle_time)

        # Record iteration start
        self.iteration_start_times.append(now)

    def record_batch_end(self) -> None:
        """Record the end of batch processing (after GPU forward/backward).

        Call this after the training step completes to mark when the next
        batch can be fetched.
        """
        now = time.time()
        self.last_batch_end_time = now

        # Calculate iteration duration
        if len(self.iteration_start_times) > 0:
            duration = now - self.iteration_start_times[-1]
            self.iteration_durations.append(duration)

    def record_batch_prep(self, duration_s: float) -> None:
        """Record time taken to prepare a batch (data loading + transfer to GPU).

        Args:
            duration_s: Time in seconds from DataLoader yield to GPU transfer complete
        """
        self.batch_prep_times.append(duration_s)

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics for logging.

        Returns:
            Dictionary with metrics suitable for wandb/tensorboard logging
        """
        summary = {}

        # Chunk loading statistics
        if self.chunk_load_times:
            summary["dataloader/chunk_load_mean_s"] = float(
                np.mean(self.chunk_load_times)
            )
            summary["dataloader/chunk_load_max_s"] = float(
                np.max(self.chunk_load_times)
            )
            summary["dataloader/chunk_load_min_s"] = float(
                np.min(self.chunk_load_times)
            )
            summary["dataloader/chunk_load_total_s"] = float(
                np.sum(self.chunk_load_times)
            )
            summary["dataloader/num_chunks_loaded"] = len(self.chunk_load_times)

        # Batch preparation statistics
        if self.batch_prep_times:
            summary["dataloader/batch_prep_mean_ms"] = float(
                np.mean(self.batch_prep_times) * 1000
            )
            summary["dataloader/batch_prep_p50_ms"] = float(
                np.percentile(self.batch_prep_times, 50) * 1000
            )
            summary["dataloader/batch_prep_p95_ms"] = float(
                np.percentile(self.batch_prep_times, 95) * 1000
            )
            summary["dataloader/batch_prep_max_ms"] = float(
                np.max(self.batch_prep_times) * 1000
            )

        # GPU idle time statistics
        if len(self.gpu_idle_times) > 0:
            summary["dataloader/gpu_idle_mean_ms"] = float(
                np.mean(self.gpu_idle_times) * 1000
            )
            summary["dataloader/gpu_idle_p50_ms"] = float(
                np.percentile(self.gpu_idle_times, 50) * 1000
            )
            summary["dataloader/gpu_idle_p95_ms"] = float(
                np.percentile(self.gpu_idle_times, 95) * 1000
            )
            summary["dataloader/gpu_idle_max_ms"] = float(
                np.max(self.gpu_idle_times) * 1000
            )

            # Calculate GPU utilization estimate
            # GPU is "idle" waiting for data if idle_time is significant compared to iteration time
            if len(self.iteration_durations) > 0:
                avg_idle = float(np.mean(self.gpu_idle_times))
                avg_iter = float(np.mean(self.iteration_durations))
                gpu_utilization = max(0.0, 1.0 - (avg_idle / (avg_iter + 1e-9)))
                summary["dataloader/gpu_utilization_estimate"] = gpu_utilization

        # Iteration timing
        if len(self.iteration_durations) > 0:
            summary["dataloader/iteration_mean_ms"] = float(
                np.mean(self.iteration_durations) * 1000
            )
            summary["dataloader/iteration_p95_ms"] = float(
                np.percentile(self.iteration_durations, 95) * 1000
            )
            summary["dataloader/batches_per_second"] = float(
                1.0 / (np.mean(self.iteration_durations) + 1e-9)
            )

        return summary

    def reset_epoch(self) -> None:
        """Reset epoch-level metrics (keep chunk load history)."""
        self.batch_prep_times.clear()
        self.gpu_idle_times.clear()
        self.iteration_start_times.clear()
        self.iteration_durations.clear()
        self.last_batch_end_time = 0.0

    def get_chunk_load_report(self) -> str:
        """Generate a human-readable report of chunk loading times.

        Returns:
            Formatted string summarizing chunk loading performance
        """
        if not self.chunk_load_times:
            return "No chunks loaded yet."

        lines = [
            f"Chunk Loading Report:",
            f"  Total chunks loaded: {len(self.chunk_load_times)}",
            f"  Mean load time: {np.mean(self.chunk_load_times):.2f}s",
            f"  Min load time: {np.min(self.chunk_load_times):.2f}s",
            f"  Max load time: {np.max(self.chunk_load_times):.2f}s",
            f"  Total time spent loading: {np.sum(self.chunk_load_times):.2f}s",
        ]

        # Show slowest chunks
        if len(self.chunk_load_times) > 3:
            slowest_idx = np.argsort(self.chunk_load_times)[-3:]
            lines.append("  Slowest chunks:")
            for idx in reversed(slowest_idx):
                iteration = self.chunk_load_iterations[idx]
                duration = self.chunk_load_times[idx]
                lines.append(f"    Iteration {iteration}: {duration:.2f}s")

        return "\n".join(lines)
