#!/usr/bin/env python3
"""
Comprehensive data loader profiling script.

Measures throughput, latency distribution, cache effects, and chunk boundary statistics
to determine if data loading is a bottleneck and where optimizations should focus.

Usage:
    python scripts/profile_dataloader.py --dataset-root /path/to/dataset
    python scripts/profile_dataloader.py --dataset-root validation_set --num-workers 0 4 8 16 24
    python scripts/profile_dataloader.py --compare-cache  # Compare cold vs warm cache
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config import get_config, init_config
from window_dataset import RandomWindowSampler, WindowDataset, worker_init_fn


@dataclass
class BatchTiming:
    """Timing for a single batch."""
    batch_idx: int
    latency_ms: float
    batch_size: int
    bytes_loaded: int


@dataclass
class ProfileResult:
    """Results from a profiling run."""
    config_name: str
    num_workers: int
    prefetch_factor: Optional[int]
    in_memory: bool
    pin_memory: bool

    num_batches: int
    total_samples: int
    total_bytes: int
    total_time_sec: float

    batch_timings: List[BatchTiming] = field(default_factory=list)

    @property
    def batches_per_sec(self) -> float:
        return self.num_batches / self.total_time_sec if self.total_time_sec > 0 else 0

    @property
    def samples_per_sec(self) -> float:
        return self.total_samples / self.total_time_sec if self.total_time_sec > 0 else 0

    @property
    def mb_per_sec(self) -> float:
        return (self.total_bytes / 1e6) / self.total_time_sec if self.total_time_sec > 0 else 0

    @property
    def latencies_ms(self) -> np.ndarray:
        return np.array([t.latency_ms for t in self.batch_timings])

    def latency_percentile(self, p: float) -> float:
        lats = self.latencies_ms
        return float(np.percentile(lats, p)) if len(lats) > 0 else 0

    def summary_dict(self) -> Dict:
        lats = self.latencies_ms
        return {
            "config": self.config_name,
            "num_workers": self.num_workers,
            "prefetch_factor": self.prefetch_factor,
            "in_memory": self.in_memory,
            "pin_memory": self.pin_memory,
            "num_batches": self.num_batches,
            "total_samples": self.total_samples,
            "total_mb": self.total_bytes / 1e6,
            "total_time_sec": round(self.total_time_sec, 3),
            "batches_per_sec": round(self.batches_per_sec, 2),
            "samples_per_sec": round(self.samples_per_sec, 2),
            "mb_per_sec": round(self.mb_per_sec, 2),
            "latency_ms": {
                "min": round(float(lats.min()), 2) if len(lats) > 0 else None,
                "p50": round(self.latency_percentile(50), 2),
                "p95": round(self.latency_percentile(95), 2),
                "p99": round(self.latency_percentile(99), 2),
                "max": round(float(lats.max()), 2) if len(lats) > 0 else None,
            },
        }


def get_batch_bytes(batch: Dict[str, torch.Tensor]) -> int:
    """Calculate total bytes in a batch."""
    total = 0
    for key, tensor in batch.items():
        if isinstance(tensor, torch.Tensor):
            total += tensor.numel() * tensor.element_size()
    return total


def analyze_chunk_boundaries(
    dataset: WindowDataset,
    chunk_frames: int = 512,
    sample_size: int = 10000
) -> Dict:
    """
    Analyze what fraction of windows cross chunk boundaries.

    With chunk_frames=512 and seq_len=256:
    - Windows starting at offsets 0-256 within a chunk are fully contained
    - Windows starting at offsets 257-511 span two chunks
    """
    seq_len = dataset.index.seq_len

    # Sample random windows
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        len(dataset),
        size=min(sample_size, len(dataset)),
        replace=False
    )

    contained = 0
    crossing = 0

    for idx in sample_indices:
        ep_idx, offset = dataset.index.window_to_episode(int(idx))

        # Determine chunk boundaries
        # Window spans frames [offset, offset + seq_len - 1]
        start_chunk = offset // chunk_frames
        end_chunk = (offset + seq_len - 1) // chunk_frames

        if start_chunk == end_chunk:
            contained += 1
        else:
            crossing += 1

    total = contained + crossing
    return {
        "chunk_frames": chunk_frames,
        "seq_len": seq_len,
        "sample_size": total,
        "contained_in_one_chunk": contained,
        "crossing_boundary": crossing,
        "pct_contained": round(100 * contained / total, 2) if total > 0 else 0,
        "pct_crossing": round(100 * crossing / total, 2) if total > 0 else 0,
        "expected_chunks_per_window": round((contained + 2 * crossing) / total, 3) if total > 0 else 0,
    }


def profile_dataloader(
    dataset_root: Path,
    num_workers: int = 0,
    batch_size: int = 256,
    stride: int = 8,
    num_batches: int = 100,
    warmup_batches: int = 5,
    prefetch_factor: Optional[int] = None,
    in_memory: bool = False,
    pin_memory: bool = False,
    start_method: Optional[str] = None,
    config_name: str = "default",
) -> ProfileResult:
    """Run a single profiling configuration."""

    dataset = WindowDataset(dataset_root, in_memory=in_memory)
    sampler = RandomWindowSampler(index=dataset.index, stride=stride)
    sampler.set_epoch(0)

    # Build multiprocessing context if needed
    mp_ctx = None
    if num_workers > 0 and start_method:
        try:
            mp_ctx = torch.multiprocessing.get_context(start_method)
        except RuntimeError:
            mp_ctx = None

    # Determine prefetch factor
    pf = prefetch_factor
    if pf is None and num_workers > 0:
        pf = 2  # PyTorch default

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        persistent_workers=(num_workers > 0),
        multiprocessing_context=mp_ctx,
        prefetch_factor=pf if num_workers > 0 else None,
    )

    batch_timings: List[BatchTiming] = []
    total_samples = 0
    total_bytes = 0
    batch_count = 0

    # Warmup
    warmup_iter = iter(loader)
    for _ in range(warmup_batches):
        try:
            batch = next(warmup_iter)
        except StopIteration:
            break

    # Timed run
    start_time = time.perf_counter()
    last_time = start_time

    for batch in loader:
        now = time.perf_counter()
        latency_ms = (now - last_time) * 1000

        batch_bytes = get_batch_bytes(batch)
        batch_size_actual = batch["X"].shape[0]

        batch_timings.append(BatchTiming(
            batch_idx=batch_count,
            latency_ms=latency_ms,
            batch_size=batch_size_actual,
            bytes_loaded=batch_bytes,
        ))

        total_samples += batch_size_actual
        total_bytes += batch_bytes
        batch_count += 1

        last_time = now

        if batch_count >= num_batches:
            break

    total_time = time.perf_counter() - start_time

    # Clean up to free memory
    del loader, dataset, sampler
    gc.collect()

    return ProfileResult(
        config_name=config_name,
        num_workers=num_workers,
        prefetch_factor=pf,
        in_memory=in_memory,
        pin_memory=pin_memory,
        num_batches=batch_count,
        total_samples=total_samples,
        total_bytes=total_bytes,
        total_time_sec=total_time,
        batch_timings=batch_timings,
    )


def print_result(result: ProfileResult, verbose: bool = False) -> None:
    """Print a single result."""
    summary = result.summary_dict()

    print(f"\n{'='*60}")
    print(f"Configuration: {summary['config']}")
    print(f"{'='*60}")
    print(f"  num_workers:    {summary['num_workers']}")
    print(f"  prefetch_factor: {summary['prefetch_factor']}")
    print(f"  in_memory:      {summary['in_memory']}")
    print(f"  pin_memory:     {summary['pin_memory']}")
    print()
    print(f"Throughput:")
    print(f"  Batches/sec:    {summary['batches_per_sec']:,.2f}")
    print(f"  Samples/sec:    {summary['samples_per_sec']:,.2f}")
    print(f"  MB/sec:         {summary['mb_per_sec']:,.2f}")
    print()
    print(f"Latency (ms per batch):")
    lat = summary['latency_ms']
    print(f"  min:  {lat['min']:>8.2f}")
    print(f"  p50:  {lat['p50']:>8.2f}")
    print(f"  p95:  {lat['p95']:>8.2f}")
    print(f"  p99:  {lat['p99']:>8.2f}")
    print(f"  max:  {lat['max']:>8.2f}")
    print()
    print(f"Total: {summary['num_batches']} batches, {summary['total_samples']:,} samples, "
          f"{summary['total_mb']:.1f} MB in {summary['total_time_sec']:.2f}s")

    if verbose:
        # Print per-batch timings
        print("\nPer-batch latencies (first 20):")
        for t in result.batch_timings[:20]:
            print(f"  batch {t.batch_idx:3d}: {t.latency_ms:7.2f}ms")


def compare_configurations(
    dataset_root: Path,
    worker_counts: List[int],
    batch_size: int,
    stride: int,
    num_batches: int,
    in_memory: bool = False,
    pin_memory: bool = False,
    prefetch_factor: Optional[int] = None,
) -> List[ProfileResult]:
    """Compare different worker configurations."""
    results = []

    for nw in worker_counts:
        config_name = f"workers={nw}"
        if in_memory:
            config_name += ",in_memory"
        if pin_memory:
            config_name += ",pinned"

        print(f"\nProfiling: {config_name}...")
        result = profile_dataloader(
            dataset_root=dataset_root,
            num_workers=nw,
            batch_size=batch_size,
            stride=stride,
            num_batches=num_batches,
            prefetch_factor=prefetch_factor,
            in_memory=in_memory,
            pin_memory=pin_memory,
            config_name=config_name,
        )
        results.append(result)
        print_result(result)

    return results


def compare_cache_warmth(
    dataset_root: Path,
    num_workers: int,
    batch_size: int,
    stride: int,
    num_batches: int,
    prefetch_factor: Optional[int] = None,
) -> Tuple[ProfileResult, ProfileResult]:
    """Compare cold cache (first pass) vs warm cache (second pass)."""

    # First pass - cold cache
    print("\n" + "="*60)
    print("COLD CACHE (first pass through data)")
    print("="*60)

    # Clear page cache if possible (requires sudo, so we skip)
    # Instead, we just note that the first run might have partial cache

    cold_result = profile_dataloader(
        dataset_root=dataset_root,
        num_workers=num_workers,
        batch_size=batch_size,
        stride=stride,
        num_batches=num_batches,
        prefetch_factor=prefetch_factor,
        config_name="cold_cache",
    )
    print_result(cold_result)

    # Second pass - warm cache (same data should be in OS page cache)
    print("\n" + "="*60)
    print("WARM CACHE (second pass - data should be cached)")
    print("="*60)

    warm_result = profile_dataloader(
        dataset_root=dataset_root,
        num_workers=num_workers,
        batch_size=batch_size,
        stride=stride,
        num_batches=num_batches,
        prefetch_factor=prefetch_factor,
        config_name="warm_cache",
    )
    print_result(warm_result)

    # Summary comparison
    print("\n" + "="*60)
    print("CACHE WARMTH COMPARISON")
    print("="*60)
    speedup = warm_result.samples_per_sec / cold_result.samples_per_sec if cold_result.samples_per_sec > 0 else 0
    print(f"  Cold cache: {cold_result.samples_per_sec:,.0f} samples/sec")
    print(f"  Warm cache: {warm_result.samples_per_sec:,.0f} samples/sec")
    print(f"  Speedup:    {speedup:.2f}x")

    return cold_result, warm_result


def print_comparison_table(results: List[ProfileResult]) -> None:
    """Print a comparison table of all results."""
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(f"{'Config':<30} {'Samples/s':>12} {'MB/s':>10} {'p50 (ms)':>10} {'p99 (ms)':>10}")
    print("-"*80)

    for r in results:
        print(f"{r.config_name:<30} {r.samples_per_sec:>12,.0f} {r.mb_per_sec:>10.1f} "
              f"{r.latency_percentile(50):>10.2f} {r.latency_percentile(99):>10.2f}")


def estimate_training_requirements(
    model_batch_time_ms: float,
    batch_size: int,
    result: ProfileResult,
) -> None:
    """
    Estimate if data loading can keep up with training.

    Args:
        model_batch_time_ms: Estimated time for forward+backward pass in ms
        batch_size: Batch size
        result: Profiling result to compare against
    """
    print("\n" + "="*60)
    print("TRAINING BOTTLENECK ANALYSIS")
    print("="*60)

    data_load_time_ms = result.latency_percentile(50)

    print(f"  Data loading p50:     {data_load_time_ms:.2f} ms/batch")
    print(f"  Model compute (est):  {model_batch_time_ms:.2f} ms/batch")
    print()

    if data_load_time_ms > model_batch_time_ms:
        overhead_pct = 100 * (data_load_time_ms - model_batch_time_ms) / model_batch_time_ms
        print(f"  STATUS: DATA LOADING IS THE BOTTLENECK")
        print(f"  Data loading is {overhead_pct:.1f}% slower than model compute")
        print(f"  Consider: more workers, in_memory mode, or format changes")
    else:
        headroom_pct = 100 * (model_batch_time_ms - data_load_time_ms) / model_batch_time_ms
        print(f"  STATUS: GPU COMPUTE IS THE BOTTLENECK (good!)")
        print(f"  Data loading has {headroom_pct:.1f}% headroom")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile data loader throughput and identify bottlenecks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic profiling with default settings
  python scripts/profile_dataloader.py --dataset-root validation_set

  # Compare different worker counts
  python scripts/profile_dataloader.py --dataset-root validation_set --num-workers 0 4 8 16 24

  # Test cache effectiveness
  python scripts/profile_dataloader.py --dataset-root validation_set --compare-cache

  # Full profiling with all options
  python scripts/profile_dataloader.py --dataset-root validation_set \\
      --num-workers 0 8 24 --num-batches 200 --compare-cache --analyze-chunks
        """
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to dataset root directory containing zarr shards.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        nargs="+",
        default=[0],
        help="Number of DataLoader workers to test. Can specify multiple values.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=8,
        help="Stride for RandomWindowSampler (default: 8).",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=100,
        help="Number of batches to profile (default: 100).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Prefetch factor (default: PyTorch default of 2).",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Cache episodes in memory on first access.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Use pinned memory for faster GPU transfer.",
    )
    parser.add_argument(
        "--compare-cache",
        action="store_true",
        help="Compare cold vs warm cache performance.",
    )
    parser.add_argument(
        "--analyze-chunks",
        action="store_true",
        help="Analyze chunk boundary crossing statistics.",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=512,
        help="Chunk size in frames for boundary analysis (default: 512).",
    )
    parser.add_argument(
        "--model-batch-ms",
        type=float,
        default=None,
        help="Estimated model batch time in ms for bottleneck analysis.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed per-batch timings.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Save results to JSON file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset_root.exists():
        print(f"Error: Dataset root does not exist: {args.dataset_root}")
        sys.exit(1)

    print("="*60)
    print("DATA LOADER PROFILER")
    print("="*60)
    print(f"Dataset:     {args.dataset_root}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Stride:      {args.stride}")
    print(f"Num batches: {args.num_batches}")
    print(f"Workers:     {args.num_workers}")

    all_results: List[ProfileResult] = []

    # Analyze chunk boundaries if requested
    if args.analyze_chunks:
        print("\n" + "="*60)
        print("CHUNK BOUNDARY ANALYSIS")
        print("="*60)

        dataset = WindowDataset(args.dataset_root, in_memory=False)
        chunk_stats = analyze_chunk_boundaries(
            dataset,
            chunk_frames=args.chunk_frames,
            sample_size=10000
        )

        print(f"  chunk_frames:              {chunk_stats['chunk_frames']}")
        print(f"  seq_len:                   {chunk_stats['seq_len']}")
        print(f"  sample_size:               {chunk_stats['sample_size']}")
        print(f"  contained_in_one_chunk:    {chunk_stats['contained_in_one_chunk']} ({chunk_stats['pct_contained']}%)")
        print(f"  crossing_boundary:         {chunk_stats['crossing_boundary']} ({chunk_stats['pct_crossing']}%)")
        print(f"  expected_chunks_per_window: {chunk_stats['expected_chunks_per_window']}")

        del dataset
        gc.collect()

    # Compare cache warmth if requested
    if args.compare_cache:
        cold, warm = compare_cache_warmth(
            dataset_root=args.dataset_root,
            num_workers=args.num_workers[0],  # Use first worker count
            batch_size=args.batch_size,
            stride=args.stride,
            num_batches=args.num_batches,
            prefetch_factor=args.prefetch_factor,
        )
        all_results.extend([cold, warm])

    # Compare worker configurations
    if len(args.num_workers) > 1 or not args.compare_cache:
        results = compare_configurations(
            dataset_root=args.dataset_root,
            worker_counts=args.num_workers,
            batch_size=args.batch_size,
            stride=args.stride,
            num_batches=args.num_batches,
            in_memory=args.in_memory,
            pin_memory=args.pin_memory,
            prefetch_factor=args.prefetch_factor,
        )
        all_results.extend(results)

    # Print comparison table
    if len(all_results) > 1:
        print_comparison_table(all_results)

    # Bottleneck analysis
    if args.model_batch_ms and all_results:
        # Use the best result for analysis
        best_result = max(all_results, key=lambda r: r.samples_per_sec)
        estimate_training_requirements(
            model_batch_time_ms=args.model_batch_ms,
            batch_size=args.batch_size,
            result=best_result,
        )

    # Save to JSON if requested
    if args.output_json:
        output_data = {
            "config": {
                "dataset_root": str(args.dataset_root),
                "batch_size": args.batch_size,
                "stride": args.stride,
                "num_batches": args.num_batches,
            },
            "results": [r.summary_dict() for r in all_results],
        }

        if args.analyze_chunks:
            output_data["chunk_analysis"] = chunk_stats

        with open(args.output_json, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    print("\n" + "="*60)
    print("PROFILING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
