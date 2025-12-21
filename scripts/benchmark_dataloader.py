from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config import get_config, init_config  # noqa: E402
from window_dataset import (
    RandomWindowSampler,
    WindowDataset,
    worker_init_fn,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark window dataloader throughput."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("validation_set"),
        help="Path to dataset root (defaults to ./validation_set).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (defaults to train.batch_size).",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=256,
        help="Number of batches to iterate during the measurement.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride to use for RandomWindowSampler (defaults to train.stride).",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable pin_memory on the DataLoader (disabled by default).",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Load each episode fully into RAM on first access to bypass zarr I/O.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (defaults to 0 for single-process loading).",
    )
    parser.add_argument(
        "--start-method",
        type=str,
        default=None,
        choices=["fork", "spawn", "forkserver", None],
        help="torch.multiprocessing start method for workers (defaults to PyTorch's platform default).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Override DataLoader prefetch_factor (PyTorch default=2 when num_workers>0).",
    )
    return parser.parse_args()


def build_loader(args: argparse.Namespace) -> torch.utils.data.DataLoader:
    try:
        config = get_config()
    except RuntimeError:
        config = init_config()
    batch_size = args.batch_size or config.train.batch_size
    stride = args.stride or config.train.stride

    dataset = WindowDataset(
        args.dataset_root,
        in_memory=args.in_memory,
    )
    sampler = RandomWindowSampler(index=dataset.index, stride=stride)
    sampler.set_epoch(0)

    mp_ctx = None
    if args.num_workers and args.start_method:
        try:
            mp_ctx = torch.multiprocessing.get_context(args.start_method)
        except RuntimeError as exc:
            print(
                f"[benchmark] Requested start method '{args.start_method}' unavailable ({exc}); using default."
            )
            mp_ctx = None

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=(args.num_workers > 0),
        multiprocessing_context=mp_ctx,
        prefetch_factor=args.prefetch_factor if args.prefetch_factor else None,
    )
    return loader


def benchmark(loader: torch.utils.data.DataLoader, num_batches: int) -> None:
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")

    warmup_batches = min(5, num_batches // 10 or 1)
    total_batches = 0

    # Warmup
    for batch in loader:
        total_batches += 1
        if total_batches >= warmup_batches:
            break

    total_batches = 0
    start = time.perf_counter()
    last_report = start
    for batch in loader:
        total_batches += 1
        now = time.perf_counter()
        elapsed_iter = now - last_report
        if elapsed_iter > 0:
            iter_bps = 1.0 / elapsed_iter
            print(
                f"batch {total_batches}/{num_batches}: {iter_bps:.2f} batches/s",
                flush=True,
            )
        last_report = now
        if total_batches >= num_batches:
            break
    elapsed = time.perf_counter() - start

    batches_per_sec = total_batches / elapsed if elapsed > 0 else float("inf")
    samples_per_sec = (
        batches_per_sec * loader.batch_size if loader.batch_size else float("nan")
    )

    print(
        f"Benchmark results: {total_batches} batches in {elapsed:.3f}s "
        f"-> {batches_per_sec:.2f} batches/s "
        f"({samples_per_sec:.2f} samples/s)"
    )


def main() -> None:
    args = parse_args()
    loader = build_loader(args)
    benchmark(loader, args.num_batches)


if __name__ == "__main__":
    main()
