#!/usr/bin/env python3
"""Utility to print DataLoader worker RSS usage."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.profiler import ProfilerActivity, profile

from config import Config
from window_dataset import make_dataloader


def _read_rss_kb(pid: int) -> Optional[int]:
    """Read VmRSS (kB) from /proc/<pid>/status."""
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return None

    for line in status_path.read_text().splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1])
    return None


def _format_mb(kb: Optional[int]) -> str:
    if kb is None:
        return "n/a"
    return f"{kb / 1024:.2f} MB"


def _capture_worker_pids(
    loader_iter: torch.utils.data._utils.fetch._BaseDataLoaderIter,
) -> List[Tuple[int, int]]:
    """Return (worker_id, pid) pairs from the loader iterator."""
    workers = getattr(loader_iter, "_workers", None)
    if not workers:
        return []

    pairs: List[Tuple[int, int]] = []
    for worker_id, proc in enumerate(workers):
        if proc is None or proc.pid is None:
            continue
        pairs.append((worker_id, proc.pid))
    return pairs


def _largest_smaps_entries(pid: int, top_n: int = 5) -> List[Tuple[str, int, int]]:
    """Return the top-N mappings by RSS (kB) for the given process."""
    smaps_path = Path(f"/proc/{pid}/smaps")
    if not smaps_path.exists():
        return []

    entries: List[Tuple[str, int, int]] = []
    current_path = "[unknown]"
    current_rss = 0
    current_size = 0

    def flush() -> None:
        if current_path:
            entries.append((current_path, current_rss, current_size))

    with smaps_path.open("r") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped[0].isdigit() or stripped[0] in "abcdef":
                # New mapping header (e.g., '7f...-7f... perms offset dev inode path')
                flush()
                parts = stripped.split()
                if len(parts) >= 6:
                    maybe_path = parts[-1]
                    current_path = maybe_path
                else:
                    current_path = "[anon]"
                current_rss = 0
                current_size = 0
                continue
            if stripped.startswith("Rss:"):
                _, value, *_ = stripped.split()
                if value.isdigit():
                    current_rss += int(value)
                continue
            if stripped.startswith("Size:"):
                _, value, *_ = stripped.split()
                if value.isdigit():
                    current_size += int(value)
                continue
    flush()

    entries.sort(key=lambda x: x[1], reverse=True)
    return entries[:top_n]


def _read_smaps_rollup(pid: int) -> Dict[str, int]:
    """Parse /proc/<pid>/smaps_rollup into a dict of metric -> kB."""
    rollup_path = Path(f"/proc/{pid}/smaps_rollup")
    if not rollup_path.exists():
        return {}
    stats: Dict[str, int] = {}
    with rollup_path.open("r") as fh:
        for line in fh:
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            parts = rest.strip().split()
            if parts and parts[0].isdigit():
                stats[key.strip()] = int(parts[0])
    return stats


def _print_rollup(stats: Dict[str, int]) -> None:
    keys = (
        "Rss",
        "Pss",
        "Private_Clean",
        "Private_Dirty",
        "Shared_Clean",
        "Shared_Dirty",
        "Swap",
    )
    summary = ", ".join(
        f"{k}={stats.get(k, 0)/1024:.1f} MB" for k in keys if k in stats
    )
    if summary:
        print(f"    rollup: {summary}")


def _print_top_maps(entries: Sequence[Tuple[str, int, int]]) -> None:
    if not entries:
        return
    print("    top mappings (path, rss, size):")
    for path, rss_kb, size_kb in entries:
        print(f"      {path}: RSS={rss_kb/1024:.2f} MB, Size={size_kb/1024:.2f} MB")


def measure_worker_memory(
    *,
    num_workers: int,
    num_batches: int,
    sleep_s: float,
    top_maps: int,
    data_root: Path,
    profile_loader: bool,
    profile_rows: int,
) -> None:
    """Create a DataLoader, consume batches, then report worker RSS."""
    cfg = Config()
    cfg.train.num_workers = num_workers
    cfg.train.prefetch_factor = max(2, cfg.train.prefetch_factor)
    cfg.zarr.out_root = str(data_root)
    loader, _, _ = make_dataloader(cfg)

    loader_iter = iter(loader)
    consumed = 0

    profiler_report: Optional[str] = None
    if profile_loader and num_batches > 0:
        try:
            with profile(
                activities=[ProfilerActivity.CPU],
                profile_memory=True,
                record_shapes=True,
            ) as prof:
                next(loader_iter)
                consumed += 1
        except StopIteration:
            pass
        except Exception as exc:  # pragma: no cover - debug helper
            profiler_report = f"[profiler failed: {exc}]"
        else:
            profiler_report = prof.key_averages().table(
                sort_by="self_cpu_memory_usage",
                row_limit=profile_rows if profile_rows > 0 else 20,
            )

    while consumed < num_batches:
        try:
            next(loader_iter)
        except StopIteration:
            break
        consumed += 1
        if sleep_s > 0:
            time.sleep(sleep_s)

    if profiler_report:
        print("Torch profiler (sorted by self_cpu_memory_usage):")
        print(profiler_report)

    worker_info = _capture_worker_pids(loader_iter)
    if not worker_info:
        print("No worker processes detected (num_workers may be zero).")
    else:
        print("Worker memory usage:")
        for worker_id, pid in worker_info:
            rss_kb = _read_rss_kb(pid)
            print(f"  worker {worker_id:02d} pid={pid}: RSS={_format_mb(rss_kb)}")
            rollup = _read_smaps_rollup(pid)
            if rollup:
                _print_rollup(rollup)
            if top_maps > 0:
                top = _largest_smaps_entries(pid, top_n=top_maps)
                _print_top_maps(top)

    if hasattr(loader_iter, "_shutdown_workers"):
        try:
            loader_iter._shutdown_workers()  # type: ignore[attr-defined]
        except RuntimeError as exc:  # pragma: no cover - diagnostic helper
            print(f"[warn] failed to shutdown workers cleanly: {exc}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument(
        "--top-maps",
        type=int,
        default=5,
        help="How many smaps entries to report per worker.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("processed_data_3500"),
        help="Zarr dataset root to load.",
    )
    parser.add_argument(
        "--profile-loader",
        action="store_true",
        help="Run the first batch inside torch.profiler.",
    )
    parser.add_argument(
        "--profile-rows", type=int, default=25, help="Row limit for the profiler table."
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    measure_worker_memory(
        num_workers=args.num_workers,
        num_batches=args.num_batches,
        sleep_s=args.sleep,
        top_maps=args.top_maps,
        data_root=args.data_root,
        profile_loader=args.profile_loader,
        profile_rows=args.profile_rows,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
