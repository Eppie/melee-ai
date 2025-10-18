#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import io
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import pstats

from column_map import ColumnMap
from config import get_config, init_config
from controller_quantization import quantize_targets
from train import build_inputs_for_gptv7
from utils import _resolve_device
from window_dataset import make_dataloader


def _parse_overrides(items: list[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def _load_initial_config(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config JSON at {cfg_path} must contain an object.")
    return data


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and hasattr(mps, "synchronize"):
            mps.synchronize()


def _create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the dataloader throughput without running model training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config_json", type=str, default=None,
                        help="Optional JSON file with initial config values.")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="Override config values (repeatable).")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to move batches onto (cpu, cuda, mps, or auto).")
    parser.add_argument("--epoch", type=int, default=0,
                        help="Epoch index to seed the sampler with.")
    parser.add_argument("--warmup-batches", type=int, default=10,
                        help="Number of batches to run before measuring throughput.")
    parser.add_argument("--max-batches", type=int, default=200,
                        help="Maximum number of batches to include in measurements. Use 0 for unlimited.")
    parser.add_argument("--stop-after-seconds", type=float, default=None,
                        help="Optional time budget (seconds) for the measured portion.")
    parser.add_argument("--report-every", type=int, default=25,
                        help="Print intermediate throughput every N measured batches (0 to disable).")
    parser.add_argument("--profile", action="store_true",
                        help="Enable cProfile around the benchmark run.")
    parser.add_argument("--profile-output", type=str, default=None,
                        help="Optional path to write raw cProfile stats.")
    parser.add_argument("--profile-sort", type=str, default="tottime",
                        help="Sort key for printed cProfile stats (e.g. tottime, cumtime).")
    parser.add_argument("--profile-top", type=int, default=30,
                        help="Number of cProfile rows to print when profiling is enabled.")
    parser.add_argument("--skip-quantize", action="store_true",
                        help="Skip controller target quantization when measuring throughput.")
    return parser


def _run_benchmark(args: argparse.Namespace) -> Dict[str, float]:
    config = get_config()
    loader, dataset, sampler = make_dataloader()
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(args.epoch)

    device = _resolve_device(args.device)
    colmap = ColumnMap.from_dataset(dataset)

    warmup_total = max(0, int(args.warmup_batches))
    max_batches = None if args.max_batches == 0 else max(args.max_batches, 0)
    report_every = max(0, int(args.report_every))
    stop_after = args.stop_after_seconds if args.stop_after_seconds and args.stop_after_seconds > 0 else None

    measured_batches = 0
    warmup_seen = 0
    measured_frames = 0
    measurement_start: Optional[float] = None

    print("=== Benchmark configuration ===")
    print(f"Dataset root: {config.zarr.out_root}")
    print(f"Sequence length: {dataset.seq_len}")
    print(f"Batch size: {config.train.batch_size}")
    print(f"num_workers: {config.train.num_workers}, pin_memory: {config.train.pin_memory}, "
          f"prefetch_factor: {config.train.prefetch_factor}")
    print(f"Device: {device.type}")
    if warmup_total:
        print(f"Warmup batches: {warmup_total}")
    print(f"Measuring up to {max_batches if max_batches is not None else '∞'} batches")
    if stop_after:
        print(f"Measurement time budget: {stop_after:.1f}s")

    try:
        for step, batch in enumerate(loader):
            X_cpu: torch.Tensor = batch["X"]
            Y_cpu: torch.Tensor = batch["Y"]

            # Mirror train.py data movement
            X = X_cpu.to(device, non_blocking=(device.type != "cpu"))
            Y = Y_cpu.to(device, non_blocking=(device.type != "cpu"))

            # Build model inputs / targets like the training loop would
            # Build inputs to ensure the same view transformations used during training run.
            inputs_td = build_inputs_for_gptv7(X, colmap)
            _ = inputs_td
            if not args.skip_quantize and Y.numel() and Y.shape[-1] > 0:
                quantize_targets(Y, colmap, input_domain="unit11")

            # Ensure asynchronous transfers are accounted for
            _synchronize_device(device)

            frames = int(X.shape[0]) * int(X.shape[1])

            if warmup_seen < warmup_total:
                warmup_seen += 1
                continue

            if measured_batches == 0:
                measurement_start = time.perf_counter()
            measured_batches += 1
            measured_frames += frames

            if report_every and measured_batches % report_every == 0:
                now = time.perf_counter()
                elapsed = now - (measurement_start or now)
                fps = measured_frames / elapsed if elapsed > 0 else float("nan")
                print(f"[batch {step + 1}] frames={measured_frames} "
                      f"({fps:.1f} fps, {measured_batches / elapsed:.2f} batches/s)")

            if max_batches is not None and measured_batches >= max_batches:
                break
            if stop_after and measurement_start is not None:
                if time.perf_counter() - measurement_start >= stop_after:
                    print(f"Stopping after reaching {stop_after:.1f}s measurement budget.")
                    break

        _synchronize_device(device)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.")

    if not measured_batches or measurement_start is None:
        raise RuntimeError("No batches were measured; adjust --warmup-batches or --max-batches.")

    total_elapsed = time.perf_counter() - measurement_start
    fps = measured_frames / total_elapsed if total_elapsed > 0 else float("nan")
    bps = measured_batches / total_elapsed if total_elapsed > 0 else float("nan")
    avg_frames_per_batch = measured_frames / measured_batches if measured_batches else float("nan")

    print("=== Benchmark results ===")
    print(f"Measured batches: {measured_batches}")
    print(f"Measured frames: {measured_frames}")
    print(f"Elapsed (measured): {total_elapsed:.3f}s")
    print(f"Throughput: {fps:.1f} frames/s")
    print(f"Batches per second: {bps:.3f}")
    print(f"Average frames per batch: {avg_frames_per_batch:.1f}")

    return {
        "measured_batches": float(measured_batches),
        "measured_frames": float(measured_frames),
        "elapsed_seconds": float(total_elapsed),
        "frames_per_second": float(fps),
        "batches_per_second": float(bps),
        "avg_frames_per_batch": float(avg_frames_per_batch),
    }


def main(argv: Optional[list[str]] = None) -> Dict[str, float]:
    parser = _create_arg_parser()
    args = parser.parse_args(argv)

    initial = _load_initial_config(args.config_json)
    overrides = _parse_overrides(args.set)
    init_config(initial=initial, cli_overrides=overrides)

    if args.profile or args.profile_output:
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            results = _run_benchmark(args)
        finally:
            profiler.disable()
            stats = pstats.Stats(profiler)
            stats.sort_stats(args.profile_sort)
            if args.profile_output:
                out_path = Path(args.profile_output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                stats.dump_stats(str(out_path))
                print(f"cProfile stats written to {out_path}")
            stream = io.StringIO()
            stats_stream = stats.stream
            stats.stream = stream
            stats.print_stats(args.profile_top)
            stats.stream = stats_stream
            print("=== Profiling (top functions) ===")
            print(stream.getvalue())
        return results

    return _run_benchmark(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
