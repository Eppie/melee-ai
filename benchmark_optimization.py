#!/usr/bin/env python3
"""Benchmark the two-pass optimization impact."""

import time
import tempfile
from pathlib import Path

import peppi_py
from filter_bad_replays import sanity_reason, quality_reason, _iter_zip_members

def benchmark_single_pass(zip_path: Path, num_files: int = 10):
    """Benchmark the old single-pass approach."""
    print("=" * 80)
    print("BENCHMARK: Single-Pass (Original)")
    print("=" * 80)

    members = list(_iter_zip_members(zip_path, None))[:num_files]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract files first
        import zipfile
        import shutil
        extracted_files = []
        with zipfile.ZipFile(zip_path) as zf:
            for member in members:
                dest = tmpdir / Path(member).name
                with zf.open(member) as src, open(dest, 'wb') as out:
                    shutil.copyfileobj(src, out, length=1024*1024)
                extracted_files.append(dest)

        # Benchmark: Single pass with all data
        start = time.perf_counter()
        passed = 0
        failed_sanity = 0
        failed_quality = 0

        for path in extracted_files:
            # OLD WAY: Parse everything at once
            game = peppi_py.read_slippi(str(path), skip_frames=False)

            if sanity_reason(game):
                failed_sanity += 1
                continue

            if quality_reason(game):
                failed_quality += 1
                continue

            passed += 1

        elapsed = time.perf_counter() - start

        print(f"\nProcessed {num_files} files in {elapsed:.3f}s")
        print(f"  - Passed: {passed}")
        print(f"  - Failed sanity: {failed_sanity}")
        print(f"  - Failed quality: {failed_quality}")
        print(f"  - Average: {elapsed / num_files * 1000:.2f} ms/file")
        print(f"  - Throughput: {num_files / elapsed:.1f} files/sec")

        return elapsed

def benchmark_two_pass(zip_path: Path, num_files: int = 10):
    """Benchmark the new two-pass approach."""
    print("\n" + "=" * 80)
    print("BENCHMARK: Two-Pass (Optimized)")
    print("=" * 80)

    members = list(_iter_zip_members(zip_path, None))[:num_files]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Extract files first
        import zipfile
        import shutil
        extracted_files = []
        with zipfile.ZipFile(zip_path) as zf:
            for member in members:
                dest = tmpdir / Path(member).name
                with zf.open(member) as src, open(dest, 'wb') as out:
                    shutil.copyfileobj(src, out, length=1024*1024)
                extracted_files.append(dest)

        # Benchmark: Two-pass approach
        start = time.perf_counter()
        passed = 0
        failed_sanity = 0
        failed_quality = 0

        for path in extracted_files:
            # NEW WAY: First pass without frames
            game = peppi_py.read_slippi(str(path), skip_frames=True)

            if sanity_reason(game):
                failed_sanity += 1
                continue

            # Second pass with frames only if needed
            game = peppi_py.read_slippi(str(path), skip_frames=False)

            if quality_reason(game):
                failed_quality += 1
                continue

            passed += 1

        elapsed = time.perf_counter() - start

        print(f"\nProcessed {num_files} files in {elapsed:.3f}s")
        print(f"  - Passed: {passed}")
        print(f"  - Failed sanity: {failed_sanity}")
        print(f"  - Failed quality: {failed_quality}")
        print(f"  - Average: {elapsed / num_files * 1000:.2f} ms/file")
        print(f"  - Throughput: {num_files / elapsed:.1f} files/sec")

        return elapsed

if __name__ == "__main__":
    zip_path = Path("/Users/eppie/PycharmProjects/nano-melee/test_replays.zip")

    print(f"Benchmarking with {zip_path.name}")
    print(f"Zip size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB\n")

    # Run multiple times to get stable results
    num_files = 37
    num_runs = 3

    single_times = []
    two_pass_times = []

    for run in range(num_runs):
        print(f"\n{'#' * 80}")
        print(f"RUN {run + 1}/{num_runs}")
        print(f"{'#' * 80}")

        single_time = benchmark_single_pass(zip_path, num_files)
        single_times.append(single_time)

        two_pass_time = benchmark_two_pass(zip_path, num_files)
        two_pass_times.append(two_pass_time)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    avg_single = sum(single_times) / len(single_times)
    avg_two_pass = sum(two_pass_times) / len(two_pass_times)

    print(f"\nSingle-Pass (avg {num_runs} runs): {avg_single:.3f}s")
    print(f"Two-Pass (avg {num_runs} runs):    {avg_two_pass:.3f}s")
    print(f"\nSpeedup: {avg_single / avg_two_pass:.2f}x")
    print(f"Time saved: {(avg_single - avg_two_pass):.3f}s ({(1 - avg_two_pass/avg_single) * 100:.1f}%)")
