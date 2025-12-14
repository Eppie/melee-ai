#!/usr/bin/env python3
"""Profile filter_bad_replays.py with sequential processing to capture actual work."""

import cProfile
import pstats
import tempfile
from pathlib import Path
from pstats import SortKey

from filter_bad_replays import _extract_and_process_member, _iter_zip_members, Character
import filter_bad_replays


def run_sequential_profiling():
    """Run cProfile on sequential zip processing to see actual work."""

    zip_path = Path("/Users/eppie/PycharmProjects/nano-melee/test_replays.zip")

    # Create temporary directories for output
    with tempfile.TemporaryDirectory(
        prefix="profile_good_"
    ) as good_dir, tempfile.TemporaryDirectory(prefix="profile_failed_") as failed_dir:
        # Override output directories
        filter_bad_replays.GOOD_DIR = Path(good_dir)
        filter_bad_replays.FAILED_DIR = Path(failed_dir)

        print(f"Profiling sequential zip processing for: {zip_path}")
        print(f"Zip size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
        print("-" * 80)

        # Get all members
        members = list(_iter_zip_members(zip_path, None))
        print(f"Processing {len(members)} files sequentially...")

        # Run with profiling
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            for member in members:
                _extract_and_process_member(zip_path, member, None)
        finally:
            profiler.disable()

        print("\n" + "=" * 80)
        print("PROFILING RESULTS (Sequential Processing)")
        print("=" * 80)

        # Save stats
        profiler.dump_stats("filter_profile_sequential.prof")
        stats = pstats.Stats(profiler)

        # Print summary
        print("\nTop 50 functions by cumulative time:")
        print("-" * 80)
        stats.sort_stats(SortKey.CUMULATIVE).print_stats(50)

        print("\n" + "=" * 80)
        print("Top 50 functions by total time:")
        print("-" * 80)
        stats.sort_stats(SortKey.TIME).print_stats(50)

        # Save detailed stats
        stats_file = "filter_profile_sequential_stats.txt"
        with open(stats_file, "w") as f:
            stats_obj = pstats.Stats(profiler, stream=f)
            f.write("=" * 80 + "\n")
            f.write("SEQUENTIAL PROFILING - TOP BY CUMULATIVE TIME\n")
            f.write("=" * 80 + "\n")
            stats_obj.sort_stats(SortKey.CUMULATIVE).print_stats(100)
            f.write("\n" + "=" * 80 + "\n")
            f.write("SEQUENTIAL PROFILING - TOP BY TOTAL TIME\n")
            f.write("=" * 80 + "\n")
            stats_obj.sort_stats(SortKey.TIME).print_stats(100)

        print("\n" + "=" * 80)
        print(f"✓ Full stats saved to: {stats_file}")
        print(f"✓ Binary profile saved to: filter_profile_sequential.prof")


if __name__ == "__main__":
    run_sequential_profiling()
