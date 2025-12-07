#!/usr/bin/env python3
"""Profile filter_bad_replays.py zip processing performance."""

import cProfile
import pstats
import shutil
import tempfile
from pathlib import Path
from pstats import SortKey

# Import the filter module
from filter_bad_replays import _process_zip_archive, Character

def run_profiling():
    """Run cProfile on the zip processing code."""

    zip_path = Path("/Users/eppie/PycharmProjects/nano-melee/test_replays.zip")

    # Create temporary directories for output
    with tempfile.TemporaryDirectory(prefix="profile_good_") as good_dir, \
         tempfile.TemporaryDirectory(prefix="profile_failed_") as failed_dir:

        # Temporarily override the output directories
        import filter_bad_replays
        original_good = filter_bad_replays.GOOD_DIR
        original_failed = filter_bad_replays.FAILED_DIR

        filter_bad_replays.GOOD_DIR = Path(good_dir)
        filter_bad_replays.FAILED_DIR = Path(failed_dir)

        print(f"Profiling zip processing for: {zip_path}")
        print(f"Zip size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"Temporary output: {good_dir}")
        print("-" * 80)

        # Run with profiling
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            _process_zip_archive(
                zip_path=zip_path,
                filename_filter=None,
                process_workers=8,  # Adjust based on your CPU
                extract_workers=4,
                allowed_chars=None,
            )
        finally:
            profiler.disable()

            # Restore original directories
            filter_bad_replays.GOOD_DIR = original_good
            filter_bad_replays.FAILED_DIR = original_failed

        print("\n" + "=" * 80)
        print("PROFILING RESULTS")
        print("=" * 80)

        # Save detailed stats to file
        stats_file = "filter_profile_stats.txt"
        profiler.dump_stats("filter_profile.prof")

        stats = pstats.Stats(profiler)

        # Print summary to console
        print("\nTop 30 functions by cumulative time:")
        print("-" * 80)
        stats.sort_stats(SortKey.CUMULATIVE).print_stats(30)

        print("\n" + "=" * 80)
        print("Top 30 functions by total time:")
        print("-" * 80)
        stats.sort_stats(SortKey.TIME).print_stats(30)

        # Save full stats to file
        with open(stats_file, 'w') as f:
            stats = pstats.Stats(profiler, stream=f)
            f.write("=" * 80 + "\n")
            f.write("TOP FUNCTIONS BY CUMULATIVE TIME\n")
            f.write("=" * 80 + "\n")
            stats.sort_stats(SortKey.CUMULATIVE).print_stats(50)
            f.write("\n" + "=" * 80 + "\n")
            f.write("TOP FUNCTIONS BY TOTAL TIME\n")
            f.write("=" * 80 + "\n")
            stats.sort_stats(SortKey.TIME).print_stats(50)
            f.write("\n" + "=" * 80 + "\n")
            f.write("CALLERS OF TOP FUNCTIONS\n")
            f.write("=" * 80 + "\n")
            stats.sort_stats(SortKey.CUMULATIVE).print_callers(30)

        print("\n" + "=" * 80)
        print(f"✓ Full profiling stats saved to: {stats_file}")
        print(f"✓ Binary profile saved to: filter_profile.prof")
        print("\nTo view interactively:")
        print("  python -m pstats filter_profile.prof")

if __name__ == "__main__":
    run_profiling()
