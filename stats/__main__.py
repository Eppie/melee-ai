"""CLI entry point for statistics computation.

Usage:
    python -m stats [options]

Examples:
    # Run all collectors on validation_set/
    python -m stats

    # Specify custom zarr directory
    python -m stats --zarr-dir processed_data/

    # Run specific collectors only
    python -m stats --collectors episode_stats action_states

    # Limit number of episodes for quick testing
    python -m stats --max-episodes 100

    # Custom output directory
    python -m stats --output-dir my_stats/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stats.config import DEFAULT_COLLECTORS, StatsConfig
from stats.run import COLLECTOR_REGISTRY, main


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute comprehensive statistics on Melee replay data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available collectors:
  column_stats      - Per-column statistics (mean, std, percentiles, run lengths)
  episode_stats     - Episode-level statistics (length, stage, outcome)
  action_states     - Action state analysis (categories, transitions)
  controller_inputs - Controller input analysis (buttons, sticks)
  cross_features    - Cross-feature correlations and joint distributions
  derived_metrics   - Derived metrics (distance, combos, advantage)
  temporal          - Temporal patterns (transitions, change rates)
  data_quality      - Data quality validation and scoring

Examples:
  python -m stats                                    # Run all collectors
  python -m stats --zarr-dir processed_data/        # Custom input
  python -m stats --collectors episode_stats        # Single collector
  python -m stats --max-episodes 1000               # Quick test
""",
    )

    parser.add_argument(
        "--zarr-dir",
        type=str,
        default="validation_set",
        help="Directory containing zarr data and index.json (default: validation_set)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="stats_output",
        help="Output directory for statistics files (default: stats_output)",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="statistics.json",
        help="Name of main JSON output file (default: statistics.json)",
    )

    parser.add_argument(
        "--collectors",
        type=str,
        nargs="+",
        choices=list(COLLECTOR_REGISTRY.keys()),
        default=None,
        help="Specific collectors to run (default: all)",
    )

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to process (default: all)",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (default: auto-detect)",
    )

    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )

    parser.add_argument(
        "--no-histograms",
        action="store_true",
        help="Exclude histogram data from output",
    )

    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Exclude example values from output",
    )

    parser.add_argument(
        "--list-collectors",
        action="store_true",
        help="List available collectors and exit",
    )

    return parser.parse_args()


def list_collectors() -> None:
    """Print available collectors and exit."""
    print("Available collectors:")
    print()
    for name, cls in COLLECTOR_REGISTRY.items():
        doc = cls.__doc__ or "No description"
        first_line = doc.strip().split("\n")[0]
        print(f"  {name:20} - {first_line}")
    print()
    print(f"Default collectors: {', '.join(DEFAULT_COLLECTORS)}")


def cli_main() -> None:
    """Main CLI entry point."""
    args = parse_args()

    if args.list_collectors:
        list_collectors()
        sys.exit(0)

    # Validate zarr directory exists
    zarr_dir = Path(args.zarr_dir)
    if not zarr_dir.exists():
        print(f"Error: Zarr directory not found: {zarr_dir}")
        sys.exit(1)

    # Check for required files
    index_path = zarr_dir / "index.jsonl"
    meta_path = zarr_dir / "meta.json"
    if not index_path.exists() or not meta_path.exists():
        print(f"Error: Required files not found in {zarr_dir}")
        print("Expected: meta.json, index.jsonl")
        print("Run the data preprocessing step first to generate these files.")
        sys.exit(1)

    # Create config
    config = StatsConfig.from_args(args)

    # Run
    try:
        main(config)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if not args.quiet:
            raise
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
