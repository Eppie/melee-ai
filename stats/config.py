"""Configuration for statistics computation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set


@dataclass
class StatsConfig:
    """Configuration for statistics computation."""

    # Input
    zarr_dir: Path = field(default_factory=lambda: Path("validation_set"))

    # Output
    output_dir: Path = field(default_factory=lambda: Path("stats_output"))
    output_json: str = "statistics.json"
    output_summary: str = "summary.txt"

    # Collectors to run (empty = all)
    collectors: Set[str] = field(default_factory=set)

    # Processing
    num_workers: int = 0  # 0 = auto-detect
    max_episodes: Optional[int] = None  # None = all
    chunk_size: int = 100  # Episodes per worker chunk

    # Memory management
    sample_large_arrays: bool = True
    max_samples_per_episode: int = 1000

    # Output options
    verbose: bool = True
    include_histograms: bool = True
    include_examples: bool = True

    def __post_init__(self) -> None:
        self.zarr_dir = Path(self.zarr_dir)
        self.output_dir = Path(self.output_dir)

    @classmethod
    def from_args(cls, args) -> "StatsConfig":
        """Create config from argparse namespace."""
        return cls(
            zarr_dir=Path(args.zarr_dir),
            output_dir=Path(args.output_dir),
            output_json=args.output_json,
            collectors=set(args.collectors) if args.collectors else set(),
            num_workers=args.workers,
            max_episodes=args.max_episodes,
            verbose=not args.quiet,
            include_histograms=not args.no_histograms,
            include_examples=not args.no_examples,
        )


# Default collectors to run
DEFAULT_COLLECTORS = [
    "column_stats",
    "episode_stats",
    "action_states",
    "controller_inputs",
    "cross_features",
    "derived_metrics",
    "temporal",
    "data_quality",
]
