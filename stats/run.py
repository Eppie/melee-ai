"""Main orchestrator for statistics computation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from tqdm import tqdm

from stats.collectors.action_states import ActionStateCollector
from stats.collectors.base import CompositeCollector, StatsCollector
from stats.collectors.column_stats import ColumnStatsCollector
from stats.collectors.controller_inputs import ControllerInputCollector
from stats.collectors.cross_feature import CrossFeatureCollector
from stats.collectors.data_quality import DataQualityCollector
from stats.collectors.derived_metrics import DerivedMetricsCollector
from stats.collectors.episode_stats import EpisodeStatsCollector
from stats.collectors.temporal import TemporalCollector
from stats.config import DEFAULT_COLLECTORS, StatsConfig
from stats.index import ValidationDatasetIndex
from stats.output.json_writer import write_json_stats, write_summary_json
from stats.output.terminal import print_summary


# Registry of available collectors
COLLECTOR_REGISTRY: Dict[str, Type[StatsCollector]] = {
    "column_stats": ColumnStatsCollector,
    "episode_stats": EpisodeStatsCollector,
    "action_states": ActionStateCollector,
    "controller_inputs": ControllerInputCollector,
    "cross_features": CrossFeatureCollector,
    "derived_metrics": DerivedMetricsCollector,
    "temporal": TemporalCollector,
    "data_quality": DataQualityCollector,
}


def create_collectors(
    feature_names: List[str],
    collector_names: Optional[List[str]] = None,
) -> List[StatsCollector]:
    """Create collector instances.

    Args:
        feature_names: List of feature column names
        collector_names: Names of collectors to create (None = all)

    Returns:
        List of collector instances
    """
    if collector_names is None:
        collector_names = DEFAULT_COLLECTORS

    collectors = []
    for name in collector_names:
        if name in COLLECTOR_REGISTRY:
            collectors.append(COLLECTOR_REGISTRY[name](feature_names))
        else:
            print(f"Warning: Unknown collector '{name}', skipping")

    return collectors


def run_statistics(config: StatsConfig) -> Dict[str, Any]:
    """Run all statistics collectors on the dataset.

    Args:
        config: Statistics configuration

    Returns:
        Dictionary of all collected statistics
    """
    start_time = time.time()

    # Load dataset index
    if config.verbose:
        print(f"Loading dataset from {config.zarr_dir}...")

    index = ValidationDatasetIndex(config.zarr_dir)

    if config.verbose:
        print(f"  Found {len(index.episodes)} episodes")
        print(f"  Features: {len(index.feature_names)}")

    # Determine which episodes to process
    episodes = index.episodes
    if config.max_episodes is not None:
        episodes = episodes[:config.max_episodes]
        if config.verbose:
            print(f"  Processing first {len(episodes)} episodes")

    # Create collectors
    collector_names = list(config.collectors) if config.collectors else None
    collectors = create_collectors(index.feature_names, collector_names)

    if config.verbose:
        print(f"\nRunning {len(collectors)} collectors:")
        for c in collectors:
            print(f"  - {c.name}")

    # Create composite collector
    composite = CompositeCollector(index.feature_names, collectors)

    # Process episodes with detailed progress bar
    total_frames = sum(ep.num_frames for ep in episodes)
    frames_processed = 0
    errors = 0

    episode_pbar = tqdm(
        episodes,
        desc="Processing episodes",
        unit="ep",
        disable=not config.verbose,
        dynamic_ncols=True,
    )

    for episode in episode_pbar:
        try:
            data = index.load_episode_data(episode)
            composite.process_episode(data, episode)
            frames_processed += episode.num_frames

            # Update progress bar with detailed stats
            episode_pbar.set_postfix(
                frames=f"{frames_processed:,}/{total_frames:,}",
                fps=f"{frames_processed / (episode_pbar.format_dict['elapsed'] + 0.001):.0f}",
                errors=errors,
            )
        except Exception as e:
            errors += 1
            if config.verbose:
                tqdm.write(f"Warning: Failed to process episode {episode.episode_id}: {e}")

    # Finalize and collect results
    if config.verbose:
        print("\nFinalizing statistics...")

    results = composite.finalize()

    # Add metadata
    elapsed = time.time() - start_time
    results["_metadata"] = {
        "zarr_dir": str(config.zarr_dir),
        "total_episodes_processed": composite._episodes_processed,
        "total_frames_processed": composite._frames_processed,
        "processing_time_seconds": elapsed,
        "collectors_used": [c.name for c in collectors],
    }

    if config.verbose:
        print(f"Completed in {elapsed:.1f}s")

    return results


def main(config: StatsConfig) -> None:
    """Main entry point for statistics computation.

    Args:
        config: Statistics configuration
    """
    # Run statistics
    results = run_statistics(config)

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Write full JSON output
    json_path = config.output_dir / config.output_json
    if config.verbose:
        print(f"\nWriting statistics to {json_path}...")
    write_json_stats(results, json_path, config)

    # Write summary JSON (without large arrays)
    summary_json_path = config.output_dir / "summary.json"
    write_summary_json(results, summary_json_path)

    # Print terminal summary
    if config.verbose:
        print_summary(results, verbose=True)

    if config.verbose:
        print(f"\nOutput written to {config.output_dir}/")
        print(f"  - {config.output_json} (full statistics)")
        print(f"  - summary.json (compact summary)")


if __name__ == "__main__":
    # Default config for quick testing
    cfg = StatsConfig()
    main(cfg)
