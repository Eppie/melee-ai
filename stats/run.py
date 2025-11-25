"""Main orchestrator for statistics computation."""

from __future__ import annotations

import multiprocessing as mp
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


def _process_episodes_worker(args):
    """Worker function to process a chunk of episodes.

    Args:
        args: Tuple of (zarr_dir, episodes, feature_names, collector_names)

    Returns:
        CompositeCollector with processed results
    """
    zarr_dir, episodes, feature_names, collector_names = args

    # Create index and collectors for this worker
    index = ValidationDatasetIndex(zarr_dir)
    collectors = create_collectors(feature_names, collector_names)
    composite = CompositeCollector(feature_names, collectors)

    # Process episodes
    for episode in episodes:
        try:
            data = index.load_episode_data(episode)
            composite.process_episode(data, episode)
        except Exception as e:
            # Silently skip errors in workers
            pass

    return composite


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
        episodes = episodes[: config.max_episodes]
        if config.verbose:
            print(f"  Processing first {len(episodes)} episodes")

    # Determine number of workers
    n_workers = config.num_workers if config.num_workers > 0 else mp.cpu_count()

    # Create collectors for reporting
    collector_names = list(config.collectors) if config.collectors else None
    collectors = create_collectors(index.feature_names, collector_names)

    if config.verbose:
        print(f"\nRunning {len(collectors)} collectors with {n_workers} workers:")
        for c in collectors:
            print(f"  - {c.name}")

    # Split episodes into chunks for parallel processing
    chunk_size = max(1, len(episodes) // (n_workers * 4))  # 4 chunks per worker for load balancing
    episode_chunks = [
        episodes[i : i + chunk_size] for i in range(0, len(episodes), chunk_size)
    ]

    if config.verbose:
        print(f"  Split {len(episodes)} episodes into {len(episode_chunks)} chunks")

    # Process chunks in parallel
    total_frames = sum(ep.num_frames for ep in episodes)

    worker_args = [
        (config.zarr_dir, chunk, index.feature_names, collector_names)
        for chunk in episode_chunks
    ]

    # Use multiprocessing pool to process chunks in parallel
    with mp.Pool(n_workers) as pool:
        if config.verbose:
            # Process with progress bar
            chunk_results = []
            for result in tqdm(
                pool.imap_unordered(_process_episodes_worker, worker_args),
                total=len(worker_args),
                desc="Processing chunks",
                unit="chunk",
                dynamic_ncols=True,
            ):
                chunk_results.append(result)
        else:
            chunk_results = pool.map(_process_episodes_worker, worker_args)

    # Merge results from all workers
    if config.verbose:
        print("\nMerging results from workers...")

    composite = chunk_results[0]
    for i in range(1, len(chunk_results)):
        composite.merge(chunk_results[i])

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
