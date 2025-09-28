#!/usr/bin/env python3
"""
Test script for Zarr storage functionality.
"""

from pathlib import Path
import numpy as np
from loguru import logger

from zarr_storage import (
    process_replays_to_zarr, 
    load_zarr_dataset, 
    get_normalization_params,
    normalize_field
)


def test_single_replay():
    """Test processing a single replay file"""
    test_replay = "/Users/eppie/PycharmProjects/new-melee-ai/test/test.slp"
    output_dir = "/Users/eppie/PycharmProjects/new-melee-ai/test_zarr_output"
    
    if not Path(test_replay).exists():
        logger.error(f"Test replay not found at {test_replay}")
        return
    
    logger.info("Testing single replay processing...")
    
    # Process the replay
    process_replays_to_zarr([test_replay], output_dir)
    
    # Load and inspect the dataset
    root, metadata = load_zarr_dataset(output_dir)
    
    logger.info(f"Dataset created successfully!")
    logger.info(f"Rows: {metadata['metadata']['num_rows']}")
    logger.info(f"Fields: {len(metadata['metadata']['field_names'])}")
    
    # Show some statistics
    stats = metadata['statistics']
    logger.info("\nField Statistics:")
    for field_name in ['frame', 'distance', 'p1_pos_x', 'p1_pos_y']:
        if field_name in stats:
            field_stats = stats[field_name]
            logger.info(f"{field_name}: {field_stats['type']} - count={field_stats['count']}, "
                      f"mean={field_stats.get('mean', 'N/A')}, std={field_stats.get('std', 'N/A')}")
    
    # Test normalization
    logger.info("\nTesting normalization...")
    frame_params = get_normalization_params(stats, 'frame')
    logger.info(f"Frame normalization params: {frame_params}")
    
    # Get some frame data and normalize it
    frame_data = root['frame'][:]
    normalized_frames = normalize_field(frame_data, frame_params, method="robust")
    logger.info(f"Frame data range: {frame_data.min()} to {frame_data.max()}")
    logger.info(f"Normalized frame data range: {normalized_frames.min():.3f} to {normalized_frames.max():.3f}")
    
    # Test different normalization methods
    zscore_frames = normalize_field(frame_data, frame_params, method="zscore")
    minmax_frames = normalize_field(frame_data, frame_params, method="minmax")
    
    logger.info(f"Z-score normalized range: {zscore_frames.min():.3f} to {zscore_frames.max():.3f}")
    logger.info(f"Min-max normalized range: {minmax_frames.min():.3f} to {minmax_frames.max():.3f}")


def test_multiple_replays():
    """Test processing multiple replay files (if available)"""
    # This would be used if you have multiple replay files
    replay_dir = Path("/Users/eppie/Downloads/replays_sorted")
    
    if not replay_dir.exists():
        logger.info("No replay directory found, skipping multiple replay test")
        return
    
    # Find some .slp files
    slp_files = list(replay_dir.rglob("*.slp"))[:5]  # Limit to 5 files for testing
    
    if not slp_files:
        logger.info("No .slp files found, skipping multiple replay test")
        return
    
    logger.info(f"Testing with {len(slp_files)} replay files...")
    
    output_dir = "/Users/eppie/PycharmProjects/new-melee-ai/test_zarr_multi_output"
    process_replays_to_zarr([str(f) for f in slp_files], output_dir)
    
    # Load and inspect
    root, metadata = load_zarr_dataset(output_dir)
    logger.info(f"Multi-replay dataset: {metadata['metadata']['num_rows']} rows")


if __name__ == "__main__":
    test_single_replay()
    test_multiple_replays()
