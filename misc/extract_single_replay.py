#!/usr/bin/env python3
"""Extract a single replay file to parquet for testing."""

import os
from pathlib import Path
from extract_replays_to_parquet import process_single_replay, write_parquet_partition

replay_file = "slippi-js/slp/actionEdgeCases.slp"
output_dir = "test_comparison_parquet"

print(f"Extracting {replay_file}...")

# Process the replay
frame_data_list = process_single_replay(replay_file)
print(f"  Extracted {len(frame_data_list)} frames")

# Write to parquet
os.makedirs(output_dir, exist_ok=True)
write_parquet_partition(frame_data_list, Path(output_dir), partition_id=0)

print(f"  Written to {output_dir}/partition_0.parquet")
print("Done!")
