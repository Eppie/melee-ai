"""
Preprocessing CLI entry point.

This module provides the command-line interface for preprocessing
raw replay data.
"""

import argparse
from pathlib import Path

from melee_ai.config import load_settings_from_args, Settings
from melee_ai.preprocessing.extractor import ReplayExtractor


def preprocess_command(args=None):
    """Main preprocessing command."""
    # Load configuration
    settings = load_settings_from_args(args)

    print("Starting preprocessing with configuration:")
    print(f"  Data root: {settings.data.data_root}")
    print(f"  Replay directory: {settings.data.replay_dir}")

    # Create extractor
    extractor = ReplayExtractor(settings)

    # Find replay files
    if settings.data.replay_dir:
        replay_dir = Path(settings.data.replay_dir)
    else:
        replay_dir = Path(settings.data.data_root) / "replays"

    if not replay_dir.exists():
        print(f"Replay directory not found: {replay_dir}")
        return

    # Find SLP files
    slp_files = list(replay_dir.glob("*.slp"))
    print(f"Found {len(slp_files)} replay files")

    if not slp_files:
        print("No replay files found")
        return

    # Process first file for testing
    test_file = slp_files[0]
    print(f"Processing test file: {test_file}")

    try:
        frames = extractor.extract_replay(str(test_file))
        if frames:
            print(f"Successfully extracted {len(frames)} frames")
            print(f"Sample frame keys: {list(frames[0].keys())[:5]}...")
        else:
            print("Failed to extract frames")
    except Exception as e:
        print(f"Error during processing: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Preprocess Melee AI data")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data-root", type=str, help="Root directory for data")
    parser.add_argument("--replay-dir", type=str, help="Directory containing replay files")

    args = parser.parse_args()
    preprocess_command(args)


if __name__ == "__main__":
    main()
