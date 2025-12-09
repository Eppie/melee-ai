"""Build dataset from .slp replay files.

Processes replays in parallel and saves as compressed .npz files.
"""

from pathlib import Path
from typing import Optional
from multiprocessing import Pool
import numpy as np
import json
from tqdm import tqdm
import sys

# Add the parent directory to sys.path to import the local libmelee folder
sys.path.insert(0, str(Path(__file__).parents[2]))

from .parse_slp import parse_slp_to_features
from .compute_targets import compute_future_deltas, normalize_deltas
from ..constants import HORIZONS, NPZ_FEATURES_KEY, NPZ_FUTURE_DELTAS_KEY, NPZ_VALID_MASK_KEY, NPZ_STAGE_KEY, NPZ_P1_CHAR_KEY, NPZ_P2_CHAR_KEY, PLAYER_FEATURES_DIM, GLOBAL_RELATIONAL_FEATURES_DIM, STAGE_HALF_WIDTH, STAGE_HALF_HEIGHT


def process_single_replay(slp_path: Path, output_dir: Path, horizons: list) -> bool:
    """Process a single replay file to .npz format.

    Args:
        slp_path: Path to .slp file
        output_dir: Directory to save .npz output
        horizons: List of horizons to compute

    Returns:
        True if successful, False if failed

    Notes:
        - Parses replay to features
        - Computes future deltas
        - Saves as compressed .npz
        - One .npz per episode
    """
    import sys
    from pathlib import Path
    # Add the parent directory to sys.path to import the local libmelee folder
    sys.path.insert(0, str(Path(__file__).parents[2].parent)) # Two levels up from build_dataset.py to nano-melee

    try:
        data = parse_slp_to_features(slp_path)
        features = data['features']
        
        # Determine the indices for player positions from the comprehensive feature vector
        # P1_X is at index 0, P1_Y at index 1
        # P2_X is at PLAYER_FEATURES_DIM, P2_Y is at PLAYER_FEATURES_DIM + 1
        positions = features[:, [0, 1, PLAYER_FEATURES_DIM, PLAYER_FEATURES_DIM + 1]]

        deltas, valid = compute_future_deltas(positions, horizons)
        
        normalized_deltas = normalize_deltas(deltas, STAGE_HALF_WIDTH, STAGE_HALF_HEIGHT)

        episode_id = slp_path.stem
        output_path = output_dir / f"{episode_id}.npz"

        np.savez_compressed(
            output_path,
            features=features.astype(np.float32),
            future_deltas=normalized_deltas.astype(np.float32),
            valid_mask=valid,
            stage=np.array([data['stage']], dtype=np.uint8),
            p1_char=np.array([data['p1_char']], dtype=np.uint8),
            p2_char=np.array([data['p2_char']], dtype=np.uint8),
        )
        return True
    except Exception as e:
        print(f"Failed {slp_path}: {e}")
        return False


def build_dataset(
    slp_dir: Path,
    output_dir: Path,
    n_workers: int = 8,
    horizons: list = HORIZONS,
) -> None:
    """Build dataset from directory of .slp files.

    Args:
        slp_dir: Directory containing .slp replay files (recursive search)
        output_dir: Output directory for .npz files
        n_workers: Number of parallel workers
        horizons: List of horizons to compute

    Notes:
        - Processes replays in parallel
        - Creates index.json with episode metadata
        - Shows progress bar
    """
    slp_files = list(slp_dir.glob('**/*.slp'))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare arguments for multiprocessing
    args = [(slp_file, output_dir, horizons) for slp_file in slp_files]

    with Pool(n_workers) as pool:
        # Use tqdm for a progress bar
        results = list(tqdm(pool.starmap(process_single_replay, args), total=len(args), desc="Processing replays"))

    print(f"Processed {sum(results)}/{len(results)} replays successfully.")

    # Build and save index
    episode_index = build_episode_index(output_dir)
    save_index(episode_index, output_dir / 'index.json')


def build_episode_index(output_dir: Path) -> list:
    """Build index of all episodes in output directory.

    Args:
        output_dir: Directory containing .npz files

    Returns:
        List of dicts with episode metadata:
            - path: str
            - num_frames: int
            - stage: int
            - p1_char: int
            - p2_char: int

    Notes:
        - Scans all .npz files
        - Loads metadata from each
        - Returns list for easy JSON serialization
    """
    index = []
    for npz_path in output_dir.glob('*.npz'):
        try:
            with np.load(npz_path) as ep:
                index.append({
                    'path': str(npz_path),
                    'num_frames': len(ep[NPZ_FEATURES_KEY]),
                    'stage': int(ep[NPZ_STAGE_KEY][0]),
                    'p1_char': int(ep[NPZ_P1_CHAR_KEY][0]),
                    'p2_char': int(ep[NPZ_P2_CHAR_KEY][0]),
                })
        except Exception as e:
            print(f"Failed to load metadata from {npz_path}: {e}")
            continue
    return index


def save_index(index: list, output_path: Path) -> None:
    """Save episode index to JSON file.

    Args:
        index: List of episode metadata dicts
        output_path: Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(index, f, indent=2)


def load_index(index_path: Path) -> list:
    """Load episode index from JSON file.

    Args:
        index_path: Path to index JSON file

    Returns:
        List of episode metadata dicts
    """
    with open(index_path) as f:
        return json.load(f)


if __name__ == '__main__':
    """Command-line interface for dataset building."""
    import argparse

    parser = argparse.ArgumentParser(description="Build dataset from .slp replay files.")
    parser.add_argument(
        "--slp-dir",
        type=Path,
        required=True,
        help="Directory containing .slp replay files (recursive search).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for .npz files and index.json.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Number of parallel workers for processing replays.",
    )

    args = parser.parse_args()

    build_dataset(
        slp_dir=args.slp_dir,
        output_dir=args.output_dir,
        n_workers=args.n_workers,
        horizons=HORIZONS, # Using HORIZONS from constants.py
    )
