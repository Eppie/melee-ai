#!/usr/bin/env python3

from __future__ import annotations

import os
import sys
import random
from pathlib import Path
from typing import Optional, List

from libmelee.melee.console import Console
from libmelee.melee.gamestate import GameState

REPLAYS_DIR = Path("/Users/eppie/Downloads/ALL_REPLAYS/FOX_vs_FOX")
PATTERN = "master-master*.slp"
SAMPLE_SIZE_DEFAULT = 50
SEED_DEFAULT = 1337  # ensures a consistent "random" subset when file list is unchanged


def pick_deterministic_sample(files: List[Path], k: int, seed: int) -> List[Path]:
    # Sort for stability, then sample with a fixed seed for deterministic selection.
    files_sorted = sorted(files, key=lambda p: str(p))
    rng = random.Random(seed)
    k = min(k, len(files_sorted))
    return rng.sample(files_sorted, k)


def count_steps_for_file(replay_path: Path) -> int:
    steps = 0
    console = Console(path=str(replay_path), is_dolphin=False, allow_old_version=True)
    console.connect()
    try:
        state: Optional[GameState] = console.step()
        while state is not None:
            steps += 1
            state = console.step()
    finally:
        # Be polite even for SLP playback
        try:
            console.stop()
        except Exception:
            pass
    return steps


def main() -> int:
    sample_size = int(os.getenv("SAMPLE_SIZE", str(SAMPLE_SIZE_DEFAULT)))
    seed = int(os.getenv("SAMPLE_SEED", str(SEED_DEFAULT)))

    files = list(REPLAYS_DIR.glob(PATTERN))
    if not files:
        print(f"No files matched {REPLAYS_DIR / PATTERN}", file=sys.stderr)
        return 2

    sample = pick_deterministic_sample(files, sample_size, seed)

    total_steps = 0
    for fp in sample:
        try:
            total_steps += count_steps_for_file(fp)
        except Exception as e:
            # Skip problematic files but keep going.
            print(f"Warning: failed on {fp}: {e}", file=sys.stderr)

    # Print ONLY the total number, per your instruction.
    print(total_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
