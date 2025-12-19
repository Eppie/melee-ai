#!/usr/bin/env python3
"""Extract Slippi replay files to denormalized Parquet format.

Each row represents one frame for one player (non-follower characters only).
Merges controller inputs and game state into a single row.

Usage:
    python extract_replays_to_parquet.py --input-dir /path/to/replays --output-dir ./parquet_output --limit 10
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger
from tqdm import tqdm

from libmelee.melee.console import Console
from libmelee.melee.enums import Button

# Configure loguru
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

# Ice Climbers character IDs (internal)
ICE_CLIMBERS_ID = 14


@dataclass
class ReplayMetadata:
    """Metadata extracted from Game Start event."""
    replay_file: str
    slippi_major: int
    slippi_minor: int
    slippi_build: int
    stage_id: int


@dataclass
class FrameData:
    """Data for a single frame for a single player."""
    # Replay metadata
    replay_file: str
    slippi_major: int
    slippi_minor: int
    slippi_build: int
    stage_id: int

    # Frame identification
    frame_number: int
    player_index: int
    port: int

    # Controller inputs
    joystick_x: float
    joystick_y: float
    cstick_x: float
    cstick_y: float
    trigger_l: float
    trigger_r: float
    button_a: bool
    button_b: bool
    button_x: bool
    button_y: bool
    button_z: bool
    button_l: bool
    button_r: bool

    # Game state
    character_id: int
    action_state: int
    action_state_frame: float
    position_x: float
    position_y: float
    facing: bool
    percent: int
    shield_size: float
    stocks: int
    hitstun_remaining: int
    on_ground: bool
    jumps_remaining: int
    lcancel_status: int
    invulnerable: bool

    # Velocity
    speed_air_x_self: float
    speed_y_self: float
    speed_ground_x_self: float
    speed_x_attack: float
    speed_y_attack: float

    # State flags
    is_fastfalling: bool
    is_in_hitlag: bool
    is_in_hitstun: bool
    is_defender_in_hitlag: bool
    is_shield_active: bool
    is_powershield: bool
    hitlag_left: float
    invulnerability_left: int
    iasa: bool
    off_stage: bool
    is_offscreen: bool

    # ECB (Environment Collision Box)
    ecb_left_x: float
    ecb_left_y: float
    ecb_right_x: float
    ecb_right_y: float
    ecb_top_x: float
    ecb_top_y: float
    ecb_bottom_x: float
    ecb_bottom_y: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for DataFrame construction."""
        return {
            "replay_file": self.replay_file,
            "slippi_major": self.slippi_major,
            "slippi_minor": self.slippi_minor,
            "slippi_build": self.slippi_build,
            "stage_id": self.stage_id,
            "frame_number": self.frame_number,
            "player_index": self.player_index,
            "port": self.port,
            "joystick_x": self.joystick_x,
            "joystick_y": self.joystick_y,
            "cstick_x": self.cstick_x,
            "cstick_y": self.cstick_y,
            "trigger_l": self.trigger_l,
            "trigger_r": self.trigger_r,
            "button_a": self.button_a,
            "button_b": self.button_b,
            "button_x": self.button_x,
            "button_y": self.button_y,
            "button_z": self.button_z,
            "button_l": self.button_l,
            "button_r": self.button_r,
            "character_id": self.character_id,
            "action_state": self.action_state,
            "action_state_frame": self.action_state_frame,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "facing": self.facing,
            "percent": self.percent,
            "shield_size": self.shield_size,
            "stocks": self.stocks,
            "hitstun_remaining": self.hitstun_remaining,
            "on_ground": self.on_ground,
            "jumps_remaining": self.jumps_remaining,
            "lcancel_status": self.lcancel_status,
            "invulnerable": self.invulnerable,
            "speed_air_x_self": self.speed_air_x_self,
            "speed_y_self": self.speed_y_self,
            "speed_ground_x_self": self.speed_ground_x_self,
            "speed_x_attack": self.speed_x_attack,
            "speed_y_attack": self.speed_y_attack,
            "is_fastfalling": self.is_fastfalling,
            "is_in_hitlag": self.is_in_hitlag,
            "is_in_hitstun": self.is_in_hitstun,
            "is_defender_in_hitlag": self.is_defender_in_hitlag,
            "is_shield_active": self.is_shield_active,
            "is_powershield": self.is_powershield,
            "hitlag_left": self.hitlag_left,
            "invulnerability_left": self.invulnerability_left,
            "iasa": self.iasa,
            "off_stage": self.off_stage,
            "is_offscreen": self.is_offscreen,
            "ecb_left_x": self.ecb_left_x,
            "ecb_left_y": self.ecb_left_y,
            "ecb_right_x": self.ecb_right_x,
            "ecb_right_y": self.ecb_right_y,
            "ecb_top_x": self.ecb_top_x,
            "ecb_top_y": self.ecb_top_y,
            "ecb_bottom_x": self.ecb_bottom_x,
            "ecb_bottom_y": self.ecb_bottom_y,
        }


def check_for_ice_climbers(replay_path: str) -> bool:
    """Check if replay contains Ice Climbers.

    Returns True if Ice Climbers detected, False otherwise.
    """
    try:
        console = Console(is_dolphin=False, allow_old_version=True, path=replay_path)
        if not console.connect():
            return False

        # Read first valid gamestate to check characters
        while True:
            gamestate = console.step()
            if gamestate is None:
                break
            if len(gamestate.players) < 2:
                continue
            if gamestate.frame < 0:
                continue

            # Check each player's character
            for player in gamestate.players.values():
                try:
                    char_id = player.character.value
                    if char_id == ICE_CLIMBERS_ID:
                        return True
                except (AttributeError, ValueError):
                    pass
            break

        return False
    except Exception:
        return False


def extract_replay_metadata(replay_path: str) -> Optional[ReplayMetadata]:
    """Extract metadata from the Game Start event.

    Returns None if replay cannot be parsed.
    """
    try:
        console = Console(is_dolphin=False, allow_old_version=True, path=replay_path)
        if not console.connect():
            logger.warning(f"Failed to connect: {os.path.basename(replay_path)}")
            return None

        gamestate = console.step()
        if gamestate is None:
            logger.warning(f"No gamestate: {os.path.basename(replay_path)}")
            return None

        # Extract stage
        stage_id = gamestate.stage.value

        return ReplayMetadata(
            replay_file=os.path.basename(replay_path),
            slippi_major=3,  # Default, libmelee doesn't expose version easily
            slippi_minor=0,
            slippi_build=0,
            stage_id=stage_id,
        )
    except Exception as e:
        logger.warning(f"Metadata extraction error {os.path.basename(replay_path)}: {e}")
        return None


def process_single_replay(replay_path: str) -> List[FrameData]:
    """Process a single replay file and extract all frame data.

    Returns list of FrameData objects, one per player per frame.
    Returns empty list if replay is invalid or contains Ice Climbers.
    """
    replay_name = os.path.basename(replay_path)

    # Check for Ice Climbers
    if check_for_ice_climbers(replay_path):
        logger.warning(f"Skipping (Ice Climbers): {replay_name}")
        return []

    # Extract metadata
    metadata = extract_replay_metadata(replay_path)
    if metadata is None:
        return []

    # Process all frames
    frames = []
    try:
        console = Console(is_dolphin=False, allow_old_version=True, path=replay_path)
        if not console.connect():
            logger.warning(f"Failed to connect: {replay_path}")
            return []

        while True:
            gamestate = console.step()
            if gamestate is None:
                break

            # Skip invalid frames
            if len(gamestate.players) < 2 or gamestate.frame < 0:
                continue

            # Process each player (skip followers)
            for player_idx, player in gamestate.players.items():
                # Skip followers (Nana)
                try:
                    if player.is_follower:
                        continue
                except AttributeError:
                    pass

                # Extract controller inputs
                try:
                    cs = player.controller_state
                    main_stick_x, main_stick_y = cs.main_stick
                    c_stick_x, c_stick_y = cs.c_stick
                    l_shoulder = cs.l_shoulder
                    r_shoulder = cs.r_shoulder
                    buttons = cs.button
                    button_a = buttons.get(Button.BUTTON_A, False)
                    button_b = buttons.get(Button.BUTTON_B, False)
                    button_x = buttons.get(Button.BUTTON_X, False)
                    button_y = buttons.get(Button.BUTTON_Y, False)
                    button_z = buttons.get(Button.BUTTON_Z, False)
                    button_l = buttons.get(Button.BUTTON_L, False)
                    button_r = buttons.get(Button.BUTTON_R, False)
                except (AttributeError, TypeError, KeyError):
                    # Skip frame if controller state unavailable
                    continue

                # Extract position and game state
                try:
                    pos_x = player.position.x
                    pos_y = player.position.y
                    char_id = player.character.value
                    action = player.action.value
                    action_frame = player.action_frame
                    facing = player.facing
                    percent = player.percent
                    shield = player.shield_strength
                    stock = player.stock
                    hitstun = player.hitstun_frames_left
                    on_ground = player.on_ground
                    jumps = player.jumps_left
                    lcancel = player.l_cancel_status
                    invuln = player.invulnerable
                    speed_air_x = player.speed_air_x_self
                    speed_y = player.speed_y_self
                    speed_ground_x = player.speed_ground_x_self
                    speed_x_atk = player.speed_x_attack
                    speed_y_atk = player.speed_y_attack
                    fastfall = bool(player.is_fastfalling)
                    hitlag = bool(player.is_in_hitlag)
                    in_hitstun = bool(player.is_in_hitstun)
                    defender_hitlag = bool(player.is_defender_in_hitlag)
                    shield_active = bool(player.is_shield_active)
                    powershield = bool(player.is_powershield)
                    hitlag_left = float(player.hitlag_left)
                    invuln_left = int(player.invulnerability_left)
                    iasa = bool(player.iasa)
                    off_stage = bool(player.off_stage)
                    offscreen = bool(player.is_offscreen)
                    # ECB (Environment Collision Box)
                    ecb_l_x, ecb_l_y = player.ecb_left
                    ecb_r_x, ecb_r_y = player.ecb_right
                    ecb_t_x, ecb_t_y = player.ecb_top
                    ecb_b_x, ecb_b_y = player.ecb_bottom
                except (AttributeError, TypeError, ValueError):
                    # Skip frame if essential state unavailable
                    continue

                # Create frame data
                frame = FrameData(
                    replay_file=metadata.replay_file,
                    slippi_major=metadata.slippi_major,
                    slippi_minor=metadata.slippi_minor,
                    slippi_build=metadata.slippi_build,
                    stage_id=metadata.stage_id,
                    frame_number=gamestate.frame,
                    player_index=player_idx,
                    port=player_idx + 1,
                    joystick_x=main_stick_x,
                    joystick_y=main_stick_y,
                    cstick_x=c_stick_x,
                    cstick_y=c_stick_y,
                    trigger_l=l_shoulder,
                    trigger_r=r_shoulder,
                    button_a=button_a,
                    button_b=button_b,
                    button_x=button_x,
                    button_y=button_y,
                    button_z=button_z,
                    button_l=button_l,
                    button_r=button_r,
                    character_id=char_id,
                    action_state=action,
                    action_state_frame=action_frame,
                    position_x=pos_x,
                    position_y=pos_y,
                    facing=facing,
                    percent=percent,
                    shield_size=shield,
                    stocks=stock,
                    hitstun_remaining=hitstun,
                    on_ground=on_ground,
                    jumps_remaining=jumps,
                    lcancel_status=lcancel,
                    invulnerable=invuln,
                    speed_air_x_self=speed_air_x,
                    speed_y_self=speed_y,
                    speed_ground_x_self=speed_ground_x,
                    speed_x_attack=speed_x_atk,
                    speed_y_attack=speed_y_atk,
                    is_fastfalling=fastfall,
                    is_in_hitlag=hitlag,
                    is_in_hitstun=in_hitstun,
                    is_defender_in_hitlag=defender_hitlag,
                    is_shield_active=shield_active,
                    is_powershield=powershield,
                    hitlag_left=hitlag_left,
                    invulnerability_left=invuln_left,
                    iasa=iasa,
                    off_stage=off_stage,
                    is_offscreen=offscreen,
                    ecb_left_x=ecb_l_x,
                    ecb_left_y=ecb_l_y,
                    ecb_right_x=ecb_r_x,
                    ecb_right_y=ecb_r_y,
                    ecb_top_x=ecb_t_x,
                    ecb_top_y=ecb_t_y,
                    ecb_bottom_x=ecb_b_x,
                    ecb_bottom_y=ecb_b_y,
                )
                frames.append(frame)

    except Exception as e:
        logger.warning(f"Processing error {replay_name}: {e}")
        return []

    return frames


def process_replay_worker(replay_path: str) -> Tuple[str, List[FrameData]]:
    """Worker function for parallel processing.

    Returns (replay_path, frames) tuple.
    """
    frames = process_single_replay(replay_path)
    return (replay_path, frames)


def write_parquet_partition(
    frames: List[FrameData], output_path: Path, partition_id: int
) -> None:
    """Write a partition of frame data to Parquet file."""
    if not frames:
        return

    # Convert to DataFrame
    records = [frame.to_dict() for frame in frames]
    df = pd.DataFrame(records)

    # Write to Parquet with compression
    output_file = output_path / f"partition_{partition_id:05d}.parquet"
    df.to_parquet(
        output_file,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    logger.info(f"Wrote partition {partition_id}: {len(df):,} rows ({df.memory_usage(deep=True).sum() / 1024**2:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Slippi replays to Parquet format"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing .slp replay files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./replay_parquet",
        help="Output directory for Parquet files (default: ./replay_parquet)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of replays to process (for testing)",
    )
    parser.add_argument(
        "--partition-size",
        type=int,
        default=100,
        help="Number of replays per Parquet partition (default: 100)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .slp files
    replay_files = sorted(input_dir.glob("*.slp"))
    if args.limit:
        replay_files = replay_files[: args.limit]

    logger.info(f"Found {len(replay_files):,} replay files")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Partition size: {args.partition_size} replays")

    # Process replays in parallel
    max_workers = args.workers or os.cpu_count()
    all_frames = []
    partition_id = 0
    processed_count = 0
    total_frames = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_replay_worker, str(path)): path
            for path in replay_files
        }

        with tqdm(total=len(replay_files), desc="Processing replays", unit="replay") as pbar:
            for future in as_completed(futures):
                replay_path, frames = future.result()
                processed_count += 1

                if frames:
                    all_frames.extend(frames)
                    total_frames += len(frames)

                # Write partition when we have enough replays processed
                if processed_count % args.partition_size == 0 and all_frames:
                    write_parquet_partition(all_frames, output_dir, partition_id)
                    all_frames = []  # Clear memory
                    partition_id += 1

                pbar.update(1)

    # Write remaining frames
    if all_frames:
        write_parquet_partition(all_frames, output_dir, partition_id)
        partition_id += 1

    logger.success(f"Processing complete!")
    logger.info(f"Total partitions: {partition_id}")
    logger.info(f"Total frames extracted: {total_frames:,}")
    logger.info(f"Average frames per replay: {total_frames / len(replay_files):.1f}")


if __name__ == "__main__":
    main()
