from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from libmelee.melee import enums
from libmelee.melee.console import Console
from libmelee.melee.controller import ControllerState
from libmelee.melee.gamestate import GameState, PlayerState
from preprocess import _preprocess_frame, _preprocess_stage, _preprocess_character, _preprocess_action, \
    _preprocess_x_y_buttons, _preprocess_l_r_buttons
from schema import Row
from stats import log_all_stats
# from to_parquet import write_rows_to_parquet


def file_hash(path: str | Path, algo: str = "md5") -> np.uint32:
    """
    Compute a 32-bit integer hash of a file, suitable as an identifier.

    Args:
        path: Path to the file as a str or pathlib.Path.
        algo: Hash algorithm (default: md5 for speed).

    Returns:
        np.uint32: deterministic 32-bit unsigned integer hash of the file.
    """
    data = Path(path).read_bytes()
    val = int.from_bytes(hashlib.new(algo, data).digest()[:4], "little", signed=False)
    return np.uint32(val)


def extract(game_state: GameState, replay_hash: Optional[np.uint32], replay_filename: Optional[str]) -> Row:
    # Common
    frame: np.int32 = _preprocess_frame(game_state.frame)
    stage: np.int32 = _preprocess_stage(game_state.stage)
    distance: np.float32 = np.float32(game_state.distance)
    players: list[int] = sorted(game_state.players.keys())
    p1_port: int = players[0]
    p2_port: int = players[1]

    def _collect_player_fields(port: int, prefix: str) -> dict:
        pl: PlayerState = game_state.players[port]
        cs: ControllerState = pl.controller_state

        # Buttons (strict access)
        b = cs.button

        # action_state, action_state_category = _preprocess_action(pl.action)

        return {
            # Core state
            f"{prefix}_action": _preprocess_action(pl.action),
            # f"{prefix}_action_category": action_state_category,
            f"{prefix}_character": _preprocess_character(pl.character),
            f"{prefix}_pos_x": np.float32(pl.position.x),
            f"{prefix}_pos_y": np.float32(pl.position.y),
            f"{prefix}_percent": np.int32(pl.percent),
            f"{prefix}_stock": np.int32(pl.stock),
            f"{prefix}_facing": np.float32(bool(pl.facing)),
            f"{prefix}_on_ground": np.float32(bool(pl.on_ground)),

            # Buttons
            f"{prefix}_button_a": np.float32(bool(b[enums.Button.BUTTON_A])),
            f"{prefix}_button_b": np.float32(bool(b[enums.Button.BUTTON_B])),
            f"{prefix}_button_xy": _preprocess_x_y_buttons(bool(b[enums.Button.BUTTON_X]),
                                                           bool(b[enums.Button.BUTTON_Y])),
            f"{prefix}_button_z": np.float32(bool(b[enums.Button.BUTTON_Z])),
            f"{prefix}_button_lr": _preprocess_l_r_buttons(bool(b[enums.Button.BUTTON_L]),
                                                           bool(b[enums.Button.BUTTON_R])),

            # Sticks / shoulders
            f"{prefix}_main_stick_x": np.float32(cs.main_stick[0]),
            f"{prefix}_main_stick_y": np.float32(cs.main_stick[1]),
            f"{prefix}_c_stick_x": np.float32(cs.c_stick[0]),
            f"{prefix}_c_stick_y": np.float32(cs.c_stick[1]),
            f"{prefix}_shoulder_analog": np.float32(cs.l_shoulder),

            # Additional state (strict access; will raise if missing)
            f"{prefix}_shield_strength": np.float32(pl.shield_strength),
            f"{prefix}_is_powershield": np.float32(bool(pl.is_powershield)),
            f"{prefix}_action_frame": np.int32(pl.action_frame),

            # New flags from PlayerState (sb1–sb5-derived)
            f"{prefix}_is_reflect_active": np.float32(bool(pl.is_reflect_active)),
            f"{prefix}_is_subaction_invulnerable": np.float32(bool(pl.is_subaction_invulnerable)),
            f"{prefix}_is_fastfalling": np.float32(bool(pl.is_fastfalling)),
            f"{prefix}_is_defender_in_hitlag": np.float32(bool(pl.is_defender_in_hitlag)),
            f"{prefix}_is_in_hitlag": np.float32(bool(pl.is_in_hitlag)),
            f"{prefix}_is_holding_character": np.float32(bool(pl.is_holding_character)),
            f"{prefix}_is_shield_active": np.float32(bool(pl.is_shield_active)),
            f"{prefix}_is_in_hitstun": np.float32(bool(pl.is_in_hitstun)),
            f"{prefix}_is_dead": np.float32(bool(pl.is_dead)),
            f"{prefix}_is_offscreen": np.float32(bool(pl.is_offscreen)),
            f"{prefix}_invulnerable": np.float32(bool(pl.invulnerable)),
            f"{prefix}_hitlag_left": np.int32(pl.hitlag_left),
            f"{prefix}_hitstun_frames_left": np.int32(pl.hitstun_frames_left),
            f"{prefix}_jumps_left": np.int32(pl.jumps_left),
            f"{prefix}_speed_air_x_self": np.float32(pl.speed_air_x_self),
            f"{prefix}_speed_y_self": np.float32(pl.speed_y_self),
            f"{prefix}_speed_x_attack": np.float32(pl.speed_x_attack),
            f"{prefix}_speed_y_attack": np.float32(pl.speed_y_attack),
            f"{prefix}_speed_ground_x_self": np.float32(pl.speed_ground_x_self),
            f"{prefix}_off_stage": np.float32(bool(pl.off_stage)),
            f"{prefix}_l_cancel_status": np.int32(pl.l_cancel_status),
        }

    fields = {
        "replay_hash": replay_hash,
        "replay_filename": replay_filename,
        "frame": frame,
        "stage": stage,
        "distance": distance,
    }
    fields.update(_collect_player_fields(p1_port, "p1"))
    fields.update(_collect_player_fields(p2_port, "p2"))

    return Row(**fields)


def process_one_replay(replay_path: str) -> Optional[list[Row]]:
    try:
        console = Console(path=str(replay_path), is_dolphin=False, allow_old_version=True)
        console.connect()
    except Exception as e:
        logger.debug(f"Error connecting to console for {replay_path}: {e}")
        return None
    rows: list[Row] = []
    try:
        replay_hash = file_hash(replay_path)
        replay_filename = Path(replay_path).name
        current_game_state: GameState = console.step()
        while current_game_state is not None:
            row = extract(current_game_state, replay_hash, replay_filename)
            rows.append(row)
            current_game_state: GameState = console.step()

    except Exception as e:
        logger.error(f"Error processing replay {replay_path}: {e}")
        return None
    finally:
        console.stop()

    return rows


def main() -> None:
    rows = process_one_replay("/Users/eppie/PycharmProjects/new-melee-ai/test/test.slp")
    logger.info(f"Processed {len(rows)} rows")
    logger.info(f"First row: {rows[0]}")
    logger.info(f"Last row: {rows[-1]}")
    logger.info(f"Row 64: {rows[64]}")
    logger.info(f"Row 100: {rows[100]}")
    logger.info(f"Row 101: {rows[101]}")
    logger.info(f"Row 110: {rows[110]}")
    log_all_stats(rows)
    # write_rows_to_parquet(rows, "melee_rows.parquet", row_cls=Row)


if __name__ == "__main__":
    main()
