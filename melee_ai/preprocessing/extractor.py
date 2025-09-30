"""
Replay extraction services for preprocessing.

This module provides classes for extracting features from raw SLP replay files,
including player state extraction and replay processing orchestration.
"""

import hashlib
from pathlib import Path
from typing import List, Optional

from libmelee.melee import enums
from libmelee.melee.console import Console
from libmelee.melee.controller import ControllerState
from libmelee.melee.gamestate import GameState, PlayerState
from melee_ai.config import Settings
from melee_ai.utils import Result, Err, Ok, guard_clause
from melee_ai.utils.logging import get_logger


class PlayerFeatureExtractor:
    """Extracts features for a single player from game state."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def extract_player_features(self, player_state: PlayerState, controller_state: ControllerState) -> dict:
        """
        Extract features for a single player.

        Args:
            player_state: Player state from libmelee
            controller_state: Controller state from libmelee

        Returns:
            Dictionary of extracted features
        """
        guard_clause(player_state is not None, "Player state cannot be None")
        guard_clause(controller_state is not None, "Controller state cannot be None")

        # Buttons
        button = controller_state.button

        return {
            # Core state
            "action": player_state.action.value,
            "character": player_state.character.value,
            "position_x": float(player_state.position.x),
            "position_y": float(player_state.position.y),
            "percent": int(player_state.percent),
            "stock": int(player_state.stock),
            "facing": float(bool(player_state.facing)),
            "on_ground": float(bool(player_state.on_ground)),

            # Buttons
            "button_a": float(bool(button[enums.Button.BUTTON_A])),
            "button_b": float(bool(button[enums.Button.BUTTON_B])),
            "button_xy": float(bool(button[enums.Button.BUTTON_X] or button[enums.Button.BUTTON_Y])),
            "button_z": float(bool(button[enums.Button.BUTTON_Z])),
            "button_lr": float(bool(button[enums.Button.BUTTON_L] or button[enums.Button.BUTTON_R])),

            # Sticks
            "main_stick_x": float(controller_state.main_stick[0]),
            "main_stick_y": float(controller_state.main_stick[1]),
            "c_stick_x": float(controller_state.c_stick[0]),
            "c_stick_y": float(controller_state.c_stick[1]),
            "shoulder_analog": float(controller_state.l_shoulder),

            # Additional state
            "shield_strength": float(player_state.shield_strength),
            "is_powershield": float(bool(player_state.is_powershield)),
            "action_frame": int(player_state.action_frame),
            "is_reflect_active": float(bool(player_state.is_reflect_active)),
            "is_subaction_invulnerable": float(bool(player_state.is_subaction_invulnerable)),
            "is_fastfalling": float(bool(player_state.is_fastfalling)),
            "is_defender_in_hitlag": float(bool(player_state.is_defender_in_hitlag)),
            "is_in_hitlag": float(bool(player_state.is_in_hitlag)),
            "is_holding_character": float(bool(player_state.is_holding_character)),
            "is_shield_active": float(bool(player_state.is_shield_active)),
            "is_in_hitstun": float(bool(player_state.is_in_hitstun)),
            "is_dead": float(bool(player_state.is_dead)),
            "is_offscreen": float(bool(player_state.is_offscreen)),
            "invulnerable": float(bool(player_state.invulnerable)),
            "hitlag_left": int(player_state.hitlag_left),
            "hitstun_frames_left": int(player_state.hitstun_frames_left),
            "jumps_left": int(player_state.jumps_left),
            "speed_air_x_self": float(player_state.speed_air_x_self),
            "speed_y_self": float(player_state.speed_y_self),
            "speed_x_attack": float(player_state.speed_x_attack),
            "speed_y_attack": float(player_state.speed_y_attack),
            "speed_ground_x_self": float(player_state.speed_ground_x_self),
            "off_stage": float(bool(player_state.off_stage)),
            "l_cancel_status": int(player_state.l_cancel_status),
        }


class ReplayExtractor:
    """Orchestrates replay file processing and feature extraction."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.player_extractor = PlayerFeatureExtractor(settings)
        self.logger = get_logger("preprocessing.extractor", settings)

    def file_hash(self, path: str | Path, algo: str = "md5") -> int:
        """
        Compute a 32-bit integer hash of a file, suitable as an identifier.

        Args:
            path: Path to the file as a str or pathlib.Path.
            algo: Hash algorithm (default: md5 for speed).

        Returns:
            32-bit unsigned integer hash of the file.
        """
        data = Path(path).read_bytes()
        val = int.from_bytes(hashlib.new(algo, data).digest()[:4], "little", signed=False)
        return val

    def extract_replay(self, replay_path: str) -> Result[List[dict], str]:
        """
        Extract frames from a single replay file.

        Args:
            replay_path: Path to the SLP replay file

        Returns:
            Result containing list of frame data or error message
        """
        guard_clause(replay_path, "Replay path cannot be empty")
        replay_path_obj = Path(replay_path)
        guard_clause(replay_path_obj.exists(), f"Replay file does not exist: {replay_path}")
        guard_clause(replay_path_obj.is_file(), f"Path is not a file: {replay_path}")

        try:
            console = Console(path=replay_path, is_dolphin=False, allow_old_version=True)
            console.connect()
        except Exception as e:
            self.logger.debug(f"Error connecting to console for {replay_path}: {e}")
            return Err(f"Failed to connect to console: {e}")

        frames = []
        try:
            replay_hash = self.file_hash(replay_path)
            replay_filename = Path(replay_path).name
            current_game_state: GameState = console.step()

            while current_game_state is not None:
                frame_result = self._extract_frame(current_game_state, replay_hash, replay_filename)
                if frame_result.is_ok():
                    frames.append(frame_result.unwrap())
                else:
                    self.logger.warning(f"Failed to extract frame: {frame_result.unwrap_or('Unknown error')}")
                current_game_state: GameState = console.step()

        except Exception as e:
            self.logger.error(f"Error processing replay {replay_path}: {e}")
            return Err(f"Processing failed: {e}")
        finally:
            console.stop()

        self.logger.info(f"Successfully extracted {len(frames)} frames from {replay_path}")
        return Ok(frames)

    def _extract_frame(self, game_state: GameState, replay_hash: int, replay_filename: str) -> Result[dict, str]:
        """
        Extract data from a single frame.

        Args:
            game_state: Current game state from libmelee
            replay_hash: Hash of the replay file
            replay_filename: Name of the replay file

        Returns:
            Result containing frame data or error message
        """
        try:
            # Get players sorted by port
            players = sorted(game_state.players.keys())
            guard_clause(len(players) == 2, f"Expected 2 players, got {len(players)}")

            p1_port, p2_port = players

            # Extract features for both players
            p1_features = self.player_extractor.extract_player_features(
                game_state.players[p1_port], game_state.players[p1_port].controller_state
            )
            p2_features = self.player_extractor.extract_player_features(
                game_state.players[p2_port], game_state.players[p2_port].controller_state
            )

            # Combine into single frame record
            frame_data = {
                "replay_hash": replay_hash,
                "replay_filename": replay_filename,
                "frame": game_state.frame + 123,  # Preprocessed frame
                "stage": game_state.stage.value,
                "distance": float(game_state.distance),
                **{f"p1_{k}": v for k, v in p1_features.items()},
                **{f"p2_{k}": v for k, v in p2_features.items()},
            }

            return Ok(frame_data)

        except Exception as e:
            return Err(f"Frame extraction failed: {e}")
