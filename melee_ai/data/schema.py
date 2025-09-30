"""
Data schema definitions for Melee AI.

This module defines the structured data format used throughout the pipeline,
from raw replay extraction to training data.
"""

import dataclasses
from typing import List, Optional

import numpy as np


@dataclasses.dataclass
class Row:
    """A single frame of game data from a replay."""

    # Common fields
    replay_hash: Optional[np.uint32] = None
    replay_filename: Optional[str] = None
    frame: np.int32 = 0
    stage: np.int32 = 0
    distance: np.float32 = 0.0

    # Player 1 fields
    p1_action: np.int32 = 0
    p1_character: np.int32 = 0
    p1_position_x: np.float32 = 0.0
    p1_position_y: np.float32 = 0.0
    p1_percent: np.int32 = 0
    p1_stock: np.int32 = 0
    p1_facing: np.float32 = 0.0
    p1_on_ground: np.float32 = 0.0
    p1_button_a: np.float32 = 0.0
    p1_button_b: np.float32 = 0.0
    p1_button_xy: np.float32 = 0.0
    p1_button_z: np.float32 = 0.0
    p1_button_lr: np.float32 = 0.0
    p1_main_stick_x: np.float32 = 0.0
    p1_main_stick_y: np.float32 = 0.0
    p1_c_stick_x: np.float32 = 0.0
    p1_c_stick_y: np.float32 = 0.0
    p1_shoulder_analog: np.float32 = 0.0
    p1_shield_strength: np.float32 = 0.0
    p1_is_powershield: np.float32 = 0.0
    p1_action_frame: np.int32 = 0
    p1_is_reflect_active: np.float32 = 0.0
    p1_is_subaction_invulnerable: np.float32 = 0.0
    p1_is_fastfalling: np.float32 = 0.0
    p1_is_defender_in_hitlag: np.float32 = 0.0
    p1_is_in_hitlag: np.float32 = 0.0
    p1_is_holding_character: np.float32 = 0.0
    p1_is_shield_active: np.float32 = 0.0
    p1_is_in_hitstun: np.float32 = 0.0
    p1_is_dead: np.float32 = 0.0
    p1_is_offscreen: np.float32 = 0.0
    p1_invulnerable: np.float32 = 0.0
    p1_hitlag_left: np.int32 = 0
    p1_hitstun_frames_left: np.int32 = 0
    p1_jumps_left: np.int32 = 0
    p1_speed_air_x_self: np.float32 = 0.0
    p1_speed_y_self: np.float32 = 0.0
    p1_speed_x_attack: np.float32 = 0.0
    p1_speed_y_attack: np.float32 = 0.0
    p1_speed_ground_x_self: np.float32 = 0.0
    p1_off_stage: np.float32 = 0.0
    p1_l_cancel_status: np.int32 = 0

    # Player 2 fields (same structure as P1)
    p2_action: np.int32 = 0
    p2_character: np.int32 = 0
    p2_position_x: np.float32 = 0.0
    p2_position_y: np.float32 = 0.0
    p2_percent: np.int32 = 0
    p2_stock: np.int32 = 0
    p2_facing: np.float32 = 0.0
    p2_on_ground: np.float32 = 0.0
    p2_button_a: np.float32 = 0.0
    p2_button_b: np.float32 = 0.0
    p2_button_xy: np.float32 = 0.0
    p2_button_z: np.float32 = 0.0
    p2_button_lr: np.float32 = 0.0
    p2_main_stick_x: np.float32 = 0.0
    p2_main_stick_y: np.float32 = 0.0
    p2_c_stick_x: np.float32 = 0.0
    p2_c_stick_y: np.float32 = 0.0
    p2_shoulder_analog: np.float32 = 0.0
    p2_shield_strength: np.float32 = 0.0
    p2_is_powershield: np.float32 = 0.0
    p2_action_frame: np.int32 = 0
    p2_is_reflect_active: np.float32 = 0.0
    p2_is_subaction_invulnerable: np.float32 = 0.0
    p2_is_fastfalling: np.float32 = 0.0
    p2_is_defender_in_hitlag: np.float32 = 0.0
    p2_is_in_hitlag: np.float32 = 0.0
    p2_is_holding_character: np.float32 = 0.0
    p2_is_shield_active: np.float32 = 0.0
    p2_is_in_hitstun: np.float32 = 0.0
    p2_is_dead: np.float32 = 0.0
    p2_is_offscreen: np.float32 = 0.0
    p2_invulnerable: np.float32 = 0.0
    p2_hitlag_left: np.int32 = 0
    p2_hitstun_frames_left: np.int32 = 0
    p2_jumps_left: np.int32 = 0
    p2_speed_air_x_self: np.float32 = 0.0
    p2_speed_y_self: np.float32 = 0.0
    p2_speed_x_attack: np.float32 = 0.0
    p2_speed_y_attack: np.float32 = 0.0
    p2_speed_ground_x_self: np.float32 = 0.0
    p2_off_stage: np.float32 = 0.0
    p2_l_cancel_status: np.int32 = 0

    def to_numpy_array(self) -> np.ndarray:
        """Convert row to numpy array for training."""
        # This would be implemented to convert the dataclass to a structured array
        # For now, return a placeholder
        return np.array([0])  # Placeholder

    @classmethod
    def from_dict(cls, data: dict) -> 'Row':
        """Create Row from dictionary."""
        return cls(**data)

    def to_dict(self) -> dict:
        """Convert Row to dictionary."""
        return dataclasses.asdict(self)


class Schema:
    """Defines the structure and validation of training data."""

    # Define the expected fields and their types
    FIELD_TYPES = {
        # Common fields
        "replay_hash": np.uint32,
        "replay_filename": str,
        "frame": np.int32,
        "stage": np.int32,
        "distance": np.float32,

        # Player 1 fields
        **{f"p1_{field}": dtype for field, dtype in {
            "action": np.int32,
            "character": np.int32,
            "position_x": np.float32,
            "position_y": np.float32,
            "percent": np.int32,
            "stock": np.int32,
            "facing": np.float32,
            "on_ground": np.float32,
            "button_a": np.float32,
            "button_b": np.float32,
            "button_xy": np.float32,
            "button_z": np.float32,
            "button_lr": np.float32,
            "main_stick_x": np.float32,
            "main_stick_y": np.float32,
            "c_stick_x": np.float32,
            "c_stick_y": np.float32,
            "shoulder_analog": np.float32,
            "shield_strength": np.float32,
            "is_powershield": np.float32,
            "action_frame": np.int32,
            "is_reflect_active": np.float32,
            "is_subaction_invulnerable": np.float32,
            "is_fastfalling": np.float32,
            "is_defender_in_hitlag": np.float32,
            "is_in_hitlag": np.float32,
            "is_holding_character": np.float32,
            "is_shield_active": np.float32,
            "is_in_hitstun": np.float32,
            "is_dead": np.float32,
            "is_offscreen": np.float32,
            "invulnerable": np.float32,
            "hitlag_left": np.int32,
            "hitstun_frames_left": np.int32,
            "jumps_left": np.int32,
            "speed_air_x_self": np.float32,
            "speed_y_self": np.float32,
            "speed_x_attack": np.float32,
            "speed_y_attack": np.float32,
            "speed_ground_x_self": np.float32,
            "off_stage": np.float32,
            "l_cancel_status": np.int32,
        }.items()},

        # Player 2 fields (same structure)
        **{f"p2_{field}": dtype for field, dtype in {
            "action": np.int32,
            "character": np.int32,
            "position_x": np.float32,
            "position_y": np.float32,
            "percent": np.int32,
            "stock": np.int32,
            "facing": np.float32,
            "on_ground": np.float32,
            "button_a": np.float32,
            "button_b": np.float32,
            "button_xy": np.float32,
            "button_z": np.float32,
            "button_lr": np.float32,
            "main_stick_x": np.float32,
            "main_stick_y": np.float32,
            "c_stick_x": np.float32,
            "c_stick_y": np.float32,
            "shoulder_analog": np.float32,
            "shield_strength": np.float32,
            "is_powershield": np.float32,
            "action_frame": np.int32,
            "is_reflect_active": np.float32,
            "is_subaction_invulnerable": np.float32,
            "is_fastfalling": np.float32,
            "is_defender_in_hitlag": np.float32,
            "is_in_hitlag": np.float32,
            "is_holding_character": np.float32,
            "is_shield_active": np.float32,
            "is_in_hitstun": np.float32,
            "is_dead": np.float32,
            "is_offscreen": np.float32,
            "invulnerable": np.float32,
            "hitlag_left": np.int32,
            "hitstun_frames_left": np.int32,
            "jumps_left": np.int32,
            "speed_air_x_self": np.float32,
            "speed_y_self": np.float32,
            "speed_x_attack": np.float32,
            "speed_y_attack": np.float32,
            "speed_ground_x_self": np.float32,
            "off_stage": np.float32,
            "l_cancel_status": np.int32,
        }.items()},
    }

    @classmethod
    def validate_row(cls, row: Row) -> bool:
        """Validate that a row conforms to the expected schema."""
        # Basic validation - check required fields exist and have correct types
        return True  # Placeholder implementation

    @classmethod
    def get_feature_columns(cls) -> List[str]:
        """Get list of feature column names."""
        return [name for name in cls.FIELD_TYPES.keys()
                if name not in ["replay_hash", "replay_filename"]]

    @classmethod
    def get_target_columns(cls) -> List[str]:
        """Get list of target column names (controller inputs)."""
        return [
            "p1_main_stick_x", "p1_main_stick_y",
            "p1_c_stick_x", "p1_c_stick_y",
            "p1_button_a", "p1_button_b", "p1_button_xy", "p1_button_z", "p1_button_lr",
            "p1_shoulder_analog"
        ]
