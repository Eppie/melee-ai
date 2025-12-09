"""Parse .slp replay files to extract raw features.

Uses libmelee to parse replay files and extract frame-by-frame game state.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, Optional

# Add the parent directory to sys.path to import the local libmelee folder
sys.path.insert(0, str(Path(__file__).parents[2]))

import libmelee
from libmelee.melee import enums
from libmelee.melee.console import Console
from ..constants import CHARACTER_PHYSICS_FIELDS, PLAYER_FEATURES_DIM, GLOBAL_RELATIONAL_FEATURES_DIM, TOTAL_FEATURE_DIM, STAGE_ID_GLOBAL_IDX, P1_CHAR_ID_GLOBAL_IDX, P2_CHAR_ID_GLOBAL_IDX
from .features import compute_relational_features


def parse_slp_to_features(slp_path: Path) -> Dict[str, np.ndarray]:
    """Parse .slp file and extract all relevant features.

    Args:
        slp_path: Path to .slp replay file

    Returns:
        Dictionary containing:
            - 'features': [n_frames, feature_dim] array of per-frame features

    Notes:
        - Uses libmelee.Console for parsing
        - Computes derived features (velocity, input deltas, button onsets)
        - Handles first frame initialization
        - Only uses libmelee, no dependencies on parent repo
    """
    console = Console(is_dolphin=False, allow_old_version=True, path=str(slp_path))
    if not console.connect():
        raise ValueError(f"Failed to connect to SLP file: {slp_path}")

    all_frames_features = []
    p1_prev_state: Optional[Dict[str, float]] = None
    p2_prev_state: Optional[Dict[str, float]] = None
    p1_prev_inputs: Optional[Dict[str, float]] = None
    p2_prev_inputs: Optional[Dict[str, float]] = None

    stage_enum: Optional[enums.Stage] = None
    stage_value: Optional[int] = None
    p1_char_id: Optional[int] = None
    p2_char_id: Optional[int] = None
    stage_bounds: Optional[Dict[str, float]] = None
    p1_char_physics: Dict[str, float] = {}
    p2_char_physics: Dict[str, float] = {}
    p1_key: Optional[int] = None
    p2_key: Optional[int] = None

    while True:
        gamestate = console.step()
        if gamestate is None:
            break
        if not is_valid_frame(gamestate):
            continue

        # Determine consistent player ordering and metadata on first valid frame
        if stage_enum is None:
            stage_enum = gamestate.stage
            stage_value = int(stage_enum.value) if hasattr(stage_enum, "value") else int(stage_enum)
            stage_bounds = get_stage_bounds(stage_enum)

            player_keys = sorted(gamestate.players.keys())
            if len(player_keys) < 2:
                continue
            p1_key, p2_key = player_keys[:2]

            p1_char_enum = gamestate.players[p1_key].character
            p2_char_enum = gamestate.players[p2_key].character
            p1_char_id = int(p1_char_enum.value)
            p2_char_id = int(p2_char_enum.value)
            p1_char_physics = load_character_physics(p1_char_enum)
            p2_char_physics = load_character_physics(p2_char_enum)

        if p1_key not in gamestate.players or p2_key not in gamestate.players:
            continue

        p1_player = gamestate.players[p1_key]
        p2_player = gamestate.players[p2_key]

        p1_features = extract_player_features(
            p1_player,
            p1_prev_state,
            p1_prev_inputs,
            p1_char_physics,
            stage_bounds,
        )
        p2_features = extract_player_features(
            p2_player,
            p2_prev_state,
            p2_prev_inputs,
            p2_char_physics,
            stage_bounds,
        )

        # Compute relational features
        relational_features = compute_relational_features(
            p1_player.position.x, p1_player.position.y,
            p2_player.position.x, p2_player.position.y,
        )

        # Global static features (IDs)
        global_ids = np.array([stage_value, p1_char_id, p2_char_id], dtype=np.float32)

        # Concatenate all features into a single vector
        # Order: [p1_features, p2_features, relational_features, global_ids]
        current_frame_features = np.concatenate([
            p1_features,
            p2_features,
            relational_features,
            global_ids
        ])
        all_frames_features.append(current_frame_features)

        # Update previous states for next iteration
        p1_prev_state = {
            'x': p1_player.position.x,
            'y': p1_player.position.y,
        }
        p2_prev_state = {
            'x': p2_player.position.x,
            'y': p2_player.position.y,
        }
        p1_prev_inputs = {
            'joystick_x': p1_player.controller_state.main_stick[0],
            'joystick_y': p1_player.controller_state.main_stick[1],
            'cstick_x': p1_player.controller_state.c_stick[0],
            'cstick_y': p1_player.controller_state.c_stick[1],
            'shoulder': max(p1_player.controller_state.l_shoulder, p1_player.controller_state.r_shoulder),
        }
        p2_prev_inputs = {
            'joystick_x': p2_player.controller_state.main_stick[0],
            'joystick_y': p2_player.controller_state.main_stick[1],
            'cstick_x': p2_player.controller_state.c_stick[0],
            'cstick_y': p2_player.controller_state.c_stick[1],
            'shoulder': max(p2_player.controller_state.l_shoulder, p2_player.controller_state.r_shoulder),
        }
    
    # Check if any frames were processed
    if not all_frames_features:
        raise ValueError(f"No valid frames found in replay: {slp_path}")

    return {
        'features': np.array(all_frames_features, dtype=np.float32),
        'stage': int(stage_enum.value) if stage_enum is not None else 0, # Keep for metadata in NPZ, will be redundant in 'features' array
        'p1_char': p1_char_id if p1_char_id is not None else 0, # Keep for metadata in NPZ
        'p2_char': p2_char_id if p2_char_id is not None else 0, # Keep for metadata in NPZ
    }


def is_valid_frame(gamestate) -> bool:
    """Check if game state is valid for feature extraction.

    Args:
        gamestate: libmelee.GameState object

    Returns:
        True if frame should be processed, False otherwise

    Notes:
        - Skip menu states, paused frames, etc.
        - Require both players to be active
    """
    if gamestate.menu_state not in [enums.Menu.IN_GAME, enums.Menu.SUDDEN_DEATH]:
        return False
    if len(gamestate.players) < 2:
        return False
    # Require both characters to be set to real characters
    player_objs = list(gamestate.players.values())
    if any(player.character is enums.Character.UNKNOWN_CHARACTER for player in player_objs[:2]):
        return False
    return True


def extract_player_features(
    player,
    prev_player_state: Optional[Dict],
    prev_inputs: Optional[Dict],
    character_physics: Dict[str, float],
    stage_bounds: Dict[str, float],
) -> np.ndarray:
    """Extract features for a single player in a single frame.

    Args:
        player: libmelee.PlayerState object
        prev_player_state: Previous frame's player state (for velocity computation)
        prev_inputs: Previous frame's inputs (for delta/onset computation)
        character_physics: Physics constants from characterdata.csv
        stage_bounds: Stage-specific bounds (ledges, blast zones)

    Returns:
        Feature vector [feature_dim] for this player

    Features extracted:
        - Position (X, Y)
        - Velocity (X, Y) - computed from position delta
        - Facing direction
        - Action state ID
        - Grounded/airborne
        - Percent, shield size, jumps remaining
        - Controller inputs (joystick, c-stick, shoulder, buttons)
        - Input changes (button onsets, analog deltas)
        - Stage proximity (distance to ledges, blast zones)
        - Character physics constants (8 values)
    """
    features = []

    # 1. Position (X, Y)
    x, y = player.position.x, player.position.y
    features.extend([x, y])

    # 2. Velocity (X, Y)
    if prev_player_state:
        vel_x = x - prev_player_state['x']
        vel_y = y - prev_player_state['y']
    else:
        vel_x, vel_y = 0.0, 0.0
    features.extend([vel_x, vel_y])

    # 3. Facing direction
    features.append(player.facing)

    # 4. Action state ID
    action_id = int(player.action.value) if hasattr(player.action, "value") else int(player.action)
    features.append(action_id)

    # 5. Grounded/airborne
    features.append(float(player.on_ground))

    # 6. Combat state
    features.extend([player.percent, player.shield_strength, player.jumps_left])

    # 7. Controller Inputs
    controller_state = player.controller_state
    features.extend([
        controller_state.main_stick[0], controller_state.main_stick[1],
        controller_state.c_stick[0], controller_state.c_stick[1],
        max(controller_state.l_shoulder, controller_state.r_shoulder),
    ])
    # Buttons (A, B, X/Y, Z, L/R) - 5 binary flags
    buttons = np.array([
        float(controller_state.button[enums.Button.BUTTON_A]),
        float(controller_state.button[enums.Button.BUTTON_B]),
        float(controller_state.button[enums.Button.BUTTON_X] or controller_state.button[enums.Button.BUTTON_Y]),
        float(controller_state.button[enums.Button.BUTTON_Z]),
        float(controller_state.button[enums.Button.BUTTON_L] or controller_state.button[enums.Button.BUTTON_R]),
    ])
    features.extend(buttons.tolist())

    # 8. Input Changes
    current_inputs_dict = {
        'joystick_x': controller_state.main_stick[0],
        'joystick_y': controller_state.main_stick[1],
        'cstick_x': controller_state.c_stick[0],
        'cstick_y': controller_state.c_stick[1],
        'shoulder': max(controller_state.l_shoulder, controller_state.r_shoulder),
    }
    features.extend(compute_input_deltas(current_inputs_dict, prev_inputs).tolist())

    # 9. Stage Proximity
    features.extend(compute_stage_proximity(x, y, stage_bounds).tolist())

    # 10. Character Physics
    features.extend([character_physics[field] for field in CHARACTER_PHYSICS_FIELDS])

    if len(features) != PLAYER_FEATURES_DIM:
        raise ValueError(f"Expected {PLAYER_FEATURES_DIM} features per player, got {len(features)}")
    return np.array(features, dtype=np.float32)


def compute_stage_proximity(
    x: float,
    y: float,
    stage_bounds: Dict[str, float],
) -> np.ndarray:
    """Compute distance to stage features (ledges, blast zones).

    Args:
        x: X position
        y: Y position
        stage_bounds: Dict with keys: left_ledge, right_ledge, top_blastzone, etc.

    Returns:
        [4] array: [dist_to_left_ledge, dist_to_right_ledge,
                    dist_to_floor, dist_to_top_blastzone]
    """
    dist_to_left_ledge = x - stage_bounds['left_ledge']
    dist_to_right_ledge = stage_bounds['right_ledge'] - x
    dist_to_floor = y - stage_bounds['stage_floor']
    dist_to_top_blastzone = stage_bounds['top_blastzone'] - y

    return np.array([
        dist_to_left_ledge,
        dist_to_right_ledge,
        dist_to_floor,
        dist_to_top_blastzone,
    ])


def compute_button_onsets(
    current_buttons: np.ndarray,
    prev_buttons: Optional[np.ndarray],
) -> np.ndarray:
    """Compute button press onsets (0->1 transitions).

    Args:
        current_buttons: [5] binary button states
        prev_buttons: [5] previous frame button states (or None for first frame)

    Returns:
        [5] binary onset flags
    """
    if prev_buttons is None:
        return np.zeros_like(current_buttons)
    return (current_buttons > 0) & (prev_buttons == 0)


def compute_input_deltas(
    current_inputs: Dict[str, float],
    prev_inputs: Optional[Dict[str, float]],
) -> np.ndarray:
    """Compute analog input changes from previous frame.

    Args:
        current_inputs: Dict with keys: joystick_x, joystick_y, cstick_x, cstick_y, shoulder
        prev_inputs: Previous frame inputs (or None)

    Returns:
        [5] array of deltas for each analog input
    """
    if prev_inputs is None:
        return np.zeros(5)

    joystick_x_delta = current_inputs['joystick_x'] - prev_inputs['joystick_x']
    joystick_y_delta = current_inputs['joystick_y'] - prev_inputs['joystick_y']
    cstick_x_delta = current_inputs['cstick_x'] - prev_inputs['cstick_x']
    cstick_y_delta = current_inputs['cstick_y'] - prev_inputs['cstick_y']
    shoulder_delta = current_inputs['shoulder'] - prev_inputs['shoulder']

    return np.array([
        joystick_x_delta,
        joystick_y_delta,
        cstick_x_delta,
        cstick_y_delta,
        shoulder_delta,
    ])


def load_character_physics(character_id: int) -> Dict[str, float]:
    """Load character physics constants from characterdata.csv.

    Args:
        character_id: Character ID (from libmelee.Character enum)

    Returns:
        Dict mapping physics field names to values

    Notes:
        - Reads from libmelee/melee/characterdata.csv
        - Returns 8 constants defined in constants.CHARACTER_PHYSICS_FIELDS
    """
    # Dynamically find the path to libmelee's characterdata.csv
    libmelee_path = Path(libmelee.__file__).parent
    character_data_path = libmelee_path / 'melee' / 'characterdata.csv'

    df = pd.read_csv(character_data_path)
    # libmelee stores character IDs under CharacterIndex
    cid = character_id.value if hasattr(character_id, "value") else character_id
    character_row = df[df['CharacterIndex'] == int(cid)].iloc[0]

    physics_data = {
        field: character_row[field]
        for field in CHARACTER_PHYSICS_FIELDS
    }
    return physics_data


def get_stage_bounds(stage_id: int) -> Dict[str, float]:
    """Get stage-specific bounds (ledges, blast zones).

    Args:
        stage_id: Stage ID (from libmelee.Stage enum)

    Returns:
        Dict with keys: left_ledge, right_ledge, stage_floor, top_blastzone, etc.

    Notes:
        - May need to hardcode common stage bounds or extract from libmelee
        - Used for stage proximity features
    """
    # Hardcoded bounds for common competitive stages
    stage_data = {
        enums.Stage.FINAL_DESTINATION: {
            'left_ledge': -60.0, 'right_ledge': 60.0, 'stage_floor': 0.0, 'top_blastzone': 190.0,
            'left_blastzone': -240.0, 'right_blastzone': 240.0, 'bottom_blastzone': -110.0,
            'half_width': 85.0, 'half_height': 70.0 # Approximate
        },
        enums.Stage.BATTLEFIELD: {
            'left_ledge': -58.0, 'right_ledge': 58.0, 'stage_floor': 0.0, 'top_blastzone': 180.0,
            'left_blastzone': -240.0, 'right_blastzone': 240.0, 'bottom_blastzone': -100.0,
            'half_width': 80.0, 'half_height': 70.0 # Approximate
        },
        enums.Stage.YOSHIS_STORY: {
            'left_ledge': -45.0, 'right_ledge': 45.0, 'stage_floor': 0.0, 'top_blastzone': 170.0,
            'left_blastzone': -220.0, 'right_blastzone': 220.0, 'bottom_blastzone': -90.0,
            'half_width': 70.0, 'half_height': 60.0 # Approximate
        },
        enums.Stage.DREAMLAND: {
            'left_ledge': -75.0, 'right_ledge': 75.0, 'stage_floor': 0.0, 'top_blastzone': 200.0,
            'left_blastzone': -280.0, 'right_blastzone': 280.0, 'bottom_blastzone': -120.0,
            'half_width': 95.0, 'half_height': 80.0 # Approximate
        },
        enums.Stage.FOUNTAIN_OF_DREAMS: {
            'left_ledge': -55.0, 'right_ledge': 55.0, 'stage_floor': 0.0, 'top_blastzone': 175.0,
            'left_blastzone': -230.0, 'right_blastzone': 230.0, 'bottom_blastzone': -95.0,
            'half_width': 75.0, 'half_height': 65.0 # Approximate
        },
        enums.Stage.POKEMON_STADIUM: {
            'left_ledge': -80.0, 'right_ledge': 80.0, 'stage_floor': 0.0, 'top_blastzone': 210.0,
            'left_blastzone': -260.0, 'right_blastzone': 260.0, 'bottom_blastzone': -130.0,
            'half_width': 100.0, 'half_height': 85.0 # Approximate
        },
    }

    if stage_id in stage_data:
        return stage_data[stage_id]
    else:
        # Fallback for unhandled stages, perhaps a default or error
        # For now, let's return a generic default or raise an error
        raise ValueError(f"Stage ID {stage_id} not found in hardcoded bounds.")
