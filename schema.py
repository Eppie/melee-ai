import dataclasses
from typing import Any, Callable, Dict

import numpy as np

from libmelee.melee import enums
from libmelee.melee.gamestate import GameState, PlayerState
from preprocess import (
    _preprocess_action,
    _preprocess_character,
    _preprocess_l_r_buttons,
    _preprocess_stage,
    _preprocess_x_y_buttons,
)

# Base specs
COMMON_SPEC = [
    ("stage", np.int32),  # Stage enum
]
_COMMON_FIELD_NAMES = [name for name, *_ in COMMON_SPEC]

BUTTONS = [
    ("button_a", np.float32),
    ("button_b", np.float32),
    # Logical OR will be applied to buttons X and Y
    ("button_xy", np.float32),
    ("button_z", np.float32),
    # Logical OR will be applied to buttons L and R
    ("button_lr", np.float32),
]

PLAYER_SPEC = [
    # Core categorical/ids (stored as ints post-preprocessing)
    ("action", np.int32),
    ("character", np.int32),
    # Geometry
    ("position_x", np.float32),
    ("position_y", np.float32),
    # Damage & state bits
    ("percent", np.int32),
    ("facing", np.float32),
    ("on_ground", np.float32),
    # Buttons
    *BUTTONS,
    # Sticks / shoulders
    ("main_stick_x", np.float32),
    ("main_stick_y", np.float32),
    ("c_stick_x", np.float32),
    ("c_stick_y", np.float32),
    ("shoulder_analog", np.float32),  # game treats L/R shoulder identically
    # Additional state
    ("shield_strength", np.float32),
    ("is_fastfalling", np.float32),
    ("is_defender_in_hitlag", np.float32),
    ("is_in_hitlag", np.float32),
    ("is_holding_character", np.float32),
    ("is_shield_active", np.float32),
    ("is_in_hitstun", np.float32),
    ("is_dead", np.float32),
    ("is_offscreen", np.float32),
    ("is_invulnerable", np.float32),
    ("jumps_left", np.int32),
    # TODO: It would be great to have these. Do we have replays with these populated? Added in version 3.5.0
    # ("speed_air_x_self", np.float32),
    # ("speed_y_self", np.float32),
    # ("speed_x_attack", np.float32),
    # ("speed_y_attack", np.float32),
    # ("speed_ground_x_self", np.float32),
    ("off_stage", np.float32),
    ("l_cancel_status", np.int32),  # TODO: Maybe one-hot encode this?
]
_PLAYER_FIELD_NAMES = [name for name, *_ in PLAYER_SPEC]
_BUTTON_A = enums.Button.BUTTON_A
_BUTTON_B = enums.Button.BUTTON_B
_BUTTON_X = enums.Button.BUTTON_X
_BUTTON_Y = enums.Button.BUTTON_Y
_BUTTON_Z = enums.Button.BUTTON_Z
_BUTTON_L = enums.Button.BUTTON_L
_BUTTON_R = enums.Button.BUTTON_R


def _require_controller_state(player: PlayerState):
    controller = player.controller_state
    if controller is None:
        raise ValueError("PlayerState is missing controller_state")
    return controller


COMMON_EXTRACTORS: Dict[str, Callable[[GameState], Any]] = {
    "stage": lambda state: _preprocess_stage(state.stage),
}

_COMMON_SPEC_NAMES = {name for name, _ in COMMON_SPEC}
if _COMMON_SPEC_NAMES != set(COMMON_EXTRACTORS):
    missing = _COMMON_SPEC_NAMES - set(COMMON_EXTRACTORS)
    extra = set(COMMON_EXTRACTORS) - _COMMON_SPEC_NAMES
    raise ValueError(
        f"COMMON_EXTRACTORS mismatch spec. missing={sorted(missing)} extra={sorted(extra)}"
    )


def _extract_common_values(game_state: GameState) -> tuple[Any, ...]:
    return (_preprocess_stage(game_state.stage),)


def _extract_player_values(player: PlayerState) -> tuple[Any, ...]:
    controller = _require_controller_state(player)
    buttons = controller.button
    button_x = bool(buttons[_BUTTON_X])
    button_y = bool(buttons[_BUTTON_Y])
    button_l = bool(buttons[_BUTTON_L])
    button_r = bool(buttons[_BUTTON_R])
    main_stick_x, main_stick_y = controller.main_stick
    c_stick_x, c_stick_y = controller.c_stick
    button_xy = float(button_x or button_y)
    button_lr = float(button_l or button_r)
    return (
        _preprocess_action(player.action),
        _preprocess_character(player.character),
        player.position.x,
        player.position.y,
        int(player.percent),
        float(player.facing),
        float(player.on_ground),
        float(buttons[_BUTTON_A]),
        float(buttons[_BUTTON_B]),
        button_xy,
        float(buttons[_BUTTON_Z]),
        button_lr,
        main_stick_x,
        main_stick_y,
        c_stick_x,
        c_stick_y,
        controller.l_shoulder,
        player.shield_strength,
        float(player.is_fastfalling),
        float(player.is_defender_in_hitlag),
        float(player.is_in_hitlag),
        float(player.is_holding_character),
        float(player.is_shield_active),
        float(player.is_in_hitstun),
        float(player.is_dead),
        float(player.is_offscreen),
        float(player.invulnerable),
        int(player.jumps_left),
        # player.speed_air_x_self,
        # player.speed_y_self,
        # player.speed_x_attack,
        # player.speed_y_attack,
        # player.speed_ground_x_self,
        float(player.off_stage),
        int(player.l_cancel_status),
    )


def extract_common_fields(game_state: GameState) -> dict[str, Any]:
    values = _extract_common_values(game_state)
    return dict(zip(_COMMON_FIELD_NAMES, values))


def extract_player_fields(player: PlayerState) -> dict[str, Any]:
    values = _extract_player_values(player)
    return dict(zip(_PLAYER_FIELD_NAMES, values))


def extract_row(game_state: GameState) -> "Row":
    player_keys = list(game_state.players.keys())
    num_players = len(player_keys)
    if num_players < 2:
        raise ValueError(f"Need at least 2 players, got {num_players}")
    if num_players == 2:
        p1_key, p2_key = player_keys
        if p1_key > p2_key:
            p1_key, p2_key = p2_key, p1_key
    else:
        p1_key, p2_key = sorted(player_keys)[:2]

    p1_state = game_state.players[p1_key]
    p2_state = game_state.players[p2_key]

    values = (
        _extract_common_values(game_state)
        + _extract_player_values(p1_state)
        + _extract_player_values(p2_state)
    )

    return Row(*values)


def _prefixed(spec, prefix: str):
    # spec elements can be (name, type) or (name, type, default/field)
    out = []
    for item in spec:
        if len(item) == 2:
            name, typ = item
            out.append((f"{prefix}{name}", typ))
        else:
            name, typ, default_or_field = item
            out.append((f"{prefix}{name}", typ, default_or_field))
    return out


# Compose all dataclass fields in the final order
_ROW_FIELDS = (
    COMMON_SPEC + _prefixed(PLAYER_SPEC, "p1_") + _prefixed(PLAYER_SPEC, "p2_")
)


def get_feature_names() -> list[str]:
    """Canonical feature ordering for model input."""
    names = []
    for field_name, *_ in COMMON_SPEC:
        names.append(field_name)
    for prefix in ["p1_", "p2_"]:
        for field_name, *_ in PLAYER_SPEC:
            names.append(f"{prefix}{field_name}")
    # Derived features append at the end so legacy column indices remain stable.
    names.append("value_target")
    return names


def get_raw_target_names() -> list[str]:
    """Raw controller targets (continuous) for dataset extraction (P1 only)."""
    controller_fields = {
        "main_stick_x",
        "main_stick_y",
        "c_stick_x",
        "c_stick_y",
        "shoulder_analog",
        "button_a",
        "button_b",
        "button_xy",
        "button_z",
        "button_lr",
    }
    return [f"p1_{field}" for field, *_ in PLAYER_SPEC if field in controller_fields]


def get_target_names() -> list[str]:
    """Quantized controller targets (P1 only) stored in the preprocessed dataset."""
    base_targets = [
        "p1_main_stick_idx",
        "p1_c_stick_idx",
        "p1_shoulder_idx",
        "p1_button_a",
        "p1_button_b",
        "p1_button_xy",
        "p1_button_z",
        "p1_button_lr",
    ]

    return base_targets


# Build the dataclass dynamically (flattened attributes), with slots for memory/perf
Row = dataclasses.make_dataclass("Row", _ROW_FIELDS, slots=True)

# Runtime assert to guarantee p1_/p2_ symmetry on import
_p1_names = [name for (name, *_rest) in _ROW_FIELDS if name.startswith("p1_")]
_p2_names = [name for (name, *_rest) in _ROW_FIELDS if name.startswith("p2_")]
assert [n[3:] for n in _p1_names] == [n[3:] for n in _p2_names], "p1/p2 spec mismatch"

# Convenience: export the authoritative specs, useful elsewhere (e.g., for dtype building)
__all__ = [
    "Row",
    "COMMON_SPEC",
    "PLAYER_SPEC",
    "extract_common_fields",
    "extract_player_fields",
    "extract_row",
    "get_feature_names",
    "get_raw_target_names",
    "get_target_names",
]
