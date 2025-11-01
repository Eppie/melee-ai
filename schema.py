import dataclasses
from typing import Any, Callable, Dict, Optional

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
    # Damage/stock & state bits
    ("percent", np.int32),
    ("stock", np.int32),
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


def _require_controller_state(player: PlayerState):
    controller = player.controller_state
    if controller is None:
        raise ValueError("PlayerState is missing controller_state")
    return controller


def _as_int32(value: Any) -> np.int32:
    return np.int32(value)


def _as_float32(value: Any) -> np.float32:
    return np.float32(value)


def _button_extractor(button: enums.Button) -> Callable[[PlayerState], np.float32]:
    def extractor(player: PlayerState) -> np.float32:
        buttons = _require_controller_state(player).button
        return np.float32(bool(buttons[button]))

    return extractor


COMMON_EXTRACTORS: Dict[str, Callable[[GameState], Any]] = {
    "stage": lambda state: _preprocess_stage(state.stage),
}

PLAYER_EXTRACTORS: Dict[str, Callable[[PlayerState], Any]] = {
    "action": lambda player: _preprocess_action(player.action),
    "character": lambda player: _preprocess_character(player.character),
    "position_x": lambda player: _as_float32(player.position.x),
    "position_y": lambda player: _as_float32(player.position.y),
    "percent": lambda player: _as_int32(player.percent),
    "stock": lambda player: _as_int32(player.stock),
    "facing": lambda player: _as_float32(player.facing),
    "on_ground": lambda player: _as_float32(player.on_ground),
    "button_a": _button_extractor(enums.Button.BUTTON_A),
    "button_b": _button_extractor(enums.Button.BUTTON_B),
    "button_xy": lambda player: _preprocess_x_y_buttons(
        bool(_require_controller_state(player).button[enums.Button.BUTTON_X]),
        bool(_require_controller_state(player).button[enums.Button.BUTTON_Y]),
    ),
    "button_z": _button_extractor(enums.Button.BUTTON_Z),
    "button_lr": lambda player: _preprocess_l_r_buttons(
        bool(_require_controller_state(player).button[enums.Button.BUTTON_L]),
        bool(_require_controller_state(player).button[enums.Button.BUTTON_R]),
    ),
    "main_stick_x": lambda player: _as_float32(
        _require_controller_state(player).main_stick[0]
    ),
    "main_stick_y": lambda player: _as_float32(
        _require_controller_state(player).main_stick[1]
    ),
    "c_stick_x": lambda player: _as_float32(
        _require_controller_state(player).c_stick[0]
    ),
    "c_stick_y": lambda player: _as_float32(
        _require_controller_state(player).c_stick[1]
    ),
    "shoulder_analog": lambda player: _as_float32(
        _require_controller_state(player).l_shoulder
    ),
    "shield_strength": lambda player: _as_float32(player.shield_strength),
    "is_fastfalling": lambda player: _as_float32(player.is_fastfalling),
    "is_defender_in_hitlag": lambda player: _as_float32(player.is_defender_in_hitlag),
    "is_in_hitlag": lambda player: _as_float32(player.is_in_hitlag),
    "is_holding_character": lambda player: _as_float32(player.is_holding_character),
    "is_shield_active": lambda player: _as_float32(player.is_shield_active),
    "is_in_hitstun": lambda player: _as_float32(player.is_in_hitstun),
    "is_dead": lambda player: _as_float32(player.is_dead),
    "is_offscreen": lambda player: _as_float32(player.is_offscreen),
    "is_invulnerable": lambda player: _as_float32(player.invulnerable),
    "jumps_left": lambda player: _as_int32(player.jumps_left),
    # "speed_air_x_self": lambda player: _as_float32(player.speed_air_x_self),
    # "speed_y_self": lambda player: _as_float32(player.speed_y_self),
    # "speed_x_attack": lambda player: _as_float32(player.speed_x_attack),
    # "speed_y_attack": lambda player: _as_float32(player.speed_y_attack),
    # "speed_ground_x_self": lambda player: _as_float32(player.speed_ground_x_self),
    "off_stage": lambda player: _as_float32(player.off_stage),
    "l_cancel_status": lambda player: _as_int32(player.l_cancel_status),
}

_COMMON_SPEC_NAMES = {name for name, _ in COMMON_SPEC}
if _COMMON_SPEC_NAMES != set(COMMON_EXTRACTORS):
    missing = _COMMON_SPEC_NAMES - set(COMMON_EXTRACTORS)
    extra = set(COMMON_EXTRACTORS) - _COMMON_SPEC_NAMES
    raise ValueError(
        f"COMMON_EXTRACTORS mismatch spec. missing={sorted(missing)} extra={sorted(extra)}"
    )

_PLAYER_SPEC_NAMES = {name for name, _ in PLAYER_SPEC}
if _PLAYER_SPEC_NAMES != set(PLAYER_EXTRACTORS):
    missing = _PLAYER_SPEC_NAMES - set(PLAYER_EXTRACTORS)
    extra = set(PLAYER_EXTRACTORS) - _PLAYER_SPEC_NAMES
    raise ValueError(
        f"PLAYER_EXTRACTORS mismatch spec. missing={sorted(missing)} extra={sorted(extra)}"
    )


def extract_common_fields(game_state: GameState) -> dict[str, Any]:
    return {name: COMMON_EXTRACTORS[name](game_state) for name, _ in COMMON_SPEC}


def extract_player_fields(player: PlayerState) -> dict[str, Any]:
    return {name: PLAYER_EXTRACTORS[name](player) for name, _ in PLAYER_SPEC}


def extract_row(game_state: GameState) -> "Row":
    players = sorted(game_state.players.keys())
    if len(players) < 2:
        raise ValueError(f"Need at least 2 players, got {len(players)}")

    p1_state = game_state.players[players[0]]
    p2_state = game_state.players[players[1]]

    fields = {
        **extract_common_fields(game_state),
        **{f"p1_{k}": v for k, v in extract_player_fields(p1_state).items()},
        **{f"p2_{k}": v for k, v in extract_player_fields(p2_state).items()},
    }

    return Row(**fields)


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
    return names


def get_target_names() -> list[str]:
    """Canonical target ordering for model output (P1 controller only)."""
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
    "get_target_names",
]
