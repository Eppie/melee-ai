"""Utilities for converting game state to model inputs and applying model outputs to
controller actions."""

from __future__ import annotations

from typing import Sequence

import numpy as np

import melee
from melee import enums

# Order of features expected by the trained model
FEATURE_COLUMNS = [
    'frame_id',
    'p1_has_temp_intang',
    'p1_is_fastfalling',
    'p1_defender_hitlag',
    'p1_in_hitlag',
    'p1_is_grabbing',
    'p1_shield_active',
    'p1_in_hitstun',
    'p1_shield_touch',
    'p1_powershield',
    'p1_is_dead',
    'p1_offscreen',
    'p1_pos_x',
    'p1_pos_y',
    'p1_action_state',
    'p1_percent',
    'p1_stocks',
    'p1_character',
    'p2_has_temp_intang',
    'p2_is_fastfalling',
    'p2_defender_hitlag',
    'p2_in_hitlag',
    'p2_is_grabbing',
    'p2_shield_active',
    'p2_in_hitstun',
    'p2_shield_touch',
    'p2_powershield',
    'p2_is_dead',
    'p2_offscreen',
    'p2_pos_x',
    'p2_pos_y',
    'p2_action_state',
    'p2_percent',
    'p2_stocks',
    'p2_character',
    'p2_btn_l_analog',
    'p2_btn_r_analog',
    'p2_btn_a',
    'p2_btn_b',
    'p2_btn_x',
    'p2_btn_y',
    'p2_btn_z',
    'p2_btn_l',
    'p2_btn_r',
    'p2_pre_joystick_x',
    'p2_pre_joystick_y',
    'p2_pre_cstick_x',
    'p2_pre_cstick_y',
]

# Order of target variables predicted by the model
TARGET_COLUMNS = [
    'p1_btn_a',
    'p1_btn_b',
    'p1_btn_x',
    'p1_btn_l',
    'p1_btn_z',
    'p1_btn_l_analog',
    'p1_btn_r_analog',
    'p1_pre_joystick_x',
    'p1_pre_joystick_y',
    'p1_pre_cstick_x',
    'p1_pre_cstick_y',
    'p1_btn_y',
    'p1_btn_r',
]


def gamestate_to_input(gs: melee.gamestate.GameState, *, p1_port: int = 1, p2_port: int = 2) -> np.ndarray:
    """Convert the current :class:`~melee.gamestate.GameState` into a feature vector.

    Parameters
    ----------
    gs:
        Current game state object provided by ``libmelee``.
    p1_port:
        Controller port of the bot (the player we are predicting actions for).
    p2_port:
        Controller port of the opponent.

    Returns
    -------
    np.ndarray
        A 1-D float32 array with features ordered as ``FEATURE_COLUMNS``.
    """
    p1 = gs.players.get(p1_port)
    p2 = gs.players.get(p2_port)
    if p1 is None or p2 is None:
        raise ValueError("Both player ports must be present in the game state")

    def _flags(pl: melee.gamestate.PlayerState) -> list[float]:
        return [
            float(pl.invulnerable),
            float(pl.is_fastfalling),
            float(pl.is_defender_in_hitlag),
            float(pl.is_in_hitlag),
            float(pl.is_holding_character),
            float(pl.is_shield_active),
            float(pl.is_in_hitstun),
            float(pl.is_touching_shield),
            float(pl.is_powershield),
            float(pl.is_dead),
            float(pl.is_offscreen),
        ]

    p1_flags = _flags(p1)
    p2_flags = _flags(p2)

    feats = [
        float(gs.frame),
        *p1_flags,
        float(p1.position.x),
        float(p1.position.y),
        float(p1.action.value if hasattr(p1.action, "value") else p1.action),
        float(p1.percent),
        float(p1.stock),
        float(p1.character.value if hasattr(p1.character, "value") else p1.character),
        *p2_flags,
        float(p2.position.x),
        float(p2.position.y),
        float(p2.action.value if hasattr(p2.action, "value") else p2.action),
        float(p2.percent),
        float(p2.stock),
        float(p2.character.value if hasattr(p2.character, "value") else p2.character),
        float(p2.controller_state.l_shoulder),
        float(p2.controller_state.r_shoulder),
        float(p2.controller_state.button[enums.Button.BUTTON_A]),
        float(p2.controller_state.button[enums.Button.BUTTON_B]),
        float(p2.controller_state.button[enums.Button.BUTTON_X]),
        float(p2.controller_state.button[enums.Button.BUTTON_Y]),
        float(p2.controller_state.button[enums.Button.BUTTON_Z]),
        float(p2.controller_state.button[enums.Button.BUTTON_L]),
        float(p2.controller_state.button[enums.Button.BUTTON_R]),
        float(p2.controller_state.main_stick[0]),
        float(p2.controller_state.main_stick[1]),
        float(p2.controller_state.c_stick[0]),
        float(p2.controller_state.c_stick[1]),
    ]

    return np.asarray(feats, dtype=np.float32)


def apply_model_output(controller: melee.controller.Controller, outputs: Sequence[float], *, threshold: float = 0.5) -> None:
    """Map model output values to controller actions.

    Parameters
    ----------
    controller:
        ``libmelee`` :class:`~melee.controller.Controller` instance to send inputs to.
    outputs:
        Sequence of predictions in the order given by ``TARGET_COLUMNS``.
    threshold:
        Probability threshold for pressing digital buttons.
    """
    if len(outputs) != len(TARGET_COLUMNS):
        raise ValueError(f"Expected {len(TARGET_COLUMNS)} outputs, got {len(outputs)}")

    # Digital button predictions – press if above threshold, otherwise release.
    button_map = {
        enums.Button.BUTTON_A: outputs[0],
        enums.Button.BUTTON_B: outputs[1],
        enums.Button.BUTTON_X: outputs[2],
        enums.Button.BUTTON_L: outputs[3],
        enums.Button.BUTTON_Z: outputs[4],
        enums.Button.BUTTON_Y: outputs[11],
        enums.Button.BUTTON_R: outputs[12],
    }
    for btn, val in button_map.items():
        if val >= threshold:
            controller.press_button(btn)
        else:
            controller.release_button(btn)

    # Analog shoulders
    controller.press_shoulder(enums.Button.BUTTON_L, float(outputs[5]))
    controller.press_shoulder(enums.Button.BUTTON_R, float(outputs[6]))

    # Analog sticks
    controller.tilt_analog(enums.Button.BUTTON_MAIN, float(outputs[7]), float(outputs[8]))
    controller.tilt_analog(enums.Button.BUTTON_C, float(outputs[9]), float(outputs[10]))
