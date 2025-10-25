from __future__ import annotations

import numpy as np

from libmelee.melee import enums
from libmelee.melee.enums import Action

MAX_FRAMES: int = (60 * 60 * 8) + 123  # Full 8 minute replay


def _preprocess_frame(frame: int) -> np.int32:
    processed_frame = frame + 123
    assert (
        0 <= processed_frame <= MAX_FRAMES
    ), f"Processed frame {processed_frame} is out of range (0, {MAX_FRAMES})"
    return np.int32(processed_frame)


def _preprocess_stage(stage: enums.Stage) -> np.int32:
    return np.int32(
        {
            enums.Stage.FINAL_DESTINATION: 0,
            enums.Stage.BATTLEFIELD: 1,
            enums.Stage.POKEMON_STADIUM: 2,
            enums.Stage.DREAMLAND: 3,
            enums.Stage.FOUNTAIN_OF_DREAMS: 4,
            enums.Stage.YOSHIS_STORY: 5,
        }[stage]
    )


def _preprocess_character(character: enums.Character) -> np.int32:
    assert 0 <= character.value <= 26
    return np.int32(character.value)


def _preprocess_action(action: Action) -> np.int32:
    assert 0 <= action.value <= 397  # 0x18d
    return np.int32(action.value)


def _preprocess_x_y_buttons(button_x: bool, button_y: bool) -> np.float32:
    """The X and Y buttons both indicate "jump", so we only need one of them"""
    return np.float32(np.logical_or(button_x, button_y))


def _preprocess_l_r_buttons(button_l: bool, button_r: bool) -> np.float32:
    """The L and R buttons both indicate "shield", so we only need one of them"""
    return np.float32(np.logical_or(button_l, button_r))
