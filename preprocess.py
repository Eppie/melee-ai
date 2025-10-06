from __future__ import annotations

import numpy as np

from libmelee.melee import enums
from libmelee.melee.enums import Action

MAX_FRAMES: int = (60 * 60 * 8) + 123  # Full 8 minute replay


def _preprocess_frame(frame: int) -> np.int32:
    processed_frame = frame + 123
    assert 0 <= processed_frame <= MAX_FRAMES, f"Processed frame {processed_frame} is out of range (0, {MAX_FRAMES})"
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


"""
Note that all stick values here are in the domain [-1,1], while the game/libmelee uses [0,1],
so conversions will be needed.
"""
FOX_STICK_64: list[tuple[float, float]] = [
    # --- Essentials ---
    (0.0, -1.0),  # 01 Hard down – ASDI-down, CC, fast-fall
    (-1.0, 0.0),  # 02 Hard left – max drift, horiz DI
    (0.0, 1.0),  # 03 Hard up – survival vs horizontals, Firefox charge positioning
    (1.0, 0.0),  # 04 Hard right – max drift, horiz DI
    (0.0, 0.0),  # 05 Neutral – buffering, re-center

    # --- Shield-drop (engine accepts -0.6875..-0.6625; one precise value suffices) ---
    (0.0, -0.6750),  # 06 Shield-drop Y (canonical pick in the allowed band)

    # --- Wavedash / waveland / ledgedash (down-toward). Keep both shallow (distance) and ~45° (Fox ledgedash) ---
    (0.9500, -0.2875),  # 07 16.84° down-right – shallowest legal; max slide if grounded
    (0.9300, -0.3500),  # 08 ~21° down-right – shallow WD
    (0.8625, -0.5000),  # 09 30° down-right – standard long WD
    (0.8125, -0.5750),  # 10 ~35° down-right – distance/consistency trade
    (0.7625, -0.6375),  # 11 ~40° down-right – reliable ground contact
    (0.7000, -0.7000),  # 12 45° down-right – Fox ledgedash (~45° best GALINT)
    (0.6375, -0.7625),  # 13 50° down-right – safer vs slants
    (-0.9500, -0.2875),  # 14 16.84° down-left – mirror
    (-0.9300, -0.3500),  # 15 ~21° down-left – mirror
    (-0.8625, -0.5000),  # 16 30° down-left – mirror
    (-0.8125, -0.5750),  # 17 ~35° down-left – mirror
    (-0.7625, -0.6375),  # 18 ~40° down-left – mirror
    (-0.7000, -0.7000),  # 19 45° down-left – Fox ledgedash mirror
    (-0.6375, -0.7625),  # 20 50° down-left – mirror

    # --- Firefox (Up+B) right/up – dense angles for recoveries ---
    (0.9000, 0.4125),  # 21 ~24.5° – shallow Firefox
    (0.8625, 0.5000),  # 22 30°
    (0.8125, 0.5750),  # 23 35°
    (0.7625, 0.6375),  # 24 40°
    (0.7000, 0.7000),  # 25 45° – baseline diagonal
    (0.6375, 0.7625),  # 26 50°
    (0.5750, 0.8125),  # 27 55°
    (0.5000, 0.8625),  # 28 60°
    (0.4125, 0.9000),  # 29 65°
    (0.3500, 0.9300),  # 30 70°
    (0.2875, 0.9500),  # 31 75° – steep threader

    # --- Firefox (Up+B) left/up – mirrors ---
    (-0.9000, 0.4125),  # 32 ~24.5° left
    (-0.8625, 0.5000),  # 33 30° left
    (-0.8125, 0.5750),  # 34 35° left
    (-0.7625, 0.6375),  # 35 40° left
    (-0.7000, 0.7000),  # 36 45° left
    (-0.6375, 0.7625),  # 37 50° left
    (-0.5750, 0.8125),  # 38 55° left
    (-0.5000, 0.8625),  # 39 60° left
    (-0.4125, 0.9000),  # 40 65° left
    (-0.3500, 0.9300),  # 41 70° left
    (-0.2875, 0.9500),  # 42 75° left

    # --- Ultra-steep Firefox angles ---
    (0.1750, 0.9750),  # 43 ~80° right/up – high thread under/over guards
    (0.0875, 0.9875),  # 44 ~85° right/up – near-vertical sweetspot
    (-0.1750, 0.9750),  # 45 ~80° left/up – mirror
    (-0.0875, 0.9875),  # 46 ~85° left/up – mirror

    # --- Universal 45° diagonals (DI/ASDI, drift, recoveries) ---
    (0.7071, 0.7071),  # 47 45° up-right – survival DI corner (approx; <=1 magnitude)
    (-0.7071, 0.7071),  # 48 135° up-left – mirror
    (-0.7071, -0.7071),  # 49 225° down-left – combo DI down-away
    (0.7071, -0.7071),  # 50 315° down-right – combo DI down-away

    # --- DI ring (8-point, ~22.5° steps; grid-aligned to 0.925/0.375 for unit radius) ---
    (0.9250, 0.3750),  # 51 ~22.5° – shallow-KB survival DI
    (0.3750, 0.9250),  # 52 ~67.5°
    (-0.3750, 0.9250),  # 53 ~112.5°
    (-0.9250, 0.3750),  # 54 ~157.5°
    (-0.9250, -0.3750),  # 55 ~202.5°
    (-0.3750, -0.9250),  # 56 ~247.5°
    (0.3750, -0.9250),  # 57 ~292.5°
    (0.9250, -0.3750),  # 58 ~337.5°

    # --- Steep ASDI/slide-off helpers (mirroring rectangle C-stick diagonals) ---
    (0.5250, 0.8500),  # 59 Up-right steep ASDI / platform slide-off
    (-0.5250, 0.8500),  # 60 Up-left steep ASDI / platform slide-off

    # --- Exact thresholds of axis activation (tilts, walk, ambiguous DI, pivot control) ---
    (0.6625, 0.0),  # 61 X=+0.6625 – tilt/walk cutoff; avoids X-smash
    (-0.6625, 0.0),  # 62 X=-0.6625 – mirror
    (0.2875, 0.0),  # 63 X=+0.2875 – minimal axis activation; dash/turn micro-timing
    (-0.2875, 0.0),  # 64 X=-0.2875 – mirror
]

# (Optional) quick sanity checks for your pipeline:
assert len(FOX_STICK_64) == 64
assert (
        max(x * x + y * y for x, y in FOX_STICK_64) <= 1.0 + 1e-9
)
# all inside unit circle

C_STICK_XY_CLUSTER_CENTERS_V0_1: np.ndarray = np.array(
    np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [0.0, 1.0],
        [-0.7, -0.7],
        [0.7, -0.7],
        [0.7, 0.7],
        [-0.7, 0.7],
    ])
)

SHOULDER_VALUES: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
