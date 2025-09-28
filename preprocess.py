from __future__ import annotations

import csv
from enum import IntEnum, auto
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from libmelee.melee import enums
from libmelee.melee.enums import Action

MAX_FRAMES: int = (60 * 60 * 8) + 123  # Full 8 minute replay


class ActionCategory(IntEnum):
    DEAD = 0
    ENTERING_STAGE = auto()
    GROUNDED_MOVEMENT = auto()
    AERIAL_MOVEMENT = auto()
    GROUNDED_LIGHT = auto()
    GROUNDED_STRONG = auto()
    AERIAL_NORMALS = auto()
    HITSTUN = auto()
    ITEMS = auto()
    SHIELDING = auto()
    OTHER_GROUNDED = auto()
    TECH = auto()
    SHIELD_BREAK = auto()
    GRAB = auto()
    THROW = auto()
    BEING_GRABBED = auto()
    DODGE_REBOUND = auto()
    BEING_THROWN = auto()
    MISSED_TECH = auto()
    EDGE = auto()
    TAUNT = auto()
    DK_SPECIFIC = auto()
    JIGGLY_SPECIFIC = auto()
    MEWTWO_SPECIFIC = auto()
    FOX_SPECIFIC = auto()
    CAPTURE = auto()
    SPECIALS = auto()
    KIRBY_SPECIFIC = auto()
    OTHER = auto()


USED_BY_CATEGORY: Mapping[ActionCategory, tuple[Action, ...]] = {
    ActionCategory.DEAD: (
        Action.DEAD_DOWN,
        Action.DEAD_LEFT,
        Action.DEAD_RIGHT,
        Action.DEAD_FLY_STAR,
        Action.DEAD_FLY,
        Action.DEAD_FLY_SPLATTER,
        Action.DEAD_FLY_SPLATTER_FLAT,
    ),
    ActionCategory.ENTERING_STAGE: (
        Action.ON_HALO_DESCENT,
        Action.ON_HALO_WAIT,
        Action.ENTRY,
        Action.ENTRY_START,
        Action.ENTRY_END,
    ),
    ActionCategory.GROUNDED_MOVEMENT: (
        Action.STANDING,
        Action.WALK_SLOW,
        Action.WALK_MIDDLE,
        Action.WALK_FAST,
        Action.TURNING,
        Action.TURNING_RUN,
        Action.DASHING,
        Action.RUNNING,
        Action.RUN_BRAKE,
        Action.KNEE_BEND,
        Action.CROUCH_START,
        Action.CROUCHING,
        Action.CROUCH_END,
        Action.ROLL_FORWARD,
        Action.ROLL_BACKWARD,
        Action.SPOTDODGE,
        Action.PLATFORM_DROP,
        Action.EDGE_TEETERING_START,
        Action.EDGE_TEETERING,
        Action.SLIDING_OFF_EDGE,
    ),
    ActionCategory.AERIAL_MOVEMENT: (
        Action.JUMPING_FORWARD,
        Action.JUMPING_BACKWARD,
        Action.JUMPING_ARIAL_FORWARD,
        Action.JUMPING_ARIAL_BACKWARD,
        Action.FALLING,
        Action.FALLING_AERIAL,
        Action.DEAD_FALL,
        Action.TUMBLING,
        Action.LANDING,
        Action.LANDING_SPECIAL,
    ),
    ActionCategory.GROUNDED_LIGHT: (
        Action.NEUTRAL_ATTACK_1,
        Action.NEUTRAL_ATTACK_2,
        Action.NEUTRAL_ATTACK_3,
        Action.LOOPING_ATTACK_START,
        Action.LOOPING_ATTACK_MIDDLE,
        Action.LOOPING_ATTACK_END,
        Action.DASH_ATTACK,
        Action.FTILT_HIGH,
        Action.FTILT_HIGH_MID,
        Action.FTILT_MID,
        Action.FTILT_LOW_MID,
        Action.FTILT_LOW,
        Action.UPTILT,
        Action.DOWNTILT,
    ),
    ActionCategory.GROUNDED_STRONG: (
        Action.FSMASH_HIGH,
        Action.FSMASH_MID_HIGH,
        Action.FSMASH_MID,
        Action.FSMASH_MID_LOW,
        Action.FSMASH_LOW,
        Action.UPSMASH,
        Action.DOWNSMASH,
    ),
    ActionCategory.AERIAL_NORMALS: (
        Action.NAIR,
        Action.FAIR,
        Action.BAIR,
        Action.UAIR,
        Action.DAIR,
        Action.NAIR_LANDING,
        Action.FAIR_LANDING,
        Action.BAIR_LANDING,
        Action.UAIR_LANDING,
        Action.DAIR_LANDING,
    ),
    ActionCategory.HITSTUN: (
        Action.DAMAGE_HIGH_1,
        Action.DAMAGE_HIGH_2,
        Action.DAMAGE_HIGH_3,
        Action.DAMAGE_NEUTRAL_1,
        Action.DAMAGE_NEUTRAL_2,
        Action.DAMAGE_NEUTRAL_3,
        Action.DAMAGE_LOW_1,
        Action.DAMAGE_LOW_2,
        Action.DAMAGE_LOW_3,
        Action.DAMAGE_AIR_1,
        Action.DAMAGE_AIR_2,
        Action.DAMAGE_AIR_3,
        Action.DAMAGE_FLY_HIGH,
        Action.DAMAGE_FLY_NEUTRAL,
        Action.DAMAGE_FLY_LOW,
        Action.DAMAGE_FLY_TOP,
        Action.DAMAGE_FLY_ROLL,
    ),
    ActionCategory.ITEMS: (
        Action.ITEM_PICKUP_LIGHT,
        Action.ITEM_THROW_LIGHT_FORWARD,
        Action.ITEM_THROW_LIGHT_BACK,
        Action.ITEM_THROW_LIGHT_HIGH,
        Action.ITEM_THROW_LIGHT_LOW,
        Action.ITEM_THROW_LIGHT_DASH,
        Action.ITEM_THROW_LIGHT_AIR_FORWARD,
        Action.ITEM_THROW_LIGHT_AIR_BACK,
        Action.ITEM_THROW_LIGHT_AIR_HIGH,
        Action.ITEM_THROW_LIGHT_AIR_LOW,
        Action.ITEM_THROW_LIGHT_SMASH_FORWARD,
        Action.ITEM_THROW_LIGHT_SMASH_BACK,
        Action.ITEM_THROW_LIGHT_SMASH_UP,
        Action.ITEM_THROW_LIGHT_SMASH_DOWN,
        Action.ITEM_THROW_LIGHT_AIR_SMASH_FORWARD,
        Action.ITEM_THROW_LIGHT_AIR_SMASH_BACK,
        Action.ITEM_THROW_LIGHT_AIR_SMASH_HIGH,
        Action.ITEM_THROW_LIGHT_AIR_SMASH_LOW,
        Action.BEAM_SWORD_SWING_3,
        Action.BEAM_SWORD_SWING_4,
        Action.ITEM_THROW_LIGHT_DROP,
    ),
    ActionCategory.SHIELDING: (
        Action.SHIELD_START,
        Action.SHIELD,
        Action.SHIELD_RELEASE,
        Action.SHIELD_STUN,
        Action.SHIELD_REFLECT,
    ),
    ActionCategory.OTHER_GROUNDED: (
        Action.TECH_MISS_UP,
        Action.LYING_GROUND_UP,
        Action.LYING_GROUND_UP_HIT,
        Action.GROUND_GETUP,
        Action.GROUND_ATTACK_UP,
        Action.GROUND_ROLL_FORWARD_UP,
        Action.GROUND_ROLL_BACKWARD_UP,
        Action.TECH_MISS_DOWN,
        Action.LYING_GROUND_DOWN,
        Action.DAMAGE_GROUND,
        Action.NEUTRAL_GETUP,
        Action.GETUP_ATTACK,
        Action.GROUND_ROLL_FORWARD_DOWN,
        Action.GROUND_ROLL_BACKWARD_DOWN,
    ),
    ActionCategory.TECH: (
        Action.NEUTRAL_TECH,
        Action.FORWARD_TECH,
        Action.BACKWARD_TECH,
        Action.WALL_TECH,
        Action.WALL_TECH_JUMP,
        Action.CEILING_TECH,
    ),
    ActionCategory.SHIELD_BREAK: (
        Action.SHIELD_BREAK_FLY,
        Action.SHIELD_BREAK_FALL,
        Action.SHIELD_BREAK_DOWN_U,
        Action.SHIELD_BREAK_DOWN_D,
        Action.SHIELD_BREAK_STAND_U,
        Action.SHIELD_BREAK_STAND_D,
        Action.SHIELD_BREAK_TEETER,
    ),
    ActionCategory.GRAB: (
        Action.GRAB,
        Action.GRAB_PULLING,
        Action.GRAB_RUNNING,
        Action.GRAB_RUNNING_PULLING,
        Action.GRAB_WAIT,
        Action.GRAB_PUMMEL,
        Action.GRAB_BREAK,
    ),
    ActionCategory.THROW: (
        Action.THROW_FORWARD,
        Action.THROW_BACK,
        Action.THROW_UP,
        Action.THROW_DOWN,
        Action.GRAB_PULLING_HIGH,
        Action.GRABBED_WAIT_HIGH,
        Action.PUMMELED_HIGH,
    ),
    ActionCategory.BEING_GRABBED: (
        Action.GRAB_PULL,
        Action.GRABBED,
        Action.GRAB_PUMMELED,
        Action.GRAB_ESCAPE,
        Action.GRAB_JUMP,
    ),
    ActionCategory.DODGE_REBOUND: (
        Action.AIRDODGE,
        Action.REBOUND_STOP,
        Action.REBOUND,
    ),
    ActionCategory.BEING_THROWN: (
        Action.THROWN_FORWARD,
        Action.THROWN_BACK,
        Action.THROWN_UP,
        Action.THROWN_DOWN,
    ),
    ActionCategory.MISSED_TECH: (
        Action.BOUNCE_WALL,
        Action.BOUNCE_CEILING,
        Action.BUMP_WALL,
        Action.BUMP_CIELING,
    ),
    ActionCategory.EDGE: (
        Action.EDGE_CATCHING,
        Action.EDGE_HANGING,
        Action.EDGE_GETUP_SLOW,
        Action.EDGE_GETUP_QUICK,
        Action.EDGE_ATTACK_SLOW,
        Action.EDGE_ATTACK_QUICK,
        Action.EDGE_ROLL_SLOW,
        Action.EDGE_ROLL_QUICK,
        Action.EDGE_JUMP_1_SLOW,
        Action.EDGE_JUMP_2_SLOW,
        Action.EDGE_JUMP_1_QUICK,
        Action.EDGE_JUMP_2_QUICK,
    ),
    ActionCategory.TAUNT: (
        Action.TAUNT_RIGHT,
        Action.TAUNT_LEFT,
    ),
    ActionCategory.DK_SPECIFIC: (
        Action.SHOULDERED_WAIT,
        Action.SHOULDERED_WALK_SLOW,
        Action.SHOULDERED_WALK_MIDDLE,
        Action.SHOULDERED_TURN,
        Action.THROWN_FF,
        Action.THROWN_FB,
        Action.THROWN_F_HIGH,
        Action.THROWN_F_LOW,
        Action.BURY,
        Action.BURY_WAIT,
        Action.BURY_JUMP,
        Action.DK_GROUND_POUND_START,
        Action.DK_GROUND_POUND,
        Action.DK_GROUND_POUND_END,
    ),
    ActionCategory.JIGGLY_SPECIFIC: (
        Action.DAMAGE_SONG,
        Action.DAMAGE_SONG_WAIT,
        Action.DAMAGE_SONG_RV,
    ),
    ActionCategory.MEWTWO_SPECIFIC: (
        Action.DAMAGE_BIND,
        Action.THROWN_MEWTWO,
        Action.THROWN_MEWTWO_AIR,
    ),
    ActionCategory.FOX_SPECIFIC: (
        Action.SHINE_TURN,
        Action.DOWN_B_STUN,
        Action.DOWN_B_AIR,
        Action.UP_B_GROUND,
    ),
    ActionCategory.CAPTURE: (
        Action.CAPTURE_CAPTAIN,
        Action.CAPTURE_YOSHI,
        Action.CAPTURE_DAMAGE_KOOPA,
        Action.CAPTURE_WAIT_KOOPA,
        Action.CAPTURE_DAMAGE_KOOPA_AIR,
        Action.CAPTURE_WAIT_KOOPA_AIR,
        Action.CAPTURE_KIRBY,
        Action.CAPTURE_WAIT_KIRBY,
        Action.CAPTURE_KOOPA_AIR_HIT,
    ),
    ActionCategory.SPECIALS: (
        Action.DAMAGE_ICE,
        Action.DAMAGE_ICE_JUMP,
        Action.DOWN_REFLECT,
        Action.LASER_GUN_PULL,
        Action.NEUTRAL_B_CHARGING,
        Action.NEUTRAL_B_ATTACKING,
        Action.NEUTRAL_B_FULL_CHARGE,
        Action.WAIT_ITEM,
        Action.NEUTRAL_B_CHARGING_AIR,
        Action.NEUTRAL_B_ATTACKING_AIR,
        Action.NEUTRAL_B_FULL_CHARGE_AIR,
        Action.SWORD_DANCE_1,
        Action.SWORD_DANCE_2_HIGH,
        Action.SWORD_DANCE_2_MID,
        Action.SWORD_DANCE_3_HIGH,
        Action.SWORD_DANCE_3_MID,
        Action.SWORD_DANCE_3_LOW,
        Action.SWORD_DANCE_4_HIGH,
        Action.SWORD_DANCE_4_MID,
        Action.SWORD_DANCE_4_LOW,
        Action.SWORD_DANCE_1_AIR,
        Action.SWORD_DANCE_2_HIGH_AIR,
        Action.DOWN_B_GROUND_START,
        Action.DOWN_B_GROUND,
        Action.SWORD_DANCE_3_MID_AIR,
        Action.SWORD_DANCE_3_LOW_AIR,
        Action.SHINE_RELEASE_AIR,
        Action.MARTH_COUNTER,
        Action.PARASOL_FALLING,
        Action.MARTH_COUNTER_FALLING,
        Action.NESS_SHEILD_START,
        Action.NESS_SHEILD_AIR,
        Action.ZITABATA,
        Action.NESS_SHEILD_AIR_END,
        Action.THROWN_KOOPA_END_F,
        Action.THROWN_KOOPA_END_B,
        Action.THROWN_KOOPA_AIR_END_F,
        Action.THROWN_KOOPA_AIR_END_B,
    ),
    ActionCategory.KIRBY_SPECIFIC: (
        Action.THROWN_KIRBY_DRINK_S_SHOT,
        Action.THROWN_KIRBY_SPIT_S_SHOT,
        Action.KIRBY_BLADE_GROUND,
        Action.KIRBY_BLADE_UP,
        Action.KIRBY_BLADE_APEX,
        Action.KIRBY_BLADE_DOWN,
        Action.KIRBY_STONE_FORMING_GROUND,
        Action.KIRBY_STONE_RESTING,
        Action.KIRBY_STONE_RELEASE,
        Action.KIRBY_STONE_FORMING_AIR,
        Action.KIRBY_STONE_FALLING,
    ),
    ActionCategory.OTHER: (
        Action.YOSHI_EGG,
        Action.THROWN_KOOPA_F,
        Action.THROWN_KOOPA_B,
        Action.THROWN_KOOPA_AIR_F,
        Action.THROWN_KOOPA_AIR_B,
        Action.THROWN_KIRBY_STAR,
        Action.THROWN_COPY_STAR,
        Action.THROWN_KIRBY,
        Action.UNKNOWN_ANIMATION,
        Action.THROWN_DOWN_2,
    ),
}

# === 2) Build dense indices and lookup maps ===
_ACTION_TO_DENSE: Dict[Action, int] = {}
_ACTION_TO_CATEGORY: Dict[Action, ActionCategory] = {}

_dense_counter = 0
for cat in ActionCategory:
    actions = USED_BY_CATEGORY.get(cat, ())
    for a in actions:
        if a in _ACTION_TO_DENSE:
            raise RuntimeError(f"Duplicate action in USED_BY_CATEGORY: {a}")
        _ACTION_TO_DENSE[a] = _dense_counter
        _ACTION_TO_CATEGORY[a] = cat
        _dense_counter += 1

NUM_USED_ACTIONS: int = _dense_counter

# Build inverse map: dense id -> original Action enum
_DENSE_TO_ACTION: Dict[int, Action] = {v: k for k, v in _ACTION_TO_DENSE.items()}


def dense_to_action(dense_id: int) -> Action:
    """
    Convert a dense index [0, NUM_USED_ACTIONS) back to the original Action enum.

    Accepts Python ints and NumPy integer types.
    Raises:
        ValueError: if the id is out of range or not in the inverse map.
    """
    try:
        return _DENSE_TO_ACTION[int(dense_id)]
    except (KeyError, ValueError):
        raise ValueError(f"dense_id {int(dense_id)} is not a valid dense Action id (0..{NUM_USED_ACTIONS - 1})")


def dense_to_action_name(dense_id: int) -> str:
    """Return the original Action enum name for a dense index (e.g., 'JUMPING_FORWARD')."""
    return dense_to_action(dense_id).name


def _preprocess_frame(frame: int) -> np.int32:
    processed_frame = frame + 123
    assert 0 <= processed_frame <= MAX_FRAMES, f"Processed frame {processed_frame} is out of range (0, {MAX_FRAMES})"
    return np.int32(processed_frame)


def _preprocess_stage(stage: enums.Stage) -> np.int32:
    return np.int32(
        {
            enums.Stage.FINAL_DESTINATION: 1,
            enums.Stage.BATTLEFIELD: 2,
            enums.Stage.POKEMON_STADIUM: 3,
            enums.Stage.DREAMLAND: 4,
            enums.Stage.FOUNTAIN_OF_DREAMS: 5,
            enums.Stage.YOSHIS_STORY: 6,
        }[stage]
    )


def _preprocess_character(character: enums.Character) -> np.int32:
    assert 0 <= character.value <= 26
    return np.int32(character.value)


# def _preprocess_action(action: Action) -> tuple[np.int32, np.int32]:
#     """
#     Map a (used) Action to:
#       1) a dense np.int32 id in [0, NUM_USED_ACTIONS)
#       2) a category id as np.int32 (ActionCategory value)
#
#     Raises:
#         ValueError: if the action is not in the 'used' set above.
#     """
#     try:
#         dense = _ACTION_TO_DENSE[action]
#         cat = _ACTION_TO_CATEGORY[action]
#     except KeyError:
#         raise ValueError(f"Action {action.name} (0x{action.value:02x}) is not in the used-action set")
#     return np.int32(dense), np.int32(int(cat))

def _preprocess_action(action: Action) -> np.int32:
    return np.int32(action.value)


def _preprocess_x_y_buttons(button_x: bool, button_y: bool) -> np.float32:
    return np.float32(np.logical_or(button_x, button_y))


def _preprocess_l_r_buttons(button_l: bool, button_r: bool) -> np.float32:
    return np.float32(np.logical_or(button_l, button_r))


def winsor_signed_sqrt(x: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    xw = np.clip(x, lo, hi)
    return np.sign(xw) * np.sqrt(np.abs(xw).astype(np.float32) + 1e-8)


def write_action_map_csv(path: Path | str = "action_map.csv") -> None:
    mapping: Dict[int, Action] = _DENSE_TO_ACTION
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dense_id", "action", "action_value"])  # headers
        for dense_id in sorted(mapping.keys()):
            act = mapping[dense_id]
            w.writerow([dense_id, act.name, int(act.value)])


if __name__ == "__main__":
    write_action_map_csv("action_map.csv")

# def category_of(action: Action) -> ActionCategory:
#     if action not in _ACTION_TO_CATEGORY:
#         raise ValueError(f"Action {action.name} (0x{action.value:02x}) is not in the used-action set")
#     return _ACTION_TO_CATEGORY[action]
#
#
# def category_name(cat_id: int) -> str:
#     return ActionCategory(cat_id).name
#
#
# def dense_size() -> int:
#     return NUM_USED_ACTIONS


from typing import Tuple
import numpy as np


def snap_replay_to_palette(
        raw_xy01: np.ndarray,
        palette11: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(raw_xy01, dtype=np.float32)
    if xy.shape[-1] != 2:
        raise ValueError("raw_xy01 must have last dimension size 2")
    P = np.asarray(palette11, dtype=np.float32)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError("palette11 must have shape (K, 2)")

    orig_shape = xy.shape
    V01 = xy.reshape(-1, 2)

    # Map [0,1] -> [-1,1], clip to unit circle (rarely necessary but safe).
    V11 = np.clip(V01 * 2.0 - 1.0, -1.0, 1.0)
    r2 = np.einsum("ij,ij->i", V11, V11)
    over = r2 > 1.0
    if np.any(over):
        V11[over] /= np.sqrt(r2[over])[..., None]

    # Precompute palette norms; vectorized nearest neighbor search.
    P_norm2 = np.einsum("ij,ij->i", P, P)  # (K,)
    V_norm2 = np.einsum("ij,ij->i", V11, V11)  # (N,)
    V_dot_PT = V11 @ P.T  # (N, K)
    d2 = V_norm2[:, None] + P_norm2[None, :] - 2.0 * V_dot_PT

    idx = np.argmin(d2, axis=1).astype(np.int32)
    snapped = P[idx]

    return snapped.reshape(orig_shape), idx.reshape(orig_shape[:-1])


def model_to_dolphin01(
        model_out: np.ndarray,
        palette11: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(model_out)

    if np.issubdtype(arr.dtype, np.integer):
        if palette11 is None:
            raise ValueError("palette11 must be provided when converting indices")
        P = np.asarray(palette11, dtype=np.float32)
        coords11 = P[arr]
    else:
        coords11 = np.asarray(model_out, dtype=np.float32)
        if coords11.shape[-1] != 2:
            raise ValueError("model_out must have last dimension size 2")

    # Clamp to unit circle to be safe.
    r2 = np.einsum("...i,...i->...", coords11, coords11)
    over = r2 > 1.0
    if np.any(over):
        coords11 = coords11.copy()
        coords11[over] /= np.sqrt(r2[over])[..., None]

    # Map [-1,1] -> [0,1]
    xy01 = np.clip(coords11 * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
    return xy01


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

    # --- Firefox (Up+B) right/up – dense angles for unfair recoveries ---
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

    # --- Ultra-steep Firefox angles (beyond rectangle-legal; superhuman recoveries) ---
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
