"""Melee game constants for statistics computation.

Contains action state names, categories, stage mappings, character names,
and known game limits for validation.
"""

from typing import Dict, FrozenSet, List

# Stage ID to name mapping (using preprocessed IDs from preprocess.py)
STAGE_NAMES: Dict[int, str] = {
    0: "Final Destination",
    1: "Battlefield",
    2: "Pokemon Stadium",
    3: "Dreamland",
    4: "Fountain of Dreams",
    5: "Yoshi's Story",
}

# Character internal ID to name mapping
CHARACTER_NAMES: Dict[int, str] = {
    0x00: "Mario",
    0x01: "Fox",
    0x02: "Captain Falcon",
    0x03: "Donkey Kong",
    0x04: "Kirby",
    0x05: "Bowser",
    0x06: "Link",
    0x07: "Sheik",
    0x08: "Ness",
    0x09: "Peach",
    0x0A: "Popo",
    0x0B: "Nana",
    0x0C: "Pikachu",
    0x0D: "Samus",
    0x0E: "Yoshi",
    0x0F: "Jigglypuff",
    0x10: "Mewtwo",
    0x11: "Luigi",
    0x12: "Marth",
    0x13: "Zelda",
    0x14: "Young Link",
    0x15: "Dr. Mario",
    0x16: "Falco",
    0x17: "Pichu",
    0x18: "Game & Watch",
    0x19: "Ganondorf",
    0x1A: "Roy",
}

# Action state ID to name mapping
# Based on libmelee enums.py Action enum
ACTION_STATE_NAMES: Dict[int, str] = {
    0x00: "DeadDown",
    0x01: "DeadLeft",
    0x02: "DeadRight",
    0x03: "DeadUp",
    0x04: "DeadFlyingStar",
    0x05: "DeadFlyingStarIce",
    0x06: "DeadFly",
    0x07: "DeadFlySplatter",
    0x08: "DeadFlySplatterFlat",
    0x09: "DeadFlySplatterIce",
    0x0A: "DeadFlySplatterFlatIce",
    0x0B: "Sleep",  # Nothing state for Sheik/Zelda
    0x0C: "Rebirth",
    0x0D: "RebirthWait",
    0x0E: "Wait",
    0x0F: "WalkSlow",
    0x10: "WalkMiddle",
    0x11: "WalkFast",
    0x12: "Turn",
    0x13: "TurnRun",
    0x14: "Dash",
    0x15: "Run",
    0x16: "RunDirect",
    0x17: "RunBrake",
    0x18: "KneeBend",
    0x19: "JumpF",
    0x1A: "JumpB",
    0x1B: "JumpAerialF",
    0x1C: "JumpAerialB",
    0x1D: "Fall",
    0x1E: "FallF",
    0x1F: "FallB",
    0x20: "FallAerial",
    0x21: "FallAerialF",
    0x22: "FallAerialB",
    0x23: "FallSpecial",
    0x24: "FallSpecialF",
    0x25: "FallSpecialB",
    0x26: "DamageFall",
    0x27: "Squat",
    0x28: "SquatWait",
    0x29: "SquatRv",
    0x2A: "Landing",
    0x2B: "LandingFallSpecial",
    0x2C: "Attack11",
    0x2D: "Attack12",
    0x2E: "Attack13",
    0x2F: "Attack100Start",
    0x30: "Attack100Loop",
    0x31: "Attack100End",
    0x32: "AttackDash",
    0x33: "AttackS3Hi",
    0x34: "AttackS3HiS",
    0x35: "AttackS3S",
    0x36: "AttackS3LwS",
    0x37: "AttackS3Lw",
    0x38: "AttackHi3",
    0x39: "AttackLw3",
    0x3A: "AttackS4Hi",
    0x3B: "AttackS4HiS",
    0x3C: "AttackS4S",
    0x3D: "AttackS4LwS",
    0x3E: "AttackS4Lw",
    0x3F: "AttackHi4",
    0x40: "AttackLw4",
    0x41: "AttackAirN",
    0x42: "AttackAirF",
    0x43: "AttackAirB",
    0x44: "AttackAirHi",
    0x45: "AttackAirLw",
    0x46: "LandingAirN",
    0x47: "LandingAirF",
    0x48: "LandingAirB",
    0x49: "LandingAirHi",
    0x4A: "LandingAirLw",
    0x4B: "DamageHi1",
    0x4C: "DamageHi2",
    0x4D: "DamageHi3",
    0x4E: "DamageN1",
    0x4F: "DamageN2",
    0x50: "DamageN3",
    0x51: "DamageLw1",
    0x52: "DamageLw2",
    0x53: "DamageLw3",
    0x54: "DamageAir1",
    0x55: "DamageAir2",
    0x56: "DamageAir3",
    0x57: "DamageFlyHi",
    0x58: "DamageFlyN",
    0x59: "DamageFlyLw",
    0x5A: "DamageFlyTop",
    0x5B: "DamageFlyRoll",
    # Items (0x5C - 0xB1)
    0x5C: "LightGet",
    0x5D: "HeavyGet",
    0x5E: "LightThrowF",
    0x5F: "LightThrowB",
    0x60: "LightThrowHi",
    0x61: "LightThrowLw",
    0x62: "LightThrowDash",
    0x63: "LightThrowDrop",
    0x64: "LightThrowAirF",
    0x65: "LightThrowAirB",
    0x66: "LightThrowAirHi",
    0x67: "LightThrowAirLw",
    0x68: "HeavyThrowF",
    0x69: "HeavyThrowB",
    0x6A: "HeavyThrowHi",
    0x6B: "HeavyThrowLw",
    # Shield states
    0xB2: "GuardOn",
    0xB3: "Guard",
    0xB4: "GuardOff",
    0xB5: "GuardSetOff",
    0xB6: "GuardReflect",
    # Tech states
    0xB7: "DownBoundU",
    0xB8: "DownWaitU",
    0xB9: "DownDamageU",
    0xBA: "DownStandU",
    0xBB: "DownAttackU",
    0xBC: "DownFowardU",
    0xBD: "DownBackU",
    0xBE: "DownSpotU",
    0xBF: "DownBoundD",
    0xC0: "DownWaitD",
    0xC1: "DownDamageD",
    0xC2: "DownStandD",
    0xC3: "DownAttackD",
    0xC4: "DownFowardD",
    0xC5: "DownBackD",
    0xC6: "DownSpotD",
    0xC7: "Passive",
    0xC8: "PassiveStandF",
    0xC9: "PassiveStandB",
    0xCA: "PassiveWall",
    0xCB: "PassiveWallJump",
    0xCC: "PassiveCeil",
    # Shield break
    0xCD: "ShieldBreakFly",
    0xCE: "ShieldBreakFall",
    0xCF: "ShieldBreakDownU",
    0xD0: "ShieldBreakDownD",
    0xD1: "ShieldBreakStandU",
    0xD2: "ShieldBreakStandD",
    0xD3: "FuraFura",
    # Grabs
    0xD4: "Catch",
    0xD5: "CatchPull",
    0xD6: "CatchDash",
    0xD7: "CatchDashPull",
    0xD8: "CatchWait",
    0xD9: "CatchAttack",
    0xDA: "CatchCut",
    0xDB: "ThrowF",
    0xDC: "ThrowB",
    0xDD: "ThrowHi",
    0xDE: "ThrowLw",
    # Being grabbed
    0xE2: "CapturePulled",
    0xE3: "CaptureWait",
    0xE4: "CaptureDamage",
    0xE5: "CaptureEscapeJump",
    0xE6: "CaptureEscape",
    # Dodges
    0xE9: "EscapeF",
    0xEA: "EscapeB",
    0xEB: "Escape",
    0xEC: "EscapeAir",
    # Thrown states
    0xEF: "ThrownF",
    0xF0: "ThrownB",
    0xF1: "ThrownHi",
    0xF2: "ThrownLw",
    # Platform/edge
    0xF4: "Pass",
    0xF5: "Ottotto",
    0xF6: "OttottoWait",
    0xF7: "StopWall",
    0xF8: "StopCeil",
    0xFB: "MissFoot",
    0xFC: "CliffCatch",
    0xFD: "CliffWait",
    0xFE: "CliffClimbSlow",
    0xFF: "CliffClimbQuick",
    0x100: "CliffAttackSlow",
    0x101: "CliffAttackQuick",
    0x102: "CliffEscapeSlow",
    0x103: "CliffEscapeQuick",
    0x104: "CliffJumpSlow1",
    0x105: "CliffJumpSlow2",
    0x106: "CliffJumpQuick1",
    0x107: "CliffJumpQuick2",
    # Taunts
    0x108: "AppealR",
    0x109: "AppealL",
    # Entry
    0x142: "Entry",
    0x143: "EntryStart",
    0x144: "EntryEnd",
    # Specials (character-specific, common IDs)
    0x156: "SpecialNStart",
    0x157: "SpecialN",
    0x158: "SpecialNEnd",
    0x15A: "SpecialAirNStart",
    0x15B: "SpecialAirN",
    0x15C: "SpecialAirNEnd",
    0x15D: "SpecialS1",
    0x15E: "SpecialS2",
    0x161: "SpecialHi",
    0x164: "SpecialAirHi",
    0x168: "SpecialLw",
    0x169: "SpecialLwLoop",
    0x16E: "SpecialAirLw",
    0x16F: "SpecialHiStart",
    0x170: "SpecialAirHiStart",
    0x171: "SpecialLwHit",  # Marth counter
}

# Action state categories for grouping
# Maps category name to frozenset of action state IDs
ACTION_STATE_CATEGORIES: Dict[str, FrozenSet[int]] = {
    "dead": frozenset(range(0x00, 0x0C)),
    "wait": frozenset([0x0E]),
    "walk": frozenset([0x0F, 0x10, 0x11]),
    "dash_run": frozenset([0x14, 0x15, 0x16, 0x17]),
    "turn": frozenset([0x12, 0x13]),
    "crouch": frozenset([0x27, 0x28, 0x29]),
    "jump_squat": frozenset([0x18]),
    "jump": frozenset([0x19, 0x1A, 0x1B, 0x1C]),
    "fall": frozenset([0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26]),
    "landing": frozenset([0x2A, 0x2B, 0x46, 0x47, 0x48, 0x49, 0x4A]),
    "jab": frozenset([0x2C, 0x2D, 0x2E, 0x2F, 0x30, 0x31]),
    "dash_attack": frozenset([0x32]),
    "tilt": frozenset([0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39]),
    "smash": frozenset([0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x40]),
    "aerial": frozenset([0x41, 0x42, 0x43, 0x44, 0x45]),
    "damage": frozenset(range(0x4B, 0x5C)),
    "shield": frozenset([0xB2, 0xB3, 0xB4, 0xB5, 0xB6]),
    "tech": frozenset(range(0xB7, 0xCD)),
    "shield_break": frozenset(range(0xCD, 0xD4)),
    "grab": frozenset([0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA]),
    "throw": frozenset([0xDB, 0xDC, 0xDD, 0xDE]),
    "grabbed": frozenset([0xE2, 0xE3, 0xE4, 0xE5, 0xE6]),
    "thrown": frozenset([0xEF, 0xF0, 0xF1, 0xF2]),
    "dodge": frozenset([0xE9, 0xEA, 0xEB, 0xEC]),
    "ledge": frozenset(range(0xFC, 0x108)),
    "taunt": frozenset([0x108, 0x109]),
    "entry": frozenset([0x0C, 0x0D, 0x142, 0x143, 0x144]),
    "special_neutral": frozenset([0x156, 0x157, 0x158, 0x15A, 0x15B, 0x15C]),
    "special_side": frozenset([0x15D, 0x15E]),
    "special_up": frozenset([0x161, 0x164, 0x16F, 0x170]),
    "special_down": frozenset([0x168, 0x169, 0x16E, 0x171]),
}

# Broader action groupings for high-level analysis
ACTION_STATE_GROUPS: Dict[str, List[str]] = {
    "neutral": ["wait", "walk", "dash_run", "turn", "crouch", "fall"],
    "offensive": [
        "jab",
        "dash_attack",
        "tilt",
        "smash",
        "aerial",
        "grab",
        "throw",
        "special_neutral",
        "special_side",
        "special_up",
        "special_down",
    ],
    "defensive": ["shield", "dodge", "tech", "ledge"],
    "disadvantage": ["damage", "grabbed", "thrown", "shield_break"],
    "movement": ["walk", "dash_run", "jump_squat", "jump", "fall", "landing"],
    "recovery": ["ledge", "special_up"],
}


def get_action_category(action_id: int) -> str:
    """Get the category name for an action state ID."""
    for category, action_ids in ACTION_STATE_CATEGORIES.items():
        if action_id in action_ids:
            return category
    return "other"


def get_action_group(action_id: int) -> str:
    """Get the high-level group for an action state ID."""
    category = get_action_category(action_id)
    for group, categories in ACTION_STATE_GROUPS.items():
        if category in categories:
            return group
    return "other"


# Known game limits for data validation
GAME_LIMITS: Dict[str, Dict[str, float]] = {
    "position_x": {"min": -300.0, "max": 300.0},
    "position_y": {"min": -200.0, "max": 350.0},
    "percent": {"min": 0.0, "max": 999.0},
    "stock": {"min": 0, "max": 99},
    "shield_strength": {"min": 0.0, "max": 60.0},
    "jumps_left": {"min": 0, "max": 6},  # Kirby/Jigglypuff have 6
    "facing": {"min": -1.0, "max": 1.0},
    "main_stick_x": {"min": 0.0, "max": 1.0},
    "main_stick_y": {"min": 0.0, "max": 1.0},
    "c_stick_x": {"min": 0.0, "max": 1.0},
    "c_stick_y": {"min": 0.0, "max": 1.0},
    "shoulder_analog": {"min": 0.0, "max": 1.0},
    "action": {"min": 0, "max": 400},
    "character": {"min": 0, "max": 26},
    "stage": {"min": 0, "max": 5},
}

# L-cancel status values
L_CANCEL_STATUS: Dict[int, str] = {
    0: "none",
    1: "successful",
    2: "unsuccessful",
}

# Button names for controller analysis
BUTTON_NAMES: List[str] = ["button_a", "button_b", "button_xy", "button_z", "button_lr"]

# Stick position regions for analysis
STICK_REGIONS: Dict[str, Dict[str, tuple]] = {
    "neutral": {"x": (0.35, 0.65), "y": (0.35, 0.65)},
    "up": {"x": (0.35, 0.65), "y": (0.65, 1.0)},
    "down": {"x": (0.35, 0.65), "y": (0.0, 0.35)},
    "left": {"x": (0.0, 0.35), "y": (0.35, 0.65)},
    "right": {"x": (0.65, 1.0), "y": (0.35, 0.65)},
    "up_left": {"x": (0.0, 0.35), "y": (0.65, 1.0)},
    "up_right": {"x": (0.65, 1.0), "y": (0.65, 1.0)},
    "down_left": {"x": (0.0, 0.35), "y": (0.0, 0.35)},
    "down_right": {"x": (0.65, 1.0), "y": (0.0, 0.35)},
}


def get_stick_region(x: float, y: float) -> str:
    """Determine which region a stick position falls into."""
    for region, bounds in STICK_REGIONS.items():
        x_min, x_max = bounds["x"]
        y_min, y_max = bounds["y"]
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return region
    return "edge"  # Outside standard regions


# Frame rate constant
FRAMES_PER_SECOND = 60
