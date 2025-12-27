"""Enum values for various Melee objects"""

from enum import Enum


class Stage(Enum):
    """A VS-mode stage"""

    NO_STAGE = 0
    FINAL_DESTINATION = 0x19
    BATTLEFIELD = 0x18
    POKEMON_STADIUM = 0x12
    DREAMLAND = 0x1A
    FOUNTAIN_OF_DREAMS = 0x8
    YOSHIS_STORY = 0x6
    RANDOM_STAGE = 0x1D  # not technically a stage, but it's useful to call it one


def to_internal_stage(stage_id):
    if stage_id == 0x03:
        return Stage.POKEMON_STADIUM
    if stage_id == 0x08:
        return Stage.YOSHIS_STORY
    if stage_id == 0x02:
        return Stage.FOUNTAIN_OF_DREAMS
    if stage_id == 0x1F:
        return Stage.BATTLEFIELD
    if stage_id == 0x20:
        return Stage.FINAL_DESTINATION
    if stage_id == 0x1C:
        return Stage.DREAMLAND
    return Stage.NO_STAGE


class Menu(Enum):
    """A primary menu scene the game can be in"""

    CHARACTER_SELECT = 0
    STAGE_SELECT = 1
    IN_GAME = 2
    SUDDEN_DEATH = 3
    POSTGAME_SCORES = 4
    MAIN_MENU = 5
    SLIPPI_ONLINE_CSS = 6
    PRESS_START = 7
    UNKNOWN_MENU = 0xFF


class SubMenu(Enum):
    """Sub-menu of a primary menu"""

    MAIN_MENU_SUBMENU = 0
    ONEP_MODE_SUBMENU = 1
    VS_MODE_SUBMENU = 2
    TROPHIES_SUBMENU = 3
    OPTION_SUBMENU = 4
    DATA_SUBMENU = 5
    REGULAR_MATCH_SUBMENU = 6
    EVENT_MATCH_SUBMENU = 7
    ONLINE_PLAY_SUBMENU = 8
    STADIUM_SUBMENU = 9
    SPECIAL_MELEE_SUBMENU = 12
    CUSTOM_RULES_SUBMENU = 13
    NAME_ENTRY_SUBMENU = 18
    RUMBLE_SUBMENU = 19
    SOUND_SUBMENU = 20
    SCREEN_DISPLAY_SUBMENU = 21
    LANGUAGE_SELECT_SUBMENU = 23
    ERASE_DATA_SUBMENU = 24
    MULTIMAN_MELEE_SUBMENU = 33
    ONLINE_CSS = 0xFE
    UNKNOWN_SUBMENU = 0xFF


class ControllerStatus(Enum):
    """One of three states a controller can be in during character select"""

    CONTROLLER_HUMAN = 0
    CONTROLLER_CPU = 1
    CONTROLLER_UNPLUGGED = 3


class ControllerType(Enum):
    """Types a controller can be in the Dolphin config

    Named pipe input is considered 'standard' input by Dolphin.
    """

    STANDARD = "6"
    GCN_ADAPTER = "12"
    UNPLUGGED = "0"


class AttackState(Enum):
    """The phases an attack can be in"""

    WINDUP = 0
    ATTACKING = 1
    COOLDOWN = 2
    NOT_ATTACKING = 3


class Character(Enum):
    """A Melee character ID.

    Note:
        Numeric values are 'internal' IDs."""

    MARIO = 0x00
    FOX = 0x01
    CPTFALCON = 0x02
    DK = 0x03
    KIRBY = 0x04
    BOWSER = 0x05
    LINK = 0x06
    SHEIK = 0x07
    NESS = 0x08
    PEACH = 0x09
    POPO = 0x0A
    NANA = 0x0B
    PIKACHU = 0x0C
    SAMUS = 0x0D
    YOSHI = 0x0E
    JIGGLYPUFF = 0x0F
    MEWTWO = 0x10
    LUIGI = 0x11
    MARTH = 0x12
    ZELDA = 0x13
    YLINK = 0x14
    DOC = 0x15
    FALCO = 0x16
    PICHU = 0x17
    GAMEANDWATCH = 0x18
    GANONDORF = 0x19
    ROY = 0x1A
    WIREFRAME_MALE = 0x1D
    WIREFRAME_FEMALE = 0x1E
    GIGA_BOWSER = 0x1F
    SANDBAG = 0x20
    UNKNOWN_CHARACTER = 0xFF


def to_internal(char_id):
    """Converts a character select-screen ID to an 'internal ID' enum

    Mostly used at the Character Select Screen
    """
    if char_id == 0x00:
        return Character.DOC
    if char_id == 0x01:
        return Character.MARIO
    if char_id == 0x02:
        return Character.LUIGI
    if char_id == 0x03:
        return Character.BOWSER
    if char_id == 0x04:
        return Character.PEACH
    if char_id == 0x05:
        return Character.YOSHI
    if char_id == 0x06:
        return Character.DK
    if char_id == 0x07:
        return Character.CPTFALCON
    if char_id == 0x08:
        return Character.GANONDORF
    if char_id == 0x09:
        return Character.FALCO
    if char_id == 0x0A:
        return Character.FOX
    if char_id == 0x0B:
        return Character.NESS
    if char_id == 0x0C:
        return Character.POPO
    if char_id == 0x0D:
        return Character.KIRBY
    if char_id == 0x0E:
        return Character.SAMUS
    if char_id == 0x0F:
        return Character.ZELDA
    if char_id == 0x10:
        return Character.LINK
    if char_id == 0x11:
        return Character.YLINK
    if char_id == 0x12:
        return Character.PICHU
    if char_id == 0x13:
        return Character.PIKACHU
    if char_id == 0x14:
        return Character.JIGGLYPUFF
    if char_id == 0x15:
        return Character.MEWTWO
    if char_id == 0x16:
        return Character.GAMEANDWATCH
    if char_id == 0x17:
        return Character.MARTH
    if char_id == 0x18:
        return Character.ROY
    return Character.UNKNOWN_CHARACTER


def from_internal(character):
    """Converts a character enum to an "external" ID.

    Mostly used at the Character Select Screen
    """
    if character == Character.DOC:
        return 0x00
    if character == Character.MARIO:
        return 0x01
    if character == Character.LUIGI:
        return 0x02
    if character == Character.BOWSER:
        return 0x03
    if character == Character.PEACH:
        return 0x04
    if character == Character.YOSHI:
        return 0x05
    if character == Character.DK:
        return 0x06
    if character == Character.CPTFALCON:
        return 0x07
    if character == Character.GANONDORF:
        return 0x08
    if character == Character.FALCO:
        return 0x09
    if character == Character.FOX:
        return 0x0A
    if character == Character.NESS:
        return 0x0B
    if character == Character.POPO:
        return 0x0C
    if character == Character.KIRBY:
        return 0x0D
    if character == Character.SAMUS:
        return 0x0E
    if character == Character.ZELDA:
        return 0x0F
    if character == Character.LINK:
        return 0x10
    if character == Character.YLINK:
        return 0x11
    if character == Character.PICHU:
        return 0x12
    if character == Character.PIKACHU:
        return 0x13
    if character == Character.JIGGLYPUFF:
        return 0x14
    if character == Character.MEWTWO:
        return 0x15
    if character == Character.GAMEANDWATCH:
        return 0x16
    if character == Character.MARTH:
        return 0x17
    if character == Character.ROY:
        return 0x18
    return 0xFF


class Button(Enum):
    """A single button on a GCN controller

    Note:
        String values represent the Dolphin input string for that button"""

    BUTTON_A = "A"
    BUTTON_B = "B"
    BUTTON_X = "X"
    BUTTON_Y = "Y"
    BUTTON_Z = "Z"
    BUTTON_L = "L"
    BUTTON_R = "R"
    BUTTON_START = "START"
    BUTTON_D_UP = "D_UP"
    BUTTON_D_DOWN = "D_DOWN"
    BUTTON_D_LEFT = "D_LEFT"
    BUTTON_D_RIGHT = "D_RIGHT"
    # Control sticks considered "buttons" here
    BUTTON_MAIN = "MAIN"
    BUTTON_C = "C"


class Action(Enum):
    """The in-game action (or animation) a character can be in

    Note:
        Numeric values (mostly) represent their in-game values"""

    DEAD_DOWN = 0x0  # Bottom blast zone KO, 100 frames. True: DEAD_DOWN
    DEAD_LEFT = 0x1  # Left blast zone KO, 100 frames. True: DEAD_LEFT
    DEAD_RIGHT = 0x2  # Right blast zone KO, 100 frames. True: DEAD_RIGHT
    DEAD_UP = 0x3  # 1P mode upward death. True: DEAD_UP
    DEAD_FLY_STAR = 0x4  # Star KO, 190 frames. True: DEAD_UP_STAR
    DEAD_FLY_STAR_ICE = 0x5  # Star KO while frozen. True: DEAD_UP_STAR_ICE
    DEAD_FLY = 0x6  # Top blast zone KO, 60 frames. True: DEAD_UP_FALL
    DEAD_FLY_SPLATTER = 0x7  # Camera splat KO. True: DEAD_UP_FALL_HIT_CAMERA
    DEAD_FLY_SPLATTER_FLAT = 0x8  # Camera splat flat. True: DEAD_UP_FALL_HIT_CAMERA_FLAT
    DEAD_FLY_SPLATTER_ICE = 0x9  # Camera splat frozen. True: DEAD_UP_FALL_ICE
    DEAD_FLY_SPLATTER_FLAT_ICE = 0xA  # Camera splat flat frozen. True: DEAD_UP_FALL_HIT_CAMERA_ICE
    NOTHING_STATE = 0xB  # Inactive (Sheik/Zelda alt, Nana when Sopo). True: SLEEP
    ON_HALO_DESCENT = 0xC  # Respawn platform descent, intangible. True: REBIRTH
    ON_HALO_WAIT = 0x0D  # Waiting on respawn platform, intangible. True: REBIRTH_WAIT
    STANDING = 0x0E  # Idle stance, 11.5% of gameplay. True: WAIT
    WALK_SLOW = 0x0F  # Slow walk, stick 0.3-0.5. True: WALK_SLOW
    WALK_MIDDLE = 0x10  # Medium walk, stick 0.5-0.8. True: WALK_MIDDLE
    WALK_FAST = 0x11  # Fast walk, stick 0.8-1.0. True: WALK_FAST
    TURNING = 0x12  # Standing turnaround, 11 frames. True: TURN
    TURNING_RUN = 0x13  # Run turnaround, skid animation. True: TURN_RUN
    DASHING = 0x14  # Initial dash, 8.4% of gameplay. True: DASH
    RUNNING = 0x15  # Full run after dash. True: RUN
    RUN_DIRECT = 0x16  # Run direction change. True: RUN_DIRECT
    RUN_BRAKE = 0x17  # Run stop, 18 frames Fox. True: RUN_BRAKE
    KNEE_BEND = 0x18  # Jumpsquat, 3 frames Fox. True: KNEE_BEND
    JUMPING_FORWARD = 0x19  # First jump facing forward. True: JUMP_F
    JUMPING_BACKWARD = 0x1A  # First jump facing backward. True: JUMP_B
    JUMPING_ARIAL_FORWARD = 0x1B  # Double jump forward. True: JUMP_AERIAL_F
    JUMPING_ARIAL_BACKWARD = 0x1C  # Double jump backward. True: JUMP_AERIAL_B
    FALLING = 0x1D  # Aerial wait, 10% of gameplay. True: FALL
    FALLING_FORWARD = 0x1E  # Fall with forward drift. True: FALL_F
    FALLING_BACKWARD = 0x1F  # Fall with backward drift. True: FALL_B
    FALLING_AERIAL = 0x20  # Post-double-jump fall, no jump. True: FALL_AERIAL
    FALLING_AERIAL_FORWARD = 0x21  # Post-double-jump forward drift. True: FALL_AERIAL_F
    FALLING_AERIAL_BACKWARD = 0x22  # Post-double-jump backward drift. True: FALL_AERIAL_B
    DEAD_FALL = 0x23  # Helpless after up-B/airdodge. True: FALL_SPECIAL
    SPECIAL_FALL_FORWARD = 0x24  # Helpless forward drift. True: FALL_SPECIAL_F
    SPECIAL_FALL_BACK = 0x25  # Helpless backward drift. True: FALL_SPECIAL_B
    TUMBLING = 0x26  # Knockback tumble, can tech. True: DAMAGE_FALL
    CROUCH_START = 0x27  # Crouch entry, 3 frames. True: SQUAT
    CROUCHING = 0x28  # Crouch hold, can cancel. True: SQUAT_WAIT
    CROUCH_END = 0x29  # Stand from crouch. True: SQUAT_RV
    LANDING = 0x2A  # Normal landing, 4 frames. True: LANDING
    LANDING_SPECIAL = 0x2B  # Wavedash landing, 10 frames. True: LANDING_FALL_SPECIAL
    NEUTRAL_ATTACK_1 = 0x2C  # Jab 1, 5% of gameplay. True: ATTACK_11
    NEUTRAL_ATTACK_2 = 0x2D  # Jab 2, chains from jab 1. True: ATTACK_12
    NEUTRAL_ATTACK_3 = 0x2E  # Jab 3 (if char has it). True: ATTACK_13
    LOOPING_ATTACK_START = 0x2F  # Rapid jab startup. True: ATTACK_100_START
    LOOPING_ATTACK_MIDDLE = 0x30  # Rapid jab loop. True: ATTACK_100_LOOP
    LOOPING_ATTACK_END = 0x31  # Rapid jab finish. True: ATTACK_100_END
    DASH_ATTACK = 0x32  # Dash attack, 25 frames. True: ATTACK_DASH
    FTILT_HIGH = 0x33  # Forward tilt angled up. True: ATTACK_S_3_HI
    FTILT_HIGH_MID = 0x34  # Forward tilt high-mid. True: ATTACK_S_3_HI_S
    FTILT_MID = 0x35  # Forward tilt neutral. True: ATTACK_S_3_S
    FTILT_LOW_MID = 0x36  # Forward tilt low-mid. True: ATTACK_S_3_LW_S
    FTILT_LOW = 0x37  # Forward tilt angled down. True: ATTACK_S_3_LW
    UPTILT = 0x38  # Up tilt, 2.1% of gameplay. True: ATTACK_HI_3
    DOWNTILT = 0x39  # Down tilt. True: ATTACK_LW_3
    FSMASH_HIGH = 0x3A  # Forward smash angled up. True: ATTACK_S_4_HI
    FSMASH_MID_HIGH = 0x3B  # Forward smash high-mid. True: ATTACK_S_4_HI_S
    FSMASH_MID = 0x3C  # Forward smash neutral, 37 frames. True: ATTACK_S_4_S
    FSMASH_MID_LOW = 0x3D  # Forward smash low-mid. True: ATTACK_S_4_LW_S
    FSMASH_LOW = 0x3E  # Forward smash angled down. True: ATTACK_S_4_LW
    UPSMASH = 0x3F  # Up smash, 41 frames. True: ATTACK_HI_4
    DOWNSMASH = 0x40  # Down smash, 39 frames. True: ATTACK_LW_4
    NAIR = 0x41  # Neutral air, 5.5% of gameplay. True: ATTACK_AIR_N
    FAIR = 0x42  # Forward air, 1.1% of gameplay. True: ATTACK_AIR_F
    BAIR = 0x43  # Back air, 3% of gameplay. True: ATTACK_AIR_B
    UAIR = 0x44  # Up air, 2% of gameplay. True: ATTACK_AIR_HI
    DAIR = 0x45  # Down air, drill. True: ATTACK_AIR_LW
    NAIR_LANDING = 0x46  # Nair landing lag, 15/7 frames. True: LANDING_AIR_N
    FAIR_LANDING = 0x47  # Fair landing lag, 17/8 frames. True: LANDING_AIR_F
    BAIR_LANDING = 0x48  # Bair landing lag, 18/9 frames. True: LANDING_AIR_B
    UAIR_LANDING = 0x49  # Uair landing lag, 15/7 frames. True: LANDING_AIR_HI
    DAIR_LANDING = 0x4A  # Dair landing lag, 20/10 frames. True: LANDING_AIR_LW
    DAMAGE_HIGH_1 = 0x4B  # Grounded hit high, light KB. True: DAMAGE_HI_1
    DAMAGE_HIGH_2 = 0x4C  # Grounded hit high, medium KB. True: DAMAGE_HI_2
    DAMAGE_HIGH_3 = 0x4D  # Grounded hit high, heavy KB. True: DAMAGE_HI_3
    DAMAGE_NEUTRAL_1 = 0x4E  # Grounded hit neutral, light KB. True: DAMAGE_N_1
    DAMAGE_NEUTRAL_2 = 0x4F  # Grounded hit neutral, medium KB. True: DAMAGE_N_2
    DAMAGE_NEUTRAL_3 = 0x50  # Grounded hit neutral, heavy KB. True: DAMAGE_N_3
    DAMAGE_LOW_1 = 0x51  # Grounded hit low, light KB. True: DAMAGE_LW_1
    DAMAGE_LOW_2 = 0x52  # Grounded hit low, medium KB. True: DAMAGE_LW_2
    DAMAGE_LOW_3 = 0x53  # Grounded hit low, heavy KB. True: DAMAGE_LW_3
    DAMAGE_AIR_1 = 0x54  # Aerial hit, light KB. True: DAMAGE_AIR_1
    DAMAGE_AIR_2 = 0x55  # Aerial hit, medium KB. True: DAMAGE_AIR_2
    DAMAGE_AIR_3 = 0x56  # Aerial hit, heavy KB. True: DAMAGE_AIR_3
    DAMAGE_FLY_HIGH = 0x57  # Tumble upward, 70-110 deg. True: DAMAGE_FLY_HI
    DAMAGE_FLY_NEUTRAL = 0x58  # Tumble horizontal, 5.2% of gameplay. True: DAMAGE_FLY_N
    DAMAGE_FLY_LOW = 0x59  # Tumble downward. True: DAMAGE_FLY_LW
    DAMAGE_FLY_TOP = 0x5A  # Tumble straight up, 5.3% of gameplay. True: DAMAGE_FLY_TOP
    DAMAGE_FLY_ROLL = 0x5B  # Reeling tumble, 30% chance at 100%+. True: DAMAGE_FLY_ROLL
    ITEM_PICKUP_LIGHT = 0x5C  # Picking up a light item
    ITEM_PICKUP_HEAVY = 0x5D  #
    ITEM_THROW_LIGHT_FORWARD = 0x5E
    ITEM_THROW_LIGHT_BACK = 0x5F
    ITEM_THROW_LIGHT_HIGH = 0x60
    ITEM_THROW_LIGHT_LOW = 0x61
    ITEM_THROW_LIGHT_DASH = 0x62
    ITEM_THROW_LIGHT_DROP = 0x63
    ITEM_THROW_LIGHT_AIR_FORWARD = 0x64
    ITEM_THROW_LIGHT_AIR_BACK = 0x65
    ITEM_THROW_LIGHT_AIR_HIGH = 0x66
    ITEM_THROW_LIGHT_AIR_LOW = 0x67
    ITEM_THROW_HEAVY_FORWARD = 0x68
    ITEM_THROW_HEAVY_BACK = 0x69
    ITEM_THROW_HEAVY_HIGH = 0x6A
    ITEM_THROW_HEAVY_LOW = 0x6B
    ITEM_THROW_LIGHT_SMASH_FORWARD = 0x6C
    ITEM_THROW_LIGHT_SMASH_BACK = 0x6D
    ITEM_THROW_LIGHT_SMASH_UP = 0x6E
    ITEM_THROW_LIGHT_SMASH_DOWN = 0x6F
    ITEM_THROW_LIGHT_AIR_SMASH_FORWARD = 0x70
    ITEM_THROW_LIGHT_AIR_SMASH_BACK = 0x71
    ITEM_THROW_LIGHT_AIR_SMASH_HIGH = 0x72
    ITEM_THROW_LIGHT_AIR_SMASH_LOW = 0x73
    ITEM_THROW_HEAVY_AIR_SMASH_FORWARD = 0x74
    ITEM_THROW_HEAVY_AIR_SMASH_BACK = 0x75
    ITEM_THROW_HEAVY_AIR_SMASH_HIGH = 0x76
    ITEM_THROW_HEAVY_AIR_SMASH_LOW = 0x77
    BEAM_SWORD_SWING_1 = 0x78
    BEAM_SWORD_SWING_2 = 0x79
    BEAM_SWORD_SWING_3 = 0x7A
    BEAM_SWORD_SWING_4 = 0x7B
    BAT_SWING_1 = 0x7C
    BAT_SWING_2 = 0x7D
    BAT_SWING_3 = 0x7E
    BAT_SWING_4 = 0x7F
    PARASOL_SWING_1 = 0x80
    PARASOL_SWING_2 = 0x81
    PARASOL_SWING_3 = 0x82
    PARASOL_SWING_4 = 0x83
    FAN_SWING_1 = 0x84
    FAN_SWING_2 = 0x85
    FAN_SWING_3 = 0x86
    FAN_SWING_4 = 0x87
    STAR_ROD_SWING_1 = 0x88
    STAR_ROD_SWING_2 = 0x89
    STAR_ROD_SWING_3 = 0x8A
    STAR_ROD_SWING_4 = 0x8B
    LIP_STICK_SWING_1 = 0x8C
    LIP_STICK_SWING_2 = 0x8D
    LIP_STICK_SWING_3 = 0x8E
    LIP_STICK_SWING_4 = 0x8F
    ITEM_PARASOL_OPEN = 0x90
    ITEM_PARASOL_FALL = 0x91
    ITEM_PARASOL_FALL_SPECIAL = 0x92
    ITEM_PARASOL_DAMAGE_FALL = 0x93
    GUN_SHOOT = 0x94
    GUN_SHOOT_AIR = 0x95
    GUN_SHOOT_EMPTY = 0x96
    GUN_SHOOT_AIR_EMPTY = 0x97
    FIRE_FLOWER_SHOOT = 0x98
    FIRE_FLOWER_SHOOT_AIR = 0x99
    ITEM_SCREW = 0x9A
    ITEM_SCREW_AIR = 0x9B
    DAMAGE_SCREW = 0x9C
    DAMAGE_SCREW_AIR = 0x9D
    ITEM_SCOPE_START = 0x9E
    ITEM_SCOPE_RAPID = 0x9F
    ITEM_SCOPE_FIRE = 0xA0
    ITEM_SCOPE_END = 0xA1
    ITEM_SCOPE_AIR_START = 0xA2
    ITEM_SCOPE_AIR_RAPID = 0xA3
    ITEM_SCOPE_AIR_FIRE = 0xA4
    ITEM_SCOPE_AIR_END = 0xA5
    ITEM_SCOPE_START_EMPTY = 0xA6
    ITEM_SCOPE_RAPID_EMPTY = 0xA7
    ITEM_SCOPE_FIRE_EMPTY = 0xA8
    ITEM_SCOPE_END_EMPTY = 0xA9
    ITEM_SCOPE_AIR_START_EMPTY = 0xAA
    ITEM_SCOPE_AIR_RAPID_EMPTY = 0xAB
    ITEM_SCOPE_AIR_FIRE_EMPTY = 0xAC
    ITEM_SCOPE_AIR_END_EMPTY = 0xAD
    LIFT_WAIT = 0xAE
    LIFT_WALK_1 = 0xAF
    LIFT_WALK_2 = 0xB0
    LIFT_TURN = 0xB1
    """
    ┌──────────────────┬───────────────────┬──────────────┐
    │ is_shield_active │ hex(action_state) │ count_star() │
    ├──────────────────┼───────────────────┼──────────────┤
    │ true             │ B2                │ 75473        │
    │ true             │ B3                │ 102492       │
    │ false            │ B4                │ 14063        │
    │ true             │ B5                │ 64927        │
    │ false            │ B6                │ 3167         │
    │ true             │ B6                │ 30358        │
    └──────────────────┴───────────────────┴──────────────┘
    """
    SHIELD_START = 0xB2  # Shield startup, 4 frames. True: GUARD_ON
    SHIELD = 0xB3  # Shield hold, 3% of gameplay. True: GUARD
    SHIELD_RELEASE = 0xB4  # Shield drop, 15 frames. True: GUARD_OFF
    SHIELD_STUN = 0xB5  # Shield stun from hit. True: GUARD_SET_OFF
    SHIELD_REFLECT = 0xB6  # Powershield, frames 1-2. True: GUARD_REFLECT
    TECH_MISS_UP = 0xB7  # Missed tech bounce face-up. True: DOWN_BOUND_U
    LYING_GROUND_UP = 0xB8  # Knockdown wait face-up. True: DOWN_WAIT_U
    LYING_GROUND_UP_HIT = 0xB9  # Hit while lying face-up. True: DOWN_DAMAGE_U
    GROUND_GETUP = 0xBA  # Neutral getup face-up. True: DOWN_STAND_U
    GROUND_ATTACK_UP = 0xBB  # Getup attack face-up. True: DOWN_ATTACK_U
    GROUND_ROLL_FORWARD_UP = 0xBC  # Getup roll forward face-up. True: DOWN_FOWARD_U
    GROUND_ROLL_BACKWARD_UP = 0xBD  # Getup roll backward face-up. True: DOWN_BACK_U
    GROUND_SPOT_UP = 0xBE  # Not commonly used. True: DOWN_SPOT_U
    TECH_MISS_DOWN = 0xBF  # Missed tech bounce face-down. True: DOWN_BOUND_D
    LYING_GROUND_DOWN = 0xC0  # Knockdown wait face-down. True: DOWN_WAIT_D
    DAMAGE_GROUND = 0xC1  # Hit while lying face-down. True: DOWN_DAMAGE_D
    NEUTRAL_GETUP = 0xC2  # Neutral getup face-down. True: DOWN_STAND_D
    GETUP_ATTACK = 0xC3  # Getup attack face-down. True: DOWN_ATTACK_D
    GROUND_ROLL_FORWARD_DOWN = 0xC4  # Getup roll forward face-down. True: DOWN_FOWARD_D
    GROUND_ROLL_BACKWARD_DOWN = 0xC5  # Getup roll backward face-down. True: DOWN_BACK_D
    GROUND_ROLL_SPOT_DOWN = 0xC6  # Not commonly used. True: DOWN_SPOT_D
    NEUTRAL_TECH = 0xC7  # Tech in place, 26 frames, intang 1-20. True: PASSIVE
    FORWARD_TECH = 0xC8  # Tech roll forward, 40 frames. True: PASSIVE_STAND_F
    BACKWARD_TECH = 0xC9  # Tech roll backward, 40 frames. True: PASSIVE_STAND_B
    WALL_TECH = 0xCA  # Wall tech, intang 1-14. True: PASSIVE_WALL
    WALL_TECH_JUMP = 0xCB  # Wall tech jump. True: PASSIVE_WALL_JUMP
    CEILING_TECH = 0xCC  # Ceiling tech, very rare. True: PASSIVE_CEIL
    SHIELD_BREAK_FLY = 0xCD  # Shield break launch. True: SHIELD_BREAK_FLY
    SHIELD_BREAK_FALL = 0xCE  # Shield break fall. True: SHIELD_BREAK_FALL
    SHIELD_BREAK_DOWN_U = 0xCF  # Shield break land face-up. True: SHIELD_BREAK_DOWN_U
    SHIELD_BREAK_DOWN_D = 0xD0  # Shield break land face-down. True: SHIELD_BREAK_DOWN_D
    SHIELD_BREAK_STAND_U = 0xD1  # Shield break stand face-up. True: SHIELD_BREAK_STAND_U
    SHIELD_BREAK_STAND_D = 0xD2  # Shield break stand face-down. True: SHIELD_BREAK_STAND_D
    SHIELD_BREAK_TEETER = 0xD3  # Shield break near edge. True: FURA_FURA (211)
    GRAB = 0xD4  # Standing grab, 74.6% JC grab. True: CATCH
    GRAB_PULLING = 0xD5  # Grab connects, pulling opp. True: CATCH_PULL
    GRAB_RUNNING = 0xD6  # Dash grab, more endlag. True: CATCH_DASH
    GRAB_RUNNING_PULLING = 0xD7  # Dash grab connects. True: CATCH_DASH_PULL
    GRAB_WAIT = 0xD8  # Holding grabbed opponent. True: CATCH_WAIT
    GRAB_PUMMEL = 0xD9  # Pummel attack, 14.3% of grabs. True: CATCH_ATTACK
    GRAB_BREAK = 0xDA  # Opponent mashes out. True: CATCH_CUT
    THROW_FORWARD = 0xDB  # Forward throw, 13.3% of throws. True: THROW_F
    THROW_BACK = 0xDC  # Back throw, 18.3% of throws. True: THROW_B
    THROW_UP = 0xDD  # Up throw, 67.1% of throws. True: THROW_HI
    THROW_DOWN = 0xDE  # Down throw, 1.3% of throws. True: THROW_LW
    GRAB_PULLING_HIGH = 0xDF  # Being grabbed pull (high). True: CAPTURE_PULLED_HI
    GRABBED_WAIT_HIGH = 0xE0  # Being held (high). True: CAPTURE_WAIT_HI
    PUMMELED_HIGH = 0xE1  # Being pummeled (high). True: CAPTURE_DAMAGE_HI
    GRAB_PULL = 0xE2  # Being grabbed pull (low). True: CAPTURE_PULLED_LW
    GRABBED = 0xE3  # Being held (low). True: CAPTURE_WAIT_LW
    GRAB_PUMMELED = 0xE4  # Being pummeled (low). True: CAPTURE_DAMAGE_LW
    GRAB_ESCAPE = 0xE5  # Mash out success. True: CAPTURE_CUT
    GRAB_JUMP = 0xE6  # Jump mash out. True: CAPTURE_JUMP
    GRAB_NECK = 0xE7  # Unused. True: CAPTURE_NECK
    GRAB_FOOT = 0xE8  # Unused. True: CAPTURE_FOOT
    ROLL_FORWARD = 0xE9  # Roll forward, intang 4-19. True: ESCAPE_F
    ROLL_BACKWARD = 0xEA  # Roll backward, intang 4-19. True: ESCAPE_B
    SPOTDODGE = 0xEB  # Spot dodge, intang 2-15. True: ESCAPE
    AIRDODGE = 0xEC  # Air dodge, 97.9% wavedash. True: ESCAPE_AIR
    REBOUND_STOP = 0xED  # Clank recoil grounded. True: REBOUND_STOP
    REBOUND = 0xEE  # Clank recoil aerial. True: REBOUND
    THROWN_FORWARD = 0xEF  # Being forward thrown. True: THROWN_F
    THROWN_BACK = 0xF0  # Being back thrown. True: THROWN_B
    THROWN_UP = 0xF1  # Being up thrown. True: THROWN_HI
    THROWN_DOWN = 0xF2  # Being down thrown. True: THROWN_LW
    THROWN_DOWN_2 = 0xF3  # Unused. True: THROWN_LW_WOMEN
    PLATFORM_DROP = 0xF4  # Platform drop-through. True: PASS
    EDGE_TEETERING_START = 0xF5  # Teeter start. True: OTTOTTO
    EDGE_TEETERING = 0xF6  # Teeter loop. True: OTTOTTO_WAIT
    BOUNCE_WALL = 0xF7  # Missed wall tech bounce. True: FLY_REFLECT_WALL
    BOUNCE_CEILING = 0xF8  # Missed ceiling tech bounce. True: FLY_REFLECT_CEIL
    BUMP_WALL = 0xF9  # Wall collision. True: STOP_WALL
    BUMP_CIELING = 0xFA  # Ceiling collision. True: STOP_CEIL
    SLIDING_OFF_EDGE = 0xFB  # Missed ledge sweetspot. True: MISS_FOOT
    EDGE_CATCHING = 0xFC  # Ledge grab, intang 1-7. True: CLIFF_CATCH
    EDGE_HANGING = 0xFD  # Ledge hang, 82.1% drop off. True: CLIFF_WAIT
    EDGE_GETUP_SLOW = 0xFE  # Ledge climb >=100%. True: CLIFF_CLIMB_SLOW
    EDGE_GETUP_QUICK = 0xFF  # Ledge climb <100%. True: CLIFF_CLIMB_QUICK
    EDGE_ATTACK_SLOW = 0x100  # Ledge attack >=100%. True: CLIFF_ATTACK_SLOW
    EDGE_ATTACK_QUICK = 0x101  # Ledge attack <100%. True: CLIFF_ATTACK_QUICK
    EDGE_ROLL_SLOW = 0x102  # Ledge roll >=100%. True: CLIFF_ESCAPE_SLOW
    EDGE_ROLL_QUICK = 0x103  # Ledge roll <100%. True: CLIFF_ESCAPE_QUICK
    EDGE_JUMP_1_SLOW = 0x104  # Ledge jump >=100% pt1. True: CLIFF_JUMP_SLOW_1
    EDGE_JUMP_2_SLOW = 0x105  # Ledge jump >=100% pt2. True: CLIFF_JUMP_SLOW_2
    EDGE_JUMP_1_QUICK = 0x106  # Ledge jump <100% pt1. True: CLIFF_JUMP_QUICK_1
    EDGE_JUMP_2_QUICK = 0x107  # Ledge jump <100% pt2. True: CLIFF_JUMP_QUICK_2
    TAUNT_RIGHT = 0x108  # Taunt facing right. True: APPEAL_R
    TAUNT_LEFT = 0x109  # Taunt facing left. True: APPEAL_L
    SHOULDERED_WAIT = 0x10A  # DK Carry
    SHOULDERED_WALK_SLOW = 0x10B
    SHOULDERED_WALK_MIDDLE = 0x10C
    SHOULDERED_WALK_FAST = 0x10D
    SHOULDERED_TURN = 0x10E
    THROWN_FF = 0x10F  # DK carry throws
    THROWN_FB = 0x110
    THROWN_F_HIGH = 0x111
    THROWN_F_LOW = 0x112
    CAPTURE_CAPTAIN = 0x113
    CAPTURE_YOSHI = 0x114
    YOSHI_EGG = 0x115
    CAPTURE_KOOPA = 0x116
    CAPTURE_DAMAGE_KOOPA = 0x117
    CAPTURE_WAIT_KOOPA = 0x118
    THROWN_KOOPA_F = 0x119
    THROWN_KOOPA_B = 0x11A
    CAPTURE_KOOPA_AIR = 0x11B
    CAPTURE_DAMAGE_KOOPA_AIR = 0x11C
    CAPTURE_WAIT_KOOPA_AIR = 0x11D
    THROWN_KOOPA_AIR_F = 0x11E
    THROWN_KOOPA_AIR_B = 0x11F
    CAPTURE_KIRBY = 0x120
    CAPTURE_WAIT_KIRBY = 0x121
    THROWN_KIRBY_STAR = 0x122
    THROWN_COPY_STAR = 0x123
    THROWN_KIRBY = 0x124
    BARREL_WAIT = 0x125
    BURY = 0x126  # Stuck from DK side-B
    BURY_WAIT = 0x127
    BURY_JUMP = 0x128
    DAMAGE_SONG = 0x129  # Put to sleep from Jiggly up-B
    DAMAGE_SONG_WAIT = 0x12A
    DAMAGE_SONG_RV = 0x12B
    DAMAGE_BIND = 0x12C  # Hit by Mewtwo's disable
    CAPTURE_MEWTWO = 0x12D  # Unused
    CAPTURE_MEWTWO_AIR = 0x12E  # Unused
    THROWN_MEWTWO = 0x12F  # Hit by Mewtwo's Confusion
    THROWN_MEWTWO_AIR = 0x130  # Hit by Mewtwo's Confusion
    WARP_STAR_JUMP = 0x131
    WARP_STAP_FALL = 0x132
    HAMMER_WAIT = 0x133
    HAMMER_WALK = 0x134
    HAMMER_TURN = 0x135
    HAMMER_KNEE_BEND = 0x136
    HAMMER_FALL = 0x137
    HAMMER_JUMP = 0x138
    HAMMER_LANDING = 0x139
    KINOKO_GIANT_START = 0x13A  # Super mushroom states
    KINOKO_GIANT_START_AIR = 0x13B
    KINOKO_GIANT_END = 0x13C
    KINOKO_GIANT_END_AIR = 0x13D
    KINOKO_SMALL_START = 0x13E  # Poison mushroom states
    KINOKO_SMALL_START_AIR = 0x13F
    KINOKO_SMALL_END = 0x140
    KINOKO_SMALL_END_AIR = 0x141
    ENTRY = 0x142  # Start of match. Can't move
    ENTRY_START = 0x143  # Start of match. Can't move
    ENTRY_END = 0x144  # Start of match. Can't move
    DAMAGE_ICE = 0x145
    DAMAGE_ICE_JUMP = 0x146
    CAPTURE_MASTERHAND = 0x147
    CAPTURE_DAMAGE_MASTERHAND = 0x148
    CAPTURE_WAIT_MASTERHAND = 0x149
    THROWN_MASTERHAND = 0x14A
    CAPTURE_KIRBY_YOSHI = 0x14B
    KIRBY_YOSHI_EGG = 0x14C
    CAPTURE_LEA_DEAD = 0x14D  # No idea what this is
    CAPTURE_LIKE_LIKE = 0x14E  # No idea what this is either
    DOWN_REFLECT = 0x14F  # Jab reset knockdown. True: DOWN_REFLECT
    CAPTURE_CRAZYHAND = 0x150
    CAPTURE_DAMAGE_CRAZYHAND = 0x151
    CAPTURE_WAIT_CRAZYHAND = 0x152
    THROWN_CRAZY_HAND = 0x153
    BARREL_CANNON_WAIT = 0x154
    LASER_GUN_PULL = 0x155  # Fox laser startup grounded. True: BLASTER_GROUND_STARTUP
    NEUTRAL_B_CHARGING = 0x156  # Fox laser loop grounded. True: BLASTER_GROUND_LOOP
    NEUTRAL_B_ATTACKING = 0x157  # Fox laser end grounded. True: BLASTER_GROUND_END
    NEUTRAL_B_FULL_CHARGE = 0x158  # Fox laser startup aerial. True: BLASTER_AIR_STARTUP
    WAIT_ITEM = 0x159  # Fox laser loop aerial. True: BLASTER_AIR_LOOP
    NEUTRAL_B_CHARGING_AIR = 0x15A  # Fox laser end aerial. True: BLASTER_AIR_END
    NEUTRAL_B_ATTACKING_AIR = 0x15B  # Fox Illusion startup grounded. True: ILLUSION_GROUND_STARTUP
    NEUTRAL_B_FULL_CHARGE_AIR = 0x15C  # Fox Illusion main grounded. True: ILLUSION_GROUND
    DOWN_B_GROUND_START = 0x168  # Shine startup, frame 1 intang. True: REFLECTOR_GROUND_STARTUP
    DOWN_B_GROUND = 0x169  # Shine hold grounded. True: REFLECTOR_GROUND_LOOP
    SHINE_TURN = 0x16C  # Shine turnaround grounded. True: REFLECTOR_GROUND_CHANGE_DIRECTION
    DOWN_B_STUN = 0x16D  # Shine startup aerial. True: REFLECTOR_AIR_STARTUP
    DOWN_B_AIR = 0x16E  # Shine hold aerial. True: REFLECTOR_AIR_LOOP
    UP_B_GROUND = 0x16F  # Shine reflect aerial. True: REFLECTOR_AIR_REFLECT
    SHINE_RELEASE_AIR = 0x170  # Shine end aerial. True: REFLECTOR_AIR_END
    SWORD_DANCE_1 = 0x15D
    SWORD_DANCE_2_HIGH = 0x15E
    SWORD_DANCE_2_MID = 0x15F
    SWORD_DANCE_3_HIGH = 0x160
    SWORD_DANCE_3_MID = 0x161
    SWORD_DANCE_3_LOW = 0x162
    SWORD_DANCE_4_HIGH = 0x163
    SWORD_DANCE_4_MID = 0x164
    SWORD_DANCE_4_LOW = 0x165
    SWORD_DANCE_1_AIR = 0x166
    SWORD_DANCE_2_HIGH_AIR = 0x167
    SWORD_DANCE_2_MID_AIR = 0x168
    SWORD_DANCE_3_HIGH_AIR = 0x169
    SWORD_DANCE_3_MID_AIR = 0x16A
    SWORD_DANCE_3_LOW_AIR = 0x16B
    SWORD_DANCE_4_HIGH_AIR = 0x16C
    SWORD_DANCE_4_MID_AIR = 0x16D
    SWORD_DANCE_4_LOW_AIR = 0x16E
    FOX_ILLUSION_START = 0x15E  # Illusion startup aerial. True: ILLUSION_STARTUP_AIR (350)
    FOX_ILLUSION = 0x15F  # Illusion main aerial. True: ILLUSION_AIR (351)
    FOX_ILLUSION_SHORTENED = 0x160  # Illusion end aerial. True: ILLUSION_AIR_END (352)
    FIREFOX_WAIT_GROUND = 0x161  # Fire Fox charge grounded. True: FIRE_FOX_GROUND_STARTUP
    FIREFOX_WAIT_AIR = 0x162  # Fire Fox charge aerial. True: FIRE_FOX_AIR_STARTUP
    FIREFOX_GROUND = 0x163  # Fire Fox travel grounded. True: FIRE_FOX_GROUND
    FIREFOX_AIR = 0x164  # Fire Fox travel aerial. True: FIRE_FOX_AIR
    UP_B_AIR = 0x170  # The upswing of the UP-B. (At least for marth)
    MARTH_COUNTER = 0x171
    PARASOL_FALLING = 0x172
    MARTH_COUNTER_FALLING = 0x173
    NESS_SHEILD_START = 0x174
    NESS_SHEILD = 0x174
    NESS_SHEILD_AIR = 0x175
    ZITABATA = 0x176  # No clue what this is
    NESS_SHEILD_AIR_END = 0x177
    THROWN_KOOPA_END_F = 0x178
    THROWN_KOOPA_END_B = 0x179
    CAPTURE_KOOPA_AIR_HIT = 0x17A
    THROWN_KOOPA_AIR_END_F = 0x17B
    THROWN_KOOPA_AIR_END_B = 0x17C
    THROWN_KIRBY_DRINK_S_SHOT = 0x17D
    THROWN_KIRBY_SPIT_S_SHOT = 0x17E
    DK_GROUND_POUND_START = 0x17F
    DK_GROUND_POUND = 0x180
    DK_GROUND_POUND_END = 0x181
    KIRBY_BLADE_GROUND = 0x184
    KIRBY_BLADE_UP = 0x185
    KIRBY_BLADE_APEX = 0x186
    KIRBY_BLADE_DOWN = 0x187
    KIRBY_STONE_FORMING_GROUND = 0x189
    KIRBY_STONE_RESTING = 0x18A
    KIRBY_STONE_RELEASE = 0x18B
    KIRBY_STONE_FORMING_AIR = 0x18C
    KIRBY_STONE_FALLING = 0x18D
    KIRBY_STONE_UNFORMING = 0x18D
    UNKNOWN_ANIMATION = 0xFFFF


class ProjectileType(Enum):
    """Primary type of prejectile or item"""

    BOB_OMB = 0x06  # Bob-omb (BombHei)
    MR_SATURN = 0x07  # Mr. Saturn (Dosei)
    BEAMSWORD = 0x0C  # Beam Sword
    MARIO_FIREBALL = 0x30  # Mario's fire
    DR_MARIO_CAPSULE = 0x31  # Dr.Mario's Capsule
    KIRBY_CUTTER = 0x32  # Kirby's Cutter beam
    KIRBY_HAMMER = 0x33  # Kirby's Hammer
    FOX_LASER = 0x36  # Fox's Laser
    FALCO_LASER = 0x37  # Falco's Laser
    FOX_SHADOW = 0x38  # Fox's shadow
    FALCO_SHADOW = 0x39  # Falco's shadow
    LINK_BOMB = 0x3A  # Link's bomb
    YLINK_BOMB = 0x3B  # Young Link's bomb
    LINK_BOOMERANG = 0x3C  # Link's boomerang
    YLINK_BOOMERANG = 0x3D  # Young Link's boomerang
    LINK_HOOKSHOT = 0x3E  # Link's Hookshot
    YLINK_HOOKSHOT = 0x3F  # Young Link's Hookshot
    ARROW = 0x40  # Arrow
    FIRE_ARROW = 0x41  # Fire Arrow
    PK_FIRE = 0x42  # PK Fire
    PK_FLASH_1 = 0x43  # PK Flash
    PK_FLASH_2 = 0x44  # PK Flash
    PK_THUNDER_HEAD = 0x45  # PK Thunder (Primary)
    PK_THUNDER_TAIL_1 = 0x46  # PK Thunder
    PK_THUNDER_TAIL_2 = 0x47  # PK Thunder
    PK_THUNDER_TAIL_3 = 0x48  # PK Thunder
    PK_THUNDER_TAIL_4 = 0x49  # PK Thunder
    LINK_ARROW = 0x4C  # Link's Arrow
    YLINK_ARROW = 0x4D  # Young Link's arrow
    PK_FLASH_EXPLOSION = 0x4E  # PK Flash (explosion)
    NEEDLE_THROWN = 0x4F  # Needle(thrown)
    PIKACHU_THUNDER = 0x51  # Pikachu's Thunder
    PICHU_THUNDER = 0x52  # Pichu's Thunder
    MARIO_CAPE = 0x53  # Mario's cape
    DR_MARIO_CAPE = 0x54  # Dr.Mario's cape
    SHEIK_SMOKE = 0x55  # Smoke (Sheik)
    YOSHI_EGG_THROWN = 0x56  # Yoshi's egg(thrown)
    YOSHI_TONGUE = 0x57  # Yoshi's Tongue??
    YOSHI_STAR = 0x58  # Yoshi's Star
    PIKACHU_THUNDERJOLT_1 = 0x59  # Pikachu's thunder (B)
    PIKACHU_THUNDERJOLT_2 = 0x5A  # Pikachu's thunder (B)
    PICHU_THUNDERJOLT_1 = 0x5B  # Pichu's thunder (B)
    PICHU_THUNDERJOLT_2 = 0x5C  # Pichu's thunder (B)
    SAMUS_BOMB = 0x5D  # Samus's bomb
    SAMUS_CHARGE_BEAM = 0x5E  # Samus's chargeshot
    SAMUS_MISSLE = 0x5F  # Missile
    SAMUS_GRAPPLE_BEAM = 0x60  # Grapple beam
    SHEIK_CHAIN = 0x61  # Sheik's chain
    TURNIP = 0x63  # Turnip
    BOWSER_FLAME = 0x64  # Bowser's flame
    NESS_BATT = 0x65  # Ness's bat
    NESS_YOYO = 0x66  # Yoyo
    PEACH_PARASOL = 0x67  # Peach's parasol
    LUIGI_FIRE = 0x69  # Luigi's fire
    ICE_BLOCK = 0x6A  # Ice(Iceclimbers)
    IC_BLIZZARD = 0x6B  # Blizzard
    ZELDA_FIRE = 0x6C  # Zelda's fire
    ZELDA_FIRE_EXPLOSION = 0x6D  # Zelda's fire (explosion)
    MEWTO_DISABLE = 0x6E  # Mewtwo's down-B
    TOAD_SPORE = 0x6F  # Toad's spore
    SHADOWBALL = 0x70  # Mewtwo's Shadowball
    IC_UP_B = 0x71  # Iceclimbers' Up  #B
    PESTICIDE = 0x72  # Pesticide
    MANHOLE = 0x73  # Manhole
    GW_FIRE = 0x74  # Fire(G&W)
    PARACHUTE = 0x75  # Parachute
    TURTLE = 0x76  # Turtle
    SPERKY = 0x77  # Sperky
    JUDGE = 0x78  # Judge
    SAUSAGE = 0x7A  # Sausage
    YLINK_MILK = 0x7B  # Milk (Young Link)
    FIREFIGHTER = 0x7C  # Firefighter(G&W)
    KIRBY_MARIO_FIRE = 0x82  # Kirby copy Mario's Fire (B)
    KIRBY_DR_MARIO_FIRE = 0x83  # Kirby copy Dr. Mario's Capsule (B)
    KIRBY_LUIGI_FIRE = 0x84  # Kirby copy Luigi's Fire (B)
    KIRBY_IC_BLOCK = 0x85  # Kirby copy IceClimber's IceCube (B)
    KIRBY_TOAD_SPORE = 0x87  # Kirby copy Toad's Spore (B)
    KIRBY_FOX_LASER = 0x88  # Kirby copy Fox's Laser (B)
    KIRBY_FALCO_LASER = 0x89  # Kirby copy Falco's Laser (B)
    KIRBY_LINK_ARROW = 0x8C  # Kirby copy Link's Arrow (B)
    KIRBY_YLINK_ARROW = 0x8D  # Kirby copy Young Link's Arrow (B)
    KIRBY_LINK_ARROW_2 = 0x8E  # Kirby copy Link's Arrow (B)
    KIRBY_YLINK_ARROW_2 = 0x8F  # Kirby copy Young Link's Arrow (B)
    KIRBY_SHADOWBALL = 0x90  # Kirby copy Mewtwo's Shadowball (B)
    KIRBY_PK_FLASH = 0x91  # Kirby copy PK Flash (B)
    KIRBY_PK_FLASH_EXPLOSION = 0x92  # Kirby copy PK Flash Explosion (B)
    KIRBY_PIKACHU_THUNDERJOLT_1 = 0x93  # Kirby copy Pikachu's Thunder (B)
    KIRBY_PIKACHU_THUNDERJOLT_2 = 0x94  # Kirby copy Pikachu's Thunder (B)
    KIRBY_PICHU_THUNDERJOLT_1 = 0x95  # Kirby copy Pichu's Thunder (B)
    KIRBY_PICHU_THUNDERJOLT_2 = 0x96  # Kirby copy Pichu's Thunder (B)
    KIRBY_SAMUS_CHARGESHOT = 0x97  # Kirby copy Samus' Chargeshot (B)
    KIRBY_SHEIK_NEEDLE_THROWN = 0x98  # Kirby copy Sheik's Needle (thrown) (B)
    KIRBY_SHEIK_NEEDLE_GROUND = 0x99  # Kirby copy Sheik's Needle (ground) (B)
    KIRBY_BOWSER_FLAME = 0x9A  # Kirby copy Bowser's Flame (B)
    KIRBY_SAUSAGE = 0x9B  # Kirby copy Mr. Game & Watch's Sausage (B)
    KIRBY_YOSHI_TONGUE = 0x9D  # Yoshi's Tongue?? (B)
    UNKNOWN_PROJECTILE = 0xFF
