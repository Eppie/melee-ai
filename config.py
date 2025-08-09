from __future__ import annotations

from pathlib import Path
from typing import List

REPLAYS_DIR = Path("/Users/eppie/Downloads/replays_sorted/FOX_vs_FOX")
OUT_DIR = Path("shards")
COMBINED_OUT_PATH = OUT_DIR / "all_frames.parquet"
CHUNK_SIZE = 1000  # number of replays per output file

COLUMNS = ['frame_id', 'p1_has_temp_intang', 'p1_is_fastfalling', 'p1_defender_hitlag', 'p1_in_hitlag',
           'p1_is_grabbing', 'p1_shield_active', 'p1_in_hitstun', 'p1_shield_touch', 'p1_powershield', 'p1_is_dead',
           'p1_offscreen', 'p1_pos_x', 'p1_pos_y', 'p1_action_state', 'p1_percent', 'p1_stocks', 'p1_character',
           'p1_btn_l_analog', 'p1_btn_r_analog', 'p1_btn_a', 'p1_btn_b', 'p1_btn_z', 'p1_btn_xy', 'p1_btn_lr',
           'p1_pre_joystick_x', 'p1_pre_joystick_y', 'p1_pre_cstick_x', 'p1_pre_cstick_y', 'p2_has_temp_intang',
           'p2_is_fastfalling', 'p2_defender_hitlag', 'p2_in_hitlag', 'p2_is_grabbing', 'p2_shield_active',
           'p2_in_hitstun', 'p2_shield_touch', 'p2_powershield', 'p2_is_dead', 'p2_offscreen', 'p2_pos_x', 'p2_pos_y',
           'p2_action_state', 'p2_percent', 'p2_stocks', 'p2_character', 'p2_btn_l_analog', 'p2_btn_r_analog',
           'p2_btn_a', 'p2_btn_b', 'p2_btn_z', 'p2_btn_xy', 'p2_btn_lr', 'p2_pre_joystick_x', 'p2_pre_joystick_y',
           'p2_pre_cstick_x', 'p2_pre_cstick_y', 'source_file']
CLASSIFICATION_TARGETS: List[str] = ['p1_btn_a', 'p1_btn_b', 'p1_btn_z', 'p1_btn_xy', 'p1_btn_lr']
REGRESSION_TARGETS: List[str] = ["p1_pre_joystick_x",
                                 "p1_pre_joystick_y",
                                 "p1_pre_cstick_x",
                                 "p1_pre_cstick_y", ]

TARGET_COLUMNS = CLASSIFICATION_TARGETS + REGRESSION_TARGETS
FEATURE_COLUMNS = TARGET_COLUMNS + [
    # 'p1_has_temp_intang',
    # 'p1_is_fastfalling',
    # 'p1_defender_hitlag',
    # 'p1_in_hitlag',
    # 'p1_is_grabbing',
    # 'p1_shield_active',
    # 'p1_in_hitstun',
    # 'p1_shield_touch',
    # 'p1_powershield',
    # 'p1_is_dead',
    # 'p1_offscreen',
    'p1_pos_x',
    'p1_pos_y',
    # 'p1_action_state',
    'p1_percent',
    # 'p1_stocks',
    'p1_character',
    # 'p2_has_temp_intang',
    # 'p2_is_fastfalling',
    # 'p2_defender_hitlag',
    # 'p2_in_hitlag',
    # 'p2_is_grabbing',
    # 'p2_shield_active',
    # 'p2_in_hitstun',
    # 'p2_shield_touch',
    # 'p2_powershield',
    # 'p2_is_dead',
    # 'p2_offscreen',
    'p2_pos_x',
    'p2_pos_y',
    # 'p2_action_state',
    'p2_percent',
    # 'p2_stocks',
    'p2_character',
    # 'p2_btn_l_analog',
    # 'p2_btn_r_analog',
    'p2_btn_a',
    'p2_btn_b',
    'p2_btn_z',
    'p2_btn_xy',
    'p2_btn_lr',
    'p2_pre_joystick_x',
    'p2_pre_joystick_y',
    'p2_pre_cstick_x',
    'p2_pre_cstick_y',
]
SEQUENCE_LENGTH = 60
BCE_INDICES: List[int] = [TARGET_COLUMNS.index(c) for c in CLASSIFICATION_TARGETS]
REG_INDICES = [TARGET_COLUMNS.index(col) for col in REGRESSION_TARGETS]
BUTTON_SLICE = slice(0, len(CLASSIFICATION_TARGETS))           # 0‥4
STICK_SLICE = slice(len(CLASSIFICATION_TARGETS), len(TARGET_COLUMNS))
BATCH_SIZE = 1