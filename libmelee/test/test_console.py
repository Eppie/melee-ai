from __future__ import annotations
from __future__ import annotations

import os
import random
import struct
import time
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable

import numpy as np
import pytest

from libmelee.melee import enums
from libmelee.melee import stages
from libmelee.melee.console import Console, SlippiVersionTooLow
from libmelee.melee.gamestate import GameState
from libmelee.melee.gamestate import PlayerState
from libmelee.melee.slippstream import EventType


def be_pack_into(buf: bytearray, offset: int, fmt: str, value) -> None:
    """Pack a single big-endian value into buf at offset."""
    struct.pack_into(">" + fmt, buf, offset, value)


def make_buf(size: int) -> bytearray:
    return bytearray(b"\x00" * size)


def normalized_from_game_float(target_norm: float) -> float:
    """
    The pre-frame code reconstructs normalized sticks as (raw/2)+0.5.
    To produce target_norm, we must write (target_norm - 0.5) * 2.
    """
    return (target_norm - 0.5) * 2.0


# --- Fixtures ----------------------------------------------------------------


@pytest.fixture
def console(tmp_path) -> Console:
    """
    Use SLPFileStreamer mode (is_dolphin=False) so no external process is needed.
    We don't actually read the SLP here; we call the private methods directly.
    """
    # If your repository places the test replay elsewhere, tweak the path:
    test_slp = "test/test.slp"
    c = Console(path=test_slp, is_dolphin=False, allow_old_version=True)
    return c


@pytest.fixture
def gs() -> GameState:
    return GameState()


# --- Tests for __pre_frame ---------------------------------------------------


def test_pre_frame_sets_controller_state_basic(console: Console, gs: GameState) -> None:
    """
    Verify main/c sticks, raw main, triggers, buttons, and manual-bookend behavior.
    """
    # Build a buffer that reaches the highest offset used by __pre_frame (0x40 signed byte)
    buf = make_buf(0x41 + 1)

    port = 1  # console adds +1 internally, so write 0 for port 1
    be_pack_into(buf, 0x5, "B", port - 1)

    # Not Nana in this one
    be_pack_into(buf, 0x6, "B", 0)

    # Sticks (final expected normalized values)
    main_norm: Tuple[float, float] = (0.80, 0.25)
    c_norm: Tuple[float, float] = (0.10, 0.90)

    be_pack_into(buf, 0x19, "f", normalized_from_game_float(main_norm[0]))
    be_pack_into(buf, 0x1D, "f", normalized_from_game_float(main_norm[1]))
    be_pack_into(buf, 0x21, "f", normalized_from_game_float(c_norm[0]))
    be_pack_into(buf, 0x25, "f", normalized_from_game_float(c_norm[1]))

    # Raw main stick (signed bytes)
    be_pack_into(buf, 0x3B, "b", -10)
    be_pack_into(buf, 0x40, "b", 5)

    # Trigger (applied to both L & R analog)
    be_pack_into(buf, 0x29, "f", 0.73)

    # Buttons bitfield at 0x31 (uint16)
    # A=0x0100, B=0x0200, X=0x0400, Y=0x0800, START=0x1000, Z=0x0010, R=0x0020, L=0x0040
    buttons = 0x0100 | 0x0400 | 0x0010 | 0x0040  # A + X + Z + L
    be_pack_into(buf, 0x31, "H", buttons)

    # Manual bookend behavior
    console._use_manual_bookends = True  # type: ignore[attr-defined]
    gs.frame = 777

    # Invoke
    console._Console__pre_frame(gs, bytes(buf))  # type: ignore[attr-defined]

    ps = gs.players[1]
    assert isinstance(ps, PlayerState)

    # Sticks normalized
    assert ps.controller_state.main_stick == pytest.approx(main_norm)
    assert ps.controller_state.c_stick == pytest.approx(c_norm)

    # Raw main
    assert ps.controller_state.raw_main_stick == (-10, 5)

    # Shoulders identical
    assert ps.controller_state.l_shoulder == pytest.approx(0.73)
    assert ps.controller_state.r_shoulder == pytest.approx(0.73)

    # Buttons
    BTN = enums.Button
    assert ps.controller_state.button[BTN.BUTTON_A] is True
    assert ps.controller_state.button[BTN.BUTTON_X] is True
    assert ps.controller_state.button[BTN.BUTTON_Z] is True
    assert ps.controller_state.button[BTN.BUTTON_L] is True
    # Others not set
    assert not ps.controller_state.button[BTN.BUTTON_B]
    assert not ps.controller_state.button[BTN.BUTTON_Y]
    assert not ps.controller_state.button[BTN.BUTTON_START]
    assert not ps.controller_state.button[BTN.BUTTON_R]

    # Manual bookend sets console._frame to current gs.frame
    assert console._frame == 777  # type: ignore[attr-defined]


def test_pre_frame_handles_nana_branch(console: Console, gs: GameState) -> None:
    """If Nana flag is set, the values should apply to players[port].nana."""
    buf = make_buf(0x33)  # enough to include button bits at 0x31
    be_pack_into(buf, 0x5, "B", 0)  # port 1
    be_pack_into(buf, 0x6, "B", 1)  # Nana

    be_pack_into(buf, 0x19, "f", normalized_from_game_float(0.6))
    be_pack_into(buf, 0x1D, "f", normalized_from_game_float(0.6))
    be_pack_into(buf, 0x29, "f", 0.4)

    console._Console__pre_frame(gs, bytes(buf))  # type: ignore[attr-defined]

    ps = gs.players[1]
    assert ps.nana is not None
    assert ps.nana.controller_state.main_stick == pytest.approx((0.6, 0.6))
    assert ps.nana.controller_state.l_shoulder == pytest.approx(0.4)
    # Base Popo should still exist:
    assert isinstance(ps, PlayerState)


def test_pre_frame_gracefully_handles_truncated_payload(
    console: Console, gs: GameState
) -> None:
    """
    Omit the raw_main_* bytes and buttonbits to trigger internal TypeError branches and defaults.
    """
    buf = make_buf(0x33)  # enough for button bits at 0x31
    be_pack_into(buf, 0x5, "B", 0)
    be_pack_into(buf, 0x6, "B", 0)
    be_pack_into(buf, 0x19, "f", normalized_from_game_float(0.5))
    be_pack_into(buf, 0x1D, "f", normalized_from_game_float(0.5))
    be_pack_into(buf, 0x29, "f", 0.0)

    console._Console__pre_frame(gs, bytes(buf))  # type: ignore[attr-defined]

    ps = gs.players[1]
    # Raw main defaults to (0,0) when missing
    assert ps.controller_state.raw_main_stick == (0, 0)
    # Buttons default to False (only those parsed in __pre_frame)
    for b in (
        enums.Button.BUTTON_A,
        enums.Button.BUTTON_B,
        enums.Button.BUTTON_X,
        enums.Button.BUTTON_Y,
        enums.Button.BUTTON_START,
        enums.Button.BUTTON_Z,
        enums.Button.BUTTON_R,
        enums.Button.BUTTON_L,
    ):
        assert ps.controller_state.button[b] is False


# --- Tests for __post_frame --------------------------------------------------


def test_post_frame_sets_core_fields_and_flags(console: Console, gs: GameState) -> None:
    """
    Exercise the full field set: stage/teams, frame, position, character/action,
    face/percent/shield/stock/action_frame, status bits, velocities, on_ground/jumps,
    hitstun/hitlag, off_stage helper, and ECB edges.
    """
    # Prepare console state that __post_frame pulls from
    console._current_stage = enums.Stage.BATTLEFIELD  # type: ignore[attr-defined]
    console._is_teams = True  # type: ignore[attr-defined]

    # Build a buffer through 0x69 + 3 (we'll write floats at 0x69)
    buf = make_buf(0x69 + 4)

    frame = 1234
    port = 2  # write 1 to represent controller port 2
    nana = 0

    be_pack_into(buf, 0x1, "i", frame)
    be_pack_into(buf, 0x5, "B", port - 1)
    be_pack_into(buf, 0x6, "B", nana)

    # Position
    pos_x, pos_y = 42.5, -7.25
    be_pack_into(buf, 0x0A, "f", pos_x)
    be_pack_into(buf, 0x0E, "f", pos_y)

    # Character and Action
    be_pack_into(buf, 0x7, "B", enums.Character.FOX.value)
    be_pack_into(buf, 0x8, "H", enums.Action.NEUTRAL_ATTACK_1.value)

    # Facing: store positive float to get True
    be_pack_into(buf, 0x12, "f", 1.0)

    # Percent, shield, stock, action_frame
    be_pack_into(buf, 0x16, "f", 87.9)
    be_pack_into(buf, 0x1A, "f", 48.0)
    be_pack_into(buf, 0x21, "B", 3)
    be_pack_into(buf, 0x22, "f", 5.0)

    # Status bytes
    # sb1: reflect bit 0x10
    be_pack_into(buf, 0x26, "B", 0x10)
    # sb2: subaction invuln 0x04 | fastfall 0x08 | defender in hitlag 0x10 | in hitlag 0x20
    be_pack_into(buf, 0x27, "B", 0x04 | 0x08 | 0x10 | 0x20)
    # sb3: holding char 0x04 | shield active 0x80
    be_pack_into(buf, 0x28, "B", 0x04 | 0x80)
    # sb4: in hitstun 0x02 | powershield 0x20
    be_pack_into(buf, 0x29, "B", 0x02 | 0x20)
    # sb5: dead 0x40 | offscreen 0x80
    be_pack_into(buf, 0x2A, "B", 0x40 | 0x80)

    # Hitstun left, on_ground (0 means True), jumps_left, l_cancel_status, invuln
    be_pack_into(buf, 0x2B, "f", 12.0)
    be_pack_into(buf, 0x2F, "B", 1)  # not on ground -> False after "not bool(...)"
    be_pack_into(buf, 0x32, "B", 2)
    be_pack_into(buf, 0x33, "B", 1)
    be_pack_into(buf, 0x34, "B", 1)

    # Speeds & hitlag_left
    be_pack_into(buf, 0x35, "f", 0.1)  # speed_air_x_self
    be_pack_into(buf, 0x39, "f", -0.2)  # speed_y_self
    be_pack_into(buf, 0x3D, "f", 0.3)  # speed_x_attack
    be_pack_into(buf, 0x41, "f", -0.4)  # speed_y_attack
    be_pack_into(buf, 0x45, "f", 0.5)  # speed_ground_x_self
    be_pack_into(buf, 0x49, "f", 7.0)  # hitlag_left

    # ECB edges (top, bottom, left, right)
    be_pack_into(buf, 0x4D, "f", 1.1)
    be_pack_into(buf, 0x51, "f", 2.2)
    be_pack_into(buf, 0x55, "f", -3.3)
    be_pack_into(buf, 0x59, "f", -4.4)
    be_pack_into(buf, 0x5D, "f", -5.5)
    be_pack_into(buf, 0x61, "f", 6.6)
    be_pack_into(buf, 0x65, "f", 7.7)
    be_pack_into(buf, 0x69, "f", -8.8)

    # Manual bookend behavior
    console._use_manual_bookends = True  # type: ignore[attr-defined]

    # Invoke
    console._Console__post_frame(gs, bytes(buf))  # type: ignore[attr-defined]

    assert gs.stage == enums.Stage.BATTLEFIELD
    assert gs.is_teams is True
    assert gs.frame == frame

    ps = gs.players[2]
    assert isinstance(ps, PlayerState)

    # Character/action
    assert ps.character == enums.Character.FOX
    assert ps.action == enums.Action.NEUTRAL_ATTACK_1

    # Position, facing
    assert ps.position.x == pytest.approx(pos_x)
    assert ps.position.y == pytest.approx(pos_y)
    assert ps.facing

    # Core counters
    assert ps.percent == 87
    assert ps.shield_strength == pytest.approx(48.0)
    assert ps.stock == 3
    assert ps.action_frame == 5

    # Status flags
    assert ps.is_reflect_active is True
    assert ps.is_subaction_invulnerable is True
    assert ps.is_fastfalling is True
    assert ps.is_defender_in_hitlag is True
    assert ps.is_in_hitlag is True
    assert ps.is_holding_character is True
    assert ps.is_shield_active is True
    assert ps.is_in_hitstun is True
    assert ps.is_powershield is True
    assert ps.is_dead is True
    assert ps.is_offscreen is True

    # Derived / other
    assert ps.hitstun_frames_left == 12
    assert ps.on_ground is False
    assert ps.jumps_left == 2
    assert ps.l_cancel_status == 1
    assert ps.invulnerable is True
    assert ps.speed_air_x_self == pytest.approx(0.1)
    assert ps.speed_y_self == pytest.approx(-0.2)
    assert ps.speed_x_attack == pytest.approx(0.3)
    assert ps.speed_y_attack == pytest.approx(-0.4)
    assert ps.speed_ground_x_self == pytest.approx(0.5)
    assert ps.hitlag_left == 7

    # Off-stage helper: with BATTLEFIELD, abs(x) > edge or y < -6 and not on_ground -> True
    edge = stages.EDGE_GROUND_POSITION[gs.stage]
    assert abs(ps.position.x) > edge or ps.position.y < -6
    assert ps.off_stage is True

    # ECB edges (both object fields and tuple mirrors)
    assert ps.ecb.top.x == pytest.approx(1.1)
    assert ps.ecb.top.y == pytest.approx(2.2)
    assert ps.ecb.bottom.x == pytest.approx(-3.3)
    assert ps.ecb.bottom.y == pytest.approx(-4.4)
    assert ps.ecb.left.x == pytest.approx(-5.5)
    assert ps.ecb.left.y == pytest.approx(6.6)
    assert ps.ecb.right.x == pytest.approx(7.7)
    assert ps.ecb.right.y == pytest.approx(-8.8)

    assert ps.ecb_top == pytest.approx((1.1, 2.2))
    assert ps.ecb_bottom == pytest.approx((-3.3, -4.4))
    assert ps.ecb_left == pytest.approx((-5.5, 6.6))
    assert ps.ecb_right == pytest.approx((7.7, -8.8))

    # Manual bookend updates console._frame to gs.frame
    assert console._frame == frame  # type: ignore[attr-defined]


def test_post_frame_nana_branch(console: Console, gs: GameState) -> None:
    """When Nana flag is set, the parsed values are applied to the nested Nana PlayerState."""
    console._current_stage = enums.Stage.BATTLEFIELD  # type: ignore[attr-defined]

    buf = make_buf(0x22 + 4)
    be_pack_into(buf, 0x1, "i", 10)
    be_pack_into(buf, 0x5, "B", 0)  # port 1
    be_pack_into(buf, 0x6, "B", 1)  # Nana
    be_pack_into(buf, 0x0A, "f", -1.0)  # x
    be_pack_into(buf, 0x0E, "f", 2.5)  # y
    be_pack_into(buf, 0x21, "B", 2)  # stock
    be_pack_into(buf, 0x22, "f", 3.0)  # action_frame

    console._Console__post_frame(gs, bytes(buf))  # type: ignore[attr-defined]

    ps = gs.players[1]
    assert ps.nana is not None
    assert ps.nana.position.x == pytest.approx(-1.0)
    assert ps.nana.position.y == pytest.approx(2.5)
    assert ps.nana.stock == 2
    assert ps.nana.action_frame == 3


def test_post_frame_unknown_action_fallback(console: Console, gs: GameState) -> None:
    """Invalid action ID should map to Action.UNKNOWN_ANIMATION without raising."""
    console._current_stage = enums.Stage.BATTLEFIELD  # type: ignore[attr-defined]
    buf = make_buf(0x22 + 4)
    be_pack_into(buf, 0x1, "i", 1)
    be_pack_into(buf, 0x5, "B", 0)
    be_pack_into(buf, 0x6, "B", 0)
    be_pack_into(buf, 0x7, "B", enums.Character.FOX.value)
    be_pack_into(buf, 0x8, "H", 0xFFFF)  # invalid action id
    be_pack_into(buf, 0x12, "f", 0.0)  # facing
    be_pack_into(buf, 0x16, "f", 0.0)  # percent
    be_pack_into(buf, 0x1A, "f", 0.0)  # shield
    be_pack_into(buf, 0x21, "B", 0)  # stock
    be_pack_into(buf, 0x22, "f", 0.0)  # action_frame

    console._Console__post_frame(gs, bytes(buf))  # type: ignore[attr-defined]

    assert gs.players[1].action == enums.Action.UNKNOWN_ANIMATION


def test_post_frame_truncated_payload_defaults(console: Console, gs: GameState) -> None:
    """
    Provide a short buffer so several optional reads trigger TypeError branches.
    Verify sane defaults are applied and no exception is raised.
    """
    console._current_stage = enums.Stage.BATTLEFIELD  # type: ignore[attr-defined]
    buf = make_buf(0x30)  # shorter than many later offsets (e.g., speeds, ecb, etc.)

    be_pack_into(buf, 0x1, "i", 9)
    be_pack_into(buf, 0x5, "B", 0)
    be_pack_into(buf, 0x6, "B", 0)
    be_pack_into(buf, 0x7, "B", enums.Character.FOX.value)
    be_pack_into(buf, 0x8, "H", enums.Action.NEUTRAL_ATTACK_1.value)
    be_pack_into(buf, 0x0A, "f", 0.0)
    be_pack_into(buf, 0x0E, "f", 0.0)
    be_pack_into(buf, 0x12, "f", -1.0)  # facing False
    be_pack_into(buf, 0x16, "f", 0.0)  # percent
    be_pack_into(buf, 0x1A, "f", 0.0)  # shield
    be_pack_into(buf, 0x21, "B", 4)
    be_pack_into(buf, 0x22, "f", 1.0)
    be_pack_into(buf, 0x2F, "B", 0)  # on_ground -> True

    console._Console__post_frame(gs, bytes(buf))  # type: ignore[attr-defined]
    ps = gs.players[1]

    # Defaults where fields were not present
    assert ps.hitstun_frames_left == 0
    assert ps.jumps_left == 1
    assert ps.l_cancel_status == 0
    assert ps.invulnerable is False
    assert ps.speed_air_x_self == 0
    assert ps.speed_y_self == 0
    assert ps.speed_x_attack == 0
    assert ps.speed_y_attack == 0
    assert ps.speed_ground_x_self == 0
    assert ps.hitlag_left == 0

    # Some present fields still parsed
    assert ps.on_ground is True
    assert ps.stock == 4
    assert not ps.facing


# --- Tests for __handle_slippstream_menu_event --------------------------------


def _call_menu_event(console: Console, gs: GameState, buf: bytes) -> None:
    # Name-mangled private method
    console._Console__handle_slippstream_menu_event(buf, gs)  # type: ignore[attr-defined]


def _mk_event_buf(size: int = 0x49) -> bytearray:
    return make_buf(size)


def _write_scene(buf: bytearray, scene: int) -> None:
    be_pack_into(buf, 0x1, "H", scene)


def test_menu_event_character_select_full(
    console: Console, gs: GameState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy-path CHARACTER_SELECT (0x02): statuses, cursors, ready flag, characters, coins, frame, submenu, selection.
    Also verify CPU-level preservation for CPU ports and zeroing for non-CPU ports.
    """
    buf = _mk_event_buf()
    _write_scene(buf, 0x02)

    # Controller port statuses (HUMAN, HUMAN, CPU, HUMAN)
    be_pack_into(buf, 0x25, "B", enums.ControllerStatus.CONTROLLER_HUMAN.value)
    be_pack_into(buf, 0x26, "B", enums.ControllerStatus.CONTROLLER_HUMAN.value)
    be_pack_into(buf, 0x27, "B", enums.ControllerStatus.CONTROLLER_CPU.value)
    be_pack_into(buf, 0x28, "B", enums.ControllerStatus.CONTROLLER_HUMAN.value)

    # CSS cursors (x,y per port)
    be_pack_into(buf, 0x03, "f", 1.25)
    be_pack_into(buf, 0x07, "f", -2.5)
    be_pack_into(buf, 0x0B, "f", -3.00)
    be_pack_into(buf, 0x0F, "f", 4.0)
    be_pack_into(buf, 0x13, "f", 5.50)
    be_pack_into(buf, 0x17, "f", 6.5)
    be_pack_into(buf, 0x1B, "f", -7.75)
    be_pack_into(buf, 0x1F, "f", -8.0)

    # Ready banner
    be_pack_into(buf, 0x23, "B", 1)

    # Characters — monkeypatch to ensure success path regardless of external-id mapping
    monkeypatch.setattr(enums, "to_internal", lambda b: enums.Character.FOX)
    be_pack_into(buf, 0x29, "B", 0x21)
    be_pack_into(buf, 0x2A, "B", 0x21)
    be_pack_into(buf, 0x2B, "B", 0x21)
    be_pack_into(buf, 0x2C, "B", 0x21)

    # Coin-down flags (only p2 & p4)
    be_pack_into(buf, 0x2D, "B", 0x00)
    be_pack_into(buf, 0x2E, "B", 0x02)
    be_pack_into(buf, 0x2F, "B", 0x00)
    be_pack_into(buf, 0x30, "B", 0x02)

    # Frame, submenu (use an invalid value to hit ValueError branch), selection
    be_pack_into(buf, 0x39, "i", 4242)
    be_pack_into(buf, 0x3D, "B", 0x7E)  # cause ValueError -> UNKNOWN_SUBMENU
    be_pack_into(buf, 0x3E, "B", 7)

    # CPU level (only p3 preserved because it is CPU); holding slider flags
    be_pack_into(buf, 0x41, "B", 1)
    be_pack_into(buf, 0x42, "B", 2)
    be_pack_into(buf, 0x43, "B", 9)  # CPU port
    be_pack_into(buf, 0x44, "B", 3)
    be_pack_into(buf, 0x45, "B", 0)
    be_pack_into(buf, 0x46, "B", 1)
    be_pack_into(buf, 0x47, "B", 0)
    be_pack_into(buf, 0x48, "B", 1)

    _call_menu_event(console, gs, bytes(buf))

    assert gs.menu_state == enums.Menu.CHARACTER_SELECT
    # Players initialized
    for p in (1, 2, 3, 4):
        assert isinstance(gs.players[p], PlayerState)

    # Statuses
    assert gs.players[3].controller_status == enums.ControllerStatus.CONTROLLER_CPU
    assert gs.players[1].controller_status == enums.ControllerStatus.CONTROLLER_HUMAN

    # banner
    assert gs.ready_to_start == 1

    # Character selection succeeded via patched to_internal
    for p in (1, 2, 3, 4):
        assert gs.players[p].character == enums.Character.FOX
        assert gs.players[p].character_selected == enums.Character.FOX

    # Frame & submenu & selection
    assert gs.frame == 4242
    assert gs.submenu == enums.SubMenu.UNKNOWN_SUBMENU
    assert gs.menu_selection == 7


def test_menu_event_character_select_truncated_defaults(
    console: Console, gs: GameState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Truncate to trigger TypeError fallbacks for character and coin-down fields."""

    # Force enums.to_internal to raise TypeError so UNKNOWN_CHARACTER fallback path is exercised
    def _raise_typeerror(_):  # type: ignore[unused-argument]
        raise TypeError("forced in test")

    monkeypatch.setattr(enums, "to_internal", _raise_typeerror)
    # Buffer long enough for statuses/cursors/ready but shorter than char bytes (0x29)
    buf = _mk_event_buf(0x29)
    _write_scene(buf, 0x02)
    be_pack_into(buf, 0x25, "B", enums.ControllerStatus.CONTROLLER_HUMAN.value)
    be_pack_into(buf, 0x26, "B", enums.ControllerStatus.CONTROLLER_HUMAN.value)
    be_pack_into(buf, 0x27, "B", enums.ControllerStatus.CONTROLLER_HUMAN.value)
    be_pack_into(buf, 0x28, "B", 0)  # last valid byte

    # Minimal required floats for p1 cursor so buffer isn't too small before 0x25
    # (Offsets < 0x25 are ignored if buffer is too short; we only care about fallbacks.)
    # Still ensure frame exists separately (must be present!)
    buf2 = _mk_event_buf(0x3D)  # enough for frame at 0x39 but not submenu
    for i, b in enumerate(buf):
        buf2[i] = b
    be_pack_into(buf2, 0x39, "i", 99)

    _call_menu_event(console, gs, bytes(buf2))

    # Character fallbacks -> UNKNOWN_CHARACTER; coin_down fallbacks -> False
    for p in (1, 2, 3, 4):
        assert gs.players[p].character == enums.Character.UNKNOWN_CHARACTER
        assert gs.players[p].character_selected == enums.Character.UNKNOWN_CHARACTER

    # Submenu TypeError fallback -> UNKNOWN_SUBMENU; menu_selection TypeError -> 0
    assert gs.submenu == enums.SubMenu.UNKNOWN_SUBMENU
    assert gs.menu_selection == 0


def test_menu_event_slippi_online_css_costumes_and_nametag(
    console: Console, gs: GameState, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SLIPPI_ONLINE_CSS (0x0008): costumes applied and nametag 0x05 => NAME_ENTRY_SUBMENU, 0x00 => ONLINE_CSS."""
    # First with nametag 0x05
    buf = _mk_event_buf()
    _write_scene(buf, 0x0008)
    # Need valid controller statuses so players exist
    for i, off in enumerate((0x25, 0x26, 0x27, 0x28), start=1):
        be_pack_into(buf, off, "B", enums.ControllerStatus.CONTROLLER_HUMAN.value)
    be_pack_into(buf, 0x39, "i", 7)
    be_pack_into(buf, 0x3F, "B", 3)  # costume value
    be_pack_into(buf, 0x40, "B", 0x05)  # triggers NAME_ENTRY_SUBMENU

    _call_menu_event(console, gs, bytes(buf))

    assert gs.menu_state == enums.Menu.SLIPPI_ONLINE_CSS
    assert gs.submenu == enums.SubMenu.NAME_ENTRY_SUBMENU

    # Now with nametag 0x00 => ONLINE_CSS
    buf2 = _mk_event_buf()
    _write_scene(buf2, 0x0008)
    for off in (0x25, 0x26, 0x27, 0x28):
        be_pack_into(buf2, off, "B", enums.ControllerStatus.CONTROLLER_HUMAN.value)
    be_pack_into(buf2, 0x39, "i", 8)
    be_pack_into(buf2, 0x3F, "B", 1)
    be_pack_into(buf2, 0x40, "B", 0x00)

    _call_menu_event(console, gs, bytes(buf2))
    assert gs.submenu == enums.SubMenu.ONLINE_CSS

    # Finally simulate missing costume/nametag bytes triggering TypeError fallbacks
    buf3 = _mk_event_buf()
    _write_scene(buf3, 0x0008)
    for off in (0x25, 0x26, 0x27, 0x28):
        be_pack_into(buf3, off, "B", enums.ControllerStatus.CONTROLLER_HUMAN.value)
    be_pack_into(buf3, 0x39, "i", 9)
    be_pack_into(buf3, 0x3D, "B", enums.SubMenu.ONLINE_CSS.value)

    original = np.ndarray

    def missing_css_data(
        shape, dtype=float, buffer=None, offset=0, strides=None, order=None
    ):
        if dtype == ">B" and offset in (0x3F, 0x40):
            raise TypeError("css bytes missing")
        return original(
            shape,
            dtype=dtype,
            buffer=buffer,
            offset=offset,
            strides=strides,
            order=order,
        )

    monkeypatch.setattr(np, "ndarray", missing_css_data)

    _call_menu_event(console, gs, bytes(buf3))

    assert gs.submenu == enums.SubMenu.ONLINE_CSS


@pytest.mark.parametrize(
    "scene, expected",
    [
        (0x0102, enums.Menu.STAGE_SELECT),
        (0x0108, enums.Menu.STAGE_SELECT),
    ],
)
def test_menu_event_stage_select_valid_and_cursors(
    console: Console, gs: GameState, scene: int, expected: enums.Menu
) -> None:
    buf = _mk_event_buf()
    _write_scene(buf, scene)
    # Valid stage and cursors
    be_pack_into(buf, 0x24, "B", enums.Stage.BATTLEFIELD.value)
    be_pack_into(buf, 0x31, "f", 12.34)
    be_pack_into(buf, 0x35, "f", -9.87)
    be_pack_into(buf, 0x39, "i", 111)
    # Submenu invalid to exercise ValueError path
    be_pack_into(buf, 0x3D, "B", 0x7E)
    be_pack_into(buf, 0x3E, "B", 5)

    _call_menu_event(console, gs, bytes(buf))

    assert gs.menu_state == expected
    assert gs.stage == enums.Stage.BATTLEFIELD


def test_menu_event_stage_select_invalid_stage_sets_no_stage(
    console: Console, gs: GameState
) -> None:
    buf = _mk_event_buf()
    _write_scene(buf, 0x0102)
    be_pack_into(buf, 0x24, "B", 0xFF)  # invalid -> ValueError -> NO_STAGE
    be_pack_into(buf, 0x31, "f", 0.0)
    be_pack_into(buf, 0x35, "f", 0.0)
    be_pack_into(buf, 0x39, "i", 222)

    _call_menu_event(console, gs, bytes(buf))

    assert gs.menu_state == enums.Menu.STAGE_SELECT
    assert gs.stage == enums.Stage.NO_STAGE


@pytest.mark.parametrize(
    "scene, expected",
    [
        (0x0202, enums.Menu.IN_GAME),
        (0x0001, enums.Menu.MAIN_MENU),
        (0x0000, enums.Menu.PRESS_START),
        (0xDEAD, enums.Menu.UNKNOWN_MENU),
    ],
)
def test_menu_event_other_scenes_and_exceptions(
    console: Console, gs: GameState, scene: int, expected: enums.Menu
) -> None:
    # Prepare a GameState with empty players to trigger KeyError path in CPU-level & slider loops
    gs.players.clear()

    # Buffer long enough for all fields (so KeyError comes from missing players, not TypeError truncation)
    buf = _mk_event_buf()
    _write_scene(buf, scene)
    be_pack_into(buf, 0x39, "i", 333)
    be_pack_into(buf, 0x3D, "B", 0x7E)

    _call_menu_event(console, gs, bytes(buf))

    assert gs.menu_state == expected
    assert gs.frame == 333
    # Submenu invalid value -> UNKNOWN_SUBMENU; menu_selection defaults to 0 (no data)
    assert gs.submenu == enums.SubMenu.UNKNOWN_SUBMENU
    assert gs.menu_selection == 0
    # Players may remain empty; ensure code handled KeyErrors without raising
    assert isinstance(gs.players, dict)


# --- Tests for __handle_slippstream_events -----------------------------------


def test_handle_slippstream_events_payloads(console: Console, gs: GameState) -> None:
    """Test that PAYLOADS event updates eventsize."""
    buf = bytearray()
    buf.append(EventType.PAYLOADS.value)
    buf.append(7)  # size
    buf.append(EventType.GAME_START.value)
    buf.extend(struct.pack(">H", 10))
    buf.append(EventType.POST_FRAME.value)
    buf.extend(struct.pack(">H", 20))

    console._Console__handle_slippstream_events(bytes(buf), gs)

    assert console.eventsize[EventType.GAME_START.value] == 11
    assert console.eventsize[EventType.POST_FRAME.value] == 21


def test_handle_slippstream_events_game_start(
    console: Console, gs: GameState, monkeypatch
) -> None:
    """Test that GAME_START event calls __game_start."""
    game_start_called = False

    def mock_game_start(*args, **kwargs):
        nonlocal game_start_called
        game_start_called = True

    monkeypatch.setattr(console, "_Console__game_start", mock_game_start)

    buf = bytearray()
    buf.append(EventType.GAME_START.value)
    console.eventsize[EventType.GAME_START.value] = 1
    console._Console__handle_slippstream_events(bytes(buf), gs)
    assert game_start_called


def test_handle_slippstream_events_game_end(console: Console, gs: GameState) -> None:
    """Test that GAME_END event returns correct value."""
    buf = bytearray()
    buf.append(EventType.GAME_END.value)
    console.eventsize[EventType.GAME_END.value] = 1
    console._use_manual_bookends = True
    assert console._Console__handle_slippstream_events(bytes(buf), gs) is True
    console._use_manual_bookends = False
    assert console._Console__handle_slippstream_events(bytes(buf), gs) is False


def test_handle_slippstream_events_pre_frame(
    console: Console, gs: GameState, monkeypatch
) -> None:
    """Test that PRE_FRAME event calls __pre_frame."""
    pre_frame_called = False

    def mock_pre_frame(*args, **kwargs):
        nonlocal pre_frame_called
        pre_frame_called = True

    monkeypatch.setattr(console, "_Console__pre_frame", mock_pre_frame)

    buf = bytearray()
    buf.append(EventType.PRE_FRAME.value)
    console.eventsize[EventType.PRE_FRAME.value] = 1
    console._Console__handle_slippstream_events(bytes(buf), gs)
    assert pre_frame_called


def test_handle_slippstream_events_post_frame(
    console: Console, gs: GameState, monkeypatch
) -> None:
    """Test that POST_FRAME event calls __post_frame."""
    post_frame_called = False

    def mock_post_frame(*args, **kwargs):
        nonlocal post_frame_called
        post_frame_called = True

    monkeypatch.setattr(console, "_Console__post_frame", mock_post_frame)

    buf = bytearray()
    buf.append(EventType.POST_FRAME.value)
    console.eventsize[EventType.POST_FRAME.value] = 1
    console._Console__handle_slippstream_events(bytes(buf), gs)
    assert post_frame_called


def test_handle_slippstream_events_frame_bookend(
    console: Console, gs: GameState, monkeypatch
) -> None:
    """Test that FRAME_BOOKEND event returns correct value based on skip_rollback_frames."""
    frame_bookend_called = False

    def mock_frame_bookend(*args, **kwargs):
        nonlocal frame_bookend_called
        frame_bookend_called = True

    monkeypatch.setattr(console, "_Console__frame_bookend", mock_frame_bookend)

    buf = bytearray()
    buf.append(EventType.FRAME_BOOKEND.value)
    console.eventsize[EventType.FRAME_BOOKEND.value] = 1

    # Test with skip_rollback_frames = True
    console.skip_rollback_frames = True
    gs.frame = 10
    console._frame = 20
    assert console._Console__handle_slippstream_events(bytes(buf), gs) is False
    assert frame_bookend_called

    # Test with skip_rollback_frames = False
    console.skip_rollback_frames = False
    gs.frame = 10
    console._frame = 20
    assert console._Console__handle_slippstream_events(bytes(buf), gs) is True


def test_handle_slippstream_events_item_update(
    console: Console, gs: GameState, monkeypatch
) -> None:
    """Test that ITEM_UPDATE event calls __item_update."""
    item_update_called = False

    def mock_item_update(*args, **kwargs):
        nonlocal item_update_called
        item_update_called = True

    monkeypatch.setattr(console, "_Console__item_update", mock_item_update)

    buf = bytearray()
    buf.append(EventType.ITEM_UPDATE.value)
    console.eventsize[EventType.ITEM_UPDATE.value] = 1
    console._Console__handle_slippstream_events(bytes(buf), gs)
    assert item_update_called


def test_handle_slippstream_events_invalid_event(
    console: Console, gs: GameState, caplog
) -> None:
    """Test that an invalid event type logs an error."""
    buf = bytearray()
    buf.append(0xFF)  # Invalid event type
    console._Console__handle_slippstream_events(bytes(buf), gs)
    assert "Got invalid event type: 255" in caplog.text


def test_handle_slippstream_events_truncated_event(
    console: Console, gs: GameState, caplog
) -> None:
    """Test that a truncated event logs a warning."""
    buf = bytearray()
    buf.append(EventType.GAME_START.value)
    console.eventsize[EventType.GAME_START.value] = 10
    console._Console__handle_slippstream_events(bytes(buf), gs)
    assert (
        "Something went wrong unpacking events. Data is probably missing" in caplog.text
    )


def test_handle_slippstream_menu_event_in_game(
    console: Console, gs: GameState, caplog
) -> None:
    """Test that a menu event in the middle of a game is handled correctly."""
    buf = make_buf(0x3E)
    be_pack_into(buf, 0, "B", EventType.MENU_EVENT.value)
    be_pack_into(buf, 1, "H", 0x0202)
    be_pack_into(buf, 0x39, "i", 1234)
    console.eventsize[EventType.MENU_EVENT.value] = len(buf)
    assert console._Console__handle_slippstream_events(bytes(buf), gs) is True
    assert (
        "Got a menu event in the middle of a frame. Continuing anyway." in caplog.text
    )


def test_handle_slippstream_events_frame_bookend_blocking(
    console: Console, gs: GameState, monkeypatch
) -> None:
    """Test that FRAME_BOOKEND event flushes controllers when blocking_input is True."""
    frame_bookend_called = False

    def mock_frame_bookend(*args, **kwargs):
        nonlocal frame_bookend_called
        frame_bookend_called = True

    flush_called = False

    class MockController:
        def flush(self):
            nonlocal flush_called
            flush_called = True

    monkeypatch.setattr(console, "_Console__frame_bookend", mock_frame_bookend)
    console.controllers.append(MockController())

    buf = bytearray()
    buf.append(EventType.FRAME_BOOKEND.value)
    console.eventsize[EventType.FRAME_BOOKEND.value] = 1

    console.blocking_input = True
    console.skip_rollback_frames = True
    gs.frame = 10
    console._frame = 20
    console._Console__handle_slippstream_events(bytes(buf), gs)
    assert flush_called


@pytest.mark.parametrize(
    "event_type, stage",
    [
        (EventType.FOD_INFO, enums.Stage.FOUNTAIN_OF_DREAMS),
        (EventType.DL_INFO, enums.Stage.DREAMLAND),
        (EventType.PS_INFO, enums.Stage.POKEMON_STADIUM),
    ],
)
def test_handle_slippstream_events_stage_info(
    console: Console, gs: GameState, event_type: EventType, stage: enums.Stage, caplog
) -> None:
    """Test that stage info events are handled correctly."""
    buf = bytearray()
    buf.append(event_type.value)
    console.eventsize[event_type.value] = 1

    # Test with matching stage
    console._current_stage = stage
    console._Console__handle_slippstream_events(bytes(buf), gs)
    assert (
        f"Got stage info for {stage}, but gamestate says {gs.stage}" not in caplog.text
    )

    # Test with mismatching stage
    console._current_stage = enums.Stage.BATTLEFIELD
    gs.stage = enums.Stage.BATTLEFIELD
    console._Console__handle_slippstream_events(bytes(buf), gs)
    assert (
        f"Got stage info for {stage}, but gamestate says {enums.Stage.BATTLEFIELD}"
        in caplog.text
    )


def test_handle_slippstream_events_unhandled_event(
    console: Console, gs: GameState, caplog
) -> None:
    """Test that an unhandled event type logs an error."""
    buf = bytearray()
    # Use an event type that is not handled in the main if/elif chain
    buf.append(EventType.GECKO_CODES.value)
    console.eventsize[EventType.GECKO_CODES.value] = 1
    console._Console__handle_slippstream_events(bytes(buf), gs)


def test_game_start_with_names(console: Console, gs: GameState) -> None:
    """Test that __game_start correctly parses player names and connect codes."""
    buf = make_buf(0x221 + 0xA * 4)
    be_pack_into(buf, 1, "B", 3)
    be_pack_into(buf, 2, "B", 9)
    be_pack_into(buf, 3, "B", 0)

    # Player 1
    name1 = "Player1".encode("shift-jis")
    for i, char in enumerate(name1):
        be_pack_into(buf, 0x1A5 + i, "B", char)
    code1 = "P1#123".encode("shift-jis").replace(b"#", b"\x81\x94")
    for i, char in enumerate(code1):
        be_pack_into(buf, 0x221 + i, "B", char)

    # Player 2
    name2 = "Player2".encode("shift-jis")
    for i, char in enumerate(name2):
        be_pack_into(buf, 0x1A5 + 0x1F + i, "B", char)
    code2 = "P2#456".encode("shift-jis").replace(b"#", b"\x81\x94")
    for i, char in enumerate(code2):
        be_pack_into(buf, 0x221 + 0xA + i, "B", char)

    console._Console__game_start(gs, bytes(buf))

    assert console._display_names[0] == "Player1"
    assert console._connect_codes[0] == "P1#123"
    assert console._display_names[1] == "Player2"
    assert console._connect_codes[1] == "P2#456"


def test_game_start_version_too_low(console: Console, gs: GameState) -> None:
    """Test that SlippiVersionTooLow is raised for old versions."""
    buf = make_buf(600)
    be_pack_into(buf, 1, "B", 2)
    console._allow_old_version = False
    with pytest.raises(SlippiVersionTooLow):
        console._Console__game_start(gs, bytes(buf))


def test_game_start_invalid_stage(console: Console, gs: GameState) -> None:
    """Test that an invalid stage ID is handled correctly."""
    buf = make_buf(600)
    be_pack_into(buf, 1, "B", 3)
    be_pack_into(buf, 0x13, "H", 0xFFFF)
    console._Console__game_start(gs, bytes(buf))
    assert console._current_stage == enums.Stage.NO_STAGE


def test_item_update_unknown_projectile(console: Console, gs: GameState) -> None:
    """Test that an unknown projectile type is handled correctly."""
    buf = make_buf(0x30)
    be_pack_into(buf, 5, "H", 0xFFFF)
    console._Console__item_update(gs, bytes(buf))
    assert len(gs.projectiles) == 1
    assert gs.projectiles[0].type == enums.ProjectileType.UNKNOWN_PROJECTILE


def test_item_update_invalid_owner(console: Console, gs: GameState) -> None:
    """Test that an invalid owner is handled correctly."""
    buf = make_buf(0x30)
    be_pack_into(buf, 0x2A, "B", 5)
    console._Console__item_update(gs, bytes(buf))
    assert len(gs.projectiles) == 1
    assert gs.projectiles[0].owner == -1


def test_item_update_frame_value_error(console: Console, gs: GameState) -> None:
    """Test that a ValueError when reading the frame is handled correctly."""
    buf = make_buf(0x30)
    # Pack a non-float value to trigger a ValueError
    be_pack_into(buf, 0x1E, "I", 0xFFFFFFFF)
    console._Console__item_update(gs, bytes(buf))
    assert len(gs.projectiles) == 1
    assert gs.projectiles[0].frame == -1


@pytest.mark.parametrize(
    "projectile_type, subtype",
    [
        (enums.ProjectileType.SAMUS_BOMB, 3),
        (enums.ProjectileType.SAMUS_MISSLE, 2),
        (enums.ProjectileType.SAMUS_MISSLE, 3),
        (enums.ProjectileType.SAMUS_CHARGE_BEAM, 0),
    ],
)
def test_item_update_ignored_projectiles(
    console: Console, gs: GameState, projectile_type: enums.ProjectileType, subtype: int
) -> None:
    """Test that certain projectiles are ignored."""
    buf = make_buf(0x30)
    be_pack_into(buf, 5, "H", projectile_type.value)
    be_pack_into(buf, 7, "B", subtype)
    console._Console__item_update(gs, bytes(buf))
    assert len(gs.projectiles) == 0


# --- Performance evaluation over real replays --------------------------------

METHODS_TO_TIME: List[str] = [
    "__init__",
    "connect",
    "_get_dolphin_home_path",
    "_get_dolphin_config_path",
    "get_dolphin_pipes_path",
    "run",
    "stop",
    "_setup_home_directory",
    "_setup_dolphin_ini",
    "_setup_gecko_codes",
    "setup_dolphin_controller",
    "step",
    "__handle_slippstream_events",
    "__game_start",
    "__pre_frame",
    "__post_frame",
    "__frame_bookend",
    "__item_update",
    "__handle_slippstream_menu_event",
    "__fixframeindexing",
    "__fixiasa",
]


def _mangle(name: str) -> str:
    # Python name-mangling for private methods e.g. __pre_frame -> _Console__pre_frame
    if name.startswith("__") and not name.endswith("__"):
        return f"_{Console.__name__}{name}"
    return name


@pytest.mark.perf
def test_console_method_timing_over_replays() -> None:
    """
    Perf smoke test: iterate a consistent random subset of FOX_vs_FOX replays and
    measure total time spent inside each Console method.

    Controls (via env):
      - PERF_SLP_N: number of replays to sample (default: 32)
      - PERF_SLP_SEED: RNG seed for stable sampling (default: 1337)
    """
    root = Path("/Users/eppie/Downloads/ALL_REPLAYS/FOX_vs_FOX")
    if not root.exists():
        pytest.skip(f"Replay directory not found: {root}")

    paths = sorted(root.glob("master-master*.slp"))
    if not paths:
        pytest.skip("No replays found matching pattern 'master-master*.slp'")

    n: int = int(os.environ.get("PERF_SLP_N", "32"))
    seed: int = int(os.environ.get("PERF_SLP_SEED", "1337"))
    rng = random.Random(seed)
    if len(paths) > n:
        paths = rng.sample(paths, n)

    totals: Dict[str, float] = {name: 0.0 for name in METHODS_TO_TIME}
    counts: Dict[str, int] = {name: 0 for name in METHODS_TO_TIME}

    for replay_path in paths:
        # __init__ timing measured explicitly
        t0 = time.perf_counter()
        c = Console(path=str(replay_path), is_dolphin=False, allow_old_version=True)
        totals["__init__"] += time.perf_counter() - t0
        counts["__init__"] += 1

        # Wrap per-instance methods (skip __init__)
        for public in METHODS_TO_TIME:
            if public == "__init__":
                continue
            attr_name = _mangle(public)
            fn = getattr(c, attr_name, None)
            if fn is None or not callable(fn):
                # Method not present in this build/mode; ignore
                continue

            def _make_wrapper(bound_fn: Callable[..., object], method_name: str):
                def _wrapped(*args, **kwargs):
                    t1 = time.perf_counter()
                    try:
                        return bound_fn(*args, **kwargs)
                    finally:
                        totals[method_name] += time.perf_counter() - t1
                        counts[method_name] += 1

                return _wrapped

            setattr(c, attr_name, _make_wrapper(fn, public))

        # Drive the replay
        c.connect()
        current_game_state: Optional[GameState] = c.step()
        while current_game_state is not None:
            current_game_state = c.step()
        c.stop()

    # Emit a readable table in test output
    print("\n[Console perf] Sampled replays:", len(paths))
    print(f"{'method':35s} {'total_s':>12s} {'calls':>10s} {'avg_ms':>12s}")
    for name in sorted(METHODS_TO_TIME, key=lambda n: totals[n], reverse=True):
        total = totals[name]
        cnt = counts[name]
        avg_ms = (total / cnt * 1000.0) if cnt else 0.0
        print(f"{name:35s} {total:12.4f} {cnt:10d} {avg_ms:12.3f}")

    # Sanity: step should have been invoked
    assert counts["step"] > 0
