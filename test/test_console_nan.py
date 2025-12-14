"""Test for console.py NaN handling in float-to-int conversions."""

from __future__ import annotations

import math
import struct

import pytest

from libmelee.melee.console import Console
from libmelee.melee.gamestate import GameState


def create_post_frame_event_with_nan(nan_offset: int) -> bytes:
    """Create a minimal valid post-frame event with NaN at specified offset.

    Args:
        nan_offset: Byte offset where to inject NaN float (e.g., 0x2B for hitstun_frames_left)

    Returns:
        Bytes representing a post-frame event
    """
    # Create event bytes large enough to contain all required fields
    # Need at least 0x50 bytes to cover all fields accessed in __post_frame
    event = bytearray(0x50)

    # Frame number at 0x1 (4 bytes, big-endian int)
    struct.pack_into(">I", event, 0x1, 100)

    # Controller port at 0x5 (1 byte) - port 0 (becomes port 1 after +1)
    event[0x5] = 0

    # Nana flag at 0x6 (1 byte) - 0 means not Nana
    event[0x6] = 0

    # Character at 0x7 (1 byte) - Fox = 1
    event[0x7] = 1

    # Action at 0x8 (2 bytes, big-endian short)
    struct.pack_into(">H", event, 0x8, 0)

    # Position at 0x0A (2 floats, 8 bytes total)
    struct.pack_into(">2f", event, 0x0A, 0.0, 0.0)

    # Facing direction at 0x12 (float)
    struct.pack_into(">f", event, 0x12, 1.0)

    # Percent and shield strength at 0x16 (2 floats)
    struct.pack_into(">2f", event, 0x16, 50.0, 60.0)

    # Stock at 0x21 (1 byte)
    event[0x21] = 4

    # Action frame at 0x22 (float) - valid value
    struct.pack_into(">f", event, 0x22, 5.0)

    # Status bytes at 0x26-0x2A (5 bytes)
    event[0x26:0x2B] = b"\x00\x00\x00\x00\x00"

    # Controller inputs at 0x19 (4 floats for stick values)
    struct.pack_into(">4f", event, 0x19, 0.5, 0.5, 0.5, 0.5)

    # Trigger at 0x29 (float)
    struct.pack_into(">f", event, 0x29, 0.0)

    # Inject NaN at the specified offset
    struct.pack_into(">f", event, nan_offset, math.nan)

    return bytes(event)


def test_hitstun_frames_left_nan_handling():
    """Test that NaN in hitstun_frames_left field doesn't crash."""
    console = Console()
    gamestate = GameState()

    # Create event with NaN at offset 0x2B (hitstun_frames_left)
    event_bytes = create_post_frame_event_with_nan(0x2B)

    # This should not raise ValueError
    console._Console__post_frame(gamestate, event_bytes)

    # The field should be 0 when NaN is encountered
    assert gamestate.players[1].hitstun_frames_left == 0


def test_action_frame_nan_handling():
    """Test that NaN in action_frame field doesn't crash."""
    console = Console()
    gamestate = GameState()

    # Create event with NaN at offset 0x22 (action_frame)
    event_bytes = create_post_frame_event_with_nan(0x22)

    # This should not raise ValueError
    console._Console__post_frame(gamestate, event_bytes)

    # The field should be 0 when NaN is encountered
    assert gamestate.players[1].action_frame == 0


def test_hitlag_left_nan_handling():
    """Test that NaN in hitlag_left field doesn't crash."""
    console = Console()
    gamestate = GameState()

    # Create event with NaN at offset 0x49 (hitlag_left)
    # Need to make the event long enough (> 0x4C)
    event_bytes = create_post_frame_event_with_nan(0x49)

    # This should not raise ValueError
    console._Console__post_frame(gamestate, event_bytes)

    # The field should be 0 when NaN is encountered
    assert gamestate.players[1].hitlag_left == 0


@pytest.mark.parametrize(
    "nan_offset,field_name",
    [
        (0x2B, "hitstun_frames_left"),
        (0x22, "action_frame"),
        (0x49, "hitlag_left"),
    ],
)
def test_all_float_to_int_conversions_handle_nan(nan_offset: int, field_name: str):
    """Parameterized test for all float-to-int conversions that might encounter NaN."""
    console = Console()
    gamestate = GameState()

    event_bytes = create_post_frame_event_with_nan(nan_offset)

    # Should not raise ValueError
    console._Console__post_frame(gamestate, event_bytes)

    # All NaN values should be converted to 0
    player_state = gamestate.players[1]
    field_value = getattr(player_state, field_name)
    assert field_value == 0, f"{field_name} should be 0 when NaN is encountered"
