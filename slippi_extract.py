from __future__ import annotations

from pathlib import Path
from typing import Mapping, NamedTuple, TypeVar, Union

import numpy as np
import peppi_py
import pyarrow as pa
from peppi_py.game import EndMethod, PlayerType
from peppi_py.frame import Pre, Post, Position

REPLAYS_DIR = Path("replays")  # root containing *.slp
OUT_DIR = Path("shards")
OUT_DIR.mkdir(exist_ok=True)

T = TypeVar('T')
Nest = Union[Mapping[str, 'Nest'], T]

from enum import Enum


class Stage(Enum):
    """A VS-mode stage """
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


class Buttons(NamedTuple):
    A: np.bool_
    B: np.bool_
    X: np.bool_
    Y: np.bool_
    Z: np.bool_
    L: np.bool_
    R: np.bool_
    D_UP: np.bool_


LIBMELEE_BUTTONS = {name: Button(name) for name in Buttons._fields}


class Stick(NamedTuple):
    x: np.float32
    y: np.float32


class Controller(NamedTuple):
    main_stick: Stick
    c_stick: Stick
    shoulder: np.float32
    buttons: Buttons


class Player(NamedTuple):
    percent: np.uint16
    facing: np.bool_
    x: np.float32
    y: np.float32
    action: np.uint16
    invulnerable: np.bool_
    character: np.uint8
    jumps_left: np.uint8
    shield_strength: np.float32
    on_ground: np.bool_
    controller: Controller


class Game(NamedTuple):
    p0: Player
    p1: Player
    stage: np.uint8


def array_from_nt(val: Union[tuple, np.ndarray]) -> pa.StructArray:
    if isinstance(val, tuple):
        values = [array_from_nt(v) for v in val]
        return pa.StructArray.from_arrays(values, names=val._fields)
    else:
        return val


BUTTON_MASKS = {
    Button.BUTTON_A: 0x0100,
    Button.BUTTON_B: 0x0200,
    Button.BUTTON_X: 0x0400,
    Button.BUTTON_Y: 0x0800,
    Button.BUTTON_START: 0x1000,
    Button.BUTTON_Z: 0x0010,
    Button.BUTTON_R: 0x0020,
    Button.BUTTON_L: 0x0040,
    Button.BUTTON_D_LEFT: 0x0001,
    Button.BUTTON_D_RIGHT: 0x0002,
    Button.BUTTON_D_DOWN: 0x0004,
    Button.BUTTON_D_UP: 0x0008,
}


def get_buttons(button_bits: np.ndarray) -> Buttons:
    return Buttons(**{
        name: np.asarray(
            np.bitwise_and(button_bits, BUTTON_MASKS[button]),
            dtype=bool)
        for name, button in LIBMELEE_BUTTONS.items()
    })


def get_slp(path: str):
    game = peppi_py.read_slippi(path)
    start = game.start

    if game.end.method in (EndMethod.NO_CONTEST,):
        return False
    if start.is_raining_bombs:
        return False
    if start.is_teams:
        return False
    if start.item_spawn_frequency != -1:
        return False
    if start.damage_ratio != 1.0:
        return False
    if start.self_destruct_score != -1:
        return False
    if start.timer != 480:
        return False
    if start.is_pal:
        return False

    stage = start.stage
    start_bitfield = start.bitfield
    players = start.players
    last_frame = game.metadata['lastFrame']
    end_method = game.end.method
    if len(players) != 2:
        return False
    for player in players:
        if player.type != PlayerType.HUMAN:
            return False
        if player.stocks != 4:
            return False
    p1 = players[0]
    p2 = players[1]
    p1_port = p1.port
    p2_port = p2.port
    p1_character = p1.character
    p2_character = p2.character
    p1_bitfield = p1.bitfield
    p2_bitfield = p2.bitfield
    print(p1_port, p1_character, p1_bitfield)
    print(p2_port, p2_character, p2_bitfield)

    print(f'stage: {stage}')
    print(f'start_bitfield: {start_bitfield}')
    print(f'last_frame: {last_frame}')
    print(f"end: {end_method}")
    frames = game.frames
    port_data = frames.ports
    p1_pre = port_data[0].leader.pre
    p1_post = port_data[0].leader.post
    p2_pre = port_data[1].leader.pre
    p2_post = port_data[1].leader.post
    print(f'p1_pre: {p1_pre}')
    print(f'p1_post: {p1_post}')
    print(f'p2_pre: {p2_pre}')
    print(f'p2_post: {p2_post}')
    exit(0)


def main() -> None:
    files = sorted(REPLAYS_DIR.rglob("*.slp"))
    for file in files[:100]:
        try:
            get_slp(str(file))
        except OSError:
            pass


if __name__ == "__main__":
    main()
