import functools
import json
import multiprocessing
import traceback
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Tuple, List, TypedDict
from typing import TypeVar, Union, Optional, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from tqdm import tqdm

from config import REPLAYS_DIR
from libmelee.melee import stages, enums
from libmelee.melee.console import Console
from libmelee.melee.controller import ControllerState
from libmelee.melee.enums import Button
from libmelee.melee.enums import Stage as StageEnum, Action as Action, Character
from libmelee.melee.framedata import FrameData
from libmelee.melee.gamestate import GameState, PlayerState

PA_TO_NT = {}

T = TypeVar('T')
Nest = Union[Mapping[str, 'Nest'], T]

from typing import NamedTuple
import numpy as np

_fd = FrameData()


@functools.lru_cache(maxsize=None)
def _char_cached(i: int) -> Character:
    return Character(i)


@functools.lru_cache(maxsize=None)
def _act_cached(i: int) -> Action:
    return Action(i)


@functools.lru_cache(maxsize=None)
def _iasa_cached(char: Character, act: Action) -> int:
    return _fd.iasa(char, act)


@functools.lru_cache(maxsize=None)
def _frame_count_cached(char: Character, act: Action) -> int:
    return _fd.frame_count(char, act)


@functools.lru_cache(maxsize=None)
def _last_hitbox_frame_cached(char: Character, act: Action) -> int:
    return _fd.last_hitbox_frame(char, act)


def _actionable_in_vals(char: Character, act: Action, af: int) -> np.uint8:
    iasa = _iasa_cached(char, act)
    if iasa == -1:
        return np.uint8(0)
    return np.uint8(min(max(0, iasa - af), 255))


def _cooldown_remaining_vals(char: Character, act: Action, af: int) -> np.uint8:
    if not _fd.is_attack(char, act):
        return np.uint8(0)
    last = _last_hitbox_frame_cached(char, act)
    if last == -1:
        return np.uint8(0)
    total = _frame_count_cached(char, act)
    rem = max(0, total - max(af, last))
    return np.uint8(min(rem, 255))


def _inject_computed_features_inplace(frames: list[Mapping[str, Any]]) -> None:
    fsg = {"p0": 0, "p1": 0}  # frames-since-grounded counters
    prev_stock = {"p0": None, "p1": None}

    for i, fr in enumerate(frames):
        stage = _as_stage_enum(int(fr["stage"]))
        for self_key, opp_key in (("p0", "p1"), ("p1", "p0")):
            me = fr[self_key];
            opp = fr[opp_key]

            # O(1) frames_since_grounded
            if bool(me.get("on_ground", False)):
                fsg[self_key] = 0
            else:
                fsg[self_key] = min(fsg[self_key] + 1, 65535)

            # stock_delta_event: “opponent lost a stock on this frame”
            stock_delta = False
            if prev_stock[opp_key] is not None:
                stock_delta = int(opp.get("stock", 0)) < int(prev_stock[opp_key])

            # cache Enums once per frame
            char = _char_cached(int(me["character"]))
            act = _act_cached(int(me["action"]))
            af = int(me.get("action_frame", 0))

            cf = ComputedFeatures(
                distance=_distance(me, opp),
                facing_opponent=_facing_opponent(me, opp),
                distance_to_blastzones=_distance_to_blastzones(me, stage),
                actionable_in=_actionable_in_vals(char, act, af),
                stock_delta_event=np.bool_(stock_delta),
                edgeguard_situation=_edgeguard_situation(frames, i, self_key, stage),
                corner_pressure=_corner_pressure(frames, i, self_key, stage),
                frames_since_grounded=np.uint16(fsg[self_key]),
                cooldown_remaining=_cooldown_remaining_vals(char, act, af),
                jumps_max=_jumps_max(me),
            )
            fr[self_key]["computed_features"] = nt_to_nest(cf)

        # update prev stocks after both players processed
        prev_stock["p0"] = int(fr["p0"].get("stock", 0))
        prev_stock["p1"] = int(fr["p1"].get("stock", 0))


def _as_stage_enum(stage_id: int) -> StageEnum:
    try:
        return StageEnum(int(stage_id))
    except Exception:
        # Fallback to Final Destination if unknown (shouldn't happen)
        return StageEnum.FINAL_DESTINATION


def _blast_zones(stage: StageEnum) -> Tuple[float, float, float, float]:
    """
    Returns (left, right, bottom, top) for the given stage.

    Your stages.BLASTZONES is (left, right, upper, lower), so we reorder.
    """
    left, right, upper, lower = stages.BLASTZONES[stage]
    # normalize to (left, right, bottom, top)
    return float(left), float(right), float(lower), float(upper)


def _nearest_ledge_x(stage: StageEnum, x: float) -> float:
    edge = float(stages.EDGE_GROUND_POSITION.get(stage, 90.0))
    return edge if x >= 0 else -edge


def _distance(p0: Mapping[str, Any], p1: Mapping[str, Any]) -> np.float32:
    dx = float(p1["x"]) - float(p0["x"])
    dy = float(p1["y"]) - float(p0["y"])
    return np.float32((dx * dx + dy * dy) ** 0.5)


def _facing_opponent(p: Mapping[str, Any], opp: Mapping[str, Any]) -> np.bool_:
    dx = float(opp["x"]) - float(p["x"])
    face_right = bool(p["facing"])
    return np.bool_((dx > 0 and face_right) or (dx < 0 and not face_right))


def _distance_to_blastzones(p: Mapping[str, Any], stage: StageEnum) -> np.float32:
    x = float(p["x"])
    y = float(p["y"])
    left, right, bottom, top = _blast_zones(stage)
    d_left = x - left
    d_right = right - x
    d_bottom = y - bottom
    d_top = top - y
    return np.float32(min(d_left, d_right, d_bottom, d_top))


def _actionable_in(p: Mapping[str, Any]) -> np.uint8:
    char = Character(int(p["character"]))
    act = Action(int(p["action"]))
    af = int(p.get("action_frame", 0))
    iasa = _fd.iasa(char, act)
    val = 0
    if iasa != -1:
        val = max(0, iasa - af)
    # If not an attack, we assume generic actions are cancellable at or before now.
    # Keep it conservative; clamp to uint8.
    return np.uint8(min(val, 255))


def _stock_delta_event(
        frames: List[Mapping[str, Any]],
        i: int,
        self_key: str,
) -> np.bool_:
    if i == 0:
        return np.bool_(False)
    prev = frames[i - 1][self_key]
    cur = frames[i][self_key]
    # "event" = opponent lost a stock this frame
    opp_key = "p1" if self_key == "p0" else "p0"
    prev_opp = frames[i - 1][opp_key]
    cur_opp = frames[i][opp_key]
    return np.bool_(int(cur_opp.get("stock", 0)) < int(prev_opp.get("stock", 0)))


def _edgeguard_situation(
        frames: List[Mapping[str, Any]],
        i: int,
        self_key: str,
        stage: StageEnum,
) -> np.bool_:
    me = frames[i][self_key]
    opp_key = "p1" if self_key == "p0" else "p0"
    opp = frames[i][opp_key]

    if not bool(opp.get("off_stage", False)):
        return np.bool_(False)
    if not bool(me.get("on_ground", False)):
        return np.bool_(False)

    ledge_x = _nearest_ledge_x(stage, float(me["x"]))
    near_ledge = abs(float(me["x"]) - ledge_x) < 20.0
    horiz_close = abs(float(me["x"]) - float(opp["x"])) < 35.0
    low_air = float(me["y"]) < 10.0  # near ground height on most stages
    return np.bool_(near_ledge and horiz_close and low_air)


def _corner_pressure(
        frames: List[Mapping[str, Any]],
        i: int,
        self_key: str,
        stage: StageEnum,
) -> np.float32:
    # pressure on opponent (how cornered opponent is)
    opp_key = "p1" if self_key == "p0" else "p0"
    opp = frames[i][opp_key]
    x = float(opp["x"])
    left, right, _, _ = _blast_zones(stage)
    w = right - left
    if w <= 1e-6:
        return np.float32(0.0)
    d = min(x - left, right - x)
    # map to [0,1], 1 = most cornered, 0 = center
    pressure = 1.0 - (2.0 * d / w)
    return np.float32(max(0.0, min(1.0, pressure)))


def _frames_since_grounded(
        frames: List[Mapping[str, Any]],
        i: int,
        self_key: str,
) -> np.uint16:
    # scan backwards to last on_ground==True
    last_ground = None
    for j in range(i, -1, -1):
        if bool(frames[j][self_key].get("on_ground", False)):
            last_ground = j
            break
    val = 0 if last_ground is None else (i - last_ground)
    return np.uint16(min(val, 65535))


def _cooldown_remaining(p: Mapping[str, Any]) -> np.uint8:
    char = Character(int(p["character"]))
    act = Action(int(p["action"]))
    af = int(p.get("action_frame", 0))

    if not _fd.is_attack(char, act):
        return np.uint8(0)

    last = _fd.last_hitbox_frame(char, act)
    if last == -1:
        return np.uint8(0)
    total = _fd.frame_count(char, act)
    rem = max(0, total - max(af, last))
    return np.uint8(min(rem, 255))


def _jumps_max(p: Mapping[str, Any]) -> np.uint8:
    char = Character(int(p["character"]))
    return np.uint8(min(_fd.max_jumps(char), 2))


class XY(NamedTuple):
    x: np.float32
    y: np.float32


class ECBNT(NamedTuple):
    top: XY
    bottom: XY
    left: XY
    right: XY


def to_xy(obj) -> XY:
    """Accept Position dataclass or a (x, y) tuple and normalize to XY(np.float32, np.float32)."""
    try:
        # Position dataclass
        return XY(np.float32(obj.x), np.float32(obj.y))
    except AttributeError:
        # Tuple-like
        x, y = obj
        return XY(np.float32(x), np.float32(y))


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


class ComputedFeatures(NamedTuple):
    distance: np.float32
    facing_opponent: np.bool_
    distance_to_blastzones: np.float32
    actionable_in: np.uint8
    stock_delta_event: np.bool_
    edgeguard_situation: np.bool_
    corner_pressure: np.float32
    frames_since_grounded: np.uint16
    cooldown_remaining: np.uint8
    jumps_max: np.uint8


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
    character_selected: np.uint8
    stock: np.int16
    action_frame: np.int16
    invulnerability_left: np.uint16
    hitlag_left: np.uint16
    hitstun_frames_left: np.int16
    speed_air_x_self: np.float32
    speed_y_self: np.float32
    speed_x_attack: np.float32
    speed_y_attack: np.float32
    speed_ground_x_self: np.float32
    off_stage: np.bool_
    iasa: np.uint16
    moonwalkwarning: np.bool_
    ecb_bottom: XY
    ecb_top: XY
    ecb_left: XY
    ecb_right: XY
    position: XY
    ecb: ECBNT
    is_powershield: np.bool_
    is_absorbing: np.bool_
    reflect_owner_doesnt_change: np.bool_
    is_reflect_active: np.bool_
    is_subaction_invulnerable: np.bool_
    is_fastfalling: np.bool_
    is_defender_in_hitlag: np.bool_
    is_in_hitlag: np.bool_
    is_holding_character: np.bool_
    is_shield_active: np.bool_
    is_in_hitstun: np.bool_
    is_touching_shield: np.bool_
    is_cloaked: np.bool_
    is_follower: np.bool_
    is_inactive: np.bool_
    is_dead: np.bool_
    is_offscreen: np.bool_
    computed_features: ComputedFeatures


class Game(NamedTuple):
    p0: Player
    p1: Player
    stage: np.uint8


@functools.lru_cache
def nt_to_pa(nt: type) -> pa.StructType:
    """Convert and register a NamedTuple (or numpy) type."""
    # NEW: handle built-in strings cleanly
    if nt is str:
        return pa.string()
    if nt is bytes:
        return pa.binary()

    if not issubclass(nt, tuple):
        return pa.from_numpy_dtype(nt)

    struct_type = pa.struct([
        (name, nt_to_pa(nt.__annotations__[name]))
        for name in nt._fields
    ])
    PA_TO_NT[struct_type] = nt
    return struct_type


GAME_TYPE = nt_to_pa(Game)


def nt_to_nest(val: Union[tuple, T]) -> Nest[T]:
    """ Converts a NamedTuple to a Nest."""
    if isinstance(val, tuple) and hasattr(val, '_fields'):
        return {k: nt_to_nest(v) for k, v in zip(val._fields, val)}
    return val


def get_stick(stick: tuple[float]) -> Stick:
    return Stick(*map(np.float32, stick))


def get_buttons(button: dict[Button, bool]) -> Buttons:
    return Buttons(**{
        name: button[lm_button]
        for name, lm_button in LIBMELEE_BUTTONS.items()
    })


def get_controller(cs: ControllerState) -> Controller:
    return Controller(
        main_stick=get_stick(cs.main_stick),
        c_stick=get_stick(cs.c_stick),
        shoulder=cs.l_shoulder,
        buttons=get_buttons(cs.button),
    )


def get_computed_features(
        frames: List[Mapping[str, Any]],
        i: int,
        self_key: str,
        stage_id: int,
) -> ComputedFeatures:
    """
    Compute all features for a single player at frame i.
    Calls per-feature helpers to keep this unit-testable.
    """
    stage = _as_stage_enum(stage_id)
    me = frames[i][self_key]
    opp = frames[i]["p1" if self_key == "p0" else "p0"]

    return ComputedFeatures(
        distance=_distance(me, opp),
        facing_opponent=_facing_opponent(me, opp),
        distance_to_blastzones=_distance_to_blastzones(me, stage),
        actionable_in=_actionable_in(me),
        stock_delta_event=_stock_delta_event(frames, i, self_key),
        edgeguard_situation=_edgeguard_situation(frames, i, self_key, stage),
        corner_pressure=_corner_pressure(frames, i, self_key, stage),
        frames_since_grounded=_frames_since_grounded(frames, i, self_key),
        cooldown_remaining=_cooldown_remaining(me),
        jumps_max=_jumps_max(me),
    )


# Optional: a small, typed zero-initializer used when we first build Players
def _empty_computed_features() -> ComputedFeatures:
    return ComputedFeatures(
        distance=np.float32(0.0),
        facing_opponent=np.bool_(False),
        distance_to_blastzones=np.float32(0.0),
        actionable_in=np.uint8(0),
        stock_delta_event=np.bool_(False),
        edgeguard_situation=np.bool_(False),
        corner_pressure=np.float32(0.0),
        frames_since_grounded=np.uint16(0),
        cooldown_remaining=np.uint8(0),
        jumps_max=np.uint8(0),
    )


def get_player(player: PlayerState) -> Player:
    if player.action == Action.UNKNOWN_ANIMATION:
        raise Exception('UNKNOWN_ANIMATION')

    # Controller (our compact Arrow-safe view)
    ctrl = get_controller(player.controller_state)

    # ECB aggregate
    ecb_struct = ECBNT(
        top=to_xy(player.ecb.top),
        bottom=to_xy(player.ecb.bottom),
        left=to_xy(player.ecb.left),
        right=to_xy(player.ecb.right),
    )

    return Player(
        percent=np.uint16(player.percent),
        facing=np.bool_(player.facing),
        x=np.float32(player.position.x),
        y=np.float32(player.position.y),
        action=np.uint16(player.action.value),
        invulnerable=np.bool_(player.invulnerable),
        character=np.uint8(player.character.value),
        jumps_left=np.uint8(player.jumps_left),
        shield_strength=np.float32(player.shield_strength),
        on_ground=np.bool_(player.on_ground),
        controller=ctrl,
        character_selected=np.uint8(player.character_selected.value),
        stock=np.int16(player.stock),
        action_frame=np.int16(player.action_frame),
        invulnerability_left=np.uint16(player.invulnerability_left),
        hitlag_left=np.uint16(player.hitlag_left),
        hitstun_frames_left=np.int16(player.hitstun_frames_left),
        speed_air_x_self=np.float32(player.speed_air_x_self),
        speed_y_self=np.float32(player.speed_y_self),
        speed_x_attack=np.float32(player.speed_x_attack),
        speed_y_attack=np.float32(player.speed_y_attack),
        speed_ground_x_self=np.float32(player.speed_ground_x_self),
        off_stage=np.bool_(player.off_stage),
        iasa=np.uint16(player.iasa),
        moonwalkwarning=np.bool_(player.moonwalkwarning),
        ecb_bottom=to_xy(player.ecb_bottom),
        ecb_top=to_xy(player.ecb_top),
        ecb_left=to_xy(player.ecb_left),
        ecb_right=to_xy(player.ecb_right),
        position=to_xy(player.position),
        ecb=ecb_struct,
        is_powershield=np.bool_(player.is_powershield),
        is_absorbing=np.bool_(player.is_absorbing),
        reflect_owner_doesnt_change=np.bool_(player.reflect_owner_doesnt_change),
        is_reflect_active=np.bool_(player.is_reflect_active),
        is_subaction_invulnerable=np.bool_(player.is_subaction_invulnerable),
        is_fastfalling=np.bool_(player.is_fastfalling),
        is_defender_in_hitlag=np.bool_(player.is_defender_in_hitlag),
        is_in_hitlag=np.bool_(player.is_in_hitlag),
        is_holding_character=np.bool_(player.is_holding_character),
        is_shield_active=np.bool_(player.is_shield_active),
        is_in_hitstun=np.bool_(player.is_in_hitstun),
        is_touching_shield=np.bool_(player.is_touching_shield),
        is_cloaked=np.bool_(player.is_cloaked),
        is_follower=np.bool_(player.is_follower),
        is_inactive=np.bool_(player.is_inactive),
        is_dead=np.bool_(player.is_dead),
        is_offscreen=np.bool_(player.is_offscreen),
        computed_features=_empty_computed_features(),
    )


def get_game(
        game: GameState,
        ports: Optional[Sequence[int]] = None,
) -> Game:
    ports = ports or sorted(game.players)
    assert len(ports) == 2
    players = {
        f'p{i}': get_player(game.players[p])
        for i, p in enumerate(ports)}
    return Game(
        stage=game.stage.value,
        **players,
    )


def get_slp(path: str) -> pa.StructArray:
    """Processes a slippi replay file."""
    console = Console(is_dolphin=False,
                      allow_old_version=True,
                      path=path)
    console.connect()

    gamestate = console.step()
    if gamestate.stage != enums.Stage.FINAL_DESTINATION:
        return None
    ports = sorted(gamestate.players)
    if len(ports) != 2:
        raise Exception(f'Not a 2-player game.')

    frames = []

    while gamestate:
        if sorted(gamestate.player) != ports:
            raise Exception(f'Ports changed on frame {len(frames)}')
        frames.append(get_game_nest(gamestate, ports))
        gamestate = console.step()

    _inject_computed_features_inplace(frames)
    # for i in range(len(frames)):
    #     stage_id = int(frames[i]["stage"])
    #     for key in ("p0", "p1"):
    #         cf = get_computed_features(frames, i, key, stage_id)
    #         Inject back into the nested dict so Arrow sees it
    # frames[i][key]["computed_features"] = nt_to_nest(cf)

    return pa.array(frames, type=GAME_TYPE)


def get_player_nest(player: PlayerState) -> Mapping[str, Any]:
    if player.action == Action.UNKNOWN_ANIMATION:
        raise Exception('UNKNOWN_ANIMATION')

    def xy(obj) -> tuple[float, float]:
        try:
            return (float(obj.x), float(obj.y))
        except AttributeError:
            x, y = obj;
            return (float(x), float(y))

    ctrl = {
        "main_stick": {"x": float(player.controller_state.main_stick[0]),
                       "y": float(player.controller_state.main_stick[1])},
        "c_stick": {"x": float(player.controller_state.c_stick[0]),
                    "y": float(player.controller_state.c_stick[1])},
        "shoulder": float(player.controller_state.l_shoulder),
        "buttons": {name: player.controller_state.button[LIBMELEE_BUTTONS[name]]
                    for name in Buttons._fields},
    }

    ecbnt = {
        "top": {"x": xy(player.ecb.top)[0], "y": xy(player.ecb.top)[1]},
        "bottom": {"x": xy(player.ecb.bottom)[0], "y": xy(player.ecb.bottom)[1]},
        "left": {"x": xy(player.ecb.left)[0], "y": xy(player.ecb.left)[1]},
        "right": {"x": xy(player.ecb.right)[0], "y": xy(player.ecb.right)[1]},
    }

    # build dict with plain Python types (Arrow handles them fine)
    return {
        "percent": int(player.percent),
        "facing": bool(player.facing),
        "x": float(player.position.x),
        "y": float(player.position.y),
        "action": int(player.action.value),
        "invulnerable": bool(player.invulnerable),
        "character": int(player.character.value),
        "jumps_left": int(player.jumps_left),
        "shield_strength": float(player.shield_strength),
        "on_ground": bool(player.on_ground),
        "controller": ctrl,
        "character_selected": int(player.character_selected.value),
        "stock": int(player.stock),
        "action_frame": int(player.action_frame),
        "invulnerability_left": int(player.invulnerability_left),
        "hitlag_left": int(player.hitlag_left),
        "hitstun_frames_left": int(player.hitstun_frames_left),
        "speed_air_x_self": float(player.speed_air_x_self),
        "speed_y_self": float(player.speed_y_self),
        "speed_x_attack": float(player.speed_x_attack),
        "speed_y_attack": float(player.speed_y_attack),
        "speed_ground_x_self": float(player.speed_ground_x_self),
        "off_stage": bool(player.off_stage),
        "iasa": int(player.iasa),
        "moonwalkwarning": bool(player.moonwalkwarning),
        "ecb_bottom": {"x": xy(player.ecb_bottom)[0], "y": xy(player.ecb_bottom)[1]},
        "ecb_top": {"x": xy(player.ecb_top)[0], "y": xy(player.ecb_top)[1]},
        "ecb_left": {"x": xy(player.ecb_left)[0], "y": xy(player.ecb_left)[1]},
        "ecb_right": {"x": xy(player.ecb_right)[0], "y": xy(player.ecb_right)[1]},
        "position": {"x": xy(player.position)[0], "y": xy(player.position)[1]},
        "ecb": ecbnt,
        # filled later
        "computed_features": nt_to_nest(_empty_computed_features()),
        "is_powershield": bool(player.is_powershield),
        "is_absorbing": bool(player.is_absorbing),
        "reflect_owner_doesnt_change": bool(player.reflect_owner_doesnt_change),
        "is_reflect_active": bool(player.is_reflect_active),
        "is_subaction_invulnerable": bool(player.is_subaction_invulnerable),
        "is_fastfalling": bool(player.is_fastfalling),
        "is_defender_in_hitlag": bool(player.is_defender_in_hitlag),
        "is_in_hitlag": bool(player.is_in_hitlag),
        "is_holding_character": bool(player.is_holding_character),
        "is_shield_active": bool(player.is_shield_active),
        "is_in_hitstun": bool(player.is_in_hitstun),
        "is_touching_shield": bool(player.is_touching_shield),
        "is_cloaked": bool(player.is_cloaked),
        "is_follower": bool(player.is_follower),
        "is_inactive": bool(player.is_inactive),
        "is_dead": bool(player.is_dead),
        "is_offscreen": bool(player.is_offscreen),
    }


def get_game_nest(game: GameState, ports: Optional[Sequence[int]] = None) -> Mapping[str, Any]:
    ports = ports or sorted(game.players)
    assert len(ports) == 2
    return {
        "stage": int(game.stage.value),
        "p0": get_player_nest(game.players[ports[0]]),
        "p1": get_player_nest(game.players[ports[1]]),
    }


def structarray_to_flat_df(arr: pa.StructArray, sep: str = "_") -> pd.DataFrame:
    """
    Convert a pyarrow StructArray (with arbitrarily‑deep nested structs)
    into a flat pandas DataFrame.

    Parameters
    ----------
    arr: pa.StructArray
        The Arrow array returned by ``get_slp``.
    sep: str
        Separator used for column names when flattening (default: "_").

    Returns
    -------
    pd.DataFrame
        One row per replay frame, one column per leaf field.
        All leaf values are kept as their native Python type (numbers,
        bools, strings, …).  Non‑numeric columns can be dropped later.
    """
    # ``to_pylist`` gives us a list of ordinary Python dicts that mirror the
    # Arrow schema, including the nested structure.
    pylist = arr.to_pylist()  # → List[Dict]

    # ``json_normalize`` recursively expands the dicts into a flat table.
    # The `sep` argument decides how deep‑field names are joined.
    flat_df = pd.json_normalize(pylist, sep=sep)

    return flat_df


def numeric_matrix_from_df(df: pd.DataFrame,
                           dtype: np.dtype = np.float32) -> np.ndarray:
    """
    Pull out all columns that are numeric (int, uint, float, bool) and
    return them as a 2‑D NumPy array of shape (frames, features).

    Booleans are cast to 0/1, integers are safely up‑cast to the requested
    ``dtype`` (by default float32 – the usual choice for PyTorch).

    Parameters
    ----------
    df: pd.DataFrame
        The flat DataFrame produced by ``structarray_to_flat_df``.
    dtype: np.dtype
        Desired NumPy data type for the final matrix.

    Returns
    -------
    np.ndarray
        ``shape == (len(df), n_features)`` – ready for ``torch.from_numpy``.
    """
    # Keep only true numeric dtypes (bool is included)
    numeric_df = df.select_dtypes(include=[np.number])

    # Cast bool → 0/1 and everything else → the requested float dtype
    mat = numeric_df.to_numpy(dtype=dtype, copy=False)

    return mat


def structarray_to_numeric_matrix_arrow(
        arr: pa.StructArray,
        dtype: np.dtype = np.float32,
        sep: str = "_",
) -> np.ndarray:
    # Put the StructArray into a single-column table named "root"
    table = pa.Table.from_arrays([arr], names=["root"])

    # Recursively flatten all struct columns
    while any(pa.types.is_struct(t) for t in table.schema.types):
        table = table.flatten()

    # Nice column names (no "root.", use your sep)
    table = table.rename_columns(
        [name.replace("root.", "").replace(".", sep) for name in table.column_names]
    )

    # Keep only numeric/boolean columns (no Python objects)
    keep_idx = [
        i for i, t in enumerate(table.schema.types)
        if pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t)
    ]
    if not keep_idx:
        return np.empty((len(table), 0), dtype=dtype)

    num = table.select(keep_idx).combine_chunks()

    # Booleans → int8, others unchanged; then materialize as NumPy without Python lists
    cols = []
    for col in num.columns:
        t = col.type
        if pa.types.is_boolean(t):
            col = pc.cast(col, pa.int8())
        cols.append(col.to_numpy(zero_copy_only=False))

    mat = np.column_stack(cols).astype(dtype, copy=False)
    return mat


class ColumnInfo(TypedDict):
    name: str  # flattened column name (with your `sep`)
    arrow_type: str  # Arrow logical type, e.g. "int32", "bool", "float64"
    numpy_dtype: str  # numpy dtype BEFORE final matrix cast, e.g. "int8"/"float32"


def structarray_to_numeric_matrix_arrow_with_cols(
        arr: pa.StructArray,
        dtype: np.dtype = np.float32,
        sep: str = "_",
) -> Tuple[np.ndarray, List[ColumnInfo]]:
    """Arrow-only flatten to a numeric matrix + ordered column metadata."""
    # Start as a single-column table
    table = pa.Table.from_arrays([arr], names=["root"])

    # Recursively flatten all struct columns
    while any(pa.types.is_struct(t) for t in table.schema.types):
        table = table.flatten()

    # Clean column names: drop "root." and join nested fields with `sep`
    col_names = [name.replace("root.", "").replace(".", sep) for name in table.column_names]
    table = table.rename_columns(col_names)

    # Keep only numeric/boolean columns
    keep_idx: List[int] = []
    for i, t in enumerate(table.schema.types):
        if pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t):
            keep_idx.append(i)

    if not keep_idx:
        return np.empty((len(table), 0), dtype=dtype), []

    num = table.select(keep_idx).combine_chunks()
    kept_names = [col_names[i] for i in keep_idx]
    kept_types = [num.schema.types[i] for i in range(len(num.schema.types))]

    cols_np: List[np.ndarray] = []
    meta: List[ColumnInfo] = []

    for name, t, col in zip(kept_names, kept_types, num.columns):
        if pa.types.is_boolean(t):
            # Cast bools to int8 to avoid object dtype in NumPy
            col_cast = pc.cast(col, pa.int8())
            np_col = col_cast.to_numpy(zero_copy_only=False)
            meta.append({"name": name, "arrow_type": "bool", "numpy_dtype": "int8"})
        else:
            np_col = col.to_numpy(zero_copy_only=False)
            meta.append({"name": name, "arrow_type": str(t), "numpy_dtype": np_col.dtype.name})
        cols_np.append(np_col)

    mat = np.column_stack(cols_np).astype(dtype, copy=False)
    return mat, meta


def save_slp_result_as_npy(
        result: pa.StructArray,
        out_path: Path,
        dtype: np.dtype = np.float32,
        sep: str = ".",
) -> None:
    """
    Save the numeric matrix to .npy and a sidecar JSON with ordered columns.

    Writes:
      - <out_path>.npy            — the (frames × features) matrix
      - <out_path>.columns.json   — [{'name', 'arrow_type', 'numpy_dtype'}, ...]
    """
    out_path = Path(out_path)
    mat, columns = structarray_to_numeric_matrix_arrow_with_cols(result, dtype=dtype, sep=sep)

    # 1) matrix
    np.save(out_path, mat)

    # 2) sidecar metadata
    meta = {
        "version": 1,
        "shape": [int(mat.shape[0]), int(mat.shape[1])],
        "matrix_numpy_dtype": np.dtype(dtype).name,
        "sep": sep,
        "columns": columns,  # ordered to align with matrix columns
    }
    sidecar = out_path.with_suffix(".columns.json")
    with sidecar.open("w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved {mat.shape[0]} frames × {mat.shape[1]} features → {out_path}")
    print(f"Wrote column metadata → {sidecar}")


def _process_one(slp_path: Path, out_dir: Path, idx: int) -> tuple[int, str]:
    """
    Called *inside* a worker process.

    Parameters
    ----------
    slp_path : Path
        Full path to the *.slp* file.
    out_dir : Path
        Directory where the ``{idx}.npy`` shard will be written.
    idx : int
        Numerical index that will become the shard name.

    Returns
    -------
    (idx, status) where *status* is either ``"OK"`` or an error message.
    """
    try:
        # -----------------------------------------------------------------
        # 1️⃣  Load the replay and turn it into a pyarrow.StructArray
        # -----------------------------------------------------------------
        result = get_slp(str(slp_path))

        # -----------------------------------------------------------------
        # 2️⃣  Convert and write the NumPy shard
        # -----------------------------------------------------------------
        out_file = out_dir / f"{idx}.npy"
        save_slp_result_as_npy(result, out_file)

        return idx, "OK"
    except Exception as exc:  # pragma: no‑cover
        # Capture the traceback so the main process can print a nice report.
        tb = traceback.format_exc()
        return idx, f"FAIL: {exc!r}\n{tb}"


def convert_all_replays_parallel(
        replays_dir: Path,
        out_dir: Path,
        max_workers: int | None = None,
        show_progress: bool = True,
) -> None:
    """
    Parallel version of the original `for i in range(len(slp_files)):` loop.

    Parameters
    ----------
    replays_dir : Path
        Root directory that contains the ``*.slp`` files (recursively searched).
    out_dir : Path
        Where the ``{i}.npy`` shards will be stored.  The folder is created
        automatically if it does not exist.
    max_workers : int | None
        Number of processes to spawn.  ``None`` → use `os.cpu_count()`.
    show_progress : bool
        Whether to display a tqdm progress bar.
    """
    # -----------------------------------------------------------------
    # 0️⃣  Prepare input & output lists
    # -----------------------------------------------------------------
    slp_files = sorted(replays_dir.rglob("*.slp"), reverse=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not slp_files:
        print(f"[⚠] No .slp files found under {replays_dir!s}")
        return

    # -----------------------------------------------------------------
    # 1️⃣  Spin up a process pool
    # -----------------------------------------------------------------
    # On macOS the default start method is "spawn", which is safe.
    # On Linux we keep the default "fork" (fast) unless the user changes it.
    ctx = multiprocessing.get_context()  # works on all platforms
    with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
    ) as executor:

        # Submit *all* jobs up‑front – the executor will queue them internally.
        futures = {
            executor.submit(_process_one, slp_path, out_dir, i): i
            for i, slp_path in enumerate(slp_files)
        }

        # -----------------------------------------------------------------
        # 2️⃣  Collect results (with optional tqdm)
        # -----------------------------------------------------------------
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(futures),
                desc="Processing replays",
                unit="replay",
                dynamic_ncols=True,
                colour="green",
            )

        # Keep a small list of failures for a final report.
        failures = []

        for future in iterator:
            idx = futures[future]  # the index we passed when submitting
            try:
                idx_ret, status = future.result()
                # sanity‑check: idx_ret should equal idx
                if status != "OK":
                    failures.append((idx_ret, status))
            except Exception as exc:  # pragma: no‑cover
                # This would be a *serializer* failure – very rare.
                failures.append((idx, f"UNHANDLED EXC: {exc!r}\n{traceback.format_exc()}"))

        # -----------------------------------------------------------------
        # 3️⃣  Summary
        # -----------------------------------------------------------------
        if failures:
            print("\n❌ Some replays failed to process:")
            for i, msg in failures:
                print(f"  • [{i}] {msg.splitlines()[0]}")
            print(f"\n🟡 Processed {len(slp_files) - len(failures)}/{len(slp_files)} replays successfully.")
        else:
            print("\n✅ All replays processed without errors!")


def main() -> None:
    OUT_DIR = Path("shards")
    convert_all_replays_parallel(REPLAYS_DIR, OUT_DIR, max_workers=None)
    # slp_files = sorted(REPLAYS_DIR.rglob("*.slp"), reverse=True)[:10000]
    # slp_file = slp_files[0]
    # slp_file = '/Users/eppie/Downloads/replays_sorted/FOX_vs_FOX/1 - Cody Schwab (Fox), Azel (Fox) - Battlefield_1752517533904.slp'
    # for i in range(len(slp_files)):
    #     slp_file = slp_files[i]
    #     print(f'Processing {slp_file} ({i + 1}/{len(slp_files)})...)')
    #     result = get_slp(slp_file)
    #     out_file = Path(f"shards/{i}.npy")
    #     if result is not None:
    #         save_slp_result_as_npy(result, out_file)

    # -----------------------------------------------------------------
    # 3️⃣  Quick sanity‑check: load it back with torch
    # -----------------------------------------------------------------
    # tensor = torch.from_numpy(np.load(out_file))  # shape: (frames, features)
    # print("Tensor shape:", tensor.shape, "dtype:", tensor.dtype)
    # Assuming you already have: result: pyarrow.StructArray
    # rv = ReplayView.from_result(result).add_decoded_columns()

    # print(sanity_checks(result).pretty())

    # Peek at the first few frames as a table
    # rv.df.head()
    # Buttons summary columns
    # add_buttons_pressed_column(rv.df, "p0")
    # add_buttons_pressed_column(rv.df, "p1")
    # if "p0.action" in rv.df and "p1.action" in rv.df:
    #     add_action_names(rv.df, "p0")
    #     add_action_names(rv.df, "p1")

    # Human-readable snapshot of a single frame (compare with your viewer)
    # for i in range(1500):
    #     print(describe_frame(result, i))

    # Quick plots (open two charts for positions, then percent, then sticks)
    # plot_positions(rv.df)
    # plot_percent(rv.df)
    # plot_sticks(rv.df, "p0")
    # plot_sticks(rv.df, "p1")


if __name__ == "__main__":
    main()
