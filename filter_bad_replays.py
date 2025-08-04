from __future__ import annotations

import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from pathlib import Path

import peppi_py
import pyarrow as pa
from peppi_py import Game
from peppi_py.game import EndMethod, PlayerType

REPLAYS_DIR = Path("/Users/eppie/Downloads/replays")
GOOD_DIR = Path("/Users/eppie/Downloads/replays_sorted")
FAILED_DIR = Path("/Users/eppie/Downloads/replays_failed")
FAILED_DIR.mkdir(exist_ok=True)
ALLOWED_STAGES = {0x03, 0x08, 0x02, 0x1F, 0x20, 0x1C}


class Character(Enum):
    """A Melee character External ID."""
    FALCON = 0x00            # Captain Falcon
    DK = 0x01                # Donkey Kong
    FOX = 0x02
    GNW = 0x03               # Mr. Game & Watch
    KIRBY = 0x04
    BOWSER = 0x05
    LINK = 0x06
    LUIGI = 0x07
    MARIO = 0x08
    MARTH = 0x09
    MEWTWO = 0x0A
    NESS = 0x0B
    PEACH = 0x0C
    PIKA = 0x0D              # Pikachu
    ICS = 0x0E               # Ice Climbers
    PUFF = 0x0F              # Jigglypuff
    SAMUS = 0x10
    YOSHI = 0x11
    ZELDA = 0x12
    SHEIK = 0x13
    FALCO = 0x14
    YLINK = 0x15             # Young Link
    DOC = 0x16               # Dr. Mario
    ROY = 0x17
    PICHU = 0x18
    GANON = 0x19             # Ganondorf
    MASTER_HAND = 0x1A
    WIREFRAME_MALE = 0x1B
    WIREFRAME_FEMALE = 0x1C
    GIGA_BOWSER = 0x1D
    CRAZY_HAND = 0x1E
    SANDBAG = 0x1F
    POPO = 0x20
    USER_SELECT_NONE = 0x21


def _move_to_failed(src: Path, reason: str) -> None:
    dest_dir = FAILED_DIR / reason
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / src.name
    if dest_path.exists():  # avoid clobbering duplicate names
        dest_path = dest_dir / f"{src.stem}_{int(time.time() * 1000)}{src.suffix}"
    shutil.move(str(src), dest_path)


def sanity_reason(game: Game) -> str | None:
    """
    Return a string describing the first sanity‑check failure, or None if the game passes.
    The string doubles as the sub‑directory name to move the .slp into.
    """
    start = game.start

    if game.end.method == EndMethod.NO_CONTEST:
        return "no_contest"
    if start.is_raining_bombs:
        return "raining_bombs"
    if start.is_teams:
        return "teams"
    if start.item_spawn_frequency != -1:
        return "item_spawn"
    if start.damage_ratio != 1.0:
        return "damage_ratio"
    if start.self_destruct_score != -1:
        return "self_destruct_score"
    if start.timer != 60 * 8:
        return "timer"
    if start.is_pal:
        return "pal"
    if len(start.players) != 2:
        return "player_count"
    if any(p.type != PlayerType.HUMAN for p in start.players):
        return "cpu_players"
    if any(p.stocks != 4 for p in start.players):
        return "stock_count"
    if start.stage not in ALLOWED_STAGES:
        return "illegal_stage"
    return None


def process_file(path: Path) -> pa.Table | None:
    """
    Parse one .slp, move it to an appropriate 'failed' folder if it flunks,
    and return a flattened pyarrow.Table (or None).  *All* exceptions are
    swallowed and cause the file to be shunted into replays_failed/corrupt.
    """
    try:
        game = peppi_py.read_slippi(str(path))

        reason = sanity_reason(game)
        if reason is not None:
            _move_to_failed(path, reason)
            return None

        # print(game)
        p1_char = game.start.players[0].character
        p2_char = game.start.players[1].character

        if p1_char > p2_char:
            p1_char, p2_char = p2_char, p1_char

        def _char_name(cid: int) -> str:
            """Map a character id to the name from the Character enum; fallback if unknown."""
            try:
                return Character(cid).name
            except ValueError:
                return f"UNKNOWN_{cid}"

        # Build subfolder name using character names (canonicalized order so P1/P2 swap goes to same folder)
        subfolder = f"{_char_name(p1_char)}_vs_{_char_name(p2_char)}"
        dest_dir = GOOD_DIR / subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)

        dest_path = dest_dir / path.name
        if dest_path.exists():  # avoid clobbering duplicate names
            dest_path = dest_dir / f"{path.stem}_{int(time.time() * 1000)}{path.suffix}"
        shutil.move(str(path), dest_path)
        return None

    except BaseException as exc:  # catch *everything*, even non‑Exception errors
        print(f"❌ {path.name}: {exc!r} – moving to 'corrupt'")
        _move_to_failed(path, "corrupt")
        return None


def main() -> None:
    files = sorted(REPLAYS_DIR.rglob("*.slp"))

    with ProcessPoolExecutor(max_workers=16) as pool:
        try:
            for _ in pool.map(process_file, files, chunksize=20):
                pass
        except BaseException as exc:
            print(f"Uncaught exception from worker threads: {exc!r}")


if __name__ == "__main__":
    main()
