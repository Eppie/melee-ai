from __future__ import annotations

import argparse
import gzip
import shutil
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Iterable

import numpy as np
import peppi_py
import pyarrow as pa
from peppi_py import Game
from peppi_py.game import EndMethod, PlayerType

REPLAYS_DIR = Path("/Users/eppie/Downloads/ALL_REPLAYS/OUT_EXTRACT")
GOOD_DIR = Path("/Users/eppie/Downloads/ALL_REPLAYS/replays_sorted")
FAILED_DIR = Path("/Users/eppie/Downloads/ALL_REPLAYS/replays_failed")
FAILED_DIR.mkdir(exist_ok=True)
ALLOWED_STAGES = {0x03, 0x08, 0x02, 0x1F, 0x20, 0x1C}
MIN_FRAME_COUNT = 3600
MAX_FRAME_COUNT = 21_600
MAIN_STICK_DEADZONE = 0.02
MIN_ACTIVE_STICK_RATIO = 0.05

FilterFailure = tuple[str, str]


class Character(Enum):
    """A Melee character External ID."""

    FALCON = 0x00  # Captain Falcon
    DK = 0x01  # Donkey Kong
    FOX = 0x02
    GNW = 0x03  # Mr. Game & Watch
    KIRBY = 0x04
    BOWSER = 0x05
    LINK = 0x06
    LUIGI = 0x07
    MARIO = 0x08
    MARTH = 0x09
    MEWTWO = 0x0A
    NESS = 0x0B
    PEACH = 0x0C
    PIKA = 0x0D  # Pikachu
    ICS = 0x0E  # Ice Climbers
    PUFF = 0x0F  # Jigglypuff
    SAMUS = 0x10
    YOSHI = 0x11
    ZELDA = 0x12
    SHEIK = 0x13
    FALCO = 0x14
    YLINK = 0x15  # Young Link
    DOC = 0x16  # Dr. Mario
    ROY = 0x17
    PICHU = 0x18
    GANON = 0x19  # Ganondorf
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


def sanity_reason(game: Game) -> FilterFailure | None:
    """
    Return (reason, message) for the first sanity‑check failure, or None if the game passes.
    """
    start = game.start
    if start is None:
        return "missing_start", "Missing game start metadata"

    end = game.end
    if end is not None and end.method == EndMethod.NO_CONTEST:
        return "no_contest", "Ended in no contest"
    if start.is_raining_bombs:
        return "raining_bombs", "Raining bombs variant enabled"
    if start.is_teams:
        return "teams", "Teams mode enabled"
    if start.item_spawn_frequency != -1:
        return "item_spawn", f"Item spawn frequency {start.item_spawn_frequency} != -1"
    if start.damage_ratio != 1.0:
        return "damage_ratio", f"Damage ratio {start.damage_ratio} != 1.0"
    if start.self_destruct_score != -1:
        return (
            "self_destruct_score",
            f"Self-destruct score {start.self_destruct_score} != -1",
        )
    if start.timer != 60 * 8:
        return "timer", f"Timer {start.timer} != 8 minutes"
    if start.is_pal:
        return "pal", "PAL region flag set"
    if len(start.players) != 2:
        return "player_count", f"{len(start.players)} players present instead of 2"
    if any(p.type != PlayerType.HUMAN for p in start.players):
        return "cpu_players", "Non-human player detected"
    if any(p.stocks != 4 for p in start.players):
        return "stock_count", "Starting stocks differ from 4"
    if start.stage not in ALLOWED_STAGES:
        return "illegal_stage", f"Stage {start.stage:#04x} not in allowed list"
    return None


def _active_stick_ratio(joystick) -> float:
    x_arr = getattr(joystick, "x", None)
    y_arr = getattr(joystick, "y", None)
    if x_arr is None or y_arr is None or len(x_arr) == 0 or len(y_arr) == 0:
        return 0.0
    x = x_arr.to_numpy(zero_copy_only=False)
    y = y_arr.to_numpy(zero_copy_only=False)
    if x.size == 0 or y.size == 0:
        return 0.0
    finite_mask = np.isfinite(x) & np.isfinite(y)
    if not finite_mask.any():
        return 0.0
    active = np.logical_or(
        np.abs(x) > MAIN_STICK_DEADZONE, np.abs(y) > MAIN_STICK_DEADZONE
    )
    active = active & finite_mask
    total = finite_mask.sum()
    if total == 0:
        return 0.0
    return float(active.sum()) / float(total)


def quality_reason(game: Game) -> FilterFailure | None:
    frames = game.frames
    if frames is None or not frames.ports:
        return "no_frame_data", "Missing per-frame controller data"

    try:
        total_frames = len(frames.ports[0].leader.pre.buttons)
    except Exception:
        return "frame_length_unreadable", "Unable to determine frame count from inputs"

    if total_frames < MIN_FRAME_COUNT:
        return "short_match", f"{total_frames} frames < required {MIN_FRAME_COUNT}"
    if total_frames > MAX_FRAME_COUNT:
        return "long_match", f"{total_frames} frames > allowed {MAX_FRAME_COUNT}"

    for idx, port in enumerate(frames.ports, start=1):
        leader = port.leader
        joystick = leader.pre.joystick

        active_ratio = _active_stick_ratio(joystick)
        if active_ratio < MIN_ACTIVE_STICK_RATIO:
            return (
                "low_input_variance",
                f"Player {idx} active main-stick frames {active_ratio * 100:.2f}% < {MIN_ACTIVE_STICK_RATIO * 100:.0f}%",
            )

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
            code, message = reason
            print(f"Skipping {path.name}: {message}")
            _move_to_failed(path, code)
            return None

        quality = quality_reason(game)
        if quality is not None:
            code, message = quality
            print(f"Skipping {path.name}: {message}")
            _move_to_failed(path, code)
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


def _iter_zip_members(
    zip_path: Path, filename_filter: str | None = None
) -> Iterable[str]:
    """Yield file names from the zip that should be extracted."""
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if filename_filter and filename_filter not in name:
                continue
            if not (name.endswith(".slp") or name.endswith(".gz")):
                continue
            yield name


def _extract_member(zip_path: Path, member_name: str, dest_dir: Path) -> Path:
    """
    Extract a single member from the archive to dest_dir.
    .gz files are decompressed to their base name to feed into process_file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_name = Path(member_name).name
    dest_path = dest_dir / target_name
    if dest_path.suffix == ".gz":
        dest_path = dest_path.with_suffix("")

    if dest_path.exists():
        dest_path = dest_path.with_name(
            f"{dest_path.stem}_{int(time.time() * 1000)}{dest_path.suffix}"
        )

    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as src:
            if target_name.endswith(".gz"):
                with gzip.open(src, "rb") as gz_in, open(dest_path, "wb") as out:
                    shutil.copyfileobj(gz_in, out)
            else:
                with open(dest_path, "wb") as out:
                    shutil.copyfileobj(src, out)

    return dest_path


def _process_zip_archive(
    zip_path: Path,
    filename_filter: str | None,
    process_workers: int,
    extract_workers: int,
) -> None:
    """
    Extract matching members from the archive in parallel and process them
    as soon as each extraction completes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_dir = Path(tmpdir)
        with ThreadPoolExecutor(
            max_workers=extract_workers
        ) as extractor, ProcessPoolExecutor(max_workers=process_workers) as processor:
            extract_futures = [
                extractor.submit(_extract_member, zip_path, member, temp_dir)
                for member in _iter_zip_members(zip_path, filename_filter)
            ]

            process_futures = []
            try:
                for fut in as_completed(extract_futures):
                    extracted_path = fut.result()
                    process_futures.append(
                        processor.submit(process_file, extracted_path)
                    )
            except BaseException as exc:
                print(f"Extraction failed: {exc!r}")
                raise

            for fut in as_completed(process_futures):
                fut.result()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter and sort Slippi replays, optionally from a zip archive."
    )
    parser.add_argument(
        "--zip-file",
        type=Path,
        help="Path to a .zip archive containing .slp or .gz replay files.",
    )
    parser.add_argument(
        "--filename-filter",
        type=str,
        default=None,
        help="Substring to filter which files from the zip are extracted/processed.",
    )
    parser.add_argument(
        "--replays-dir",
        type=Path,
        default=REPLAYS_DIR,
        help="Directory to scan for .slp files when not using --zip-file.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of worker processes for replay validation.",
    )
    parser.add_argument(
        "--extract-workers",
        type=int,
        default=4,
        help="Number of threads to extract files from zip archives.",
    )
    args = parser.parse_args()

    if args.zip_file:
        _process_zip_archive(
            zip_path=args.zip_file,
            filename_filter=args.filename_filter,
            process_workers=args.workers,
            extract_workers=args.extract_workers,
        )
        return

    files = sorted(args.replays_dir.rglob("*.slp"))

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        try:
            for _ in pool.map(process_file, files, chunksize=20):
                pass
        except BaseException as exc:
            print(f"Uncaught exception from worker threads: {exc!r}")


if __name__ == "__main__":
    main()
