#!/usr/bin/env python3
"""Run a headless CPU vs CPU Melee match and log average FPS."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from libmelee.melee.console import Console
from libmelee.melee.controller import Controller
from libmelee.melee.enums import Character, ControllerType, Menu, Stage
from libmelee.melee.menuhelper import MenuHelper

LOG_INTERVAL_FRAMES = 500


def _default_dolphin_path() -> Path | None:
    """Pick a bundled Dolphin binary if available."""
    for candidate in (Path("dolphin-emu-nogui"), Path("dolphin-emu")):
        if candidate.exists():
            return candidate.resolve()
    return None


def _parse_args() -> argparse.Namespace:
    default_iso = Path("melee.iso")
    default_dolphin = _default_dolphin_path()

    parser = argparse.ArgumentParser(
        description="Launch a headless Dolphin instance and time a CPU vs CPU match.",
    )
    parser.add_argument(
        "--dolphin-path",
        type=Path,
        default=default_dolphin,
        required=default_dolphin is None,
        help="Path to the dolphin-emu(-nogui) executable.",
    )
    parser.add_argument(
        "--iso",
        type=Path,
        default=default_iso if default_iso.exists() else None,
        required=not default_iso.exists(),
        help="Path to the Melee ISO.",
    )
    parser.add_argument(
        "--cpu-level",
        type=int,
        default=9,
        help="CPU difficulty level (1-9).",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=LOG_INTERVAL_FRAMES,
        help="Frames between FPS logs.",
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.cpu_level < 1 or args.cpu_level > 9:
        raise ValueError("CPU level must be between 1 and 9.")
    if args.dolphin_path is None or not args.dolphin_path.exists():
        raise FileNotFoundError(
            "Dolphin binary not found. Provide --dolphin-path to a valid executable."
        )
    args.dolphin_path = args.dolphin_path.resolve()
    if args.iso is None or not args.iso.exists():
        raise FileNotFoundError("Melee ISO not found. Provide a valid --iso path.")
    args.iso = args.iso.resolve()


def _create_console(dolphin_path: Path) -> Console:
    dolphin_home = Path.cwd() / "dolphin-home" / "User"
    dolphin_home.mkdir(parents=True, exist_ok=True)

    return Console(
        path=str(dolphin_path),
        dolphin_home_path=str(dolphin_home),
        slippi_address="127.0.0.1",
        save_replays=False,
        copy_home_directory=False,
        tmp_home_directory=False,
        blocking_input=True,
        gfx_backend="Null",
        disable_audio=True,
        emulation_speed=0.0,  # unlock FPS cap
        infinite_time=False,
        use_exi_inputs=True,
        enable_ffw=True,
    )


def log_fps(frame: int, frames_since_log: int, elapsed: float) -> None:
    fps = frames_since_log / elapsed if elapsed > 0 else float("inf")
    print(
        f"[Frame {frame}] Average FPS over last {frames_since_log} frames: {fps:.2f}"
    )


def main() -> int:
    args = _parse_args()
    _validate_args(args)

    console = _create_console(args.dolphin_path)
    controllers = {
        1: Controller(console=console, port=1, type=ControllerType.STANDARD),
        2: Controller(console=console, port=2, type=ControllerType.STANDARD),
    }
    menu_helper = MenuHelper()

    console.run(iso_path=str(args.iso))

    print("Connecting to console...")
    if not console.connect():
        print("ERROR: Failed to connect to the console.")
        console.stop()
        return 1

    for controller in controllers.values():
        if not controller.connect():
            print("ERROR: Failed to connect controller.")
            console.stop()
            return 1

    print("Console and controllers connected. Preparing CPU match...")

    frame_counter = 0
    last_log_frame = 0
    last_log_time = None
    game_started = False

    try:
        while True:
            gamestate = console.step()
            if gamestate is None:
                continue

            if gamestate.menu_state in (Menu.IN_GAME, Menu.SUDDEN_DEATH):
                if not game_started:
                    print("Game started. Tracking FPS...")
                    game_started = True
                    frame_counter = 0
                    last_log_frame = 0
                    last_log_time = time.perf_counter()

                frame_counter += 1
                if frame_counter - last_log_frame >= args.log_interval:
                    now = time.perf_counter()
                    elapsed = now - last_log_time if last_log_time is not None else 0.0
                    log_fps(frame_counter, frame_counter - last_log_frame, elapsed)
                    last_log_frame = frame_counter
                    last_log_time = now
            else:
                if game_started:
                    print(f"Game ended after {frame_counter} frames.")
                    break

                menu_helper.menu_helper_simple(
                    gamestate=gamestate,
                    controller=controllers[1],
                    character_selected=Character.FOX,
                    stage_selected=Stage.FINAL_DESTINATION,
                    cpu_level=args.cpu_level,
                    costume=1,
                    autostart=True,
                    swag=False,
                )
                menu_helper.choose_character(
                    character=Character.FOX,
                    gamestate=gamestate,
                    controller=controllers[2],
                    cpu_level=args.cpu_level,
                    costume=2,
                    swag=False,
                    start=True,
                )
    finally:
        for controller in controllers.values():
            controller.disconnect()
        console.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
