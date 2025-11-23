#!/usr/bin/python3
import argparse
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))

from config.config import init_config
from libmelee.melee.console import Console
from libmelee.melee.controller import Controller
from libmelee.melee.enums import Character, Stage, Menu, ControllerType
from libmelee.melee.menuhelper import MenuHelper
from model_interface import (
    GPTInferenceEngine,
    apply_model_outputs_to_game,
    collect_raw_inputs_from_gamestate,
)


@dataclass
class ProfilerConfig:
    a: int = 0


FRAME_LOG_INTERVAL = 60


def _format_ko_window(values):
    if not values:
        return "n/a"
    mn = min(values)
    mx = max(values)
    med = median(values)
    return f"{mn:.0f}/{med:.0f}/{mx:.0f}"


def log_player_snapshot(gamestate, ports, death_counts, death_percents, fps=None):
    """Print KO count/percent info for the tracked ports."""
    snapshot = []
    for port in ports:
        player_state = gamestate.players.get(port)
        if player_state is None:
            snapshot.append(f"P{port}: no data")
            continue
        percent = getattr(player_state, "percent", None)
        percent_str = percent if percent is not None else "?"
        deaths = death_counts.get(port, 0)
        ko_window = _format_ko_window(death_percents.get(port))
        snapshot.append(
            f"P{port}: {deaths} deaths (KO% min/med/max: {ko_window}), {percent_str}%"
        )
    fps_str = "FPS: n/a" if fps is None else f"FPS: {fps:.2f}"
    print(f"[Frame {gamestate.frame} | {fps_str}] " + " | ".join(snapshot))


if __name__ == "__main__":
    init_config()
    default_dolphin_path = Path(
        "/home/eppie/slippi-Ishiiruka/build/Binaries/dolphin-emu"
    )
    default_dolphin_path = (
        str(default_dolphin_path) if default_dolphin_path.exists() else None
    )
    default_dolphin_home = REPO_ROOT / "dolphin-home" / "User"
    default_dolphin_home.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser(description="Example of libmelee in action")
    parser.add_argument(
        "--debug",
        "-d",
        action="store_true",
        help="Debug mode. Creates a CSV of all game states",
    )
    parser.add_argument(
        "--address", "-a", default="127.0.0.1", help="IP address of Slippi/Wii"
    )
    parser.add_argument(
        "--dolphin_executable_path",
        "-e",
        default=default_dolphin_path,
        help="Path to the dolphin-emu-nogui executable",
    )
    parser.add_argument("--iso", default=None, type=str, help="Path to melee iso.")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=Path(
            "/Users/eppie/PycharmProjects/nano-melee/checkpoints/model_ep006_025002.pt"
        ),
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--button-threshold",
        default=0.45,
        type=float,
        help="Sigmoid threshold for button activation",
    )
    parser.add_argument(
        "--warmup-frames",
        default=256,
        type=int,
        help="Number of frames to buffer before using the model output",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        type=str,
        help="Dataset directory with meta.json; defaults to checkpoint config value",
    )

    args = parser.parse_args()
    engine = GPTInferenceEngine(
        checkpoint_path=args.checkpoint,
    )
    console = Console(
        path=args.dolphin_executable_path,
        dolphin_home_path=str(default_dolphin_home),
        slippi_address=args.address,
        save_replays=args.debug,
        copy_home_directory=False,
        tmp_home_directory=False,
        blocking_input=True,
        gfx_backend="Null",
        disable_audio=True,
        infinite_time=True,
        use_exi_inputs=True,
        enable_ffw=True,
    )
    ports = [1, 2]

    controllers = {
        1: Controller(
            console=console,
            port=1,
            type=ControllerType.STANDARD,
        ),
        2: Controller(console=console, port=2, type=ControllerType.STANDARD),
    }

    def signal_handler(sig, frame):
        for controller in controllers.values():
            controller.disconnect()
        console.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    console.run(iso_path=args.iso)

    # Connect to the console
    print("Connecting to console...")
    if not console.connect():
        print("ERROR: Failed to connect to the console.")
        sys.exit(-1)
    print("Console connected")
    for controller in controllers.values():
        if not controller.connect():
            print("ERROR: Failed to connect the controller.")
            sys.exit(-1)
    print("Controller connected")

    menu_helper = MenuHelper()

    BOT_PORT = 1
    OPP_PORT = 2
    # Main loop
    previous_gamestate = None
    frames_since_last_log = 0
    last_log_time = None
    death_counts = defaultdict(int)
    death_percents = defaultdict(list)
    prev_is_dead: dict[int, bool] = {}
    prev_percent: dict[int, float] = {}
    while True:
        # "step" to the next frame
        gamestate = console.step()
        if gamestate is None:
            continue

        # The console object keeps track of how long your bot is taking to process frames
        #   And can warn you if it's taking too long
        if console.processingtime * 1000 > 12:
            # print("WARNING: Last frame took " + str(console.processingtime * 1000) + "ms to process.")
            pass

        # What menu are we in?
        if gamestate.menu_state in [Menu.IN_GAME, Menu.SUDDEN_DEATH]:
            for port in ports:
                player_state = gamestate.players.get(port)
                if player_state is None:
                    continue
                prior_percent = prev_percent.get(port, player_state.percent)
                was_dead = prev_is_dead.get(port, player_state.is_dead)
                if not was_dead and player_state.is_dead:
                    death_counts[port] += 1
                    death_percents[port].append(prior_percent)
                prev_is_dead[port] = player_state.is_dead
                prev_percent[port] = player_state.percent
            frames_since_last_log += 1
            if frames_since_last_log >= FRAME_LOG_INTERVAL:
                now = time.perf_counter()
                fps = None
                if last_log_time is not None and now > last_log_time:
                    elapsed = now - last_log_time
                    fps = frames_since_last_log / elapsed
                log_player_snapshot(
                    gamestate, ports, death_counts, death_percents, fps=fps
                )
                last_log_time = now
                frames_since_last_log = 0
            raw_model_inputs = collect_raw_inputs_from_gamestate(
                gamestate, BOT_PORT, OPP_PORT
            )
            controller_state = engine.predict_from_raw(raw_model_inputs)
            apply_model_outputs_to_game(controllers[BOT_PORT], controller_state)
            previous_gamestate = gamestate

        else:
            frames_since_last_log = 0
            last_log_time = None
            death_counts = defaultdict(int)
            death_percents = defaultdict(list)
            prev_is_dead.clear()
            prev_percent.clear()

            menu_helper.menu_helper_simple(
                gamestate,
                controllers[1],
                Character.FOX,
                Stage.POKEMON_STADIUM,
                costume=1,
                autostart=False,
                swag=False,
            )
            menu_helper.choose_character(
                character=Character.FOX,
                gamestate=gamestate,
                controller=controllers[2],
                cpu_level=9,
                costume=1,
                swag=False,
                start=True,
            )
