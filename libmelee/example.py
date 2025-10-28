#!/usr/bin/python3
import argparse
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
from config import init_config
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


if __name__ == "__main__":
    init_config()
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
        default=None,
        help="The directory where dolphin is",
    )
    parser.add_argument("--iso", default=None, type=str, help="Path to melee iso.")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=Path(
            "/Users/eppie/PycharmProjects/nano-melee/checkpoints/model_ep007_010001.pt"
        ),
        help="Path to trained model checkpoint (.pt)",
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
        warmup_frames=args.warmup_frames,
        data_root=args.data_root,
    )
    console = Console(
        path=args.dolphin_executable_path,
        # dolphin_home_path=str(default_dolphin_home),
        slippi_address=args.address,
        save_replays=args.debug,
        copy_home_directory=False,
        tmp_home_directory=False,
        blocking_input=True,
        # gfx_backend="Null",
        # disable_audio=True,
        # infinite_time=True,
        # use_exi_inputs=True,
        # enable_ffw=True,
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
    while True:
        # "step" to the next frame
        gamestate = console.step()
        if gamestate is None:
            continue

        # The console object keeps track of how long your bot is taking to process frames
        #   And can warn you if it's taking too long
        if console.processingtime * 1000 > 12:
            print(
                "WARNING: Last frame took "
                + str(console.processingtime * 1000)
                + "ms to process."
            )

        # What menu are we in?
        if gamestate.menu_state in [Menu.IN_GAME, Menu.SUDDEN_DEATH]:
            raw_model_inputs = collect_raw_inputs_from_gamestate(
                gamestate, BOT_PORT, OPP_PORT
            )
            controller_state = engine.predict_from_raw(raw_model_inputs)
            apply_model_outputs_to_game(controllers[BOT_PORT], controller_state)
            previous_gamestate = gamestate

        else:

            menu_helper.menu_helper_simple(
                gamestate,
                controllers[1],
                Character.FOX,
                Stage.FINAL_DESTINATION,
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
