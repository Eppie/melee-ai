#!/usr/bin/python3
import argparse
import signal
import sys

import numpy as np

import melee
from inference import CHECKPOINT_PATH, InferenceEngine, gamestate_to_input, apply_model_output, debug_dump_model_inputs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Example of libmelee in action')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Debug mode. Creates a CSV of all game states')
    parser.add_argument('--address', '-a', default="127.0.0.1",
                        help='IP address of Slippi/Wii')
    parser.add_argument('--dolphin_executable_path', '-e', default=None,
                        help='The directory where dolphin is')
    parser.add_argument('--iso', default=None, type=str,
                        help='Path to melee iso.')

    args = parser.parse_args()
    console = melee.Console(
        path=args.dolphin_executable_path,
        slippi_address=args.address,
        save_replays=args.debug,
        copy_home_directory=False,
        tmp_home_directory=False
    )
    ports = [1, 2]

    controllers = {
        1: melee.Controller(
            console=console,
            port=1,
            type=melee.ControllerType.STANDARD,
        ),
        2: melee.Controller(
            console=console,
            port=2,
            type=melee.ControllerType.STANDARD)
    }


    def signal_handler(sig, frame):
        for controller in controllers.values():
            controller.disconnect()
        console.stop()
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)

    # Run the console
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

    menu_helper = melee.MenuHelper()
    engine = InferenceEngine(CHECKPOINT_PATH, threshold=0.5)
    BOT_PORT = 1
    OPP_PORT = 2
    # Main loop
    while True:
        # "step" to the next frame
        gamestate = console.step()
        if gamestate is None:
            continue

        # The console object keeps track of how long your bot is taking to process frames
        #   And can warn you if it's taking too long
        if console.processingtime * 1000 > 12:
            print("WARNING: Last frame took " + str(console.processingtime * 1000) + "ms to process.")

        # What menu are we in?
        if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
            feat: np.ndarray = gamestate_to_input(gamestate, p1_port=BOT_PORT, p2_port=OPP_PORT)

            engine.push_frame(feat)

            if engine.ready():
                outputs: np.ndarray = engine.infer()  # (C,), aligned with TARGET_COLUMNS
                # Send to bot controller
                apply_model_output(controllers[BOT_PORT], outputs.tolist())
                window = np.stack(list(engine.buffer), axis=0)  # (T=60, F)
                debug_dump_model_inputs(window, show_window_stats=True, which_frame="last")
            # melee.techskill.multishine(ai_state=gamestate.players[port], controller=controller)

        else:
            for port, controller in controllers.items():
                menu_helper.menu_helper_simple(
                    gamestate,
                    controller,
                    melee.Character.FOX,
                    melee.Stage.YOSHIS_STORY,
                    costume=port,
                    autostart=port == 1,
                    swag=False)
