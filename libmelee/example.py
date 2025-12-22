#!/usr/bin/python3
import argparse
import random
import signal
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from tensordict import TensorDict

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
from config.config import init_config
from libmelee.melee.console import Console
from libmelee.melee.controller import Controller
from libmelee.melee.enums import (
    Character,
    ControllerStatus,
    ControllerType,
    Menu,
    Stage,
)
from libmelee.melee.menuhelper import MenuHelper
from train import find_latest_checkpoint
from model_interface import (
    GPTInferenceEngine,
    apply_model_outputs_to_game,
    collect_raw_inputs_from_gamestate,
)


@dataclass
class ProfilerConfig:
    a: int = 0


LEGAL_TOURNAMENT_STAGES = [
    Stage.BATTLEFIELD,
    Stage.YOSHIS_STORY,
    Stage.POKEMON_STADIUM,
    Stage.DREAMLAND,
    Stage.FINAL_DESTINATION,
    Stage.FOUNTAIN_OF_DREAMS,
]

SUPPORTED_CHARS = [
    Character.FOX,
       # Character.FALCO,
       # Character.CPTFALCON,
       # Character.JIGGLYPUFF,
       # Character.MARTH,
    #    Character.SHEIK,
]


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
        default=None,
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--self-play",
        "-s",
        action="store_true",
        help="Enable self-play mode (two model instances vs each other)",
    )
    parser.add_argument(
        "--checkpoint2",
        "-c2",
        type=Path,
        default=None,
        help="Path to second model checkpoint for self-play (defaults to same as --checkpoint)",
    )

    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        default_dir = Path("checkpoints")
        latest = find_latest_checkpoint(default_dir)
        if latest is None:
            raise FileNotFoundError(
                f"No checkpoint provided and none found in {default_dir.resolve()}"
            )
        checkpoint_path = latest

    # Initialize primary model
    engine = GPTInferenceEngine(
        checkpoint_path=checkpoint_path,
    )

    # Initialize second model for self-play if enabled
    engine2 = None
    use_batched_inference = False
    if args.self_play:
        checkpoint_path2 = args.checkpoint2 if args.checkpoint2 else checkpoint_path

        # If both checkpoints are the same, we can share the model and use batched inference
        if checkpoint_path2 == checkpoint_path:
            print(
                f"Self-play mode enabled: Port 1 vs Port 2 (same model, batched inference)"
            )
            use_batched_inference = True
            # Create second engine but share the model from the first engine
            engine2 = GPTInferenceEngine.__new__(GPTInferenceEngine)
            # Copy attributes from engine
            engine2.__dict__.update(engine.__dict__.copy())
            # Create separate buffers and history for port 2
            from collections import deque

            engine2.buffer = deque(maxlen=engine.seq_len)
            engine2.frame_history = deque(maxlen=engine.seq_len)
            engine2._prev_controller_features = {}
            engine2._frames_seen = 0
            engine2._death_counter = 0
            engine2._prev_stock = None
            # Share the model reference
            engine2.model = engine.model
        else:
            print(f"Self-play mode enabled: Port 1 vs Port 2 (different models)")
            engine2 = GPTInferenceEngine(
                checkpoint_path=checkpoint_path2,
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
    current_stage = random.choice(LEGAL_TOURNAMENT_STAGES)
    bot_char = random.choice(SUPPORTED_CHARS)
    opp_char = random.choice(SUPPORTED_CHARS)

    # Ensure CPU doesn't pick Sheik. If Sheik is chosen, change to Zelda.
    if not args.self_play and opp_char is Character.SHEIK:
        print("WARNING: CPU cannot pick Sheik. Changing opponent to Zelda.")
        opp_char = Character.ZELDA

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
            if args.self_play and engine2 is not None and use_batched_inference:
                # Batched inference: prepare inputs from both engines, run single forward pass
                raw_model_inputs1 = collect_raw_inputs_from_gamestate(
                    gamestate, BOT_PORT, OPP_PORT
                )
                raw_model_inputs2 = collect_raw_inputs_from_gamestate(
                    gamestate, OPP_PORT, BOT_PORT
                )

                # Prepare inputs from both engines (updates their buffers)
                inputs_td1, _, _, record1 = engine.prepare_only(raw_model_inputs1)
                inputs_td2, _, _, record2 = engine2.prepare_only(raw_model_inputs2)

                # Check if both are ready for inference
                if (
                    inputs_td1 is not None
                    and inputs_td2 is not None
                    and len(engine.buffer) >= engine.warmup_frames
                    and len(engine2.buffer) >= engine2.warmup_frames
                ):
                    # Batch inputs along batch dimension
                    batched_td = TensorDict({}, batch_size=[2])
                    for key in inputs_td1.keys():
                        batched_td[key] = torch.cat(
                            [inputs_td1[key], inputs_td2[key]], dim=0
                        )

                    # Single forward pass
                    with torch.inference_mode():
                        outputs = engine.model(batched_td)

                    # Extract outputs for each player
                    outputs1 = TensorDict({}, batch_size=[1])
                    outputs2 = TensorDict({}, batch_size=[1])
                    for key in outputs.keys():
                        outputs1[key] = outputs[key][0:1]
                        outputs2[key] = outputs[key][1:2]

                    # Decode outputs
                    controller_state1 = engine.decode_only(outputs1, record1)
                    controller_state2 = engine2.decode_only(outputs2, record2)

                    apply_model_outputs_to_game(
                        controllers[BOT_PORT], controller_state1
                    )
                    apply_model_outputs_to_game(
                        controllers[OPP_PORT], controller_state2
                    )
                else:
                    # During warmup, use neutral controllers
                    from model_interface import ControllerState

                    apply_model_outputs_to_game(
                        controllers[BOT_PORT], ControllerState.neutral()
                    )
                    apply_model_outputs_to_game(
                        controllers[OPP_PORT], ControllerState.neutral()
                    )

            elif args.self_play and engine2 is not None:
                # Non-batched inference: run each engine separately
                raw_model_inputs1 = collect_raw_inputs_from_gamestate(
                    gamestate, BOT_PORT, OPP_PORT
                )
                raw_model_inputs2 = collect_raw_inputs_from_gamestate(
                    gamestate, OPP_PORT, BOT_PORT
                )
                controller_state1 = engine.predict_from_raw(raw_model_inputs1)
                controller_state2 = engine2.predict_from_raw(raw_model_inputs2)
                apply_model_outputs_to_game(controllers[BOT_PORT], controller_state1)
                apply_model_outputs_to_game(controllers[OPP_PORT], controller_state2)
            else:
                # Single player vs CPU
                raw_model_inputs = collect_raw_inputs_from_gamestate(
                    gamestate, BOT_PORT, OPP_PORT
                )
                controller_state = engine.predict_from_raw(raw_model_inputs)
                apply_model_outputs_to_game(controllers[BOT_PORT], controller_state)

            previous_gamestate = gamestate

        else:
            if previous_gamestate and previous_gamestate.menu_state in [
                Menu.IN_GAME,
                Menu.SUDDEN_DEATH,
            ]:
                current_stage = random.choice(LEGAL_TOURNAMENT_STAGES)
                bot_char = random.choice(SUPPORTED_CHARS)
                opp_char = random.choice(SUPPORTED_CHARS)
                print(
                    f"Picking stage: {current_stage}, bot: {bot_char}, opp: {opp_char}"
                )

            # Check if we are ready to start
            autostart = False
            if 1 in gamestate.players and 2 in gamestate.players:
                # Port 1
                p1_state = gamestate.players[1]
                p1_ready = (p1_state.character == bot_char) and p1_state.coin_down

                # Port 2
                p2_state = gamestate.players[2]
                if args.self_play:
                    p2_ready = (p2_state.character == opp_char) and p2_state.coin_down
                else:
                    # CPU Check
                    p2_ready = (
                        (p2_state.character == opp_char)
                        and (
                            p2_state.controller_status
                            == ControllerStatus.CONTROLLER_CPU
                        )
                        and (p2_state.cpu_level == 9)
                    )

                autostart = p1_ready and p2_ready
                if gamestate.frame % 60 == 0:  # Print once per second
                    print(f"Frame: {gamestate.frame}, Menu: {gamestate.menu_state}")
                    print(
                        f"P1 ({bot_char}): {p1_state.character}, Coin: {p1_state.coin_down} -> Ready: {p1_ready}"
                    )
                    print(
                        f"P2 ({opp_char}): {p2_state.character}, Status: {p2_state.controller_status}, Level: {p2_state.cpu_level} -> Ready: {p2_ready}"
                    )
                    print(f"Autostart: {autostart}")

            menu_helper.menu_helper_simple(
                gamestate,
                controllers[1],
                bot_char,
                current_stage,
                costume=1,
                autostart=autostart,
                swag=False,
            )
            # Configure port 2: model in self-play mode, CPU otherwise
            if args.self_play:
                menu_helper.menu_helper_simple(
                    gamestate,
                    controllers[2],
                    opp_char,
                    current_stage,
                    costume=2,
                    autostart=False,
                    swag=False,
                )
            else:
                menu_helper.choose_character(
                    character=opp_char,
                    gamestate=gamestate,
                    controller=controllers[2],
                    cpu_level=9,
                    costume=2,
                    swag=False,
                    start=False,
                )
        previous_gamestate = gamestate
