"""Stateless simulation worker for distributed PPO training.

Each worker manages a single Dolphin emulator instance and acts as a physics
engine. Workers send game states to the central inference coordinator and
receive actions for both players.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)
from libmelee.melee.console import Console
from libmelee.melee.controller import Controller
from libmelee.melee.enums import Button, Character, ControllerType, Menu, Stage
from libmelee.melee.gamestate import GameState
from libmelee.melee.menuhelper import MenuHelper
from model_interface import ControllerState, collect_raw_inputs_from_gamestate
from schema import get_feature_names
from train.value_head import build_reward_feature_index, compute_frame_rewards
from column_map import ColumnMap


@dataclass
class WorkerConfig:
    """Configuration for a simulation worker."""

    worker_id: int
    dolphin_path: str
    iso_path: str
    learner_port: int = 1
    opponent_port: int = 2


def run_worker(
    worker_id: int,
    config: WorkerConfig,
    state_queue: mp.Queue,
    action_queue: mp.Queue,
    control_queue: mp.Queue,
) -> None:
    """Main worker process function.

    This runs in a separate process and manages a single Dolphin instance.
    The worker loop:
    1. Extracts game state and converts to features
    2. Sends (worker_id, features, reward, done) to state_queue
    3. Waits for (p1_actions, p2_actions) from action_queue
    4. Applies actions to both controllers
    5. Advances the game by one frame

    Args:
        worker_id: Unique identifier for this worker
        config: Worker configuration
        state_queue: Queue to send states to coordinator
        action_queue: Queue to receive actions from coordinator
        control_queue: Queue for control signals (shutdown, pause, etc.)
    """
    # Initialize config for this worker process (needed for spawn method)
    from config.config import init_config

    init_config()

    worker = SimulationWorker(
        worker_id=worker_id,
        config=config,
        state_queue=state_queue,
        action_queue=action_queue,
        control_queue=control_queue,
    )

    try:
        worker.run()
    except KeyboardInterrupt:
        print(f"[Worker {worker_id}] Interrupted")
    except Exception as e:
        print(f"[Worker {worker_id}] Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        worker.shutdown()


class SimulationWorker:
    """Stateless simulation worker managing a single Dolphin instance.

    This worker has no model - it only manages the emulator and translates
    between game states and the central inference coordinator.
    """

    def __init__(
        self,
        worker_id: int,
        config: WorkerConfig,
        state_queue: mp.Queue,
        action_queue: mp.Queue,
        control_queue: mp.Queue,
    ):
        """Initialize the simulation worker.

        Args:
            worker_id: Unique identifier for this worker
            config: Worker configuration
            state_queue: Queue to send states to coordinator
            action_queue: Queue to receive actions from coordinator
            control_queue: Queue for control signals
        """
        self.worker_id = worker_id
        self.config = config
        self.state_queue = state_queue
        self.action_queue = action_queue
        self.control_queue = control_queue

        self.learner_port = config.learner_port
        self.opponent_port = config.opponent_port

        # Feature configuration
        self.feature_names = get_feature_names()
        self.feature_dim = len(self.feature_names)

        # Reward computation
        from config.config import get_config

        cfg = get_config()
        target_names = []  # We don't need targets for reward computation
        self.colmap = ColumnMap(self.feature_names, target_names)
        self.reward_feature_idx = build_reward_feature_index(self.colmap)

        # Dolphin console
        self.console: Optional[Console] = None
        self.controllers: Dict[int, Controller] = {}
        self.menu_helper = MenuHelper()

        # Action palettes
        self._main_stick_palette = np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32)
        self._c_stick_palette = np.asarray(C_STICK_QUANTIZED, dtype=np.float32)
        self._shoulder_centers = SHOULDER_QUANTIZED

        # State tracking for reward computation
        self.prev_features: Optional[torch.Tensor] = None
        self.frame_count = 0

        # Running flag
        self.running = False

    def initialize_console(self) -> None:
        """Initialize Dolphin console and controllers."""
        print(f"[Worker {self.worker_id}] Initializing Dolphin...")

        # Unique home directory per worker to avoid conflicts
        dolphin_home = Path.cwd() / f"dolphin-home-{self.worker_id}" / "User"
        dolphin_home.mkdir(parents=True, exist_ok=True)

        self.console = Console(
            path=self.config.dolphin_path,
            dolphin_home_path=str(dolphin_home),
            slippi_address="127.0.0.1",
            slippi_port=51441 + self.worker_id,  # Unique port per worker
            save_replays=False,
            copy_home_directory=False,
            tmp_home_directory=False,
            blocking_input=True,
            gfx_backend="Null",
            disable_audio=True,
            emulation_speed=0.0,  # Unlock FPS
            infinite_time=True,  # Matches never end due to time
            use_exi_inputs=True,
            enable_ffw=True,
        )

        # Create controllers
        for port in [self.learner_port, self.opponent_port]:
            self.controllers[port] = Controller(
                console=self.console,
                port=port,
                type=ControllerType.STANDARD,
            )

        # Start console
        self.console.run(iso_path=self.config.iso_path)

        # Connect
        if not self.console.connect():
            raise RuntimeError(
                f"[Worker {self.worker_id}] Failed to connect to console"
            )

        for controller in self.controllers.values():
            if not controller.connect():
                raise RuntimeError(
                    f"[Worker {self.worker_id}] Failed to connect controller"
                )

        print(f"[Worker {self.worker_id}] Console initialized")

    def shutdown(self) -> None:
        """Shutdown the Dolphin console."""
        if self.console:
            for controller in self.controllers.values():
                try:
                    controller.disconnect()
                except:
                    pass
            try:
                self.console.stop()
            except:
                pass
            self.console = None
        print(f"[Worker {self.worker_id}] Shutdown complete")

    def run(self) -> None:
        """Main worker loop."""
        self.initialize_console()
        self.running = True

        print(f"[Worker {self.worker_id}] Starting main loop")

        while self.running:
            # Check for control signals (non-blocking)
            try:
                signal = self.control_queue.get_nowait()
                if signal == "shutdown":
                    print(f"[Worker {self.worker_id}] Received shutdown signal")
                    break
                elif signal == "pause":
                    # Wait for resume signal
                    while True:
                        signal = self.control_queue.get()
                        if signal == "resume":
                            break
                        elif signal == "shutdown":
                            self.running = False
                            break
            except:
                pass  # No control signal

            if not self.running:
                break

            # Get game state
            gamestate = self.console.step()
            if gamestate is None:
                continue

            # Handle menus
            if gamestate.menu_state not in [Menu.IN_GAME, Menu.SUDDEN_DEATH]:
                self._navigate_menu(gamestate)
                continue

            # Process game frame
            self._process_frame(gamestate)

    def _navigate_menu(self, gamestate: GameState) -> None:
        """Navigate menus to start a match."""
        # P1 selects character and stage
        self.menu_helper.menu_helper_simple(
            gamestate,
            self.controllers[self.learner_port],
            Character.FOX,
            Stage.FINAL_DESTINATION,
            costume=1,
            autostart=False,
            swag=False,
        )

        # P2 selects character
        self.menu_helper.choose_character(
            character=Character.FOX,
            gamestate=gamestate,
            controller=self.controllers[self.opponent_port],
            cpu_level=0,
            costume=2,
            swag=False,
            start=True,
        )

    def _process_frame(self, gamestate: GameState) -> None:
        """Process a single game frame.

        1. Extract features
        2. Compute reward from previous state
        3. Check for episode end
        4. Send state to coordinator
        5. Receive and apply actions
        """
        self.frame_count += 1

        # Extract features
        raw_inputs = collect_raw_inputs_from_gamestate(
            gamestate,
            self.learner_port,
            self.opponent_port,
        )
        features = self._frame_to_tensor(raw_inputs.transformed)

        # Compute reward (from state change)
        reward = 0.0
        if self.prev_features is not None:
            reward = self._compute_reward(self.prev_features, features)

        # Episode continues indefinitely (infinite stocks/time in PPO)
        # Episodes are sliced arbitrarily by the coordinator
        done = False
        p1 = gamestate.players.get(self.learner_port)
        p2 = gamestate.players.get(self.opponent_port)

        if p1 and p2:
            # Print percent every 120 frames (2 seconds of game time)
            if self.frame_count % 120 == 0:
                print(
                    f"[Worker {self.worker_id}] Frame {self.frame_count} | "
                    f"P1: {p1.percent:.1f}% ({p1.stock} stocks) | "
                    f"P2: {p2.percent:.1f}% ({p2.stock} stocks)"
                )

        # Send state to coordinator (convert tensor to numpy to avoid file descriptor issues)
        features_np = features.numpy()
        self.state_queue.put((self.worker_id, features_np, reward, done))

        # Wait for actions
        p1_actions, p2_actions = self.action_queue.get()

        # Apply actions
        self._apply_actions(self.controllers[self.learner_port], p1_actions)
        self._apply_actions(self.controllers[self.opponent_port], p2_actions)

        # Update state
        self.prev_features = features

        # Handle episode end
        if done:
            self._restart_match()

    def _frame_to_tensor(self, raw_inputs: Dict[str, float]) -> torch.Tensor:
        """Convert raw feature dict to tensor."""
        frame = torch.zeros(self.feature_dim, dtype=torch.float32)
        for idx, name in enumerate(self.feature_names):
            frame[idx] = float(raw_inputs.get(name, 0.0))
        return frame

    def _compute_reward(
        self,
        prev_features: torch.Tensor,
        curr_features: torch.Tensor,
    ) -> float:
        """Compute reward from state transition.

        Uses the same reward computation as the original implementation.
        """
        # Stack for batch format expected by compute_frame_rewards
        states = torch.stack([prev_features, curr_features], dim=0).unsqueeze(
            0
        )  # [1, 2, F]
        rewards = compute_frame_rewards(states, self.reward_feature_idx)  # [1, 1]
        return float(rewards[0, 0].item())

    def _apply_actions(
        self,
        controller: Controller,
        actions: Dict[str, torch.Tensor],
    ) -> None:
        """Apply action dict to controller."""
        controller_state = self._actions_to_controller_state(actions)

        controller.release_all()

        if controller_state.button_a:
            controller.press_button(Button.BUTTON_A)
        if controller_state.button_b:
            controller.press_button(Button.BUTTON_B)
        if controller_state.button_xy:
            controller.press_button(Button.BUTTON_X)
        if controller_state.button_z:
            controller.press_button(Button.BUTTON_Z)
        if controller_state.button_lr:
            controller.press_button(Button.BUTTON_L)

        controller.tilt_analog(
            Button.BUTTON_MAIN,
            controller_state.main_stick_x,
            controller_state.main_stick_y,
        )
        controller.tilt_analog(
            Button.BUTTON_C,
            controller_state.c_stick_x,
            controller_state.c_stick_y,
        )
        controller.press_shoulder(Button.BUTTON_L, controller_state.shoulder_analog)

    def _actions_to_controller_state(
        self,
        actions: Dict,
    ) -> ControllerState:
        """Convert action dict (tensors or primitives) to ControllerState."""

        # Helper to extract int from tensor or primitive
        def to_int(val):
            if isinstance(val, int):
                return val
            return int(val.item())

        # Helper to extract bool list from tensor or primitive
        def to_bool_list(val):
            if isinstance(val, list):
                return val
            return val.tolist()

        # Main stick
        main_idx = to_int(actions["main_stick"])
        main_xy = self._main_stick_palette[main_idx]
        main_xy = (main_xy * 0.5 + 0.5).astype(np.float32)  # [-1,1] -> [0,1]

        # C-stick
        c_idx = to_int(actions["c_stick"])
        c_xy = self._c_stick_palette[c_idx]
        c_xy = (c_xy * 0.5 + 0.5).astype(np.float32)

        # Shoulder
        shoulder_idx = to_int(actions["shoulder"])
        shoulder_val = float(self._shoulder_centers[shoulder_idx])

        # Buttons
        buttons = to_bool_list(actions["buttons"])

        return ControllerState(
            main_stick_x=float(main_xy[0]),
            main_stick_y=float(main_xy[1]),
            c_stick_x=float(c_xy[0]),
            c_stick_y=float(c_xy[1]),
            shoulder_analog=shoulder_val,
            button_a=bool(buttons[0]),
            button_b=bool(buttons[1]),
            button_xy=bool(buttons[2]),
            button_z=bool(buttons[3]),
            button_lr=bool(buttons[4]),
        )

    def _restart_match(self) -> None:
        """Restart the match after it ends.

        Note: With infinite stocks/time, matches never actually end naturally.
        This is here for manual reset scenarios.
        """
        print(f"[Worker {self.worker_id}] Restarting match...")
        self.prev_features = None
        self.frame_count = 0
        # The menu navigation will happen automatically in the main loop
        # when we detect we're no longer in-game
