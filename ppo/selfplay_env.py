"""Self-play environment wrapper for PPO training."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict

from column_map import ColumnMap
from config import get_config
from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)
from libmelee.melee.console import Console
from libmelee.melee.controller import Controller
from libmelee.melee.enums import Button, Character, ControllerType, Stage
from libmelee.melee.gamestate import GameState
from libmelee.melee.menuhelper import MenuHelper
from model.nano_gpt import GPT
from model_interface import (
    ControllerState,
    collect_raw_inputs_from_gamestate,
)
from ppo.opponent_pool import OpponentPool
from ppo.trajectory import TrajectoryBuffer
from schema import get_feature_names, get_target_names
from train.batch_utils import build_model_inputs
from train.value_head import build_reward_feature_index


class SelfPlayEnvironment:
    """Self-play environment for PPO training.

    Manages:
    - Dolphin console and controllers
    - Two players: learner (P1) and opponent (P2)
    - Trajectory collection
    - Reward computation
    """

    def __init__(
        self,
        learner_model: GPT,
        opponent_pool: OpponentPool,
        dolphin_path: str,
        iso_path: str,
        device: torch.device,
        seq_len: int = 256,
        warmup_frames: int = 128,
        max_episode_frames: int = 18000,
    ):
        """Initialize self-play environment.

        Args:
            learner_model: The model being trained (player 1)
            opponent_pool: Pool of opponent models
            dolphin_path: Path to dolphin executable
            iso_path: Path to Melee ISO
            device: Device to run models on
            seq_len: Sequence length for model context
            warmup_frames: Number of frames to buffer before predictions
            max_episode_frames: Maximum frames per episode
        """
        self.learner_model = learner_model
        self.opponent_pool = opponent_pool
        self.device = device
        self.seq_len = seq_len
        self.warmup_frames = warmup_frames
        self.max_episode_frames = max_episode_frames

        # Feature configuration
        self.feature_names = get_feature_names()
        self.target_names = get_target_names()
        self.feature_dim = len(self.feature_names)

        # Dolphin console
        self.dolphin_path = dolphin_path
        self.iso_path = iso_path
        self.console: Optional[Console] = None
        self.controllers: Dict[int, Controller] = {}
        self.menu_helper = MenuHelper()

        # Ports
        self.learner_port = 1
        self.opponent_port = 2

        # Model buffers (separate for each player)
        self.learner_buffer: deque[torch.Tensor] = deque(maxlen=seq_len)
        self.opponent_buffer: deque[torch.Tensor] = deque(maxlen=seq_len)

        # Trajectory collection
        self.trajectory_buffer = TrajectoryBuffer()

        # Episode state
        self.frame_count = 0
        self.episode_reward = 0.0
        self.previous_gamestate: Optional[GameState] = None
        self.last_log_frame = 0  # For progress logging

        # Track previous state for reward deltas
        self.prev_p1_percent = 0.0
        self.prev_p2_percent = 0.0
        self.prev_p1_stock = 4
        self.prev_p2_stock = 4

        # Reward computation
        self.colmap = ColumnMap(self.feature_names, self.target_names)

        # Opponent model (loaded at episode start)
        self.opponent_model: Optional[GPT] = None

        # Action palettes
        self._main_stick_palette = np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32)
        self._c_stick_palette = np.asarray(C_STICK_QUANTIZED, dtype=np.float32)
        self._shoulder_centers = SHOULDER_QUANTIZED

    def initialize_console(self) -> None:
        """Initialize Dolphin console and controllers."""
        print("Initializing Dolphin console...")

        dolphin_home = Path.cwd() / "dolphin-home" / "User"
        dolphin_home.mkdir(parents=True, exist_ok=True)

        self.console = Console(
            path=self.dolphin_path,
            dolphin_home_path=str(dolphin_home),
            slippi_address="127.0.0.1",
            save_replays=False,
            copy_home_directory=False,
            tmp_home_directory=False,
            blocking_input=True,
            gfx_backend="Null",
            disable_audio=True,
            infinite_time=False,  # Use normal time limit
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
        self.console.run(iso_path=self.iso_path)

        # Connect
        if not self.console.connect():
            raise RuntimeError("Failed to connect to console")

        for controller in self.controllers.values():
            if not controller.connect():
                raise RuntimeError("Failed to connect controller")

        print("Console initialized successfully")

    def shutdown_console(self) -> None:
        """Shutdown Dolphin console."""
        if self.console:
            for controller in self.controllers.values():
                controller.disconnect()
            self.console.stop()
            self.console = None

    def reset_episode(self) -> None:
        """Reset for a new episode.

        - Load a random opponent from the pool
        - Clear buffers
        - Reset episode state
        """
        print(f"\n{'='*60}")
        print("Starting new episode...")

        # Load random opponent (if pool not empty)
        if not self.opponent_pool.is_empty():
            opponent_path, opponent_meta = self.opponent_pool.sample_opponent()
            print(
                f"Selected opponent: {opponent_path.name} (metadata: {opponent_meta})"
            )

            # Create fresh opponent model instance
            self.opponent_model = GPT(get_config()).to(self.device)
            self.opponent_pool.load_opponent_model(self.opponent_model, opponent_path)
            self.opponent_model.eval()
        else:
            print(
                "Warning: Opponent pool is empty. Using self-play against current model."
            )
            self.opponent_model = self.learner_model

        # Clear buffers
        self.learner_buffer.clear()
        self.opponent_buffer.clear()

        # Reset episode state
        self.frame_count = 0
        self.episode_reward = 0.0
        self.previous_gamestate = None
        self.last_log_frame = 0

        # Reset reward tracking
        self.prev_p1_percent = 0.0
        self.prev_p2_percent = 0.0
        self.prev_p1_stock = 4
        self.prev_p2_stock = 4

        # Finish any incomplete trajectory
        if len(self.trajectory_buffer.current_trajectory) > 0:
            self.trajectory_buffer.finish_trajectory()

        print(f"{'='*60}\n")

    def _frame_to_tensor(self, raw_inputs: Dict[str, float]) -> torch.Tensor:
        """Convert raw feature dict to tensor."""
        frame = torch.zeros(self.feature_dim, dtype=torch.float32)
        for idx, name in enumerate(self.feature_names):
            frame[idx] = float(raw_inputs.get(name, 0.0))
        return frame

    def _build_model_inputs(
        self,
        buffer: deque[torch.Tensor],
    ) -> Optional[TensorDict]:
        """Build model inputs from buffer."""
        if len(buffer) < self.warmup_frames:
            return None

        frames = list(buffer)
        stacked = torch.stack(frames, dim=0).unsqueeze(0).to(self.device)  # [1, T, F]

        return build_model_inputs(stacked, self.colmap)

    def _sample_action(
        self,
        outputs: TensorDict,
        exploration: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
        """Sample actions from model outputs.

        Args:
            outputs: Model output TensorDict
            exploration: If True, sample from distribution. If False, take argmax.

        Returns:
            Tuple of (action_logits, actions_taken, log_prob)
        """
        # Extract logits (last timestep)
        main_logits = outputs["main_stick"][0, -1]  # [num_bins]
        c_logits = outputs["c_stick"][0, -1]
        button_logits = outputs["buttons"][0, -1]  # [num_buttons]
        shoulder_logits = outputs.get("shoulder")
        if shoulder_logits is not None:
            shoulder_logits = shoulder_logits[0, -1]

        action_logits = {
            "main_stick": main_logits,
            "c_stick": c_logits,
            "buttons": button_logits,
        }
        if shoulder_logits is not None:
            action_logits["shoulder"] = shoulder_logits

        # Sample actions
        actions_taken = {}

        if exploration:
            # Sample from distributions
            main_probs = torch.softmax(main_logits, dim=-1)
            actions_taken["main_stick"] = torch.multinomial(main_probs, 1).squeeze(-1)

            c_probs = torch.softmax(c_logits, dim=-1)
            actions_taken["c_stick"] = torch.multinomial(c_probs, 1).squeeze(-1)

            if shoulder_logits is not None:
                shoulder_probs = torch.softmax(shoulder_logits, dim=-1)
                actions_taken["shoulder"] = torch.multinomial(
                    shoulder_probs, 1
                ).squeeze(-1)

            # Sample buttons independently
            button_probs = torch.sigmoid(button_logits)
            actions_taken["buttons"] = torch.bernoulli(button_probs).bool()
        else:
            # Argmax (deterministic)
            actions_taken["main_stick"] = torch.argmax(main_logits, dim=-1)
            actions_taken["c_stick"] = torch.argmax(c_logits, dim=-1)
            if shoulder_logits is not None:
                actions_taken["shoulder"] = torch.argmax(shoulder_logits, dim=-1)
            actions_taken["buttons"] = (torch.sigmoid(button_logits) > 0.5).bool()

        # Compute log probability
        log_prob = self._compute_log_prob(action_logits, actions_taken)

        return action_logits, actions_taken, log_prob

    def _compute_log_prob(
        self,
        action_logits: Dict[str, torch.Tensor],
        actions_taken: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute log probability of actions."""
        from ppo.ppo_loss import compute_log_probs

        # Add batch dimension
        logits_batched = {k: v.unsqueeze(0) for k, v in action_logits.items()}
        actions_batched = {k: v.unsqueeze(0) for k, v in actions_taken.items()}

        log_prob = compute_log_probs(logits_batched, actions_batched)
        return log_prob.squeeze(0)

    # TODO: Some of this might be duplicated from model_interface.py
    def _actions_to_controller_state(
        self,
        actions: Dict[str, torch.Tensor],
    ) -> ControllerState:
        """Convert action tensors to ControllerState."""
        # Sticks
        main_idx = int(actions["main_stick"].item())
        main_xy = self._main_stick_palette[main_idx]
        main_xy = (main_xy * 0.5 + 0.5).astype(np.float32)  # [-1,1] -> [0,1]

        c_idx = int(actions["c_stick"].item())
        c_xy = self._c_stick_palette[c_idx]
        c_xy = (c_xy * 0.5 + 0.5).astype(np.float32)

        # Shoulder
        if "shoulder" in actions:
            shoulder_idx = int(actions["shoulder"].item())
            shoulder_val = float(self._shoulder_centers[shoulder_idx])
        else:
            shoulder_val = 0.0

        # Buttons
        buttons = actions["buttons"].cpu().tolist()

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

    def _apply_controller_state(
        self,
        controller: Controller,
        state: ControllerState,
    ) -> None:
        """Apply ControllerState to game controller."""
        controller.release_all()

        if state.button_a:
            controller.press_button(Button.BUTTON_A)
        if state.button_b:
            controller.press_button(Button.BUTTON_B)
        if state.button_xy:
            controller.press_button(Button.BUTTON_X)
        if state.button_z:
            controller.press_button(Button.BUTTON_Z)
        if state.button_lr:
            controller.press_button(Button.BUTTON_L)

        controller.tilt_analog(
            Button.BUTTON_MAIN, state.main_stick_x, state.main_stick_y
        )
        controller.tilt_analog(Button.BUTTON_C, state.c_stick_x, state.c_stick_y)
        controller.press_shoulder(Button.BUTTON_L, state.shoulder_analog)

    def step(self, gamestate: GameState) -> Tuple[bool, Dict[str, float]]:
        """Execute one step of the environment.

        Args:
            gamestate: Current game state

        Returns:
            Tuple of (done, metrics)
        """
        self.frame_count += 1

        # Collect raw inputs
        raw_inputs = collect_raw_inputs_from_gamestate(
            gamestate,
            self.learner_port,
            self.opponent_port,
        )

        # Convert to tensor and add to buffers
        learner_frame = self._frame_to_tensor(raw_inputs.transformed)
        self.learner_buffer.append(learner_frame)

        # For opponent, we swap p1/p2 in the features
        opponent_raw = dict(raw_inputs.transformed)
        for key in list(opponent_raw.keys()):
            if key.startswith("p1_"):
                opponent_key = "p2_" + key[3:]
                if opponent_key in opponent_raw:
                    opponent_raw[key], opponent_raw[opponent_key] = (
                        opponent_raw[opponent_key],
                        opponent_raw[key],
                    )
        opponent_frame = self._frame_to_tensor(opponent_raw)
        self.opponent_buffer.append(opponent_frame)

        # Build model inputs
        learner_inputs = self._build_model_inputs(self.learner_buffer)
        opponent_inputs = self._build_model_inputs(self.opponent_buffer)

        # Get actions
        if learner_inputs is not None:
            with torch.no_grad():
                learner_outputs = self.learner_model(learner_inputs)
                learner_logits, learner_actions, learner_log_prob = self._sample_action(
                    learner_outputs, exploration=True
                )
                learner_value = learner_outputs.get("value", torch.zeros(1, 1, 1))[
                    0, -1, 0
                ]

            learner_controller = self._actions_to_controller_state(learner_actions)
            self._apply_controller_state(
                self.controllers[self.learner_port], learner_controller
            )
        else:
            # Warmup: neutral controller
            learner_logits = None
            learner_actions = None
            learner_log_prob = None
            learner_value = None

        if opponent_inputs is not None and self.opponent_model is not None:
            with torch.no_grad():
                opponent_outputs = self.opponent_model(opponent_inputs)
                _, opponent_actions, _ = self._sample_action(
                    opponent_outputs,
                    exploration=False,  # Opponent uses deterministic policy
                )

            opponent_controller = self._actions_to_controller_state(opponent_actions)
            self._apply_controller_state(
                self.controllers[self.opponent_port], opponent_controller
            )

        # Compute reward
        # TODO: Use reward computation from value_head.py
        reward = self._compute_reward(gamestate)
        self.episode_reward += reward

        # Check if episode is done
        done = False

        # Episode ends if:
        # 1. Max frames reached
        if self.frame_count >= self.max_episode_frames:
            done = True
            print(f"Episode ended: max frames ({self.max_episode_frames}) reached")

        # 2. Game ended (stocks depleted)
        p1 = gamestate.players.get(self.learner_port)
        p2 = gamestate.players.get(self.opponent_port)
        if p1 and p2:
            if p1.stock == 0 or p2.stock == 0:
                done = True
                winner = "Learner" if p1.stock > 0 else "Opponent"
                print(
                    f"Episode ended: {winner} won! (P1 stocks: {p1.stock}, P2 stocks: {p2.stock})"
                )

        # Store step in trajectory (only after warmup)
        if learner_logits is not None and learner_actions is not None:
            self.trajectory_buffer.add_step(
                state=learner_frame,
                action_logits=learner_logits,
                action_taken=learner_actions,
                log_prob=learner_log_prob,
                value=learner_value,
                reward=reward,
                done=done,
            )

        metrics = {
            "episode/frame": self.frame_count,
            "episode/reward": reward,
            "episode/total_reward": self.episode_reward,
        }

        if done and p1 and p2:
            metrics["episode/learner_stocks"] = p1.stock
            metrics["episode/opponent_stocks"] = p2.stock
            metrics["episode/learner_percent"] = p1.percent
            metrics["episode/opponent_percent"] = p2.percent

        self.previous_gamestate = gamestate

        # Progress logging every 1000 frames
        if self.frame_count - self.last_log_frame >= 1000:
            self.last_log_frame = self.frame_count
            if p1 and p2:
                print(
                    f"  Frame {self.frame_count}: "
                    f"Learner({p1.stock} stocks, {p1.percent:.0f}%) vs "
                    f"Opponent({p2.stock} stocks, {p2.percent:.0f}%) | "
                    f"Total reward: {self.episode_reward:.3f}"
                )

        return done, metrics

    def navigate_menu(self, gamestate: GameState) -> None:
        """Navigate menus to get into game."""
        # P1 (learner) selects character and stage
        self.menu_helper.menu_helper_simple(
            gamestate,
            self.controllers[self.learner_port],
            Character.FOX,
            Stage.FINAL_DESTINATION,
            costume=1,
            autostart=False,
            swag=False,
        )

        # P2 (opponent) selects character
        self.menu_helper.choose_character(
            character=Character.FOX,
            gamestate=gamestate,
            controller=self.controllers[self.opponent_port],
            cpu_level=0,  # Not CPU, controlled by our model
            costume=2,
            swag=False,
            start=True,
        )
