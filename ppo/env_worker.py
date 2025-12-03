"""EnvWorker: Thread managing single Dolphin instance lifecycle."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Optional

import numpy as np

from column_map import ColumnMap
from libmelee.melee import enums
from libmelee.melee.console import Console
from libmelee.melee.controller import Controller
from libmelee.melee.gamestate import GameState
from model_interface import collect_raw_inputs_from_gamestate
from schema import get_feature_names
from train.value_head import compute_frame_rewards, build_reward_feature_index

from .config import PPOConfig
from .rollout import RolloutBuffer
from .shared_memory import ActionData, SharedMemorySlab


# Stage enum mapping (abbreviations → libmelee enums)
STAGE_MAP = {
    "FD": enums.Stage.FINAL_DESTINATION,
    "BF": enums.Stage.BATTLEFIELD,
    "YS": enums.Stage.YOSHIS_STORY,
    "FoD": enums.Stage.FOUNTAIN_OF_DREAMS,
    "PS": enums.Stage.POKEMON_STADIUM,
    "DL": enums.Stage.DREAMLAND,
}

# Character enum mapping
CHARACTER_MAP = {
    "FOX": enums.Character.FOX,
    "FALCO": enums.Character.FALCO,
    "MARTH": enums.Character.MARTH,
    "SHEIK": enums.Character.SHEIK,
    "FALCON": enums.Character.CPTFALCON,
}


class EnvWorker:
    """
    Thread managing one Dolphin instance and its I/O.

    Responsibilities:
    - Initialize and restart Dolphin subprocess
    - Step Dolphin, extract features
    - Write features to shared memory slab
    - Wait for CRD to compute actions
    - Apply actions to controller
    - Compute rewards and store rollout
    - Handle warmup masking
    """

    def __init__(
        self,
        shard_id: int,
        env_id: int,  # 0-7 within shard
        shared_slab: SharedMemorySlab,
        config: PPOConfig,
        column_map: ColumnMap,
    ):
        self.shard_id = shard_id
        self.env_id = env_id
        self.global_env_id = shard_id * config.envs_per_shard + env_id
        self.slab = shared_slab
        self.config = config
        self.column_map = column_map

        # Dolphin lifecycle
        self.console: Optional[Console] = None
        self.ego_controller: Optional[Controller] = None
        self.opp_controller: Optional[Controller] = None

        # Ring buffer state
        self.t_local = 0  # Local frame counter
        self.t_mod = 0  # Ring position (0-255)
        self.is_warm = False

        # Rollout buffer (1024 frames)
        self.rollout = RolloutBuffer(
            rollout_length=config.rollout_length,
            feature_dim=908,
        )

        # Restart tracking
        self.frames_since_restart = 0

        # Feature names for featurization
        self.feature_names = get_feature_names()

        # Reward computation
        self.reward_idx = build_reward_feature_index(self.column_map)

        # Stage selection
        self.current_stage = self._select_stage()

    def _select_stage(self) -> enums.Stage:
        """Select random stage from tournament legal pool."""
        stage_abbrev = random.choice(self.config.stages)
        return STAGE_MAP[stage_abbrev]

    def _start_dolphin(self):
        """Initialize Dolphin subprocess and controllers."""
        # Create console
        self.console = Console(
            path=self.config.dolphin_path,
            slippi_port=51441 + self.global_env_id,  # Unique port per env
            logger=None,  # Suppress logging for now
        )

        # Start Dolphin
        self.console.run(
            iso_path=str(Path(self.config.iso_path).expanduser()),
        )

        # Connect controllers
        self.ego_controller = Controller(
            console=self.console,
            port=self.config.bot_port,
            type=enums.ControllerType.STANDARD,
        )
        self.opp_controller = Controller(
            console=self.console,
            port=self.config.opp_port,
            type=enums.ControllerType.STANDARD,
        )

        # Connect to console
        if not self.console.connect():
            raise RuntimeError(
                f"ENV {self.global_env_id}: Failed to connect to Dolphin"
            )

        # Select characters and stage
        character = CHARACTER_MAP[self.config.character]

        # Navigate menus (simplified - assumes fast-start or auto-pilot)
        # In production, you'd implement full menu navigation
        # For now, assume Dolphin is configured to skip menus or use savestate
        print(
            f"[ENV {self.global_env_id}] Dolphin started: "
            f"{self.config.character} vs {self.config.character} on {self.current_stage.name}"
        )

    def _restart_dolphin(self):
        """Restart Dolphin to prevent memory leaks."""
        print(f"[ENV {self.global_env_id}] Restarting Dolphin (periodic cleanup)")

        if self.console:
            try:
                self.console.stop()
            except Exception as e:
                print(f"[ENV {self.global_env_id}] Error stopping Dolphin: {e}")

        # Reset state
        self.frames_since_restart = 0
        self.t_local = 0
        self.t_mod = 0
        self.is_warm = False
        self.rollout.reset()

        # Select new stage
        self.current_stage = self._select_stage()

        # Restart Dolphin
        self._start_dolphin()

    def _featurize(self, gamestate: GameState) -> np.ndarray:
        """
        Extract 908-float feature vector from gamestate.

        Reuses existing model_interface infrastructure.
        """
        model_inputs = collect_raw_inputs_from_gamestate(
            gamestate,
            bot_port=self.config.bot_port,
            opp_port=self.config.opp_port,
        )

        # Convert transformed dict to numpy array in correct order
        features = np.array(
            [model_inputs.transformed[k] for k in self.feature_names],
            dtype=np.float32,
        )

        return features

    def _compute_reward(self, gamestate: GameState, features: np.ndarray) -> float:
        """
        Compute zero-sum reward for current frame.

        Returns: ego_reward - opp_reward
        """
        # For now, simple zero-sum based on damage dealt
        # TODO: Use compute_frame_rewards when we have proper integration

        ego = gamestate.players.get(self.config.bot_port)
        opp = gamestate.players.get(self.config.opp_port)

        if ego is None or opp is None:
            return 0.0

        # Damage dealt: opponent's damage increase
        # Stock taken: opponent died
        reward = 0.0

        # Track damage changes (simple heuristic)
        # In production, track per-env state to compute deltas
        # For now, placeholder
        reward += 0.0  # TODO: implement proper reward tracking

        return reward

    def _apply_neutral_action(self):
        """Apply neutral controller state (warmup)."""
        # Release all buttons
        self.ego_controller.release_all()

        # Neutral sticks (0.5 = center)
        self.ego_controller.tilt_analog(
            button=enums.Button.BUTTON_MAIN,
            x=0.5,
            y=0.5,
        )
        self.ego_controller.tilt_analog(
            button=enums.Button.BUTTON_C,
            x=0.5,
            y=0.5,
        )

        # Similarly for opponent (controlled by us)
        self.opp_controller.release_all()
        self.opp_controller.tilt_analog(enums.Button.BUTTON_MAIN, 0.5, 0.5)
        self.opp_controller.tilt_analog(enums.Button.BUTTON_C, 0.5, 0.5)

    def _apply_action(self, action: ActionData, is_ego: bool = True):
        """
        Decode quantized action and apply to controller.

        Args:
            action: ActionData from CRD
            is_ego: If True, apply to ego controller; else opponent
        """
        from controller_quantization import (
            CONTROL_STICK_QUANTIZED,
            C_STICK_QUANTIZED,
            SHOULDER_QUANTIZED,
        )

        controller = self.ego_controller if is_ego else self.opp_controller

        # Decode main stick (quantized index → [-1, 1] → [0, 1])
        main_xy = CONTROL_STICK_QUANTIZED[action.main_idx]
        main_01_x = (main_xy[0] + 1) / 2  # [-1, 1] → [0, 1]
        main_01_y = (main_xy[1] + 1) / 2

        # Decode c-stick
        c_xy = C_STICK_QUANTIZED[action.c_idx]
        c_01_x = (c_xy[0] + 1) / 2
        c_01_y = (c_xy[1] + 1) / 2

        # Decode shoulder
        shoulder_val = SHOULDER_QUANTIZED[action.shoulder_idx]

        # Apply to controller
        controller.tilt_analog(enums.Button.BUTTON_MAIN, main_01_x, main_01_y)
        controller.tilt_analog(enums.Button.BUTTON_C, c_01_x, c_01_y)
        controller.press_shoulder(enums.Button.BUTTON_L, shoulder_val)

        # Apply buttons
        controller.release_all()  # Clear previous buttons
        if action.buttons[0]:
            controller.press_button(enums.Button.BUTTON_A)
        if action.buttons[1]:
            controller.press_button(enums.Button.BUTTON_B)
        if action.buttons[2]:
            controller.press_button(enums.Button.BUTTON_X)
        if action.buttons[3]:
            controller.press_button(enums.Button.BUTTON_Z)
        if action.buttons[4]:
            controller.press_button(enums.Button.BUTTON_L)

    def _store_rollout_frame(
        self,
        features: np.ndarray,
        action: ActionData,
        reward: float,
    ):
        """Store frame in rollout buffer."""
        if not self.rollout.complete:
            self.rollout.append(
                features=features,
                action=action.to_numpy_struct(),
                logp=action.logp,
                value=action.value,
                reward=reward,
                mask=self.is_warm,
            )

    def run(self):
        """Main worker loop."""
        # Start Dolphin
        self._start_dolphin()

        print(f"[ENV {self.global_env_id}] Worker loop started")

        while True:
            try:
                # 1. Step Dolphin and get gamestate
                gamestate = self.console.step()
                if gamestate is None:
                    time.sleep(0.001)
                    continue

                # 2. Featurize
                features = self._featurize(gamestate)

                # 3. Write to shared ring buffer at position t_mod
                self.slab.features[self.env_id, self.t_mod, :] = features

                # 4. Mark ready for CRD
                self.slab.ready_flags[self.env_id] = 1

                # 5. Wait for CRD to compute action (busy-wait)
                while self.slab.ready_flags[self.env_id] == 1:
                    time.sleep(0.0001)  # Brief sleep to yield CPU

                # 6. Read actions from shared slot
                ego_action = self.slab.read_action(self.env_id, is_ego=True)
                opp_action = self.slab.read_action(self.env_id, is_ego=False)

                # 7. Apply actions to controllers
                if not self.is_warm:
                    self._apply_neutral_action()
                else:
                    self._apply_action(ego_action, is_ego=True)
                    self._apply_action(opp_action, is_ego=False)

                # 8. Compute reward
                reward = self._compute_reward(gamestate, features)

                # 9. Store in rollout buffer
                self._store_rollout_frame(features, ego_action, reward)

                # 10. Update ring position
                self.t_mod = (self.t_mod + 1) % self.config.context_length
                self.t_local += 1
                self.frames_since_restart += 1

                # Mark warm after warmup period
                if self.t_local >= self.config.warmup_frames:
                    self.is_warm = True

                # 11. Check if rollout complete
                if self.is_warm and self.rollout.complete:
                    # Compute advantages
                    self.rollout.compute_advantages(
                        gamma=self.config.gamma,
                        gae_lambda=self.config.gae_lambda,
                    )
                    # TODO: Signal CRD that rollout is ready
                    # For now, reset and continue
                    self.rollout.reset()

                # 12. Restart Dolphin periodically
                if self.frames_since_restart >= self.config.restart_interval:
                    self._restart_dolphin()

            except KeyboardInterrupt:
                print(f"[ENV {self.global_env_id}] Interrupted, shutting down")
                break
            except Exception as e:
                print(f"[ENV {self.global_env_id}] Error in main loop: {e}")
                import traceback

                traceback.print_exc()
                # Attempt restart
                self._restart_dolphin()

        # Cleanup
        if self.console:
            self.console.stop()
