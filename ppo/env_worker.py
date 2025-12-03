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
from libmelee.melee.menuhelper import MenuHelper
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
        rollout_queue,  # Queue to send completed rollouts
        config: PPOConfig,
        column_map: ColumnMap,
    ):
        self.shard_id = shard_id
        self.env_id = env_id
        self.global_env_id = shard_id * config.envs_per_shard + env_id
        self.slab = shared_slab
        self.rollout_queue = rollout_queue
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
            feature_dim=config.feature_dim,
        )

        # Restart tracking
        self.frames_since_restart = 0

        # Feature names for featurization
        self.feature_names = get_feature_names()

        # Reward computation
        self.reward_idx = build_reward_feature_index(self.column_map)

        # Reward tracking state
        self.prev_ego_percent = 0.0
        self.prev_opp_percent = 0.0
        self.prev_ego_stock = 4
        self.prev_opp_stock = 4

        # Stage selection
        self.current_stage = self._select_stage()

        # Menu navigation
        self.menu_helper = MenuHelper()
        self.was_in_menu = True  # Track menu state transitions

    def _select_stage(self) -> enums.Stage:
        """Select random stage from tournament legal pool."""
        stage_abbrev = random.choice(self.config.stages)
        return STAGE_MAP[stage_abbrev]

    def _start_dolphin(self):
        """Initialize Dolphin subprocess and controllers."""
        # Create console with optimal settings for headless AI training
        self.console = Console(
            path=self.config.dolphin_path,
            slippi_port=51441 + self.global_env_id,  # Unique port per env
            logger=None,  # Suppress logging for now
            # Headless settings (no graphics/audio)
            gfx_backend="Null",  # No rendering
            disable_audio=True,  # No sound
            fullscreen=False,  # Don't fullscreen
            # Performance settings
            blocking_input=True,  # Wait for AI inputs (precise control)
            online_delay=0,  # No input delay
            save_replays=False,  # Don't save replays (saves disk space)
            emulation_speed=1.0,  # Normal speed (can set to 0 for unlimited)
            # Dolphin configuration
            tmp_home_directory=True,  # Isolated dolphin config per instance
            copy_home_directory=False,  # Fresh directory
            setup_gecko_codes=True,  # Setup necessary gecko codes
        )

        # Start Dolphin
        self.console.run(
            iso_path=str(Path(self.config.iso_path).expanduser()),
        )

        # Create controllers
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

        # Connect controllers (CRITICAL - must be after console.connect())
        if not self.ego_controller.connect():
            raise RuntimeError(
                f"ENV {self.global_env_id}: Failed to connect ego controller"
            )
        if not self.opp_controller.connect():
            raise RuntimeError(
                f"ENV {self.global_env_id}: Failed to connect opp controller"
            )

        print(
            f"[ENV {self.global_env_id}] Dolphin started: "
            f"{self.config.character} vs {self.config.character} on {self.current_stage.name}"
        )
        print(
            f"[ENV {self.global_env_id}] Controllers connected on ports "
            f"{self.config.bot_port} and {self.config.opp_port}"
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
        Extract feature_dim-float feature vector from gamestate.

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
        Compute zero-sum reward for current frame based on damage and stocks.

        Reward structure:
        - Damage dealt to opponent: +damage_delta
        - Damage taken: -damage_delta
        - Stock taken from opponent: +1.0
        - Stock lost: -1.0

        Returns: ego_reward - opp_reward (zero-sum)
        """
        ego = gamestate.players.get(self.config.bot_port)
        opp = gamestate.players.get(self.config.opp_port)

        if ego is None or opp is None:
            return 0.0

        reward = 0.0

        # Damage rewards (normalized by 100 for scale)
        ego_damage_delta = ego.percent - self.prev_ego_percent
        opp_damage_delta = opp.percent - self.prev_opp_percent

        # Positive reward for damaging opponent, negative for taking damage
        reward += (opp_damage_delta - ego_damage_delta) / 100.0

        # Stock rewards (large bonus/penalty)
        ego_stock_delta = ego.stock - self.prev_ego_stock
        opp_stock_delta = opp.stock - self.prev_opp_stock

        # Positive reward for taking opponent's stock, negative for losing stock
        reward += (self.prev_opp_stock - opp.stock) * 1.0  # Opponent lost stock (ego took it)
        reward -= (self.prev_ego_stock - ego.stock) * 1.0  # Ego lost stock

        # Update tracking state
        self.prev_ego_percent = ego.percent
        self.prev_opp_percent = opp.percent
        self.prev_ego_stock = ego.stock
        self.prev_opp_stock = opp.stock

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
        from controller_utils import (
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

        # Apply buttons FIRST (before setting analogs)
        # Release all button presses from previous frame
        controller.release_button(enums.Button.BUTTON_A)
        controller.release_button(enums.Button.BUTTON_B)
        controller.release_button(enums.Button.BUTTON_X)
        controller.release_button(enums.Button.BUTTON_Y)
        controller.release_button(enums.Button.BUTTON_Z)

        # Press buttons for this frame
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

        # Apply analog inputs (AFTER buttons to avoid being cleared)
        controller.tilt_analog(enums.Button.BUTTON_MAIN, main_01_x, main_01_y)
        controller.tilt_analog(enums.Button.BUTTON_C, c_01_x, c_01_y)
        controller.press_shoulder(enums.Button.BUTTON_L, shoulder_val)

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

                # 2. Handle menu navigation (if not in game)
                if gamestate.menu_state not in [enums.Menu.IN_GAME, enums.Menu.SUDDEN_DEATH]:
                    # Log menu state periodically to debug
                    menu_frame_counter = getattr(self, '_menu_frame_counter', 0)
                    if menu_frame_counter % 120 == 0:  # Every 2 seconds at 60fps
                        print(
                            f"[ENV {self.env_id}] Still in menus (frame {menu_frame_counter}): "
                            f"menu_state={gamestate.menu_state}, "
                            f"submenu={gamestate.submenu}, "
                            f"menu_selection={gamestate.menu_selection}, "
                            f"frame={gamestate.frame}"
                        )
                    self._menu_frame_counter = menu_frame_counter + 1

                    # Handle PRESS_START screen explicitly (menu_helper_simple doesn't handle it)
                    if gamestate.menu_state == enums.Menu.PRESS_START:
                        # Press START on both controllers to get past title screen
                        if gamestate.frame % 2 == 1:  # Only press on odd frames
                            self.ego_controller.press_button(enums.Button.BUTTON_START)
                            self.opp_controller.press_button(enums.Button.BUTTON_START)
                        else:
                            self.ego_controller.release_all()
                            self.opp_controller.release_all()
                    elif gamestate.menu_state == enums.Menu.MAIN_MENU:
                        # Handle stuck in ONLINE_PLAY_SUBMENU - back out with B
                        if gamestate.submenu == enums.SubMenu.ONLINE_PLAY_SUBMENU:
                            if gamestate.frame % 2 == 1:
                                self.ego_controller.press_button(enums.Button.BUTTON_B)
                                self.opp_controller.press_button(enums.Button.BUTTON_B)
                            else:
                                self.ego_controller.release_all()
                                self.opp_controller.release_all()
                        else:
                            # Use normal menu helper for other submenus
                            self.menu_helper.menu_helper_simple(
                                gamestate,
                                self.ego_controller,
                                CHARACTER_MAP[self.config.character],
                                self.current_stage,
                                costume=1,
                                autostart=True,
                                swag=False,
                            )
                            self.menu_helper.menu_helper_simple(
                                gamestate,
                                self.opp_controller,
                                CHARACTER_MAP[self.config.character],
                                self.current_stage,
                                costume=2,
                                autostart=True,
                                swag=False,
                            )
                    else:
                        # Navigate menus for both players (character select, stage select, etc)
                        self.menu_helper.menu_helper_simple(
                            gamestate,
                            self.ego_controller,
                            CHARACTER_MAP[self.config.character],
                            self.current_stage,
                            costume=1,
                            autostart=True,  # Auto-start since both are bots
                            swag=False,
                        )
                        self.menu_helper.menu_helper_simple(
                            gamestate,
                            self.opp_controller,
                            CHARACTER_MAP[self.config.character],
                            self.current_stage,
                            costume=2,
                            autostart=True,  # Auto-start since both are bots
                            swag=False,
                        )

                    self.was_in_menu = True
                    continue  # Skip to next frame during menu navigation (DON'T signal READY)

                # 3. Log when match starts (transition from menu to in-game)
                if self.was_in_menu:
                    print(
                        f"[ENV {self.env_id}] Match started! "
                        f"{self.config.character} vs {self.config.character} on {self.current_stage.name}"
                    )
                    self.was_in_menu = False
                    self._menu_frame_counter = 0  # Reset menu frame counter
                    # Reset frame counter when entering match
                    self.t_local = 0
                    self.t_mod = 0
                    self.is_warm = False
                    self.rollout.reset()
                    # Reset reward tracking state
                    self.prev_ego_percent = 0.0
                    self.prev_opp_percent = 0.0
                    self.prev_ego_stock = 4
                    self.prev_opp_stock = 4

                # 4. Featurize (only in-game)
                features = self._featurize(gamestate)

                # Debug: Check if we're getting valid gamestate
                if self.t_local % 500 == 0:
                    ego_player = gamestate.players.get(self.config.bot_port)
                    opp_player = gamestate.players.get(self.config.opp_port)
                    print(
                        f"[ENV {self.env_id}] Frame {self.t_local} GAMESTATE: "
                        f"menu={gamestate.menu_state}, "
                        f"ego_exists={ego_player is not None}, "
                        f"opp_exists={opp_player is not None}"
                    )
                    if ego_player:
                        print(
                            f"[ENV {self.env_id}]   Ego raw: pos=({ego_player.position.x:.1f},{ego_player.position.y:.1f}) "
                            f"pct={ego_player.percent:.0f}% stock={ego_player.stock}"
                        )

                # 5. Write to shared ring buffer at position t_mod
                self.slab.features[self.env_id, self.t_mod, :] = features

                # 6. Mark ready for CRD
                self.slab.ready_flags[self.env_id] = 1

                # 7. Wait for CRD to compute action (busy-wait)
                while self.slab.ready_flags[self.env_id] == 1:
                    time.sleep(0.0001)  # Brief sleep to yield CPU

                # 8. Read actions from shared slot
                ego_action = self.slab.read_action(self.env_id, is_ego=True)
                opp_action = self.slab.read_action(self.env_id, is_ego=False)

                # 9. Apply actions to controllers
                if not self.is_warm:
                    self._apply_neutral_action()
                else:
                    self._apply_action(ego_action, is_ego=True)
                    self._apply_action(opp_action, is_ego=False)

                # 10. Compute reward
                reward = self._compute_reward(gamestate, features)

                # 11. Store in rollout buffer
                self._store_rollout_frame(features, ego_action, reward)

                # 12. Update ring position
                self.t_mod = (self.t_mod + 1) % self.config.context_length
                self.t_local += 1
                self.frames_since_restart += 1

                # Mark warm after warmup period
                if self.t_local >= self.config.warmup_frames:
                    self.is_warm = True

                # Periodic logging (every 500 frames)
                if self.t_local % 500 == 0:
                    ego_player = gamestate.players.get(self.config.bot_port)
                    opp_player = gamestate.players.get(self.config.opp_port)
                    if ego_player and opp_player:
                        print(
                            f"[ENV {self.env_id}] Frame {self.t_local}: "
                            f"Ego: pos=({ego_player.position.x:.1f},{ego_player.position.y:.1f}) "
                            f"pct={ego_player.percent:.0f}% stock={ego_player.stock} | "
                            f"Opp: pos=({opp_player.position.x:.1f},{opp_player.position.y:.1f}) "
                            f"pct={opp_player.percent:.0f}% stock={opp_player.stock} | "
                            f"Action: main={ego_action.main_idx} c={ego_action.c_idx} "
                            f"btns={ego_action.buttons} | warm={self.is_warm}"
                        )

                # 13. Check if rollout complete
                if self.is_warm and self.rollout.complete:
                    # Compute advantages
                    self.rollout.compute_advantages(
                        gamma=self.config.gamma,
                        gae_lambda=self.config.gae_lambda,
                    )

                    # Send completed rollout to shard for coordinator collection
                    print(
                        f"[ENV {self.env_id}] Rollout complete "
                        f"({self.rollout.mask.sum()}/{self.rollout.rollout_length} valid frames), "
                        f"sending to coordinator"
                    )
                    self.rollout_queue.put(self.rollout)

                    # Create new rollout buffer for next collection
                    self.rollout = RolloutBuffer(
                        rollout_length=self.config.rollout_length,
                        feature_dim=self.config.feature_dim,
                    )

                # 14. Restart Dolphin periodically
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
