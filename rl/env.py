"""
MeleeEnv: Gym-like environment wrapper for Super Smash Bros. Melee.

This environment provides a standard RL interface for training agents using libmelee.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Tuple, Optional

from libmelee.melee.console import Console
from libmelee.melee.controller import Controller
from libmelee.melee.enums import Character, ControllerType, Menu, Stage
from libmelee.melee.gamestate import GameState
from libmelee.melee.menuhelper import MenuHelper

from config import Config, get_config
from model_interface import collect_raw_inputs_from_gamestate, ModelFrameInputs


# Legal tournament stages for competitive play
LEGAL_STAGES = [
    Stage.BATTLEFIELD,
    Stage.YOSHIS_STORY,
    Stage.POKEMON_STADIUM,
    Stage.DREAMLAND,
    Stage.FINAL_DESTINATION,
    Stage.FOUNTAIN_OF_DREAMS,
]


class MeleeEnv:
    """
    Gym-like environment for Super Smash Bros. Melee RL training.
    
    The environment manages a Dolphin instance and provides a standard interface:
    - reset(): Start a new game
    - step(action): Apply action and advance one frame
    - close(): Clean up resources
    """

    def __init__(
        self,
        iso_path: str,
        dolphin_path: str,
        agent_port: int = 1,
        opponent_port: int = 2,
        character: Character = Character.FOX,
        opponent_character: Character = Character.FOX,
        stage: Optional[Stage] = None,
        render: bool = False,
        save_replays: bool = False,
    ):
        """
        Initialize the Melee environment.

        Args:
            iso_path: Path to Melee ISO file
            dolphin_path: Path to Dolphin executable
            agent_port: Controller port for the agent (1 or 2)
            opponent_port: Controller port for the opponent (1 or 2)
            character: Character for the agent
            opponent_character: Character for the opponent
            stage: Stage to play on (None for random legal stage)
            render: Whether to render graphics (False for headless)
            save_replays: Whether to save replays
        """
        self.iso_path = iso_path
        self.dolphin_path = dolphin_path
        self.agent_port = agent_port
        self.opponent_port = opponent_port
        self.character = character
        self.opponent_character = opponent_character
        self.stage = stage
        self.render = render
        self.save_replays = save_replays

        # Initialize console
        self.console = Console(
            path=dolphin_path,
            slippi_address="127.0.0.1",
            save_replays=save_replays,
            copy_home_directory=False,
            tmp_home_directory=False,
            blocking_input=True,
            gfx_backend="Null" if not render else None,
            disable_audio=not render,
        )

        # Initialize controllers
        self.controllers = {
            agent_port: Controller(
                console=self.console,
                port=agent_port,
                type=ControllerType.STANDARD,
            ),
            opponent_port: Controller(
                console=self.console,
                port=opponent_port,
                type=ControllerType.STANDARD,
            ),
        }

        self.menu_helper = MenuHelper()
        self.current_stage: Optional[Stage] = None
        self.gamestate: Optional[GameState] = None
        self.previous_gamestate: Optional[GameState] = None
        
        # Episode tracking
        self.episode_frames = 0
        self.episode_reward = 0.0
        
        # Previous state for reward calculation
        self.prev_agent_percent: float = 0.0
        self.prev_opponent_percent: float = 0.0
        self.prev_agent_stock: int = 4
        self.prev_opponent_stock: int = 4

        # Start console
        self.console.run(iso_path=iso_path)
        if not self.console.connect():
            raise RuntimeError("Failed to connect to Dolphin console")
        
        for controller in self.controllers.values():
            if not controller.connect():
                raise RuntimeError("Failed to connect controller")

    def reset(self) -> Tuple[Dict[str, float], Dict]:
        """
        Reset the environment for a new episode.

        Returns:
            observation: Dict of feature values
            info: Additional information
        """
        # Select random stage if not specified
        if self.stage is None:
            self.current_stage = random.choice(LEGAL_STAGES)
        else:
            self.current_stage = self.stage

        # Reset episode tracking
        self.episode_frames = 0
        self.episode_reward = 0.0
        self.prev_agent_percent = 0.0
        self.prev_opponent_percent = 0.0
        self.prev_agent_stock = 4
        self.prev_opponent_stock = 4
        self.previous_gamestate = None

        # Wait for game to start
        while True:
            self.gamestate = self.console.step()
            if self.gamestate is None:
                continue

            if self.gamestate.menu_state in [Menu.IN_GAME, Menu.SUDDEN_DEATH]:
                break

            # Navigate menus
            self.menu_helper.menu_helper_simple(
                self.gamestate,
                self.controllers[self.agent_port],
                self.character,
                self.current_stage,
                costume=0,
                autostart=False,
                swag=False,
            )
            
            self.menu_helper.choose_character(
                character=self.opponent_character,
                gamestate=self.gamestate,
                controller=self.controllers[self.opponent_port],
                cpu_level=9,  # Will be overridden by policy in actual training
                costume=1,
                swag=False,
                start=True,
            )

        # Get initial observation
        obs = self._get_observation()
        info = {
            "stage": self.current_stage.name if self.current_stage else "UNKNOWN",
            "episode_frames": 0,
        }

        return obs, info

    def step(
        self, action: Dict[str, any]
    ) -> Tuple[Dict[str, float], float, bool, bool, Dict]:
        """
        Take one step in the environment.

        Args:
            action: Dictionary with action components:
                - main_stick_idx: int (index into quantized grid)
                - c_stick_idx: int (index into quantized grid)
                - buttons: Dict[str, bool] (button states)
                - shoulder: float (analog trigger value)

        Returns:
            observation: Dict of feature values
            reward: Scalar reward
            terminated: Whether episode ended naturally (stock out)
            truncated: Whether episode was truncated (time limit, etc.)
            info: Additional information
        """
        # Apply action to agent controller (action application will be handled by caller)
        # For now, we just step the console
        self.previous_gamestate = self.gamestate
        self.gamestate = self.console.step()

        if self.gamestate is None:
            # Return safe defaults if gamestate is None
            return self._get_observation(), 0.0, False, False, {}

        self.episode_frames += 1

        # Check if we're still in game
        in_game = self.gamestate.menu_state in [Menu.IN_GAME, Menu.SUDDEN_DEATH]
        
        if not in_game:
            # Episode ended, return to menu
            obs = self._get_observation()
            reward = 0.0
            terminated = True
            truncated = False
            info = {
                "episode_frames": self.episode_frames,
                "episode_reward": self.episode_reward,
            }
            return obs, reward, terminated, truncated, info

        # Compute reward
        reward = self._compute_reward()
        self.episode_reward += reward

        # Check termination conditions
        agent_player = self.gamestate.players.get(self.agent_port)
        opponent_player = self.gamestate.players.get(self.opponent_port)
        
        terminated = False
        if agent_player and opponent_player:
            agent_stock = getattr(agent_player, "stock", 0)
            opponent_stock = getattr(opponent_player, "stock", 0)
            
            # Episode ends when either player runs out of stocks
            if agent_stock == 0 or opponent_stock == 0:
                terminated = True

        # Get observation
        obs = self._get_observation()

        info = {
            "episode_frames": self.episode_frames,
            "agent_stock": agent_stock if agent_player else 0,
            "opponent_stock": opponent_stock if opponent_player else 0,
        }

        return obs, reward, terminated, False, info

    def _get_observation(self) -> Dict[str, float]:
        """
        Extract observation from current gamestate.

        Returns:
            Dictionary of feature values suitable for the model
        """
        if self.gamestate is None:
            # Return zero observation if no gamestate
            from schema import get_feature_names
            return {name: 0.0 for name in get_feature_names()}

        # Use existing model_interface function to extract features
        frame_inputs = collect_raw_inputs_from_gamestate(
            self.gamestate,
            bot_port=self.agent_port,
            opp_port=self.opponent_port,
        )
        
        # Return the transformed features as dict
        return dict(frame_inputs.transformed)

    def _compute_reward(self) -> float:
        """
        Compute reward based on game state changes.

        Reward components:
        - Damage dealt to opponent: +0.01 per percent
        - Damage taken: -0.01 per percent
        - Stock taken: +1.0
        - Stock lost: -1.0
        - Small time penalty: -0.001 per frame (encourages aggression)
        """
        if self.gamestate is None or self.previous_gamestate is None:
            return 0.0

        agent_player = self.gamestate.players.get(self.agent_port)
        opponent_player = self.gamestate.players.get(self.opponent_port)

        if not agent_player or not opponent_player:
            return 0.0

        # Get current values
        agent_percent = float(getattr(agent_player, "percent", 0))
        opponent_percent = float(getattr(opponent_player, "percent", 0))
        agent_stock = int(getattr(agent_player, "stock", 0))
        opponent_stock = int(getattr(opponent_player, "stock", 0))

        reward = 0.0

        # Damage rewards (opponent damage dealt is good, agent damage taken is bad)
        damage_dealt = opponent_percent - self.prev_opponent_percent
        damage_taken = agent_percent - self.prev_agent_percent
        
        reward += damage_dealt * 0.01
        reward -= damage_taken * 0.01

        # Stock rewards
        stocks_taken = self.prev_opponent_stock - opponent_stock
        stocks_lost = self.prev_agent_stock - agent_stock
        
        reward += stocks_taken * 1.0
        reward -= stocks_lost * 1.0

        # Small time penalty to encourage aggression
        reward -= 0.001

        # Update previous values
        self.prev_agent_percent = agent_percent
        self.prev_opponent_percent = opponent_percent
        self.prev_agent_stock = agent_stock
        self.prev_opponent_stock = opponent_stock

        return reward

    def close(self):
        """Clean up environment resources."""
        for controller in self.controllers.values():
            controller.disconnect()
        self.console.stop()
