import os
import shutil
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from libmelee.melee.console import Console
from libmelee.melee.controller import Controller
from libmelee.melee.enums import ControllerType, Character, Stage, Menu, Button
from libmelee.melee.menuhelper import MenuHelper

from config.config import init_config, get_config
from schema import get_feature_names, get_target_names
from model_interface import collect_raw_inputs_from_gamestate, ControllerState
from train.value_head import build_reward_feature_index, compute_frame_rewards
from column_map import ColumnMap
from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)

# Initialize config (needed for feature extraction logic dependent on global config)
init_config()

class BatchedDolphinEnv:
    """
    Manages N Dolphin instances within a single process.
    Presents a vectorized API: step(actions_batch) -> obs_batch, reward_batch, done_batch.
    """
    def __init__(
        self, 
        num_envs: int, 
        worker_id: int, 
        dolphin_path: str, 
        iso_path: str,
        base_port: int = 51441
    ):
        self.num_envs = num_envs
        self.worker_id = worker_id
        self.dolphin_path = dolphin_path
        self.iso_path = iso_path
        
        # Internal state
        self.consoles: List[Console] = []
        self.controllers: List[Dict[int, Controller]] = [] # [{1: Ctl, 2: Ctl}, ...]
        self.menu_helpers: List[MenuHelper] = []
        
        # Feature/Reward Setup
        self.feature_names = get_feature_names()
        self.feature_dim = len(self.feature_names)
        self.target_names = get_target_names()
        self.colmap = ColumnMap(self.feature_names, self.target_names)
        self.reward_feature_idx = build_reward_feature_index(self.colmap)
        
        # Quantization Palettes (Pre-loaded as numpy)
        self.stick_palette = np.array(CONTROL_STICK_QUANTIZED, dtype=np.float32)
        self.c_stick_palette = np.array(C_STICK_QUANTIZED, dtype=np.float32)
        self.shoulder_palette = np.array(SHOULDER_QUANTIZED, dtype=np.float32)

        # State tracking for rewards [N, F]
        self.prev_obs: Optional[np.ndarray] = None
        self.frame_counts = np.zeros(num_envs, dtype=int)
        
        # Initialize Environments
        self._init_envs(base_port)

    def _init_envs(self, base_port: int) -> None:
        """Spawns the Dolphin processes."""
        print(f"Worker {self.worker_id}: Spawning {self.num_envs} Dolphin instances...")
        
        for i in range(self.num_envs):
            # Unique Identity
            env_global_id = self.worker_id * self.num_envs + i
            user_dir = Path.cwd() / f"dolphin-home-{self.worker_id}-{i}" / "User"
            slippi_port = base_port + env_global_id
            
            # Ensure clean state
            if user_dir.exists():
                # Optional: clean up if needed, or rely on Dolphin to handle it
                pass
            user_dir.mkdir(parents=True, exist_ok=True)

            console = Console(
                path=self.dolphin_path,
                dolphin_home_path=str(user_dir),
                slippi_address="127.0.0.1",
                slippi_port=slippi_port,
                save_replays=False,
                blocking_input=True,
                gfx_backend="Null",
                disable_audio=True,
                emulation_speed=0.0, # Unlock speed
                infinite_time=True,
                use_exi_inputs=True,
                enable_ffw=True,
            )
            
            # Controllers (P1=Learner, P2=Opponent)
            ctls = {}
            for port in [1, 2]:
                ctls[port] = Controller(console, port, ControllerType.STANDARD)

            # Start
            console.run(iso_path=self.iso_path)
            if not console.connect():
                raise RuntimeError(f"Failed to connect to Dolphin (Worker {self.worker_id}, Env {i})")
            
            for c in ctls.values():
                if not c.connect():
                     raise RuntimeError(f"Failed to connect Controller (Worker {self.worker_id}, Env {i})")
            
            self.consoles.append(console)
            self.controllers.append(ctls)
            self.menu_helpers.append(MenuHelper())
            
        print(f"Worker {self.worker_id}: All {self.num_envs} instances ready.")
        
        # Initialize buffer
        self.prev_obs = np.zeros((self.num_envs, self.feature_dim), dtype=np.float32)

    def step(self, p1_actions: Dict[str, np.ndarray], p2_actions: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advance all environments by 1 frame.
        
        Args:
            p1_actions: Dict of batched actions [N, ...]
            p2_actions: Dict of batched actions [N, ...]
            
        Returns:
            obs: [N, F]
            rewards: [N]
            dones: [N]
        """
        curr_obs_list = []
        
        # 1. Apply Actions & Step
        for i in range(self.num_envs):
            # Extract action for this env
            a1 = self._extract_action(p1_actions, i)
            a2 = self._extract_action(p2_actions, i)
            
            # Apply
            self._apply_controller(self.controllers[i][1], a1)
            self._apply_controller(self.controllers[i][2], a2)
            
            # Step
            # Note: This blocks until frame is received. 
            # In a single thread, this serializes the waiting time.
            gamestate = self.consoles[i].step()
            
            # Handle Menus / Reset
            if gamestate is None or gamestate.menu_state not in [Menu.IN_GAME, Menu.SUDDEN_DEATH]:
                self._handle_menu(i, gamestate)
                # If we are in menu, we might yield zeros or the last frame. 
                # For PPO, usually we want to pause or reset. 
                # Here we just push zeros if not in game, or try to get back in game.
                # A simple strategy: return zeros, reward 0, done=True (to reset LSTM state)
                features = np.zeros(self.feature_dim, dtype=np.float32)
            else:
                # Extract Features
                raw = collect_raw_inputs_from_gamestate(gamestate, 1, 2)
                features = self._dict_to_array(raw.transformed)
                self.frame_counts[i] += 1
                
            curr_obs_list.append(features)

        # 2. Aggregate
        curr_obs = np.stack(curr_obs_list) # [N, F]
        
        # 3. Compute Rewards (Vectorized)
        # Convert to torch for the shared reward function
        prev_t = torch.from_numpy(self.prev_obs).unsqueeze(1) # [N, 1, F]
        curr_t = torch.from_numpy(curr_obs).unsqueeze(1)      # [N, 1, F]
        stack = torch.cat([prev_t, curr_t], dim=1)            # [N, 2, F]
        
        with torch.no_grad():
            # compute_frame_rewards returns [N, L]. L=2. 
            # Reward for transition 0->1 is stored at index 0.
            # Index 1 is zero because there is no future frame.
            full_rewards = compute_frame_rewards(stack, self.reward_feature_idx) # [N, 2]
            rewards = full_rewards[:, 0].numpy() # [N]
            
        # 4. Update State
        self.prev_obs = curr_obs.copy()
        
        # 5. Dones (Infinite Time, so handled by rollout length usually, but check menu)
        # For this simple version, we rely on the coordinator to slice trajectories.
        # We return done=False unless we actually reset.
        dones = np.zeros(self.num_envs, dtype=bool) 
        
        return curr_obs, rewards, dones

    def _extract_action(self, action_dict: Dict[str, np.ndarray], idx: int) -> Dict:
        """Slices the batch dict to get a single action dict."""
        return {k: v[idx] for k, v in action_dict.items()}

    def _apply_controller(self, controller: Controller, action: Dict) -> None:
        """Applies a single action dict to a controller."""
        controller.release_all()
        
        # Sticks
        main_idx = action['main_stick']
        c_idx = action['c_stick']
        
        mx, my = self.stick_palette[main_idx]
        cx, cy = self.c_stick_palette[c_idx]
        
        # Remap -1..1 to 0..1 for libmelee if needed, or does libmelee take -1..1?
        # Controller.tilt_analog takes 0..1
        mx = mx * 0.5 + 0.5
        my = my * 0.5 + 0.5
        cx = cx * 0.5 + 0.5
        cy = cy * 0.5 + 0.5
        
        controller.tilt_analog(Button.BUTTON_MAIN, mx, my)
        controller.tilt_analog(Button.BUTTON_C, cx, cy)
        
        # Shoulder
        s_idx = action['shoulder']
        s_val = self.shoulder_palette[s_idx]
        controller.press_shoulder(Button.BUTTON_L, s_val)
        
        # Buttons (Bool array)
        # [A, B, X, Z, L]
        btns = action['buttons']
        if btns[0]: controller.press_button(Button.BUTTON_A)
        if btns[1]: controller.press_button(Button.BUTTON_B)
        if btns[2]: controller.press_button(Button.BUTTON_X)
        if btns[3]: controller.press_button(Button.BUTTON_Z)
        if btns[4]: controller.press_button(Button.BUTTON_L)

    def _handle_menu(self, idx: int, gamestate) -> None:
        """Navigates menus if needed."""
        # P1
        self.menu_helpers[idx].menu_helper_simple(
            gamestate, self.controllers[idx][1], 
            Character.FOX, Stage.FINAL_DESTINATION, 
            costume=1, autostart=False, swag=False
        )
        # P2
        self.menu_helpers[idx].choose_character(
            Character.FOX, gamestate, self.controllers[idx][2],
            cpu_level=0, costume=2, swag=False, start=True
        )

    def _dict_to_array(self, feature_dict: Dict[str, float]) -> np.ndarray:
        """Fast conversion of dict to numpy array using schema order."""
        arr = np.zeros(self.feature_dim, dtype=np.float32)
        for i, name in enumerate(self.feature_names):
            arr[i] = feature_dict.get(name, 0.0)
        return arr

    def close(self) -> None:
        """Cleanup."""
        for c in self.consoles:
            c.stop()
        self.consoles = []
