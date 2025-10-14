#!/usr/bin/env python3
"""Self-play PPO training script for Melee AI.

This script trains a model using Proximal Policy Optimization (PPO) with self-play.
The model plays against itself or past versions of itself, collecting experience
and updating via PPO objectives.
"""

import argparse
import json
import math
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from column_map import ColumnMap
from config import get_config, init_config
from controller_quantization import quantize_targets
from libmelee.melee.console import Console
from libmelee.melee.controller import Controller
from libmelee.melee.enums import Character, ControllerType, Menu, Stage
from libmelee.melee.gamestate import GameState
from libmelee.melee.menuhelper import MenuHelper
from model.gpt import GPTv7
from model_interface import (
    ControllerState,
    GPTInferenceEngine,
    apply_model_outputs_to_game,
    collect_raw_inputs_from_gamestate,
)
from controller_utils import CONTROL_STICK_QUANTIZED, C_STICK_QUANTIZED, SHOULDER_QUANTIZED
from train import (
    build_inputs_for_gptv7,
    build_reward_feature_index,
    compute_frame_rewards,
    _resolve_device,
)
from utils import print_model_diagram


@dataclass
class Experience:
    """Single timestep of experience from self-play."""
    observation: torch.Tensor  # [F] - raw features
    action_main_stick: int  # predicted main stick class
    action_c_stick: int  # predicted c-stick class
    action_buttons: torch.Tensor  # [K_buttons] - binary
    action_shoulder: int  # predicted shoulder class
    log_prob: float  # log probability of taken action
    value: float  # value estimate at this state
    reward: float  # reward received (computed later)
    done: bool  # episode termination flag


@dataclass
class GameEpisode:
    """Complete episode of gameplay."""
    experiences: List[Experience]
    total_reward: float
    episode_length: int
    winner_port: int
    final_stocks: Dict[int, int]


@dataclass
class PPOBatch:
    """Batch of experiences formatted for PPO training."""
    observations: torch.Tensor  # [B, L, F]
    actions_main: torch.Tensor  # [B, L]
    actions_c: torch.Tensor  # [B, L]
    actions_buttons: torch.Tensor  # [B, L, K_buttons]
    actions_shoulder: torch.Tensor  # [B, L]
    old_log_probs: torch.Tensor  # [B, L]
    advantages: torch.Tensor  # [B, L]
    returns: torch.Tensor  # [B, L]
    old_values: torch.Tensor  # [B, L]


class ExperienceBuffer:
    """Circular buffer for collecting self-play experiences."""
    
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.episodes: deque = deque(maxlen=max_size)
        
    def add_episode(self, episode: GameEpisode):
        """Add a complete episode to the buffer."""
        self.episodes.append(episode)
        
    def sample_batch(self, batch_size: int, seq_len: int) -> Optional[PPOBatch]:
        """Sample a batch of experiences for training."""
        if len(self.episodes) == 0:
            return None
            
        # Flatten all experiences
        all_experiences = []
        for episode in self.episodes:
            all_experiences.extend(episode.experiences)
            
        if len(all_experiences) < seq_len:
            return None
            
        # Sample random windows
        sampled_sequences = []
        for _ in range(batch_size):
            max_start = len(all_experiences) - seq_len
            start_idx = np.random.randint(0, max_start + 1)
            sequence = all_experiences[start_idx:start_idx + seq_len]
            sampled_sequences.append(sequence)
            
        # Convert to PPOBatch
        return self._sequences_to_batch(sampled_sequences)
    
    def _sequences_to_batch(self, sequences: List[List[Experience]]) -> PPOBatch:
        """Convert list of sequences to a batched tensor format."""
        B = len(sequences)
        L = len(sequences[0])
        F = sequences[0][0].observation.shape[0]
        K_buttons = sequences[0][0].action_buttons.shape[0]
        
        obs = torch.stack([
            torch.stack([exp.observation for exp in seq])
            for seq in sequences
        ])  # [B, L, F]
        
        actions_main = torch.tensor([
            [exp.action_main_stick for exp in seq]
            for seq in sequences
        ], dtype=torch.long)
        
        actions_c = torch.tensor([
            [exp.action_c_stick for exp in seq]
            for seq in sequences
        ], dtype=torch.long)
        
        actions_buttons = torch.stack([
            torch.stack([exp.action_buttons for exp in seq])
            for seq in sequences
        ])  # [B, L, K_buttons]
        
        actions_shoulder = torch.tensor([
            [exp.action_shoulder for exp in seq]
            for seq in sequences
        ], dtype=torch.long)
        
        old_log_probs = torch.tensor([
            [exp.log_prob for exp in seq]
            for seq in sequences
        ], dtype=torch.float32)
        
        old_values = torch.tensor([
            [exp.value for exp in seq]
            for seq in sequences
        ], dtype=torch.float32)
        
        rewards = torch.tensor([
            [exp.reward for exp in seq]
            for seq in sequences
        ], dtype=torch.float32)
        
        # Compute returns and advantages using GAE
        returns, advantages = self._compute_gae(rewards, old_values)
        
        return PPOBatch(
            observations=obs,
            actions_main=actions_main,
            actions_c=actions_c,
            actions_buttons=actions_buttons,
            actions_shoulder=actions_shoulder,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            old_values=old_values,
        )
    
    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation (GAE)."""
        config = get_config()
        gamma = config.rl.gamma
        gae_lambda = config.rl.gae_lambda
        
        B, L = rewards.shape
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Compute advantages using GAE
        gae = 0
        for t in reversed(range(L)):
            if t == L - 1:
                next_value = 0  # Terminal state
            else:
                next_value = values[:, t + 1]
                
            delta = rewards[:, t] + gamma * next_value - values[:, t]
            gae = delta + gamma * gae_lambda * gae
            advantages[:, t] = gae
            
        # Returns are advantages + values
        returns = advantages + values
        
        # Normalize advantages (per batch)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def clear(self):
        """Clear all episodes from buffer."""
        self.episodes.clear()
        
    def __len__(self) -> int:
        return len(self.episodes)


def experience_to_controller_state(exp: Experience, button_thresholds: Optional[Dict[str, float]] = None) -> ControllerState:
    """Convert Experience actions to ControllerState for controller application.
    
    Args:
        exp: Experience object with action predictions
        button_thresholds: Optional dict mapping button names to threshold values.
                          If None, uses default 0.5 for all buttons.
    """
    # Dequantize stick positions
    main_x, main_y = CONTROL_STICK_QUANTIZED[exp.action_main_stick]
    c_x, c_y = C_STICK_QUANTIZED[exp.action_c_stick]
    shoulder = SHOULDER_QUANTIZED[exp.action_shoulder]
    
    # Convert to controller range [0, 1] from [-1, 1]
    main_x = (main_x + 1.0) / 2.0
    main_y = (main_y + 1.0) / 2.0
    c_x = (c_x + 1.0) / 2.0
    c_y = (c_y + 1.0) / 2.0
    
    # Get thresholds (default to 0.5 if not provided)
    if button_thresholds is None:
        button_thresholds = {
            "button_a": 0.5,
            "button_b": 0.5,
            "button_xy": 0.5,
            "button_z": 0.5,
            "button_lr": 0.5,
        }
    
    # Extract button states using appropriate thresholds
    # Order: A, B, X/Y, Z, L/R
    buttons = exp.action_buttons.cpu().numpy() if isinstance(exp.action_buttons, torch.Tensor) else exp.action_buttons
    button_a = bool(buttons[0] > button_thresholds["button_a"]) if len(buttons) > 0 else False
    button_b = bool(buttons[1] > button_thresholds["button_b"]) if len(buttons) > 1 else False
    button_xy = bool(buttons[2] > button_thresholds["button_xy"]) if len(buttons) > 2 else False
    button_z = bool(buttons[3] > button_thresholds["button_z"]) if len(buttons) > 3 else False
    button_lr = bool(buttons[4] > button_thresholds["button_lr"]) if len(buttons) > 4 else False
    
    return ControllerState(
        main_stick_x=float(main_x),
        main_stick_y=float(main_y),
        c_stick_x=float(c_x),
        c_stick_y=float(c_y),
        shoulder_analog=float(shoulder),
        button_a=button_a,
        button_b=button_b,
        button_xy=button_xy,
        button_lr=button_lr,
        button_z=button_z,
    )


class SelfPlayPPOTrainer:
    """Main trainer for self-play PPO."""
    
    def __init__(
        self,
        model: GPTv7,
        device: torch.device,
        colmap: ColumnMap,
        checkpoint_dir: Path,
        button_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.colmap = colmap
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.button_thresholds = button_thresholds
        
        config = get_config()
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.rl.learning_rate,
            betas=config.train.betas,
            weight_decay=config.train.weight_decay,
        )
        
        # GradScaler for AMP
        from torch import GradScaler
        self.scaler = GradScaler(device='cuda', enabled=config.train.use_amp)
        
        # Create a second model instance for player 2 (opponent)
        # This will be loaded from checkpoints during self-play
        self.model_p2 = None
        self.started_from_checkpoint = False  # Track if we loaded an initial checkpoint
        
        self.experience_buffer = ExperienceBuffer(max_size=config.rl.buffer_size)
        self.opponent_checkpoints = deque(maxlen=config.rl.opponent_pool_size)
        
        # Separate experience tracking for each player
        self.episode_rewards_p1 = []
        self.episode_rewards_p2 = []
        self.episode_lengths = []
        self.win_rate_p1 = 0.0
        
        # Training step counter
        self.global_step = 0
        
    def collect_episode(
        self,
        console: Console,
        controllers: Dict[int, Controller],
    ) -> Tuple[GameEpisode, GameEpisode]:
        """Collect a single episode of self-play experience from both players.
        
        Returns:
            Tuple of (player1_episode, player2_episode)
        """
        experiences_p1 = []
        experiences_p2 = []
        menu_helper = MenuHelper()
        episode_reward_p1 = 0.0
        episode_reward_p2 = 0.0
        frame_count = 0
        game_started = False
        
        # Track previous gamestate for reward computation
        previous_gamestate = None
        warmup_frames = 0
        WARMUP_THRESHOLD = 256  # Buffer frames before using model
        
        print("Starting self-play episode collection (both players)...")
        print("Navigating menus...")
        
        while True:
            gamestate = console.step()
            if gamestate is None:
                continue
                
            # Check if we're in game
            if gamestate.menu_state in [Menu.IN_GAME, Menu.SUDDEN_DEATH]:
                if not game_started:
                    print("Game started! Beginning warmup phase...")
                    print(f"Warmup: {WARMUP_THRESHOLD} frames before model control")
                    game_started = True
                    
                warmup_frames += 1
                
                # During warmup, release controllers (let game AI take over briefly)
                if warmup_frames < WARMUP_THRESHOLD:
                    controllers[1].release_all()
                    controllers[2].release_all()
                
                # Collect experience from both players (only after warmup)
                if warmup_frames >= WARMUP_THRESHOLD:
                    if warmup_frames == WARMUP_THRESHOLD:
                        print("Warmup complete! Models now controlling both players...")
                    
                    # Collect inputs for both players
                    raw_inputs_p1 = collect_raw_inputs_from_gamestate(
                        gamestate, 1, 2  # port, opponent_port
                    )
                    raw_inputs_p2 = collect_raw_inputs_from_gamestate(
                        gamestate, 2, 1  # port, opponent_port
                    )
                    
                    # Get model predictions for both players
                    with torch.no_grad():
                        # Check if we can batch both players together (same model)
                        model_for_p2 = self.model_p2 if self.model_p2 is not None else self.model
                        
                        if model_for_p2 is self.model:
                            # Both players use same model - batch inference for 2x speedup
                            exp_p1, exp_p2 = self._predict_and_collect_experience_batched(
                                raw_inputs_p1, raw_inputs_p2, self.model
                            )
                        else:
                            # Different models - must run separately
                            exp_p1 = self._predict_and_collect_experience(raw_inputs_p1, model=self.model)
                            exp_p2 = self._predict_and_collect_experience(raw_inputs_p2, model=model_for_p2)
                        
                    # Convert experiences to controller states and apply
                    controller_state_p1 = experience_to_controller_state(exp_p1, self.button_thresholds)
                    controller_state_p2 = experience_to_controller_state(exp_p2, self.button_thresholds)
                    apply_model_outputs_to_game(controllers[1], controller_state_p1)
                    apply_model_outputs_to_game(controllers[2], controller_state_p2)
                    
                    # Compute rewards if we have previous state
                    if previous_gamestate is not None:
                        reward_p1 = self._compute_reward(previous_gamestate, gamestate, 1)  # bot_port
                        reward_p2 = self._compute_reward(previous_gamestate, gamestate, 2)  # bot_port
                        episode_reward_p1 += reward_p1
                        episode_reward_p2 += reward_p2
                        
                        # Update the last experiences with rewards
                        if len(experiences_p1) > 0:
                            experiences_p1[-1].reward = reward_p1
                        if len(experiences_p2) > 0:
                            experiences_p2[-1].reward = reward_p2
                    
                    experiences_p1.append(exp_p1)
                    experiences_p2.append(exp_p2)
                    frame_count += 1
                    
                    # Print progress every 60 frames (1 second of game time)
                    if frame_count % 60 == 0:
                        p1_stocks = gamestate.players[1].stock if 1 in gamestate.players else 0
                        p2_stocks = gamestate.players[2].stock if 2 in gamestate.players else 0
                        p1_percent = gamestate.players[1].percent if 1 in gamestate.players else 0
                        p2_percent = gamestate.players[2].percent if 2 in gamestate.players else 0
                        print(f"  Frame {frame_count}: P1({p1_stocks}stocks, {p1_percent:.0f}%) "
                              f"P2({p2_stocks}stocks, {p2_percent:.0f}%) | "
                              f"Rewards: P1={episode_reward_p1:.2f} P2={episode_reward_p2:.2f}")
                    
                previous_gamestate = gamestate
                    
            elif game_started:
                # Game ended
                print(f"\n[Episode Complete]")
                print(f"  Total frames: {frame_count}")
                print(f"  Game duration: {frame_count/60:.1f} seconds")
                print(f"  P1 total reward: {episode_reward_p1:.2f}")
                print(f"  P2 total reward: {episode_reward_p2:.2f}")
                print(f"  Avg reward per frame - P1: {episode_reward_p1/max(frame_count, 1):.4f}, P2: {episode_reward_p2/max(frame_count, 1):.4f}")
                break
            else:
                # Navigate menus - release all buttons first, then use menu helper
                # Don't apply model outputs during menu navigation
                controllers[1].release_all()
                controllers[2].release_all()
                
                menu_helper.menu_helper_simple(
                    gamestate,
                    controllers[1],
                    Character.FOX,
                    Stage.BATTLEFIELD,
                    costume=1,
                    autostart=True,
                    swag=False,
                )
                
                menu_helper.menu_helper_simple(
                    gamestate,
                    controllers[2],
                    Character.FOX,
                    Stage.BATTLEFIELD,
                    costume=2,
                    autostart=True,
                    swag=False,
                )
                    
        # Determine winner
        final_stocks = {
            1: gamestate.players[1].stock if 1 in gamestate.players else 0,
            2: gamestate.players[2].stock if 2 in gamestate.players else 0,
        }
        winner_port = 1 if final_stocks[1] > final_stocks[2] else 2
        
        episode_p1 = GameEpisode(
            experiences=experiences_p1,
            total_reward=episode_reward_p1,
            episode_length=frame_count,
            winner_port=winner_port,
            final_stocks=final_stocks,
        )
        
        episode_p2 = GameEpisode(
            experiences=experiences_p2,
            total_reward=episode_reward_p2,
            episode_length=frame_count,
            winner_port=winner_port,
            final_stocks=final_stocks,
        )
        
        return episode_p1, episode_p2
    
    def _predict_and_collect_experience_batched(
        self,
        raw_inputs_p1: Dict[str, np.ndarray],
        raw_inputs_p2: Dict[str, np.ndarray],
        model: GPTv7,
    ) -> Tuple[Experience, Experience]:
        """Run batched model inference for both players simultaneously.
        
        This is 2x faster than running inference twice separately.
        """
        # Convert both players' raw inputs to tensors
        def inputs_to_tensor(raw_inputs):
            flattened_values = []
            for v in raw_inputs.values():
                if isinstance(v, np.ndarray):
                    flattened_values.append(v.flatten())
                else:
                    flattened_values.append(np.array([v]))
            return torch.from_numpy(np.concatenate(flattened_values)).float()
        
        obs_p1 = inputs_to_tensor(raw_inputs_p1)
        obs_p2 = inputs_to_tensor(raw_inputs_p2)
        
        # Stack to create batch of 2
        observations = torch.stack([obs_p1, obs_p2], dim=0)  # [2, F]
        
        # Add sequence dimension: [2, 1, F]
        X = observations.unsqueeze(1).to(self.device)
        
        # Build model inputs (batched)
        inputs_td = build_inputs_for_gptv7(X, self.colmap)
        
        # Single forward pass for both players
        pred = model(inputs_td)
        
        # Sample actions for both players
        # pred["main_stick"]: [2, 1, K_main]
        # pred["c_stick"]: [2, 1, K_c]
        # pred["buttons"]: [2, 1, K_buttons]
        # pred["shoulder"]: [2, 1, K_shoulder]
        # pred["value"]: [2, 1, 1] if present
        
        def sample_actions_from_batch(pred, batch_idx):
            """Extract actions for one player from batched predictions."""
            main_logits = pred["main_stick"][batch_idx, 0]
            c_logits = pred["c_stick"][batch_idx, 0]
            button_logits = pred["buttons"][batch_idx, 0]
            shoulder_logits = pred["shoulder"][batch_idx, 0]
            
            # Sample from categorical distributions
            main_stick_idx = torch.distributions.Categorical(logits=main_logits).sample().item()
            c_stick_idx = torch.distributions.Categorical(logits=c_logits).sample().item()
            shoulder_idx = torch.distributions.Categorical(logits=shoulder_logits).sample().item()
            
            # Sample buttons independently (binary)
            button_probs = torch.sigmoid(button_logits)
            button_samples = torch.bernoulli(button_probs)
            
            # Compute log probs for PPO
            main_log_prob = torch.distributions.Categorical(logits=main_logits).log_prob(
                torch.tensor(main_stick_idx, device=main_logits.device)
            )
            c_log_prob = torch.distributions.Categorical(logits=c_logits).log_prob(
                torch.tensor(c_stick_idx, device=c_logits.device)
            )
            shoulder_log_prob = torch.distributions.Categorical(logits=shoulder_logits).log_prob(
                torch.tensor(shoulder_idx, device=shoulder_logits.device)
            )
            
            button_log_probs = (
                button_samples * torch.log(button_probs + 1e-8) +
                (1 - button_samples) * torch.log(1 - button_probs + 1e-8)
            ).sum()
            
            total_log_prob = main_log_prob + c_log_prob + button_log_probs + shoulder_log_prob
            
            # Get value if available
            value = pred["value"][batch_idx, 0, 0] if "value" in pred else torch.tensor(0.0, device=self.device)
            
            return Experience(
                observation=observations[batch_idx].cpu().numpy(),
                action_logits={
                    "main_stick": main_logits.cpu(),
                    "c_stick": c_logits.cpu(),
                    "buttons": button_logits.cpu(),
                    "shoulder": shoulder_logits.cpu(),
                },
                action_log_probs=total_log_prob.item(),
                action_main_stick=main_stick_idx,
                action_c_stick=c_stick_idx,
                action_buttons=button_samples.cpu().numpy(),
                action_shoulder=shoulder_idx,
                value=value.item(),
                reward=0.0,
                done=False,
            )
        
        # Extract experiences for both players
        exp_p1 = sample_actions_from_batch(pred, 0)
        exp_p2 = sample_actions_from_batch(pred, 1)
        
        return exp_p1, exp_p2
    
    def _predict_and_collect_experience(
        self,
        raw_inputs: Dict[str, np.ndarray],
        model: Optional[GPTv7] = None,
    ) -> Experience:
        """Run model inference and collect experience.
        
        Args:
            raw_inputs: Raw game state inputs
            model: Model to use for prediction (defaults to self.model)
        """
        if model is None:
            model = self.model
            
        # Convert raw inputs to tensor (handle both arrays and scalars)
        flattened_values = []
        for v in raw_inputs.values():
            if isinstance(v, np.ndarray):
                flattened_values.append(v.flatten())
            else:
                # Scalar value
                flattened_values.append(np.array([v]))
        
        observation = torch.from_numpy(
            np.concatenate(flattened_values)
        ).float()
        
        # Prepare input for model (add batch and sequence dims)
        X = observation.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, F]
        
        # Build model inputs
        inputs_td = build_inputs_for_gptv7(X, self.colmap)
        
        # Forward pass
        with torch.no_grad():
            pred = model(inputs_td)
            
        # Sample actions from distributions
        main_logits = pred["main_stick"][0, 0]  # [K_main]
        c_logits = pred["c_stick"][0, 0]  # [K_c]
        button_logits = pred["buttons"][0, 0]  # [K_buttons]
        shoulder_logits = pred["shoulder"][0, 0]  # [K_shoulder]
        
        # Sample actions
        main_probs = F.softmax(main_logits, dim=0)
        c_probs = F.softmax(c_logits, dim=0)
        button_probs = torch.sigmoid(button_logits)
        shoulder_probs = F.softmax(shoulder_logits, dim=0)
        
        action_main = torch.multinomial(main_probs, 1).item()
        action_c = torch.multinomial(c_probs, 1).item()
        action_buttons = (button_probs > 0.5).float()
        action_shoulder = torch.multinomial(shoulder_probs, 1).item()
        
        # Compute log probability of taken action
        log_prob_main = torch.log(main_probs[action_main] + 1e-8).item()
        log_prob_c = torch.log(c_probs[action_c] + 1e-8).item()
        log_prob_buttons = torch.sum(
            action_buttons * torch.log(button_probs + 1e-8) +
            (1 - action_buttons) * torch.log(1 - button_probs + 1e-8)
        ).item()
        log_prob_shoulder = torch.log(shoulder_probs[action_shoulder] + 1e-8).item()
        log_prob = log_prob_main + log_prob_c + log_prob_buttons + log_prob_shoulder
        
        # Get value estimate
        value = pred["value"][0, 0, 0].item() if "value" in pred else 0.0
        
        # Create experience (reward will be filled in later)
        exp = Experience(
            observation=observation.cpu(),
            action_main_stick=action_main,
            action_c_stick=action_c,
            action_buttons=action_buttons.cpu(),
            action_shoulder=action_shoulder,
            log_prob=log_prob,
            value=value,
            reward=0.0,  # Will be filled in
            done=False,
        )
        
        return exp
    
    def _compute_reward(
        self,
        prev_gamestate: GameState,
        curr_gamestate: GameState,
        bot_port: int,
    ) -> float:
        """Compute reward for a single timestep."""
        config = get_config()
        reward = config.rl.reward_per_frame
        
        prev_player = prev_gamestate.players.get(bot_port)
        curr_player = curr_gamestate.players.get(bot_port)
        opp_port = 3 - bot_port
        prev_opp = prev_gamestate.players.get(opp_port)
        curr_opp = curr_gamestate.players.get(opp_port)
        
        if not all([prev_player, curr_player, prev_opp, curr_opp]):
            return reward
            
        # Damage rewards
        damage_dealt = max(0, curr_opp.percent - prev_opp.percent)
        damage_taken = max(0, curr_player.percent - prev_player.percent)
        reward += damage_dealt * config.rl.reward_damage_dealt
        reward += damage_taken * config.rl.reward_damage_taken
        
        # Stock rewards
        stocks_taken = max(0, prev_opp.stock - curr_opp.stock)
        stocks_lost = max(0, prev_player.stock - curr_player.stock)
        reward += stocks_taken * config.rl.reward_stock_taken
        reward += stocks_lost * config.rl.reward_stock_lost
        
        return reward
    
    def train_on_batch(self, batch: PPOBatch) -> Dict[str, float]:
        """Perform PPO update on a batch of experiences."""
        config = get_config()
        
        # Move batch to device
        obs = batch.observations.to(self.device)
        actions_main = batch.actions_main.to(self.device)
        actions_c = batch.actions_c.to(self.device)
        actions_buttons = batch.actions_buttons.to(self.device)
        actions_shoulder = batch.actions_shoulder.to(self.device)
        old_log_probs = batch.old_log_probs.to(self.device)
        advantages = batch.advantages.to(self.device)
        returns = batch.returns.to(self.device)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        
        # PPO epochs
        for ppo_epoch in range(config.rl.ppo_epochs):
            # Forward pass
            autocast_device = 'cuda' if self.device.type in ('cuda', 'mps') else 'cpu'
            amp_dtype = torch.float16 if config.train.amp_dtype == "float16" else torch.bfloat16
            
            with autocast(device_type=autocast_device, dtype=amp_dtype, enabled=config.train.use_amp):
                inputs_td = build_inputs_for_gptv7(obs, self.colmap)
                pred = self.model(inputs_td)
                
                B, L = actions_main.shape
                
                # Get logits
                main_logits = pred["main_stick"]  # [B, L, K_main]
                c_logits = pred["c_stick"]  # [B, L, K_c]
                button_logits = pred["buttons"]  # [B, L, K_buttons]
                shoulder_logits = pred["shoulder"]  # [B, L, K_shoulder]
                value_pred = pred["value"] if "value" in pred else torch.zeros(B, L, 1, device=self.device)
                
                # Compute new log probabilities
                main_log_probs = F.log_softmax(main_logits, dim=-1)
                c_log_probs = F.log_softmax(c_logits, dim=-1)
                button_probs = torch.sigmoid(button_logits)
                shoulder_log_probs = F.log_softmax(shoulder_logits, dim=-1)
                
                # Gather log probs of taken actions
                new_log_prob_main = torch.gather(
                    main_log_probs, 2, actions_main.unsqueeze(-1)
                ).squeeze(-1)
                new_log_prob_c = torch.gather(
                    c_log_probs, 2, actions_c.unsqueeze(-1)
                ).squeeze(-1)
                new_log_prob_shoulder = torch.gather(
                    shoulder_log_probs, 2, actions_shoulder.unsqueeze(-1)
                ).squeeze(-1)
                
                # Button log probs
                new_log_prob_buttons = torch.sum(
                    actions_buttons * torch.log(button_probs + 1e-8) +
                    (1 - actions_buttons) * torch.log(1 - button_probs + 1e-8),
                    dim=-1
                )
                
                # Total log prob
                new_log_probs = (
                    new_log_prob_main + new_log_prob_c +
                    new_log_prob_buttons + new_log_prob_shoulder
                )
                
                # PPO policy loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 1 - config.rl.ppo_epsilon, 1 + config.rl.ppo_epsilon
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(value_pred.squeeze(-1), returns)
                
                # Entropy bonus (encourage exploration)
                entropy_main = -(main_log_probs * torch.exp(main_log_probs)).sum(dim=-1).mean()
                entropy_c = -(c_log_probs * torch.exp(c_log_probs)).sum(dim=-1).mean()
                entropy_shoulder = -(shoulder_log_probs * torch.exp(shoulder_log_probs)).sum(dim=-1).mean()
                entropy = entropy_main + entropy_c + entropy_shoulder
                
                # Total loss
                loss = (
                    policy_loss +
                    config.rl.value_loss_coef * value_loss -
                    config.rl.entropy_coef * entropy
                )
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if config.rl.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.rl.max_grad_norm
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            
        self.global_step += 1
        
        metrics = {
            "policy_loss": total_policy_loss / config.rl.ppo_epochs,
            "value_loss": total_value_loss / config.rl.ppo_epochs,
            "entropy": total_entropy / config.rl.ppo_epochs,
            "total_loss": (total_policy_loss + total_value_loss) / config.rl.ppo_epochs,
        }
        
        return metrics
    
    def save_checkpoint(self, episode: int):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"ppo_model_ep{episode:04d}.pt"
        
        print(f"  Saving checkpoint to {checkpoint_path.name}...")
        checkpoint_data = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "episode": episode,
            "global_step": self.global_step,
            "win_rate_p1": self.win_rate_p1,
            "config": get_config().to_dict(),
        }
        torch.save(checkpoint_data, checkpoint_path)
        
        # Calculate checkpoint size
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"  Checkpoint saved ({size_mb:.1f} MB)")
        
        # Add to opponent pool
        self.opponent_checkpoints.append(checkpoint_path)
        print(f"  Opponent pool now contains {len(self.opponent_checkpoints)} checkpoints")
    
    def train(
        self,
        console: Console,
        controllers: Dict[int, Controller],
        num_episodes: int = 1000,
    ):
        """Main training loop with true self-play (both players using models)."""
        config = get_config()
        
        print("Starting PPO self-play training...")
        print(f"Device: {self.device}")
        print(f"Episodes: {num_episodes}")
        print(f"Buffer size: {config.rl.buffer_size}")
        print("Mode: True self-play (model vs model)")
        
        for episode in range(num_episodes):
            print(f"\n{'='*70}")
            print(f"Episode {episode + 1}/{num_episodes} ({100*(episode+1)/num_episodes:.1f}% complete)")
            print(f"{'='*70}")
            print(f"Training step: {self.global_step}")
            print(f"Avg P1 reward (last 10): {np.mean(self.episode_rewards_p1[-10:]) if len(self.episode_rewards_p1) > 0 else 0:.2f}")
            print(f"Avg P2 reward (last 10): {np.mean(self.episode_rewards_p2[-10:]) if len(self.episode_rewards_p2) > 0 else 0:.2f}")
            print(f"Avg episode length: {np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) > 0 else 0:.0f} frames")
            print(f"Opponent pool size: {len(self.opponent_checkpoints)}")
            print()
            
            # Choose opponent for player 2
            # If we started from a checkpoint and P2 already has a model, keep it for first few episodes
            if self.started_from_checkpoint and self.model_p2 is not None and episode < 5:
                print(f"[Opponent] P2 using initial checkpoint (episode {episode + 1}/5 warmup)")
            elif episode > 0 and config.rl.self_play_enabled and len(self.opponent_checkpoints) > 0:
                # Sample random past checkpoint for player 2
                opponent_ckpt = np.random.choice(list(self.opponent_checkpoints))
                print(f"[Opponent] P2 using past checkpoint: {opponent_ckpt.name}")
                
                # Load checkpoint into model_p2
                ckpt = torch.load(opponent_ckpt, map_location=self.device)
                if self.model_p2 is None:
                    # Create a new model instance for player 2
                    self.model_p2 = GPTv7().to(self.device)
                self.model_p2.load_state_dict(ckpt['model'])
                self.model_p2.eval()
            else:
                # Both players use the same (current) model
                if self.model_p2 is None:
                    print("[Opponent] P2 using current model (mirror match)")
                else:
                    print("[Opponent] P2 using same checkpoint as P1")
            
            print()
            
            # Collect episode from both players
            episode_p1, episode_p2 = self.collect_episode(console, controllers)
            
            # Add both players' experiences to buffer
            self.experience_buffer.add_episode(episode_p1)
            self.experience_buffer.add_episode(episode_p2)
            
            # Track metrics
            self.episode_rewards_p1.append(episode_p1.total_reward)
            self.episode_rewards_p2.append(episode_p2.total_reward)
            self.episode_lengths.append(episode_p1.episode_length)
            
            # Update win rate for player 1 (the one we're primarily training)
            if episode_p1.winner_port == 1:
                self.win_rate_p1 = self.win_rate_p1 * 0.95 + 0.05  # Exponential moving average
            else:
                self.win_rate_p1 = self.win_rate_p1 * 0.95
            
            print(f"\n[Episode Results]")
            print(f"  Winner: Port {episode_p1.winner_port} {'(P1)' if episode_p1.winner_port == 1 else '(P2)'}")
            print(f"  Final stocks: P1={episode_p1.final_stocks[1]}, P2={episode_p1.final_stocks[2]}")
            print(f"  Episode length: {episode_p1.episode_length} frames ({episode_p1.episode_length/60:.1f}s)")
            print(f"  P1 cumulative reward: {episode_p1.total_reward:.2f}")
            print(f"  P2 cumulative reward: {episode_p2.total_reward:.2f}")
            print(f"  P1 Win rate (EMA): {self.win_rate_p1:.3f}")
            print(f"  Experience buffer: {len(self.experience_buffer)} episodes, "
                  f"{sum(len(ep.experiences) for ep in self.experience_buffer.episodes)} total frames")
            
            # Train more frequently - every episode once we have minimum data
            min_episodes_before_training = max(2, config.rl.batch_size // 200)  # Need at least a few episodes
            if len(self.experience_buffer) >= min_episodes_before_training:
                print(f"\n--- Training Update {self.global_step + 1} ---")
                
                # Multiple training batches per episode for faster learning
                num_training_batches = 3  # Train on 3 batches per episode
                for batch_idx in range(num_training_batches):
                    batch = self.experience_buffer.sample_batch(
                        config.rl.batch_size, config.seq_len
                    )
                    if batch is not None:
                        print(f"  Training batch {batch_idx + 1}/{num_training_batches}...")
                        metrics = self.train_on_batch(batch)
                        print(f"  Policy loss: {metrics['policy_loss']:.4f}, "
                              f"Value loss: {metrics['value_loss']:.4f}, "
                              f"Entropy: {metrics['entropy']:.4f}")
                    else:
                        print(f"  Could not sample batch {batch_idx + 1}")
                
                print(f"Training update complete. Global step: {self.global_step}")
            else:
                print(f"Waiting for more data before training ({len(self.experience_buffer)}/{min_episodes_before_training} episodes)")
            
            # Save checkpoint more frequently (every 5 episodes)
            if (episode + 1) % 5 == 0:
                print(f"\n[Checkpoint] Saving model at episode {episode + 1}...")
                self.save_checkpoint(episode + 1)
            
            # Update opponent pool more frequently (every 10 episodes)
            if (episode + 1) % 10 == 0 and episode > 0:
                print(f"[Opponent Pool] Adding checkpoint to opponent pool...")
                self.save_checkpoint(episode + 1)


def main():
    parser = argparse.ArgumentParser(description='Self-play PPO training for Melee AI')
    parser.add_argument('--dolphin_executable_path', '-e', default=None,
                        help='Path to Dolphin executable')
    parser.add_argument('--iso', default=None, type=str, required=True,
                        help='Path to Melee ISO')
    parser.add_argument('--device', default='mps',
                        help='Device to run on (cpu/cuda/mps)')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes to train')
    parser.add_argument('--checkpoint', '-c', type=Path, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--data-root', type=str, default=None,
                        help='Dataset directory with meta.json')
    
    args = parser.parse_args()
    
    # Initialize config
    init_config()
    config = get_config()
    
    # Setup device
    device = _resolve_device(args.device)
    print(f"Using device: {device}")
    
    # Load column map from dataset
    from window_dataset import ZarrCorpusIndex
    data_root = args.data_root or config.zarr.out_root
    corpus_index = ZarrCorpusIndex(data_root)
    
    from column_map import ColumnMap
    
    # Create a mock dataset object that has the structure ColumnMap expects
    class MockDataset:
        def __init__(self, feature_names, target_names):
            self._feature_names_sel = feature_names
            self._target_names_sel = target_names
            # Create a mock index object
            self.index = type('obj', (object,), {
                'feature_names': feature_names,
                'target_names': target_names
            })
    
    mock_ds = MockDataset(
        feature_names=corpus_index.feature_names,
        target_names=corpus_index.target_names
    )
    colmap = ColumnMap.from_dataset(mock_ds)
    
    # Create model
    model = GPTv7()
    
    # Load button thresholds if available
    button_thresholds_path = Path("button_thresholds.json")
    button_thresholds = None
    if button_thresholds_path.exists():
        print(f"Loading button thresholds from {button_thresholds_path}")
        with open(button_thresholds_path, 'r') as f:
            threshold_data = json.load(f)
            button_thresholds = threshold_data.get("threshold_map", None)
            if button_thresholds:
                print(f"Button thresholds loaded:")
                for btn, thresh in button_thresholds.items():
                    print(f"  {btn}: {thresh:.3f}")
    else:
        print("No button_thresholds.json found, using default 0.5 for all buttons")
    
    # Load checkpoint if provided
    initial_checkpoint = None
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        initial_checkpoint = args.checkpoint
        print("Both players will start with this checkpoint")
    
    # Create trainer
    checkpoint_dir = Path(config.train.out_dir) / "ppo_checkpoints"
    trainer = SelfPlayPPOTrainer(model, device, colmap, checkpoint_dir, button_thresholds)
    
    # If a checkpoint was provided, load it for player 2 as well
    if initial_checkpoint:
        print("Loading same checkpoint for player 2...")
        trainer.model_p2 = GPTv7().to(device)
        trainer.model_p2.load_state_dict(ckpt['model'])
        trainer.model_p2.eval()
        trainer.started_from_checkpoint = True
        print("Both players initialized with the same checkpoint")
        print("Will use same checkpoint for first 5 episodes before introducing opponent diversity")
    
    # Setup console and controllers
    console = Console(
        path=args.dolphin_executable_path,
        slippi_address="127.0.0.1",
        save_replays=False,
        copy_home_directory=False,
        tmp_home_directory=False,
        blocking_input=True,
    )
    
    controllers = {
        1: Controller(console=console, port=1, type=ControllerType.STANDARD),
        2: Controller(console=console, port=2, type=ControllerType.STANDARD),
    }
    
    def signal_handler(sig, frame):
        print("\nShutting down...")
        for controller in controllers.values():
            controller.disconnect()
        console.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start console
    console.run(iso_path=args.iso)
    print("Connecting to console...")
    if not console.connect():
        print("ERROR: Failed to connect to console")
        sys.exit(1)
    
    for controller in controllers.values():
        if not controller.connect():
            print("ERROR: Failed to connect controller")
            sys.exit(1)
    
    print("Connected!")
    
    # Train (both players controlled by models)
    trainer.train(console, controllers, args.episodes)


if __name__ == "__main__":
    main()

