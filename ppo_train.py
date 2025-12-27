#!/usr/bin/env python3
"""Simple PPO training script for Melee-AI.

Trains a GPT model to play Super Smash Bros. Melee using Proximal Policy Optimization.
Self-contained single-file implementation with:
- Single Dolphin emulator instance
- Fox vs CPU Fox (level 9)
- Rollouts end on stock change (either player loses a stock)
- Online learning (train after each stock exchange)
- 4 PPO epochs per rollout
- Checkpoint saving every N rollouts
"""

from __future__ import annotations

import argparse
import random
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.amp import autocast
from torch.distributions import Bernoulli, Categorical

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from column_map import ColumnMap
from config.config import init_config, get_config, init_config_from_checkpoint
from utils import match_state_dict_keys
from config.reward_config import RewardConfig
from constants import (
    LEGAL_TOURNAMENT_STAGES,
    SUPPORTED_CHARS,
    BOT_PORT,
    OPP_PORT,
    MIN_EPISODE_LENGTH,
)
from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)
from model.sampling import action_info_to_controller_state
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
from model.nano_gpt import GPT
from model_interface import (
    ControllerState,
    GPTInferenceEngine,
    apply_model_outputs_to_game,
    collect_raw_inputs_from_gamestate,
    model_to_dolphin01,
)
from train import find_latest_checkpoint
from train.batch_utils import build_model_inputs
from train.setup import configure_amp, parse_cli_overrides
from train.value_head import (
    build_reward_feature_index,
    compute_frame_rewards,
    compute_reward_components,
    compute_tech_rewards,
    compute_tech_reward_components,
)


# ============================================================================
# Configuration and Constants
# ============================================================================

@dataclass
class PPOConfig:
    """PPO-specific hyperparameters."""

    # Learning
    learning_rate: float = 1e-5  # Very low for online RL fine-tuning from imitation

    # PPO parameters
    clip_epsilon: float = 0.2  # Clipping range for policy ratio
    num_epochs: int = 4  # Number of epochs to train on each rollout
    mini_batch_size: int = 256  # Mini-batch size (increased for M4 Max efficiency)
    value_coef: float = 0.5  # Coefficient for value loss
    entropy_coef: float = 0.01  # Coefficient for entropy bonus

    # Mixed precision - DISABLED by default for inference
    # AMP adds overhead for single-sample inference (batch_size=1) that outweighs
    # float16 benefits. The autocast context manager overhead at 60Hz is significant.
    use_amp: bool = False

    # Note: We use simple discounted returns (like slippi-ai), not GAE

    # Training stability
    grad_clip: float = 0.5  # Gradient clipping max norm
    normalize_advantages: bool = True  # Normalize advantages
    max_kl: float = float('inf')  # KL threshold DISABLED for tech experiment (was 0.05)
    clip_value_loss: bool = True  # Clip value function updates (prevents value collapse)

    # Teacher KL penalty (keeps policy close to imitation behavior)
    kl_teacher_weight: float = 0.0  # DISABLED for tech experiment (was 0.1)

    # Checkpointing
    checkpoint_interval: int = 10  # Save checkpoint every N rollouts (stocks)
    checkpoint_dir: Path = Path("checkpoints_ppo")

    # Logging
    log_interval: int = 1  # Log every rollout (each rollout = one stock taken)


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ActionInfo:
    """Actions taken and their log probabilities."""

    # Action indices
    main_stick_idx: int
    c_stick_idx: int
    buttons: torch.Tensor  # [5] binary
    shoulder_idx: int

    # Log probabilities
    main_log_prob: torch.Tensor
    c_log_prob: torch.Tensor
    buttons_log_probs: torch.Tensor  # [5]
    shoulder_log_prob: torch.Tensor


@dataclass
class RolloutStep:
    """Single timestep in a rollout."""

    # Single frame features [F] (not full sequence!)
    features: torch.Tensor

    # Actions taken
    action_info: ActionInfo

    # Value prediction
    value_pred: float

    # Reward signal
    reward: float
    done: bool


@dataclass
class Rollout:
    """Complete rollout data for one stock exchange (not a full game)."""

    steps: List[RolloutStep]
    episode_length: int
    total_reward: float
    winner: int  # 1 or 2

    # Computed after collection
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None

    # Raw advantage statistics (before normalization)
    raw_advantage_mean: float = 0.0
    raw_advantage_std: float = 0.0
    raw_advantage_min: float = 0.0
    raw_advantage_max: float = 0.0


@dataclass
class PPOLossComponents:
    """PPO loss components for logging."""

    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor

    # Per-component breakdowns
    main_policy_loss: torch.Tensor
    c_policy_loss: torch.Tensor
    buttons_policy_loss: torch.Tensor
    shoulder_policy_loss: torch.Tensor

    main_entropy: float
    c_entropy: float
    buttons_entropy: float
    shoulder_entropy: float

    # Diagnostics
    main_clip_frac: float
    c_clip_frac: float
    buttons_clip_frac: float
    shoulder_clip_frac: float

    approx_kl: float
    grad_norm: float

    # Value function diagnostics
    value_pred_mean: float = 0.0
    value_pred_std: float = 0.0
    returns_mean: float = 0.0
    returns_std: float = 0.0
    uev: float = 1.0  # Unexplained Variance: value_loss / var(returns), lower is better

    # Raw advantage statistics (before normalization)
    raw_advantage_mean: float = 0.0
    raw_advantage_std: float = 0.0
    raw_advantage_min: float = 0.0
    raw_advantage_max: float = 0.0

    # Teacher KL penalty (if using teacher model)
    teacher_kl: float = 0.0


@dataclass
class TrainingStats:
    """Rolling training statistics across rollouts (each rollout = one stock taken)."""

    episode_returns: deque  # Last N rollout total rewards
    episode_lengths: deque  # Last N rollout lengths (frames)
    win_count: int = 0  # Stocks taken by bot
    loss_count: int = 0  # Stocks lost by bot
    kl_history: deque = None  # Last N KL values

    def __post_init__(self):
        if self.kl_history is None:
            self.kl_history = deque(maxlen=100)

    @classmethod
    def create(cls, window_size: int = 100) -> "TrainingStats":
        """Create new TrainingStats with given window size."""
        return cls(
            episode_returns=deque(maxlen=window_size),
            episode_lengths=deque(maxlen=window_size),
            win_count=0,
            loss_count=0,
            kl_history=deque(maxlen=window_size),
        )

    def update(self, rollout: "Rollout", loss_components: "PPOLossComponents", bot_port: int):
        """Update stats with results from an episode."""
        self.episode_returns.append(rollout.total_reward)
        self.episode_lengths.append(rollout.episode_length)
        if rollout.winner == bot_port:
            self.win_count += 1
        else:
            self.loss_count += 1
        self.kl_history.append(loss_components.approx_kl)

    @property
    def win_rate(self) -> float:
        """Win rate as a fraction [0, 1]."""
        total = self.win_count + self.loss_count
        return self.win_count / total if total > 0 else 0.0

    @property
    def avg_return(self) -> float:
        """Average return over recent episodes."""
        return sum(self.episode_returns) / len(self.episode_returns) if self.episode_returns else 0.0

    @property
    def avg_length(self) -> float:
        """Average episode length over recent episodes."""
        return sum(self.episode_lengths) / len(self.episode_lengths) if self.episode_lengths else 0.0

    @property
    def avg_kl(self) -> float:
        """Average KL over recent episodes."""
        return sum(self.kl_history) / len(self.kl_history) if self.kl_history else 0.0

    def summary(self) -> str:
        """One-line summary of training progress."""
        total_stocks = self.win_count + self.loss_count
        return (
            f"Stocks Taken: {self.win_rate:.1%} ({self.win_count}/{total_stocks}) | "
            f"Avg Return: {self.avg_return:+.2f} | "
            f"Avg Length: {self.avg_length:.0f} frames | "
            f"Avg KL: {self.avg_kl:.4f}"
        )


# ============================================================================
# Action Sampling
# ============================================================================

ActionSelection = Literal["stochastic", "greedy"]


def _pack_actions_for_cpu(
    main_idx: torch.Tensor,
    c_idx: torch.Tensor,
    shoulder_idx: torch.Tensor,
    button_samples: torch.Tensor,
) -> Tuple[int, int, int, torch.Tensor]:
    """Move actions to CPU once for controller application."""
    indices = torch.stack([main_idx, c_idx, shoulder_idx]).to(
        dtype=button_samples.dtype
    )
    actions_gpu = torch.cat([indices, button_samples], dim=0)
    actions_cpu = actions_gpu.detach().cpu()
    buttons_cpu = actions_cpu[3:].to(torch.float32)
    return (
        int(actions_cpu[0].item()),
        int(actions_cpu[1].item()),
        int(actions_cpu[2].item()),
        buttons_cpu,
    )


def sample_actions_with_logprobs(
    outputs: TensorDict,
) -> ActionInfo:
    """Sample actions from model outputs and compute log probabilities.

    Args:
        outputs: Model outputs with keys ["main_stick", "c_stick", "buttons", "shoulder"]
    Returns:
        ActionInfo with sampled actions and log probabilities
    """
    # Extract last timestep logits [1, seq_len, K] -> [K]
    # Note: outputs are already on device, no need to call .to(device)
    main_logits = outputs["main_stick"][0, -1]  # [64]
    c_logits = outputs["c_stick"][0, -1]  # [9]
    button_logits = outputs["buttons"][0, -1]  # [5]
    shoulder_logits = outputs["shoulder"][0, -1]  # [5]

    # Sample all categorical actions using multinomial (faster than Categorical for single samples)
    main_idx = torch.multinomial(torch.softmax(main_logits, dim=-1), 1).squeeze(-1)
    c_idx = torch.multinomial(torch.softmax(c_logits, dim=-1), 1).squeeze(-1)
    shoulder_idx = torch.multinomial(torch.softmax(shoulder_logits, dim=-1), 1).squeeze(-1)

    # Compute log probabilities using log_softmax (avoids creating distribution objects)
    main_log_softmax = torch.log_softmax(main_logits, dim=-1)
    c_log_softmax = torch.log_softmax(c_logits, dim=-1)
    shoulder_log_softmax = torch.log_softmax(shoulder_logits, dim=-1)

    main_log_prob = main_log_softmax[main_idx]
    c_log_prob = c_log_softmax[c_idx]
    shoulder_log_prob = shoulder_log_softmax[shoulder_idx]

    # Buttons: Independent Bernoulli for each button
    button_probs = torch.sigmoid(button_logits)
    button_samples = torch.bernoulli(button_probs)

    # Compute log probabilities for buttons
    button_log_probs = torch.where(
        button_samples == 1,
        torch.log(button_probs + 1e-8),
        torch.log(1 - button_probs + 1e-8),
    )

    main_idx_cpu, c_idx_cpu, shoulder_idx_cpu, buttons_cpu = _pack_actions_for_cpu(
        main_idx, c_idx, shoulder_idx, button_samples
    )

    return ActionInfo(
        main_stick_idx=main_idx_cpu,
        c_stick_idx=c_idx_cpu,
        buttons=buttons_cpu,
        shoulder_idx=shoulder_idx_cpu,
        main_log_prob=main_log_prob.detach(),
        c_log_prob=c_log_prob.detach(),
        buttons_log_probs=button_log_probs.detach(),
        shoulder_log_prob=shoulder_log_prob.detach(),
    )


def greedy_actions_with_logprobs(outputs: TensorDict) -> ActionInfo:
    """Select greedy actions and compute log probabilities."""
    main_logits = outputs["main_stick"][0, -1]  # [64]
    c_logits = outputs["c_stick"][0, -1]  # [9]
    button_logits = outputs["buttons"][0, -1]  # [5]
    shoulder_logits = outputs["shoulder"][0, -1]  # [5]

    main_idx = torch.argmax(main_logits, dim=-1)
    c_idx = torch.argmax(c_logits, dim=-1)
    shoulder_idx = torch.argmax(shoulder_logits, dim=-1)

    main_log_softmax = torch.log_softmax(main_logits, dim=-1)
    c_log_softmax = torch.log_softmax(c_logits, dim=-1)
    shoulder_log_softmax = torch.log_softmax(shoulder_logits, dim=-1)

    main_log_prob = main_log_softmax[main_idx]
    c_log_prob = c_log_softmax[c_idx]
    shoulder_log_prob = shoulder_log_softmax[shoulder_idx]

    button_probs = torch.sigmoid(button_logits)
    button_samples = (button_probs >= 0.5).to(button_probs.dtype)
    button_log_probs = torch.where(
        button_samples == 1,
        torch.log(button_probs + 1e-8),
        torch.log(1 - button_probs + 1e-8),
    )

    main_idx_cpu, c_idx_cpu, shoulder_idx_cpu, buttons_cpu = _pack_actions_for_cpu(
        main_idx, c_idx, shoulder_idx, button_samples
    )

    return ActionInfo(
        main_stick_idx=main_idx_cpu,
        c_stick_idx=c_idx_cpu,
        buttons=buttons_cpu,
        shoulder_idx=shoulder_idx_cpu,
        main_log_prob=main_log_prob.detach(),
        c_log_prob=c_log_prob.detach(),
        buttons_log_probs=button_log_probs.detach(),
        shoulder_log_prob=shoulder_log_prob.detach(),
    )


def select_actions_with_logprobs(
    outputs: TensorDict,
    mode: ActionSelection,
) -> ActionInfo:
    """Select actions according to the requested rollout mode."""
    if mode == "stochastic":
        return sample_actions_with_logprobs(outputs)
    if mode == "greedy":
        return greedy_actions_with_logprobs(outputs)
    raise ValueError(f"Unknown action selection mode: {mode}")


# ============================================================================
# Rollout Collection
# ============================================================================

def collect_rollout(
    console: Console,
    controllers: Dict[int, Controller],
    engine: GPTInferenceEngine,
    model: GPT,
    column_map: ColumnMap,
    reward_config: RewardConfig,
    current_stage: Stage,
    bot_char: Character,
    opp_char: Character,
    action_selection: ActionSelection,
    amp_ctx,
) -> Rollout:
    """Collect a complete rollout (one stock exchange).

    Args:
        console: Dolphin console
        controllers: Controller dict
        engine: Inference engine
        model: GPT model
        column_map: Feature column mapping
        reward_config: Reward configuration
        current_stage: Stage to play on
        bot_char: Bot character
        opp_char: Opponent character
        action_selection: Rollout action selection mode
        use_amp: Whether to use AMP during inference

    Returns:
        Complete rollout with rewards
    """
    steps: List[RolloutStep] = []
    frame_count = 0
    prev_p1_action = None
    prev_p2_action = None

    # Build reward feature index once
    reward_idx = build_reward_feature_index(column_map)

    # Reset engine buffers
    engine.buffer.clear()
    engine.frame_history.clear()

    # Menu helper for navigation
    menu_helper = MenuHelper()

    while True:
        gamestate = console.step()
        if gamestate is None:
            continue

        # Handle menu navigation
        if gamestate.menu_state not in [Menu.IN_GAME, Menu.SUDDEN_DEATH]:
            # Check if both players are ready before autostarting
            autostart = False
            if BOT_PORT in gamestate.players and OPP_PORT in gamestate.players:
                p1_state = gamestate.players[BOT_PORT]
                p1_ready = (p1_state.character == bot_char) and p1_state.coin_down

                p2_state = gamestate.players[OPP_PORT]
                p2_ready = (
                    (p2_state.character == opp_char)
                    and (p2_state.controller_status == ControllerStatus.CONTROLLER_CPU)
                    and (p2_state.cpu_level == 9)
                )

                autostart = p1_ready and p2_ready

            # Navigate menus
            menu_helper.menu_helper_simple(
                gamestate,
                controllers[BOT_PORT],
                bot_char,
                current_stage,
                costume=1,
                autostart=autostart,
                swag=False,
            )
            # Configure opponent as CPU
            menu_helper.choose_character(
                character=opp_char,
                gamestate=gamestate,
                controller=controllers[OPP_PORT],
                cpu_level=9,
                costume=2,
                swag=False,
                start=False,
            )
            continue

        # ===== COLLECT FEATURES =====
        raw_inputs = collect_raw_inputs_from_gamestate(
            gamestate, BOT_PORT, OPP_PORT
        )

        # Prepare model inputs (updates buffer AND returns TensorDict)
        inputs_td = engine.prepare_inputs(raw_inputs.transformed)

        # Store current frame for rollout (just this frame, not the full sequence!)
        current_frame = engine.buffer[-1].detach() if len(engine.buffer) > 0 else None

        frame_count += 1

        # ===== PERIODIC LOGGING =====
        if frame_count % 60 == 0:  # Once per second
            if BOT_PORT in gamestate.players and OPP_PORT in gamestate.players:
                bot_player = gamestate.players[BOT_PORT]
                opp_player = gamestate.players[OPP_PORT]
                print(
                    f"Frame {frame_count:4d} | "
                    f"Bot: {bot_player.stock:1d} stocks, {bot_player.percent:5.1f}% | "
                    f"Opp: {opp_player.stock:1d} stocks, {opp_player.percent:5.1f}%"
                )

        # ===== CHECK EPISODE TERMINATION =====
        p1_action = raw_inputs.raw.get("p1_action")
        p2_action = raw_inputs.raw.get("p2_action")

        done = False
        winner = None

        if prev_p1_action is not None and p1_action is not None and p2_action is not None:
            # Detect stock loss: transition to dying action (≤ 0x0A)
            p1_died = (prev_p1_action > 0x0A) and (p1_action <= 0x0A)
            p2_died = (prev_p2_action > 0x0A) and (p2_action <= 0x0A)

            if p1_died:
                done = True
                winner = OPP_PORT
            elif p2_died:
                done = True
                winner = BOT_PORT

        prev_p1_action = p1_action
        prev_p2_action = p2_action

        # ===== MODEL INFERENCE =====
        if inputs_td is None or len(engine.buffer) < engine.warmup_frames:
            # Warmup: use neutral controller
            controller_state = ControllerState.neutral()
            apply_model_outputs_to_game(controllers[BOT_PORT], controller_state)

            if done:
                # Rollout ended during warmup (very rare)
                break
            continue

        # Get model device (engine.prepare_inputs already moves tensors to device)
        device = next(model.parameters()).device

        # Forward pass with optional AMP (float16 on MPS for Apple Silicon optimization)
        with torch.inference_mode(), autocast(
            device_type=amp_ctx.device_type, dtype=amp_ctx.dtype, enabled=amp_ctx.enabled
        ):
            outputs = model(inputs_td)

        # Sample or select actions and get log probs
        action_info = select_actions_with_logprobs(outputs, action_selection)

        # Apply to game
        controller_state = action_info_to_controller_state(action_info)
        apply_model_outputs_to_game(controllers[BOT_PORT], controller_state)

        # Store step - ONLY store current frame [F], not full sequence!
        steps.append(RolloutStep(
            features=current_frame,  # Just current frame [F], not [seq_len, F]
            action_info=action_info,
            value_pred=outputs["value"][0, -1, 0].item(),
            reward=0.0,  # Will compute after episode ends
            done=done,
        ))

        if done:
            break

    # ===== POST-EPISODE PROCESSING =====
    # Now compute rewards and transfer to CPU
    # TECH-ONLY REWARD: Using compute_tech_rewards instead of compute_frame_rewards
    warmup = engine.warmup_frames
    total_frames = frame_count
    print(f"\n[Rollout ended] Total: {total_frames} frames ({warmup} warmup + {len(steps)} collected)")
    print("[Post-process] Computing TECH-ONLY rewards...")
    post_start = time.perf_counter()
    device = next(model.parameters()).device
    rewards = torch.zeros(len(steps), dtype=torch.float32)
    if steps:
        features = torch.stack([s.features for s in steps], dim=0)  # [L, F]
        features_device = features.to(device)
        # TECH-ONLY: Use tech rewards instead of full reward function
        rewards_tensor = compute_tech_rewards(
            features_device.unsqueeze(0), reward_idx
        )
        rewards = rewards_tensor[0].detach().cpu()
        components = compute_tech_reward_components(
            features_device.unsqueeze(0), reward_idx
        )
        components_cpu = {key: value[0].detach().cpu() for key, value in components.items()}
    else:
        components_cpu = {}

    for i, step in enumerate(steps):
        step.reward = float(rewards[i].item())
        step.features = step.features.cpu()

    # Compute total reward
    total_reward = float(rewards.sum().item()) if steps else 0.0
    post_elapsed = time.perf_counter() - post_start
    print(f"[Rewards computed] {post_elapsed:.2f}s | Total reward: {total_reward:+.2f}")
    if components_cpu:
        print("[TECH Reward breakdown]")
        for key in ("p1_tech_success", "p1_tech_miss", "p2_tech_success", "p2_tech_miss"):
            values = components_cpu.get(key)
            if values is None:
                continue
            nonzero = int((values != 0).sum().item())
            total = float(values.sum().item())
            print(
                f"  {key:>16}: count={int(total):2d} (frames={nonzero:4d})"
            )
        total_components = float(components_cpu["total"].sum().item())
        print(f"  {'total':>16}: sum={total_components:+.2f}")

    return Rollout(
        steps=steps,
        episode_length=len(steps),
        total_reward=total_reward,
        winner=winner or BOT_PORT,  # Default to bot if no clear winner
    )


# ============================================================================
# Advantage Computation (Simple Discounted Returns)
# ============================================================================

@dataclass
class AdvantageResult:
    """Result of advantage computation with diagnostics."""
    advantages: torch.Tensor  # [L] normalized advantages for policy gradient
    returns: torch.Tensor  # [L] returns for value training
    raw_advantage_mean: float
    raw_advantage_std: float
    raw_advantage_min: float
    raw_advantage_max: float


def compute_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    normalize: bool = True,
) -> AdvantageResult:
    """Compute discounted returns and advantages (slippi-ai approach).

    Uses simple discounted returns with bootstrapping:
      Return[t] = reward[t] + gamma * Return[t+1]
      Advantage[t] = Return[t] - Value[t]

    This is simpler and more stable than GAE.

    Args:
        rewards: [L] per-frame rewards
        values: [L] value predictions
        gamma: Discount factor
        normalize: Whether to normalize advantages

    Returns:
        AdvantageResult with advantages, returns, and raw advantage statistics
    """
    L = len(rewards)
    device = rewards.device

    # Compute discounted returns (backward pass)
    # Terminal state has no bootstrap value (episode ends on stock loss)
    returns = torch.zeros(L, device=device, dtype=rewards.dtype)
    running_return = 0.0
    for t in reversed(range(L)):
        running_return = rewards[t].item() + gamma * running_return
        returns[t] = running_return

    # Advantages = Returns - Values (simple TD advantage)
    advantages = returns - values

    # Store raw advantage statistics before normalization
    raw_advantage_mean = advantages.mean().item()
    raw_advantage_std = advantages.std().item()
    raw_advantage_min = advantages.min().item()
    raw_advantage_max = advantages.max().item()

    # Normalize advantages (critical for stability)
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return AdvantageResult(
        advantages=advantages,
        returns=returns,
        raw_advantage_mean=raw_advantage_mean,
        raw_advantage_std=raw_advantage_std,
        raw_advantage_min=raw_advantage_min,
        raw_advantage_max=raw_advantage_max,
    )


def compute_transition_reward(
    prev_features: torch.Tensor,
    curr_features: torch.Tensor,
    reward_idx,
    reward_config: RewardConfig,
    device: torch.device,
) -> float:
    """Compute the reward for the transition prev -> curr.

    compute_frame_rewards assigns the transition reward to the previous frame index,
    so we read index 0 from the two-frame window.
    """
    X_for_reward = torch.stack([prev_features, curr_features], dim=0).unsqueeze(0)
    X_for_reward = X_for_reward.to(device)
    rewards_tensor = compute_frame_rewards(X_for_reward, reward_idx, reward_config)
    return rewards_tensor[0, 0].item()


# ============================================================================
# PPO Loss Computation
# ============================================================================

def compute_log_probs_for_actions(
    outputs: TensorDict,
    rollout: Rollout,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Compute log probabilities for actions taken in rollout.

    Args:
        outputs: Model outputs [L, seq_len, K]
        rollout: Rollout with action info
        device: Device for tensors

    Returns:
        Dict with keys ["main", "c", "buttons", "shoulder"] and log prob tensors
    """
    L = len(rollout.steps)

    # Extract logits for last timestep of each sequence
    main_logits = outputs["main_stick"][:, -1, :]  # [L, 64]
    c_logits = outputs["c_stick"][:, -1, :]  # [L, 9]
    button_logits = outputs["buttons"][:, -1, :]  # [L, 5]
    shoulder_logits = outputs["shoulder"][:, -1, :]  # [L, 5]

    # Extract taken actions from rollout
    main_actions = torch.tensor(
        [s.action_info.main_stick_idx for s in rollout.steps],
        dtype=torch.long,
        device=device,
    )
    c_actions = torch.tensor(
        [s.action_info.c_stick_idx for s in rollout.steps],
        dtype=torch.long,
        device=device,
    )
    shoulder_actions = torch.tensor(
        [s.action_info.shoulder_idx for s in rollout.steps],
        dtype=torch.long,
        device=device,
    )
    button_actions = torch.stack(
        [s.action_info.buttons for s in rollout.steps], dim=0
    ).to(device)  # [L, 5]

    # Compute log probabilities
    # Main stick
    main_dist = Categorical(logits=main_logits)
    main_log_probs = main_dist.log_prob(main_actions)  # [L]

    # C-stick
    c_dist = Categorical(logits=c_logits)
    c_log_probs = c_dist.log_prob(c_actions)  # [L]

    # Shoulder
    shoulder_dist = Categorical(logits=shoulder_logits)
    shoulder_log_probs = shoulder_dist.log_prob(shoulder_actions)  # [L]

    # Buttons: compute log prob for each button independently
    button_probs = torch.sigmoid(button_logits)  # [L, 5]
    button_log_probs = torch.where(
        button_actions == 1,
        torch.log(button_probs + 1e-8),
        torch.log(1 - button_probs + 1e-8),
    )  # [L, 5]

    return {
        "main": main_log_probs,
        "c": c_log_probs,
        "buttons": button_log_probs,
        "shoulder": shoulder_log_probs,
    }


def compute_policy_entropy(outputs: TensorDict) -> Dict[str, float]:
    """Compute entropy for each policy component.

    Args:
        outputs: Model outputs

    Returns:
        Dict with entropy values for each component
    """
    # Extract logits
    main_logits = outputs["main_stick"][:, -1, :]  # [L, 64]
    c_logits = outputs["c_stick"][:, -1, :]  # [L, 9]
    button_logits = outputs["buttons"][:, -1, :]  # [L, 5]
    shoulder_logits = outputs["shoulder"][:, -1, :]  # [L, 5]

    # Categorical entropies
    main_dist = Categorical(logits=main_logits)
    main_entropy = main_dist.entropy().mean().item()

    c_dist = Categorical(logits=c_logits)
    c_entropy = c_dist.entropy().mean().item()

    shoulder_dist = Categorical(logits=shoulder_logits)
    shoulder_entropy = shoulder_dist.entropy().mean().item()

    # Button entropy (Bernoulli)
    button_probs = torch.sigmoid(button_logits)
    button_entropy = -(
        button_probs * torch.log(button_probs + 1e-8)
        + (1 - button_probs) * torch.log(1 - button_probs + 1e-8)
    ).mean().item()

    return {
        "main": main_entropy,
        "c": c_entropy,
        "buttons": button_entropy,
        "shoulder": shoulder_entropy,
    }


def compute_ppo_loss_with_gradient_accumulation(
    rollout: Rollout,
    model: GPT,
    column_map: ColumnMap,
    ppo_config: PPOConfig,
    optimizer: torch.optim.Optimizer,
    seq_len: int,
    amp_ctx,
    teacher_model: Optional[GPT] = None,
) -> PPOLossComponents:
    """Compute PPO loss with gradient accumulation across mini-batches.

    Only trains on steps that have full seq_len history (no padding needed).
    Windows are shuffled for better training dynamics.

    Args:
        rollout: Complete rollout with advantages and returns
        model: GPT model
        column_map: Feature column mapping
        ppo_config: PPO configuration
        optimizer: Optimizer used for updates
        seq_len: Sequence length for training windows
        teacher_model: Optional frozen teacher model for KL penalty
    """
    device = next(model.parameters()).device
    L = len(rollout.steps)

    # ===== IDENTIFY VALID TRAINING INDICES =====
    # Only train on steps that have full seq_len history
    # Step i is valid if we can get seq_len frames ending at i: [i-seq_len+1, i]
    # This requires (i - seq_len + 1) >= 0, i.e., i >= seq_len - 1
    first_valid_idx = seq_len - 1
    if first_valid_idx >= L:
        print(f"    [Warning] Rollout too short for training: {L} steps < {seq_len} seq_len")
        # Return dummy loss components
        return PPOLossComponents(
            total_loss=torch.tensor(0.0),
            policy_loss=torch.tensor(0.0),
            value_loss=torch.tensor(0.0),
            entropy=torch.tensor(0.0),
            main_policy_loss=torch.tensor(0.0),
            c_policy_loss=torch.tensor(0.0),
            buttons_policy_loss=torch.tensor(0.0),
            shoulder_policy_loss=torch.tensor(0.0),
            main_entropy=0.0, c_entropy=0.0, buttons_entropy=0.0, shoulder_entropy=0.0,
            main_clip_frac=0.0, c_clip_frac=0.0, buttons_clip_frac=0.0, shoulder_clip_frac=0.0,
            approx_kl=0.0, grad_norm=0.0,
        )

    valid_indices = list(range(first_valid_idx, L))
    num_valid = len(valid_indices)

    # Shuffle for better training dynamics (like imitation learning)
    random.shuffle(valid_indices)

    mini_batch_size = min(ppo_config.mini_batch_size, num_valid)
    num_batches = (num_valid + mini_batch_size - 1) // mini_batch_size

    # Accumulate metrics (scalars only, no tensors!)
    total_policy_losses = {"main": 0.0, "c": 0.0, "buttons": 0.0, "shoulder": 0.0}
    total_value_loss = 0.0
    total_entropy = {"main": 0.0, "c": 0.0, "buttons": 0.0, "shoulder": 0.0}
    total_clip_fracs = {"main": 0.0, "c": 0.0, "buttons": 0.0, "shoulder": 0.0}
    total_kl = 0.0
    total_teacher_kl = 0.0
    # Value function diagnostics
    total_value_pred_mean = 0.0
    total_value_pred_std = 0.0
    total_returns_mean = 0.0
    total_returns_std = 0.0
    total_uev = 0.0

    # ===== PRE-COMPUTE ALL FEATURES ONCE =====
    precompute_start = time.perf_counter()

    # Stack all features into a single contiguous tensor
    all_features = torch.stack([s.features for s in rollout.steps], dim=0)  # [L, F]

    precompute_elapsed = time.perf_counter() - precompute_start
    print(f"    [Precompute] Stacked {L} frames, {num_valid} valid training windows "
          f"(steps {first_valid_idx}-{L-1}) in {precompute_elapsed:.3f}s")

    for batch_idx in range(num_batches):
        batch_start = time.perf_counter()

        # Get shuffled indices for this batch
        batch_start_pos = batch_idx * mini_batch_size
        batch_end_pos = min(batch_start_pos + mini_batch_size, num_valid)
        batch_indices = valid_indices[batch_start_pos:batch_end_pos]
        batch_len = len(batch_indices)

        # Build sequences for this batch: for each valid index i, window is [i-seq_len+1 : i+1]
        sequences = []
        for idx in batch_indices:
            start = idx - seq_len + 1
            end = idx + 1
            seq = all_features[start:end]  # [seq_len, F]
            sequences.append(seq)
        features_batch = torch.stack(sequences, dim=0).to(device)  # [batch_len, seq_len, F]
        inputs_td = build_model_inputs(features_batch, column_map)

        build_time = time.perf_counter() - batch_start

        # Forward pass with AMP
        fwd_start = time.perf_counter()
        with autocast(
            device_type=amp_ctx.device_type, dtype=amp_ctx.dtype, enabled=amp_ctx.enabled
        ):
            outputs = model(inputs_td)

        # Teacher forward pass (if using teacher model)
        teacher_outputs = None
        if teacher_model is not None:
            with torch.no_grad(), autocast(
                device_type=amp_ctx.device_type, dtype=amp_ctx.dtype, enabled=amp_ctx.enabled
            ):
                teacher_outputs = teacher_model(inputs_td)
        fwd_time = time.perf_counter() - fwd_start

        # Gather steps, advantages, returns for this batch (using shuffled indices)
        batch_steps = [rollout.steps[i] for i in batch_indices]
        batch_advantages = rollout.advantages[batch_indices]
        batch_returns = rollout.returns[batch_indices]

        # Create mini-rollout for loss computation
        mini_rollout = Rollout(
            steps=batch_steps,
            episode_length=batch_len,
            total_reward=0.0,
            winner=0,
        )
        mini_rollout.advantages = batch_advantages
        mini_rollout.returns = batch_returns

        # Compute losses for this batch
        loss_start = time.perf_counter()
        batch_loss_components = _compute_batch_loss(
            outputs, mini_rollout, device, ppo_config, teacher_outputs
        )
        loss_time = time.perf_counter() - loss_start

        # Backward pass (gradient accumulation)
        bwd_start = time.perf_counter()
        batch_loss = batch_loss_components["total_loss"]
        (batch_loss / num_batches).backward()  # Normalize by num_batches
        bwd_time = time.perf_counter() - bwd_start

        # Accumulate metrics (scalars only)
        for key in total_policy_losses:
            total_policy_losses[key] += batch_loss_components["policy_losses"][key].item() * batch_len
        total_value_loss += batch_loss_components["value_loss"].item() * batch_len
        for key in total_entropy:
            total_entropy[key] += batch_loss_components["entropies"][key] * batch_len
        for key in total_clip_fracs:
            total_clip_fracs[key] += batch_loss_components["clip_fracs"][key] * batch_len
        total_kl += batch_loss_components["kl"] * batch_len
        total_teacher_kl += batch_loss_components["teacher_kl"] * batch_len
        # Value diagnostics
        total_value_pred_mean += batch_loss_components["value_pred_mean"] * batch_len
        total_value_pred_std += batch_loss_components["value_pred_std"] * batch_len
        total_returns_mean += batch_loss_components["returns_mean"] * batch_len
        total_returns_std += batch_loss_components["returns_std"] * batch_len
        total_uev += batch_loss_components["uev"] * batch_len

        batch_total = time.perf_counter() - batch_start
        print(f"      Batch {batch_idx+1}/{num_batches}: "
              f"build={build_time:.2f}s fwd={fwd_time:.2f}s loss={loss_time:.2f}s bwd={bwd_time:.2f}s "
              f"total={batch_total:.2f}s")

        # Free memory (let Python GC handle it - empty_cache() is expensive on MPS)
        del outputs, inputs_td, batch_loss_components, features_batch, sequences

    # Free stacked features
    del all_features

    # Average metrics over num_valid (not L)
    for key in total_policy_losses:
        total_policy_losses[key] /= num_valid
    total_value_loss /= num_valid
    for key in total_entropy:
        total_entropy[key] /= num_valid
    for key in total_clip_fracs:
        total_clip_fracs[key] /= num_valid
    total_kl /= num_valid
    total_teacher_kl /= num_valid
    # Value diagnostics
    total_value_pred_mean /= num_valid
    total_value_pred_std /= num_valid
    total_returns_mean /= num_valid
    total_returns_std /= num_valid
    total_uev /= num_valid

    # Return components (as scalars/floats for logging)
    total_policy_loss = sum(total_policy_losses.values())
    total_entropy_val = sum(total_entropy.values())
    total_loss = total_policy_loss + ppo_config.value_coef * total_value_loss - ppo_config.entropy_coef * total_entropy_val

    return PPOLossComponents(
        total_loss=torch.tensor(total_loss),  # Dummy tensor for compatibility
        policy_loss=torch.tensor(total_policy_loss),
        value_loss=torch.tensor(total_value_loss),
        entropy=torch.tensor(total_entropy_val),
        main_policy_loss=torch.tensor(total_policy_losses["main"]),
        c_policy_loss=torch.tensor(total_policy_losses["c"]),
        buttons_policy_loss=torch.tensor(total_policy_losses["buttons"]),
        shoulder_policy_loss=torch.tensor(total_policy_losses["shoulder"]),
        main_entropy=total_entropy["main"],
        c_entropy=total_entropy["c"],
        buttons_entropy=total_entropy["buttons"],
        shoulder_entropy=total_entropy["shoulder"],
        main_clip_frac=total_clip_fracs["main"],
        c_clip_frac=total_clip_fracs["c"],
        buttons_clip_frac=total_clip_fracs["buttons"],
        shoulder_clip_frac=total_clip_fracs["shoulder"],
        approx_kl=total_kl,
        grad_norm=0.0,
        # Value function diagnostics
        value_pred_mean=total_value_pred_mean,
        value_pred_std=total_value_pred_std,
        returns_mean=total_returns_mean,
        returns_std=total_returns_std,
        uev=total_uev,
        # Teacher KL
        teacher_kl=total_teacher_kl,
    )


def _compute_batch_loss(
    outputs: TensorDict,
    mini_rollout: Rollout,
    device: torch.device,
    ppo_config: PPOConfig,
    teacher_outputs: Optional[TensorDict] = None,
) -> dict:
    """Compute loss for a single mini-batch.

    Args:
        outputs: Model outputs for current policy
        mini_rollout: Mini-batch rollout with actions and advantages
        device: Computation device
        ppo_config: PPO configuration
        teacher_outputs: Optional outputs from frozen teacher model for KL penalty
    """
    # Compute new log probabilities
    new_log_probs = compute_log_probs_for_actions(outputs, mini_rollout, device)

    # Extract old log probabilities
    old_main_log_probs = torch.stack(
        [s.action_info.main_log_prob for s in mini_rollout.steps], dim=0
    ).to(device)
    old_c_log_probs = torch.stack(
        [s.action_info.c_log_prob for s in mini_rollout.steps], dim=0
    ).to(device)
    old_shoulder_log_probs = torch.stack(
        [s.action_info.shoulder_log_prob for s in mini_rollout.steps], dim=0
    ).to(device)
    old_buttons_log_probs = torch.stack(
        [s.action_info.buttons_log_probs for s in mini_rollout.steps], dim=0
    ).to(device)

    old_log_probs = {
        "main": old_main_log_probs,
        "c": old_c_log_probs,
        "buttons": old_buttons_log_probs,
        "shoulder": old_shoulder_log_probs,
    }

    advantages = mini_rollout.advantages.to(device)

    # Policy losses
    policy_losses = {}
    clip_fracs = {}

    for key in ["main", "c", "shoulder"]:
        ratio = torch.exp(new_log_probs[key] - old_log_probs[key])
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - ppo_config.clip_epsilon, 1.0 + ppo_config.clip_epsilon) * advantages
        policy_losses[key] = -torch.min(surr1, surr2).mean()
        clip_fracs[key] = (torch.abs(ratio - 1.0) > ppo_config.clip_epsilon).float().mean().item()

    # Buttons
    button_ratio = torch.exp(new_log_probs["buttons"] - old_log_probs["buttons"])
    button_surr1 = button_ratio * advantages.unsqueeze(-1)
    button_surr2 = torch.clamp(button_ratio, 1.0 - ppo_config.clip_epsilon, 1.0 + ppo_config.clip_epsilon) * advantages.unsqueeze(-1)
    policy_losses["buttons"] = -torch.min(button_surr1, button_surr2).mean()
    clip_fracs["buttons"] = (torch.abs(button_ratio - 1.0) > ppo_config.clip_epsilon).float().mean().item()

    # Value loss (with optional clipping)
    value_preds = outputs["value"][:, -1, 0]
    value_targets = mini_rollout.returns.to(device)

    if ppo_config.clip_value_loss:
        # Extract old value predictions from rollout
        old_values = torch.tensor(
            [s.value_pred for s in mini_rollout.steps], device=device, dtype=value_preds.dtype
        )
        # Clip value predictions to be within clip_epsilon of old values
        value_pred_clipped = old_values + torch.clamp(
            value_preds - old_values,
            -ppo_config.clip_epsilon,
            ppo_config.clip_epsilon,
        )
        # Use max of clipped and unclipped losses (pessimistic)
        value_losses = (value_preds - value_targets) ** 2
        value_losses_clipped = (value_pred_clipped - value_targets) ** 2
        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
    else:
        value_loss = ((value_preds - value_targets) ** 2).mean()

    # Value function diagnostics
    value_pred_mean = value_preds.mean().item()
    value_pred_std = value_preds.std().item()
    returns_mean = value_targets.mean().item()
    returns_std = value_targets.std().item()
    # UEV = value_loss / variance(returns), lower is better (0 = perfect, 1 = predicting mean)
    returns_var = value_targets.var().item() + 1e-8
    uev = value_loss.item() / returns_var

    # Entropy
    entropies = compute_policy_entropy(outputs)

    # KL divergence (improved approximation using Schulman's formula)
    kl = 0.0
    for k in ["main", "c", "shoulder"]:
        log_ratio = new_log_probs[k] - old_log_probs[k]
        ratio = torch.exp(log_ratio)
        kl += ((ratio - 1) - log_ratio).mean().item()  # More accurate KL approx
    # Buttons
    log_ratio_btn = new_log_probs["buttons"] - old_log_probs["buttons"]
    ratio_btn = torch.exp(log_ratio_btn)
    kl += ((ratio_btn - 1) - log_ratio_btn).mean().item()

    # Teacher KL penalty (if using teacher model)
    teacher_kl = 0.0
    if teacher_outputs is not None:
        teacher_log_probs = compute_log_probs_for_actions(teacher_outputs, mini_rollout, device)
        for k in ["main", "c", "shoulder"]:
            # Forward KL: KL(policy || teacher) = E_policy[log(policy) - log(teacher)]
            teacher_kl += (new_log_probs[k] - teacher_log_probs[k]).mean().item()
        teacher_kl += (new_log_probs["buttons"] - teacher_log_probs["buttons"]).mean().item()

    # Total loss
    total_policy_loss = sum(policy_losses.values())
    total_entropy = sum(entropies.values())
    total_loss = (
        total_policy_loss
        + ppo_config.value_coef * value_loss
        - ppo_config.entropy_coef * total_entropy
        + ppo_config.kl_teacher_weight * teacher_kl
    )

    return {
        "total_loss": total_loss,
        "teacher_kl": teacher_kl,
        "policy_losses": policy_losses,
        "value_loss": value_loss,
        "entropies": entropies,
        "clip_fracs": clip_fracs,
        "kl": kl,
        # Value diagnostics
        "value_pred_mean": value_pred_mean,
        "value_pred_std": value_pred_std,
        "returns_mean": returns_mean,
        "returns_std": returns_std,
        "uev": uev,
    }


def compute_ppo_loss(
    rollout: Rollout,
    model: GPT,
    column_map: ColumnMap,
    ppo_config: PPOConfig,
    seq_len: int,
    amp_ctx,
) -> PPOLossComponents:
    """Compute PPO loss for a rollout using mini-batches.

    Only trains on steps that have full seq_len history.
    Windows are shuffled for better training dynamics.

    Args:
        rollout: Complete rollout with advantages and returns
        model: GPT model
        column_map: Feature column mapping
        ppo_config: PPO configuration
        seq_len: Sequence length for training windows

    Returns:
        PPO loss components
    """
    device = next(model.parameters()).device
    L = len(rollout.steps)

    first_valid_idx = seq_len - 1
    if first_valid_idx >= L:
        print(f"    [Warning] Rollout too short for training: {L} steps < {seq_len} seq_len")
        return PPOLossComponents(
            total_loss=torch.tensor(0.0, device=device),
            policy_loss=torch.tensor(0.0, device=device),
            value_loss=torch.tensor(0.0, device=device),
            entropy=torch.tensor(0.0, device=device),
            main_policy_loss=torch.tensor(0.0, device=device),
            c_policy_loss=torch.tensor(0.0, device=device),
            buttons_policy_loss=torch.tensor(0.0, device=device),
            shoulder_policy_loss=torch.tensor(0.0, device=device),
            main_entropy=0.0, c_entropy=0.0, buttons_entropy=0.0, shoulder_entropy=0.0,
            main_clip_frac=0.0, c_clip_frac=0.0, buttons_clip_frac=0.0, shoulder_clip_frac=0.0,
            approx_kl=0.0, grad_norm=0.0,
        )

    valid_indices = list(range(first_valid_idx, L))
    num_valid = len(valid_indices)
    random.shuffle(valid_indices)

    mini_batch_size = min(ppo_config.mini_batch_size, num_valid)

    # Split rollout into mini-batches
    num_batches = (num_valid + mini_batch_size - 1) // mini_batch_size

    # Accumulate losses as tensors (for gradient tracking)
    accumulated_policy_losses = {"main": [], "c": [], "buttons": [], "shoulder": []}
    accumulated_value_losses = []
    accumulated_entropies = {"main": [], "c": [], "buttons": [], "shoulder": []}
    accumulated_clip_fracs = {"main": [], "c": [], "buttons": [], "shoulder": []}
    accumulated_kl = []
    batch_sizes = []

    all_features = torch.stack([s.features for s in rollout.steps], dim=0)  # [L, F]

    for batch_idx in range(num_batches):
        batch_start_pos = batch_idx * mini_batch_size
        batch_end_pos = min(batch_start_pos + mini_batch_size, num_valid)
        batch_indices = valid_indices[batch_start_pos:batch_end_pos]
        batch_len = len(batch_indices)

        # Reconstruct sequences for each timestep in the batch
        # For each step i, we need frames [i-seq_len+1 : i+1]
        # Use the configured rollout/training sequence length.
        sequences = []

        for idx in batch_indices:
            start_frame_idx = idx - seq_len + 1
            end_frame_idx = idx + 1
            sequences.append(all_features[start_frame_idx:end_frame_idx])

        features_batch = torch.stack(sequences, dim=0).to(device)  # [batch_len, seq_len, F]
        inputs_td = build_model_inputs(features_batch, column_map)

        # Forward pass with AMP
        with autocast(
            device_type=amp_ctx.device_type, dtype=amp_ctx.dtype, enabled=amp_ctx.enabled
        ):
            outputs = model(inputs_td)

        # Create a mini-rollout for this batch
        batch_steps = [rollout.steps[i] for i in batch_indices]
        mini_rollout = Rollout(
            steps=batch_steps,
            episode_length=batch_len,
            total_reward=0.0,
            winner=0,
        )
        mini_rollout.advantages = rollout.advantages[batch_indices]
        mini_rollout.returns = rollout.returns[batch_indices]

        # Compute new log probabilities for this batch
        new_log_probs = compute_log_probs_for_actions(outputs, mini_rollout, device)

        # Extract old log probabilities from batch
        old_main_log_probs = torch.stack(
            [s.action_info.main_log_prob for s in batch_steps], dim=0
        ).to(device)
        old_c_log_probs = torch.stack(
            [s.action_info.c_log_prob for s in batch_steps], dim=0
        ).to(device)
        old_shoulder_log_probs = torch.stack(
            [s.action_info.shoulder_log_prob for s in batch_steps], dim=0
        ).to(device)
        old_buttons_log_probs = torch.stack(
            [s.action_info.buttons_log_probs for s in batch_steps], dim=0
        ).to(device)

        old_log_probs = {
            "main": old_main_log_probs,
            "c": old_c_log_probs,
            "buttons": old_buttons_log_probs,
            "shoulder": old_shoulder_log_probs,
        }

        # Get advantages for this batch
        advantages = mini_rollout.advantages.to(device)

        # ===== POLICY LOSS (Clipped Surrogate Objective) =====
        for key in ["main", "c", "shoulder"]:
            ratio = torch.exp(new_log_probs[key] - old_log_probs[key])
            surr1 = ratio * advantages
            surr2 = torch.clamp(
                ratio,
                1.0 - ppo_config.clip_epsilon,
                1.0 + ppo_config.clip_epsilon,
            ) * advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            accumulated_policy_losses[key].append(policy_loss * batch_len)

            # Clip fraction (diagnostic, can be scalar)
            clip_frac = (torch.abs(ratio - 1.0) > ppo_config.clip_epsilon).float().mean()
            accumulated_clip_fracs[key].append(clip_frac * batch_len)

        # Buttons
        button_ratio = torch.exp(new_log_probs["buttons"] - old_log_probs["buttons"])
        button_surr1 = button_ratio * advantages.unsqueeze(-1)
        button_surr2 = torch.clamp(
            button_ratio,
            1.0 - ppo_config.clip_epsilon,
            1.0 + ppo_config.clip_epsilon,
        ) * advantages.unsqueeze(-1)
        button_policy_loss = -torch.min(button_surr1, button_surr2).mean()
        accumulated_policy_losses["buttons"].append(button_policy_loss * batch_len)

        clip_frac = (torch.abs(button_ratio - 1.0) > ppo_config.clip_epsilon).float().mean()
        accumulated_clip_fracs["buttons"].append(clip_frac * batch_len)

        # ===== VALUE LOSS (with optional clipping) =====
        value_preds = outputs["value"][:, -1, 0]
        value_targets = mini_rollout.returns.to(device)

        if ppo_config.clip_value_loss:
            # Extract old value predictions from rollout
            old_values = torch.tensor(
                [s.value_pred for s in batch_steps], device=device, dtype=value_preds.dtype
            )
            # Clip value predictions to be within clip_epsilon of old values
            value_pred_clipped = old_values + torch.clamp(
                value_preds - old_values,
                -ppo_config.clip_epsilon,
                ppo_config.clip_epsilon,
            )
            # Use max of clipped and unclipped losses (pessimistic)
            value_losses = (value_preds - value_targets) ** 2
            value_losses_clipped = (value_pred_clipped - value_targets) ** 2
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = ((value_preds - value_targets) ** 2).mean()
        accumulated_value_losses.append(value_loss * batch_len)

        # ===== ENTROPY =====
        entropies = compute_policy_entropy(outputs)
        for key in entropies:
            accumulated_entropies[key].append(entropies[key] * batch_len)

        # ===== KL DIVERGENCE =====
        kl = 0.5 * sum(
            ((new_log_probs[k] - old_log_probs[k]) ** 2).mean()
            for k in ["main", "c", "shoulder"]
        ) + 0.5 * ((new_log_probs["buttons"] - old_log_probs["buttons"]) ** 2).mean()
        accumulated_kl.append(kl * batch_len)

        batch_sizes.append(batch_len)

        # Free memory (let Python GC handle it - empty_cache() is expensive on MPS)
        del outputs, features_batch, inputs_td

    del all_features

    # Average across all timesteps by summing weighted losses
    total_policy_losses = {}
    for key in accumulated_policy_losses:
        total_policy_losses[key] = sum(accumulated_policy_losses[key]) / num_valid

    total_value_loss = sum(accumulated_value_losses) / num_valid

    total_entropy = {}
    for key in accumulated_entropies:
        total_entropy[key] = sum(accumulated_entropies[key]) / num_valid

    total_clip_fracs = {}
    for key in accumulated_clip_fracs:
        total_clip_fracs[key] = (sum(accumulated_clip_fracs[key]) / num_valid).item()

    total_kl = (sum(accumulated_kl) / num_valid).item()

    # Compute total loss (as tensor with grad)
    total_policy_loss = sum(total_policy_losses.values())
    total_entropy_val = sum(total_entropy.values())

    total_loss = (
        total_policy_loss
        + ppo_config.value_coef * total_value_loss
        - ppo_config.entropy_coef * total_entropy_val
    )

    return PPOLossComponents(
        total_loss=total_loss,
        policy_loss=total_policy_loss,
        value_loss=total_value_loss,
        entropy=total_entropy_val,
        main_policy_loss=total_policy_losses["main"],
        c_policy_loss=total_policy_losses["c"],
        buttons_policy_loss=total_policy_losses["buttons"],
        shoulder_policy_loss=total_policy_losses["shoulder"],
        main_entropy=total_entropy["main"],
        c_entropy=total_entropy["c"],
        buttons_entropy=total_entropy["buttons"],
        shoulder_entropy=total_entropy["shoulder"],
        main_clip_frac=total_clip_fracs["main"],
        c_clip_frac=total_clip_fracs["c"],
        buttons_clip_frac=total_clip_fracs["buttons"],
        shoulder_clip_frac=total_clip_fracs["shoulder"],
        approx_kl=total_kl,
        grad_norm=0.0,  # Will be filled in after backward pass
    )


# ============================================================================
# Metrics and Logging
# ============================================================================

def compute_discounted_returns(
    rewards: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute Monte Carlo discounted returns from per-step rewards."""
    if rewards.numel() == 0:
        return rewards
    returns = torch.zeros_like(rewards)
    running = 0.0
    for idx in range(rewards.numel() - 1, -1, -1):
        running = rewards[idx].item() + gamma * running
        returns[idx] = running
    return returns


def print_ppo_metrics(
    episode_count: int,
    rollout: Rollout,
    loss_components: PPOLossComponents,
):
    """Print detailed PPO training metrics.

    Args:
        episode_count: Current rollout number (each rollout = one stock taken)
        rollout: Rollout data (one stock exchange)
        loss_components: Loss components from training
    """
    # Each "rollout" ends when a stock is taken (by either player)
    stock_taker = "Bot" if rollout.winner == BOT_PORT else "CPU"

    print(f"\n{'='*80}")
    print(f"Rollout {episode_count:04d} | Length: {rollout.episode_length:3d} frames | "
          f"Reward: {rollout.total_reward:+7.2f} | Stock taken by: {stock_taker}")
    print(f"{'='*80}")

    # Loss components
    print(f"Losses:")
    print(f"  Total:  {loss_components.total_loss.item():7.4f}")
    print(f"  Policy: {loss_components.policy_loss.item():7.4f} "
          f"(main={loss_components.main_policy_loss.item():.4f}, "
          f"c={loss_components.c_policy_loss.item():.4f}, "
          f"btn={loss_components.buttons_policy_loss.item():.4f}, "
          f"shoulder={loss_components.shoulder_policy_loss.item():.4f})")
    print(f"  Value:  {loss_components.value_loss.item():7.4f}")
    print(f"  Entropy:{loss_components.entropy.item():7.4f} "
          f"(main={loss_components.main_entropy:.4f}, "
          f"c={loss_components.c_entropy:.4f}, "
          f"btn={loss_components.buttons_entropy:.4f}, "
          f"shoulder={loss_components.shoulder_entropy:.4f})")

    rewards = torch.tensor([s.reward for s in rollout.steps], dtype=torch.float32)
    gamma = get_config().reward.gamma
    discounted_returns = compute_discounted_returns(rewards, gamma)
    reward_nonzero = int((rewards != 0).sum().item()) if rewards.numel() > 0 else 0
    reward_mean = rewards.mean().item() if rewards.numel() > 0 else 0.0
    reward_abs_mean = rewards.abs().mean().item() if rewards.numel() > 0 else 0.0
    reward_min = rewards.min().item() if rewards.numel() > 0 else 0.0
    reward_max = rewards.max().item() if rewards.numel() > 0 else 0.0

    disc_total = discounted_returns[0].item() if discounted_returns.numel() > 0 else 0.0
    disc_mean = discounted_returns.mean().item() if discounted_returns.numel() > 0 else 0.0
    disc_min = discounted_returns.min().item() if discounted_returns.numel() > 0 else 0.0
    disc_max = discounted_returns.max().item() if discounted_returns.numel() > 0 else 0.0

    print("Rewards:")
    print(f"  Non-zero: {reward_nonzero:4d}/{rollout.episode_length:4d} | "
          f"Mean: {reward_mean:+.6f} | Abs mean: {reward_abs_mean:.6f} | "
          f"Min/Max: {reward_min:+.4f}/{reward_max:+.4f}")
    print(f"  Discounted (gamma={gamma:.5f}): "
          f"Total: {disc_total:+.4f} | Mean: {disc_mean:+.6f} | "
          f"Min/Max: {disc_min:+.4f}/{disc_max:+.4f}")

    # Value function diagnostics
    print(f"Value Function:")
    print(f"  Predictions: mean={loss_components.value_pred_mean:+.4f}, std={loss_components.value_pred_std:.4f}")
    print(f"  Returns:     mean={loss_components.returns_mean:+.4f}, std={loss_components.returns_std:.4f}")
    uev_quality = "good" if loss_components.uev < 0.5 else "ok" if loss_components.uev < 1.0 else "poor"
    print(f"  UEV:         {loss_components.uev:.4f} ({uev_quality}, lower is better)")

    # Raw advantage statistics
    print(f"Advantages (raw, before normalization):")
    print(f"  Mean: {rollout.raw_advantage_mean:+.4f}, Std: {rollout.raw_advantage_std:.4f}")
    print(f"  Min/Max: {rollout.raw_advantage_min:+.4f}/{rollout.raw_advantage_max:+.4f}")

    # Diagnostics
    print(f"Diagnostics:")
    print(f"  Clip Fractions: main={loss_components.main_clip_frac:.3f}, "
          f"c={loss_components.c_clip_frac:.3f}, "
          f"btn={loss_components.buttons_clip_frac:.3f}, "
          f"shoulder={loss_components.shoulder_clip_frac:.3f}")
    print(f"  Approx KL:      {loss_components.approx_kl:.6f}")
    print(f"  Grad Norm:      {loss_components.grad_norm:.4f}")
    if loss_components.teacher_kl != 0.0:
        print(f"  Teacher KL:     {loss_components.teacher_kl:+.6f}")

    print(f"{'='*80}\n")


# Button names for diagnostics
BUTTON_NAMES = ["A", "B", "X/Y", "Z", "L/R"]


def print_action_diagnostics(rollout: Rollout):
    """Print detailed action diagnostics for a rollout (one stock exchange).

    Shows:
    - Button press statistics (count and percentage for each button)
    - Shoulder value distribution (percentage for each of 5 values)
    - Top 4 stick positions for main and c stick (in [0,1] domain)

    Args:
        rollout: Rollout data (one stock exchange) containing steps with action_info
    """
    if not rollout.steps:
        return

    n_frames = len(rollout.steps)

    # ===== BUTTON STATISTICS =====
    button_counts = [0] * 5
    for step in rollout.steps:
        buttons = step.action_info.buttons
        for i in range(5):
            if buttons[i].item() > 0.5:
                button_counts[i] += 1

    print("Action Diagnostics:")
    print("  Buttons:")
    for i, name in enumerate(BUTTON_NAMES):
        count = button_counts[i]
        pct = 100.0 * count / n_frames if n_frames > 0 else 0.0
        print(f"    {name:4s}: {count:4d} frames ({pct:5.1f}%)")

    # ===== SHOULDER STATISTICS =====
    shoulder_counts = [0] * 5  # 5 discrete values
    for step in rollout.steps:
        idx = step.action_info.shoulder_idx
        if 0 <= idx < 5:
            shoulder_counts[idx] += 1

    print("  Shoulder:")
    shoulder_values = [0.0, 0.31, 0.42, 0.55, 1.0]
    for i, val in enumerate(shoulder_values):
        count = shoulder_counts[i]
        pct = 100.0 * count / n_frames if n_frames > 0 else 0.0
        print(f"    {val:.2f}: {count:4d} frames ({pct:5.1f}%)")

    # ===== MAIN STICK STATISTICS =====
    from collections import Counter
    main_stick_counts = Counter()
    for step in rollout.steps:
        idx = step.action_info.main_stick_idx
        main_stick_counts[idx] += 1

    # Get top 4 positions
    top_main = main_stick_counts.most_common(4)
    print("  Main Stick (top 4):")
    for idx, count in top_main:
        pct = 100.0 * count / n_frames if n_frames > 0 else 0.0
        # Get stick position in [-1, 1] then convert to [0, 1]
        if idx < len(CONTROL_STICK_QUANTIZED):
            x, y = CONTROL_STICK_QUANTIZED[idx]
            x_01 = (x + 1.0) / 2.0
            y_01 = (y + 1.0) / 2.0
            print(f"    idx {idx:2d}: ({x_01:.2f}, {y_01:.2f}) - {count:4d} frames ({pct:5.1f}%)")
        else:
            print(f"    idx {idx:2d}: (unknown) - {count:4d} frames ({pct:5.1f}%)")

    # ===== C STICK STATISTICS =====
    c_stick_counts = Counter()
    for step in rollout.steps:
        idx = step.action_info.c_stick_idx
        c_stick_counts[idx] += 1

    # Get top 4 positions
    top_c = c_stick_counts.most_common(4)
    print("  C Stick (top 4):")
    for idx, count in top_c:
        pct = 100.0 * count / n_frames if n_frames > 0 else 0.0
        # Get stick position in [-1, 1] then convert to [0, 1]
        if idx < len(C_STICK_QUANTIZED):
            x, y = C_STICK_QUANTIZED[idx]
            x_01 = (x + 1.0) / 2.0
            y_01 = (y + 1.0) / 2.0
            print(f"    idx {idx:2d}: ({x_01:.2f}, {y_01:.2f}) - {count:4d} frames ({pct:5.1f}%)")
        else:
            print(f"    idx {idx:2d}: (unknown) - {count:4d} frames ({pct:5.1f}%)")

    print()


def save_ppo_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    episode_count: int,
    checkpoint_dir: Path,
    config,
):
    """Save PPO training checkpoint.

    Args:
        model: GPT model
        optimizer: Optimizer
        episode_count: Current episode count
        checkpoint_dir: Directory to save checkpoint
        config: Configuration object
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"ppo_ep{episode_count:04d}.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "episode": episode_count,
        "config": config.model_dump() if hasattr(config, "model_dump") else None,
    }, checkpoint_path)

    print(f"Saved checkpoint: {checkpoint_path}")


def find_latest_ppo_checkpoint(directory: Path) -> Optional[Path]:
    """Find the newest PPO checkpoint in directory.

    Args:
        directory: Directory to search for PPO checkpoints.

    Returns:
        Path to the most recent PPO checkpoint or None if not found.
    """
    directory = directory.expanduser()
    if not directory.exists():
        return None
    candidates = [p for p in directory.glob("ppo_ep*.pt") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_ppo_checkpoint(
    checkpoint_path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    """Load PPO training checkpoint.

    Args:
        checkpoint_path: Path to PPO checkpoint file.
        model: GPT model to load weights into.
        optimizer: Optimizer to load state into.
        device: Device to load tensors to.

    Returns:
        Episode count from checkpoint.
    """
    print(f"Loading PPO checkpoint: {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model weights (handle torch.compile prefix mismatch)
    model_state = checkpoint_data["model_state_dict"]
    model_state = match_state_dict_keys(model_state, model)
    model.load_state_dict(model_state)

    # Load optimizer state
    optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

    episode_count = checkpoint_data.get("episode", 0)
    print(f"Resumed from episode {episode_count}")

    return episode_count


# ============================================================================
# Main Training Loop
# ============================================================================

def main():
    """Main PPO training loop."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="PPO training for Melee-AI")
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=None,
        help="Path to initial checkpoint (defaults to latest in checkpoints/)",
    )
    parser.add_argument(
        "--iso",
        default=None,
        type=str,
        help="Path to Melee ISO",
    )
    parser.add_argument(
        "--dolphin_executable_path",
        "-e",
        default=None,
        help="Path to Dolphin executable",
    )
    parser.add_argument(
        "--rollout-mode",
        choices=("stochastic", "greedy"),
        default="stochastic",
        help="Action selection during rollouts: stochastic (policy sampling) or greedy (argmax).",
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=Path,
        default=None,
        help="Path to PPO checkpoint to resume from (or 'latest' to find most recent in checkpoints_ppo/)",
    )
    parser.add_argument(
        "--teacher",
        "-t",
        type=Path,
        default=None,
        help="Path to teacher model checkpoint for KL penalty (defaults to --checkpoint)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start from a fresh untrained model (random weights) instead of a checkpoint",
    )
    args, remaining = parser.parse_known_args()

    # Parse CLI overrides
    overrides = parse_cli_overrides(remaining)

    # Handle --fresh flag vs checkpoint loading
    if args.fresh:
        # ===== FRESH MODEL: Start from random weights =====
        print("\n" + "=" * 60)
        print("FRESH MODEL - Starting from random weights!")
        print("=" * 60)

        # Initialize config with defaults (not from checkpoint)
        init_config(overrides=overrides)
        config = get_config()
        checkpoint_path = None

        # Create fresh model
        model = GPT(config)

        # Determine device
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        model = model.to(device)
        model.train()
        print(f"Fresh model created. Device: {device}")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create column map from default feature/target names
        from model_interface import _DEFAULT_FEATURE_NAMES, _DEFAULT_TARGET_NAMES
        feature_names = list(_DEFAULT_FEATURE_NAMES)
        target_names = list(_DEFAULT_TARGET_NAMES)
        colmap = ColumnMap(feature_names, target_names)

        # Create a minimal engine-like object for inference
        class FreshEngine:
            def __init__(self, model, colmap, device, seq_len=256):
                self.model = model
                self.colmap = colmap
                self.device = device
                self.seq_len = seq_len
                self.warmup_frames = 256
                self.buffer: deque[torch.Tensor] = deque(maxlen=seq_len)
                self.frame_history: deque = deque(maxlen=seq_len)
                self.feature_names = colmap.feat_names
                self.feature_dim = len(colmap.feat_names)

            def prepare_inputs(self, raw_inputs):
                """Prepare model inputs from transformed features dict."""
                # Convert dict to tensor (same as GPTInferenceEngine._frame_to_tensor)
                frame = torch.zeros(self.feature_dim, dtype=torch.float32)
                for idx, name in enumerate(self.feature_names):
                    frame[idx] = raw_inputs[name]

                self.buffer.append(frame)

                if len(self.buffer) < self.warmup_frames:
                    return None

                # Stack and build model inputs
                stacked = torch.stack(list(self.buffer), dim=0)  # [seq_len, F]
                batch = stacked.unsqueeze(0).to(self.device)  # [1, seq_len, F]

                from train.batch_utils import build_model_inputs
                return build_model_inputs(batch, self.colmap)

        engine = FreshEngine(model, colmap, device, seq_len=config.seq_len)

    else:
        # ===== CHECKPOINT MODEL: Load from existing checkpoint =====
        checkpoint_path = args.checkpoint
        if checkpoint_path is None:
            default_dir = Path("checkpoints")
            latest = find_latest_checkpoint(default_dir)
            if latest is None:
                raise FileNotFoundError(
                    f"No checkpoint provided and none found in {default_dir.resolve()}"
                )
            checkpoint_path = latest

        # Initialize config FROM CHECKPOINT
        init_config_from_checkpoint(checkpoint_path, overrides=overrides)
        config = get_config()

        # Load model from checkpoint
        print(f"Loading model from: {checkpoint_path}")
        engine = GPTInferenceEngine(checkpoint_path=checkpoint_path)
        model = engine.model
        model.train()  # Switch to training mode
        colmap = engine.colmap

        device = next(model.parameters()).device
        print(f"Model loaded. Device: {device}")

    # Print reward configuration
    rc = config.reward
    print("\n" + "=" * 60)
    print("TECH-ONLY REWARD EXPERIMENT")
    print("=" * 60)
    print("  Reward function: compute_tech_rewards()")
    print("  +1 for successful tech (neutral, forward, backward, wall, ceiling)")
    print("  -1 for missed tech (DownBoundU, DownBoundD)")
    print("  Zero-sum: ego rewards minus opponent rewards")
    print("")
    print("  Settings:")
    print(f"    gamma:            {rc.gamma}")
    print(f"    max_kl:           DISABLED (inf)")
    print(f"    kl_teacher_weight: DISABLED (0.0)")
    if args.fresh:
        print("    model:            FRESH (random weights)")
    else:
        print(f"    model:            {checkpoint_path}")
    print("=" * 60 + "\n")

    ppo_config = PPOConfig()
    seq_len = config.seq_len
    if seq_len != 256:
        raise ValueError(
            f"Expected config.seq_len=256 for PPO, got {seq_len}."
        )

    # Load frozen teacher model for KL penalty (if enabled)
    teacher_model: Optional[GPT] = None
    if ppo_config.kl_teacher_weight > 0:
        teacher_checkpoint = args.teacher if args.teacher is not None else checkpoint_path
        print(f"Loading frozen teacher model for KL penalty (weight={ppo_config.kl_teacher_weight})...")
        print(f"  Teacher checkpoint: {teacher_checkpoint}")
        # Create a new model instance with the same config
        teacher_model = GPT(config)
        # Load checkpoint weights (handle torch.compile prefix mismatch)
        checkpoint_data = torch.load(teacher_checkpoint, map_location=device, weights_only=False)
        model_state = checkpoint_data.get("model", checkpoint_data)
        model_state = match_state_dict_keys(model_state, teacher_model)
        teacher_model.load_state_dict(model_state)
        teacher_model.to(device)
        teacher_model.eval()  # Always in eval mode
        # Freeze all parameters
        for param in teacher_model.parameters():
            param.requires_grad = False
        print("Teacher model loaded and frozen.")

    # Configure AMP for mixed precision training
    # Override config's use_amp with PPO's setting
    original_use_amp = config.train.use_amp
    config.train.use_amp = ppo_config.use_amp
    amp_ctx = configure_amp(config, device)
    config.train.use_amp = original_use_amp  # Restore

    if seq_len > model.block_size:
        raise ValueError(
            f"Sequence length {seq_len} exceeds model.block_size {model.block_size}."
        )

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ppo_config.learning_rate,
        eps=1e-5,
    )

    # Handle PPO checkpoint resume
    resume_episode = 0
    if args.resume is not None:
        # Handle 'latest' as special value
        if str(args.resume) == "latest":
            resume_path = find_latest_ppo_checkpoint(ppo_config.checkpoint_dir)
            if resume_path is None:
                raise FileNotFoundError(
                    f"No PPO checkpoint found in {ppo_config.checkpoint_dir.resolve()}"
                )
        else:
            resume_path = args.resume
            if not resume_path.exists():
                raise FileNotFoundError(f"PPO checkpoint not found: {resume_path}")

        resume_episode = load_ppo_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            device=device,
        )

    # Setup Dolphin console
    console = Console(
        path=args.dolphin_executable_path,
        slippi_address="127.0.0.1",
        save_replays=False,
        copy_home_directory=False,
        tmp_home_directory=False,
        blocking_input=True,
    )

    # Setup controllers
    controllers = {
        BOT_PORT: Controller(
            console=console,
            port=BOT_PORT,
            type=ControllerType.STANDARD,
        ),
        OPP_PORT: Controller(
            console=console,
            port=OPP_PORT,
            type=ControllerType.STANDARD,
        ),
    }

    # Signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\nShutting down...")
        for controller in controllers.values():
            controller.disconnect()
        console.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start console
    console.run(iso_path=args.iso)

    # Connect to console
    print("Connecting to console...")
    if not console.connect():
        print("ERROR: Failed to connect to the console.")
        sys.exit(-1)
    print("Console connected")

    # Connect controllers
    for controller in controllers.values():
        if not controller.connect():
            print("ERROR: Failed to connect controller.")
            sys.exit(-1)
    print("Controllers connected")

    # Training loop
    episode_count = resume_episode
    training_stats = TrainingStats.create(window_size=100)

    # Random stage and characters
    current_stage = random.choice(LEGAL_TOURNAMENT_STAGES)
    bot_char = random.choice(SUPPORTED_CHARS)
    opp_char = random.choice(SUPPORTED_CHARS)

    print(f"\nStarting PPO training...")
    print(f"Stage: {current_stage}, Bot: {bot_char}, Opponent: {opp_char}")
    print(f"PPO Config: {ppo_config}\n")

    while True:
        # ===== 1. COLLECT ROLLOUT =====
        model.eval()
        rollout = collect_rollout(
            console=console,
            controllers=controllers,
            engine=engine,
            model=model,
            column_map=colmap,
            reward_config=config.reward,
            current_stage=current_stage,
            bot_char=bot_char,
            opp_char=opp_char,
            action_selection=args.rollout_mode,
            amp_ctx=amp_ctx,
        )
        model.train()

        episode_count += 1

        # Check if episode is long enough for meaningful training
        if rollout.episode_length < MIN_EPISODE_LENGTH:
            print(f"Rollout {episode_count} too short ({rollout.episode_length} < {MIN_EPISODE_LENGTH} frames), skipping...")
            continue

        # ===== 2. COMPUTE ADVANTAGES =====
        print(f"[Advantages] Computing for {rollout.episode_length} frames...")
        gae_start = time.perf_counter()

        rewards = torch.tensor(
            [s.reward for s in rollout.steps], dtype=torch.float32
        )
        values = torch.tensor(
            [s.value_pred for s in rollout.steps], dtype=torch.float32
        )

        adv_result = compute_advantages(
            rewards=rewards,
            values=values,
            gamma=config.reward.gamma,
            normalize=ppo_config.normalize_advantages,
        )

        rollout.advantages = adv_result.advantages
        rollout.returns = adv_result.returns
        rollout.raw_advantage_mean = adv_result.raw_advantage_mean
        rollout.raw_advantage_std = adv_result.raw_advantage_std
        rollout.raw_advantage_min = adv_result.raw_advantage_min
        rollout.raw_advantage_max = adv_result.raw_advantage_max
        adv_elapsed = time.perf_counter() - gae_start
        print(f"[Advantages] Done in {adv_elapsed:.2f}s "
              f"(raw: mean={adv_result.raw_advantage_mean:+.4f}, "
              f"std={adv_result.raw_advantage_std:.4f})")

        # ===== 3. PPO TRAINING (K epochs) =====
        first_valid_idx = seq_len - 1
        num_valid = max(0, rollout.episode_length - first_valid_idx)
        if num_valid > 0:
            num_batches = (num_valid + ppo_config.mini_batch_size - 1) // ppo_config.mini_batch_size
        else:
            num_batches = 0
        print(f"[Training] Starting {ppo_config.num_epochs} PPO epochs "
              f"({num_batches} batches/epoch, batch_size={ppo_config.mini_batch_size}, "
              f"valid_windows={num_valid})")
        train_start = time.perf_counter()

        loss_components = None
        for epoch in range(ppo_config.num_epochs):
            epoch_start = time.perf_counter()

            # Process rollout in mini-batches with gradient accumulation
            optimizer.zero_grad()

            loss_components = compute_ppo_loss_with_gradient_accumulation(
                rollout=rollout,
                model=model,
                column_map=colmap,
                ppo_config=ppo_config,
                optimizer=optimizer,
                seq_len=seq_len,
                amp_ctx=amp_ctx,
                teacher_model=teacher_model,
            )

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=ppo_config.grad_clip,
            )
            loss_components.grad_norm = grad_norm.item()

            optimizer.step()

            epoch_elapsed = time.perf_counter() - epoch_start
            print(f"  Epoch {epoch+1}/{ppo_config.num_epochs}: "
                  f"loss={loss_components.total_loss.item():.4f}, "
                  f"policy={loss_components.policy_loss.item():.4f}, "
                  f"value={loss_components.value_loss.item():.4f}, "
                  f"kl={loss_components.approx_kl:.4f}, "
                  f"time={epoch_elapsed:.2f}s")

            # KL-based early stopping
            if loss_components.approx_kl > ppo_config.max_kl:
                print(f"  [Early Stop] KL {loss_components.approx_kl:.4f} > max_kl {ppo_config.max_kl}")
                break

        train_elapsed = time.perf_counter() - train_start
        epochs_completed = epoch + 1
        print(f"[Training] Complete in {train_elapsed:.2f}s ({epochs_completed}/{ppo_config.num_epochs} epochs)")

        # ===== 4. LOGGING =====
        if loss_components is not None:
            training_stats.update(rollout, loss_components, BOT_PORT)

        if episode_count % ppo_config.log_interval == 0 and loss_components is not None:
            print_ppo_metrics(episode_count, rollout, loss_components)
            print_action_diagnostics(rollout)
            print(f"Training Progress: {training_stats.summary()}")
            print()

        # ===== MEMORY CLEANUP =====
        # Note: Don't call torch.mps.empty_cache() here - it's expensive and
        # unnecessary on MPS. The caching allocator handles memory efficiently.
        # Only call empty_cache() if you encounter actual OOM errors.

        # Delete rollout and loss components to free memory
        del rollout
        if loss_components is not None:
            del loss_components

        # ===== 5. CHECKPOINTING =====
        if episode_count % ppo_config.checkpoint_interval == 0:
            save_ppo_checkpoint(
                model=model,
                optimizer=optimizer,
                episode_count=episode_count,
                checkpoint_dir=ppo_config.checkpoint_dir,
                config=config,
            )

        # Pick new stage/character for next episode
        current_stage = random.choice(LEGAL_TOURNAMENT_STAGES)
        bot_char = random.choice(SUPPORTED_CHARS)
        opp_char = random.choice(SUPPORTED_CHARS)


if __name__ == "__main__":
    main()
