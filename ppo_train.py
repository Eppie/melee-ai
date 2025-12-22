#!/usr/bin/env python3
"""Simple PPO training script for Melee-AI.

Trains a GPT model to play Super Smash Bros. Melee using Proximal Policy Optimization.
Self-contained single-file implementation with:
- Single Dolphin emulator instance
- Fox vs CPU Fox (level 9)
- Episodes end on stock loss
- Online learning (train after each episode)
- 4 PPO epochs per rollout
- Checkpoint saving every N episodes
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

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.amp import autocast
from torch.distributions import Bernoulli, Categorical

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from column_map import ColumnMap
from config.config import init_config, get_config
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
)


# ============================================================================
# Configuration and Constants
# ============================================================================

@dataclass
class PPOConfig:
    """PPO-specific hyperparameters."""

    # Learning
    learning_rate: float = 3e-4

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

    # GAE parameters
    gae_lambda: float = 0.95  # Lambda for GAE

    # Training stability
    grad_clip: float = 0.5  # Gradient clipping max norm
    normalize_advantages: bool = True  # Normalize advantages

    # Checkpointing
    checkpoint_interval: int = 10  # Save checkpoint every N episodes
    checkpoint_dir: Path = Path("checkpoints_ppo")

    # Logging
    log_interval: int = 1  # Log every episode


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
    """Complete episode rollout."""

    steps: List[RolloutStep]
    episode_length: int
    total_reward: float
    winner: int  # 1 or 2

    # Computed after collection
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None


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


def action_info_to_controller_state(
    action_info: ActionInfo,
    engine: GPTInferenceEngine,
) -> ControllerState:
    """Convert ActionInfo to ControllerState for game application.

    Args:
        action_info: Sampled actions
        engine: Inference engine with quantization tables

    Returns:
        ControllerState with normalized [0, 1] values
    """
    # Convert stick indices to coordinates using quantization tables
    main_coords = CONTROL_STICK_QUANTIZED[action_info.main_stick_idx]
    c_coords = C_STICK_QUANTIZED[action_info.c_stick_idx]
    shoulder_val = SHOULDER_QUANTIZED[action_info.shoulder_idx]

    # Convert to Dolphin [0, 1] coordinates
    main_x, main_y = model_to_dolphin01(main_coords)
    c_x, c_y = model_to_dolphin01(c_coords)

    # Extract button values
    buttons = action_info.buttons

    return ControllerState(
        main_stick_x=main_x,
        main_stick_y=main_y,
        c_stick_x=c_x,
        c_stick_y=c_y,
        shoulder_analog=shoulder_val,
        button_a=bool(buttons[0].item()),
        button_b=bool(buttons[1].item()),
        button_xy=bool(buttons[2].item()),
        button_z=bool(buttons[3].item()),
        button_lr=bool(buttons[4].item()),
    )


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
    """Collect a complete episode rollout.

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
                # Episode ended during warmup (very rare)
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
        controller_state = action_info_to_controller_state(action_info, engine)
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
    warmup = engine.warmup_frames
    total_frames = frame_count
    print(f"\n[Episode ended] Total: {total_frames} frames ({warmup} warmup + {len(steps)} collected)")
    print("[Post-process] Computing rewards...")
    post_start = time.perf_counter()
    device = next(model.parameters()).device
    rewards = torch.zeros(len(steps), dtype=torch.float32)
    if steps:
        features = torch.stack([s.features for s in steps], dim=0)  # [L, F]
        features_device = features.to(device)
        rewards_tensor = compute_frame_rewards(
            features_device.unsqueeze(0), reward_idx, reward_config
        )
        rewards = rewards_tensor[0].detach().cpu()
        components = compute_reward_components(
            features_device.unsqueeze(0), reward_idx, reward_config
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
        print("[Reward breakdown]")
        for key in ("damage", "stock", "hitlag", "low_shield", "hitstun"):
            values = components_cpu.get(key)
            if values is None:
                continue
            nonzero = int((values != 0).sum().item())
            total = float(values.sum().item())
            abs_total = float(values.abs().sum().item())
            print(
                f"  {key:>10}: sum={total:+.4f} | "
                f"abs_sum={abs_total:.4f} | nonzero={nonzero:4d}"
            )
        total_components = float(components_cpu["total"].sum().item())
        print(f"  {'total':>10}: sum={total_components:+.4f}")

    return Rollout(
        steps=steps,
        episode_length=len(steps),
        total_reward=total_reward,
        winner=winner or BOT_PORT,  # Default to bot if no clear winner
    )


# ============================================================================
# Advantage Computation (GAE)
# ============================================================================

def compute_gae_advantages(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    normalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE) advantages and returns.

    Args:
        rewards: [L] per-frame rewards
        values: [L] value predictions
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        normalize: Whether to normalize advantages

    Returns:
        Tuple of (advantages [L], returns [L])
    """
    L = len(rewards)
    advantages = torch.zeros(L, device=rewards.device)

    # Compute TD errors: δ_t = r_t + γV(s_{t+1}) - V(s_t)
    deltas = torch.zeros(L, device=rewards.device)
    deltas[:-1] = rewards[:-1] + gamma * values[1:] - values[:-1]
    deltas[-1] = rewards[-1] - values[-1]  # Terminal state

    # Compute GAE: A_t = Σ_{i=0}^{∞} (γλ)^i δ_{t+i}
    gae = 0.0
    for t in reversed(range(L)):
        gae = deltas[t] + gamma * gae_lambda * gae
        advantages[t] = gae

    # Compute returns: R_t = A_t + V(s_t)
    returns = advantages + values

    # Normalize advantages (critical for stability)
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


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
    """
    device = next(model.parameters()).device
    L = len(rollout.steps)

    # ===== IDENTIFY VALID TRAINING INDICES =====
    # Only train on steps that have full seq_len history
    # Step i is valid if we can get seq_len frames ending at i: [i-seq_len+1, i]
    # This requires (i - seq_len + 1) >= 0, i.e., i >= seq_len - 1
    first_valid_idx = seq_len - 1
    if first_valid_idx >= L:
        print(f"    [Warning] Episode too short for training: {L} steps < {seq_len} seq_len")
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
            outputs, mini_rollout, device, ppo_config
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
    )


def _compute_batch_loss(
    outputs: TensorDict,
    mini_rollout: Rollout,
    device: torch.device,
    ppo_config: PPOConfig,
) -> dict:
    """Compute loss for a single mini-batch."""
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

    # Value loss
    value_preds = outputs["value"][:, -1, 0]
    value_targets = mini_rollout.returns.to(device)
    value_loss = ((value_preds - value_targets) ** 2).mean()

    # Entropy
    entropies = compute_policy_entropy(outputs)

    # KL divergence
    kl = 0.5 * sum(
        ((new_log_probs[k] - old_log_probs[k]) ** 2).mean().item()
        for k in ["main", "c", "shoulder"]
    ) + 0.5 * ((new_log_probs["buttons"] - old_log_probs["buttons"]) ** 2).mean().item()

    # Total loss
    total_policy_loss = sum(policy_losses.values())
    total_entropy = sum(entropies.values())
    total_loss = total_policy_loss + ppo_config.value_coef * value_loss - ppo_config.entropy_coef * total_entropy

    return {
        "total_loss": total_loss,
        "policy_losses": policy_losses,
        "value_loss": value_loss,
        "entropies": entropies,
        "clip_fracs": clip_fracs,
        "kl": kl,
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
        print(f"    [Warning] Episode too short for training: {L} steps < {seq_len} seq_len")
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

        # ===== VALUE LOSS =====
        value_preds = outputs["value"][:, -1, 0]
        value_targets = mini_rollout.returns.to(device)
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
        episode_count: Current episode number
        rollout: Episode rollout
        loss_components: Loss components from training
    """
    winner_str = f"Bot (P{BOT_PORT})" if rollout.winner == BOT_PORT else f"CPU (P{OPP_PORT})"

    print(f"\n{'='*80}")
    print(f"Episode {episode_count:04d} | Length: {rollout.episode_length:3d} frames | "
          f"Reward: {rollout.total_reward:+7.2f} | Winner: {winner_str}")
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

    # Diagnostics
    print(f"Diagnostics:")
    print(f"  Clip Fractions: main={loss_components.main_clip_frac:.3f}, "
          f"c={loss_components.c_clip_frac:.3f}, "
          f"btn={loss_components.buttons_clip_frac:.3f}, "
          f"shoulder={loss_components.shoulder_clip_frac:.3f}")
    print(f"  Approx KL:      {loss_components.approx_kl:.6f}")
    print(f"  Grad Norm:      {loss_components.grad_norm:.4f}")

    print(f"{'='*80}\n")


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
    args, remaining = parser.parse_known_args()

    # Initialize config with CLI overrides (--set key=value)
    overrides = parse_cli_overrides(remaining)
    init_config(overrides=overrides)
    config = get_config()
    ppo_config = PPOConfig()
    seq_len = config.seq_len
    if seq_len != 256:
        raise ValueError(
            f"Expected config.seq_len=256 for PPO, got {seq_len}."
        )

    # Load model from checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        default_dir = Path("checkpoints")
        latest = find_latest_checkpoint(default_dir)
        if latest is None:
            raise FileNotFoundError(
                f"No checkpoint provided and none found in {default_dir.resolve()}"
            )
        checkpoint_path = latest

    print(f"Loading model from: {checkpoint_path}")
    engine = GPTInferenceEngine(checkpoint_path=checkpoint_path)
    model = engine.model
    model.train()  # Switch to training mode

    device = next(model.parameters()).device
    print(f"Model loaded. Device: {device}")

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
    episode_count = 0

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
            column_map=engine.colmap,
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
            print(f"Episode {episode_count} too short ({rollout.episode_length} < {MIN_EPISODE_LENGTH} frames), skipping...")
            continue

        # ===== 2. COMPUTE ADVANTAGES =====
        print(f"[GAE] Computing advantages for {rollout.episode_length} frames...")
        gae_start = time.perf_counter()

        rewards = torch.tensor(
            [s.reward for s in rollout.steps], dtype=torch.float32
        )
        values = torch.tensor(
            [s.value_pred for s in rollout.steps], dtype=torch.float32
        )

        advantages, returns = compute_gae_advantages(
            rewards=rewards,
            values=values,
            gamma=config.reward.gamma,
            gae_lambda=ppo_config.gae_lambda,
            normalize=ppo_config.normalize_advantages,
        )

        rollout.advantages = advantages
        rollout.returns = returns
        gae_elapsed = time.perf_counter() - gae_start
        print(f"[GAE] Done in {gae_elapsed:.2f}s")

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
                column_map=engine.colmap,
                ppo_config=ppo_config,
                optimizer=optimizer,
                seq_len=seq_len,
                amp_ctx=amp_ctx,
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
                  f"time={epoch_elapsed:.2f}s")

        train_elapsed = time.perf_counter() - train_start
        print(f"[Training] Complete in {train_elapsed:.2f}s")

        # ===== 4. LOGGING =====
        if episode_count % ppo_config.log_interval == 0 and loss_components is not None:
            print_ppo_metrics(episode_count, rollout, loss_components)

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
