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
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Bernoulli, Categorical

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.append(str(REPO_ROOT))

from column_map import ColumnMap
from config.config import init_config, get_config
from config.reward_config import RewardConfig
from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)
from libmelee.melee.console import Console
from libmelee.melee.controller import Controller
from libmelee.melee.enums import (
    Character,
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
from train.value_head import build_reward_feature_index, compute_frame_rewards


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
    value_coef: float = 0.5  # Coefficient for value loss
    entropy_coef: float = 0.01  # Coefficient for entropy bonus

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


LEGAL_TOURNAMENT_STAGES = [
    Stage.BATTLEFIELD,
    Stage.YOSHIS_STORY,
    Stage.POKEMON_STADIUM,
    Stage.DREAMLAND,
    Stage.FINAL_DESTINATION,
    Stage.FOUNTAIN_OF_DREAMS,
]

SUPPORTED_CHARS = [Character.FOX]

BOT_PORT = 1
OPP_PORT = 2


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
    main_log_prob: float
    c_log_prob: float
    buttons_log_probs: torch.Tensor  # [5]
    shoulder_log_prob: float


@dataclass
class RolloutStep:
    """Single timestep in a rollout."""

    # Full feature sequence needed for model input [seq_len, F]
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

def sample_actions_with_logprobs(
    outputs: TensorDict,
    device: torch.device,
) -> ActionInfo:
    """Sample actions from model outputs and compute log probabilities.

    Args:
        outputs: Model outputs with keys ["main_stick", "c_stick", "buttons", "shoulder"]
        device: Device for tensor operations

    Returns:
        ActionInfo with sampled actions and log probabilities
    """
    # Extract last timestep logits [1, seq_len, K] -> [K]
    main_logits = outputs["main_stick"][0, -1].to(device)  # [64]
    c_logits = outputs["c_stick"][0, -1].to(device)  # [9]
    button_logits = outputs["buttons"][0, -1].to(device)  # [5]
    shoulder_logits = outputs["shoulder"][0, -1].to(device)  # [5]

    # Main stick: Categorical distribution
    main_dist = Categorical(logits=main_logits)
    main_idx = main_dist.sample()
    main_log_prob = main_dist.log_prob(main_idx)

    # C-stick: Categorical distribution
    c_dist = Categorical(logits=c_logits)
    c_idx = c_dist.sample()
    c_log_prob = c_dist.log_prob(c_idx)

    # Shoulder: Categorical distribution
    shoulder_dist = Categorical(logits=shoulder_logits)
    shoulder_idx = shoulder_dist.sample()
    shoulder_log_prob = shoulder_dist.log_prob(shoulder_idx)

    # Buttons: Independent Bernoulli for each button
    button_probs = torch.sigmoid(button_logits)
    button_samples = torch.bernoulli(button_probs)

    # Compute log probabilities for buttons
    # log(p) if sampled 1, log(1-p) if sampled 0
    button_log_probs = torch.where(
        button_samples == 1,
        torch.log(button_probs + 1e-8),
        torch.log(1 - button_probs + 1e-8),
    )

    return ActionInfo(
        main_stick_idx=main_idx.item(),
        c_stick_idx=c_idx.item(),
        buttons=button_samples.cpu(),
        shoulder_idx=shoulder_idx.item(),
        main_log_prob=main_log_prob.item(),
        c_log_prob=c_log_prob.item(),
        buttons_log_probs=button_log_probs.cpu(),
        shoulder_log_prob=shoulder_log_prob.item(),
    )


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
    main_x, main_y = model_to_dolphin01(main_coords[0], main_coords[1])
    c_x, c_y = model_to_dolphin01(c_coords[0], c_coords[1])

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
            # Navigate menus
            menu_helper.menu_helper_simple(
                gamestate,
                controllers[BOT_PORT],
                bot_char,
                current_stage,
                costume=1,
                autostart=True,
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

        # Prepare model inputs (updates buffer)
        engine.prepare_inputs(raw_inputs.transformed)
        frame_count += 1

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
        if len(engine.buffer) < engine.warmup_frames:
            # Warmup: use neutral controller
            controller_state = ControllerState.neutral()
            apply_model_outputs_to_game(controllers[BOT_PORT], controller_state)

            if done:
                # Episode ended during warmup (very rare)
                break
            continue

        # Stack buffered frames
        features_batch = torch.stack(list(engine.buffer), dim=0)  # [seq_len, F]
        inputs_td = build_model_inputs(
            features_batch.unsqueeze(0), column_map
        )  # [1, seq_len, F]

        # Forward pass
        with torch.inference_mode():
            outputs = model(inputs_td)

        # Sample actions and get log probs
        action_info = sample_actions_with_logprobs(outputs, model.device)

        # Apply to game
        controller_state = action_info_to_controller_state(action_info, engine)
        apply_model_outputs_to_game(controllers[BOT_PORT], controller_state)

        # Get value prediction
        value_pred = outputs["value"][0, -1, 0].item()

        # Compute reward (needs at least 2 frames)
        if len(steps) > 0:
            # Build feature tensor for reward computation
            # Need [1, 2, F] with previous and current frame
            X_for_reward = features_batch[-2:].unsqueeze(0)  # [1, 2, F]
            rewards_tensor = compute_frame_rewards(
                X_for_reward, reward_idx, reward_config
            )  # [1, 2]
            frame_reward = rewards_tensor[0, -1].item()
        else:
            frame_reward = 0.0

        # Store step
        steps.append(RolloutStep(
            features=features_batch.clone(),  # Full sequence
            action_info=action_info,
            value_pred=value_pred,
            reward=frame_reward,
            done=done,
        ))

        if done:
            break

    # Compute total reward
    total_reward = sum(s.reward for s in steps)

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


def compute_ppo_loss(
    rollout: Rollout,
    model: GPT,
    column_map: ColumnMap,
    ppo_config: PPOConfig,
) -> PPOLossComponents:
    """Compute PPO loss for a rollout.

    Args:
        rollout: Complete rollout with advantages and returns
        model: GPT model
        column_map: Feature column mapping
        ppo_config: PPO configuration

    Returns:
        PPO loss components
    """
    device = model.device
    L = len(rollout.steps)

    # Build batch of inputs [L, seq_len, F]
    features_batch = torch.stack(
        [s.features for s in rollout.steps], dim=0
    ).to(device)
    inputs_td = build_model_inputs(features_batch, column_map)

    # Forward pass
    outputs = model(inputs_td)

    # Compute new log probabilities
    new_log_probs = compute_log_probs_for_actions(outputs, rollout, device)

    # Extract old log probabilities from rollout
    old_main_log_probs = torch.tensor(
        [s.action_info.main_log_prob for s in rollout.steps], device=device
    )
    old_c_log_probs = torch.tensor(
        [s.action_info.c_log_prob for s in rollout.steps], device=device
    )
    old_shoulder_log_probs = torch.tensor(
        [s.action_info.shoulder_log_prob for s in rollout.steps], device=device
    )
    old_buttons_log_probs = torch.stack(
        [s.action_info.buttons_log_probs for s in rollout.steps], dim=0
    ).to(device)

    old_log_probs = {
        "main": old_main_log_probs,
        "c": old_c_log_probs,
        "buttons": old_buttons_log_probs,
        "shoulder": old_shoulder_log_probs,
    }

    # Get advantages
    advantages = rollout.advantages.to(device)  # [L]

    # ===== POLICY LOSS (Clipped Surrogate Objective) =====
    policy_losses = {}
    clip_fractions = {}

    for key in ["main", "c", "shoulder"]:
        # Compute importance ratio
        ratio = torch.exp(new_log_probs[key] - old_log_probs[key])  # [L]

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(
            ratio,
            1.0 - ppo_config.clip_epsilon,
            1.0 + ppo_config.clip_epsilon,
        ) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()
        policy_losses[key] = policy_loss

        # Clip fraction (diagnostic)
        clip_fractions[key] = (
            (torch.abs(ratio - 1.0) > ppo_config.clip_epsilon).float().mean().item()
        )

    # Buttons: handle per-button (averaged)
    button_ratio = torch.exp(new_log_probs["buttons"] - old_log_probs["buttons"])  # [L, 5]
    button_surr1 = button_ratio * advantages.unsqueeze(-1)
    button_surr2 = torch.clamp(
        button_ratio,
        1.0 - ppo_config.clip_epsilon,
        1.0 + ppo_config.clip_epsilon,
    ) * advantages.unsqueeze(-1)
    button_policy_loss = -torch.min(button_surr1, button_surr2).mean()
    policy_losses["buttons"] = button_policy_loss
    clip_fractions["buttons"] = (
        (torch.abs(button_ratio - 1.0) > ppo_config.clip_epsilon).float().mean().item()
    )

    total_policy_loss = sum(policy_losses.values())

    # ===== VALUE LOSS (MSE) =====
    value_preds = outputs["value"][:, -1, 0]  # [L]
    value_targets = rollout.returns.to(device)  # [L]
    value_loss = ((value_preds - value_targets) ** 2).mean()

    # ===== ENTROPY BONUS =====
    entropies = compute_policy_entropy(outputs)
    total_entropy = sum(entropies.values())

    # ===== TOTAL LOSS =====
    total_loss = (
        total_policy_loss
        + ppo_config.value_coef * value_loss
        - ppo_config.entropy_coef * total_entropy
    )

    # ===== DIAGNOSTICS =====
    # Approximate KL divergence
    approx_kl = 0.5 * sum(
        ((new_log_probs[k] - old_log_probs[k]) ** 2).mean().item()
        for k in ["main", "c", "shoulder"]
    ) + 0.5 * ((new_log_probs["buttons"] - old_log_probs["buttons"]) ** 2).mean().item()

    return PPOLossComponents(
        total_loss=total_loss,
        policy_loss=total_policy_loss,
        value_loss=value_loss,
        entropy=torch.tensor(total_entropy),
        main_policy_loss=policy_losses["main"],
        c_policy_loss=policy_losses["c"],
        buttons_policy_loss=policy_losses["buttons"],
        shoulder_policy_loss=policy_losses["shoulder"],
        main_entropy=entropies["main"],
        c_entropy=entropies["c"],
        buttons_entropy=entropies["buttons"],
        shoulder_entropy=entropies["shoulder"],
        main_clip_frac=clip_fractions["main"],
        c_clip_frac=clip_fractions["c"],
        buttons_clip_frac=clip_fractions["buttons"],
        shoulder_clip_frac=clip_fractions["shoulder"],
        approx_kl=approx_kl,
        grad_norm=0.0,  # Will be filled in after backward pass
    )


# ============================================================================
# Metrics and Logging
# ============================================================================

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
    args = parser.parse_args()

    # Initialize config
    init_config()
    config = get_config()
    ppo_config = PPOConfig()

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

    print(f"Model loaded. Device: {model.device}")

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
        )

        episode_count += 1

        # Check if episode is long enough
        if rollout.episode_length < 2:
            print(f"Episode {episode_count} too short ({rollout.episode_length} frames), skipping...")
            continue

        # ===== 2. COMPUTE ADVANTAGES =====
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

        # ===== 3. PPO TRAINING (K epochs) =====
        loss_components = None
        for epoch in range(ppo_config.num_epochs):
            # Compute loss
            loss_components = compute_ppo_loss(
                rollout=rollout,
                model=model,
                column_map=engine.colmap,
                ppo_config=ppo_config,
            )

            # Backward pass
            optimizer.zero_grad()
            loss_components.total_loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=ppo_config.grad_clip,
            )
            loss_components.grad_norm = grad_norm.item()

            optimizer.step()

        # ===== 4. LOGGING =====
        if episode_count % ppo_config.log_interval == 0 and loss_components is not None:
            print_ppo_metrics(episode_count, rollout, loss_components)

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
