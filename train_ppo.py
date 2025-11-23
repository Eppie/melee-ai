#!/usr/bin/env python3
"""PPO self-play training script for Melee AI."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.amp import autocast
from torch.cuda.amp import GradScaler

from config import get_config, init_config
from libmelee.melee.enums import Menu
from model.nano_gpt import GPT
from ppo.opponent_pool import OpponentPool
from ppo.ppo_loss import compute_total_ppo_loss
from ppo.selfplay_env import SelfPlayEnvironment
from ppo.trajectory import Trajectory
from schema import get_target_names
from train.wandb_utils import WandbConfig, WandbLogger, init_wandb
from utils import _resolve_device


def load_initial_checkpoint(
    model: GPT,
    checkpoint_path: Optional[Path],
    device: torch.device,
) -> int:
    """Load initial checkpoint if provided.

    Returns:
        Episode number to start from
    """
    if checkpoint_path is None or not checkpoint_path.exists():
        print("Starting training from scratch")
        return 0

    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    episode = ckpt.get("episode", 0)
    print(f"Resumed from episode {episode}")

    return episode


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    episode: int,
    out_dir: Path,
) -> Path:
    """Save training checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"ppo_checkpoint_ep{episode:04d}.pt"

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": episode,
        "config": get_config().to_dict(),
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    return checkpoint_path


def run_episode_worker(
    worker_id: int,
    model_state_dict_path: Path,
    opponent_state_dict_path: Optional[Path],
    dolphin_path: Path,
    iso_path: Path,
    seq_len: int,
    max_episode_frames: int,
) -> Tuple[Optional[Trajectory], Dict[str, float]]:
    """Worker function to run a single episode in a separate process.

    Args:
        worker_id: Worker ID (for Dolphin port offset)
        model_state_dict_path: Path to saved learner model state dict
        opponent_state_dict_path: Path to saved opponent model state dict (or None)
        dolphin_path: Path to Dolphin executable
        iso_path: Path to Melee ISO
        seq_len: Sequence length for model
        max_episode_frames: Maximum frames per episode

    Returns:
        Tuple of (trajectory, episode_metrics)
    """
    # Re-initialize config in worker process
    init_config()
    from config import get_config

    config = get_config()

    # Reconstruct model
    device = torch.device("cpu")  # Workers use CPU
    model = GPT(config).to(device)
    model.load_state_dict(
        torch.load(model_state_dict_path, map_location=device, weights_only=True)
    )
    model.eval()

    # Load opponent model if provided
    opponent_model = None
    if opponent_state_dict_path is not None and opponent_state_dict_path.exists():
        opponent_model = GPT(config).to(device)
        opponent_model.load_state_dict(
            torch.load(opponent_state_dict_path, map_location=device, weights_only=True)
        )
        opponent_model.eval()

    # Create environment with port offset to avoid conflicts
    from ppo.opponent_pool import OpponentPool
    import shutil

    # Create a temporary opponent pool for this worker
    pool_dir = Path(tempfile.mkdtemp())
    opponent_pool = OpponentPool(max_size=1, pool_dir=pool_dir)
    if opponent_model is not None:
        opponent_pool.add_opponent(opponent_model, metadata={"worker": worker_id})

    env = SelfPlayEnvironment(
        learner_model=model,
        opponent_pool=opponent_pool,
        dolphin_path=dolphin_path,
        iso_path=iso_path,
        device=device,
        seq_len=seq_len,
        warmup_frames=128,
        max_episode_frames=max_episode_frames,
    )

    # Initialize console
    env.initialize_console()

    try:
        # Run episode
        print(f"[Worker {worker_id}] Starting episode...")
        episode_metrics = run_episode(env)

        # Get trajectory
        trajectories = env.trajectory_buffer.get_trajectories()
        trajectory = trajectories[0] if len(trajectories) > 0 else None

        print(
            f"[Worker {worker_id}] Episode complete: {episode_metrics.get('episode/frames', 0)} frames"
        )

        return trajectory, episode_metrics

    finally:
        # Cleanup
        env.console.stop()
        shutil.rmtree(pool_dir, ignore_errors=True)


def run_episode(
    env: SelfPlayEnvironment,
) -> Dict[str, float]:
    """Run one episode of self-play.

    Returns:
        Episode metrics
    """
    env.reset_episode()

    episode_metrics = {
        "episode/frames": 0,
        "episode/total_reward": 0.0,
    }

    # Wait for game to start
    print("Waiting for game to start...")
    while True:
        gamestate = env.console.step()
        if gamestate is None:
            continue

        if gamestate.menu_state in [Menu.IN_GAME, Menu.SUDDEN_DEATH]:
            print("Game started!")
            break
        else:
            # Navigate menus
            env.navigate_menu(gamestate)

    # Play game
    print("Playing episode...")
    done = False
    i = 0
    while not done:
        gamestate = env.console.step()
        if gamestate is None:
            continue

        # Check if we're still in game
        if gamestate.menu_state not in [Menu.IN_GAME, Menu.SUDDEN_DEATH]:
            print("Game ended (left game state)")
            done = True
            break

        # Step environment
        done, step_metrics = env.step(gamestate)

        # Update episode metrics
        for key, value in step_metrics.items():
            if key.startswith("episode/"):
                episode_metrics[key] = value

    print(
        f"Episode finished: {env.frame_count} frames, total reward: {env.episode_reward:.2f}"
    )

    return episode_metrics


def build_sequence_windows(
    trajectories: List[Trajectory],
    seq_len: int,
    stride: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build overlapping sequence windows from trajectories for proper transformer training.

    Instead of treating each step independently, this creates sliding windows of
    length seq_len that preserve temporal context. The model can then use attention
    over the full sequence history.

    Args:
        trajectories: List of trajectories with computed GAE
        seq_len: Length of each sequence window (should match model's seq_len)
        stride: Step size between windows (smaller = more overlap, more data)
        device: Device to place tensors on

    Returns:
        Dictionary with windowed tensors:
            - states: [num_windows, seq_len, F]
            - advantages: [num_windows, seq_len]
            - returns: [num_windows, seq_len]
            - old_log_probs: [num_windows, seq_len]
            - values: [num_windows, seq_len]
            - action_logits_*: [num_windows, seq_len, ...]
            - actions_*: [num_windows, seq_len, ...]
            - valid_mask: [num_windows, seq_len] - True for valid positions
    """
    all_windows = {
        "states": [],
        "advantages": [],
        "returns": [],
        "old_log_probs": [],
        "values": [],
        "valid_mask": [],
    }

    # Collect action head keys from first valid trajectory
    action_keys = []
    for traj in trajectories:
        if len(traj.steps) > 0:
            for head in ["main_stick", "c_stick", "buttons", "shoulder"]:
                if head in traj.steps[0].action_logits:
                    action_keys.append(head)
            break

    for head in action_keys:
        all_windows[f"action_logits_{head}"] = []
        all_windows[f"actions_{head}"] = []

    for traj in trajectories:
        if traj.advantages is None or traj.returns is None:
            continue

        T = len(traj.steps)
        if T == 0:
            continue

        # Convert trajectory to tensors (on CPU first for efficiency)
        states = torch.stack([step.state for step in traj.steps])  # [T, F]
        advantages = traj.advantages  # [T]
        returns = traj.returns  # [T]
        old_log_probs = torch.stack([step.log_prob for step in traj.steps]).squeeze(-1)  # [T]
        values = torch.stack([step.value for step in traj.steps]).squeeze(-1)  # [T]

        action_logits = {}
        actions = {}
        for head in action_keys:
            action_logits[head] = torch.stack([step.action_logits[head] for step in traj.steps])
            actions[head] = torch.stack([step.action_taken[head] for step in traj.steps])

        # Create sliding windows
        # For short trajectories, pad from the beginning
        if T < seq_len:
            # Pad trajectory to seq_len
            pad_len = seq_len - T
            states = torch.cat([states[0:1].expand(pad_len, -1), states], dim=0)
            advantages = torch.cat([torch.zeros(pad_len), advantages], dim=0)
            returns = torch.cat([returns[0:1].expand(pad_len), returns], dim=0)
            old_log_probs = torch.cat([old_log_probs[0:1].expand(pad_len), old_log_probs], dim=0)
            values = torch.cat([values[0:1].expand(pad_len), values], dim=0)

            for head in action_keys:
                action_logits[head] = torch.cat(
                    [action_logits[head][0:1].expand(pad_len, -1), action_logits[head]], dim=0
                )
                # Handle both 1D (indices) and 2D (buttons) action tensors
                if actions[head].dim() == 1:
                    actions[head] = torch.cat(
                        [actions[head][0:1].expand(pad_len), actions[head]], dim=0
                    )
                else:
                    # 2D tensor like buttons [T, num_buttons]
                    actions[head] = torch.cat(
                        [actions[head][0:1].expand(pad_len, -1), actions[head]], dim=0
                    )

            # Valid mask: only the original (non-padded) positions are valid for loss
            valid_mask = torch.cat([torch.zeros(pad_len, dtype=torch.bool), torch.ones(T, dtype=torch.bool)])

            # Single window for short trajectory
            all_windows["states"].append(states.unsqueeze(0))
            all_windows["advantages"].append(advantages.unsqueeze(0))
            all_windows["returns"].append(returns.unsqueeze(0))
            all_windows["old_log_probs"].append(old_log_probs.unsqueeze(0))
            all_windows["values"].append(values.unsqueeze(0))
            all_windows["valid_mask"].append(valid_mask.unsqueeze(0))

            for head in action_keys:
                all_windows[f"action_logits_{head}"].append(action_logits[head].unsqueeze(0))
                all_windows[f"actions_{head}"].append(actions[head].unsqueeze(0))
        else:
            # Create sliding windows with specified stride
            for start in range(0, T - seq_len + 1, stride):
                end = start + seq_len
                all_windows["states"].append(states[start:end].unsqueeze(0))
                all_windows["advantages"].append(advantages[start:end].unsqueeze(0))
                all_windows["returns"].append(returns[start:end].unsqueeze(0))
                all_windows["old_log_probs"].append(old_log_probs[start:end].unsqueeze(0))
                all_windows["values"].append(values[start:end].unsqueeze(0))
                # All positions valid in full-length windows
                all_windows["valid_mask"].append(torch.ones(seq_len, dtype=torch.bool).unsqueeze(0))

                for head in action_keys:
                    all_windows[f"action_logits_{head}"].append(action_logits[head][start:end].unsqueeze(0))
                    all_windows[f"actions_{head}"].append(actions[head][start:end].unsqueeze(0))

            # Include final window if trajectory doesn't divide evenly
            if (T - seq_len) % stride != 0:
                start = T - seq_len
                all_windows["states"].append(states[start:].unsqueeze(0))
                all_windows["advantages"].append(advantages[start:].unsqueeze(0))
                all_windows["returns"].append(returns[start:].unsqueeze(0))
                all_windows["old_log_probs"].append(old_log_probs[start:].unsqueeze(0))
                all_windows["values"].append(values[start:].unsqueeze(0))
                all_windows["valid_mask"].append(torch.ones(seq_len, dtype=torch.bool).unsqueeze(0))

                for head in action_keys:
                    all_windows[f"action_logits_{head}"].append(action_logits[head][start:].unsqueeze(0))
                    all_windows[f"actions_{head}"].append(actions[head][start:].unsqueeze(0))

    # Concatenate all windows and move to device
    result = {}
    for key, tensors in all_windows.items():
        if len(tensors) > 0:
            result[key] = torch.cat(tensors, dim=0).to(device)

    return result


def train_on_trajectories(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    trajectories: List[Trajectory],
    device: torch.device,
    logger: WandbLogger,
    episode: int,
    feature_names: List[str],
) -> Dict[str, float]:
    """Train model on collected trajectories using PPO with proper sequence handling.

    This function processes trajectories as overlapping windows of seq_len, allowing
    the transformer to use attention over the full sequence history during PPO updates.
    This prevents the "lobotomization" that occurs when training on individual steps.

    Args:
        model: The learner model to train
        optimizer: Optimizer
        scaler: Gradient scaler for AMP
        trajectories: List of collected trajectories
        device: Device to train on
        logger: Wandb logger
        episode: Current episode number
        feature_names: List of feature names for column mapping

    Returns:
        Training metrics
    """
    from column_map import ColumnMap
    from train.batch_utils import build_model_inputs

    config = get_config()
    ppo_cfg = config.ppo

    if len(trajectories) == 0:
        print("Warning: No trajectories to train on")
        return {}

    print(f"\nTraining on {len(trajectories)} trajectories...")

    # Compute GAE for all trajectories
    for traj in trajectories:
        traj.compute_gae(
            gamma=config.rl.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            normalize=ppo_cfg.normalize_advantages,
        )

        # Debug: Check for NaN/inf in trajectory
        if torch.isnan(traj.advantages).any() or torch.isinf(traj.advantages).any():
            print(f"WARNING: NaN or Inf detected in advantages!")
            print(
                f"  Advantages stats: min={traj.advantages.min():.4f}, max={traj.advantages.max():.4f}, mean={traj.advantages.mean():.4f}"
            )
        if torch.isnan(traj.returns).any() or torch.isinf(traj.returns).any():
            print(f"WARNING: NaN or Inf detected in returns!")
            print(
                f"  Returns stats: min={traj.returns.min():.4f}, max={traj.returns.max():.4f}, mean={traj.returns.mean():.4f}"
            )

    # Build sequence windows instead of flattening to individual steps
    # Use stride of seq_len // 4 for good overlap (4x data augmentation)
    window_stride = max(1, config.seq_len // 4)
    windows = build_sequence_windows(
        trajectories,
        seq_len=config.seq_len,
        stride=window_stride,
        device=device,
    )

    if "states" not in windows or windows["states"].shape[0] == 0:
        print("Warning: No valid windows created from trajectories")
        return {}

    num_windows = windows["states"].shape[0]
    seq_len = windows["states"].shape[1]
    total_steps = sum(len(traj) for traj in trajectories)

    print(f"Created {num_windows} sequence windows of length {seq_len} (stride={window_stride})")
    print(f"Total original steps: {total_steps}")

    # Create column map for building model inputs
    target_names = get_target_names()
    colmap = ColumnMap(feature_names, target_names)

    # Training metrics
    metrics = {
        "train/total_steps": total_steps,
        "train/num_trajectories": len(trajectories),
        "train/num_windows": num_windows,
    }

    # Number of positions to mask at the start of each sequence for warmup
    # The transformer needs some context before its predictions are reliable
    warmup_positions = config.seq_len // 4  # Mask first 25% of sequence

    # PPO epochs
    for ppo_epoch in range(ppo_cfg.ppo_epochs):
        # Shuffle windows (not individual steps!)
        perm = torch.randperm(num_windows, device=device)

        # Mini-batches of windows
        num_minibatches = max(1, num_windows // ppo_cfg.minibatch_size)
        epoch_losses = []

        for mb_idx in range(num_minibatches):
            start_idx = mb_idx * ppo_cfg.minibatch_size
            end_idx = min((mb_idx + 1) * ppo_cfg.minibatch_size, num_windows)
            mb_indices = perm[start_idx:end_idx]
            mb_size = len(mb_indices)

            # Get minibatch of windows [MB, seq_len, ...]
            mb_states = windows["states"][mb_indices]  # [MB, seq_len, F]
            mb_advantages = windows["advantages"][mb_indices]  # [MB, seq_len]
            mb_returns = windows["returns"][mb_indices]  # [MB, seq_len]
            mb_old_log_probs = windows["old_log_probs"][mb_indices]  # [MB, seq_len]
            mb_valid_mask = windows["valid_mask"][mb_indices]  # [MB, seq_len]
            mb_old_values = windows["values"][mb_indices]  # [MB, seq_len]

            # Get old action logits and actions taken
            mb_old_action_logits = {}
            mb_actions_taken = {}
            for key in windows.keys():
                if key.startswith("action_logits_"):
                    head = key.replace("action_logits_", "")
                    mb_old_action_logits[head] = windows[key][mb_indices]  # [MB, seq_len, ...]
                if key.startswith("actions_"):
                    head = key.replace("actions_", "")
                    mb_actions_taken[head] = windows[key][mb_indices]  # [MB, seq_len]

            # Create loss mask: valid positions after warmup
            loss_mask = mb_valid_mask.clone()
            loss_mask[:, :warmup_positions] = False  # Mask warmup positions

            # Skip if no valid positions for loss
            if not loss_mask.any():
                continue

            with autocast(device_type=device.type, enabled=False):
                # Build model inputs with full sequence
                model_inputs = build_model_inputs(mb_states, colmap)

                # Forward pass with full sequence context
                outputs = model(model_inputs)

                # Extract logits and values for all positions [MB, seq_len, ...]
                new_action_logits = {}
                for head in ["main_stick", "c_stick", "buttons", "shoulder"]:
                    if head in outputs:
                        new_action_logits[head] = outputs[head]  # [MB, seq_len, num_classes]

                new_values = outputs.get(
                    "value", torch.zeros(mb_size, seq_len, 1, device=device)
                )[:, :, 0]  # [MB, seq_len]

                # Check for NaN in model outputs
                has_nan_output = False
                for head, logits in new_action_logits.items():
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print(
                            f"\nWARNING: NaN/Inf detected in {head} logits from model forward pass!"
                        )
                        print(
                            f"  {head}: min={logits.min():.4f}, max={logits.max():.4f}, nan_count={torch.isnan(logits).sum()}"
                        )
                        has_nan_output = True
                if torch.isnan(new_values).any() or torch.isinf(new_values).any():
                    print(f"\nWARNING: NaN/Inf detected in value head output!")
                    print(
                        f"  values: min={new_values.min():.4f}, max={new_values.max():.4f}, nan_count={torch.isnan(new_values).sum()}"
                    )
                    has_nan_output = True

                if has_nan_output:
                    nan_params = sum(
                        1 for p in model.parameters() if torch.isnan(p).any()
                    )
                    print(f"  Model has {nan_params} parameters with NaN")
                    continue

                # Compute PPO loss with masking for valid positions
                loss, loss_metrics = compute_total_ppo_loss(
                    new_action_logits=new_action_logits,
                    new_values=new_values,
                    old_action_logits=mb_old_action_logits,
                    old_values=mb_old_values,
                    actions_taken=mb_actions_taken,
                    old_log_probs=mb_old_log_probs,
                    advantages=mb_advantages,
                    returns=mb_returns,
                    clip_ratio=ppo_cfg.clip_ratio,
                    entropy_coef=ppo_cfg.entropy_coef,
                    value_coef=config.rl.value_loss_coef,
                    value_clip=ppo_cfg.value_clip,
                    loss_mask=loss_mask,  # Pass mask to loss function
                )

            # Check for NaN loss BEFORE backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"\nWARNING: NaN/Inf loss detected in minibatch {mb_idx}, epoch {ppo_epoch}"
                )
                print(
                    f"  Advantages: min={mb_advantages.min():.4f}, max={mb_advantages.max():.4f}, mean={mb_advantages.mean():.4f}, std={mb_advantages.std():.4f}"
                )
                print(
                    f"  Returns: min={mb_returns.min():.4f}, max={mb_returns.max():.4f}"
                )
                print(
                    f"  Old log probs: min={mb_old_log_probs.min():.4f}, max={mb_old_log_probs.max():.4f}"
                )
                print(
                    f"  New values: min={new_values.min():.4f}, max={new_values.max():.4f}"
                )
                continue

            optimizer.zero_grad()

            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Check for NaN gradients
            nan_grads = 0
            inf_grads = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        nan_grads += 1
                    if torch.isinf(param.grad).any():
                        inf_grads += 1

            if nan_grads > 0 or inf_grads > 0:
                print(
                    f"\nWARNING: NaN/Inf gradients detected! nan_grads={nan_grads}, inf_grads={inf_grads}"
                )
                print(f"  Skipping optimizer step to prevent model corruption")
                optimizer.zero_grad()
                continue

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), ppo_cfg.max_grad_norm
            )

            # Log gradient norm
            if mb_idx == 0 and ppo_epoch == 0:
                metrics["train/grad_norm"] = grad_norm.item()

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

            # Log metrics from this minibatch
            if mb_idx == 0 and ppo_epoch == 0:
                for key, value in loss_metrics.items():
                    metrics[key] = value

        if len(epoch_losses) > 0:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(
                f"  Epoch {ppo_epoch + 1}/{ppo_cfg.ppo_epochs}: avg loss = {avg_loss:.4f}"
            )
            metrics[f"train/ppo_epoch_{ppo_epoch}_loss"] = avg_loss
        else:
            print(
                f"  Epoch {ppo_epoch + 1}/{ppo_cfg.ppo_epochs}: no valid batches (all NaN)"
            )

    # Log metrics
    logger.log_metrics(metrics, step=episode)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="PPO self-play training for Melee AI")
    parser.add_argument(
        "--checkpoint", type=Path, default=None, help="Initial checkpoint to load"
    )
    parser.add_argument(
        "--dolphin-path", type=str, required=True, help="Path to Dolphin executable"
    )
    parser.add_argument("--iso", type=str, required=True, help="Path to Melee ISO")
    parser.add_argument(
        "--num-episodes", type=int, default=1000, help="Number of episodes to train"
    )
    parser.add_argument(
        "--save-every", type=int, default=10, help="Save checkpoint every N episodes"
    )
    parser.add_argument(
        "--add-to-pool-every",
        type=int,
        default=5,
        help="Add model to opponent pool every N episodes",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("checkpoints/ppo"),
        help="Output directory for checkpoints",
    )

    args = parser.parse_args()

    # Initialize config
    init_config()
    config = get_config()

    # Setup device
    device = _resolve_device(None)
    print(f"Using device: {device}")

    # Create model
    model = GPT(config).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Load initial checkpoint if provided
    start_episode = load_initial_checkpoint(model, args.checkpoint, device)
    start_episode += 1

    # Optimizer (use PPO-specific learning rate)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.ppo.lr,
        betas=config.train.betas,
        weight_decay=config.train.weight_decay,
    )
    print(f"Using PPO learning rate: {config.ppo.lr:.2e}")

    # Gradient scaler for AMP
    scaler = GradScaler(enabled=config.train.use_amp)

    # Opponent pool
    pool_dir = args.out_dir / "opponent_pool"
    opponent_pool = OpponentPool(
        max_size=config.ppo.pool_size,
        pool_dir=pool_dir,
    )

    # Initialize with current model if pool is empty
    if opponent_pool.is_empty():
        print("Opponent pool is empty. Adding initial model...")
        opponent_pool.add_opponent(
            model,
            metadata={"episode": start_episode, "note": "initial_model"},
        )

    # Initialize wandb
    wandb_cfg = WandbConfig(
        project="melee-ai-ppo",
        name=f"ppo_selfplay_ep{start_episode}",
        mode="online",
    )
    wandb_run = init_wandb(
        config=wandb_cfg,
        run_dir=args.out_dir,
        hyperparameters={
            "train": dict(vars(config.train)),
            "model": dict(vars(config.model)),
            "ppo": dict(vars(config.ppo)),
            "rl": dict(vars(config.rl)),
        },
    )
    logger = WandbLogger(wandb_run, enabled=wandb_run is not None)

    # Determine if we're using parallel or sequential collection
    num_workers = config.ppo.num_workers
    use_parallel = num_workers > 1

    print(
        f"Using {'parallel' if use_parallel else 'sequential'} trajectory collection with {num_workers} worker(s)"
    )

    # For sequential mode, create a single environment
    env = None
    if not use_parallel:
        env = SelfPlayEnvironment(
            learner_model=model,
            opponent_pool=opponent_pool,
            dolphin_path=args.dolphin_path,
            iso_path=args.iso,
            device=device,
            seq_len=config.seq_len,
            warmup_frames=128,
            max_episode_frames=config.ppo.max_episode_frames,
        )
        env.initialize_console()

    try:
        # Training loop
        for episode_idx in range(start_episode, start_episode + args.num_episodes):
            episode_num = episode_idx + 1  # 1-indexed for display
            print(f"\n{'='*80}")
            print(f"Episode {episode_num}/{start_episode + args.num_episodes}")
            print(f"{'='*80}")

            # Set model to eval mode for data collection
            model.eval()

            # Collect trajectories (parallel or sequential)
            trajectories = []
            all_episode_metrics = []

            if use_parallel:
                # Parallel collection using multiprocessing
                print(f"Collecting {num_workers} trajectories in parallel...")

                # Save model weights to temp files for workers
                model_temp = Path(tempfile.mkdtemp()) / "learner_model.pt"
                torch.save(model.state_dict(), model_temp)

                # Sample opponent and save if it exists
                opponent_temp = None
                if not opponent_pool.is_empty():
                    opponent_checkpoint_path, _ = opponent_pool.sample_opponent()
                    # Load opponent model
                    opponent_model = GPT(config).to(device)
                    opponent_pool.load_opponent_model(
                        opponent_model, opponent_checkpoint_path
                    )
                    opponent_temp = Path(model_temp).parent / "opponent_model.pt"
                    torch.save(opponent_model.state_dict(), opponent_temp)

                # Create worker arguments
                worker_args = [
                    (
                        worker_id,
                        model_temp,
                        opponent_temp,
                        args.dolphin_path,
                        args.iso,
                        config.seq_len,
                        config.ppo.max_episode_frames,
                    )
                    for worker_id in range(num_workers)
                ]

                # Run workers in parallel
                with mp.Pool(processes=num_workers) as pool:
                    results = pool.starmap(run_episode_worker, worker_args)

                # Collect results
                for trajectory, metrics in results:
                    if trajectory is not None:
                        trajectories.append(trajectory)
                    all_episode_metrics.append(metrics)

                # Cleanup temp files
                import shutil

                shutil.rmtree(model_temp.parent, ignore_errors=True)

                # Aggregate metrics (average across workers)
                episode_metrics = {}
                if all_episode_metrics:
                    for key in all_episode_metrics[0].keys():
                        values = [m.get(key, 0.0) for m in all_episode_metrics]
                        episode_metrics[key] = sum(values) / len(values)

            else:
                # Sequential collection (original behavior)
                episode_metrics = run_episode(env)
                trajectories = env.trajectory_buffer.get_trajectories()

            # Log episode metrics
            logger.log_metrics(episode_metrics, step=episode_num)

            print(f"Collected {len(trajectories)} trajectories")

            # Train on trajectories
            if len(trajectories) > 0:
                model.train()
                # Get feature names from env or schema
                from schema import get_feature_names

                feature_names = (
                    env.feature_names if env is not None else get_feature_names()
                )

                train_metrics = train_on_trajectories(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    trajectories=trajectories,
                    device=device,
                    logger=logger,
                    episode=episode_num,
                    feature_names=feature_names,
                )

            # Clear trajectory buffer (sequential mode only)
            if not use_parallel and env is not None:
                env.trajectory_buffer.clear()

            # Add to opponent pool
            if episode_num % args.add_to_pool_every == 0:
                opponent_pool.add_opponent(
                    model,
                    metadata={
                        "episode": episode_num,
                        "total_reward": episode_metrics.get(
                            "episode/total_reward", 0.0
                        ),
                    },
                )

                # Log pool stats
                pool_stats = opponent_pool.get_pool_stats()
                logger.log_metrics(
                    {
                        "pool/size": pool_stats["pool_size"],
                        "pool/total_created": pool_stats["total_opponents_created"],
                    },
                    step=episode_num,
                )

            # Save checkpoint
            if episode_num % args.save_every == 0:
                save_checkpoint(model, optimizer, episode_num, args.out_dir)

        print("\nTraining complete!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    finally:
        # Cleanup
        if env is not None:
            print("Shutting down environment...")
            env.shutdown_console()

        # Save final checkpoint
        save_checkpoint(model, optimizer, episode_num, args.out_dir)

        print("Done!")


if __name__ == "__main__":
    main()
