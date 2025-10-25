#!/usr/bin/env python3
"""PPO self-play training script for Melee AI."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import multiprocessing as mp
import tempfile

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
from train.wandb_utils import WandbConfig, WandbLogger, init_wandb, WANDB_AVAILABLE
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
    model.load_state_dict(torch.load(model_state_dict_path, map_location=device, weights_only=True))
    model.eval()
    
    # Load opponent model if provided
    opponent_model = None
    if opponent_state_dict_path is not None and opponent_state_dict_path.exists():
        opponent_model = GPT(config).to(device)
        opponent_model.load_state_dict(torch.load(opponent_state_dict_path, map_location=device, weights_only=True))
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
        slippi_port=51441 + worker_id,  # Offset port for each worker
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
        
        print(f"[Worker {worker_id}] Episode complete: {episode_metrics.get('episode/frames', 0)} frames")
        
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
    """Train model on collected trajectories using PPO.

    Args:
        model: The learner model to train
        optimizer: Optimizer
        scaler: Gradient scaler for AMP
        trajectories: List of collected trajectories
        device: Device to train on
        logger: Wandb logger
        episode: Current episode number

    Returns:
        Training metrics
    """
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
            print(f"  Advantages stats: min={traj.advantages.min():.4f}, max={traj.advantages.max():.4f}, mean={traj.advantages.mean():.4f}")
        if torch.isnan(traj.returns).any() or torch.isinf(traj.returns).any():
            print(f"WARNING: NaN or Inf detected in returns!")
            print(f"  Returns stats: min={traj.returns.min():.4f}, max={traj.returns.max():.4f}, mean={traj.returns.mean():.4f}")

    # Convert trajectories to tensors and concatenate
    all_data = []
    for traj in trajectories:
        try:
            data = traj.to_tensors(device)
            all_data.append(data)
        except ValueError as e:
            print(f"Warning: Skipping trajectory: {e}")
            continue

    if len(all_data) == 0:
        print("Warning: No valid trajectory data to train on")
        return {}

    # Concatenate all trajectories
    batch = {
        key: torch.cat([d[key] for d in all_data], dim=0) for key in all_data[0].keys()
    }

    total_steps = batch["states"].shape[0]
    print(f"Total training steps: {total_steps}")

    # Store old action logits and values (used for PPO loss)
    with torch.no_grad():
        # We already have these from trajectory collection, but for value clipping
        # we might need fresh values. For simplicity, use stored values.
        old_values = batch["values"]

    # Training metrics
    metrics = {
        "train/total_steps": total_steps,
        "train/num_trajectories": len(trajectories),
    }

    # PPO epochs
    for ppo_epoch in range(ppo_cfg.ppo_epochs):
        # Shuffle data
        perm = torch.randperm(total_steps)

        # Mini-batches
        num_minibatches = max(1, total_steps // ppo_cfg.minibatch_size)
        epoch_losses = []

        for mb_idx in range(num_minibatches):
            start_idx = mb_idx * ppo_cfg.minibatch_size
            end_idx = min((mb_idx + 1) * ppo_cfg.minibatch_size, total_steps)
            mb_indices = perm[start_idx:end_idx]

            # Get minibatch
            mb_states = batch["states"][mb_indices]  # [MB, F]
            mb_advantages = batch["advantages"][mb_indices]
            mb_returns = batch["returns"][mb_indices]
            mb_old_log_probs = batch["old_log_probs"][mb_indices]

            # Get old action logits
            mb_old_action_logits = {}
            mb_actions_taken = {}
            for key in batch.keys():
                if key.startswith("action_logits_"):
                    head = key.replace("action_logits_", "")
                    mb_old_action_logits[head] = batch[key][mb_indices]
                if key.startswith("actions_"):
                    head = key.replace("actions_", "")
                    mb_actions_taken[head] = batch[key][mb_indices]

            mb_old_values = old_values[mb_indices]

            # Build model inputs from states
            # States are raw features [MB, F], need to add sequence dimension
            mb_states_seq = mb_states.unsqueeze(1)  # [MB, 1, F]

            # For simplicity, we'll do a forward pass per sample (not ideal for efficiency)
            # In production, you'd want to batch this properly with the full sequence context
            # For now, we'll accumulate loss over the minibatch

            # This is a simplification - ideally you'd maintain sequence context
            # But for PPO on single steps, we can treat each as independent

            with autocast(device_type=device.type, enabled=False):
                # Build inputs (this needs proper handling of sequences)
                # For simplicity, we'll forward each sample independently

                # We need to call the model with proper sequence input
                # Let's create a batch with each state as a single-step sequence
                from column_map import ColumnMap
                from train.batch_utils import build_model_inputs

                # Create column map with proper target names
                target_names = get_target_names()
                colmap = ColumnMap(feature_names, target_names)

                # Build model inputs
                model_inputs = build_model_inputs(mb_states_seq, colmap)

                # Forward pass
                outputs = model(model_inputs)

                # Extract new logits and values
                new_action_logits = {}
                for head in ["main_stick", "c_stick", "buttons", "shoulder"]:
                    if head in outputs:
                        new_action_logits[head] = outputs[head][
                            :, -1, :
                        ]  # Take last timestep

                new_values = outputs.get(
                    "value", torch.zeros(mb_states.shape[0], 1, 1)
                )[:, -1, 0]
                
                # Check for NaN in model outputs
                has_nan_output = False
                for head, logits in new_action_logits.items():
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print(f"\nWARNING: NaN/Inf detected in {head} logits from model forward pass!")
                        print(f"  {head}: min={logits.min():.4f}, max={logits.max():.4f}, nan_count={torch.isnan(logits).sum()}")
                        has_nan_output = True
                if torch.isnan(new_values).any() or torch.isinf(new_values).any():
                    print(f"\nWARNING: NaN/Inf detected in value head output!")
                    print(f"  values: min={new_values.min():.4f}, max={new_values.max():.4f}, nan_count={torch.isnan(new_values).sum()}")
                    has_nan_output = True
                
                if has_nan_output:
                    # Check model parameters
                    nan_params = sum(1 for p in model.parameters() if torch.isnan(p).any())
                    print(f"  Model has {nan_params} parameters with NaN")
                    # Skip this batch
                    continue

                # Compute PPO loss
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
                )
            
            # Check for NaN loss BEFORE backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\nWARNING: NaN/Inf loss detected in minibatch {mb_idx}, epoch {ppo_epoch}")
                print(f"  Advantages: min={mb_advantages.min():.4f}, max={mb_advantages.max():.4f}, mean={mb_advantages.mean():.4f}, std={mb_advantages.std():.4f}")
                print(f"  Returns: min={mb_returns.min():.4f}, max={mb_returns.max():.4f}")
                print(f"  Old log probs: min={mb_old_log_probs.min():.4f}, max={mb_old_log_probs.max():.4f}")
                print(f"  New values: min={new_values.min():.4f}, max={new_values.max():.4f}")
                # Skip this batch entirely (don't touch optimizer/scaler)
                continue

            # Forward pass with new policy
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
                print(f"\nWARNING: NaN/Inf gradients detected! nan_grads={nan_grads}, inf_grads={inf_grads}")
                print(f"  Skipping optimizer step to prevent model corruption")
                optimizer.zero_grad()
                continue

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.max_grad_norm)
            
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
            print(f"  Epoch {ppo_epoch + 1}/{ppo_cfg.ppo_epochs}: no valid batches (all NaN)")

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
    logger = WandbLogger(wandb_run, enabled=WANDB_AVAILABLE and wandb_run is not None)

    # Determine if we're using parallel or sequential collection
    num_workers = config.ppo.num_workers
    use_parallel = num_workers > 1
    
    print(f"Using {'parallel' if use_parallel else 'sequential'} trajectory collection with {num_workers} worker(s)")
    
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
                    opponent_model = opponent_pool.sample()
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
                feature_names = env.feature_names if env is not None else get_feature_names()
                
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
