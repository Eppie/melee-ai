#!/usr/bin/env python3
"""PPO self-play training script for Melee AI.

Supports two modes:
1. Sequential/parallel episode-based training (original)
2. Distributed mode with centralized GPU inference (new)

Use --distributed flag to enable the new architecture.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.amp import autocast
from torch.amp import GradScaler

from config import get_config, init_config
from libmelee.melee.enums import Menu
from model.nano_gpt import GPT
from ppo.opponent_pool import OpponentPool
from ppo.ppo_loss import compute_total_ppo_loss
from ppo.selfplay_env import SelfPlayEnvironment
from ppo.trajectory import Trajectory
from schema import get_feature_names, get_target_names
from train.wandb_utils import WandbConfig, WandbLogger, init_wandb
from utils import _resolve_device, strip_compiled_prefix


# =============================================================================
# Checkpoint Management
# =============================================================================


def load_initial_checkpoint(
    model: GPT,
    checkpoint_path: Optional[Path],
    device: torch.device,
) -> int:
    """Load initial checkpoint if provided.

    Returns:
        Episode/rollout number to start from
    """
    if checkpoint_path is None or not checkpoint_path.exists():
        print("Starting training from scratch")
        return 0

    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = strip_compiled_prefix(ckpt["model"])
    model.load_state_dict(model_state)

    episode = ckpt.get("episode", ckpt.get("rollout", 0))
    print(f"Resumed from step {episode}")

    return episode


def save_checkpoint(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    out_dir: Path,
    prefix: str = "ppo_checkpoint",
) -> Path:
    """Save training checkpoint."""
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{prefix}_{step:06d}.pt"

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": step,
        "rollout": step,
        "config": get_config().to_dict(),
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")

    return checkpoint_path


# =============================================================================
# Distributed Training Mode
# =============================================================================


def run_distributed_training(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    opponent_pool: OpponentPool,
    device: torch.device,
    logger: WandbLogger,
    args,
) -> None:
    """Run training with centralized GPU inference architecture.

    This mode uses:
    - InferenceCoordinator: Batched GPU inference for all workers
    - SimulationWorker: Stateless emulator processes
    - TrajectorySlicer: Fixed-length rollouts with bootstrapping
    """
    from ppo.inference_coordinator import InferenceCoordinator
    from ppo.simulation_worker import SimulationWorker, WorkerConfig, run_worker
    from ppo.trajectory_slicer import TrajectorySlicer

    config = get_config()
    ppo_cfg = config.ppo
    num_workers = ppo_cfg.num_workers

    print(f"\n{'='*60}")
    print("DISTRIBUTED PPO TRAINING")
    print(f"{'='*60}")
    print(f"Workers: {num_workers}")
    print(f"Rollout length: {ppo_cfg.rollout_length} frames")
    print(f"Opponent rotation: every {ppo_cfg.opponent_rotation_interval} frames")
    print(f"{'='*60}\n")

    # Compile model for faster inference (2-3x speedup)
    print("Compiling model for inference...")
    model = torch.compile(model, mode='reduce-overhead')

    # Create inference coordinator
    coordinator = InferenceCoordinator(
        learner_model=model,
        device=device,
        num_workers=num_workers,
        seq_len=config.seq_len,
        warmup_frames=ppo_cfg.warmup_frames,
    )

    # Load initial opponent
    if not opponent_pool.is_empty():
        opponent_path, _ = opponent_pool.sample_opponent()
        coordinator.load_opponent(opponent_path)

    # Create trajectory slicer
    slicer = TrajectorySlicer(
        coordinator=coordinator,
        rollout_length=ppo_cfg.rollout_length,
        gamma=config.rl.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        normalize_advantages=ppo_cfg.normalize_advantages,
    )

    # Create queues for IPC
    state_queue = mp.Queue()  # Workers -> Coordinator
    action_queues = {
        i: mp.Queue() for i in range(num_workers)
    }  # Coordinator -> Workers
    control_queues = {i: mp.Queue() for i in range(num_workers)}  # Control signals

    # Spawn worker processes
    workers = []
    for worker_id in range(num_workers):
        worker_config = WorkerConfig(
            worker_id=worker_id,
            dolphin_path=args.dolphin_path,
            iso_path=args.iso,
        )
        p = mp.Process(
            target=run_worker,
            args=(
                worker_id,
                worker_config,
                state_queue,
                action_queues[worker_id],
                control_queues[worker_id],
            ),
        )
        p.start()
        workers.append(p)
        print(f"Started worker {worker_id} (PID: {p.pid})")

    # Give workers time to initialize
    print("Waiting for workers to initialize...")
    time.sleep(10)

    total_frames = 0
    rollout_num = 0
    frames_since_opponent_rotation = 0

    try:
        while total_frames < args.max_frames:
            rollout_num += 1
            print(f"\n{'='*60}")
            print(f"ROLLOUT {rollout_num} | Total frames: {total_frames:,}")
            print(f"{'='*60}")

            # Collect rollout
            model.eval()
            rollout = slicer.collect_rollout(state_queue, action_queues)
            total_frames += rollout.total_frames
            frames_since_opponent_rotation += rollout.total_frames

            # Pause workers for training
            for control_queue in control_queues.values():
                control_queue.put("pause")

            # Prepare training data
            windows = slicer.prepare_training_data(rollout, config.seq_len, device)

            if "states" not in windows or windows["states"].shape[0] == 0:
                print("Warning: No valid training windows")
                # Resume workers
                for control_queue in control_queues.values():
                    control_queue.put("resume")
                continue

            # Train on rollout
            model.train()
            train_metrics = train_on_windows(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                windows=windows,
                device=device,
                ppo_cfg=ppo_cfg,
            )

            # Log metrics
            metrics = {
                "rollout/num": rollout_num,
                "rollout/total_frames": total_frames,
                "rollout/frames_collected": rollout.total_frames,
                **train_metrics,
            }
            logger.log_metrics(metrics, step=rollout_num)

            # Clear step records
            slicer.clear_records()

            # Resume workers
            for control_queue in control_queues.values():
                control_queue.put("resume")

            # Opponent rotation
            if frames_since_opponent_rotation >= ppo_cfg.opponent_rotation_interval:
                frames_since_opponent_rotation = 0
                if not opponent_pool.is_empty():
                    opponent_path, meta = opponent_pool.sample_opponent()
                    coordinator.load_opponent(opponent_path)
                    print(f"Rotated opponent to: {opponent_path.name}")

                # Also add current model to pool
                opponent_pool.add_opponent(
                    model,
                    metadata={
                        "rollout": rollout_num,
                        "total_frames": total_frames,
                    },
                )

            # Save checkpoint
            if rollout_num % args.save_every == 0:
                save_checkpoint(model, optimizer, rollout_num, args.out_dir)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    finally:
        # Shutdown workers
        print("Shutting down workers...")
        for control_queue in control_queues.values():
            control_queue.put("shutdown")

        for worker in workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()

        # Save final checkpoint
        save_checkpoint(model, optimizer, rollout_num, args.out_dir)

        print("Distributed training complete!")


def train_on_windows(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    windows: Dict[str, torch.Tensor],
    device: torch.device,
    ppo_cfg,
) -> Dict[str, float]:
    """Train on pre-built sequence windows.

    Args:
        model: Model to train
        optimizer: Optimizer
        scaler: Gradient scaler
        windows: Pre-built training windows
        device: Training device
        ppo_cfg: PPO configuration

    Returns:
        Training metrics
    """
    from column_map import ColumnMap
    from train.batch_utils import build_model_inputs

    config = get_config()
    feature_names = get_feature_names()
    target_names = get_target_names()
    colmap = ColumnMap(feature_names, target_names)

    num_windows = windows["states"].shape[0]
    seq_len = windows["states"].shape[1]
    warmup_positions = seq_len // 4

    print(f"Training on {num_windows} windows")

    metrics = {"train/num_windows": num_windows}
    all_losses = []

    for ppo_epoch in range(ppo_cfg.ppo_epochs):
        perm = torch.randperm(num_windows, device=device)
        num_minibatches = max(1, num_windows // ppo_cfg.minibatch_size)
        epoch_losses = []

        for mb_idx in range(num_minibatches):
            start_idx = mb_idx * ppo_cfg.minibatch_size
            end_idx = min((mb_idx + 1) * ppo_cfg.minibatch_size, num_windows)
            mb_indices = perm[start_idx:end_idx]
            mb_size = len(mb_indices)

            # Get minibatch
            mb_states = windows["states"][mb_indices]
            mb_advantages = windows["advantages"][mb_indices]
            mb_returns = windows["returns"][mb_indices]
            mb_old_log_probs = windows["old_log_probs"][mb_indices]
            mb_valid_mask = windows["valid_mask"][mb_indices]
            mb_old_values = windows["values"][mb_indices]

            # Actions
            mb_actions_taken = {}
            for key in windows.keys():
                if key.startswith("actions_"):
                    head = key.replace("actions_", "")
                    mb_actions_taken[head] = windows[key][mb_indices]

            # Loss mask
            loss_mask = mb_valid_mask.clone()
            loss_mask[:, :warmup_positions] = False

            if not loss_mask.any():
                continue

            with autocast(device_type=device.type, enabled=False):
                model_inputs = build_model_inputs(mb_states, colmap)
                outputs = model(model_inputs)

                new_action_logits = {}
                for head in ["main_stick", "c_stick", "buttons", "shoulder"]:
                    if head in outputs:
                        new_action_logits[head] = outputs[head]

                new_values = outputs.get(
                    "value", torch.zeros(mb_size, seq_len, 1, device=device)
                )[:, :, 0]

                # Skip if NaN
                has_nan = any(
                    torch.isnan(v).any() or torch.isinf(v).any()
                    for v in new_action_logits.values()
                )
                if has_nan or torch.isnan(new_values).any():
                    print("Warning: NaN in model outputs, skipping batch")
                    continue

                loss, loss_metrics = compute_total_ppo_loss(
                    new_action_logits=new_action_logits,
                    new_values=new_values,
                    old_values=mb_old_values,
                    actions_taken=mb_actions_taken,
                    old_log_probs=mb_old_log_probs,
                    advantages=mb_advantages,
                    returns=mb_returns,
                    clip_ratio=ppo_cfg.clip_ratio,
                    entropy_coef=ppo_cfg.entropy_coef,
                    value_coef=config.rl.value_loss_coef,
                    value_clip=ppo_cfg.value_clip,
                    loss_mask=loss_mask,
                )

            if torch.isnan(loss) or torch.isinf(loss):
                print("Warning: NaN loss, skipping batch")
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Check gradients
            has_nan_grad = any(
                p.grad is not None
                and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                for p in model.parameters()
            )
            if has_nan_grad:
                print("Warning: NaN gradients, skipping step")
                optimizer.zero_grad()
                continue

            torch.nn.utils.clip_grad_norm_(model.parameters(), ppo_cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

            if mb_idx == 0 and ppo_epoch == 0:
                for key, value in loss_metrics.items():
                    metrics[key] = value

        if epoch_losses:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(
                f"  Epoch {ppo_epoch + 1}/{ppo_cfg.ppo_epochs}: loss = {avg_loss:.4f}"
            )
            all_losses.extend(epoch_losses)

    if all_losses:
        metrics["train/avg_loss"] = sum(all_losses) / len(all_losses)

    return metrics


# =============================================================================
# Episode-Based Training Mode (Original)
# =============================================================================


def run_episode_worker(
    worker_id: int,
    model_state_dict_path: Path,
    opponent_state_dict_path: Optional[Path],
    dolphin_path: Path,
    iso_path: Path,
    seq_len: int,
    max_episode_frames: int,
) -> Tuple[Optional[Trajectory], Dict[str, float]]:
    """Worker function to run a single episode in a separate process."""
    init_config()
    from config import get_config

    config = get_config()
    device = torch.device("cpu")
    model = GPT(config).to(device)
    model_state = strip_compiled_prefix(
        torch.load(model_state_dict_path, map_location=device, weights_only=True)
    )
    model.load_state_dict(model_state)
    model.eval()

    opponent_model = None
    if opponent_state_dict_path is not None and opponent_state_dict_path.exists():
        opponent_model = GPT(config).to(device)
        opponent_state = strip_compiled_prefix(
            torch.load(opponent_state_dict_path, map_location=device, weights_only=True)
        )
        opponent_model.load_state_dict(opponent_state)
        opponent_model.eval()

    import shutil

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

    env.initialize_console()

    try:
        print(f"[Worker {worker_id}] Starting episode...")
        episode_metrics = run_episode(env)
        trajectories = env.trajectory_buffer.get_trajectories()
        trajectory = trajectories[0] if len(trajectories) > 0 else None
        print(
            f"[Worker {worker_id}] Episode complete: {episode_metrics.get('episode/frames', 0)} frames"
        )
        return trajectory, episode_metrics

    finally:
        env.console.stop()
        shutil.rmtree(pool_dir, ignore_errors=True)


def run_episode(env: SelfPlayEnvironment) -> Dict[str, float]:
    """Run one episode of self-play."""
    env.reset_episode()

    episode_metrics = {
        "episode/frames": 0,
        "episode/total_reward": 0.0,
    }

    print("Waiting for game to start...")
    while True:
        gamestate = env.console.step()
        if gamestate is None:
            continue

        if gamestate.menu_state in [Menu.IN_GAME, Menu.SUDDEN_DEATH]:
            print("Game started!")
            break
        else:
            env.navigate_menu(gamestate)

    print("Playing episode...")
    done = False
    while not done:
        gamestate = env.console.step()
        if gamestate is None:
            continue

        if gamestate.menu_state not in [Menu.IN_GAME, Menu.SUDDEN_DEATH]:
            print("Game ended (left game state)")
            done = True
            break

        done, step_metrics = env.step(gamestate)

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
    """Build overlapping sequence windows from trajectories."""
    all_windows = {
        "states": [],
        "advantages": [],
        "returns": [],
        "old_log_probs": [],
        "values": [],
        "valid_mask": [],
    }

    action_keys = []
    for traj in trajectories:
        if len(traj.steps) > 0:
            for head in ["main_stick", "c_stick", "buttons", "shoulder"]:
                if head in traj.steps[0].action_taken:
                    action_keys.append(head)
            break

    for head in action_keys:
        all_windows[f"actions_{head}"] = []

    for traj in trajectories:
        if traj.advantages is None or traj.returns is None:
            continue

        T = len(traj.steps)
        if T == 0:
            continue

        states = torch.stack([step.state for step in traj.steps])
        advantages = traj.advantages
        returns = traj.returns
        old_log_probs = torch.stack([step.log_prob for step in traj.steps]).squeeze(-1)
        values = torch.stack([step.value for step in traj.steps]).squeeze(-1)

        actions = {}
        for head in action_keys:
            actions[head] = torch.stack(
                [step.action_taken[head] for step in traj.steps]
            )

        if T < seq_len:
            pad_len = seq_len - T
            states = torch.cat([states[0:1].expand(pad_len, -1), states], dim=0)
            advantages = torch.cat([torch.zeros(pad_len), advantages], dim=0)
            returns = torch.cat([returns[0:1].expand(pad_len), returns], dim=0)
            old_log_probs = torch.cat(
                [old_log_probs[0:1].expand(pad_len), old_log_probs], dim=0
            )
            values = torch.cat([values[0:1].expand(pad_len), values], dim=0)

            for head in action_keys:
                if actions[head].dim() == 1:
                    actions[head] = torch.cat(
                        [actions[head][0:1].expand(pad_len), actions[head]], dim=0
                    )
                else:
                    actions[head] = torch.cat(
                        [actions[head][0:1].expand(pad_len, -1), actions[head]], dim=0
                    )

            valid_mask = torch.cat(
                [
                    torch.zeros(pad_len, dtype=torch.bool),
                    torch.ones(T, dtype=torch.bool),
                ]
            )

            all_windows["states"].append(states.unsqueeze(0))
            all_windows["advantages"].append(advantages.unsqueeze(0))
            all_windows["returns"].append(returns.unsqueeze(0))
            all_windows["old_log_probs"].append(old_log_probs.unsqueeze(0))
            all_windows["values"].append(values.unsqueeze(0))
            all_windows["valid_mask"].append(valid_mask.unsqueeze(0))

            for head in action_keys:
                all_windows[f"actions_{head}"].append(actions[head].unsqueeze(0))
        else:
            for start in range(0, T - seq_len + 1, stride):
                end = start + seq_len
                all_windows["states"].append(states[start:end].unsqueeze(0))
                all_windows["advantages"].append(advantages[start:end].unsqueeze(0))
                all_windows["returns"].append(returns[start:end].unsqueeze(0))
                all_windows["old_log_probs"].append(
                    old_log_probs[start:end].unsqueeze(0)
                )
                all_windows["values"].append(values[start:end].unsqueeze(0))
                all_windows["valid_mask"].append(
                    torch.ones(seq_len, dtype=torch.bool).unsqueeze(0)
                )

                for head in action_keys:
                    all_windows[f"actions_{head}"].append(
                        actions[head][start:end].unsqueeze(0)
                    )

            if (T - seq_len) % stride != 0:
                start = T - seq_len
                all_windows["states"].append(states[start:].unsqueeze(0))
                all_windows["advantages"].append(advantages[start:].unsqueeze(0))
                all_windows["returns"].append(returns[start:].unsqueeze(0))
                all_windows["old_log_probs"].append(old_log_probs[start:].unsqueeze(0))
                all_windows["values"].append(values[start:].unsqueeze(0))
                all_windows["valid_mask"].append(
                    torch.ones(seq_len, dtype=torch.bool).unsqueeze(0)
                )

                for head in action_keys:
                    all_windows[f"actions_{head}"].append(
                        actions[head][start:].unsqueeze(0)
                    )

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
    """Train model on collected trajectories using PPO."""
    from column_map import ColumnMap
    from train.batch_utils import build_model_inputs

    config = get_config()
    ppo_cfg = config.ppo

    if len(trajectories) == 0:
        print("Warning: No trajectories to train on")
        return {}

    print(f"\nTraining on {len(trajectories)} trajectories...")

    for traj in trajectories:
        traj.compute_gae(
            gamma=config.rl.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            normalize=ppo_cfg.normalize_advantages,
        )

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

    print(
        f"Created {num_windows} sequence windows of length {seq_len} (stride={window_stride})"
    )
    print(f"Total original steps: {total_steps}")

    target_names = get_target_names()
    colmap = ColumnMap(feature_names, target_names)

    metrics = {
        "train/total_steps": total_steps,
        "train/num_trajectories": len(trajectories),
        "train/num_windows": num_windows,
    }

    warmup_positions = config.seq_len // 4

    for ppo_epoch in range(ppo_cfg.ppo_epochs):
        perm = torch.randperm(num_windows, device=device)
        num_minibatches = max(1, num_windows // ppo_cfg.minibatch_size)
        epoch_losses = []

        for mb_idx in range(num_minibatches):
            start_idx = mb_idx * ppo_cfg.minibatch_size
            end_idx = min((mb_idx + 1) * ppo_cfg.minibatch_size, num_windows)
            mb_indices = perm[start_idx:end_idx]
            mb_size = len(mb_indices)

            mb_states = windows["states"][mb_indices]
            mb_advantages = windows["advantages"][mb_indices]
            mb_returns = windows["returns"][mb_indices]
            mb_old_log_probs = windows["old_log_probs"][mb_indices]
            mb_valid_mask = windows["valid_mask"][mb_indices]
            mb_old_values = windows["values"][mb_indices]

            mb_actions_taken = {}
            for key in windows.keys():
                if key.startswith("actions_"):
                    head = key.replace("actions_", "")
                    mb_actions_taken[head] = windows[key][mb_indices]

            loss_mask = mb_valid_mask.clone()
            loss_mask[:, :warmup_positions] = False

            if not loss_mask.any():
                continue

            with autocast(device_type=device.type, enabled=False):
                model_inputs = build_model_inputs(mb_states, colmap)
                outputs = model(model_inputs)

                new_action_logits = {}
                for head in ["main_stick", "c_stick", "buttons", "shoulder"]:
                    if head in outputs:
                        new_action_logits[head] = outputs[head]

                new_values = outputs.get(
                    "value", torch.zeros(mb_size, seq_len, 1, device=device)
                )[:, :, 0]

                has_nan_output = False
                for head, logits in new_action_logits.items():
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        has_nan_output = True
                if torch.isnan(new_values).any() or torch.isinf(new_values).any():
                    has_nan_output = True

                if has_nan_output:
                    continue

                loss, loss_metrics = compute_total_ppo_loss(
                    new_action_logits=new_action_logits,
                    new_values=new_values,
                    old_values=mb_old_values,
                    actions_taken=mb_actions_taken,
                    old_log_probs=mb_old_log_probs,
                    advantages=mb_advantages,
                    returns=mb_returns,
                    clip_ratio=ppo_cfg.clip_ratio,
                    entropy_coef=ppo_cfg.entropy_coef,
                    value_coef=config.rl.value_loss_coef,
                    value_clip=ppo_cfg.value_clip,
                    loss_mask=loss_mask,
                )

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            nan_grads = sum(
                1
                for p in model.parameters()
                if p.grad is not None
                and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            )

            if nan_grads > 0:
                optimizer.zero_grad()
                continue

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), ppo_cfg.max_grad_norm
            )

            if mb_idx == 0 and ppo_epoch == 0:
                metrics["train/grad_norm"] = grad_norm.item()

            scaler.step(optimizer)
            scaler.update()

            epoch_losses.append(loss.item())

            if mb_idx == 0 and ppo_epoch == 0:
                for key, value in loss_metrics.items():
                    metrics[key] = value

        if len(epoch_losses) > 0:
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(
                f"  Epoch {ppo_epoch + 1}/{ppo_cfg.ppo_epochs}: avg loss = {avg_loss:.4f}"
            )
            metrics[f"train/ppo_epoch_{ppo_epoch}_loss"] = avg_loss

    logger.log_metrics(metrics, step=episode)
    return metrics


def run_episode_based_training(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    opponent_pool: OpponentPool,
    device: torch.device,
    logger: WandbLogger,
    args,
    start_episode: int,
) -> None:
    """Run original episode-based training."""
    config = get_config()
    num_workers = config.ppo.num_workers
    use_parallel = num_workers > 1

    print(
        f"Using {'parallel' if use_parallel else 'sequential'} trajectory collection with {num_workers} worker(s)"
    )

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
        for episode_idx in range(start_episode, start_episode + args.num_episodes):
            episode_num = episode_idx + 1
            print(f"\n{'='*80}")
            print(f"Episode {episode_num}/{start_episode + args.num_episodes}")
            print(f"{'='*80}")

            model.eval()

            trajectories = []
            all_episode_metrics = []

            if use_parallel:
                print(f"Collecting {num_workers} trajectories in parallel...")

                model_temp = Path(tempfile.mkdtemp()) / "learner_model.pt"
                torch.save(model.state_dict(), model_temp)

                opponent_temp = None
                if not opponent_pool.is_empty():
                    opponent_checkpoint_path, _ = opponent_pool.sample_opponent()
                    opponent_model = GPT(config).to(device)
                    opponent_pool.load_opponent_model(
                        opponent_model, opponent_checkpoint_path
                    )
                    opponent_temp = Path(model_temp).parent / "opponent_model.pt"
                    torch.save(opponent_model.state_dict(), opponent_temp)

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

                with mp.Pool(processes=num_workers) as pool:
                    results = pool.starmap(run_episode_worker, worker_args)

                for trajectory, metrics in results:
                    if trajectory is not None:
                        trajectories.append(trajectory)
                    all_episode_metrics.append(metrics)

                import shutil

                shutil.rmtree(model_temp.parent, ignore_errors=True)

                episode_metrics = {}
                if all_episode_metrics:
                    for key in all_episode_metrics[0].keys():
                        values = [m.get(key, 0.0) for m in all_episode_metrics]
                        episode_metrics[key] = sum(values) / len(values)

            else:
                episode_metrics = run_episode(env)
                trajectories = env.trajectory_buffer.get_trajectories()

            logger.log_metrics(episode_metrics, step=episode_num)

            print(f"Collected {len(trajectories)} trajectories")

            if len(trajectories) > 0:
                model.train()
                feature_names = (
                    env.feature_names if env is not None else get_feature_names()
                )

                train_on_trajectories(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    trajectories=trajectories,
                    device=device,
                    logger=logger,
                    episode=episode_num,
                    feature_names=feature_names,
                )

            if not use_parallel and env is not None:
                env.trajectory_buffer.clear()

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

                pool_stats = opponent_pool.get_pool_stats()
                logger.log_metrics(
                    {
                        "pool/size": pool_stats["pool_size"],
                        "pool/total_created": pool_stats["total_opponents_created"],
                    },
                    step=episode_num,
                )

            if episode_num % args.save_every == 0:
                save_checkpoint(model, optimizer, episode_num, args.out_dir)

        print("\nTraining complete!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    finally:
        if env is not None:
            print("Shutting down environment...")
            env.shutdown_console()

        save_checkpoint(model, optimizer, episode_num, args.out_dir)

        print("Done!")


# =============================================================================
# Main Entry Point
# =============================================================================


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
        "--num-episodes",
        type=int,
        default=1000,
        help="Number of episodes to train (episode mode)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10_000_000,
        help="Maximum frames to train (distributed mode)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N episodes/rollouts",
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
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use distributed training with centralized GPU inference",
    )

    args = parser.parse_args()

    # Initialize config
    init_config()
    config = get_config()

    # Override distributed mode from CLI
    if args.distributed:
        # Create new config with distributed_mode enabled
        from config.config import set_config, Config

        config_dict = config.to_dict()
        config_dict["ppo"]["distributed_mode"] = True
        new_config = Config.model_validate(config_dict)
        set_config(new_config)
        config = new_config

    # Setup device
    device = _resolve_device(None)
    print(f"Using device: {device}")

    # Enable TF32 for faster matmul on Ampere+ GPUs (new PyTorch 2.9+ API)
    if device.type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'

    # Create model
    model = GPT(config).to(device)
    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Load initial checkpoint if provided
    start_episode = load_initial_checkpoint(model, args.checkpoint, device)
    start_episode += 1

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.ppo.lr,
        betas=config.train.betas,
        weight_decay=config.train.weight_decay,
    )
    print(f"Using PPO learning rate: {config.ppo.lr:.2e}")

    # Gradient scaler
    scaler = GradScaler('cuda', enabled=config.train.use_amp)

    # Opponent pool
    pool_dir = args.out_dir / "opponent_pool"
    opponent_pool = OpponentPool(
        max_size=config.ppo.pool_size,
        pool_dir=pool_dir,
    )

    if opponent_pool.is_empty():
        print("Opponent pool is empty. Adding initial model...")
        opponent_pool.add_opponent(
            model,
            metadata={"episode": start_episode, "note": "initial_model"},
        )

    # Initialize wandb
    mode = "distributed" if config.ppo.distributed_mode else "episode"
    wandb_cfg = WandbConfig(
        project="melee-ai-ppo",
        name=f"ppo_{mode}_{start_episode}",
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

    # Run training
    if config.ppo.distributed_mode:
        run_distributed_training(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            opponent_pool=opponent_pool,
            device=device,
            logger=logger,
            args=args,
        )
    else:
        run_episode_based_training(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            opponent_pool=opponent_pool,
            device=device,
            logger=logger,
            args=args,
            start_episode=start_episode,
        )


if __name__ == "__main__":
    # Set spawn method for CUDA multiprocessing compatibility
    try:
        mp.set_start_method('spawn', force=False)
    except RuntimeError:
        # Start method already set
        pass
    main()
