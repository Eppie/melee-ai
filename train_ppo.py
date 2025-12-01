#!/usr/bin/env python3
"""PPO self-play training script for Melee AI.

Uses distributed mode with centralized GPU inference and async training.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.amp import autocast

from config import get_config, init_config
from model.nano_gpt import GPT
from ppo.opponent_pool import OpponentPool
from ppo.ppo_loss import compute_total_ppo_loss
from schema import get_feature_names, get_target_names
from train.wandb_utils import WandbConfig, WandbLogger, init_wandb
from utils import _resolve_device, match_state_dict_keys, Profiler


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
    model_state = match_state_dict_keys(ckpt["model"], model)
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
    model = torch.compile(model, mode="reduce-overhead")

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

    # Create profilers for main loop operations
    prof_checkpoint = Profiler(burnin=0)
    prof_opponent_rotation = Profiler(burnin=0)

    # Create async trainer if enabled
    async_trainer = None
    if ppo_cfg.async_training:
        from ppo.async_trainer import AsyncPPOTrainer

        print("Starting async training process...")
        async_trainer = AsyncPPOTrainer(
            model_state_dict=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            config_dict=config.to_dict(),
        )
        print(
            f"Async trainer started (gradient accumulation: {ppo_cfg.gradient_accumulation_steps})"
        )

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

            # Handle training based on mode
            if async_trainer is not None:
                # ASYNC MODE: Submit rollout and check for updates
                submitted = async_trainer.submit_rollout(rollout)
                if not submitted:
                    print("Warning: Async trainer queue full, rollout not submitted")

                # Check for parameter updates (non-blocking)
                new_params = async_trainer.get_updated_parameters()
                if new_params is not None:
                    print("Applying updated parameters from async trainer")
                    model.load_state_dict(new_params)
                    coordinator.learner_model.load_state_dict(new_params)

                # Check for metrics (non-blocking)
                train_metrics = async_trainer.get_metrics()
                if train_metrics is not None:
                    metrics = {
                        "rollout/num": rollout_num,
                        "rollout/total_frames": total_frames,
                        "rollout/frames_collected": rollout.total_frames,
                        "train/avg_loss": train_metrics.avg_loss,
                        "train/policy_loss": train_metrics.policy_loss,
                        "train/value_loss": train_metrics.value_loss,
                        "train/entropy": train_metrics.entropy,
                        "train/approx_kl": train_metrics.approx_kl,
                        "train/clipped_fraction": train_metrics.clipped_fraction,
                        "train/num_windows": train_metrics.num_windows,
                        "train/reverted": int(train_metrics.reverted),
                        # Profiling stats
                        "train/time_prepare_windows_ms": train_metrics.time_prepare_windows_ms,
                        "train/time_combine_windows_ms": train_metrics.time_combine_windows_ms,
                        "train/time_forward_ms": train_metrics.time_forward_ms,
                        "train/time_loss_ms": train_metrics.time_loss_ms,
                        "train/time_backward_ms": train_metrics.time_backward_ms,
                        "train/time_optimizer_ms": train_metrics.time_optimizer_ms,
                        "train/time_total_training_ms": train_metrics.time_total_training_ms,
                    }
                    logger.log_metrics(metrics, step=rollout_num)

                    # Print training performance summary
                    print(f"\n{'='*60}")
                    print("TRAINING PERFORMANCE")
                    print(f"{'='*60}")
                    print(
                        f"Total training time: {train_metrics.time_total_training_ms:.1f}ms"
                    )
                    print(
                        f"  Prepare windows:   {train_metrics.time_prepare_windows_ms:.1f}ms"
                    )
                    print(
                        f"  Combine windows:   {train_metrics.time_combine_windows_ms:.1f}ms"
                    )
                    print(f"  Forward pass:      {train_metrics.time_forward_ms:.1f}ms")
                    print(f"  Loss computation:  {train_metrics.time_loss_ms:.1f}ms")
                    print(
                        f"  Backward pass:     {train_metrics.time_backward_ms:.1f}ms"
                    )
                    print(
                        f"  Optimizer step:    {train_metrics.time_optimizer_ms:.1f}ms"
                    )
                    print(f"{'='*60}\n")

                    if train_metrics.reverted:
                        print("⚠️  Training update reverted due to high KL divergence")

                # Clear step records
                slicer.clear_records()

            else:
                # SYNC MODE: Pause workers, train, resume (original behavior)
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
                with prof_opponent_rotation:
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
                print(
                    f"Opponent rotation took {prof_opponent_rotation.last_duration() * 1000:.1f}ms"
                )

            # Save checkpoint
            if rollout_num % args.save_every == 0:
                with prof_checkpoint:
                    save_checkpoint(model, optimizer, rollout_num, args.out_dir)
                print(
                    f"Checkpoint save took {prof_checkpoint.last_duration() * 1000:.1f}ms"
                )

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    finally:
        # Shutdown async trainer first
        if async_trainer is not None:
            print("Shutting down async trainer...")
            async_trainer.shutdown()

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

        # Print overall profiling summary
        print(f"\n{'='*60}")
        print("OVERALL PROFILING SUMMARY")
        print(f"{'='*60}")
        if prof_checkpoint.num_calls > 0:
            print(
                f"Checkpoints: {prof_checkpoint.num_calls} saves, avg {prof_checkpoint.mean_time() * 1000:.1f}ms, total {prof_checkpoint.total_time():.2f}s"
            )
        if prof_opponent_rotation.num_calls > 0:
            print(
                f"Opponent rotation: {prof_opponent_rotation.num_calls} rotations, avg {prof_opponent_rotation.mean_time() * 1000:.1f}ms, total {prof_opponent_rotation.total_time():.2f}s"
            )
        print(f"{'='*60}\n")

        print("Distributed training complete!")


def train_on_windows(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    windows: Dict[str, torch.Tensor],
    device: torch.device,
    ppo_cfg,
) -> Dict[str, float]:
    """Train on pre-built sequence windows.

    Args:
        model: Model to train
        optimizer: Optimizer
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
    warmup_positions = seq_len

    print(f"Training on {num_windows} windows")

    metrics = {"train/num_windows": num_windows}
    all_losses = []

    # Create profilers
    prof_forward = Profiler(burnin=0)
    prof_loss = Profiler(burnin=0)
    prof_backward = Profiler(burnin=0)
    prof_optimizer = Profiler(burnin=0)

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

            # Forward pass
            with prof_forward:
                with autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=True
                ):
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

            # Loss computation
            with prof_loss:
                with autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=True
                ):
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

            # Backward pass
            with prof_backward:
                optimizer.zero_grad()
                loss.backward()

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

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), ppo_cfg.max_grad_norm
                )

            # Optimizer step
            with prof_optimizer:
                optimizer.step()

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

    # Add profiling stats
    metrics["train/time_forward_ms"] = prof_forward.mean_time() * 1000
    metrics["train/time_loss_ms"] = prof_loss.mean_time() * 1000
    metrics["train/time_backward_ms"] = prof_backward.mean_time() * 1000
    metrics["train/time_optimizer_ms"] = prof_optimizer.mean_time() * 1000

    # Print training performance summary
    print(f"\n{'='*60}")
    print("TRAINING PERFORMANCE")
    print(f"{'='*60}")
    print(
        f"Forward pass:      {prof_forward.mean_time() * 1000:.1f}ms (total: {prof_forward.total_time():.2f}s, {prof_forward.num_calls} calls)"
    )
    print(
        f"Loss computation:  {prof_loss.mean_time() * 1000:.1f}ms (total: {prof_loss.total_time():.2f}s, {prof_loss.num_calls} calls)"
    )
    print(
        f"Backward pass:     {prof_backward.mean_time() * 1000:.1f}ms (total: {prof_backward.total_time():.2f}s, {prof_backward.num_calls} calls)"
    )
    print(
        f"Optimizer step:    {prof_optimizer.mean_time() * 1000:.1f}ms (total: {prof_optimizer.total_time():.2f}s, {prof_optimizer.num_calls} calls)"
    )
    total_training = (
        prof_forward.total_time()
        + prof_loss.total_time()
        + prof_backward.total_time()
        + prof_optimizer.total_time()
    )
    print(f"Total training:    {total_training:.2f}s")
    print(f"{'='*60}\n")

    return metrics


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
        "--max-frames",
        type=int,
        default=10_000_000,
        help="Maximum frames to train",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N rollouts",
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

    # Enable TF32 for faster matmul on Ampere+ GPUs (new PyTorch 2.9+ API)
    if device.type == "cuda":
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"

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
    wandb_cfg = WandbConfig(
        project="melee-ai-ppo",
        name=f"ppo_distributed_{start_episode}",
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

    # Run distributed training
    run_distributed_training(
        model=model,
        optimizer=optimizer,
        opponent_pool=opponent_pool,
        device=device,
        logger=logger,
        args=args,
    )


if __name__ == "__main__":
    # Set fork method for CUDA multiprocessing on Linux (spawn causes issues with CUDA tensors)
    try:
        mp.set_start_method("fork", force=False)
    except RuntimeError:
        # Start method already set
        pass
    main()
