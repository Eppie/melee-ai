"""Asynchronous PPO trainer that runs in a separate process.

This module implements the AsyncPPOTrainer which runs PPO training in a separate process,
allowing data collection to continue in parallel. This eliminates the pause-train-resume cycle
and significantly improves throughput.

Key features:
- Separate process for training (no GIL contention)
- Gradient accumulation across multiple rollouts
- Safety mechanisms (checkpoint reversion, early stopping)
- Optional teacher distillation for preventing catastrophic forgetting
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.amp import autocast

from column_map import ColumnMap
from config import get_config
from model.nano_gpt import GPT
from ppo.ppo_loss import compute_total_ppo_loss
from ppo.trajectory_slicer import RolloutSlice
from schema import get_feature_names, get_target_names
from train.batch_utils import build_model_inputs
from utils import _resolve_device, Profiler

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics from a training step."""

    avg_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clipped_fraction: float
    num_windows: int
    reverted: bool = False

    # Profiling stats (in milliseconds for easier reading)
    time_prepare_windows_ms: float = 0.0
    time_combine_windows_ms: float = 0.0
    time_forward_ms: float = 0.0
    time_loss_ms: float = 0.0
    time_backward_ms: float = 0.0
    time_optimizer_ms: float = 0.0
    time_total_training_ms: float = 0.0


class AsyncPPOTrainer:
    """Asynchronous PPO trainer running in a separate process.

    This trainer runs in a dedicated process with its own CUDA context,
    allowing it to train while the main process continues data collection.

    Architecture:
        Main Process                  Training Process
        ============                  ================
        Collect rollout        <--    Train on rollouts
        Submit to queue        -->    Accumulate gradients
        Check for updates      <--    Send updated params
        Apply new params       <--    Send metrics
    """

    def __init__(
        self,
        model_state_dict: Dict[str, torch.Tensor],
        optimizer_state: Optional[Dict] = None,
        config_dict: Optional[Dict] = None,
    ):
        """Initialize the async trainer.

        Args:
            model_state_dict: Initial model parameters
            optimizer_state: Optional optimizer state to resume from
            config_dict: Configuration dictionary (if None, uses current config)
        """
        # Communication queues
        # Queue should be larger than gradient_accumulation_steps to prevent deadlock
        self.rollout_queue: mp.Queue = mp.Queue(maxsize=10)  # Buffer for multiple rollouts
        self.parameter_queue: mp.Queue = mp.Queue(maxsize=1)  # Latest params
        self.metrics_queue: mp.Queue = mp.Queue(maxsize=20)  # Metrics stream
        self.shutdown_event: mp.Event = mp.Event()

        # Get config
        if config_dict is None:
            config_dict = get_config().to_dict()

        # Spawn training process
        self.process = mp.Process(
            target=self._training_process,
            args=(
                model_state_dict,
                optimizer_state,
                config_dict,
                self.rollout_queue,
                self.parameter_queue,
                self.metrics_queue,
                self.shutdown_event,
            ),
            daemon=False,  # Ensure clean shutdown
        )
        self.process.start()

        logger.info(f"AsyncPPOTrainer started with PID {self.process.pid}")
        print(f"AsyncPPOTrainer process started with PID {self.process.pid}")

    def submit_rollout(self, rollout: RolloutSlice) -> bool:
        """Submit a rollout for training (non-blocking).

        Args:
            rollout: Rollout to train on

        Returns:
            True if submitted successfully, False if queue is full
        """
        # Check if process is alive
        if not self.process.is_alive():
            logger.error(f"Training process is dead! Exit code: {self.process.exitcode}")
            print(f"ERROR: Training process is dead! Exit code: {self.process.exitcode}")
            return False

        try:
            self.rollout_queue.put_nowait(rollout)
            return True
        except queue.Full:
            logger.warning("Rollout queue full, skipping submission")
            return False

    def get_updated_parameters(self) -> Optional[Dict[str, torch.Tensor]]:
        """Check for parameter updates (non-blocking).

        Returns:
            Updated state_dict if available, None otherwise
        """
        try:
            return self.parameter_queue.get_nowait()
        except queue.Empty:
            return None

    def get_metrics(self) -> Optional[TrainingMetrics]:
        """Check for training metrics (non-blocking).

        Returns:
            Training metrics if available, None otherwise
        """
        try:
            return self.metrics_queue.get_nowait()
        except queue.Empty:
            return None

    def shutdown(self, timeout: float = 10.0):
        """Shutdown the training process gracefully.

        Args:
            timeout: Maximum time to wait for process to terminate
        """
        logger.info("Shutting down AsyncPPOTrainer...")
        self.shutdown_event.set()
        self.process.join(timeout=timeout)

        if self.process.is_alive():
            logger.warning("Training process did not terminate, forcing...")
            self.process.terminate()
            self.process.join(timeout=1.0)

        if self.process.is_alive():
            logger.error("Could not terminate training process")
        else:
            logger.info("AsyncPPOTrainer shut down successfully")

    @staticmethod
    def _training_process(
        model_state: Dict[str, torch.Tensor],
        optimizer_state: Optional[Dict],
        config_dict: Dict,
        rollout_queue: mp.Queue,
        param_queue: mp.Queue,
        metrics_queue: mp.Queue,
        shutdown: mp.Event,
    ):
        """Training process loop.

        This runs in a separate process with its own CUDA context.
        It accumulates rollouts and trains when enough have been collected.
        """
        print("[AsyncTrainer] Training process _training_process() started", flush=True)

        # Rebuild config in this process
        from config import Config

        config = Config.model_validate(config_dict)

        # Set as global config
        from config import config as config_module

        config_module._GLOBAL_CONFIG = config
        ppo_cfg = config.ppo
        rl_cfg = config.rl

        # Set up device
        device = _resolve_device()
        logger.info(f"Training process using device: {device}")

        # Rebuild model in this process
        model = GPT(config)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.train()

        # Build optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=ppo_cfg.lr, weight_decay=0.0
        )
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)

        # Optional: Load teacher model if configured
        teacher_model = None
        if (
            hasattr(ppo_cfg, "teacher_kl_weight")
            and ppo_cfg.teacher_kl_weight > 0
            and hasattr(ppo_cfg, "teacher_checkpoint_path")
            and ppo_cfg.teacher_checkpoint_path is not None
        ):
            teacher_path = Path(ppo_cfg.teacher_checkpoint_path)
            if teacher_path.exists():
                logger.info(f"Loading teacher model from {teacher_path}")
                teacher_model = GPT(config)
                ckpt = torch.load(teacher_path, map_location=device, weights_only=False)
                teacher_model.load_state_dict(ckpt["model"])
                teacher_model = teacher_model.to(device)
                teacher_model.eval()
                logger.info("Teacher model loaded for distillation")
            else:
                logger.warning(f"Teacher checkpoint not found: {teacher_path}")

        # Get gradient accumulation steps
        gradient_accumulation_steps = getattr(ppo_cfg, "gradient_accumulation_steps", 3)

        # Create helper instances
        feature_names = get_feature_names()
        target_names = get_target_names()
        colmap = ColumnMap(feature_names, target_names)

        # Gradient accumulation buffer
        accumulated_rollouts: List[RolloutSlice] = []
        step_count = 0

        logger.info("Training process ready")

        # Main training loop
        print(f"[AsyncTrainer] Entering training loop", flush=True)
        while not shutdown.is_set():
            try:
                # Get rollout with timeout
                rollout = rollout_queue.get(timeout=1.0)
                print(f"[AsyncTrainer] Received rollout with {len(rollout.observations)} frames", flush=True)
            except queue.Empty:
                continue

            # Accumulate rollouts
            accumulated_rollouts.append(rollout)
            logger.debug(
                f"Accumulated {len(accumulated_rollouts)}/{gradient_accumulation_steps} rollouts"
            )

            # Train when we have enough rollouts
            if len(accumulated_rollouts) >= gradient_accumulation_steps:
                logger.info(
                    f"Training on {len(accumulated_rollouts)} accumulated rollouts"
                )

                try:
                    # Train with accumulated rollouts
                    metrics = _train_with_accumulation(
                        model=model,
                        teacher_model=teacher_model,
                        optimizer=optimizer,
                        rollouts=accumulated_rollouts,
                        config=config,
                        colmap=colmap,
                        device=device,
                    )

                    # Send updated parameters back to main process
                    try:
                        # Clear old params if queue is full
                        try:
                            param_queue.get_nowait()
                        except queue.Empty:
                            pass

                        # Move state_dict to CPU for pickling across processes
                        state_dict_cpu = {
                            k: v.cpu() for k, v in model.state_dict().items()
                        }
                        param_queue.put(state_dict_cpu)
                        logger.debug("Sent updated parameters to main process")
                    except Exception as e:
                        logger.error(f"Failed to send parameters: {e}")

                    # Send metrics
                    try:
                        metrics_queue.put(metrics)
                    except queue.Full:
                        logger.warning("Metrics queue full, dropping metrics")

                    step_count += 1

                except Exception as e:
                    logger.error(f"Training error: {e}", exc_info=True)

                # Clear accumulation buffer
                accumulated_rollouts = []

        logger.info("Training process shutting down")


def _train_with_accumulation(
    model: GPT,
    teacher_model: Optional[GPT],
    optimizer: torch.optim.Optimizer,
    rollouts: List[RolloutSlice],
    config,
    colmap: ColumnMap,
    device: torch.device,
) -> TrainingMetrics:
    """Train on accumulated rollouts with gradient accumulation.

    This implements the core PPO training loop with several enhancements:
    - Gradient accumulation across multiple rollouts
    - Safety checks (checkpoint reversion on high KL)
    - Early stopping on high clipped fraction
    - Optional teacher distillation

    Args:
        model: Model to train
        teacher_model: Optional teacher for distillation
        optimizer: Optimizer
        rollouts: List of rollouts to train on
        config: Full config object
        colmap: Column mapping
        device: Training device

    Returns:
        Training metrics
    """
    ppo_cfg = config.ppo
    rl_cfg = config.rl

    # Create profilers for this training run
    prof_prepare = Profiler(burnin=0)
    prof_combine = Profiler(burnin=0)
    prof_total = Profiler(burnin=0)

    with prof_total:
        # Prepare training windows from all rollouts
        all_windows = []
        for rollout in rollouts:
            with prof_prepare:
                windows = _prepare_windows(rollout, config, device)
            all_windows.append(windows)

        # Combine windows from all rollouts
        with prof_combine:
            combined_windows = _combine_windows(all_windows)

        # Save checkpoint before training (for potential reversion)
        checkpoint_state = None
        if hasattr(ppo_cfg, "max_mean_actor_kl") and ppo_cfg.max_mean_actor_kl > 0:
            checkpoint_state = model.state_dict()

        # Run PPO epochs with gradient accumulation
        metrics = _ppo_training_loop(
            model=model,
            teacher_model=teacher_model,
            optimizer=optimizer,
            windows=combined_windows,
            ppo_cfg=ppo_cfg,
            rl_cfg=rl_cfg,
            colmap=colmap,
            device=device,
        )

        # Safety check: revert if KL divergence too high
        reverted = False
        if (
            checkpoint_state is not None
            and hasattr(ppo_cfg, "max_mean_actor_kl")
            and metrics.approx_kl > ppo_cfg.max_mean_actor_kl
        ):
            logger.warning(
                f"Actor KL {metrics.approx_kl:.4f} > {ppo_cfg.max_mean_actor_kl:.4f}, reverting checkpoint"
            )
            model.load_state_dict(checkpoint_state)
            reverted = True

    metrics.reverted = reverted

    # Add profiling stats to metrics (convert to ms)
    metrics.time_prepare_windows_ms = prof_prepare.mean_time() * 1000
    metrics.time_combine_windows_ms = prof_combine.mean_time() * 1000
    metrics.time_total_training_ms = prof_total.total_time() * 1000

    return metrics


def _prepare_windows(
    rollout: RolloutSlice, config, device: torch.device
) -> Dict[str, torch.Tensor]:
    """Prepare training windows from a rollout slice.

    This converts the rollout into overlapping windows suitable for training.
    """
    from ppo.trajectory import Trajectory, Step

    # Convert rollout to trajectory format
    all_steps = []
    for worker_id, steps in rollout.worker_steps.items():
        all_steps.extend(steps)

    if not all_steps:
        return {}

    # Create trajectory
    trajectory = Trajectory(steps=all_steps)

    # Compute GAE advantages
    trajectory.compute_gae(
        gamma=config.rl.gamma,
        gae_lambda=config.ppo.gae_lambda,
        normalize=config.ppo.normalize_advantages,
    )

    # Build windows
    seq_len = config.seq_len
    stride = seq_len // 2  # 50% overlap

    num_steps = len(trajectory.steps)
    if num_steps < seq_len:
        return {}

    windows = {
        "states": [],
        "advantages": [],
        "returns": [],
        "old_log_probs": [],
        "values": [],
        "valid_mask": [],
    }

    # Add action tensors
    for head in ["main_stick", "c_stick", "buttons", "shoulder"]:
        windows[f"actions_{head}"] = []

    # Create overlapping windows
    for i in range(0, num_steps - seq_len + 1, stride):
        window_steps = trajectory.steps[i : i + seq_len]

        # Stack states
        states = torch.stack([step.state for step in window_steps])  # [T, F]
        windows["states"].append(states)

        # Stack advantages and returns
        adv_slice = trajectory.advantages[i : i + seq_len]  # [T]
        ret_slice = trajectory.returns[i : i + seq_len]  # [T]
        windows["advantages"].append(adv_slice)
        windows["returns"].append(ret_slice)

        # Stack old log probs and values
        old_log_probs = torch.stack([step.log_prob for step in window_steps]).squeeze(
            -1
        )
        values = torch.stack([step.value for step in window_steps]).squeeze(-1)
        windows["old_log_probs"].append(old_log_probs)
        windows["values"].append(values)

        # Valid mask (all True for now)
        valid_mask = torch.ones(seq_len, dtype=torch.bool)
        windows["valid_mask"].append(valid_mask)

        # Stack actions
        for head in ["main_stick", "c_stick", "buttons", "shoulder"]:
            actions = torch.stack([step.action_taken[head] for step in window_steps])
            windows[f"actions_{head}"].append(actions)

    # Stack all windows
    if windows["states"]:
        for key in windows:
            windows[key] = torch.stack(windows[key]).to(device)

    return windows


def _combine_windows(
    windows_list: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Combine windows from multiple rollouts."""
    if not windows_list:
        return {}

    # Filter out empty windows
    windows_list = [w for w in windows_list if w]
    if not windows_list:
        return {}

    # Concatenate along batch dimension
    combined = {}
    for key in windows_list[0].keys():
        tensors = [w[key] for w in windows_list if key in w]
        if tensors:
            combined[key] = torch.cat(tensors, dim=0)

    return combined


def _ppo_training_loop(
    model: GPT,
    teacher_model: Optional[GPT],
    optimizer: torch.optim.Optimizer,
    windows: Dict[str, torch.Tensor],
    ppo_cfg,
    rl_cfg,
    colmap: ColumnMap,
    device: torch.device,
) -> TrainingMetrics:
    """Run PPO training loop on windows.

    Args:
        model: Model to train
        teacher_model: Optional teacher for distillation
        optimizer: Optimizer
        windows: Pre-built training windows
        ppo_cfg: PPO configuration
        rl_cfg: RL configuration
        colmap: Column mapping
        device: Training device

    Returns:
        Training metrics
    """
    if not windows or "states" not in windows:
        logger.warning("No windows to train on")
        return TrainingMetrics(
            avg_loss=0.0,
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            approx_kl=0.0,
            clipped_fraction=0.0,
            num_windows=0,
        )

    num_windows = windows["states"].shape[0]
    seq_len = windows["states"].shape[1]
    warmup_positions = ppo_cfg.warmup_frames

    logger.info(f"Training on {num_windows} windows, seq_len={seq_len}")

    all_losses = []
    all_approx_kl = []
    all_clipped_frac = []
    epoch_metrics = {}

    # Create profilers for detailed timing
    prof_forward = Profiler(burnin=0)
    prof_loss = Profiler(burnin=0)
    prof_backward = Profiler(burnin=0)
    prof_optimizer = Profiler(burnin=0)

    # Run PPO epochs
    for ppo_epoch in range(ppo_cfg.ppo_epochs):
        # Shuffle windows
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

            # Loss mask (skip warmup positions)
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
                logger.warning("NaN in model outputs, skipping batch")
                continue

            # Compute loss
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
                        value_coef=rl_cfg.value_loss_coef,
                        value_clip=ppo_cfg.value_clip,
                        loss_mask=loss_mask,
                    )

            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN loss, skipping batch")
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
                    logger.warning("NaN gradients, skipping step")
                    optimizer.zero_grad()
                    continue

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), ppo_cfg.max_grad_norm
                )

            # Optimizer step
            with prof_optimizer:
                optimizer.step()

            # Record metrics
            epoch_losses.append(loss.item())
            all_approx_kl.append(loss_metrics.get("approx_kl", 0.0))
            all_clipped_frac.append(loss_metrics.get("clipped_fraction", 0.0))

            # Save metrics from first minibatch
            if mb_idx == 0 and ppo_epoch == 0:
                epoch_metrics = loss_metrics

        # Check early stopping
        if epoch_losses:
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            all_losses.extend(epoch_losses)

            # Early stopping if too many samples clipped
            if all_clipped_frac:
                avg_clipped = sum(all_clipped_frac[-num_minibatches:]) / num_minibatches
                max_clipped = getattr(ppo_cfg, "max_clipped_fraction", 0.5)
                if avg_clipped > max_clipped:
                    logger.warning(
                        f"High clipped fraction: {avg_clipped:.2f} > {max_clipped:.2f}, "
                        f"stopping at epoch {ppo_epoch + 1}"
                    )
                    break

    # Compute final metrics
    avg_loss = sum(all_losses) / len(all_losses) if all_losses else 0.0
    approx_kl = sum(all_approx_kl) / len(all_approx_kl) if all_approx_kl else 0.0
    clipped_frac = (
        sum(all_clipped_frac) / len(all_clipped_frac) if all_clipped_frac else 0.0
    )

    return TrainingMetrics(
        avg_loss=avg_loss,
        policy_loss=epoch_metrics.get("policy_loss", 0.0),
        value_loss=epoch_metrics.get("value_loss", 0.0),
        entropy=epoch_metrics.get("entropy", 0.0),
        approx_kl=approx_kl,
        clipped_fraction=clipped_frac,
        num_windows=num_windows,
        # Add profiling stats (convert to ms)
        time_forward_ms=prof_forward.mean_time() * 1000,
        time_loss_ms=prof_loss.mean_time() * 1000,
        time_backward_ms=prof_backward.mean_time() * 1000,
        time_optimizer_ms=prof_optimizer.mean_time() * 1000,
    )
