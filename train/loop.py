"""Training loop orchestration helpers."""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Dict

import torch
from loguru import logger

from train.checkpoint import maybe_checkpoint_batch, maybe_checkpoint_epoch
from train.components import EpochContext, TrainingState
from train.logging import emit_logging, prepare_logging_bundle
from train.lr_schedule import _update_learning_rate
from train.profiling import print_profiling_results
from train.setup import initialize_training_components, print_config
from train.step import perform_backward_pass, perform_forward_pass
from train.wandb_utils import finish_wandb
from window_dataset import worker_init_fn


def _prepare_batch(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        "X": batch["X"].to(device, non_blocking=True),
        "Y": batch["Y"].to(device, non_blocking=True),
    }


def _compute_training_progress(
    epoch: int,
    iteration: int,
    total_batches: int,
    total_epochs: int,
    warmup_epochs: int,
    cooldown_epochs: int,
) -> float:
    warmup_epochs = max(warmup_epochs, 0)
    cooldown_epochs = max(cooldown_epochs, 0)
    if total_epochs <= warmup_epochs + cooldown_epochs:
        return 1.0 if epoch >= warmup_epochs else 0.0

    if epoch < warmup_epochs:
        return 0.0
    if epoch >= total_epochs - cooldown_epochs:
        return 1.0

    effective_epochs = total_epochs - warmup_epochs - cooldown_epochs
    epoch_offset = epoch - warmup_epochs
    if total_batches <= 0:
        progress = (epoch_offset + 1) / float(effective_epochs)
    else:
        batch_fraction = (iteration + 1) / float(total_batches)
        progress = (epoch_offset + batch_fraction) / float(effective_epochs)
    return float(min(max(progress, 0.0), 1.0))


def _should_log(current_iter: int) -> bool:
    return current_iter % 100 == 0


def _update_epoch_statistics(epoch_ctx: EpochContext, forward_result) -> None:
    epoch_ctx.add_loss(forward_result.loss)
    batch_size, sequence_length, _ = forward_result.pred["main_stick"].shape
    epoch_ctx.frames_since_last_log += float(batch_size * sequence_length)


def run_epoch(state: TrainingState, epoch: int) -> TrainingState:
    components = state.components
    config = components.config
    dataset = components.dataset
    components.sampler.set_epoch(epoch)
    components.model.train()

    try:
        total_batches = len(components.loader)
    except TypeError:
        total_batches = 0
    total_batches = max(int(total_batches), 1)

    epoch_ctx = EpochContext()
    epoch_ctx.last_log_time = time.time()
    if epoch == state.resume_epoch:
        epoch_ctx.applied_skip = state.resume_iter
        epoch_ctx.skip_remaining = state.resume_iter
        if state.resume_iter:
            try:
                components.sampler.set_start_offset(state.resume_iter)
                logger.info(
                    f"Resuming epoch {epoch + 1}: skipping first {state.resume_iter} batches via sampler offset."
                )
                epoch_ctx.skip_remaining = 0
            except Exception as exc:
                logger.warning(
                    f"Sampler offset failed ({exc}); falling back to loading batches for skip."
                )
                logger.info(
                    f"Resuming epoch {epoch + 1}: skipping first {state.resume_iter} batches by consuming them (may take time)."
                )

    iteration = 0

    # Main training loop
    for batch in components.loader:
        if epoch_ctx.skip_remaining:
            epoch_ctx.skip_remaining -= 1
            iteration += 1
            continue

        # Determine if we should profile this step
        should_profile = (
            components.profiling_enabled
            and components.profiling_step_count < 1000
        )

        # Get profiler references (or nullcontext for zero overhead when disabled)
        prof = components.profilers
        ctx = lambda name: prof[name] if should_profile else nullcontext()

        with ctx("total_step"):
            # Data preparation
            with ctx("data_prep"):
                batch_tensors = _prepare_batch(batch, components.device)

            # Progress computation
            with ctx("progress_calc"):
                progress = _compute_training_progress(
                    epoch,
                    iteration,
                    total_batches,
                    config.train.epochs,
                    config.train.schedule_warmup_epochs,
                    config.train.schedule_cooldown_epochs,
                )

            # Determine if we should log this iteration (needed before forward pass)
            current_iter = epoch_ctx.applied_skip + epoch_ctx.iters_processed
            log_this_iter = _should_log(current_iter)

            # Forward pass
            with ctx("forward"):
                forward_result = perform_forward_pass(
                    components,
                    batch_tensors,
                    progress=progress,
                    in_warmup=epoch < config.train.schedule_warmup_epochs,
                    collect_diagnostics=log_this_iter,
                )

            # Learning rate update
            with ctx("lr_update"):
                lr = _update_learning_rate(components, state.global_step)

            # Backward pass
            with ctx("backward"):
                grad_stats = perform_backward_pass(
                    components,
                    forward_result.loss,
                    collect_grad_stats=components.logger.enabled
                    and log_this_iter,
                )

            epoch_ctx.iters_processed += 1

            # Statistics update
            with ctx("stats_update"):
                _update_epoch_statistics(epoch_ctx, forward_result)
                state.global_step += 1

                # Update variance trackers without forcing a sync every step
                components.loss_accumulator.add(forward_result.loss.detach())
                if grad_stats and "total_norm" in grad_stats:
                    components.gradient_variance_tracker.add(
                        grad_stats["total_norm"]
                    )

            # Checkpointing
            completed_batches = (
                epoch_ctx.applied_skip + epoch_ctx.iters_processed
            )
            with ctx("checkpoint"):
                maybe_checkpoint_batch(
                    components,
                    epoch,
                    iteration,
                    completed_batches,
                    state.global_step,
                )

            # Logging
            if log_this_iter:
                with ctx("logging"):
                    # Flush accumulated losses once per log
                    for (
                        loss_value
                    ) in components.loss_accumulator.get_and_reset():
                        components.loss_variance_tracker.add(loss_value)

                    now = time.time()
                    dt = max(1e-9, now - epoch_ctx.last_log_time)
                    frames_per_s = epoch_ctx.frames_since_last_log / dt
                    avg_loss_running = epoch_ctx.get_avg_loss()
                    bundle = prepare_logging_bundle(
                        components=components,
                        forward_result=forward_result,
                        epoch=epoch,
                        completed_batches=completed_batches,
                        total_batches=total_batches,
                        lr=lr,
                        frames_per_s=frames_per_s,
                        avg_loss_running=avg_loss_running,
                        grad_stats=grad_stats,
                        global_step=state.global_step,
                        epoch_ctx=epoch_ctx,
                    )
                    emit_logging(
                        components=components,
                        bundle=bundle,
                        grad_stats=grad_stats,
                        global_step=state.global_step,
                        epoch_ctx=epoch_ctx,
                    )
                    epoch_ctx.last_log_time = now
                    epoch_ctx.frames_since_last_log = 0.0

        # Increment profiling step count and check if profiling just completed
        if should_profile:
            components.profiling_step_count += 1
            if components.profiling_step_count >= 1000:
                logger.info("PROFILING COMPLETE - 1000 steps profiled")
                print_profiling_results(components.profilers)

        iteration += 1

    # Flush any remaining losses at epoch end
    for loss_value in components.loss_accumulator.get_and_reset():
        components.loss_variance_tracker.add(loss_value)

    if epoch == state.resume_epoch:
        state.resume_iter = 0
        state.resume_epoch = -1

    if epoch_ctx.iters_processed:
        avg_epoch_loss = epoch_ctx.get_avg_loss()
        logger.info(
            f"[epoch {epoch + 1}] avg_loss {avg_epoch_loss:.4f} ({epoch_ctx.iters_processed} iters)"
        )

    maybe_checkpoint_epoch(components, epoch, state.global_step)

    return state


def train_loop(
    model,
    loader,
    ds,
    sampler,
    *,
    debug: bool = False,
) -> None:
    components, start_epoch, global_step, start_iter = initialize_training_components(
        model, loader, ds, sampler, debug
    )
    config = components.config

    def _total_batches_for_epoch(epoch: int) -> int:
        try:
            return max(int(len(components.loader)), 1)
        except Exception:
            return 1

    # If resume_iter is beyond the current epoch length, advance epochs accordingly
    while start_epoch < config.train.epochs:
        tb = _total_batches_for_epoch(start_epoch)
        if start_iter < tb:
            break
        logger.warning(
            f"[resume] resume_iter {start_iter} exceeds total_batches {tb} for epoch {start_epoch + 1}; "
            "advancing to next epoch."
        )
        start_iter -= tb
        start_epoch += 1

    if start_epoch >= config.train.epochs:
        logger.info(
            f"All requested epochs ({config.train.epochs}) already completed (start_epoch={start_epoch}); exiting."
        )
        if not debug:
            finish_wandb()
            components.local_logger.close()
        return

    state = TrainingState(
        components=components,
        global_step=global_step,
        resume_epoch=start_epoch,
        resume_iter=start_iter,
    )

    if start_epoch >= config.train.epochs:
        logger.info(
            f"All requested epochs ({config.train.epochs}) already completed (start_epoch={start_epoch}); exiting."
        )
        if not components.debug:
            finish_wandb()
            components.local_logger.close()
        return

    print_config(config)

    for epoch in range(start_epoch, config.train.epochs):
        state = run_epoch(state, epoch)

    if not components.debug:
        finish_wandb()
        components.local_logger.close()
