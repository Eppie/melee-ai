"""Training loop orchestration helpers."""

from __future__ import annotations

import time
from typing import Dict

import torch

from train.checkpoint import maybe_checkpoint_batch, maybe_checkpoint_epoch
from train.components import EpochContext, TrainingState
from train.logging import emit_logging, prepare_logging_bundle
from train.lr_schedule import _update_learning_rate
from train.setup import initialize_training_components, print_config
from train.step import perform_backward_pass, perform_forward_pass
from train.wandb_utils import finish_wandb


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

    if hasattr(components.sampler, "set_epoch"):
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
        if state.resume_iter and hasattr(components.sampler, "set_start_offset"):
            try:
                components.sampler.set_start_offset(state.resume_iter)
                print(
                    f"Resuming epoch {epoch + 1}: skipping first {state.resume_iter} batches via sampler offset."
                )
                epoch_ctx.skip_remaining = 0
            except Exception as exc:
                print(
                    f"Sampler offset failed ({exc}); falling back to loading batches for skip."
                )
        elif state.resume_iter:
            print(
                f"Resuming epoch {epoch + 1}: skipping first {state.resume_iter} batches by consuming them (may take time)."
            )

    for iteration, batch in enumerate(components.loader):
        if epoch_ctx.skip_remaining:
            epoch_ctx.skip_remaining -= 1
            continue

        batch_tensors = _prepare_batch(batch, components.device)
        progress = _compute_training_progress(
            epoch,
            iteration,
            total_batches,
            config.train.epochs,
            config.train.schedule_warmup_epochs,
            config.train.schedule_cooldown_epochs,
        )
        forward_result = perform_forward_pass(
            components,
            batch_tensors,
            progress=progress,
            in_warmup=epoch < config.train.schedule_warmup_epochs,
        )

        current_iter = epoch_ctx.applied_skip + epoch_ctx.iters_processed
        log_this_iter = _should_log(current_iter)
        lr = _update_learning_rate(components, state.global_step)
        grad_stats = perform_backward_pass(
            components,
            forward_result.loss,
            collect_grad_stats=components.logger.enabled and log_this_iter,
        )

        epoch_ctx.iters_processed += 1
        _update_epoch_statistics(epoch_ctx, forward_result)
        state.global_step += 1

        # Update variance trackers
        loss_value = float(forward_result.loss.detach().cpu().item())
        components.loss_variance_tracker.add(loss_value)
        if grad_stats and "total_norm" in grad_stats:
            components.gradient_variance_tracker.add(grad_stats["total_norm"])

        completed_batches = epoch_ctx.applied_skip + epoch_ctx.iters_processed
        maybe_checkpoint_batch(
            components,
            epoch,
            iteration,
            completed_batches,
            state.global_step,
        )

        if log_this_iter:
            now = time.time()
            dt = max(1e-9, now - epoch_ctx.last_log_time)
            frames_per_s = epoch_ctx.frames_since_last_log / dt
            avg_loss_running = epoch_ctx.get_avg_loss()
            bundle = prepare_logging_bundle(
                components=components,
                forward_result=forward_result,
                epoch=epoch,
                completed_batches=completed_batches,
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

    if epoch == state.resume_epoch:
        state.resume_iter = 0
        state.resume_epoch = -1

    if epoch_ctx.iters_processed:
        avg_epoch_loss = epoch_ctx.get_avg_loss()
        print(
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

    state = TrainingState(
        components=components,
        global_step=global_step,
        resume_epoch=start_epoch,
        resume_iter=start_iter,
    )

    if start_epoch >= config.train.epochs:
        print(
            f"All requested epochs ({config.train.epochs}) already completed (start_epoch={start_epoch}); exiting."
        )
        if not components.debug:
            finish_wandb()
        return

    print_config(config)

    for epoch in range(start_epoch, config.train.epochs):
        state = run_epoch(state, epoch)

    if not components.debug:
        finish_wandb()
