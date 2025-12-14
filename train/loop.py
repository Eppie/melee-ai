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
    chunked = (
        dataset is not None
        and dataset._total_chunks is not None
        and dataset._total_chunks > 1
    )
    components.sampler.set_epoch(epoch)
    components.model.train()

    if chunked:
        stride = config.train.stride
        batch_size = config.train.batch_size
        total_batches = dataset.total_batches_for_epoch(
            stride=stride, batch_size=batch_size, epoch=epoch
        )
    else:
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
        if (
            not chunked
            and state.resume_iter
        ):
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
        elif state.resume_iter and not chunked:
            logger.info(
                f"Resuming epoch {epoch + 1}: skipping first {state.resume_iter} batches by consuming them (may take time)."
            )

    iteration = 0

    def _build_loader_for_chunk():
        mp_ctx = None
        start_method = config.train.worker_start_method
        if config.train.num_workers and config.train.num_workers > 0 and start_method:
            try:
                mp_ctx = torch.multiprocessing.get_context(start_method)
            except RuntimeError as exc:
                logger.warning(
                    f"[dataloader] Requested start method '{start_method}' unavailable ({exc}); using default."
                )
                mp_ctx = None

        pin_memory = config.train.pin_memory
        num_workers = config.train.num_workers
        prefetch_factor = None
        if num_workers and num_workers > 0:
            prefetch_factor = config.train.prefetch_factor
            max_prefetch_mb = config.train.max_loader_prefetch_mb
            if max_prefetch_mb:
                batch_bytes = max(
                    1, dataset.estimate_batch_bytes(config.train.batch_size)
                )
                max_prefetch_bytes = max_prefetch_mb * 1024 * 1024
                total_batches_budget = max_prefetch_bytes // batch_bytes
                budget_saturated = False
                if total_batches_budget == 0:
                    total_batches_budget = 1
                    budget_saturated = True
                allowed_per_worker = total_batches_budget // num_workers
                if allowed_per_worker == 0:
                    allowed_per_worker = 1
                    budget_saturated = True
                if allowed_per_worker < prefetch_factor:
                    approx_batch_mb = batch_bytes / (1024**2)
                    logger.warning(
                        "[dataloader] Reducing prefetch_factor from "
                        f"{prefetch_factor} to {allowed_per_worker} to honor "
                        f"{max_prefetch_mb} MiB prefetch budget (batch ≈ "
                        f"{approx_batch_mb:.2f} MiB)."
                    )
                    if budget_saturated:
                        logger.warning(
                            "[dataloader] Consider lowering train.num_workers or "
                            "batch_size, or increase train.max_loader_prefetch_mb "
                            "if you need more throughput."
                        )
                    prefetch_factor = allowed_per_worker

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            sampler=components.sampler,
            num_workers=config.train.num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            persistent_workers=(config.train.persistent_workers and not chunked),
            multiprocessing_context=mp_ctx,
            prefetch_factor=prefetch_factor,
        )

    if not chunked:
        loaders = [components.loader]
    else:
        loaders = None  # not used; chunk loop below

    if chunked:
        stride = config.train.stride
        batch_size = config.train.batch_size
        epoch_mod = epoch % max(1, stride)
        skip_batches = epoch_ctx.skip_remaining

        # Multi-chunk overlap: determine number of overlapping chunks
        num_overlapping = dataset._num_overlapping_chunks
        total_chunks = int(dataset._total_chunks)

        # Calculate number of iterations needed to cover all chunks with overlap
        # For overlapping chunks, we advance by 1 chunk each iteration
        # Example: With 10 chunks and 2 overlapping: [0,1], [1,2], [2,3], ..., [9,0]
        if num_overlapping > 1:
            # We need total_chunks iterations to ensure all chunks are visited
            num_iterations = total_chunks
        else:
            # Single chunk mode: iterate through each chunk
            num_iterations = total_chunks

        for iteration_idx in range(num_iterations):
            # Calculate which chunks to load for this iteration
            if num_overlapping > 1:
                # Multi-chunk overlap: load overlapping set
                chunk_indices = [
                    (iteration_idx + offset) % total_chunks
                    for offset in range(num_overlapping)
                ]
            else:
                # Single chunk mode: load one chunk at a time
                chunk_indices = [iteration_idx % total_chunks]

            # Count batches for this iteration (across all chunks in the set)
            iteration_batches = 0
            for chunk_idx in chunk_indices:
                iteration_batches += dataset.count_chunk_batches(
                    chunk_idx,
                    stride=stride,
                    batch_size=batch_size,
                    epoch_mod=epoch_mod,
                )

            # Skip if resuming from checkpoint
            if skip_batches >= iteration_batches:
                skip_batches -= iteration_batches
                continue

            try:
                # Load chunks (supports multi-chunk overlap) with timing
                chunk_load_start = time.time()
                dataset.load_chunks(chunk_indices)
                chunk_load_duration = time.time() - chunk_load_start

                # Record chunk load time
                components.dataloader_metrics.record_chunk_load(
                    chunk_load_duration, iteration=iteration_idx
                )

                # Update sampler with active episodes
                components.sampler.set_active_episodes(
                    dataset._active_episode_indices.tolist()
                )
                components.sampler.set_epoch(epoch)
                if skip_batches > 0:
                    components.sampler.set_start_offset(skip_batches * batch_size)
                    skip_batches = 0
                epoch_ctx.skip_remaining = 0
            except Exception as exc:
                logger.error(
                    f"[dataloader] Failed to load chunks {chunk_indices} ({exc}); skipping."
                )
                continue

            loader = _build_loader_for_chunk()

            for batch in loader:
                # Record batch start for idle time tracking
                components.dataloader_metrics.record_batch_start()

                # Determine if we should profile this step
                should_profile = (
                    components.profiling_enabled
                    and components.profiling_step_count < 1000
                )

                # Get profiler references (or nullcontext for zero overhead when disabled)
                prof = components.profilers
                ctx = lambda name: prof[name] if should_profile else nullcontext()

                with ctx("total_step"):
                    # Data preparation with timing
                    with ctx("data_prep"):
                        batch_prep_start = time.time()
                        batch_tensors = _prepare_batch(batch, components.device)
                        batch_prep_duration = time.time() - batch_prep_start
                        components.dataloader_metrics.record_batch_prep(
                            batch_prep_duration
                        )

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

                    # Forward pass
                    with ctx("forward"):
                        forward_result = perform_forward_pass(
                            components,
                            batch_tensors,
                            progress=progress,
                            in_warmup=epoch < config.train.schedule_warmup_epochs,
                        )

                    current_iter = epoch_ctx.applied_skip + epoch_ctx.iters_processed
                    log_this_iter = _should_log(current_iter)

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
                            for loss_value in components.loss_accumulator.get_and_reset():
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

                # Record batch end for timing metrics
                components.dataloader_metrics.record_batch_end()

                iteration += 1

    # Non-chunked path
    if not chunked:
        for loader in loaders:
            for batch in loader:
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

                    # Forward pass
                    with ctx("forward"):
                        forward_result = perform_forward_pass(
                            components,
                            batch_tensors,
                            progress=progress,
                            in_warmup=epoch < config.train.schedule_warmup_epochs,
                        )

                    current_iter = epoch_ctx.applied_skip + epoch_ctx.iters_processed
                    log_this_iter = _should_log(current_iter)

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
                            for loss_value in components.loss_accumulator.get_and_reset():
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

    # Flush any remaining losses at epoch end (covers both chunked and non-chunked paths)
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
        dataset = components.dataset
        chunked = (
            dataset is not None
            and dataset._total_chunks is not None
            and dataset._total_chunks > 1
        )
        if chunked:
            return max(
                1,
                dataset.total_batches_for_epoch(
                    stride=config.train.stride,
                    batch_size=config.train.batch_size,
                    epoch=epoch,
                ),
            )
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
        return

    print_config(config)

    for epoch in range(start_epoch, config.train.epochs):
        state = run_epoch(state, epoch)

    if not components.debug:
        finish_wandb()
