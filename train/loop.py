"""Training loop orchestration helpers."""

from __future__ import annotations

import time
from contextlib import nullcontext
import warnings
from typing import Dict

import torch
from torch.profiler import ProfilerActivity, profile

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
    return current_iter % 5 == 0


def _update_epoch_statistics(epoch_ctx: EpochContext, forward_result) -> None:
    epoch_ctx.epoch_loss += forward_result.loss.item()
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

    loader_iter = iter(components.loader)
    iteration = 0
    while True:
        iter_start = time.perf_counter()
        try:
            batch = next(loader_iter)
        except StopIteration:
            break
        timing_load = time.perf_counter()

        if epoch_ctx.skip_remaining:
            epoch_ctx.skip_remaining -= 1
            iteration += 1
            continue

        batch_tensors = _prepare_batch(batch, components.device)
        timing_to_device = time.perf_counter()
        progress = _compute_training_progress(
            epoch,
            iteration,
            total_batches,
            config.train.epochs,
            config.train.schedule_warmup_epochs,
            config.train.schedule_cooldown_epochs,
        )
        current_iter = epoch_ctx.applied_skip + epoch_ctx.iters_processed
        log_this_iter = _should_log(current_iter)

        profiler_summary = None
        profiler_shapes = None
        profiler_device = None
        profiler_device_note = None
        profiler_stacks = None
        profiler_stack_path = None
        profiler_stack_note = None

        profile_activities = [ProfilerActivity.CPU]
        device_metric = None
        if components.device.type == "cuda":
            profile_activities.append(ProfilerActivity.CUDA)
            device_metric = "self_cuda_time_total"
        elif components.device.type == "mps" and hasattr(
            ProfilerActivity, "PrivateUse1"
        ):
            profile_activities.append(ProfilerActivity.PrivateUse1)
            device_metric = "self_privateuse1_time_total"

        prof_ctx = (
            profile(
                activities=profile_activities,
                record_shapes=True,
                with_stack=True,
            )
            if log_this_iter
            else nullcontext()
        )

        with warnings.catch_warnings():
            # MPS profiling may emit "privateuseone is not a valid device option" on some builds.
            warnings.filterwarnings(
                "ignore", message="The privateuseone is not a valid device option."
            )
            with prof_ctx as prof:
                forward_result = perform_forward_pass(
                    components,
                    batch_tensors,
                    progress=progress,
                    in_warmup=epoch < config.train.schedule_warmup_epochs,
                )
                timing_forward = time.perf_counter()

                lr = _update_learning_rate(components, state.global_step)
                grad_stats, backward_timing = perform_backward_pass(
                    components,
                    forward_result.loss,
                    collect_grad_stats=components.logger.enabled and log_this_iter,
                )
                timing_backward = time.perf_counter()

        if log_this_iter and hasattr(prof, "key_averages"):
            events_by_stack = prof.key_averages(group_by_stack_n=5)
            profiler_summary = events_by_stack.table(
                sort_by="self_cpu_time_total", row_limit=15
            )
            events_by_shape = prof.key_averages(group_by_input_shape=True)
            profiler_shapes = events_by_shape.table(
                sort_by="self_cpu_time_total", row_limit=15
            )
            def _format_device_table(
                events, metric_candidates, row_limit=15
            ) -> tuple[str | None, str | None]:
                rows = []
                for evt in events:
                    dev_val = 0.0
                    metric_used = None
                    for metric in metric_candidates:
                        metric_used = metric
                        dev_val = float(getattr(evt, metric, 0.0) or 0.0)
                        if dev_val > 0.0:
                            break
                    if dev_val <= 0.0:
                        continue
                    # Try to fetch the corresponding total time.
                    total_candidates = [
                        metric_used.replace("self_", ""),
                        metric_used.replace("self_", "") + "_total",
                        "device_time_total",
                        "cuda_time_total",
                        "privateuse1_time_total",
                        "xpu_time_total",
                    ]
                    dev_total_val = 0.0
                    for cand in total_candidates:
                        dev_total_val = float(getattr(evt, cand, 0.0) or 0.0)
                        if dev_total_val > 0.0:
                            break
                    cpu_self = float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0)
                    cpu_total = float(getattr(evt, "cpu_time_total", 0.0) or 0.0)
                    rows.append(
                        (
                            dev_val / 1e6,
                            dev_total_val / 1e6,
                            cpu_self / 1e6,
                            cpu_total / 1e6,
                            evt.count,
                            evt.name,
                        )
                    )
                if not rows:
                    note = "no device timings captured (backend may not support device profiling)"
                    return None, note
                rows.sort(key=lambda r: r[0], reverse=True)
                header = (
                    f"{'Name':<45} {'self_dev_ms':>12} {'dev_ms':>12} "
                    f"{'self_cpu_ms':>12} {'cpu_ms':>12} {'#':>8}"
                )
                lines = [header]
                for entry in rows[:row_limit]:
                    dev_self_ms, dev_total_ms, cpu_self_ms, cpu_total_ms, calls, name = (
                        entry
                    )
                    lines.append(
                        f"{name:<45} {dev_self_ms:12.3f} {dev_total_ms:12.3f} "
                        f"{cpu_self_ms:12.3f} {cpu_total_ms:12.3f} {int(calls):8d}"
                    )
                return "\n".join(lines), None

            if device_metric:
                metric_candidates = [
                    device_metric,
                    "self_device_time_total",
                    "self_cuda_time_total",
                ]
                profiler_device, profiler_device_note = _format_device_table(
                    events_by_stack, metric_candidates
                )
            # Persist raw stacks (if available) so they can be inspected even if the log is truncated.
            try:
                profile_dir = components.out_dir / "profiler"
                profile_dir.mkdir(parents=True, exist_ok=True)
                stack_file = profile_dir / f"epoch{epoch + 1:03d}_iter{epoch_ctx.applied_skip + epoch_ctx.iters_processed:06d}_stacks.txt"
                prof.export_stacks(str(stack_file), "self_cpu_time_total")
                if stack_file.exists() and stack_file.stat().st_size > 0:
                    profiler_stack_path = stack_file
                elif stack_file.exists():
                    profiler_stack_note = "stack export was empty (likely unsupported by this build)"
            except Exception as exc:
                profiler_stack_note = f"error saving stacks: {exc}"

            raw_events = sorted(
                prof.events(), key=lambda e: e.self_cpu_time_total, reverse=True
            )
            stack_lines = []
            for evt in raw_events:
                if not getattr(evt, "stack", None):
                    continue
                shape_info = (
                    f" shapes={evt.input_shapes}" if evt.input_shapes else ""
                )
                stack_lines.append(
                    f"{evt.name}{shape_info} self_cpu={evt.self_cpu_time_total/1e6:.2f}ms:\n"
                    + "\n".join(f"    {frame}" for frame in evt.stack)
                )
                if len(stack_lines) >= 5:
                    break
            profiler_stacks = "\n".join(stack_lines) if stack_lines else None

        epoch_ctx.iters_processed += 1
        _update_epoch_statistics(epoch_ctx, forward_result)
        state.global_step += 1
        iteration += 1

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
            avg_loss_running = epoch_ctx.epoch_loss / max(1, epoch_ctx.iters_processed)
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
                timing_ms={
                    "load_ms": 1000.0 * (timing_load - iter_start),
                    "to_device_ms": 1000.0 * (timing_to_device - timing_load),
                    "forward_ms": 1000.0 * (timing_forward - timing_to_device),
                    "backward_ms": 1000.0 * (timing_backward - timing_forward),
                },
                forward_timing_ms=forward_result.timing_ms,
                backward_timing_ms=backward_timing,
                profiler_summary=profiler_summary,
                profiler_shapes=profiler_shapes,
                profiler_device=profiler_device,
                profiler_device_note=profiler_device_note,
                profiler_stacks=profiler_stacks,
                profiler_stack_path=str(profiler_stack_path)
                if profiler_stack_path is not None
                else None,
                profiler_stack_note=profiler_stack_note,
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
        avg_epoch_loss = epoch_ctx.epoch_loss / max(1, epoch_ctx.iters_processed)
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
