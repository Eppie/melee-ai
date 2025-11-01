#!/usr/bin/env python3
"""Utility script for orchestrating instrumented training runs.

Features
--------
- Execute single or multiple training runs with configuration overrides.
- Estimate FLOP usage for each batch/epoch using a lightweight analytical model.
- Stop early when a target loss is met.
- Search over hyperparameter grids to find fast/cheap configurations.
- Produce structured JSON/CLI reports summarising each run.

The implementation reuses the core model, dataset, and loss construction from
``train.py`` but layers experiment-management and bookkeeping logic around it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from column_map import ColumnMap
from config import Config, get_config, init_config, reset_config_for_tests
from loss import _compute_ce_weights, _compute_pos_weights
from model.nano_gpt import GPT
# Train module utilities
from train.batch_utils import build_model_inputs, quantize_controller_targets
from train.metrics import MetricsAccumulator
from utils import _resolve_device
from window_dataset import make_dataloader

TRAINING_FLOP_MULTIPLIER = 3.0  # forward + backward (approximation)
TOKEN_RATE_EMA_DECAY = 0.9

_FLOP_UNITS: Tuple[Tuple[float, str], ...] = (
    (1e15, "PFLOP"),
    (1e12, "TFLOP"),
    (1e9, "GFLOP"),
    (1e6, "MFLOP"),
    (1e3, "KFLOP"),
)

_MAIN_METRIC_KEYS: Tuple[str, ...] = (
    "acc_main",
    "acc_main_rand",
    "acc_main_maj",
    "acc_main_rep",
)
_C_METRIC_KEYS: Tuple[str, ...] = (
    "acc_c",
    "acc_c_rand",
    "acc_c_maj",
    "acc_c_rep",
)
_BUTTON_METRIC_KEYS: Tuple[str, ...] = (
    "btn_em",
    "btn_prec_micro",
    "btn_rec_micro",
    "btn_f1_micro",
    "btn_f1_macro",
    "btn_em_rand",
    "btn_f1_micro_rand",
    "btn_f1_macro_rand",
    "btn_em_maj",
    "btn_f1_micro_maj",
    "btn_f1_macro_maj",
    "btn_em_rep",
    "btn_f1_micro_rep",
    "btn_f1_macro_rep",
)
_SHOULDER_METRIC_KEYS: Tuple[str, ...] = (
    "acc_shoulder",
    "acc_shoulder_rand",
    "acc_shoulder_maj",
)
_RUNNING_METRIC_KEYS = (
    set(_MAIN_METRIC_KEYS)
    | set(_C_METRIC_KEYS)
    | set(_BUTTON_METRIC_KEYS)
    | set(_SHOULDER_METRIC_KEYS)
)

_LOSS_METRIC_KEYS: Tuple[str, ...] = (
    "loss",
    "loss_total",
    "loss_main",
    "loss_c",
    "loss_buttons",
    "loss_shoulder",
)
_OTHER_DIRECT_METRIC_KEYS: Tuple[str, ...] = (
    "lr",
    "learning_rate",
    "global_step",
    "tokens_per_second",
    "flops_per_second",
)


@dataclass(frozen=True)
class StoppingObjective:
    metric: str
    operator: str
    threshold: float
    raw: str

    def is_satisfied(
        self, metrics: Mapping[str, float], *, tol: float = 1e-6
    ) -> Tuple[bool, float]:
        if self.metric not in metrics:
            raise KeyError(
                f"Objective metric '{self.metric}' not available; available metrics: {sorted(metrics.keys())}"
            )
        value = float(metrics[self.metric])
        op = self.operator
        if op == ">=":
            satisfied = value >= self.threshold
        elif op == "<=":
            satisfied = value <= self.threshold
        elif op == ">":
            satisfied = value > self.threshold
        elif op == "<":
            satisfied = value < self.threshold
        else:  # op == "=="
            satisfied = abs(value - self.threshold) <= tol
        return satisfied, value


@dataclass
class TrainingRunResult:
    run_id: str
    overrides: Dict[str, str]
    target_loss: Optional[float]
    target_objectives: List[str]
    steps: int
    epochs_completed: int
    final_loss: float
    reached_target_loss: bool
    reached_target_objectives: bool
    interrupted: bool
    stopped_due_to_cap: bool
    stopped_due_to_time_limit: bool
    time_seconds: float
    estimated_forward_flops: float
    estimated_training_flops: float
    average_loss: Optional[float]
    min_loss: Optional[float]
    max_loss: Optional[float]
    per_step_losses: List[float]
    total_tokens: int
    tokens_per_second: Optional[float]
    stop_reason: str
    parameter_count: int
    config_snapshot: Dict[str, Any]
    metrics_summary: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["config_snapshot"] = self.config_snapshot
        return data


def append_result_jsonl(path: Path, result: TrainingRunResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(result.to_dict())
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.write("\n")


def parse_key_value(text: str) -> Tuple[str, str]:
    if "=" not in text:
        raise ValueError(f"Expected KEY=VALUE, got '{text}'.")
    key, value = text.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key:
        raise ValueError(f"Empty key in override '{text}'.")
    return key, value


_OBJECTIVE_OPERATORS: Tuple[str, ...] = (">=", "<=", ">", "<", "==")


def parse_objective(expr: str) -> StoppingObjective:
    text = expr.strip()
    for op in _OBJECTIVE_OPERATORS:
        if op in text:
            metric, threshold = text.split(op, 1)
            metric = metric.strip()
            threshold = threshold.strip()
            if not metric:
                raise ValueError(
                    f"Objective '{expr}' missing metric name before '{op}'."
                )
            try:
                value = float(threshold)
            except ValueError as exc:
                raise ValueError(
                    f"Objective '{expr}' has invalid threshold '{threshold}'."
                ) from exc
            return StoppingObjective(
                metric=metric, operator=op, threshold=value, raw=text
            )
    raise ValueError(
        f"Objective '{expr}' must contain a comparison operator (one of {' '.join(_OBJECTIVE_OPERATORS)})."
    )


def format_flops(value: float, *, per_second: bool = False) -> str:
    abs_value = abs(value)
    for scale, name in _FLOP_UNITS:
        if abs_value >= scale:
            formatted = value / scale
            suffix = f"{name}{'s' if not per_second else ''}"
            return f"{formatted:.2f} {suffix}{'/s' if per_second else ''}"
    suffix = "FLOPs"
    return f"{value:.0f} {suffix}{'/s' if per_second else ''}"


def format_overrides(overrides: Mapping[str, str]) -> str:
    if not overrides:
        return "(none)"
    items = sorted(overrides.items())
    return ", ".join(f"{k}={v}" for k, v in items)


def parse_search_items(items: Sequence[str]) -> Dict[str, List[str]]:
    search_space: Dict[str, List[str]] = {}
    for raw in items:
        key, value = parse_key_value(raw)
        options = [v.strip() for v in value.split(",") if v.strip()]
        if not options:
            raise ValueError(f"No values provided for search key '{key}'.")
        search_space[key] = options
    return search_space


def expand_search_space(space: Mapping[str, Sequence[str]]) -> Iterator[Dict[str, str]]:
    if not space:
        yield {}
        return
    keys = list(space.keys())
    value_lists = [list(space[k]) for k in keys]
    for combo in itertools.product(*value_lists):
        yield {k: v for k, v in zip(keys, combo)}


def _swiglu_flops(
    input_dim: int, inner_dim: int, output_dim: int, tokens: int
) -> float:
    if tokens == 0:
        return 0.0
    # swiglu = 2 linear projections + elementwise silu + mul + output linear
    linear_w1 = 2.0 * tokens * input_dim * inner_dim
    linear_v1 = 2.0 * tokens * input_dim * inner_dim
    linear_w2 = 2.0 * tokens * inner_dim * output_dim
    activations = 4.0 * tokens * inner_dim
    return linear_w1 + linear_v1 + linear_w2 + activations


def _activation_ffn_flops(
    activation: str,
    input_dim: int,
    inner_dim: int,
    output_dim: int,
    tokens: int,
) -> float:
    if tokens <= 0:
        return 0.0
    act = activation.lower()
    if act == "swiglu":
        return _swiglu_flops(input_dim, inner_dim, output_dim, tokens)
    if act == "geglu":
        gelu_cost = 6.0 * tokens * inner_dim
        return (
            2.0 * tokens * input_dim * inner_dim * 2
            + 2.0 * tokens * inner_dim * output_dim
            + gelu_cost
        )
    if act == "gelu":
        gelu_cost = 6.0 * tokens * inner_dim
        return (
            2.0 * tokens * input_dim * inner_dim
            + gelu_cost
            + 2.0 * tokens * inner_dim * output_dim
        )
    return 2.0 * tokens * input_dim * inner_dim + 2.0 * tokens * inner_dim * output_dim


def estimate_forward_flops(cfg: Config, batch_size: int, seq_len: int) -> float:
    if batch_size == 0 or seq_len == 0:
        return 0.0
    tokens = batch_size * seq_len
    model_cfg = cfg.model
    D = model_cfg.n_embd
    H = model_cfg.n_head
    d_head = D // H

    flops = 0.0

    # Input projection G -> D
    flops += 2.0 * tokens * model_cfg.input_size * D

    # Transformer blocks
    attn_type = model_cfg.attention_type.lower()
    if attn_type == "mqa":
        kv_heads = 1
    elif attn_type == "gqa":
        kv_heads = model_cfg.n_kv_head or max(1, H // 4)
    else:
        kv_heads = H
    groups = max(1, H // kv_heads)

    # Q projection always full size
    q_proj = 2.0 * tokens * D * D
    k_proj = 2.0 * tokens * D * (kv_heads * d_head)
    v_proj = 2.0 * tokens * D * (kv_heads * d_head)
    o_proj = 2.0 * tokens * D * D

    attn_scores = 2.0 * batch_size * H * seq_len * seq_len * d_head
    attn_values = 2.0 * batch_size * H * seq_len * seq_len * d_head
    attn_softmax = 4.0 * batch_size * H * seq_len * seq_len

    attn_total = (
        q_proj + k_proj + v_proj + o_proj + attn_scores + attn_values + attn_softmax
    )

    # Positional encoding overhead
    pe_type = model_cfg.pe_type.lower()
    if pe_type == "rope":
        # approx cost of rotations (~6 ops per element)
        attn_total += 6.0 * batch_size * H * seq_len * d_head * 2
    elif pe_type == "alibi":
        attn_total += batch_size * H * seq_len * seq_len

    ffn_mult = 2
    activation = cfg.model.ffn_activation.lower()
    inner_dim = max(1, int(math.ceil(ffn_mult * D)))
    mlp_total = _activation_ffn_flops(activation, D, inner_dim, D, tokens)

    residual_cost = 4.0 * tokens * D  # two residual adds per block
    block_total = attn_total + mlp_total + residual_cost
    flops += model_cfg.n_layer * block_total

    # Heads
    target_shapes = model_cfg.target_shapes_by_head
    shoulder_out = int(target_shapes.get("shoulder"))
    c_out = int(target_shapes.get("c_stick"))
    main_out = int(target_shapes.get("main_stick"))
    btn_out = int(target_shapes.get("buttons"))

    shoulder_hidden = max(1, D // 2)
    flops += _swiglu_flops(D, shoulder_hidden, shoulder_out, tokens)

    c_in = D + shoulder_out
    c_hidden = max(1, c_in // 2)
    flops += _swiglu_flops(c_in, c_hidden, c_out, tokens)

    main_in = D + shoulder_out + c_out
    main_hidden = max(1, main_in // 2)
    flops += _swiglu_flops(main_in, main_hidden, main_out, tokens)

    btn_in = D + shoulder_out + c_out + main_out
    btn_hidden = max(1, max(btn_in // 2, btn_out * 2))
    flops += _swiglu_flops(btn_in, btn_hidden, btn_out, tokens)
    flops += 4.0 * tokens * btn_out  # sigmoid + logistic odds (~4 ops/token)

    return flops


def _shutdown_loader_iter(loader_iter: Any) -> None:
    if loader_iter is None:
        return
    shutdown = getattr(loader_iter, "_shutdown_workers", None)
    if shutdown is not None:
        try:
            shutdown()  # type: ignore[misc]
        except Exception:
            pass


def _shutdown_loader(loader: Any) -> None:
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        _shutdown_loader_iter(iterator)
        try:
            loader._iterator = None  # type: ignore[attr-defined]
        except Exception:
            pass


def _build_profiler_context(
    cfg: Config, run_id: str, *, verbose: bool
) -> Tuple[Any, Optional[Path]]:
    prof_cfg = cfg.profile
    if not getattr(prof_cfg, "enable", False) or not hasattr(torch, "profiler"):
        return nullcontext(), None

    activities = [torch.profiler.ProfilerActivity.CPU]
    mps_activity = getattr(torch.profiler.ProfilerActivity, "MPS", None)
    if mps_activity is not None:
        activities.append(mps_activity)

    base_dir = prof_cfg.out_dir or cfg.train.out_dir or "profiles"
    profile_root = Path(base_dir) / "profiles" / run_id
    profile_root.mkdir(parents=True, exist_ok=True)

    schedule = torch.profiler.schedule(
        wait=prof_cfg.wait,
        warmup=prof_cfg.warmup,
        active=prof_cfg.active,
        repeat=prof_cfg.repeat,
    )
    handler = torch.profiler.tensorboard_trace_handler(
        str(profile_root), worker_name=run_id
    )

    if verbose:
        print(f"[{run_id}] Profiling enabled; writing traces to {profile_root}")

    context = torch.profiler.profile(
        activities=tuple(activities),
        schedule=schedule,
        on_trace_ready=handler,
        record_shapes=prof_cfg.record_shapes,
        with_stack=prof_cfg.with_stack,
        profile_memory=prof_cfg.profile_memory,
    )
    return context, profile_root


def run_training_once(
    run_id: str,
    *,
    base_initial: Optional[Mapping[str, Any]],
    overrides: Mapping[str, str],
    target_loss: Optional[float],
    objectives: Sequence[StoppingObjective],
    device: torch.device,
    verbose: bool,
    time_limit_seconds: Optional[float] = None,
) -> TrainingRunResult:
    reset_config_for_tests()
    cfg = init_config(initial=base_initial, cli_overrides=overrides, freeze=False)

    if time_limit_seconds is not None and time_limit_seconds <= 0:
        raise ValueError(
            f"[{run_id}] time limit must be positive; got {time_limit_seconds}"
        )

    user_set_num_workers = any(key.endswith("train.num_workers") for key in overrides)
    user_set_pin_memory = any(key.endswith("train.pin_memory") for key in overrides)

    # Force settings that play nicely with early-stop and requested device.
    if cfg.train.num_workers > 0 and getattr(cfg.train, "persistent_workers", False):
        cfg.train.persistent_workers = False
    if (
        device.type == "mps"
        and getattr(cfg.train, "pin_memory", False)
        and not user_set_pin_memory
    ):
        cfg.train.pin_memory = False
    if (
        device.type != "cuda"
        and cfg.train.num_workers != 0
        and not user_set_num_workers
    ):
        cfg.train.num_workers = 0
    if cfg.train.num_workers == 0:
        cfg.train.prefetch_factor = 2
        cfg.train.persistent_workers = False

    cfg.freeze()

    objective_specs = [obj.raw for obj in objectives]
    required_metrics = {obj.metric for obj in objectives}
    need_main_metrics = bool(required_metrics & set(_MAIN_METRIC_KEYS))
    need_c_metrics = bool(required_metrics & set(_C_METRIC_KEYS))
    need_button_metrics = bool(required_metrics & set(_BUTTON_METRIC_KEYS))
    need_shoulder_metrics = bool(required_metrics & set(_SHOULDER_METRIC_KEYS))
    needs_running_metrics = any(
        [need_main_metrics, need_c_metrics, need_button_metrics, need_shoulder_metrics]
    )
    metrics_tracker: Optional[MetricsAccumulator] = None
    latest_metrics: Dict[str, float] = {}
    objectives_met = False

    try:
        model = GPT(cfg).to(device)
    except ValueError as exc:
        if verbose:
            print(f"Skipping {run_id}: invalid configuration ({exc}).")
        return TrainingRunResult(
            run_id=run_id,
            overrides=dict(overrides),
            target_loss=target_loss,
            target_objectives=objective_specs,
            steps=0,
            epochs_completed=0,
            final_loss=float("nan"),
            reached_target_loss=False,
            reached_target_objectives=False,
            interrupted=False,
            stopped_due_to_cap=False,
            stopped_due_to_time_limit=False,
            time_seconds=0.0,
            estimated_forward_flops=0.0,
            estimated_training_flops=0.0,
            average_loss=None,
            min_loss=None,
            max_loss=None,
            per_step_losses=[],
            total_tokens=0,
            tokens_per_second=None,
            stop_reason="invalid_config",
            parameter_count=0,
            config_snapshot=cfg.to_dict(),
            metrics_summary={},
        )
    parameter_count = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"[{run_id}] model parameter count: {parameter_count:,}")
    loader, ds, sampler = make_dataloader()
    colmap = ColumnMap.from_dataset(ds)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=cfg.train.betas,
        weight_decay=cfg.train.weight_decay,
    )

    steps_per_epoch = cfg.train.steps_per_epoch or math.ceil(len(loader))
    total_steps_cap = cfg.train.max_steps or (cfg.train.epochs * steps_per_epoch)

    global_step = 0
    epochs_completed = 0
    final_loss = float("inf")

    forward_flops = 0.0
    training_flops = 0.0
    loss_history: List[float] = []
    total_tokens = 0
    token_rate_ema: Optional[float] = None
    flop_rate_ema: Optional[float] = None
    last_batch_wall = time.perf_counter()

    start_time = time.perf_counter()
    reached_target_loss = False
    interrupted = False
    stopped_due_to_cap = False
    stopped_due_to_time_limit = False
    stop_reason = "completed"

    profiler_ctx, _ = _build_profiler_context(cfg, run_id, verbose=verbose)

    with profiler_ctx as profiler:
        try:
            for epoch in range(cfg.train.epochs):
                epochs_completed = epoch + 1
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch)
                model.train()

                loader_iter = iter(loader)
                progress = None
                steps_this_epoch = 0
                try:
                    try:
                        default_total = len(loader)  # type: ignore[arg-type]
                    except TypeError:
                        default_total = None
                    max_epoch_iters = cfg.train.steps_per_epoch or default_total
                    progress = tqdm(
                        total=max_epoch_iters,
                        desc=f"[{run_id}] epoch {epoch + 1}",
                        leave=False,
                        dynamic_ncols=False,
                        mininterval=0.5,
                    )
                    while True:
                        if time_limit_seconds is not None:
                            elapsed = time.perf_counter() - start_time
                            if elapsed >= time_limit_seconds:
                                stopped_due_to_time_limit = True
                                stop_reason = "time_limit"
                                break
                        if (
                            cfg.train.steps_per_epoch is not None
                            and steps_this_epoch >= cfg.train.steps_per_epoch
                        ):
                            break
                        if (
                            cfg.train.max_steps is not None
                            and global_step >= cfg.train.max_steps
                        ):
                            break
                        try:
                            batch = next(loader_iter)
                        except StopIteration:
                            break

                        X = batch["X"].to(device, non_blocking=True)
                        Y = batch["Y"].to(device, non_blocking=True)

                        inputs_td = build_model_inputs(X, colmap)
                        target_info = quantize_controller_targets(
                            Y, colmap, input_domain="unit11"
                        )

                        pred = model(inputs_td)
                        B, L, _ = pred["main_stick"].shape
                        total_tokens += int(B * L)

                        logits_main = pred["main_stick"].reshape(B * L, -1)
                        target_main = target_info["main_idx"].reshape(B * L)
                        main_weights = _compute_ce_weights(
                            target_main, target_info["main_K"]
                        )
                        loss_main = torch.nn.functional.cross_entropy(
                            logits_main,
                            target_main,
                            reduction="mean",
                            label_smoothing=cfg.train.label_smoothing,
                            weight=main_weights,
                        )

                        logits_c = pred["c_stick"].reshape(B * L, -1)
                        target_c = target_info["c_idx"].reshape(B * L)
                        c_weights = _compute_ce_weights(target_c, target_info["c_K"])
                        loss_c = torch.nn.functional.cross_entropy(
                            logits_c,
                            target_c,
                            reduction="mean",
                            label_smoothing=cfg.train.label_smoothing,
                            weight=c_weights,
                        )

                        logits_btn = pred["buttons"]
                        target_btn = target_info["buttons"]
                        pos_weight = _compute_pos_weights(target_btn)
                        loss_btn = torch.nn.functional.binary_cross_entropy_with_logits(
                            logits_btn,
                            target_btn,
                            reduction="mean",
                            pos_weight=pos_weight,
                        )

                        loss_s = torch.zeros((), device=device)
                        if (
                            "shoulder" in pred.keys()
                            and target_info["shoulder_K"] > 0
                            and target_info["shoulder_idx"] is not None
                        ):
                            logits_s = pred["shoulder"].reshape(B * L, -1)
                            target_s = target_info["shoulder_idx"].reshape(B * L)
                            loss_s = torch.nn.functional.cross_entropy(
                                logits_s,
                                target_s,
                                reduction="mean",
                                label_smoothing=cfg.train.label_smoothing,
                            )

                        loss = loss_main + loss_c + loss_btn + loss_s

                        # lr = cosine_lr_schedule(global_step, total_steps_cap, cfg.train.lr, cfg.train.warmup_steps)
                        lr = cfg.train.lr
                        for pg in opt.param_groups:
                            pg["lr"] = lr

                        opt.zero_grad(set_to_none=True)
                        loss.backward()

                        if cfg.train.grad_clip is not None and cfg.train.grad_clip > 0:
                            clip_grad_norm_(model.parameters(), cfg.train.grad_clip)

                        opt.step()

                        global_step += 1
                        final_loss = float(loss.item())

                        loss_history.append(final_loss)

                        # ----- Metrics will be computed lazily only when needed for objectives -----

                        batch_forward_flops = estimate_forward_flops(cfg, B, L)
                        forward_flops += batch_forward_flops
                        training_flops += batch_forward_flops * TRAINING_FLOP_MULTIPLIER

                        now = time.perf_counter()
                        step_duration = now - last_batch_wall
                        last_batch_wall = now
                        tokens_this_step = int(B * L)
                        batch_training_flops = (
                            batch_forward_flops * TRAINING_FLOP_MULTIPLIER
                        )
                        inst_token_rate: Optional[float] = None
                        inst_flop_rate: Optional[float] = None
                        if step_duration > 0 and tokens_this_step > 0:
                            inst_token_rate = tokens_this_step / step_duration
                            if token_rate_ema is None:
                                token_rate_ema = inst_token_rate
                            else:
                                token_rate_ema = (
                                    TOKEN_RATE_EMA_DECAY * token_rate_ema
                                    + (1.0 - TOKEN_RATE_EMA_DECAY) * inst_token_rate
                                )
                        if step_duration > 0 and batch_training_flops > 0:
                            inst_flop_rate = batch_training_flops / step_duration
                            if flop_rate_ema is None:
                                flop_rate_ema = inst_flop_rate
                            else:
                                flop_rate_ema = (
                                    TOKEN_RATE_EMA_DECAY * flop_rate_ema
                                    + (1.0 - TOKEN_RATE_EMA_DECAY) * inst_flop_rate
                                )

                        steps_this_epoch += 1
                        token_rate_display = (
                            f"{token_rate_ema:7.0f}"
                            if token_rate_ema is not None
                            else "   n/a"
                        )
                        flop_rate_display = (
                            format_flops(flop_rate_ema, per_second=True)
                            if flop_rate_ema is not None
                            else "n/a"
                        )
                        if progress is not None:
                            progress.update(1)

                        objective_status: List[
                            Tuple[StoppingObjective, bool, float]
                        ] = []
                        current_metrics: Dict[str, float] = {}

                        if objectives:

                            def _maybe_set_metric(name: str, value: float) -> None:
                                if name in required_metrics:
                                    current_metrics[name] = float(value)

                            # Loss metrics (shared aliases)
                            loss_main_value: float = loss_main.item()
                            loss_c_value: float = loss_c.item()
                            loss_btn_value: float = loss_btn.item()
                            loss_shoulder_value: float = loss_s.item()

                            for key in _LOSS_METRIC_KEYS:
                                if key == "loss" or key == "loss_total":
                                    _maybe_set_metric(key, final_loss)
                                elif key == "loss_main":
                                    _maybe_set_metric(key, loss_main_value)
                                elif key == "loss_c":
                                    _maybe_set_metric(key, loss_c_value)
                                elif key == "loss_buttons":
                                    _maybe_set_metric(key, loss_btn_value)
                                elif key == "loss_shoulder":
                                    _maybe_set_metric(key, loss_shoulder_value)

                            for key in _OTHER_DIRECT_METRIC_KEYS:
                                if key in required_metrics:
                                    if key in {"lr", "learning_rate"}:
                                        current_metrics[key] = float(lr)
                                    elif key == "global_step":
                                        current_metrics[key] = float(global_step)
                                    elif key == "tokens_per_second":
                                        if token_rate_ema is not None:
                                            current_metrics[key] = float(token_rate_ema)
                                        elif inst_token_rate is not None:
                                            current_metrics[key] = float(
                                                inst_token_rate
                                            )
                                    elif key == "flops_per_second":
                                        if flop_rate_ema is not None:
                                            current_metrics[key] = float(flop_rate_ema)
                                        elif inst_flop_rate is not None:
                                            current_metrics[key] = float(inst_flop_rate)

                            if needs_running_metrics:
                                if metrics_tracker is None:
                                    metrics_tracker = MetricsAccumulator(
                                        int(target_info["main_K"]),
                                        int(target_info["c_K"]),
                                        int(target_info["buttons_K"]),
                                        int(target_info["shoulder_K"]),
                                        device=device,
                                    )

                                if need_main_metrics:
                                    pred_main_idx = pred["main_stick"].argmax(dim=-1)
                                    true_main_idx = target_info["main_idx"].reshape(
                                        B, L
                                    )
                                    metrics_tracker.update_stick_metrics(
                                        pred_main_idx.reshape(-1),
                                        true_main_idx.reshape(-1),
                                        stick_type="main",
                                    )

                                if need_c_metrics:
                                    pred_c_idx = pred["c_stick"].argmax(dim=-1)
                                    true_c_idx = target_info["c_idx"].reshape(B, L)
                                    metrics_tracker.update_stick_metrics(
                                        pred_c_idx.reshape(-1),
                                        true_c_idx.reshape(-1),
                                        stick_type="c",
                                    )

                                if need_button_metrics:
                                    btn_logits = pred["buttons"]
                                    btn_true = target_info["buttons"]
                                    btn_probs = pred.get("buttons_probs")
                                    if btn_probs is None:
                                        btn_probs = torch.sigmoid(btn_logits)
                                    metrics_tracker.update_button_metrics(
                                        btn_true, btn_probs > 0.5, btn_logits
                                    )

                                if need_shoulder_metrics:
                                    shoulder_logits = pred.get("shoulder")
                                    shoulder_idx = target_info.get("shoulder_idx")
                                    if (
                                        shoulder_logits is not None
                                        and shoulder_idx is not None
                                        and int(target_info["shoulder_K"]) > 0
                                    ):
                                        shoulder_pred_idx = shoulder_logits.argmax(
                                            dim=-1
                                        ).reshape(-1)
                                        metrics_tracker.update_shoulder_metrics(
                                            shoulder_pred_idx,
                                            shoulder_idx.reshape(-1),
                                        )

                                summary = metrics_tracker.get_summary()
                                for key in required_metrics & _RUNNING_METRIC_KEYS:
                                    if key not in summary:
                                        raise ValueError(
                                            f"[{run_id}] metric '{key}' unavailable; available metrics: {sorted(summary.keys())}"
                                        )
                                    current_metrics[key] = float(summary[key])

                            all_met = True
                            objective_status = []
                            for obj in objectives:
                                if obj.metric not in current_metrics:
                                    raise ValueError(
                                        f"[{run_id}] objective '{obj.raw}' refers to unknown metric. "
                                        f"Available metrics this step: {sorted(current_metrics.keys())}"
                                    )
                                met, value = obj.is_satisfied(current_metrics)
                                objective_status.append((obj, met, value))
                                if not met:
                                    all_met = False

                            latest_metrics = current_metrics.copy()

                            if all_met:
                                objectives_met = True
                                stop_reason = "objective_met"
                                if verbose:
                                    objective_str = ", ".join(
                                        obj.raw for obj in objectives
                                    )
                                    print(
                                        f"[{run_id}] objectives met at step {global_step}: {objective_str}"
                                    )
                        else:
                            latest_metrics = {"loss": final_loss}

                        if progress is not None:
                            postfix_parts = [
                                f"loss={final_loss:.4f}",
                                f"lr={lr:.2e}",
                                f"tok/s={token_rate_display}",
                                f"flops/s={flop_rate_display}",
                            ]
                            if objective_status:
                                obj_parts = []
                                for obj, met, value in objective_status:
                                    mark = "✓" if met else ""
                                    obj_parts.append(f"{obj.metric}={value:.3f}{mark}")
                                postfix_parts.append("obj:" + ",".join(obj_parts))
                            progress.set_postfix_str(" ".join(postfix_parts))

                        if profiler is not None:
                            profiler.step()

                        if target_loss is not None and final_loss <= target_loss:
                            reached_target_loss = True
                            stop_reason = "target_met"
                            if verbose:
                                print(
                                    f"[{run_id}] target loss reached at step {global_step}: {final_loss:.4f}"
                                )
                            _shutdown_loader_iter(loader_iter)
                            loader_iter = None
                            break

                        if objectives_met:
                            _shutdown_loader_iter(loader_iter)
                            loader_iter = None
                            break

                        if global_step >= total_steps_cap:
                            stopped_due_to_cap = True
                            stop_reason = "max_steps_reached"
                            if verbose:
                                print(
                                    f"[{run_id}] max step cap reached at step {global_step}"
                                )
                            _shutdown_loader_iter(loader_iter)
                            loader_iter = None
                            break

                        if time_limit_seconds is not None:
                            elapsed = time.perf_counter() - start_time
                            if elapsed >= time_limit_seconds:
                                stopped_due_to_time_limit = True
                                stop_reason = "time_limit"
                                _shutdown_loader_iter(loader_iter)
                                loader_iter = None
                                break
                finally:
                    if progress is not None:
                        progress.close()
                    _shutdown_loader_iter(loader_iter)
                    loader_iter = None

                if (
                    reached_target_loss
                    or objectives_met
                    or global_step >= total_steps_cap
                    or stopped_due_to_time_limit
                ):
                    break
        except KeyboardInterrupt:
            interrupted = True
            if verbose:
                print(f"[{run_id}] received KeyboardInterrupt; wrapping up run.")
        finally:
            _shutdown_loader(loader)

    elapsed = time.perf_counter() - start_time

    # Clean-up device memory when using CUDA to avoid accumulation between runs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if interrupted:
        stop_reason = "interrupted"
    elif objectives_met:
        stop_reason = "objective_met"
    elif reached_target_loss:
        stop_reason = "target_met"
    elif stopped_due_to_cap:
        stop_reason = "max_steps_reached"
    elif stopped_due_to_time_limit:
        stop_reason = "time_limit"
    elif epochs_completed >= cfg.train.epochs:
        stop_reason = "epochs_completed"
    else:
        stop_reason = stop_reason or "completed"

    return TrainingRunResult(
        run_id=run_id,
        overrides=dict(overrides),
        target_loss=target_loss,
        target_objectives=objective_specs,
        steps=global_step,
        epochs_completed=epochs_completed,
        final_loss=final_loss,
        reached_target_loss=reached_target_loss,
        reached_target_objectives=objectives_met,
        interrupted=interrupted,
        stopped_due_to_cap=stopped_due_to_cap,
        stopped_due_to_time_limit=stopped_due_to_time_limit,
        time_seconds=elapsed,
        estimated_forward_flops=forward_flops,
        estimated_training_flops=training_flops,
        average_loss=(sum(loss_history) / len(loss_history)) if loss_history else None,
        min_loss=min(loss_history) if loss_history else None,
        max_loss=max(loss_history) if loss_history else None,
        per_step_losses=loss_history,
        total_tokens=total_tokens,
        tokens_per_second=(
            (total_tokens / elapsed) if (elapsed > 0 and total_tokens > 0) else None
        ),
        stop_reason=stop_reason,
        parameter_count=parameter_count,
        config_snapshot=get_config().to_dict(),
        metrics_summary=latest_metrics,
    )


def summarise_results(results: Sequence[TrainingRunResult]) -> str:
    if not results:
        return "No runs executed."
    valid_results = [r for r in results if r.stop_reason != "invalid_config"]
    if not valid_results:
        return "No successful runs."
    best_by_flops = min(valid_results, key=lambda r: r.estimated_training_flops)
    best_by_time = min(valid_results, key=lambda r: r.time_seconds)
    best_by_loss = min(valid_results, key=lambda r: r.final_loss)
    detail_lines = []
    for res in valid_results:
        status_bits = []
        if res.reached_target_loss:
            status_bits.append("target met")
        if res.reached_target_objectives:
            status_bits.append("objective met")
        if res.stopped_due_to_cap:
            status_bits.append("max steps reached")
        if res.stopped_due_to_time_limit:
            status_bits.append("time limit")
        if res.interrupted:
            status_bits.append("interrupted")
        status = ", ".join(status_bits) if status_bits else "completed"
        target_text = f"{res.target_loss:.4f}" if res.target_loss is not None else "n/a"
        objective_text = (
            ", ".join(res.target_objectives) if res.target_objectives else "n/a"
        )
        flops_rate = res.estimated_training_flops / max(res.time_seconds, 1e-9)
        mean_loss = res.average_loss
        mean_loss_text = f"{mean_loss:.4f}" if mean_loss is not None else "n/a"
        tokens_rate = res.tokens_per_second
        tokens_rate_text = f"{tokens_rate:,.0f}" if tokens_rate is not None else "n/a"
        reason_text = (
            res.stop_reason.replace("_", " ") if res.stop_reason else "unknown"
        )
        detail_lines.append(
            (
                f"{res.run_id}: loss {res.final_loss:.4f} (target {target_text}) | objective {objective_text} | mean {mean_loss_text} | steps {res.steps} | "
                f"epochs {res.epochs_completed} | time {res.time_seconds:.1f}s | "
                f"total {format_flops(res.estimated_training_flops)} | {format_flops(flops_rate, per_second=True)} | "
                f"tokens/s {tokens_rate_text} | {status} | reason {reason_text} | overrides {format_overrides(res.overrides)}"
            )
        )
    best_by_flops_rate = format_flops(
        best_by_flops.estimated_training_flops / max(best_by_flops.time_seconds, 1e-9),
        per_second=True,
    )
    best_by_time_rate = format_flops(
        best_by_time.estimated_training_flops / max(best_by_time.time_seconds, 1e-9),
        per_second=True,
    )
    best_by_flops_tokens = (
        f"{best_by_flops.tokens_per_second:,.0f}"
        if best_by_flops.tokens_per_second is not None
        else "n/a"
    )
    best_by_time_tokens = (
        f"{best_by_time.tokens_per_second:,.0f}"
        if best_by_time.tokens_per_second is not None
        else "n/a"
    )
    best_by_loss_tokens = (
        f"{best_by_loss.tokens_per_second:,.0f}"
        if best_by_loss.tokens_per_second is not None
        else "n/a"
    )
    best_by_flops_reason = (
        best_by_flops.stop_reason.replace("_", " ")
        if best_by_flops.stop_reason
        else "unknown"
    )
    best_by_time_reason = (
        best_by_time.stop_reason.replace("_", " ")
        if best_by_time.stop_reason
        else "unknown"
    )
    best_by_loss_reason = (
        best_by_loss.stop_reason.replace("_", " ")
        if best_by_loss.stop_reason
        else "unknown"
    )

    lines = [
        f"Executed {len(results)} run(s).",
        (
            f"Best loss:  {best_by_loss.run_id} | loss {best_by_loss.final_loss:.4f} | "
            f"time {best_by_loss.time_seconds:.1f}s | total {format_flops(best_by_loss.estimated_training_flops)} | "
            f"tokens/s {best_by_loss_tokens} | reason {best_by_loss_reason}"
        ),
        (
            f"Best FLOPs: {best_by_flops.run_id} | loss {best_by_flops.final_loss:.4f} | "
            f"time {best_by_flops.time_seconds:.1f}s | total {format_flops(best_by_flops.estimated_training_flops)} | "
            f"{best_by_flops_rate} | tokens/s {best_by_flops_tokens} | reason {best_by_flops_reason}"
        ),
        (
            f"Best time:  {best_by_time.run_id} | loss {best_by_time.final_loss:.4f} | "
            f"time {best_by_time.time_seconds:.1f}s | total {format_flops(best_by_time.estimated_training_flops)} | "
            f"{best_by_time_rate} | tokens/s {best_by_time_tokens} | reason {best_by_time_reason}"
        ),
        "---",
        *detail_lines,
        "Parameter counts:",
        *[f"{res.run_id}: {res.parameter_count:,}" for res in valid_results],
    ]
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config-json", type=Path, default=None, help="Optional base config JSON file."
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values (repeatable).",
    )
    p.add_argument(
        "--search",
        action="append",
        default=[],
        metavar="KEY=V1,V2",
        help="Hyperparameter grid; repeat to define multiple axes.",
    )
    p.add_argument(
        "--target-loss", type=float, default=1, help="Early-stop when loss <= target."
    )
    p.add_argument(
        "--target-objective",
        action="append",
        default=[],
        metavar="METRIC{>=,<=,>,<,==}VALUE",
        help=(
            "Early-stop when the given metric objective is met. "
            "Examples: --target-objective acc_main>=0.92 --target-objective loss<=1.5"
        ),
    )
    p.add_argument(
        "--time-limit-seconds",
        type=float,
        default=None,
        help="Optional wall-clock time limit (seconds) applied to each run.",
    )
    p.add_argument(
        "--device", type=str, default=None, help="Device to use (cpu/cuda/mps)."
    )
    p.add_argument(
        "--report", type=Path, default=None, help="Optional path to save JSON report."
    )
    p.add_argument(
        "--results-log",
        type=Path,
        default=None,
        help="Optional path to stream TrainingRunResult objects as JSONL (one per run).",
    )
    p.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable periodic stdout logging during training.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    initial_data: Optional[Dict[str, Any]] = None
    if args.config_json is not None:
        with args.config_json.open("r", encoding="utf-8") as f:
            initial_data = json.load(f)

    base_overrides = dict(parse_key_value(s) for s in args.set)
    search_space = parse_search_items(args.search)

    device = _resolve_device()
    verbose = not args.no_verbose
    results_log_path = args.results_log
    objectives = [parse_objective(expr) for expr in args.target_objective]

    if results_log_path is not None:
        results_log_path.parent.mkdir(parents=True, exist_ok=True)
        with results_log_path.open("w", encoding="utf-8"):
            pass

    results: List[TrainingRunResult] = []

    search_combos = list(expand_search_space(search_space))
    total_runs = len(search_combos)
    if verbose:
        print(f"Planned runs: {total_runs}")
    for idx, combo in enumerate(search_combos, start=1):
        combined_overrides = {**base_overrides, **combo}
        run_id = f"run_{idx:03d}"
        if verbose:
            print(f"Starting {run_id} with overrides: {combined_overrides}")
        result = run_training_once(
            run_id,
            base_initial=initial_data,
            overrides=combined_overrides,
            target_loss=args.target_loss,
            objectives=objectives,
            device=device,
            verbose=verbose,
            time_limit_seconds=args.time_limit_seconds,
        )
        results.append(result)
        if results_log_path is not None:
            append_result_jsonl(results_log_path, result)
        if verbose and result.stop_reason != "invalid_config":
            target_text = (
                f"{result.target_loss:.4f}" if result.target_loss is not None else "n/a"
            )
            status_bits = []
            if result.reached_target_loss:
                status_bits.append("target met")
            if result.reached_target_objectives:
                status_bits.append("objective met")
            if result.stopped_due_to_cap:
                status_bits.append("max steps reached")
            if result.interrupted:
                status_bits.append("interrupted")
            status = ", ".join(status_bits) if status_bits else "completed"
            flops_rate = result.estimated_training_flops / max(
                result.time_seconds, 1e-9
            )
            overrides_str = format_overrides(result.overrides)
            mean_loss = (
                f"{result.average_loss:.4f}"
                if result.average_loss is not None
                else "n/a"
            )
            tokens_rate_str = (
                f"{result.tokens_per_second:,.0f}"
                if result.tokens_per_second is not None
                else "n/a"
            )
            reason_text = (
                result.stop_reason.replace("_", " ")
                if result.stop_reason
                else "unknown"
            )
            objective_parts: List[str] = []
            for expr in result.target_objectives:
                try:
                    obj = parse_objective(expr)
                except ValueError:
                    objective_parts.append(expr)
                    continue
                metric_val = result.metrics_summary.get(obj.metric)
                if metric_val is not None:
                    objective_parts.append(f"{expr} (value {metric_val:.4f})")
                else:
                    objective_parts.append(expr)
            objective_text = ", ".join(objective_parts) if objective_parts else "n/a"
            print(
                f"Finished {run_id}:\n"
                f"  loss {result.final_loss:.4f} (target {target_text}) | mean {mean_loss}\n"
                f"  steps {result.steps} | epochs {result.epochs_completed}\n"
                f"  time {result.time_seconds:.1f}s | total {format_flops(result.estimated_training_flops)} | {format_flops(flops_rate, per_second=True)}\n"
                f"  throughput tokens/s {tokens_rate_str}\n"
                f"  status {status} | reason {reason_text}\n"
                f"  objectives {objective_text}\n"
                f"  overrides {overrides_str}"
            )

    summary = summarise_results(results)
    print(summary)

    if args.report is not None:
        payload = {
            "summary": summary,
            "results": [r.to_dict() for r in results],
        }
        args.report.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if verbose:
            print(f"Report written to {args.report}")

    if results_log_path is not None and verbose:
        print(f"Per-run JSONL log written to {results_log_path}")


if __name__ == "__main__":
    main()
