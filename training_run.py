"""Shared training utilities for command-line and W&B orchestrators."""

from __future__ import annotations

import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from column_map import ColumnMap
from config import Config, init_config, reset_config_for_tests
from controller_quantization import quantize_targets
from loss import _compute_ce_weights, _compute_pos_weights
from model.gpt import GPTv7
from train import (
    RunningMetrics,
    build_inputs_for_gptv7,
    cosine_lr_schedule,
    _sorted_checkpoint_paths,
)
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
    "loss_aux",
)
_OTHER_DIRECT_METRIC_KEYS: Tuple[str, ...] = (
    "lr",
    "learning_rate",
    "global_step",
    "tokens_per_second",
    "flops_per_second",
)

_OBJECTIVE_OPERATORS: Tuple[str, ...] = (">=", "<=", ">", "<", "==")


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
            except ValueError as exc:  # pragma: no cover - defensive, rarely hit
                raise ValueError(
                    f"Objective '{expr}' has non-numeric threshold '{threshold}'."
                ) from exc
            return StoppingObjective(metric=metric, operator=op, threshold=value, raw=expr)
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
    flops += q_proj + k_proj + v_proj + o_proj

    # Attention scores + softmax + value weighting
    attn_tokens = tokens * H * seq_len
    flops += 2.0 * attn_tokens * d_head
    flops += 2.0 * attn_tokens  # softmax exp + normalisation
    flops += 2.0 * attn_tokens * d_head

    if attn_type in {"mqa", "gqa"} and kv_heads < H:
        # Account for grouped computation differences (rough estimate)
        flops *= max(1.0, groups)

    # Feed-forward layers
    inner_dim = int(model_cfg.ffn_mult * D)
    flops += _activation_ffn_flops(model_cfg.ffn_activation, D, inner_dim, D, tokens)

    # Output heads (per token)
    if "main_stick" in model_cfg.target_shapes_by_head:
        flops += 2.0 * tokens * D * model_cfg.target_shapes_by_head["main_stick"][0]
    if "c_stick" in model_cfg.target_shapes_by_head:
        flops += 2.0 * tokens * D * model_cfg.target_shapes_by_head["c_stick"][0]
    if "buttons" in model_cfg.target_shapes_by_head:
        flops += 2.0 * tokens * D * model_cfg.target_shapes_by_head["buttons"][0]
    if "shoulder" in model_cfg.target_shapes_by_head:
        flops += 2.0 * tokens * D * model_cfg.target_shapes_by_head["shoulder"][0]

    return flops


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


def _swiglu_flops(
    input_dim: int,
    inner_dim: int,
    output_dim: int,
    tokens: int,
) -> float:
    if tokens == 0:
        return 0.0
    # swiglu = 2 linear projections + elementwise silu + mul + output linear
    linear_w1 = 2.0 * tokens * input_dim * inner_dim
    linear_v1 = 2.0 * tokens * input_dim * inner_dim
    linear_w2 = 2.0 * tokens * inner_dim * output_dim
    activations = 4.0 * tokens * inner_dim
    return linear_w1 + linear_v1 + linear_w2 + activations


def gather_objective_metrics(metrics: Mapping[str, float], keys: Iterable[str]) -> Dict[str, float]:
    return {k: float(metrics[k]) for k in keys if k in metrics}


def run_training_once(
    run_id: str,
    *,
    base_initial: Optional[Mapping[str, Any]],
    overrides: Mapping[str, str],
    target_loss: Optional[float],
    objectives: Sequence[StoppingObjective],
    device: torch.device,
    verbose: bool,
) -> TrainingRunResult:
    reset_config_for_tests()
    cfg = init_config(initial=base_initial, cli_overrides=overrides, freeze=False)

    user_set_num_workers = any(key.endswith("train.num_workers") for key in overrides)
    user_set_pin_memory = any(key.endswith("train.pin_memory") for key in overrides)

    # Force settings that play nicely with early-stop and requested device.
    if cfg.train.num_workers > 0 and getattr(cfg.train, "persistent_workers", False):
        cfg.train.persistent_workers = False
    if device.type == "mps" and getattr(cfg.train, "pin_memory", False) and not user_set_pin_memory:
        cfg.train.pin_memory = False
    if device.type != "cuda" and cfg.train.num_workers != 0 and not user_set_num_workers:
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
    metrics_tracker: Optional[RunningMetrics] = None
    latest_metrics: Dict[str, float] = {}
    objectives_met = False

    if cfg.model.use_moe and cfg.model.moe_num_active > cfg.model.moe_num_experts:
        if verbose:
            print(
                f"Skipping {run_id}: moe_num_active ({cfg.model.moe_num_active}) "
                f"exceeds moe_num_experts ({cfg.model.moe_num_experts})."
            )
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

    try:
        model = GPTv7().to(device)
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
    total_tokens = 0
    loss_history: List[float] = []
    interrupted = False
    stop_reason = "completed"
    stopped_due_to_cap = False

    context = nullcontext()
    run_start_time = time.perf_counter()
    last_batch_wall = run_start_time
    token_rate_ema: Optional[float] = None
    flop_rate_ema: Optional[float] = None

    try:
        with context:
            for epoch in range(cfg.train.epochs):
                epochs_completed = epoch
                steps_this_epoch = 0
                loader_iter = iter(loader)
                if verbose:
                    print(f"[{run_id}] Starting epoch {epoch + 1}/{cfg.train.epochs}")
                progress = None
                if verbose:
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
                    except TypeError:
                        progress = None

                while True:
                    if (
                        cfg.train.steps_per_epoch is not None
                        and steps_this_epoch >= cfg.train.steps_per_epoch
                    ):
                        break
                    if cfg.train.max_steps is not None and global_step >= cfg.train.max_steps:
                        break
                    try:
                        batch = next(loader_iter)
                    except StopIteration:
                        break

                    X = batch["X"].to(device, non_blocking=True)
                    Y = batch["Y"].to(device, non_blocking=True)

                    inputs_td = build_inputs_for_gptv7(X, colmap)
                    target_info = quantize_targets(Y, colmap, input_domain="unit11")

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

                    loss_aux = torch.zeros((), device=device)
                    if cfg.model.use_moe and "moe_aux_loss" in pred.keys():
                        loss_aux = pred["moe_aux_loss"].mean() * cfg.model.moe_aux_loss_weight

                    loss = loss_main + loss_c + loss_btn + loss_s + loss_aux

                    lr = cosine_lr_schedule(
                        global_step, total_steps_cap, cfg.train.lr, cfg.train.warmup_steps
                    )
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
                    batch_training_flops = batch_forward_flops * TRAINING_FLOP_MULTIPLIER
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
                    if progress is not None:
                        progress.update(1)

                    objective_status: List[Tuple[StoppingObjective, bool, float]] = []
                    current_metrics: Dict[str, float] = {}

                    if objectives:

                        def _maybe_set_metric(name: str, value: float) -> None:
                            if name in required_metrics:
                                current_metrics[name] = float(value)

                        # Loss metrics (shared aliases)
                        loss_main_value = float(loss_main.detach().item())
                        loss_c_value = float(loss_c.detach().item())
                        loss_btn_value = float(loss_btn.detach().item())
                        loss_shoulder_value = float(loss_s.detach().item())
                        loss_aux_value = float(loss_aux.detach().item()) if loss_aux is not None else 0.0

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
                            elif key == "loss_aux":
                                _maybe_set_metric(key, loss_aux_value)

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
                                        current_metrics[key] = float(inst_token_rate)
                                elif key == "flops_per_second":
                                    if flop_rate_ema is not None:
                                        current_metrics[key] = float(flop_rate_ema)
                                    elif inst_flop_rate is not None:
                                        current_metrics[key] = float(inst_flop_rate)

                        if needs_running_metrics:
                            if metrics_tracker is None:
                                metrics_tracker = RunningMetrics(
                                    int(target_info["main_K"]),
                                    int(target_info["c_K"]),
                                    int(target_info["buttons_K"]),
                                    int(target_info["shoulder_K"]),
                                    device=device,
                                )

                            if need_main_metrics:
                                pred_main_idx = pred["main_stick"].argmax(dim=-1)
                                true_main_idx = target_info["main_idx"].reshape(B, L)
                                main_major = metrics_tracker._majority_label(
                                    metrics_tracker.main_label_counts
                                )
                                main_rand = torch.randint(
                                    high=int(target_info["main_K"]), size=(B * L,), device=device
                                )
                                main_rep = torch.zeros_like(true_main_idx)
                                rep_mask_main = torch.ones((B, L), dtype=torch.bool, device=device)
                                rep_mask_main[:, 0] = False
                                if L > 1:
                                    main_rep[:, 1:] = true_main_idx[:, :-1]
                                metrics_tracker.update_main(
                                    pred_main_idx.reshape(-1),
                                    true_main_idx.reshape(-1),
                                    main_rand,
                                    main_major,
                                    main_rep.reshape(-1),
                                    rep_mask_main.reshape(-1),
                                )

                            if need_c_metrics:
                                pred_c_idx = pred["c_stick"].argmax(dim=-1)
                                true_c_idx = target_info["c_idx"].reshape(B, L)
                                c_major = metrics_tracker._majority_label(
                                    metrics_tracker.c_label_counts
                                )
                                c_rand = torch.randint(
                                    high=int(target_info["c_K"]), size=(B * L,), device=device
                                )
                                c_rep = torch.zeros_like(true_c_idx)
                                rep_mask_c = torch.ones((B, L), dtype=torch.bool, device=device)
                                rep_mask_c[:, 0] = False
                                if L > 1:
                                    c_rep[:, 1:] = true_c_idx[:, :-1]
                                metrics_tracker.update_c(
                                    pred_c_idx.reshape(-1),
                                    true_c_idx.reshape(-1),
                                    c_rand,
                                    c_major,
                                    c_rep.reshape(-1),
                                    rep_mask_c.reshape(-1),
                                )

                            if need_button_metrics:
                                btn_logits = pred["buttons"]
                                btn_true = target_info["buttons"]
                                metrics_tracker.update_buttons(btn_logits, btn_true)

                            if need_shoulder_metrics and target_info["shoulder_idx"] is not None:
                                pred_s_idx = pred["shoulder"].argmax(dim=-1)
                                true_s_idx = target_info["shoulder_idx"].reshape(B, L)
                                s_major = metrics_tracker._majority_label(
                                    metrics_tracker.shoulder_label_counts
                                )
                                s_rand = torch.randint(
                                    high=int(target_info["shoulder_K"]), size=(B * L,), device=device
                                )
                                metrics_tracker.update_shoulder(
                                    pred_s_idx.reshape(-1),
                                    true_s_idx.reshape(-1),
                                    s_rand,
                                    s_major,
                                )

                        for obj in objectives:
                            satisfied, value = obj.is_satisfied(current_metrics)
                            objective_status.append((obj, satisfied, value))

                        if all(status for _, status, _ in objective_status):
                            objectives_met = True
                            latest_metrics.update(current_metrics)
                            stop_reason = "objective_met"
                            break

                    if target_loss is not None and final_loss <= target_loss:
                        stop_reason = "target_loss_met"
                        break

                if progress is not None:
                    progress.close()

                if stop_reason in {"target_loss_met", "objective_met"}:
                    break

            stopped_due_to_cap = bool(
                (cfg.train.max_steps is not None and global_step >= cfg.train.max_steps)
                or (
                    cfg.train.steps_per_epoch is not None
                    and cfg.train.steps_per_epoch <= steps_this_epoch
                    and cfg.train.epochs == epoch + 1
                )
            )
    except KeyboardInterrupt:
        interrupted = True
        stop_reason = "interrupted"

    time_seconds = time.perf_counter() - run_start_time
    average_loss = float(sum(loss_history) / len(loss_history)) if loss_history else None
    min_loss = float(min(loss_history)) if loss_history else None
    max_loss = float(max(loss_history)) if loss_history else None
    tokens_per_second = None
    if time_seconds > 0 and total_tokens > 0:
        tokens_per_second = total_tokens / time_seconds

    metrics_summary: Dict[str, float] = {}
    if metrics_tracker is not None:
        metrics_summary.update(metrics_tracker.compute_summary())
    metrics_summary.update(latest_metrics)

    return TrainingRunResult(
        run_id=run_id,
        overrides=dict(overrides),
        target_loss=target_loss,
        target_objectives=objective_specs,
        steps=global_step,
        epochs_completed=epochs_completed,
        final_loss=final_loss,
        reached_target_loss=bool(target_loss is not None and final_loss <= target_loss),
        reached_target_objectives=objectives_met,
        interrupted=interrupted,
        stopped_due_to_cap=stopped_due_to_cap,
        time_seconds=time_seconds,
        estimated_forward_flops=forward_flops,
        estimated_training_flops=training_flops,
        average_loss=average_loss,
        min_loss=min_loss,
        max_loss=max_loss,
        per_step_losses=loss_history,
        total_tokens=total_tokens,
        tokens_per_second=tokens_per_second,
        stop_reason=stop_reason,
        parameter_count=parameter_count,
        config_snapshot=cfg.to_dict(),
        metrics_summary=metrics_summary,
    )


def latest_checkpoint_from_result(result: TrainingRunResult) -> Optional[Path]:
    train_cfg = result.config_snapshot.get("train")
    if not isinstance(train_cfg, Mapping):
        return None
    out_dir = train_cfg.get("out_dir")
    if not isinstance(out_dir, str) or not out_dir:
        return None
    directory = Path(out_dir).expanduser().resolve()
    if not directory.exists():
        return None
    checkpoints = _sorted_checkpoint_paths(directory)
    if not checkpoints:
        return None
    return checkpoints[0]

