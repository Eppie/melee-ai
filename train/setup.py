"""Utilities for configuring the training loop entrypoints."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from pprint import pformat
from typing import Dict, Sequence, Tuple

import torch
from loguru import logger
from torch.amp import GradScaler
from torch.amp.autocast_mode import is_autocast_available

from column_map import ColumnMap
from config.config import get_config
from model.compile_utils import maybe_torch_compile
from model.nano_gpt import GPT
from train.batch_utils import SampleWeightRatios
from train.checkpoint import _load_latest_checkpoint
from train.components import AMPContext, TrainingComponents
from train.wandb_utils import LocalLogger, WandbConfig, WandbLogger, init_wandb
from utils import _resolve_device, Profiler


def parse_cli_overrides(argv: Sequence[str]) -> Dict[str, str]:
    """
    Parses CLI arguments for --set KEY=VALUE overrides.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ns, _ = parser.parse_known_args(argv)

    overrides: Dict[str, str] = {}
    for item in ns.set:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected KEY=VALUE.")
        key, value = item.split("=", 1)
        overrides[key.strip()] = value.strip()
    return overrides


def _make_printable_config(value):
    """Recursively convert complex config values into printable representations."""
    if isinstance(value, dict):
        return {k: _make_printable_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_make_printable_config(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_make_printable_config(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def print_config(config: object) -> None:
    """Pretty prints the resolved Pydantic configuration."""
    printable_config = _make_printable_config(config.model_dump(mode="python"))
    logger.info(
        "Resolved training configuration:\n"
        + pformat(printable_config, indent=2, width=100)
    )


def configure_amp(config, device: torch.device) -> AMPContext:
    """Configures and reports the AMP (Automatic Mixed Precision) context."""
    if device.type in ("cuda", "mps") and is_autocast_available(device.type):
        device_type = device.type
    else:
        device_type = "cpu"

    # Parse dtype string from config
    dtype_str = config.train.amp_dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(dtype_str, torch.float16)

    amp_context = AMPContext(
        enabled=config.train.use_amp,
        device_type=device_type,
        dtype=dtype,
    )

    logger.info(f"Using PyTorch {torch.__version__}")
    if amp_context.enabled:
        backend = amp_context.device_type.upper()
        logger.info(f"AMP enabled with {dtype_str} on {backend} backend")
    else:
        logger.info(
            f"AMP requested but disabled for device '{device.type}';"
            " falling back to full precision."
        )

    return amp_context


def configure_performance_settings(config, device: torch.device) -> None:
    """Configure global PyTorch performance settings based on config."""

    # Configure torch.compile cache directory to avoid recompilation
    if config.train.torch_compile and "TORCHINDUCTOR_CACHE_DIR" not in os.environ:
        cache_dir = Path.home() / ".cache" / "torch" / "inductor"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
        logger.info(f"torch.compile cache enabled: {cache_dir}")

    # Use legacy API for TF32 settings to avoid mixing APIs
    if device.type == "cuda":
        # Enable TF32 for matmul (faster on Ampere+ GPUs)
        torch.backends.cuda.matmul.allow_tf32 = True
        # Enable TF32 for convolutions as well
        torch.backends.cudnn.allow_tf32 = True

    # Enable cudnn.benchmark for faster convolutions with consistent input sizes
    if config.train.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("cudnn.benchmark enabled for faster CUDA operations")


def build_optimizer(model: GPT, config) -> torch.optim.Optimizer:
    """
    Builds the AdamW optimizer with proper weight decay handling.

    Separates parameters into two groups:
    1. Decayed: Weights of Linear and Embedding layers
    2. No Decay: Biases, LayerNorm/RMSNorm weights, and other 1D tensors
    """
    # Separate parameters into decay and no-decay groups
    decay_params = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Common heuristic: decay 2D+ tensors (weights), skip 1D (biases, layernorms)
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": config.train.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    fused = torch.cuda.is_available()
    return torch.optim.AdamW(
        optim_groups,
        lr=config.train.lr,
        betas=config.train.betas,
        fused=fused,
    )


def initialize_training_components(
    model: GPT,
    loader,
    ds,
    sampler,
    debug: bool,
) -> Tuple[TrainingComponents, int, int, int]:
    """
    Initializes and wires together the objects needed for training.
    """
    config = get_config()
    device = _resolve_device(None)
    model = model.to(device)

    # Configure global performance settings (cudnn.benchmark, etc.)
    configure_performance_settings(config, device)

    # Optionally compile model with torch.compile for faster execution
    model = maybe_torch_compile(
        model,
        label="GPT",
        enable=config.train.torch_compile,
        mode=config.train.torch_compile_mode,
    )

    amp = configure_amp(config, device)

    column_map = ColumnMap.from_dataset(ds)
    value_idx = column_map.value_idx
    lw_cfg = config.loss_weights
    button_overrides = {
        "button_z": lw_cfg.button_z,
        "button_b": lw_cfg.button_b,
        "button_a": lw_cfg.button_a,
        "button_xy": lw_cfg.button_xy,
        "button_lr": lw_cfg.button_lr,
    }
    ratios = SampleWeightRatios(
        main_change=lw_cfg.main_change,
        c_change=lw_cfg.c_change,
        shoulder_change=lw_cfg.shoulder_change,
        buttons_change_default=lw_cfg.buttons_change_default,
        buttons_change_per_key=button_overrides,
        hold_base=lw_cfg.hold_base,
        value_change=lw_cfg.value_change,
    )

    optimizer = build_optimizer(model, config)
    # GradScaler is only needed for float16, not bfloat16
    use_grad_scaler = amp.enabled and amp.dtype == torch.float16
    scaler_device = amp.device_type if use_grad_scaler else "cpu"
    scaler = GradScaler(device=scaler_device, enabled=use_grad_scaler)

    steps_per_epoch = math.ceil(len(loader))
    total_steps = config.train.epochs * steps_per_epoch

    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_step_file = out_dir / "last_step.txt"

    wandb_run = None
    if not debug:
        wandb_cfg = WandbConfig(
            project=config.train.wandb_project,
            name=config.train.run_name,
            mode=config.train.wandb_mode,
        )
        wandb_run = init_wandb(
            config=wandb_cfg,
            run_dir=out_dir,
            hyperparameters={
                "train": dict(vars(config.train)),
                "model": dict(vars(config.model)),
                "seq_len": config.seq_len,
            },
        )
    wandb_logger = WandbLogger(wandb_run, enabled=not debug and wandb_run is not None)

    # Initialize local file logger for training metrics
    local_log_path = out_dir / "training_metrics.jsonl"
    local_logger = LocalLogger(local_log_path, enabled=not debug)

    allow_partial_load = config.train.allow_partial_checkpoint_load
    start_epoch, global_step, start_iter = _load_latest_checkpoint(
        out_dir,
        model,
        optimizer,
        scaler,
        device,
        allow_partial_load=allow_partial_load,
    )
    try:
        if last_step_file.exists():
            persisted = int(last_step_file.read_text().strip())
            global_step = max(global_step, persisted)
    except Exception:
        pass

    components = TrainingComponents(
        config=config,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        logger=wandb_logger,
        local_logger=local_logger,
        device=device,
        amp=amp,
        ratios=ratios,
        column_map=column_map,
        value_idx=value_idx,
        dataset=ds,
        loader=loader,
        sampler=sampler,
        total_steps=total_steps,
        out_dir=out_dir,
        last_step_file=last_step_file,
        debug=debug,
    )

    # Initialize profilers (each with burnin=1 to exclude first step)
    profiler_names = [
        "total_step",
        "data_prep",
        "progress_calc",
        "forward",
        "lr_update",
        "backward",
        "stats_update",
        "checkpoint",
        "logging",
    ]
    for name in profiler_names:
        components.profilers[name] = Profiler(burnin=1, ema_alpha=0.1)

    return components, start_epoch, global_step, start_iter
