"""Utilities for configuring the training loop entrypoints."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from pprint import pformat
from typing import Dict, Sequence, Tuple

import torch
from torch.amp import GradScaler
from torch.amp.autocast_mode import is_autocast_available

from column_map import ColumnMap
from config import get_config
from model.nano_gpt import GPT
from train.batch_utils import SampleWeightRatios
from train.checkpoint import _load_latest_checkpoint
from train.components import AMPContext, TrainingComponents
from train.wandb_utils import WandbConfig, WandbLogger, init_wandb
from utils import _resolve_device


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
    print("Resolved training configuration:")
    print(pformat(printable_config, indent=2, width=100))


def configure_amp(config, device: torch.device) -> AMPContext:
    """Configures and reports the AMP (Automatic Mixed Precision) context."""
    if device.type in ("cuda", "mps") and is_autocast_available(device.type):
        device_type = device.type
    else:
        device_type = "cpu"

    amp_context = AMPContext(
        enabled=config.train.use_amp,
        device_type=device_type,
        dtype=torch.float16,
    )

    print(f"Using PyTorch {torch.__version__}")
    if amp_context.enabled:
        backend = amp_context.device_type.upper()
        print(f"AMP enabled with float16 on {backend} backend")
    else:
        print(
            f"AMP requested but disabled for device '{device.type}';"
            " falling back to full precision."
        )

    return amp_context


def build_optimizer(model: GPT, config) -> torch.optim.Optimizer:
    """Builds the AdamW optimizer from the training configuration."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        betas=config.train.betas,
        weight_decay=config.train.weight_decay,
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
    scaler_device = amp.device_type if amp.enabled else "cpu"
    scaler = GradScaler(device=scaler_device, enabled=amp.enabled)

    steps_per_epoch = math.ceil(len(loader))
    total_steps = config.train.epochs * steps_per_epoch

    out_dir = Path(config.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    last_step_file = out_dir / "last_step.txt"

    wandb_run = None
    if not debug:
        wandb_cfg = WandbConfig(
            project=getattr(config.train, "wandb_project", "melee-ai"),
            name=getattr(config.train, "run_name", None),
            mode=getattr(config.train, "wandb_mode", "online"),
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
    logger = WandbLogger(wandb_run, enabled=not debug and wandb_run is not None)

    start_epoch, global_step, start_iter = _load_latest_checkpoint(
        out_dir, model, optimizer, scaler, device
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
        logger=logger,
        device=device,
        amp=amp,
        ratios=ratios,
        column_map=column_map,
        value_idx=value_idx,
        loader=loader,
        sampler=sampler,
        total_steps=total_steps,
        out_dir=out_dir,
        last_step_file=last_step_file,
        debug=debug,
    )

    return components, start_epoch, global_step, start_iter
