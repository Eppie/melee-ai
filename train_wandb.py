#!/usr/bin/env python3
"""W&B-connected entry point for supervised training sweeps.

This script mirrors the CLI for :mod:`training_manager` but sources most
overrides from the active Weights & Biases configuration. It is intended to be
used in two modes:

* As the ``program`` for a W&B sweep/agent, where configuration values are
  injected via ``wandb.config``.
* Manually, for debugging a sweep locally with ``--mode offline`` or
  ``--mode disabled`` so results are still recorded without uploading.

The implementation delegates the heavy lifting to
``training_run.run_training_once`` to ensure behaviour stays consistent with
other orchestration tooling.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import wandb

from training_run import (
    StoppingObjective,
    TrainingRunResult,
    append_result_jsonl,
    latest_checkpoint_from_result,
    parse_key_value,
    parse_objective,
    run_training_once,
)
from utils import _resolve_device


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, default=None, help="Optional base config JSON file.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values (repeatable).",
    )
    parser.add_argument("--target-loss", type=float, default=None, help="Early-stop when loss <= target.")
    parser.add_argument(
        "--target-objective",
        action="append",
        default=[],
        metavar="METRIC{>=,<=,>,<,==}VALUE",
        help="Early-stop when the given metric objective is met.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu/cuda/mps).")
    parser.add_argument("--project", type=str, default=None, help="W&B project name.")
    parser.add_argument("--entity", type=str, default=None, help="W&B entity (team/user).")
    parser.add_argument("--group", type=str, default=None, help="Optional W&B group for the run.")
    parser.add_argument("--name", type=str, default=None, help="Optional explicit W&B run name.")
    parser.add_argument("--job-type", type=str, default=None, help="Optional W&B job type.")
    parser.add_argument("--notes", type=str, default=None, help="Free-form notes stored with the run.")
    parser.add_argument("--tags", action="append", default=[], help="Additional W&B tags (repeatable).")
    parser.add_argument(
        "--mode",
        choices=("online", "offline", "disabled"),
        default=None,
        help="Set WANDB_MODE before initialising (use 'offline' for local debugging).",
    )
    parser.add_argument("--resume", type=str, default=None, help="Optional W&B resume identifier.")
    parser.add_argument(
        "--results-log",
        type=Path,
        default=None,
        help="Optional path to stream the TrainingRunResult as JSONL.",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Disable stdout logging from the underlying training run.",
    )
    return parser


def _stringify_override(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if value is None:
        return "null"
    if isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def _coerce_for_wandb_config(overrides: Mapping[str, str]) -> Dict[str, Any]:
    coerced: Dict[str, Any] = {}
    for key, value in overrides.items():
        try:
            coerced[key] = json.loads(value)
        except json.JSONDecodeError:
            coerced[key] = value
    return coerced


def _collect_wandb_overrides(config: Mapping[str, Any], skip: Optional[Sequence[str]] = None) -> Dict[str, str]:
    skip_set = set(skip or ())
    overrides: Dict[str, str] = {}
    for key, value in config.items():
        if key in skip_set or key.startswith("_"):
            continue
        overrides[key] = _stringify_override(value)
    return overrides


def _normalise_objectives(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence):
        values: List[str] = []
        for item in raw:
            if item is None:
                continue
            values.extend(_normalise_objectives(item))
        return values
    return [str(raw)]


def _summarise_run(result: TrainingRunResult) -> str:
    target_text = f"{result.target_loss:.4f}" if result.target_loss is not None else "n/a"
    tokens_rate = (
        f"{result.tokens_per_second:,.0f}" if result.tokens_per_second is not None else "n/a"
    )
    status_bits: List[str] = []
    if result.reached_target_loss:
        status_bits.append("target met")
    if result.reached_target_objectives:
        status_bits.append("objective met")
    if result.stopped_due_to_cap:
        status_bits.append("max steps reached")
    if result.interrupted:
        status_bits.append("interrupted")
    status = ", ".join(status_bits) if status_bits else "completed"
    return (
        f"loss {result.final_loss:.4f} (target {target_text}) | steps {result.steps} | epochs {result.epochs_completed} | "
        f"time {result.time_seconds:.1f}s | tokens/s {tokens_rate} | status {status} | reason {result.stop_reason}"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode is not None:
        os.environ["WANDB_MODE"] = args.mode

    cli_overrides = dict(parse_key_value(s) for s in args.set)
    initial_data: Optional[Dict[str, Any]] = None
    if args.config_json is not None:
        with args.config_json.open("r", encoding="utf-8") as fh:
            initial_data = json.load(fh)

    init_config = _coerce_for_wandb_config(cli_overrides)
    tags = list(dict.fromkeys(args.tags))  # preserve order, remove duplicates
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        group=args.group,
        name=args.name,
        job_type=args.job_type,
        notes=args.notes,
        tags=tags or None,
        config=init_config,
        resume=args.resume,
    )

    wandb_run = run if run is not None else wandb.run
    try:
        wandb_config = dict(wandb.config)
    except Exception:
        wandb_config = {}

    target_loss = args.target_loss
    if target_loss is None and "target_loss" in wandb_config:
        try:
            target_loss = float(wandb_config.pop("target_loss"))
        except (TypeError, ValueError):
            pass

    objective_exprs: List[str] = list(args.target_objective)
    if "target_objective" in wandb_config:
        objective_exprs.extend(_normalise_objectives(wandb_config.pop("target_objective")))

    device_name = args.device or wandb_config.pop("device", None)

    wandb_overrides = _collect_wandb_overrides(
        wandb_config,
        skip=("target_loss", "target_objective", "device"),
    )
    overrides = dict(wandb_overrides)
    overrides.update(cli_overrides)

    verbose = not args.no_verbose
    objectives: List[StoppingObjective] = [parse_objective(expr) for expr in objective_exprs]
    device = _resolve_device(device_name)
    run_id = wandb_run.id if wandb_run is not None else "manual"

    results_log_path = args.results_log
    if results_log_path is not None:
        results_log_path.parent.mkdir(parents=True, exist_ok=True)

    result = run_training_once(
        run_id,
        base_initial=initial_data,
        overrides=overrides,
        target_loss=target_loss,
        objectives=objectives,
        device=device,
        verbose=verbose,
    )

    if verbose:
        print(_summarise_run(result))

    if results_log_path is not None:
        append_result_jsonl(results_log_path, result)

    if wandb_run is not None:
        wandb.log(
            {
                "train/final_loss": result.final_loss,
                "train/steps": result.steps,
                "train/epochs_completed": result.epochs_completed,
                "train/time_seconds": result.time_seconds,
                "train/tokens_per_second": result.tokens_per_second,
                "train/target_loss": target_loss,
            }
        )
        if result.per_step_losses:
            wandb.log({"train/loss_history": wandb.Histogram(result.per_step_losses)})

        for key, value in result.metrics_summary.items():
            wandb.summary[f"metrics/{key}"] = value
        wandb.summary["final_loss"] = result.final_loss
        wandb.summary["training_time_seconds"] = result.time_seconds
        wandb.summary["tokens_per_second"] = result.tokens_per_second
        wandb.summary["stop_reason"] = result.stop_reason
        wandb.summary["parameter_count"] = result.parameter_count

        run_dir = Path(wandb_run.dir or ".")
        result_path = run_dir / "training_result.json"
        result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        artifact = wandb.Artifact(f"training-result-{run_id}", type="training-result")
        artifact.add_file(str(result_path))
        wandb.log_artifact(artifact)

        ckpt_path = latest_checkpoint_from_result(result)
        if ckpt_path is not None and ckpt_path.exists():
            ckpt_artifact = wandb.Artifact(f"checkpoint-{run_id}", type="model")
            ckpt_artifact.add_file(str(ckpt_path))
            wandb.log_artifact(ckpt_artifact)

    wandb.finish()


if __name__ == "__main__":
    main()


