#!/usr/bin/env python3
"""W&B-connected entry point that delegates to ``training_manager``.

This script initialises a Weights & Biases run, translates sweep configuration
values into ``training_manager`` CLI overrides, and then executes the manager's
main entry point. Structured reports produced by the manager are inserted back
into the active W&B run so sweeps can optimise against the recorded metrics.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import wandb

import training_manager


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", type=Path, default=None, help="Optional base config JSON file.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values before applying sweep settings (repeatable).",
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
        "--no-verbose",
        action="store_true",
        help="Disable stdout logging coming from training_manager.",
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mode is not None:
        os.environ["WANDB_MODE"] = args.mode

    cli_overrides = dict(training_manager.parse_key_value(item) for item in args.set)

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

    wandb.define_metric("train/final_loss", summary="min")
    wandb.define_metric("final_loss", summary="min")

    target_loss = args.target_loss
    if target_loss is None and "target_loss" in wandb_config:
        try:
            target_loss = float(wandb_config.pop("target_loss"))
        except (TypeError, ValueError):
            pass

    target_objective_values = list(args.target_objective)
    if "target_objective" in wandb_config:
        raw_obj = wandb_config.pop("target_objective")
        if isinstance(raw_obj, str):
            target_objective_values.append(raw_obj)
        elif isinstance(raw_obj, Sequence):
            for item in raw_obj:
                if item is None:
                    continue
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    target_objective_values.extend(str(x) for x in item if x is not None)
                else:
                    target_objective_values.append(str(item))
        elif raw_obj is not None:
            target_objective_values.append(str(raw_obj))

    device_name = args.device or wandb_config.pop("device", None)

    wandb_overrides = _collect_wandb_overrides(
        wandb_config,
        skip=("target_loss", "target_objective", "device"),
    )
    overrides = dict(wandb_overrides)
    overrides.update(cli_overrides)

    results_dir = Path(wandb_run.dir or ".") if wandb_run is not None else Path.cwd()
    results_dir.mkdir(parents=True, exist_ok=True)
    report_path = results_dir / "training_manager_report.json"
    results_log_path = results_dir / "training_manager_results.jsonl"

    tm_argv = []
    if args.config_json is not None:
        tm_argv.extend(["--config-json", str(args.config_json)])
    for key, value in overrides.items():
        tm_argv.extend(["--set", f"{key}={value}"])
    if target_loss is not None:
        tm_argv.extend(["--target-loss", f"{target_loss}"])
    for objective in target_objective_values:
        tm_argv.extend(["--target-objective", objective])
    if device_name is not None:
        tm_argv.extend(["--device", device_name])
    if args.no_verbose:
        tm_argv.append("--no-verbose")
    tm_argv.extend(["--time-limit-seconds", "60"])
    tm_argv.extend(["--report", str(report_path)])
    tm_argv.extend(["--results-log", str(results_log_path)])

    training_manager.main(tm_argv)

    if report_path.exists():
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
    else:
        payload = None

    if payload and isinstance(payload, dict):
        results = payload.get("results")
        if isinstance(results, list) and results:
            final_result = results[-1]
            metrics_summary = final_result.get("metrics_summary", {})
            wandb.log({
                "train/final_loss": final_result.get("final_loss"),
                "train/steps": final_result.get("steps"),
                "train/epochs_completed": final_result.get("epochs_completed"),
                "train/time_seconds": final_result.get("time_seconds"),
                "train/tokens_per_second": final_result.get("tokens_per_second"),
                "train/target_loss": final_result.get("target_loss"),
            })
            for key, value in metrics_summary.items():
                wandb.summary[f"metrics/{key}"] = value
            wandb.summary["final_loss"] = final_result.get("final_loss")
            wandb.summary["training_time_seconds"] = final_result.get("time_seconds")
            wandb.summary["tokens_per_second"] = final_result.get("tokens_per_second")
            wandb.summary["stop_reason"] = final_result.get("stop_reason")
            wandb.summary["parameter_count"] = final_result.get("parameter_count")

        summary_text = payload.get("summary")
        if isinstance(summary_text, str):
            wandb.summary["training_manager_summary"] = summary_text

        report_artifact = wandb.Artifact(f"training-manager-report-{wandb_run.id if wandb_run else 'manual'}", type="training-report")
        report_artifact.add_file(str(report_path))
        if results_log_path.exists():
            report_artifact.add_file(str(results_log_path))
        wandb.log_artifact(report_artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
