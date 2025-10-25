"""Wandb integration utilities for optional logging."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional wandb import
try:
    import wandb  # type: ignore

    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False


@dataclass
class WandbConfig:
    """Configuration for wandb initialization."""

    project: str = "melee-ai"
    entity: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[List[str]] = None
    mode: Optional[str] = None  # "online", "offline", "disabled"
    resume_id: Optional[str] = None
    job_type: Optional[str] = None
    notes: Optional[str] = None


def init_wandb(
    config: WandbConfig,
    run_dir: Path,
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Initialize wandb with persistent run ID and resume support.

    Args:
        config: Wandb configuration
        run_dir: Directory for run outputs (for storing run ID)
        hyperparameters: Optional hyperparameters to log

    Returns:
        wandb run object or None if wandb unavailable/disabled
    """
    if not WANDB_AVAILABLE or config.mode == "disabled":
        return None

    try:
        # Set mode if specified
        if config.mode is not None:
            os.environ["WANDB_MODE"] = config.mode

        # Prepare init kwargs
        init_kwargs: Dict[str, Any] = {
            "project": config.project,
            "dir": str(run_dir),
        }

        if config.entity is not None:
            init_kwargs["entity"] = config.entity
        if config.name is not None:
            init_kwargs["name"] = config.name
        if config.group is not None:
            init_kwargs["group"] = config.group
        if config.job_type is not None:
            init_kwargs["job_type"] = config.job_type
        if config.notes is not None:
            init_kwargs["notes"] = config.notes
        if config.tags:
            init_kwargs["tags"] = config.tags
        if hyperparameters:
            init_kwargs["config"] = hyperparameters

        # Handle run ID persistence for resume
        run_id_file = run_dir / "wandb_run_id.txt"
        resume_run_id = config.resume_id

        # 1) Try stored run ID
        if not resume_run_id and run_id_file.exists():
            try:
                resume_run_id = run_id_file.read_text().strip() or None
            except Exception:
                pass

        # 2) Try environment variables
        if not resume_run_id:
            resume_run_id = os.environ.get("WANDB_RUN_ID") or os.environ.get(
                "WANDB_RESUME_ID"
            )

        # 3) Try to discover from latest-run symlink
        if not resume_run_id:
            try:
                latest = (run_dir / "wandb" / "latest-run").resolve(strict=True)
                meta_path = latest / "files" / "wandb-metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    resume_run_id = meta.get("id") or meta.get("run_id")
                if not resume_run_id:
                    # Parse from directory name: run-YYYYMMDD_HHMMSS-<id>
                    base = latest.name
                    if "-" in base:
                        resume_run_id = base.split("-")[-1]
            except Exception:
                pass

        # Set resume parameters if we have a run ID
        if resume_run_id:
            init_kwargs["id"] = resume_run_id
            init_kwargs["resume"] = "allow"

        # Initialize wandb
        run = wandb.init(**init_kwargs)

        # Save run ID for future resume
        if run is not None and hasattr(run, "id"):
            try:
                run_id_file.parent.mkdir(parents=True, exist_ok=True)
                run_id_file.write_text(str(run.id))
            except Exception:
                pass

        return run

    except Exception as e:
        print(f"Warning: wandb initialization failed: {e}")
        return None


def finish_wandb() -> None:
    """Finish wandb run gracefully."""
    if WANDB_AVAILABLE and wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass


class WandbLogger:
    """Optional wandb logger with no-op behavior when unavailable.

    Provides a consistent interface whether wandb is available or not.
    """

    def __init__(self, wandb_run: Optional[Any] = None, enabled: bool = True):
        """Initialize logger.

        Args:
            wandb_run: Wandb run object (from wandb.init())
            enabled: Whether logging is enabled
        """
        self.run = wandb_run if WANDB_AVAILABLE else None
        self.enabled = enabled and self.run is not None

    def log_metrics(
        self, metrics: Dict[str, float], step: int, commit: bool = True
    ) -> None:
        """Log metrics to wandb.

        Args:
            metrics: Dictionary of metric names to values
            step: Global step number
            commit: Whether to commit the log (push to server)
        """
        if not self.enabled or self.run is None:
            return

        try:
            wandb.log(metrics, step=step, commit=commit)
        except Exception:
            pass

    def log_gradients(self, grad_stats: Dict[str, float], step: int) -> None:
        """Log gradient statistics to wandb.

        Args:
            grad_stats: Dictionary of gradient statistics
            step: Global step number
        """
        if not self.enabled:
            return

        # Prefix with "gradients/" for organization
        prefixed = {f"gradients/{k}": v for k, v in grad_stats.items()}
        self.log_metrics(prefixed, step, commit=False)

    def log_loss_components(self, losses: Dict[str, float], step: int) -> None:
        """Log loss components to wandb.

        Args:
            losses: Dictionary of loss component names to values
            step: Global step number
        """
        if not self.enabled:
            return

        # Prefix with "loss/" for organization
        prefixed = {f"loss/{k}": v for k, v in losses.items()}
        self.log_metrics(prefixed, step, commit=False)

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to wandb config.

        Args:
            params: Dictionary of hyperparameter names to values
        """
        if not self.enabled or self.run is None:
            return

        try:
            for key, value in params.items():
                wandb.config[key] = value
        except Exception:
            pass

    def should_log_this_step(self, step: int, frequency: int = 10) -> bool:
        """Check if we should log at this step based on frequency.

        Args:
            step: Current step number
            frequency: Log every N steps

        Returns:
            True if we should log
        """
        return self.enabled and (step % frequency == 0)

    def watch_model(
        self, model: Any, log: str = "gradients", log_freq: int = 100
    ) -> None:
        """Watch model for gradient/parameter tracking.

        Args:
            model: Model to watch
            log: What to log ("gradients", "parameters", "all")
            log_freq: How often to log
        """
        if not self.enabled or self.run is None:
            return

        try:
            wandb.watch(model, log=log, log_freq=log_freq)
        except Exception:
            pass
