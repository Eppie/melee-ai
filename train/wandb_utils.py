"""Wandb integration utilities for optional logging."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # Import lazily so training can proceed without wandb installed
    import wandb

    WANDB_AVAILABLE = True
except Exception:  # pragma: no cover - wandb missing in some environments
    wandb = None  # type: ignore[assignment]
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
    """Initialise a Weights & Biases run, persisting the run ID for automatic resume.

    Example:
        Suppose ``run_dir`` already contains ``wandb_run_id.txt`` with the value ``"abc123"``.
        Calling ``init_wandb`` with ``config.project="demo"`` and ``config.mode="online"`` loads the
        stored ID, sets ``WANDB_MODE=online`` in the environment, and calls ``wandb.init(project="demo",
        dir=str(run_dir), id="abc123", resume="allow")``. The returned run object is then cached so
        subsequent launches reuse the same dashboard entry.

    Args:
        config: Wandb configuration describing project metadata.
        run_dir: Directory where wandb files (including run ID) are stored.
        hyperparameters: Optional dictionary logged to wandb's config section.

    Returns:
        Wandb run object or ``None`` if wandb is disabled.
    """
    if config.mode == "disabled" or not WANDB_AVAILABLE:
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
    """Terminate the active wandb run if logging is enabled.

    Example:
        After a run created by :func:`init_wandb`, calling ``finish_wandb()`` invokes
        ``wandb.finish()`` inside a ``try`` block. If wandb was never initialised or raised an
        exception, the call is skipped, leaving the application unharmed. This demonstrates the
        defensive behaviour against missing dependencies.
    """
    if wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass


class WandbLogger:
    """Optional wandb logger with no-op behavior. Provides a consistent interface.
    """

    def __init__(self, wandb_run: Optional[Any] = None, enabled: bool = True):
        """Wrap a wandb run with convenience methods that degrade to no-ops.

        Example:
            Creating ``WandbLogger(run, enabled=True)`` stores ``run`` and sets ``self.enabled`` to
            ``True``. Passing ``enabled=False`` forces ``self.enabled`` to ``False`` even when a run is
            provided, making subsequent :meth:`log_metrics` calls skip wandb entirely. This example
            highlights how the constructor decides whether logging should occur.

        Args:
            wandb_run: Wandb run object (from :func:`wandb.init`).
            enabled: Whether logging should be performed.
        """
        self.run = wandb_run
        self.enabled = enabled and self.run is not None

    def log_metrics(
        self, metrics: Dict[str, float], step: int, commit: bool = True
    ) -> None:
        """Send scalar metrics to wandb when logging is enabled.

        Example:
            With ``metrics={"loss": 0.5}`` and ``step=100``, calling ``log_metrics`` executes
            ``wandb.log({"loss": 0.5}, step=100, commit=True)``. If ``self.enabled`` is ``False`` the
            function returns immediately, illustrating the guard that prevents wandb usage when the
            dependency is absent or disabled.

        Args:
            metrics: Dictionary of metric names to values.
            step: Global step number associated with the metrics.
            commit: Whether to commit the log immediately (pushing to the server).
        """
        if not self.enabled or self.run is None:
            return

        try:
            wandb.log(metrics, step=step, commit=commit)
        except Exception:
            pass

    def log_gradients(self, grad_stats: Dict[str, float], step: int) -> None:
        """Prefix gradient statistics with ``gradients/`` and log them via :meth:`log_metrics`.

        Example:
            Given ``grad_stats={"total_norm": 3.2}`` and ``step=50``, the method first builds
            ``{"gradients/total_norm": 3.2}`` and then calls ``log_metrics(..., commit=False)`` so the
            gradient entry is batched with other logs. If logging is disabled the method exits without
            modification. The example shows the exact transformation of keys and subsequent logging.

        Args:
            grad_stats: Dictionary of gradient statistics.
            step: Global step number.
        """
        if not self.enabled:
            return

        # Prefix with "gradients/" for organization
        prefixed = {f"gradients/{k}": v for k, v in grad_stats.items()}
        self.log_metrics(prefixed, step, commit=False)
