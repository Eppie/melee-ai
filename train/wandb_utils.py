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


# TODO: Make proper use of the code here
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
        subsequent launches reuse the same dashboard entry. If wandb is unavailable, the function
        simply returns ``None``, illustrating both control-flow branches.

    Args:
        config: Wandb configuration describing project metadata.
        run_dir: Directory where wandb files (including run ID) are stored.
        hyperparameters: Optional dictionary logged to wandb's config section.

    Returns:
        Wandb run object or ``None`` if wandb is unavailable or disabled.
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
    """Terminate the active wandb run if logging is enabled.

    Example:
        After a run created by :func:`init_wandb`, calling ``finish_wandb()`` invokes
        ``wandb.finish()`` inside a ``try`` block. If wandb was never initialised or raised an
        exception, the call is skipped, leaving the application unharmed. This demonstrates the
        defensive behaviour against missing dependencies.
    """
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
        self.run = wandb_run if WANDB_AVAILABLE else None
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

    def log_loss_components(self, losses: Dict[str, float], step: int) -> None:
        """Record individual loss components under the ``loss/`` namespace.

        Example:
            For ``losses={"main": 0.3, "buttons": 0.2}`` the method constructs
            ``{"loss/main": 0.3, "loss/buttons": 0.2}`` and invokes :meth:`log_metrics` with
            ``commit=False``. This allows callers to combine loss logging with other statistics within
            the same wandb step.

        Args:
            losses: Dictionary of loss component names to values.
            step: Global step number.
        """
        if not self.enabled:
            return

        # Prefix with "loss/" for organization
        prefixed = {f"loss/{k}": v for k, v in losses.items()}
        self.log_metrics(prefixed, step, commit=False)

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Persist hyperparameters in ``wandb.config`` for reproducibility.

        Example:
            When ``params={"lr": 0.001, "batch_size": 64}``, the method iterates over the dictionary
            and assigns ``wandb.config["lr"] = 0.001`` and ``wandb.config["batch_size"] = 64``. If
            logging is disabled, the loop is skipped entirely. This showcases the mapping from input
            dictionary to wandb's configuration namespace.

        Args:
            params: Dictionary of hyperparameter names to values.
        """
        if not self.enabled or self.run is None:
            return

        try:
            for key, value in params.items():
                wandb.config[key] = value
        except Exception:
            pass

    def should_log_this_step(self, step: int, frequency: int = 10) -> bool:
        """Return ``True`` when ``step`` is a multiple of ``frequency`` and logging is enabled.

        Example:
            With ``frequency=5`` and ``self.enabled=True``, calling ``should_log_this_step(10)`` returns
            ``True`` because ``10 % 5 == 0``. Calling ``should_log_this_step(11)`` returns ``False``.
            If ``self.enabled`` were ``False`` both calls would return ``False``. This demonstrates the
            precise boolean condition used to decide whether to log.

        Args:
            step: Current step number.
            frequency: Log every ``N`` steps.

        Returns:
            ``True`` if logging should happen this step, ``False`` otherwise.
        """
        return self.enabled and (step % frequency == 0)

    def watch_model(
        self, model: Any, log: str = "gradients", log_freq: int = 100
    ) -> None:
        """Register a model with wandb's watch API when logging is active.

        Example:
            ``watch_model(model, log="all", log_freq=50)`` triggers ``wandb.watch`` with the same
            arguments, enabling parameter and gradient histograms in the UI. If wandb is disabled the
            method returns without side effects, illustrating the guard that prevents unnecessary API
            calls.

        Args:
            model: Model to watch.
            log: What to log (``"gradients"``, ``"parameters"``, or ``"all"``).
            log_freq: Frequency with which wandb should log model statistics.
        """
        if not self.enabled or self.run is None:
            return

        try:
            wandb.watch(model, log=log, log_freq=log_freq)
        except Exception:
            pass
