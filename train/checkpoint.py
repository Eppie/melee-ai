from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
from torch.amp import GradScaler
from torch.optim import Optimizer

from config.gpt_config import _schema_feature_dims
from train.gradients import _move_optimizer_state_to_device
from train.components import TrainingComponents
from train.validation import maybe_run_validation
from utils import match_state_dict_keys
import torch.nn as nn


def _sorted_checkpoint_paths(directory: Path) -> List[Path]:
    """Return checkpoint files ordered from newest to oldest with a concrete example.

    Example:
        If ``directory`` contains ``[run1.pt, run2.pt]`` where ``run1.pt`` was modified at
        ``12:00`` and ``run2.pt`` at ``12:05``, calling ``_sorted_checkpoint_paths`` yields the list
        ``[run2.pt, run1.pt]``. The function gathers every ``*.pt`` file, inspects the modification
        timestamps via ``Path.stat().st_mtime``, sorts in descending order, and returns the paths in
        that precise sequence.

    Args:
        directory: Folder that potentially contains checkpoint ``.pt`` files.

    Returns:
        List of checkpoint paths sorted by modification time (newest first).
    """
    checkpoints = [p for p in directory.glob("*.pt") if p.is_file()]
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints


def _prune_checkpoints(directory: Path, keep: int = 10) -> None:
    """Delete older checkpoints once more than ``keep`` files exist.

    Example:
        When ``directory`` holds ``[epoch1.pt, epoch2.pt, epoch3.pt]`` ordered newest to oldest and
        ``keep`` is ``2``, the helper first calls :func:`_sorted_checkpoint_paths` to obtain the list
        ``[epoch3.pt, epoch2.pt, epoch1.pt]``. It then iterates over every entry beyond the first two
        (in this case ``epoch1.pt``) and unlinks it, leaving only the most recent checkpoints on
        disk.

    Args:
        directory: Folder that stores checkpoint ``.pt`` files.
        keep: Maximum number of recent checkpoints to retain.
    """
    if keep <= 0:
        return
    checkpoints = _sorted_checkpoint_paths(directory)
    for old_ckpt in checkpoints[keep:]:
        try:
            old_ckpt.unlink()
        except OSError as err:
            print(f"Warning: failed to remove old checkpoint {old_ckpt}: {err}")


def _latest_checkpoint(directory: Path) -> Optional[Path]:
    """Find the newest checkpoint in ``directory`` if one exists.

    Example:
        Given a folder containing ``ckpt_10.pt`` and ``ckpt_11.pt`` with modification times of
        ``10:00`` and ``10:05`` respectively, ``_latest_checkpoint`` expands the path (handling
        ``~``), filters to files ending in ``.pt``, sorts them by timestamp, and returns the
        ``Path`` for ``ckpt_11.pt``. If no checkpoint files are present, it returns ``None``.

    Args:
        directory: Directory to search for checkpoints.

    Returns:
        Path to the most recent checkpoint or ``None`` if no files are found.
    """
    directory = directory.expanduser()
    if not directory.exists():
        return None
    candidates = [p for p in directory.glob("*.pt") if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def load_config_from_checkpoint(
    checkpoint_path: Union[Path, str],
) -> Optional[Dict[str, Any]]:
    """Load the config dictionary from a checkpoint file.

    Args:
        checkpoint_path: Path to a checkpoint file.

    Returns:
        The config dictionary stored in the checkpoint, or None if no config is found.

    Example:
        >>> config_dict = load_config_from_checkpoint("checkpoints/model.pt")
        >>> if config_dict:
        ...     config = Config.model_validate(config_dict)
    """
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.exists():
        return None

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    return ckpt.get("config")


def load_config_from_latest_checkpoint(
    directory: Union[Path, str],
) -> Optional[Dict[str, Any]]:
    """Load the config dictionary from the latest checkpoint in a directory.

    This function is useful for restoring the full configuration when resuming
    training. Call this before initializing the global config to ensure the
    model and training parameters match the checkpoint.

    Args:
        directory: Directory containing checkpoint files.

    Returns:
        The config dictionary from the latest checkpoint, or None if no
        checkpoint exists or the checkpoint has no config.

    Example:
        >>> from config.config import Config, init_config
        >>> config_dict = load_config_from_latest_checkpoint("checkpoints/")
        >>> if config_dict:
        ...     # Use checkpoint config as base, with optional overrides
        ...     config = Config.model_validate(config_dict)
        ... else:
        ...     config = init_config()
    """
    directory = Path(directory).expanduser()
    latest = _latest_checkpoint(directory)
    if latest is None:
        return None

    return load_config_from_checkpoint(latest)


def _load_latest_checkpoint(
    directory: Path,
    model: torch.nn.Module,
    optimizer: Optimizer,
    scaler: GradScaler,
    device: torch.device,
    *,
    allow_partial_load: bool = False,
) -> Tuple[int, int, int]:
    """Load the newest checkpoint and restore model, optimizer, and scaler state.

    Example:
        Imagine ``directory`` contains ``epoch_10.pt`` representing epoch 10 with
        ``{"model": ..., "optimizer": ..., "scaler": ..., "global_step": 1280}``. When invoked,
        the helper selects that file, calls :func:`torch.load`, and feeds the stored state dictionaries
        into ``model.load_state_dict`` and ``optimizer.load_state_dict``. After adjusting for legacy
        metadata, it returns ``(10, 1280, 0)`` indicating training should resume at epoch 10,
        global step 1280, and iteration 0. If the directory is empty, ``(0, 0, 0)`` is returned
        instead.

    Args:
        directory: Folder containing checkpoints.
        model: Model instance whose parameters should be restored.
        optimizer: Optimizer instance whose state should be restored.
        scaler: Gradient scaler for mixed precision training.
        device: Device to move optimizer state tensors onto.

    Returns:
        Tuple ``(start_epoch, global_step, start_iter)`` describing where to resume training.
    """
    checkpoints = _sorted_checkpoint_paths(directory)
    if not checkpoints:
        return 0, 0, 0

    latest = checkpoints[0]
    print(f"Resuming from checkpoint: {latest}")
    ckpt = torch.load(latest, map_location="cpu")

    model_state = ckpt.get("model")
    if model_state:
        # Handle torch.compile() prefix mismatch (both directions)
        model_state = match_state_dict_keys(model_state, model)

        original_state = model_state.copy()

        try:
            # Always try strict loading first
            try:
                model.load_state_dict(model_state, strict=True)
            except RuntimeError as e:
                # Get expected keys
                model_keys = set(model.state_dict().keys())
                ckpt_keys = set(model_state.keys())
                missing_keys = list(model_keys - ckpt_keys)
                unexpected_keys = list(ckpt_keys - model_keys)

                # Check for shape mismatches in error message
                error_msg = str(e)

                incompatible = model.load_state_dict(filtered_state, strict=False)
                missing = list(getattr(incompatible, "missing_keys", ()))
                unexpected = list(getattr(incompatible, "unexpected_keys", ()))

                # If validation passed, show what was loaded
                if allow_partial_load:
                    if missing:
                        preview = ", ".join(missing[:5])
                        more = "..." if len(missing) > 5 else ""
                        print(
                            f"Checkpoint is missing {len(missing)} parameter(s); "
                            f"initialising from current model weights: {preview}{more}"
                        )
                    if unexpected:
                        preview = ", ".join(unexpected[:5])
                        more = "..." if len(unexpected) > 5 else ""
                        print(
                            f"Checkpoint has {len(unexpected)} unexpected parameter(s); ignoring: {preview}{more}"
                        )
        except RuntimeError as err:
            raise RuntimeError(
                "Checkpoint parameters do not match the current model. "
                "Set train.allow_partial_checkpoint_load=True if this is intentional."
            ) from err

    opt_state = ckpt.get("optimizer")
    if opt_state:
        optimizer.load_state_dict(opt_state)
        _move_optimizer_state_to_device(optimizer, device)

    scaler_state = ckpt.get("scaler")
    if scaler_state:
        scaler.load_state_dict(scaler_state)

    resume_epoch = ckpt["resume_epoch"]
    resume_iter = ckpt["resume_iter"]

    start_epoch = int(resume_epoch)
    start_iter = max(int(resume_iter), 0)
    global_step = int(ckpt["global_step"])
    return max(start_epoch, 0), max(global_step, 0), start_iter


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: Union[Dict[str, Any], "Config"],
    optimizer: Optional[Optimizer] = None,
    scaler: Optional[GradScaler] = None,
    epoch: int = 0,
    global_step: int = 0,
    **kwargs,
) -> None:
    """Persist the model state and optional training metadata to ``path``.

    Example:
        Calling ``save_checkpoint(Path("checkpoints/epoch_5.pt"), model, config, optimizer, epoch=5, global_step=640)``
        produces a dictionary containing at least the keys ``"model"``, ``"epoch"``,
        ``"resume_epoch"``, ``"resume_iter"``, and ``"global_step"``. The helper ensures the parent
        directory exists, then uses :func:`torch.save` to serialize the dictionary. Reloading the file
        later will reproduce the exact state, demonstrating how inputs are collected and written to
        disk step by step.

    Args:
        path: Destination path for the checkpoint file.
        model: Model whose parameters should be saved.
        config: Configuration object or dictionary to embed in the checkpoint.
            If a Config object is passed, it will be serialized to a dictionary.
        optimizer: Optional optimizer whose state should also be serialized.
        scaler: Optional gradient scaler to persist for mixed precision runs.
        epoch: Epoch number to record.
        global_step: Global training step to record.
        **kwargs: Additional key-value pairs to merge into the saved dictionary.
    """
    # Serialize Config object to dict if needed
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    elif hasattr(config, "model_dump"):
        config_dict = config.model_dump()
    else:
        config_dict = config

    ckpt = {
        "model": model.state_dict(),
        "epoch": epoch,
        "resume_epoch": epoch,
        "resume_iter": 0,
        "global_step": global_step,
        "config": config_dict,
    }

    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()

    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()

    # Add any extra kwargs
    ckpt.update(kwargs)

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def maybe_checkpoint_batch(
    components: TrainingComponents,
    epoch: int,
    iteration_index: int,
    completed_batches: int,
    global_step: int,
) -> None:
    """Save a checkpoint mid-epoch on a fixed interval."""
    if iteration_index % 5000 != 0:
        return
    ckpt_path = (
        components.out_dir / f"model_ep{epoch + 1:03d}_{completed_batches:06d}.pt"
    )
    save_checkpoint(
        path=ckpt_path,
        model=components.model,
        config=components.config,
        optimizer=components.optimizer,
        scaler=components.scaler,
        epoch=epoch + 1,
        global_step=global_step,
        resume_epoch=epoch,
        resume_iter=completed_batches,
        iteration=completed_batches,
    )
    _prune_checkpoints(components.out_dir, keep=10)

    # Run validation and log metrics to wandb
    maybe_run_validation(components, global_step)


def maybe_checkpoint_epoch(
    components: TrainingComponents,
    epoch: int,
    global_step: int,
) -> None:
    """Persist a checkpoint at the end of an epoch when requested."""
    ckpt_path = components.out_dir / f"model_ep{epoch + 1:03d}_000000.pt"
    save_checkpoint(
        path=ckpt_path,
        model=components.model,
        config=components.config,
        optimizer=components.optimizer,
        scaler=components.scaler,
        epoch=epoch + 1,
        global_step=global_step,
        resume_epoch=epoch + 1,
        resume_iter=0,
        iteration=0,
    )
    _prune_checkpoints(components.out_dir, keep=10)
    components.last_step_file.write_text(str(global_step))

    # Run validation and log metrics to wandb
    maybe_run_validation(components, global_step)
