from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer

from train.gradients import _move_optimizer_state_to_device


def _sorted_checkpoint_paths(directory: Path) -> List[Path]:
    checkpoints = [p for p in directory.glob("*.pt") if p.is_file()]
    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints


def _prune_checkpoints(directory: Path, keep: int = 10) -> None:
    if keep <= 0:
        return
    checkpoints = _sorted_checkpoint_paths(directory)
    for old_ckpt in checkpoints[keep:]:
        try:
            old_ckpt.unlink()
        except OSError as err:
            print(f"Warning: failed to remove old checkpoint {old_ckpt}: {err}")


def _latest_checkpoint(directory: Path) -> Optional[Path]:
    directory = directory.expanduser()
    if not directory.exists():
        return None
    candidates = [p for p in directory.glob('*.pt') if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_latest_checkpoint(
        directory: Path,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scaler: GradScaler,
        device: torch.device,
) -> Tuple[int, int, int]:
    checkpoints = _sorted_checkpoint_paths(directory)
    if not checkpoints:
        return 0, 0, 0

    latest = checkpoints[0]
    print(f"Resuming from checkpoint: {latest}")
    ckpt = torch.load(latest, map_location="cpu")

    model_state = ckpt.get("model")
    if model_state:
        incompatible = model.load_state_dict(model_state, strict=False)
        missing = list(getattr(incompatible, "missing_keys", ()))
        unexpected = list(getattr(incompatible, "unexpected_keys", ()))
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

    opt_state = ckpt.get("optimizer")
    if opt_state:
        optimizer.load_state_dict(opt_state)
        _move_optimizer_state_to_device(optimizer, device)

    scaler_state = ckpt.get("scaler")
    if scaler_state:
        scaler.load_state_dict(scaler_state)

    resume_epoch = ckpt.get("resume_epoch", None)
    if resume_epoch is None:
        raw_epoch = int(ckpt.get("epoch", 0))
        resume_epoch = raw_epoch
        resume_iter = ckpt.get("resume_iter", ckpt.get("iteration", 0))
        if "resume_iter" not in ckpt and "iteration" not in ckpt:
            # Legacy checkpoints stored the *next* epoch to run. Adjust so we resume from the
            # previous epoch and start at the beginning of that epoch.
            if raw_epoch > 0:
                resume_epoch = raw_epoch - 1
            resume_iter = 0
    else:
        resume_iter = ckpt.get("resume_iter", 0)

    start_epoch = int(resume_epoch)
    start_iter = max(int(resume_iter), 0)
    global_step = int(ckpt.get("global_step", 0))
    return max(start_epoch, 0), max(global_step, 0), start_iter

def save_checkpoint(
        path: Path,
        model: torch.nn.Module,
        optimizer: Optional[Optimizer] = None,
        scaler: Optional[GradScaler] = None,
        epoch: int = 0,
        global_step: int = 0,
        config: Optional[dict] = None,
        **kwargs
) -> None:
    """Save model checkpoint with optional optimizer and scaler state."""
    ckpt = {
        "model": model.state_dict(),
        "epoch": epoch,
        "resume_epoch": epoch,
        "resume_iter": 0,
        "global_step": global_step,
    }
    
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()
    
    if config is not None:
        ckpt["config"] = config
    
    # Add any extra kwargs
    ckpt.update(kwargs)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
