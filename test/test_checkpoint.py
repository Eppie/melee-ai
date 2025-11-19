from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.amp import GradScaler

from train.checkpoint import _load_latest_checkpoint


def _write_mismatched_checkpoint(path: Path, model: torch.nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler(enabled=False)
    state = model.state_dict()
    partial_state = {"weight": state["weight"].clone()}  # drop bias to force missing keys
    ckpt = {
        "model": partial_state,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "resume_epoch": 1,
        "resume_iter": 0,
        "global_step": 10,
    }
    torch.save(ckpt, path)


def test_load_latest_checkpoint_is_strict_by_default(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    ckpt_path = tmp_path / "strict.pt"
    _write_mismatched_checkpoint(ckpt_path, model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler(enabled=False)

    with pytest.raises(RuntimeError):
        _load_latest_checkpoint(
            tmp_path,
            model,
            optimizer,
            scaler,
            torch.device("cpu"),
        )


def test_load_latest_checkpoint_allows_partial_when_requested(tmp_path: Path) -> None:
    model = torch.nn.Linear(4, 2)
    ckpt_path = tmp_path / "partial.pt"
    _write_mismatched_checkpoint(ckpt_path, model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler(enabled=False)

    start_epoch, global_step, start_iter = _load_latest_checkpoint(
        tmp_path,
        model,
        optimizer,
        scaler,
        torch.device("cpu"),
        allow_partial_load=True,
    )

    assert (start_epoch, global_step, start_iter) == (1, 10, 0)
