from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.amp import GradScaler

from config.config import (
    Config,
    reset_config,
    init_config_from_checkpoint,
    set_config,
)
from train.checkpoint import (
    _load_latest_checkpoint,
    save_checkpoint,
    load_config_from_checkpoint,
    load_config_from_latest_checkpoint,
)


def _write_mismatched_checkpoint(path: Path, model: torch.nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler(enabled=False)
    state = model.state_dict()
    partial_state = {
        "weight": state["weight"].clone()
    }  # drop bias to force missing keys
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


@pytest.fixture(autouse=True)
def cleanup_global_config():
    """Reset global config before and after each test."""
    reset_config()
    yield
    reset_config()


def test_save_checkpoint_saves_full_config(tmp_path: Path) -> None:
    """Test that save_checkpoint saves the full Config object."""
    model = torch.nn.Linear(4, 2)
    config = Config()

    # Modify some config values to verify they're saved
    config.train.lr = 1e-5
    config.model.n_layer = 12
    config.seq_len = 128

    ckpt_path = tmp_path / "full_config.pt"
    save_checkpoint(
        path=ckpt_path,
        model=model,
        config=config,
        epoch=5,
        global_step=1000,
    )

    # Load and verify the config was saved correctly
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert "config" in ckpt
    config_dict = ckpt["config"]

    # Verify nested config values
    assert config_dict["train"]["lr"] == 1e-5
    assert config_dict["model"]["n_layer"] == 12
    assert config_dict["seq_len"] == 128
    # Verify other sub-configs are present
    assert "zarr" in config_dict
    assert "loss_weights" in config_dict
    assert "features" in config_dict


def test_load_config_from_checkpoint(tmp_path: Path) -> None:
    """Test loading config from a specific checkpoint file."""
    model = torch.nn.Linear(4, 2)
    config = Config()
    config.train.batch_size = 256
    config.model.n_head = 16

    ckpt_path = tmp_path / "config_test.pt"
    save_checkpoint(path=ckpt_path, model=model, config=config)

    loaded_config_dict = load_config_from_checkpoint(ckpt_path)
    assert loaded_config_dict is not None
    assert loaded_config_dict["train"]["batch_size"] == 256
    assert loaded_config_dict["model"]["n_head"] == 16


def test_load_config_from_checkpoint_nonexistent_file(tmp_path: Path) -> None:
    """Test that loading from nonexistent file returns None."""
    result = load_config_from_checkpoint(tmp_path / "nonexistent.pt")
    assert result is None


def test_load_config_from_latest_checkpoint(tmp_path: Path) -> None:
    """Test loading config from the latest checkpoint in a directory."""
    model = torch.nn.Linear(4, 2)

    # Create first checkpoint
    config1 = Config()
    config1.train.lr = 1e-4
    save_checkpoint(path=tmp_path / "ckpt1.pt", model=model, config=config1)

    import time

    time.sleep(0.01)  # Ensure different mtime

    # Create second (newer) checkpoint
    config2 = Config()
    config2.train.lr = 1e-5
    save_checkpoint(path=tmp_path / "ckpt2.pt", model=model, config=config2)

    # Should load from ckpt2 (newest)
    loaded_config_dict = load_config_from_latest_checkpoint(tmp_path)
    assert loaded_config_dict is not None
    assert loaded_config_dict["train"]["lr"] == 1e-5


def test_load_config_from_latest_checkpoint_empty_dir(tmp_path: Path) -> None:
    """Test that loading from empty directory returns None."""
    result = load_config_from_latest_checkpoint(tmp_path)
    assert result is None


def test_init_config_from_checkpoint(tmp_path: Path) -> None:
    """Test initializing global config from a checkpoint."""
    model = torch.nn.Linear(4, 2)
    config = Config()
    config.train.lr = 2e-4
    config.model.dropout = 0.1
    config.seq_len = 512

    ckpt_path = tmp_path / "init_test.pt"
    save_checkpoint(path=ckpt_path, model=model, config=config)

    # Initialize global config from checkpoint
    loaded = init_config_from_checkpoint(ckpt_path)

    assert loaded.train.lr == 2e-4
    assert loaded.model.dropout == 0.1
    assert loaded.seq_len == 512


def test_init_config_from_checkpoint_with_overrides(tmp_path: Path) -> None:
    """Test that CLI overrides work on checkpoint config."""
    model = torch.nn.Linear(4, 2)
    config = Config()
    config.train.lr = 1e-4
    config.train.epochs = 10

    ckpt_path = tmp_path / "override_test.pt"
    save_checkpoint(path=ckpt_path, model=model, config=config)

    # Load with overrides
    loaded = init_config_from_checkpoint(
        ckpt_path,
        overrides={"train.lr": "5e-5", "train.epochs": "20"},
    )

    # Overrides should take effect
    assert loaded.train.lr == 5e-5
    assert loaded.train.epochs == 20


def test_init_config_from_checkpoint_nonexistent(tmp_path: Path) -> None:
    """Test that loading from nonexistent checkpoint raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        init_config_from_checkpoint(tmp_path / "nonexistent.pt")


def test_init_config_from_checkpoint_missing_config(tmp_path: Path) -> None:
    """Test that loading checkpoint without config raises ValueError."""
    # Create a checkpoint without config
    ckpt_path = tmp_path / "no_config.pt"
    torch.save({"model": {}, "epoch": 1, "global_step": 100}, ckpt_path)

    with pytest.raises(ValueError, match="does not contain a config"):
        init_config_from_checkpoint(ckpt_path)


def test_config_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    """Test that saving and loading preserves all config fields."""
    from config.config import TrainConfig, GPTConfig, LossConfig, RLConfig, PPOConfig

    model = torch.nn.Linear(4, 2)

    # Create config with modified values (frozen=False by default in Config model_config)
    original = Config(
        seq_len=384,
        train=TrainConfig(
            lr=3e-4,
            batch_size=64,
            warmup_steps=5000,
        ),
        model=GPTConfig(
            n_layer=6,
            n_head=4,
            n_embd=256,
            dropout=0.05,
        ),
        loss_weights=LossConfig(
            main_change=3.0,
            button_z=15.0,
        ),
        rl=RLConfig(
            gamma=0.99,
        ),
        ppo=PPOConfig(
            clip_ratio=0.1,
        ),
    )

    ckpt_path = tmp_path / "round_trip.pt"
    save_checkpoint(path=ckpt_path, model=model, config=original)

    loaded = init_config_from_checkpoint(ckpt_path, freeze=False)

    # Verify all modified fields
    assert loaded.seq_len == 384
    assert loaded.train.lr == 3e-4
    assert loaded.train.batch_size == 64
    assert loaded.train.warmup_steps == 5000
    assert loaded.model.n_layer == 6
    assert loaded.model.n_head == 4
    assert loaded.model.n_embd == 256
    assert loaded.model.dropout == 0.05
    assert loaded.loss_weights.main_change == 3.0
    assert loaded.loss_weights.button_z == 15.0
    assert loaded.rl.gamma == 0.99
    assert loaded.ppo.clip_ratio == 0.1


def test_set_config(tmp_path: Path) -> None:
    """Test that set_config properly sets the global config."""
    config = Config()
    config.train.lr = 9e-5

    result = set_config(config)

    assert result is config
    from config.config import get_config

    assert get_config() is config
    assert get_config().train.lr == 9e-5
