from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

import window_dataset


@pytest.fixture
def zarr_corpus(tmp_path: Path) -> Path:
    """Creates a temporary Zarr corpus for testing."""
    data_dir = tmp_path / "zarr_corpus"
    data_dir.mkdir()

    # Create meta.json
    meta = {
        "build_config": {"seq_len": 4},
        "schema": {
            "features": ["p1_main_stick_x", "p1_main_stick_y"],
            "targets": ["p1_main_stick_x", "p1_main_stick_y"],
        },
    }
    with (data_dir / "meta.json").open("w") as f:
        json.dump(meta, f)

    # Create lengths.npy and wins_per_ep.npy
    lengths = np.array([10], dtype=np.int64)
    np.save(data_dir / "lengths.npy", lengths)
    wins = np.array([7], dtype=np.int64)
    np.save(data_dir / "wins_per_ep.npy", wins)

    total_windows = int(wins.sum())
    window_index = np.empty((total_windows, 2), dtype=np.int64)
    cursor = 0
    for ep_idx, num_windows in enumerate(wins):
        if num_windows == 0:
            continue
        next_cursor = cursor + int(num_windows)
        window_index[cursor:next_cursor, 0] = ep_idx
        window_index[cursor:next_cursor, 1] = np.arange(num_windows, dtype=np.int64)
        cursor = next_cursor
    np.save(data_dir / "window_index.npy", window_index)

    # Create index.jsonl
    with (data_dir / "index.jsonl").open("w") as f:
        f.write('{"episode_id": 0, "shard_id": 0, "frames": 10}\n')

    # Create shard_00000.zarr
    shard_dir = data_dir / "shard_00000.zarr"
    shard_dir.mkdir()
    root = zarr.open_group(str(shard_dir), mode="w")
    ep_group = root.create_group("ep_000000")
    x_data = np.random.rand(10, 2).astype(np.float32)
    y_data = np.random.rand(10, 2).astype(np.float32)
    ep_group.create_array("X", data=x_data)
    ep_group.create_array("Y", data=y_data)

    return data_dir


def test_window_dataset_double_quantization(zarr_corpus: Path):
    """
    Tests that the WindowDataset does not apply feature transforms to the target tensor (Y).
    This prevents the double quantization issue.
    """
    # Create a WindowDataset instance (transforms are now hardcoded and always applied)
    dataset = window_dataset.WindowDataset(zarr_corpus)

    # Get a window from the dataset
    window = dataset[0]
    x_window = window["X"]
    y_window = window["Y"]

    # Get the original data from the Zarr corpus
    root = zarr.open_group(str(zarr_corpus / "shard_00000.zarr"), mode="r")
    original_x = root["ep_000000/X"][:4]
    original_y = root["ep_000000/Y"][:4]

    # Assert that the X tensor has been transformed (i.e., it's different from the original)
    assert not np.allclose(
        x_window.numpy(), original_x
    ), "X tensor should be transformed"

    # Assert that the Y tensor has NOT been transformed (i.e., it's the same as the original)
    assert np.allclose(
        y_window.numpy(), original_y
    ), "Y tensor should not be transformed"


def test_window_to_episode_prefers_window_index(
    monkeypatch: pytest.MonkeyPatch, zarr_corpus: Path
) -> None:
    dataset = window_dataset.WindowDataset(zarr_corpus)
    assert dataset.index._window_index is not None

    def fail_searchsorted(*args, **kwargs):
        raise AssertionError(
            "searchsorted should not be invoked when window_index is available"
        )

    monkeypatch.setattr(window_dataset.np, "searchsorted", fail_searchsorted)
    for idx in range(dataset.index.total_windows):
        ep_idx, offset = dataset.index.window_to_episode(idx)
        assert ep_idx == 0
        assert offset == idx
