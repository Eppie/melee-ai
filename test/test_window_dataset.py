from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from window_dataset import WindowDataset


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
        "array_layout": {
            "features": {"raw": "X_raw", "transformed": "X_transformed"},
            "targets": {"raw": "Y_raw", "transformed": "Y_quantized"},
        },
        "target_quantization": {
            "version": 1,
            "dataset": "Y_quantized",
            "fields": {
                "main_idx": {"offset": 0},
                "c_idx": {"offset": 1},
                "shoulder_idx": {"offset": 2},
            },
            "buttons": {"offset": 3, "count": 2},
            "column_count": 5,
            "main_K": 5,
            "c_K": 5,
            "buttons_K": 2,
            "shoulder_K": 3,
        },
    }
    with (data_dir / "meta.json").open("w") as f:
        json.dump(meta, f)

    # Create lengths.npy and wins_per_ep.npy
    lengths = np.array([10], dtype=np.int64)
    np.save(data_dir / "lengths.npy", lengths)
    wins = np.array([7], dtype=np.int64)
    np.save(data_dir / "wins_per_ep.npy", wins)

    # Create index.jsonl
    with (data_dir / "index.jsonl").open("w") as f:
        f.write('{"episode_id": 0, "shard_id": 0, "frames": 10}\n')

    # Create shard_00000.zarr
    shard_dir = data_dir / "shard_00000.zarr"
    shard_dir.mkdir()
    root = zarr.open_group(str(shard_dir), mode="w")
    ep_group = root.create_group("ep_000000")
    x_raw = np.random.rand(10, 2).astype(np.float32)
    y_raw = np.random.rand(10, 2).astype(np.float32)
    x_transformed = (x_raw * 2.0).astype(np.float32)
    ep_group.create_array("X_raw", data=x_raw)
    ep_group.create_array("X_transformed", data=x_transformed)
    ep_group.create_array("Y_raw", data=y_raw)
    main_idx = (np.arange(10) % 5).astype(np.float32)
    c_idx = (np.arange(10) % 3).astype(np.float32)
    shoulder_idx = (np.arange(10) % 2).astype(np.float32)
    buttons = np.random.rand(10, 2).astype(np.float32)
    y_quantized = np.zeros((10, 5), dtype=np.float32)
    y_quantized[:, 0] = main_idx
    y_quantized[:, 1] = c_idx
    y_quantized[:, 2] = shoulder_idx
    y_quantized[:, 3:] = buttons
    ep_group.create_array("Y_quantized", data=y_quantized)

    return data_dir


def test_window_dataset_uses_transformed_arrays(zarr_corpus: Path):
    """WindowDataset should surface the transformed feature/target arrays."""
    dataset = WindowDataset(zarr_corpus)
    window = dataset[0]
    x_window = window["X"]
    target_info = window["target_info"]

    root = zarr.open_group(str(zarr_corpus / "shard_00000.zarr"), mode="r")
    transformed_x = root["ep_000000/X_transformed"][:4]
    quantized = root["ep_000000/Y_quantized"][:4]

    assert np.allclose(
        x_window.numpy(), transformed_x
    ), "Features should match the stored transformed array"
    assert np.array_equal(
        target_info["main_idx"].numpy(), quantized[:, 0].astype(np.int64)
    ), "Main indices should match transformed storage"
    assert np.array_equal(
        target_info["c_idx"].numpy(), quantized[:, 1].astype(np.int64)
    ), "C-stick indices should match transformed storage"
    assert np.allclose(
        target_info["buttons"].numpy(), quantized[:, 3:]
    ), "Button targets should match transformed storage"
    assert np.array_equal(
        target_info["shoulder_idx"].numpy(), quantized[:, 2].astype(np.int64)
    ), "Shoulder indices should match transformed storage"
