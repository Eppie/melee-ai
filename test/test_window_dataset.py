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
            "targets": {
                "raw": "Y_raw",
                "transformed": {
                    "type": "quantized_v1",
                    "main_idx": "Y_main_idx",
                    "c_idx": "Y_c_idx",
                    "buttons": "Y_buttons",
                    "shoulder_idx": "Y_shoulder_idx",
                },
            },
        },
        "target_quantization": {
            "version": 1,
            "layout": {
                "type": "quantized_v1",
                "main_idx": "Y_main_idx",
                "c_idx": "Y_c_idx",
                "buttons": "Y_buttons",
                "shoulder_idx": "Y_shoulder_idx",
            },
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
    ep_group.create_array("Y_main_idx", data=(np.arange(10, dtype=np.int16) % 5))
    ep_group.create_array("Y_c_idx", data=(np.arange(10, dtype=np.int16) % 7))
    buttons = np.random.rand(10, 2).astype(np.float32)
    ep_group.create_array("Y_buttons", data=buttons)
    ep_group.create_array("Y_shoulder_idx", data=(np.arange(10, dtype=np.int16) % 3))

    return data_dir


def test_window_dataset_uses_transformed_arrays(zarr_corpus: Path):
    """WindowDataset should surface the transformed feature/target arrays."""
    dataset = WindowDataset(zarr_corpus)
    window = dataset[0]
    x_window = window["X"]
    target_info = window["target_info"]

    root = zarr.open_group(str(zarr_corpus / "shard_00000.zarr"), mode="r")
    transformed_x = root["ep_000000/X_transformed"][:4]
    expected_main = root["ep_000000/Y_main_idx"][:4]
    expected_c = root["ep_000000/Y_c_idx"][:4]
    expected_buttons = root["ep_000000/Y_buttons"][:4]
    expected_shoulder = root["ep_000000/Y_shoulder_idx"][:4]

    assert np.allclose(
        x_window.numpy(), transformed_x
    ), "Features should match the stored transformed array"
    assert np.array_equal(
        target_info["main_idx"].numpy(), expected_main
    ), "Main indices should match transformed storage"
    assert np.array_equal(
        target_info["c_idx"].numpy(), expected_c
    ), "C-stick indices should match transformed storage"
    assert np.allclose(
        target_info["buttons"].numpy(), expected_buttons
    ), "Button targets should match transformed storage"
    assert np.array_equal(
        target_info["shoulder_idx"].numpy(), expected_shoulder
    ), "Shoulder indices should match transformed storage"
