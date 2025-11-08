from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from config import FeatureConfig
from feature_transforms import feature_spec_from_config
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
    # Create a FeatureConfig with a stick_palette transform
    feature_config = FeatureConfig(
        transforms=[
            {
                "transform": "stick_palette",
                "features": ["main_stick_x", "main_stick_y"],
                "palette": "fox_main",
            }
        ]
    )
    feature_spec = feature_spec_from_config(feature_config)

    # Create a WindowDataset instance
    dataset = WindowDataset(zarr_corpus, feature_transforms=feature_spec)

    # Get a window from the dataset
    window = dataset[0]
    x_window = window["X"]
    y_window = window["Y"]

    # Get the original data from the Zarr corpus
    root = zarr.open_group(str(zarr_corpus / "shard_00000.zarr"), mode="r")
    original_x = root["ep_000000/X"][:4]
    original_y = root["ep_000000/Y"][:4]

    # Assert that the X tensor has been transformed (i.e., it's different from the original)
    assert not np.allclose(x_window.numpy(), original_x), "X tensor should be transformed"

    # Assert that the Y tensor has NOT been transformed (i.e., it's the same as the original)
    assert np.allclose(y_window.numpy(), original_y), "Y tensor should not be transformed"
