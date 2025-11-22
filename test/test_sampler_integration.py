from __future__ import annotations

from pathlib import Path
from typing import Tuple

import json
import numpy as np
import pytest
import zarr
import torch

from column_map import ColumnMap
from config import get_config, init_config, reset_config
from controller_quantization import quantize_targets, target_transform_hash
from feature_transforms import feature_spec_from_config, transform_hash
from schema import get_feature_names, get_target_names
from window_dataset import (
    RandomWindowSampler,
    WindowDataset,
    _apply_feature_transforms,
)
from zarr_storage import (
    Schema,
    _rows_to_dense,
    process_one_episode,
)


@pytest.fixture(autouse=True)
def _reset_config():
    reset_config()
    init_config()
    yield
    reset_config()


def _write_zarr_dataset(
    base_dir: Path,
    X_raw: np.ndarray,
    Y_raw: np.ndarray,
    seq_len: int,
    feat_names: list[str],
    targ_names: list[str],
) -> Path:
    data_dir = base_dir / "zarr_corpus"
    data_dir.mkdir()

    feature_spec = feature_spec_from_config(get_config().features)
    X_proc = _apply_feature_transforms(
        np.ascontiguousarray(X_raw, dtype=np.float32),
        feat_names,
        feature_spec,
    )
    colmap = ColumnMap(feat_names, targ_names)
    target_info = quantize_targets(
        torch.from_numpy(Y_raw).unsqueeze(0), colmap, input_domain="unit01"
    )

    meta = {
        "build_config": {"seq_len": seq_len},
        "schema": {"features": feat_names, "targets": targ_names},
        "feat_dtypes": ["float32"] * len(feat_names),
        "targ_dtypes": ["float32"] * len(targ_names),
        "transforms": {
            "features": transform_hash(feature_spec),
            "targets": target_transform_hash(input_domain="unit01"),
        },
        "processed": {
            "features": "X_proc",
            "targets": {"packed": "Y_proc/proc"},
        },
    }
    (data_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    lengths = np.array([X_raw.shape[0]], dtype=np.int64)
    wins = np.array([max(0, X_raw.shape[0] - seq_len + 1)], dtype=np.int64)
    np.save(data_dir / "lengths.npy", lengths)
    np.save(data_dir / "wins_per_ep.npy", wins)

    total_windows = int(wins.sum())
    window_index = np.empty((total_windows, 2), dtype=np.int64)
    for idx in range(total_windows):
        window_index[idx, 0] = 0
        window_index[idx, 1] = idx
    np.save(data_dir / "window_index.npy", window_index)

    (data_dir / "index.jsonl").write_text(
        '{"episode_id": 0, "shard_id": 0, "frames": %d}\n' % lengths[0],
        encoding="utf-8",
    )

    shard_dir = data_dir / "shard_00000.zarr"
    root = zarr.open_group(str(shard_dir), mode="w")
    ep_group = root.create_group("ep_000000")
    ep_group.create_array("X", data=X_raw.astype(np.float32, copy=False))
    ep_group.create_array("X_proc", data=X_proc.astype(np.float32, copy=False))
    ep_group.create_array("Y", data=Y_raw.astype(np.float32, copy=False))
    y_proc = ep_group.create_group("Y_proc")
    proc = torch.cat(
        [
            torch.from_numpy(X_proc),
            torch.stack(
                [
                    target_info["main_idx"].squeeze(0).to(torch.float32),
                    target_info["c_idx"].squeeze(0).to(torch.float32),
                    target_info["shoulder_idx"].squeeze(0).to(torch.float32),
                ],
                dim=-1,
            ),
            target_info["buttons"].squeeze(0),
        ],
        dim=1,
    ).cpu().numpy()
    y_proc.create_array("proc", data=proc)
    y_proc.attrs["feat_dim"] = X_proc.shape[1]
    y_proc.attrs["buttons_dim"] = target_info["buttons"].shape[-1]

    return data_dir


@pytest.fixture(scope="module")
def slp_zarr(tmp_path_factory: pytest.TempPathFactory) -> Tuple[Path, np.ndarray, np.ndarray, int, list[str]]:
    cfg = init_config()
    schema = Schema(features=get_feature_names(), targets=get_target_names())
    slp_path = Path(__file__).with_name("test.slp")
    rows = process_one_episode(str(slp_path))

    X_raw, Y_raw, _, _, feat_names, targ_names = _rows_to_dense(rows, schema)
    seq_len = 8
    max_frames = seq_len + 6
    X_raw = X_raw[:max_frames]
    Y_raw = Y_raw[:max_frames]

    data_dir = _write_zarr_dataset(
        tmp_path_factory.mktemp("slp_zarr"),
        X_raw,
        Y_raw,
        seq_len,
        feat_names,
        targ_names,
    )
    return data_dir, X_raw, Y_raw, seq_len, feat_names


def test_random_window_sampler_produces_expected_windows(slp_zarr) -> None:
    data_dir, X_raw, Y_raw, seq_len, feat_names = slp_zarr

    feature_spec = feature_spec_from_config(get_config().features)
    transformed = _apply_feature_transforms(
        np.ascontiguousarray(X_raw, dtype=np.float32),
        feat_names,
        feature_spec,
    )
    dataset = WindowDataset(data_dir, feature_transforms=feature_spec)
    sampler = RandomWindowSampler(index=dataset.index, stride=1)
    sampler.set_epoch(0)
    root = zarr.open_group(str(data_dir / "shard_00000.zarr"), mode="r")

    expected_order = np.arange(dataset.index.total_windows, dtype=np.int64)
    seed = (sampler.epoch * 0x9E3779B97F4A7C15 + 4242) % (2**63 - 1)
    rng = np.random.default_rng(seed)
    rng.shuffle(expected_order)

    sampled_indices = list(iter(sampler))
    assert sampled_indices == expected_order.tolist()

    for idx in sampled_indices[: min(5, len(sampled_indices))]:
        ep_idx, start = dataset.index.window_to_episode(idx)
        assert ep_idx == 0

        window = dataset[idx]
        expected_X = root["ep_000000/X_proc"][start : start + seq_len]
        assert np.allclose(window["X"].numpy(), expected_X, atol=1e-6)
        target_info = window["target_info"]
        expected_target_info = quantize_targets(
            torch.from_numpy(Y_raw[start : start + seq_len]).unsqueeze(0),
            ColumnMap(feat_names, get_target_names()),
            input_domain="unit01",
        )
        assert torch.equal(
            target_info["main_idx"], expected_target_info["main_idx"].squeeze(0)
        )
        assert torch.equal(
            target_info["c_idx"], expected_target_info["c_idx"].squeeze(0)
        )
        assert torch.equal(
            target_info["shoulder_idx"],
            expected_target_info["shoulder_idx"].squeeze(0),
        )
        assert torch.allclose(
            target_info["buttons"], expected_target_info["buttons"].squeeze(0)
        )
