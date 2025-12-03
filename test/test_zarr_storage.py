from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from config.config import get_config, init_config, reset_config
from schema import get_feature_names, get_raw_target_names
from zarr_storage import (
    EPISODES_PER_REPLAY,
    Schema,
    _process_episode_task,
    build_dataset,
)


@pytest.fixture(autouse=True)
def _reset_config():
    """Ensure each test sees a fresh RL config singleton."""
    reset_config()
    init_config(freeze=False)
    yield
    reset_config()


def test_process_episode_task_emits_original_and_flipped_perspectives():
    slp_path = Path(__file__).with_name("test.slp")
    schema = Schema(features=get_feature_names(), targets=get_raw_target_names())

    episodes = _process_episode_task(str(slp_path), schema)

    assert len(episodes) == EPISODES_PER_REPLAY
    orig, flipped = episodes
    assert orig.features.shape == flipped.features.shape

    name2idx = {name: idx for idx, name in enumerate(orig.feature_names)}
    frame_idx = -1  # inspect final available frame to avoid pre-game padding
    for base_field in ("percent", "stock", "action"):
        p1_key = f"p1_{base_field}"
        p2_key = f"p2_{base_field}"
        orig_p1 = orig.features[frame_idx, name2idx[p1_key]]
        orig_p2 = orig.features[frame_idx, name2idx[p2_key]]
        flipped_p1 = flipped.features[frame_idx, name2idx[p1_key]]
        flipped_p2 = flipped.features[frame_idx, name2idx[p2_key]]

        assert flipped_p1 == pytest.approx(orig_p2)
        assert flipped_p2 == pytest.approx(orig_p1)

    assert not np.allclose(orig.features, flipped.features)


def test_build_dataset_writes_two_episodes_per_replay(tmp_path, monkeypatch):
    monkeypatch.setenv("ZARR_USE_THREADS", "1")
    config = get_config()
    config.zarr.shard_size = EPISODES_PER_REPLAY  # one replay per shard
    config.seq_len = 32

    schema = Schema(features=get_feature_names(), targets=get_raw_target_names())
    slp_path = Path(__file__).with_name("test.slp")
    out_dir = tmp_path / "out"

    build_dataset([str(slp_path)], schema, str(out_dir))

    index_lines = (out_dir / "index.jsonl").read_text().strip().splitlines()
    assert len(index_lines) == EPISODES_PER_REPLAY

    lengths = np.load(out_dir / "lengths.npy")
    assert lengths.shape == (EPISODES_PER_REPLAY,)

    wins_per_ep = np.load(out_dir / "wins_per_ep.npy")
    assert wins_per_ep.shape == (EPISODES_PER_REPLAY,)

    shard = zarr.open_group(str(out_dir / "shard_00000.zarr"), mode="r")
    for ep in (f"ep_{i:06d}" for i in range(EPISODES_PER_REPLAY)):
        assert ep in shard
