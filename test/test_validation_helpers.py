from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from config import init_config, reset_config
from validation import (
    _apply_checkpoint_config,
    _nearest_diffs_within_window,
    _normalize_checkpoint_config,
    _patch_model_config_from_state_dict,
)


@pytest.fixture(autouse=True)
def clear_global_config():
    reset_config()
    yield
    reset_config()


def test_normalize_checkpoint_config_handles_varied_inputs():
    payload = {"train": {"lr": 0.05}}
    assert _normalize_checkpoint_config(payload) == payload

    as_json = json.dumps(payload)
    assert _normalize_checkpoint_config(as_json) == payload

    as_bytes = as_json.encode("utf-8")
    assert _normalize_checkpoint_config(as_bytes) == payload


def test_apply_checkpoint_config_merges_nested_sections():
    cfg = init_config(freeze=False)
    _apply_checkpoint_config(
        cfg,
        {
            "train": {"lr": 0.123},
            "model": {"n_layer": 3},
        },
    )

    assert cfg.train.lr == 0.123
    assert cfg.model.n_layer == 3


def test_patch_model_config_from_state_dict_infers_shapes():
    cfg = init_config(freeze=False)
    state_dict = {
        "projection_down.weight": torch.zeros(64, 10),
        "blocks.0.attn.weight": torch.zeros(1),
        "blocks.2.ff.weight": torch.zeros(1),
        "blocks.5.attn.weight": torch.zeros(1),
    }

    _patch_model_config_from_state_dict(cfg, state_dict)

    assert cfg.model.n_embd == 64
    assert cfg.model.n_layer == 6  # max block index + 1


def test_nearest_diffs_within_window_basic_and_filtering():
    true_frames = [10, 20, 30]
    pred_frames = [9, 22, 29]

    diffs = _nearest_diffs_within_window(true_frames, pred_frames, window=2)
    np.testing.assert_array_equal(diffs, np.array([-1, 2, -1], dtype=np.int32))

    diffs_tight = _nearest_diffs_within_window(true_frames, pred_frames, window=1)
    np.testing.assert_array_equal(diffs_tight, np.array([-1, -1], dtype=np.int32))

    diffs_empty = _nearest_diffs_within_window([], pred_frames, window=2)
    assert diffs_empty.size == 0
