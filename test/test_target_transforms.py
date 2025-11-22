from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from column_map import ColumnMap
from controller_quantization import quantize_targets
from schema import get_feature_names, get_target_names
from zarr_storage import Schema, _rows_to_dense, process_one_episode


def _load_targets(batch_len: int = 50):
    schema = Schema(features=get_feature_names(), targets=get_target_names())
    rows = process_one_episode(str(Path(__file__).with_name("test.slp")))
    _, Y_raw, _, _, feat_names, targ_names = _rows_to_dense(rows, schema)
    colmap = ColumnMap(feat_names, targ_names)
    return torch.from_numpy(Y_raw[:batch_len]).unsqueeze(0), colmap


def test_quantize_targets_matches_expected_indices():
    Y, colmap = _load_targets(batch_len=60)
    out = quantize_targets(Y, colmap, input_domain="unit01")

    # Early frames stay centered (no movement)
    assert out["main_idx"][0, :10].tolist() == [3] * 10
    assert out["c_idx"][0, :10].tolist() == [0] * 10
    assert out["shoulder_idx"][0, :10].tolist() == [0] * 10

    # Later frames exercise other palette entries (slice corresponds to frames 30–49)
    assert out["main_idx"][0, 30:34].tolist() == [24, 20, 3, 3]
    assert out["main_idx"][0, 36:40].tolist() == [12, 12, 56, 56]
    assert out["shoulder_idx"][0, 33:38].tolist() == [4, 4, 4, 4, 4]


def test_quantize_targets_clamps_button_probabilities():
    _, colmap = _load_targets(batch_len=2)
    num_targets = len(colmap.targ_names)
    Y = torch.zeros((1, 2, num_targets), dtype=torch.float32)

    # Set button probabilities outside [0, 1] to confirm clamping.
    if not colmap.y_buttons:
        raise AssertionError("ColumnMap did not expose button target columns.")
    Y[0, 0, colmap.y_buttons[0]] = -0.5
    Y[0, 1, colmap.y_buttons[0]] = 1.5

    out = quantize_targets(Y, colmap, input_domain="unit01")
    buttons = out["buttons"]
    assert buttons[0, 0, colmap.y_buttons[0]].item() == 0.0
    assert buttons[0, 1, colmap.y_buttons[0]].item() == 1.0
