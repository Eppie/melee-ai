from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from column_map import ColumnMap
from config import get_config, init_config, reset_config
from feature_transforms import feature_spec_from_config
from schema import get_feature_names, get_target_names
from train.value_head import (
    build_reward_feature_index,
    compute_frame_rewards,
    compute_value_targets,
)
from window_dataset import _apply_feature_transforms
from zarr_storage import (
    Schema,
    _rows_to_dense,
    process_one_episode,
    _swap_row_players,
)


def _build_colmap() -> ColumnMap:
    """Return a :class:`ColumnMap` using the canonical schema names."""
    return ColumnMap(get_feature_names(), get_target_names())


def _zeros_feature_tensor(colmap: ColumnMap, seq_len: int) -> torch.Tensor:
    """Allocate a zero tensor with shape [1, seq_len, F]."""
    return torch.zeros((1, seq_len, len(colmap.feat_names)), dtype=torch.float32)


@pytest.fixture(autouse=True)
def _reset_config():
    """Ensure each test sees a fresh RL config singleton."""
    reset_config()
    init_config()
    yield
    reset_config()


@pytest.fixture
def reward_setup():
    colmap = _build_colmap()
    idx = build_reward_feature_index(colmap)
    return colmap, idx


@pytest.fixture(scope="module")
def replay_reward_data():
    """Load test/test.slp and precompute rewards for integration tests."""
    reset_config()
    init_config()
    cfg = get_config()
    schema = Schema(features=get_feature_names(), targets=get_target_names())
    slp_path = Path(__file__).with_name("test.slp")
    rows = process_one_episode(str(slp_path))
    X_np, _, _, _, feat_names, targ_names = _rows_to_dense(rows, schema)
    colmap = ColumnMap(feat_names, targ_names)
    idx = build_reward_feature_index(colmap)
    feature_spec = feature_spec_from_config(cfg.features)

    def transform(arr: np.ndarray) -> np.ndarray:
        buf = np.ascontiguousarray(arr, dtype=np.float32)
        if feature_spec is None:
            return buf
        return _apply_feature_transforms(buf, feat_names, feature_spec)

    X = torch.from_numpy(transform(X_np)).unsqueeze(0)
    rewards = compute_frame_rewards(X, colmap, idx=idx).squeeze(0)

    swapped_rows = [_swap_row_players(row) for row in rows]
    X_swapped, _, _, _, _, _ = _rows_to_dense(swapped_rows, schema)
    swapped_rewards = compute_frame_rewards(
        torch.from_numpy(transform(X_swapped)).unsqueeze(0), colmap, idx=idx
    ).squeeze(0)

    reset_config()
    return {
        "rewards": rewards,
        "swapped_rewards": swapped_rewards,
    }


def test_compute_frame_rewards_damage_and_stock(reward_setup):
    """Damage deltas and stock losses convert to the configured rewards."""
    colmap, idx = reward_setup
    assert idx.p2_percent is not None and idx.p2_stock is not None

    seq_len = 4
    X = _zeros_feature_tensor(colmap, seq_len)

    # Opponent percent rises by 2, then by 3 -> convert to rewards at frames 1 and 2.
    X[0, :, idx.p2_percent] = torch.tensor([0.0, 2.0, 5.0, 5.0])
    # Opponent loses one stock between frames 1 and 2.
    X[0, :, idx.p2_stock] = torch.tensor([4.0, 4.0, 3.0, 3.0])

    rewards = compute_frame_rewards(X, colmap, idx=idx).squeeze(0)

    cfg = get_config().rl
    expected = torch.tensor(
        [
            0.0,  # no prior frame for damage calculation
            2.0 * 100.0 * float(cfg.reward_damage_dealt),
            3.0 * 100.0 * float(cfg.reward_damage_dealt) + float(cfg.reward_stock_taken),
            0.0,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(rewards, expected)


def test_compute_frame_rewards_shield_penalty(reward_setup):
    """Low shield strength applies a per-frame penalty only to the ego player."""
    colmap, idx = reward_setup
    assert idx.p1_shield_strength is not None
    assert idx.p2_shield_strength is not None

    seq_len = 3
    X = _zeros_feature_tensor(colmap, seq_len)
    X[0, :, idx.p2_shield_strength] = 1.0  # keep opponent shielded to avoid penalties
    X[0, :, idx.p1_shield_strength] = torch.tensor([1.0, 0.1, 0.4])

    rewards = compute_frame_rewards(X, colmap, idx=idx).squeeze(0)

    cfg = get_config().rl
    # penalty = clamp(1 - 2 * shield, 0, 1) * reward_low_shield
    expected = torch.tensor(
        [
            0.0,
            (1 - 2 * 0.1) * float(cfg.reward_low_shield),
            (1 - 2 * 0.4) * float(cfg.reward_low_shield),
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(rewards, expected)


def test_compute_frame_rewards_hitlag_terms(reward_setup):
    """Attacking frames award the configured opponent hitlag bonus."""
    colmap, idx = reward_setup
    assert idx.p1_is_in_hitlag is not None
    assert idx.p1_is_defender_in_hitlag is not None
    assert idx.p2_is_in_hitlag is not None
    assert idx.p2_is_defender_in_hitlag is not None

    seq_len = 3
    X = _zeros_feature_tensor(colmap, seq_len)

    # Frame 1: p1 hits p2 (opponent hitlag metric == 1).
    X[0, 1, idx.p2_is_in_hitlag] = 1.0
    # Frame 2: roles swap, p2 hits p1.
    X[0, 2, idx.p1_is_in_hitlag] = 1.0

    rewards = compute_frame_rewards(X, colmap, idx=idx).squeeze(0)

    cfg = get_config().rl
    expected = torch.tensor(
        [
            0.0,
            float(cfg.reward_hitlag_opponent),
            -float(cfg.reward_hitlag_opponent),
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(rewards, expected)


def test_compute_value_targets_sequence_gamma(reward_setup):
    """Discounted returns should honor the provided gamma factor."""
    colmap, idx = reward_setup
    seq_len = 4
    X = _zeros_feature_tensor(colmap, seq_len)
    X[0, :, idx.p2_percent] = torch.tensor([0.0, 1.0, 1.0, 2.0])

    gamma = 0.9
    returns = compute_value_targets(
        X, colmap, gamma=gamma, reward_idx=idx
    ).squeeze(0).squeeze(-1)

    expected_rewards = torch.tensor([0.0, 2.0, 0.0, 2.0])

    manual = torch.empty(seq_len)
    for t in range(seq_len):
        total = 0.0
        for k in range(t, seq_len):
            total += (gamma ** (k - t)) * expected_rewards[k]
        manual[t] = total

    expected_returns = manual
    assert torch.allclose(returns, expected_returns, atol=1e-6)


def test_replay_rewards_shape_and_sparsity(replay_reward_data):
    rewards = replay_reward_data["rewards"]
    assert rewards.shape[0] == 6956
    assert torch.count_nonzero(rewards).item() == 385


@pytest.mark.parametrize(
    ("frame", "value"),
    [
        (32, 0.02),
        (37, 0.02),
        (6918, -0.25),
    ],
)
def test_replay_rewards_matches_known_frames(replay_reward_data, frame, value):
    rewards = replay_reward_data["rewards"]
    assert pytest.approx(value, abs=1e-6) == float(rewards[frame])


def test_replay_rewards_zero_sum_after_player_swap(replay_reward_data):
    rewards = replay_reward_data["rewards"]
    swapped = replay_reward_data["swapped_rewards"]
    assert torch.allclose(swapped, -rewards, atol=1e-6)
