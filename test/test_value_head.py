from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from column_map import ColumnMap
from config.config import get_config, init_config, reset_config
from feature_transforms import apply_feature_transforms
from schema import get_feature_names, get_target_names
from train.value_head import (
    build_reward_feature_index,
    compute_frame_rewards,
    compute_value_targets,
)
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
    return colmap, idx, get_config().reward


@pytest.fixture(scope="module")
def replay_reward_data():
    """Load test/test.slp and precompute rewards for integration tests."""
    reset_config()
    init_config()
    cfg = get_config()
    from schema import get_raw_target_names

    schema = Schema(features=get_feature_names(), targets=get_raw_target_names())
    slp_path = Path(__file__).with_name("test.slp")
    rows = process_one_episode(str(slp_path))
    X_np, _, _, _, feat_names, _ = _rows_to_dense(rows, schema)
    # Use quantized target layout for reward indexing
    colmap = ColumnMap(feat_names, get_target_names())
    idx = build_reward_feature_index(colmap)

    def transform(arr: np.ndarray) -> np.ndarray:
        buf = np.ascontiguousarray(arr, dtype=np.float32)
        return apply_feature_transforms(buf, feat_names)

    X = torch.from_numpy(transform(X_np)).unsqueeze(0)
    rewards = compute_frame_rewards(X, idx=idx, reward_cfg=cfg.reward).squeeze(0)

    swapped_rows = [_swap_row_players(row) for row in rows]
    X_swapped, _, _, _, _, _ = _rows_to_dense(swapped_rows, schema)
    swapped_rewards = compute_frame_rewards(
        torch.from_numpy(transform(X_swapped)).unsqueeze(0),
        idx=idx,
        reward_cfg=cfg.reward,
    ).squeeze(0)

    reset_config()
    return {
        "rewards": rewards,
        "swapped_rewards": swapped_rewards,
    }


def test_compute_frame_rewards_damage_and_stock(reward_setup):
    """Damage deltas and stock losses convert to the configured rewards."""
    colmap, idx, reward_cfg = reward_setup
    assert idx.p2_percent is not None and idx.p2_action is not None

    seq_len = 4
    X = _zeros_feature_tensor(colmap, seq_len)

    # Opponent percent rises by 2, then by 3 -> convert to rewards at frames 1 and 2.
    X[0, :, idx.p2_percent] = torch.tensor([0.0, 2.0, 5.0, 5.0])
    # Opponent loses one stock between frames 1 and 2 (action transitions from alive to dying)
    # Action > 0xA means alive, action <= 0xA means dying
    X[0, :, idx.p2_action] = torch.tensor(
        [50.0, 50.0, 5.0, 5.0]
    )  # Alive -> Alive -> Dying -> Dying

    rewards = compute_frame_rewards(X, idx=idx, reward_cfg=reward_cfg).squeeze(0)

    cfg = reward_cfg
    expected = torch.tensor(
        [
            2.0 * float(cfg.reward_damage_dealt),
            3.0 * float(cfg.reward_damage_dealt) + float(cfg.reward_stock_taken),
            0.0,
            0.0,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(rewards, expected)


def test_compute_frame_rewards_shield_penalty(reward_setup):
    """Low shield strength applies a per-frame penalty only to the ego player."""
    colmap, idx, reward_cfg = reward_setup
    assert idx.p1_shield_strength is not None
    assert idx.p2_shield_strength is not None

    seq_len = 3
    X = _zeros_feature_tensor(colmap, seq_len)
    X[0, :, idx.p2_shield_strength] = 1.0  # keep opponent shielded to avoid penalties
    X[0, :, idx.p1_shield_strength] = torch.tensor([1.0, 0.1, 0.4])

    rewards = compute_frame_rewards(X, idx=idx, reward_cfg=reward_cfg).squeeze(0)

    cfg = reward_cfg
    # penalty = clamp(1 - 2 * shield, 0, 1) * reward_low_shield
    expected = torch.tensor(
        [
            (1 - 2 * 0.1) * float(cfg.reward_low_shield),
            (1 - 2 * 0.4) * float(cfg.reward_low_shield),
            0.0,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(rewards, expected)


def test_compute_frame_rewards_hitlag_terms(reward_setup):
    """Attacking frames award the configured opponent hitlag bonus."""
    colmap, idx, reward_cfg = reward_setup
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

    rewards = compute_frame_rewards(X, idx=idx, reward_cfg=reward_cfg).squeeze(0)

    cfg = reward_cfg
    expected = torch.tensor(
        [
            float(cfg.reward_hitlag_opponent),
            -float(cfg.reward_hitlag_opponent),
            0.0,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(rewards, expected)


def test_compute_frame_rewards_hitstun_curve(reward_setup):
    """Hitstun rewards follow the configured curve: no reward until min_frames,
    peak at peak_frames, then decline to zero at max_frames."""
    colmap, idx, reward_cfg = reward_setup
    assert idx.p2_is_in_hitstun is not None

    # Configure curve: min=15, peak=400, max=600, peak_value=0.04
    cfg = reward_cfg.model_copy(update={
        "reward_hitstun_min_frames": 15,
        "reward_hitstun_peak_frames": 400,
        "reward_hitstun_max_frames": 600,
        "reward_hitstun_peak_value": 0.04,
    })

    # Create sequence with continuous hitstun
    seq_len = 650
    X = _zeros_feature_tensor(colmap, seq_len)
    # All frames in hitstun
    X[0, :, idx.p2_is_in_hitstun] = 1.0

    rewards = compute_frame_rewards(X, idx=idx, reward_cfg=cfg).squeeze(0)

    # Test key points on the curve
    # Frames 0-14: no reward (below min_frames)
    assert torch.allclose(rewards[0], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(rewards[14], torch.tensor(0.0), atol=1e-6)

    # Frame 15: just reached min_frames, should start getting reward
    assert rewards[15].item() > 0.0
    assert rewards[15].item() < 0.04

    # Frame 399: at peak (streak of 400), should be maximum reward
    # Frame 400: streak of 401, slightly past peak
    assert torch.allclose(rewards[399], torch.tensor(0.04), atol=1e-4)
    assert rewards[400].item() < 0.04  # Past peak, declining
    assert rewards[400].item() > 0.039  # But still very close to peak

    # Frame 500: streak of 501, midway in decline (between 400 and 600)
    expected_500 = 0.04 * (600 - 501) / (600 - 400)  # Linear interpolation
    assert torch.allclose(rewards[500], torch.tensor(expected_500), atol=1e-4)

    # Frame 600+: beyond max_frames, no reward
    assert torch.allclose(rewards[600], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(rewards[649], torch.tensor(0.0), atol=1e-6)


def test_compute_frame_rewards_hitstun_resets_on_break(reward_setup):
    """Hitstun streak counter resets when hitstun is interrupted."""
    colmap, idx, reward_cfg = reward_setup
    assert idx.p2_is_in_hitstun is not None

    cfg = reward_cfg.model_copy(update={
        "reward_hitstun_min_frames": 10,
        "reward_hitstun_peak_frames": 50,
        "reward_hitstun_max_frames": 100,
        "reward_hitstun_peak_value": 0.1,
    })

    seq_len = 75
    X = _zeros_feature_tensor(colmap, seq_len)

    # Pattern: 25 frames hitstun, 5 frames break, 25 frames hitstun
    X[0, 0:25, idx.p2_is_in_hitstun] = 1.0  # First combo
    X[0, 25:30, idx.p2_is_in_hitstun] = 0.0  # Break
    X[0, 30:55, idx.p2_is_in_hitstun] = 1.0  # Second combo

    rewards = compute_frame_rewards(X, idx=idx, reward_cfg=cfg).squeeze(0)

    # Frame 24: streak of 25, should have reward
    assert rewards[24].item() > 0.0

    # Frames 25-29: no hitstun, no reward
    assert torch.allclose(rewards[25:30], torch.zeros(5), atol=1e-6)

    # Frame 30: first frame of second combo (streak=1), below min, no reward
    assert torch.allclose(rewards[30], torch.tensor(0.0), atol=1e-6)

    # Frame 40: streak of 11 in second combo, should have reward
    assert rewards[40].item() > 0.0

    # Second combo's peak reward should be lower than if it continued from first
    # (since streak resets)
    assert rewards[40].item() < rewards[24].item()


def test_compute_value_targets_sequence_gamma(reward_setup):
    """Discounted returns should honor the provided gamma factor."""
    colmap, _, reward_cfg = reward_setup
    seq_len = 4
    X = _zeros_feature_tensor(colmap, seq_len)
    reward_col = colmap.value_idx
    assert reward_col is not None

    stored_returns = torch.tensor([2.5, 1.5, 0.5, 0.25], dtype=torch.float32)
    X[0, :, reward_col] = stored_returns

    gamma = 0.9
    returns = (
        compute_value_targets(
            X, colmap, reward_cfg=reward_cfg.model_copy(update={"gamma": gamma}), reward_idx=reward_col
        )
        .squeeze(0)
        .squeeze(-1)
    )

    assert torch.allclose(returns, stored_returns, atol=1e-6)


def test_compute_value_targets_fallback_reward_computation(reward_setup):
    """Legacy datasets without stored rewards should recompute values on the fly."""
    colmap, idx, reward_cfg = reward_setup
    seq_len = 4
    X = _zeros_feature_tensor(colmap, seq_len)
    X[0, :, idx.p2_percent] = torch.tensor([0.0, 1.0, 1.0, 2.0])

    gamma = 0.9
    returns = (
        compute_value_targets(
            X,
            colmap,
            reward_cfg=reward_cfg.model_copy(update={"gamma": gamma}),
            reward_idx=None,
            reward_features=idx,
        )
        .squeeze(0)
        .squeeze(-1)
    )

    cfg = reward_cfg
    expected_rewards = torch.tensor(
        [
            float(cfg.reward_damage_dealt),
            0.0,
            float(cfg.reward_damage_dealt),
            0.0,
        ],
        dtype=torch.float32,
    )
    manual = torch.empty(seq_len)
    for t in range(seq_len):
        total = 0.0
        for k in range(t, seq_len):
            total += (gamma ** (k - t)) * expected_rewards[k]
        manual[t] = total

    assert torch.allclose(returns, manual, atol=1e-6)


def test_replay_rewards_shape_and_sparsity(replay_reward_data):
    rewards = replay_reward_data["rewards"]
    assert rewards.shape[0] == 6956
    # Updated count includes hitstun rewards (was 382 before hitstun rewards added)
    assert torch.count_nonzero(rewards).item() == 1450


@pytest.mark.parametrize(
    ("frame", "value"),
    [
        (31, -0.0784),
        (36, -0.0784),
        (
            6917,
            -1.0,
        ),  # Updated: action-based death detection at frame 6918 gives reward at 6917 (reward_stock_taken=1.0)
    ],
)
def test_replay_rewards_matches_known_frames(replay_reward_data, frame, value):
    rewards = replay_reward_data["rewards"]
    assert pytest.approx(value, abs=1e-6) == float(rewards[frame])


def test_replay_rewards_zero_sum_after_player_swap(replay_reward_data):
    rewards = replay_reward_data["rewards"]
    swapped = replay_reward_data["swapped_rewards"]
    assert torch.allclose(swapped, -rewards, atol=1e-6)
