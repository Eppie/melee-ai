from __future__ import annotations

from typing import Tuple

import pytest
import torch

from column_map import ColumnMap
from config.reward_config import RewardConfig
from schema import get_feature_names, get_target_names
from train.value_head import (
    RewardFeatureIdx,
    build_reward_feature_index,
    compute_frame_rewards,
    compute_reward_components,
)


def _build_colmap_and_idx() -> Tuple[ColumnMap, RewardFeatureIdx]:
    colmap = ColumnMap(get_feature_names(), get_target_names())
    idx = build_reward_feature_index(colmap)
    return colmap, idx


def _make_base_features(
    num_frames: int,
    colmap: ColumnMap,
    idx: RewardFeatureIdx,
) -> torch.Tensor:
    features = torch.zeros(
        (1, num_frames, len(colmap.feat_names)), dtype=torch.float32
    )
    assert idx.p1_action is not None
    assert idx.p2_action is not None
    assert idx.p1_shield_strength is not None
    assert idx.p2_shield_strength is not None
    features[:, :, idx.p1_action] = 20.0
    features[:, :, idx.p2_action] = 20.0
    features[:, :, idx.p1_shield_strength] = 0.5
    features[:, :, idx.p2_shield_strength] = 0.5
    return features


def test_reward_damage_positive_on_opponent_percent_increase() -> None:
    colmap, idx = _build_colmap_and_idx()
    assert idx.p2_percent is not None
    X = _make_base_features(num_frames=2, colmap=colmap, idx=idx)
    X[0, 0, idx.p2_percent] = 0.0
    X[0, 1, idx.p2_percent] = 0.125  # 12.5% scaled by 1/100

    cfg = RewardConfig()
    rewards = compute_frame_rewards(X, idx, cfg)

    assert rewards[0, 0].item() == pytest.approx(
        cfg.reward_damage_dealt * 0.125
    )
    assert rewards[0, 1].item() == pytest.approx(0.0)


def test_reward_damage_negative_on_self_percent_increase() -> None:
    colmap, idx = _build_colmap_and_idx()
    assert idx.p1_percent is not None
    X = _make_base_features(num_frames=2, colmap=colmap, idx=idx)
    X[0, 0, idx.p1_percent] = 0.0
    X[0, 1, idx.p1_percent] = 0.15  # 15% scaled by 1/100

    cfg = RewardConfig()
    rewards = compute_frame_rewards(X, idx, cfg)

    assert rewards[0, 0].item() == pytest.approx(
        -cfg.reward_damage_dealt * 0.15
    )
    assert rewards[0, 1].item() == pytest.approx(0.0)


def test_reward_stock_taken_on_death_transition() -> None:
    colmap, idx = _build_colmap_and_idx()
    assert idx.p2_action is not None
    cfg = RewardConfig()
    X = _make_base_features(num_frames=2, colmap=colmap, idx=idx)
    X[0, 0, idx.p2_action] = 20.0
    X[0, 1, idx.p2_action] = 5.0

    rewards = compute_frame_rewards(X, idx, cfg)

    assert rewards[0, 0].item() == pytest.approx(cfg.reward_stock_taken)
    assert rewards[0, 1].item() == pytest.approx(0.0)


def test_reward_hitlag_positive_when_opponent_in_hitlag() -> None:
    colmap, idx = _build_colmap_and_idx()
    assert idx.p2_is_in_hitlag is not None
    assert idx.p2_is_defender_in_hitlag is not None
    cfg = RewardConfig()
    X = _make_base_features(num_frames=2, colmap=colmap, idx=idx)
    X[0, 1, idx.p2_is_in_hitlag] = 1.0
    X[0, 1, idx.p2_is_defender_in_hitlag] = 0.0

    rewards = compute_frame_rewards(X, idx, cfg)

    assert rewards[0, 0].item() == pytest.approx(cfg.reward_hitlag_opponent)
    assert rewards[0, 1].item() == pytest.approx(0.0)


def test_reward_low_shield_penalty_applies() -> None:
    colmap, idx = _build_colmap_and_idx()
    assert idx.p1_shield_strength is not None
    cfg = RewardConfig()
    X = _make_base_features(num_frames=2, colmap=colmap, idx=idx)
    X[0, 1, idx.p1_shield_strength] = 0.0

    rewards = compute_frame_rewards(X, idx, cfg)

    assert rewards[0, 0].item() == pytest.approx(cfg.reward_low_shield)
    assert rewards[0, 1].item() == pytest.approx(0.0)


def test_reward_hitstun_curve_positive_for_opponent_hitstun() -> None:
    colmap, idx = _build_colmap_and_idx()
    assert idx.p2_is_in_hitstun is not None
    cfg = RewardConfig(
        reward_damage_dealt=0.0,
        reward_stock_taken=0.0,
        reward_hitlag_opponent=0.0,
        reward_low_shield=0.0,
        reward_hitstun_min_frames=1,
        reward_hitstun_peak_frames=2,
        reward_hitstun_max_frames=3,
        reward_hitstun_peak_value=0.04,
    )
    X = _make_base_features(num_frames=2, colmap=colmap, idx=idx)
    X[0, 0, idx.p2_is_in_hitstun] = 1.0
    X[0, 1, idx.p2_is_in_hitstun] = 1.0

    rewards = compute_frame_rewards(X, idx, cfg)

    assert rewards[0].sum().item() > 0.0


def test_reward_components_sum_matches_total() -> None:
    colmap, idx = _build_colmap_and_idx()
    assert idx.p2_percent is not None
    assert idx.p2_action is not None
    X = _make_base_features(num_frames=3, colmap=colmap, idx=idx)
    X[0, 0, idx.p2_percent] = 0.0
    X[0, 1, idx.p2_percent] = 0.10
    X[0, 2, idx.p2_percent] = 0.10
    X[0, 0, idx.p2_action] = 20.0
    X[0, 1, idx.p2_action] = 5.0
    X[0, 2, idx.p2_action] = 5.0

    cfg = RewardConfig()
    rewards = compute_frame_rewards(X, idx, cfg)
    components = compute_reward_components(X, idx, cfg)
    total_components = components["total"]

    assert torch.allclose(rewards, total_components)
