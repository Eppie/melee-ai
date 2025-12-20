from __future__ import annotations

import pytest
import torch

from column_map import ColumnMap
from config.reward_config import RewardConfig
from ppo_train import compute_transition_reward
from schema import get_feature_names, get_target_names
from train.value_head import build_reward_feature_index


def test_transition_reward_uses_prev_frame_signal() -> None:
    colmap = ColumnMap(get_feature_names(), get_target_names())
    idx = build_reward_feature_index(colmap)
    assert idx.p2_percent is not None
    assert idx.p1_action is not None
    assert idx.p2_action is not None

    cfg = RewardConfig(
        reward_stock_taken=0.0,
        reward_hitlag_opponent=0.0,
        reward_low_shield=0.0,
        reward_hitstun_peak_value=0.0,
    )

    features = torch.zeros((2, len(colmap.feat_names)), dtype=torch.float32)
    features[0, idx.p1_action] = 20.0
    features[0, idx.p2_action] = 20.0
    features[1, idx.p1_action] = 20.0
    features[1, idx.p2_action] = 20.0
    features[0, idx.p2_percent] = 0.0
    features[1, idx.p2_percent] = 0.10  # 10% scaled by 1/100

    reward = compute_transition_reward(
        features[0], features[1], idx, cfg, torch.device("cpu")
    )

    assert reward == pytest.approx(cfg.reward_damage_dealt * 0.10)
