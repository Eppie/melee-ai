"""
Compute auxiliary task targets from raw data (X tensor)
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO_ROOT))
import torch
from typing import Dict
from column_map import ColumnMap
from train.imitation_strategy import (
    UniformStrategy,
    ValueWeightedStrategy,
    ValueAdvantageStrategy,
    ValueFilterStrategy,
    HybridStrategy,
)


def compute_aux_targets(
    X: torch.Tensor, colmap: ColumnMap, config
) -> Dict[str, torch.Tensor]:
    """
    Compute all auxiliary task targets from the X tensor.

    Args:
        X: (B, L, F) input features
        colmap: column mapping for feature access
        config: global config with aux_tasks settings

    Returns:
        Dict with keys:
            - opponent_action: (B, L) - next opponent action index
            - damage_diff: (B, L, 1) - net damage over next N frames
            - action_effectiveness: (B, L, 1) - will action cause hitlag?
    """
    B, L, F = X.shape
    device = X.device
    aux_cfg = config.aux_tasks

    targets = {}

    # 1. Opponent action prediction (next frame's action)
    if aux_cfg.enable_opponent_action:
        try:
            p2_action_idx = colmap.feat_names.index("p2_action")
            # Target is the action at t+1 (shift left)
            opponent_actions = X[:, :, p2_action_idx].long()  # (B, L)
            # Shift: target[t] = action[t+1]
            targets["opponent_action"] = torch.zeros_like(opponent_actions)
            targets["opponent_action"][:, :-1] = opponent_actions[:, 1:]
            # Last frame has no target (we'll mask it out in loss)
        except ValueError:
            # If p2_action not in features, skip this task
            pass

    # 2. Damage differential prediction
    if aux_cfg.enable_damage_diff:
        try:
            p1_percent_idx = colmap.feat_names.index("p1_percent")
            p2_percent_idx = colmap.feat_names.index("p2_percent")
            n_frames = aux_cfg.damage_diff_n_frames

            p1_percent = X[:, :, p1_percent_idx]  # (B, L)
            p2_percent = X[:, :, p2_percent_idx]  # (B, L)

            # Pad to allow looking ahead
            p1_padded = torch.nn.functional.pad(p1_percent, (0, n_frames), value=0)
            p2_padded = torch.nn.functional.pad(p2_percent, (0, n_frames), value=0)

            # Compute damage differential: (p2_damage - p1_damage)
            # Positive = we dealt more damage, negative = we took more damage
            p1_future = p1_padded[:, n_frames : n_frames + L]
            p2_future = p2_padded[:, n_frames : n_frames + L]

            p2_damage_dealt = p2_future - p2_percent  # how much opponent took
            p1_damage_taken = p1_future - p1_percent  # how much we took

            damage_diff = p2_damage_dealt - p1_damage_taken  # (B, L)
            targets["damage_diff"] = damage_diff.unsqueeze(-1)  # (B, L, 1)

        except ValueError:
            pass

    # 3. Action effectiveness (will our action cause opponent hitlag?)
    if aux_cfg.enable_action_effectiveness:
        try:
            p2_hitlag_idx = colmap.feat_names.index("p2_hitlag_left")
            k_frames = aux_cfg.action_effectiveness_k_frames

            p2_hitlag = X[:, :, p2_hitlag_idx]  # (B, L)

            # Check if opponent will be in hitlag within next K frames
            # Create a sliding window view
            effectiveness = torch.zeros((B, L), device=device)

            for t in range(L):
                # Look ahead up to k_frames
                end = min(t + k_frames + 1, L)
                # If opponent has any hitlag in [t+1, t+k], action was effective
                future_hitlag = p2_hitlag[:, t + 1 : end]  # (B, window_size)
                effectiveness[:, t] = (future_hitlag > 0).any(dim=1).float()

            targets["action_effectiveness"] = effectiveness.unsqueeze(-1)  # (B, L, 1)

        except ValueError:
            pass

    return targets


def compute_imitation_weights(
    values: torch.Tensor, strategy_config, strategy_instance=None
) -> torch.Tensor:
    """
    Compute sample weights for imitation learning.

    Args:
        values: (B, L, 1) or (B, L) predicted values from value head
        strategy_config: config.imitation
        strategy_instance: optional pre-created strategy object

    Returns:
        weights: (B, L) sample weights
    """

    if strategy_instance is not None:
        return strategy_instance.compute_weights(values)

    # Create strategy from config
    strategy_type = strategy_config.strategy

    if strategy_type == "uniform":
        strategy = UniformStrategy()

    elif strategy_type == "value_weighted":
        strategy = ValueWeightedStrategy(
            k=strategy_config.value_k,
            temperature=strategy_config.value_temperature,
            use_exp=strategy_config.value_use_exp,
        )

    elif strategy_type == "value_advantage":
        strategy = ValueAdvantageStrategy(
            n_steps=strategy_config.advantage_n_steps,
            alpha=strategy_config.advantage_alpha,
            use_gae=strategy_config.advantage_use_gae,
            gamma=strategy_config.gae_gamma,
            lambda_=strategy_config.gae_lambda,
        )

    elif strategy_type == "value_filter":
        strategy = ValueFilterStrategy(
            percentile=strategy_config.filter_percentile,
            soft=strategy_config.filter_soft,
            temperature=strategy_config.filter_temperature,
        )

    elif strategy_type == "hybrid":
        # Create sub-strategies
        strategies = []
        for strat_name in strategy_config.hybrid_strategies:
            if strat_name == "value_weighted":
                strategies.append(
                    ValueWeightedStrategy(
                        k=strategy_config.value_k,
                        temperature=strategy_config.value_temperature,
                        use_exp=strategy_config.value_use_exp,
                    )
                )
            elif strat_name == "value_advantage":
                strategies.append(
                    ValueAdvantageStrategy(
                        n_steps=strategy_config.advantage_n_steps,
                        alpha=strategy_config.advantage_alpha,
                    )
                )
            elif strat_name == "value_filter":
                strategies.append(
                    ValueFilterStrategy(
                        percentile=strategy_config.filter_percentile,
                        soft=strategy_config.filter_soft,
                    )
                )

        strategy = HybridStrategy(
            strategies=strategies,
            weights=strategy_config.hybrid_weights,
        )

    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return strategy.compute_weights(values)
