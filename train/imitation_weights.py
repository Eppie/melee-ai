"""Value-based sample weighting for imitation learning.

This module computes per-frame weights based on the value_target feature to prioritize
learning from high-value states and critical decisions. Used to train models that imitate
winning play rather than average play.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from config.imitation_config import ImitationConfig


def compute_value_weighted_weights(
    values: Tensor,
    *,
    k: float,
    temperature: float,
    use_exp: bool,
) -> Tensor:
    """Weight samples by their value (higher value = higher weight).

    Args:
        values: [B, L] value targets
        k: Scaling factor (higher = more emphasis on high values)
        temperature: Temperature for softening distribution
        use_exp: If True, use exp(k*value); else use linear scaling

    Returns:
        [B, L] normalized weights
    """
    if use_exp:
        # Exponential weighting: strongly emphasizes high-value states
        # Subtract max for numerical stability
        scaled = k * values / temperature
        scaled = scaled - scaled.max()  # prevent overflow
        weights = torch.exp(scaled)
    else:
        # Linear weighting: gentler emphasis
        weights = (k * values / temperature).clamp_min(0.0)

    # Normalize to mean=1 to maintain loss scale
    return weights / (weights.mean() + 1e-8)


def compute_value_filter_weights(
    values: Tensor,
    *,
    percentile: float,
    soft: bool,
    temperature: float,
) -> Tensor:
    """Filter to only train on top percentile by value.

    Args:
        values: [B, L] value targets
        percentile: Percentile cutoff (e.g., 30 = only train on top 70%)
        soft: If True, use smooth sigmoid cutoff; else hard threshold
        temperature: Temperature for soft filtering

    Returns:
        [B, L] binary or smooth weights
    """
    # Compute threshold across entire batch
    threshold = torch.quantile(values.flatten(), percentile / 100.0)

    if soft:
        # Smooth sigmoid transition around threshold
        weights = torch.sigmoid((values - threshold) / temperature)
    else:
        # Hard cutoff: 1 if above threshold, 0 otherwise
        weights = (values >= threshold).float()

    return weights


def compute_advantage_weights(
    values: Tensor,
    *,
    n_steps: int,
    alpha: float,
    use_gae: bool = False,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tensor:
    """Weight by advantage (temporal difference in value).

    Frames where value increases (good decisions) get higher weight.

    Args:
        values: [B, L] value targets
        n_steps: Number of steps for TD error (if not using GAE)
        alpha: Scaling factor for advantage weighting
        use_gae: If True, use GAE; else simple n-step TD
        gamma: Discount factor for GAE
        gae_lambda: Lambda for GAE

    Returns:
        [B, L] weights based on advantage magnitude
    """
    B, L = values.shape

    if use_gae:
        # GAE advantage estimation (more sophisticated)
        advantages = torch.zeros_like(values)
        deltas = torch.zeros_like(values)

        # Compute TD errors
        deltas[:, :-1] = values[:, 1:] - values[:, :-1]

        # Compute GAE
        gae = torch.zeros(B, device=values.device, dtype=values.dtype)
        for t in reversed(range(L - 1)):
            gae = deltas[:, t] + gamma * gae_lambda * gae
            advantages[:, t] = gae

    else:
        # Simple n-step advantage: V(t) - V(t+n)
        # Positive advantage = value increased (good decision)
        advantages = torch.zeros_like(values)
        n = min(n_steps, L - 1)

        if n > 0:
            # Advantage = future value - current value
            # Positive = making progress, negative = losing ground
            advantages[:, :-n] = values[:, n:] - values[:, :-n]

    # Weight by absolute advantage (both improvements and mistakes are informative)
    # But prioritize improvements (positive advantage) more
    weights = torch.where(
        advantages > 0,
        1.0 + alpha * advantages,  # boost improvements
        1.0 + alpha * 0.5 * torch.abs(advantages),  # light weight on mistakes
    )

    return weights.clamp_min(0.0)


def compute_hybrid_weights(
    values: Tensor,
    config: ImitationConfig,
) -> Tensor:
    """Combine multiple weighting strategies.

    Args:
        values: [B, L] value targets
        config: Imitation config with hybrid settings

    Returns:
        [B, L] combined weights
    """
    if len(config.hybrid_strategies) != len(config.hybrid_weights):
        raise ValueError("hybrid_strategies and hybrid_weights must have same length")

    weights_list = []

    for strategy_name in config.hybrid_strategies:
        if strategy_name == "value_weighted":
            w = compute_value_weighted_weights(
                values,
                k=config.value_k,
                temperature=config.value_temperature,
                use_exp=config.value_use_exp,
            )
        elif strategy_name == "value_filter":
            w = compute_value_filter_weights(
                values,
                percentile=config.filter_percentile,
                soft=config.filter_soft,
                temperature=config.filter_temperature,
            )
        elif strategy_name == "value_advantage":
            w = compute_advantage_weights(
                values,
                n_steps=config.advantage_n_steps,
                alpha=config.advantage_alpha,
                use_gae=config.advantage_use_gae,
                gamma=config.gae_gamma,
                gae_lambda=config.gae_lambda,
            )
        elif strategy_name == "uniform":
            w = torch.ones_like(values)
        else:
            raise ValueError(f"Unknown strategy in hybrid: {strategy_name}")

        weights_list.append(w)

    # Combine strategies: multiply each weight by its hybrid coefficient, then multiply together
    combined = torch.ones_like(values)
    for w, coef in zip(weights_list, config.hybrid_weights):
        # Raise weight to power of coefficient (geometric mean when coefs sum to 1)
        combined = combined * torch.pow(w, coef)

    # Normalize to mean=1
    return combined / (combined.mean() + 1e-8)


def compute_imitation_weights(
    X: Tensor,
    value_idx: int,
    config: ImitationConfig,
) -> Tensor:
    """Compute per-sample weights based on imitation learning strategy.

    Args:
        X: [B, L, F] input features
        value_idx: Column index of value_target feature
        config: Imitation learning configuration

    Returns:
        [B, L] sample weights (normalized to mean=1)
    """
    # Extract value targets
    values = X[:, :, value_idx]  # [B, L]

    strategy = config.strategy

    if strategy == "uniform":
        # Baseline: no weighting
        return torch.ones_like(values)

    elif strategy == "value_weighted":
        return compute_value_weighted_weights(
            values,
            k=config.value_k,
            temperature=config.value_temperature,
            use_exp=config.value_use_exp,
        )

    elif strategy == "value_filter":
        return compute_value_filter_weights(
            values,
            percentile=config.filter_percentile,
            soft=config.filter_soft,
            temperature=config.filter_temperature,
        )

    elif strategy == "value_advantage":
        return compute_advantage_weights(
            values,
            n_steps=config.advantage_n_steps,
            alpha=config.advantage_alpha,
            use_gae=config.advantage_use_gae,
            gamma=config.gae_gamma,
            gae_lambda=config.gae_lambda,
        )

    elif strategy == "hybrid":
        return compute_hybrid_weights(values, config)

    else:
        raise ValueError(f"Unknown imitation strategy: {strategy}")
