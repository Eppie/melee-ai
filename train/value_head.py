"""RL/value head utilities for computing rewards and value targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from column_map import ColumnMap
from config import get_config


@dataclass(frozen=True)
class RewardFeatureIdx:
    """Cached indices for reward computation features."""

    p1_stock: Optional[int] = None
    p2_stock: Optional[int] = None
    p1_percent: Optional[int] = None
    p2_percent: Optional[int] = None
    p1_in_hitlag: Optional[int] = None
    p2_in_hitlag: Optional[int] = None
    p1_in_defender_hitlag: Optional[int] = None
    p2_in_defender_hitlag: Optional[int] = None
    p1_shield_strength: Optional[int] = None


def build_reward_feature_index(colmap: ColumnMap) -> RewardFeatureIdx:
    """Resolve feature indices once and reuse; avoids per-call .index() overhead."""
    names = colmap.feat_names

    def idx(name: str) -> Optional[int]:
        return names.index(name) if name in names else None

    return RewardFeatureIdx(
        p1_stock=idx("p1_stock"),
        p2_stock=idx("p2_stock"),
        p1_percent=idx("p1_percent"),
        p2_percent=idx("p2_percent"),
        p1_in_hitlag=idx("p1_in_hitlag"),
        p2_in_hitlag=idx("p2_in_hitlag"),
        p1_in_defender_hitlag=idx("p1_in_defender_hitlag"),
        p2_in_defender_hitlag=idx("p2_in_defender_hitlag"),
        p1_shield_strength=idx("p1_shield_strength"),
    )


def compute_frame_rewards(
    X: torch.Tensor,
    colmap: ColumnMap,
    *,
    idx: Optional[RewardFeatureIdx] = None,
) -> torch.Tensor:
    """Compute per-frame rewards based on game state changes.

    Vectorized & allocation-lean version for efficient computation.

    Args:
        X: [B, L, F] input features
        colmap: Column mapping
        idx: Optional pre-computed reward feature indices

    Returns:
        [B, L] reward tensor
    """
    B, L, _F = X.shape
    device = X.device
    dtype = X.dtype

    if idx is None:
        # Resolve on the fly (still cheap), or pass a cached `idx` from caller for max perf.
        idx = build_reward_feature_index(colmap)

    cfg = get_config().rl
    rw = torch.full(
        (B, L), float(cfg.reward_per_frame), device=device, dtype=dtype
    )  # base per-frame reward

    if L > 1:
        # --- Damage deltas (vectorized with torch.diff) ---
        if (idx.p1_percent is not None) and (idx.p2_percent is not None):
            # deltas are current - previous over time dim
            d_p2 = torch.diff(X[:, :, idx.p2_percent], dim=1)  # [B, L-1]
            d_p1 = torch.diff(X[:, :, idx.p1_percent], dim=1)  # [B, L-1]
            # Scale percents from [0,1] back to 0..100 if that matches your preprocessing
            # Then apply weights, in-place add to t>=1 frames
            if float(cfg.reward_damage_dealt) != 0.0:
                rw[:, 1:].add_(d_p2.mul_(100.0).mul_(float(cfg.reward_damage_dealt)))
            if float(cfg.reward_damage_taken) != 0.0:
                rw[:, 1:].add_(d_p1.mul_(100.0).mul_(float(cfg.reward_damage_taken)))

        # --- Stock changes (vectorized) ---
        if (idx.p1_stock is not None) and (idx.p2_stock is not None):
            d_p2_stock = torch.diff(X[:, :, idx.p2_stock], dim=1)  # [B,L-1]
            d_p1_stock = torch.diff(X[:, :, idx.p1_stock], dim=1)  # [B,L-1]
            # Lost stock ⇒ delta = -1. Clamp the positive part of (-delta)
            stock_taken = (-d_p2_stock).clamp_min_(0)  # opponent lost stock
            stock_lost = (-d_p1_stock).clamp_min_(0)  # we lost stock
            if float(cfg.reward_stock_taken) != 0.0:
                rw[:, 1:].add_(stock_taken.mul_(float(cfg.reward_stock_taken)))
            if float(cfg.reward_stock_lost) != 0.0:
                rw[:, 1:].add_(stock_lost.mul_(float(cfg.reward_stock_lost)))

    # --- Hitlag rewards/penalties (no diffs, per-frame) ---
    if (
        (idx.p1_in_hitlag is not None)
        and (idx.p1_in_defender_hitlag is not None)
        and (idx.p2_in_hitlag is not None)
        and (idx.p2_in_defender_hitlag is not None)
    ):
        # metric: in_hitlag - in_defender_hitlag; reward when equals 1
        p1_metric = X[:, :, idx.p1_in_hitlag] - X[:, :, idx.p1_in_defender_hitlag]
        p2_metric = X[:, :, idx.p2_in_hitlag] - X[:, :, idx.p2_in_defender_hitlag]

        if float(cfg.reward_hitlag_self) != 0.0:
            rw.add_((p1_metric == 1).to(dtype).mul_(float(cfg.reward_hitlag_self)))
        if float(cfg.reward_hitlag_opponent) != 0.0:
            rw.add_((p2_metric == 1).to(dtype).mul_(float(cfg.reward_hitlag_opponent)))

    # --- Shield penalty (fused math; no mask tensor needed) ---
    if idx.p1_shield_strength is not None and float(cfg.reward_low_shield) != 0.0:
        s = X[:, :, idx.p1_shield_strength]  # [B, L] in [0,1]
        # penalty_multiplier = clamp(1 - 2*shield, 0, 1)
        pen = (1.0 - 2.0 * s).clamp_min_(0.0).clamp_max_(1.0)
        rw.add_(pen.mul_(float(cfg.reward_low_shield)))

    return rw


# Cache for gamma powers to avoid recomputation
_GAMMA_POW_CACHE: Dict[Tuple[int, float, torch.dtype, str, int], torch.Tensor] = {}


def _gamma_cache_key(
    length: int, gamma: float, device: torch.device, dtype: torch.dtype
) -> Tuple[int, float, torch.dtype, str, int]:
    """Create cache key for gamma powers."""
    dev = torch.device(device)
    return (
        int(length),
        float(gamma),
        dtype,
        dev.type,
        dev.index if dev.index is not None else -1,
    )


def _get_gamma_powers(
    length: int, gamma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Get cached gamma powers or compute and cache them."""
    if length <= 0:
        return torch.empty((0,), device=device, dtype=dtype)

    key = _gamma_cache_key(length, gamma, device, dtype)
    cached = _GAMMA_POW_CACHE.get(key)
    if cached is not None and cached.device == device and cached.dtype == dtype:
        return cached

    compute_dtype = dtype
    if device.type == "cpu" and dtype == torch.float16:
        compute_dtype = torch.float32

    arange = torch.arange(length, device=device, dtype=compute_dtype)
    gamma_scalar = torch.as_tensor(gamma, device=device, dtype=compute_dtype)
    powers = torch.pow(gamma_scalar, arange)
    if compute_dtype != dtype:
        powers = powers.to(dtype=dtype)
    _GAMMA_POW_CACHE[key] = powers
    return powers


def compute_value_targets(
    X: torch.Tensor,
    colmap: ColumnMap,
    gamma: float = 0.995,
    *,
    reward_idx: Optional[RewardFeatureIdx] = None,
) -> torch.Tensor:
    """Compute discounted returns in O(B·L) using cached gamma powers and fused scans.

    Args:
        X: [B, L, F] input features
        colmap: Column mapping
        gamma: Discount factor
        reward_idx: Optional pre-computed reward feature indices

    Returns:
        [B, L, 1] value targets (discounted returns)
    """
    B, L, _ = X.shape
    device = X.device
    dtype = X.dtype

    if L == 0:
        return torch.empty((B, 0, 1), device=device, dtype=dtype)

    rewards = compute_frame_rewards(X, colmap, idx=reward_idx)  # [B, L]
    gamma_val = float(gamma)
    gamma_powers = _get_gamma_powers(L, gamma_val, device, rewards.dtype)

    if abs(gamma_val) < 1e-12:
        returns = rewards.clone()
    else:
        weighted = rewards * gamma_powers  # broadcast multiply
        discounted = torch.cumsum(weighted.flip(1), dim=1).flip(1)
        returns = discounted / gamma_powers.clamp_min(1e-12)

    terminal_bonus = 1.0
    terminal_vec = gamma_powers.flip(0)
    returns = returns + terminal_bonus * terminal_vec

    return returns.unsqueeze(-1)  # [B, L, 1]
