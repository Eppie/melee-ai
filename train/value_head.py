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
    """Resolve frequently accessed feature indices from a :class:`ColumnMap` in one pass.

    Example:
        If ``colmap.feat_names`` equals ``["p1_stock", "p2_stock", "p1_percent"]``, calling
        ``build_reward_feature_index`` returns ``RewardFeatureIdx(p1_stock=0, p2_stock=1,
        p1_percent=2, ...)`` while any missing names (such as ``p2_percent``) remain ``None``. The
        example demonstrates how the helper searches each feature name and records the integer
        position so later reward computations can index into tensors without repeated list lookups.

    Args:
        colmap: Column mapping that lists feature names in order.

    Returns:
        :class:`RewardFeatureIdx` populated with index values where available.
    """
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
    """Compute reward signals per frame using stock, damage, hitlag, and shield features.

    Example:
        Consider ``B=1`` and ``L=3`` with stocks ``[4, 4, 3]`` and opponent percent damage
        ``[0.10, 0.20, 0.25]`` (normalized 0–1). Using configuration weights
        ``reward_stock_taken = 2`` and ``reward_damage_dealt = 0.5``:

        #. ``torch.diff`` finds a stock drop of ``1`` at frame 2, so ``rw[:, 2]`` gains ``+2``.
        #. Damage deltas are ``[+0.10, +0.05]``; the second frame adds ``0.10 * 100 * 0.5 = 5`` and
           the third frame adds ``0.05 * 100 * 0.5 = 2.5``.
        #. Summing the base reward ``reward_per_frame`` (call it ``r``) with these bonuses yields a
           reward vector ``[r, r + 5, r + 2 + 2.5]``.

        The example showcases each tensor operation—diffs, clamps, and adds—and how they manipulate
        the inputs to produce the per-frame rewards.

    Args:
        X: ``[B, L, F]`` input feature tensor.
        colmap: Column mapping describing feature positions.
        idx: Optional cached feature indices from :func:`build_reward_feature_index`.

    Returns:
        ``[B, L]`` tensor of per-frame rewards.
    """
    B, L, _F = X.shape
    device = X.device
    dtype = X.dtype
    # TODO: Make this mandatory
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
    """Produce a hashable key describing the gamma power request for caching.

    Example:
        For ``length=4``, ``gamma=0.99``, ``device=torch.device('cuda', 0)``, and ``dtype=torch.float32``
        the function returns ``(4, 0.99, torch.float32, 'cuda', 0)``. Requesting gamma powers with the
        same inputs later reuses the cached tensor because the key matches exactly. This example shows
        how each argument contributes to the key tuple.

    Args:
        length: Number of time steps required.
        gamma: Discount factor.
        device: Target device for the cached tensor.
        dtype: Desired floating-point dtype.

    Returns:
        Tuple uniquely identifying the gamma power request.
    """
    dev = torch.device(device)
    return (
        int(length),
        float(gamma),
        dtype,
        dev.type,
        dev.index if dev.index is not None else -1,
    )


# TODO: Can we pre-compute this and save a branch?
def _get_gamma_powers(
    length: int, gamma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return the vector ``[1, gamma, gamma^2, ...]`` either from cache or by recomputation.

    Example:
        Requesting ``_get_gamma_powers(3, 0.9, cpu_device, torch.float32)`` first builds the cache key
        ``(3, 0.9, torch.float32, 'cpu', -1)``. If absent, it creates the tensor
        ``tensor([1.0, 0.9, 0.81])`` via ``torch.pow`` and stores it in ``_GAMMA_POW_CACHE``. A second
        call with the same arguments returns the cached tensor without recomputing. The example
        highlights the control flow between cache hits and misses.

    Args:
        length: Number of gamma powers needed.
        gamma: Discount factor.
        device: Target device for the returned tensor.
        dtype: Desired floating-point dtype of the result.

    Returns:
        Tensor of shape ``[length]`` containing successive gamma powers.
    """
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
    """Compute discounted returns by summing future rewards with geometric decay.

    Example:
        Suppose ``compute_frame_rewards`` yields ``[[1.0, 2.0, 3.0]]`` with ``gamma=0.9``. The helper
        obtains gamma powers ``[1.0, 0.9, 0.81]`` and performs a reversed cumulative sum:

        * Weighted rewards become ``[[1.0, 1.8, 2.43]]``.
        * The reversed ``cumsum`` generates ``[[5.23, 4.23, 2.43]]``.
        * Dividing by the gamma powers recovers the standard discounted returns
          ``[[5.23, 4.7, 3.0]]``.

        Finally the method adds the optional terminal bonus (contributing ``[0.81, 0.9, 1.0]`` in this
        example) and unsqueezes the last dimension to produce ``[[[6.04], [5.6], [4.0]]]``. This
        detailed walkthrough mirrors the tensor manipulations used in the implementation.

    Args:
        X: ``[B, L, F]`` input features.
        colmap: Column mapping describing feature positions.
        gamma: Discount factor used for future rewards.
        reward_idx: Optional cached feature indices for faster reward computation.

    Returns:
        ``[B, L, 1]`` tensor of discounted returns.
    """
    B, L, _ = X.shape
    device = X.device
    dtype = X.dtype

    if L == 0:
        return torch.empty((B, 0, 1), device=device, dtype=dtype)

    rewards = compute_frame_rewards(X, colmap, idx=reward_idx)  # [B, L]
    gamma_val = float(gamma)
    gamma_powers = _get_gamma_powers(L, gamma_val, device, rewards.dtype)

    # TODO: Do we really need this?
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
