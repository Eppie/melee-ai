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
    p1_is_in_hitlag: Optional[int] = None
    p2_is_in_hitlag: Optional[int] = None
    p1_is_defender_in_hitlag: Optional[int] = None
    p2_is_defender_in_hitlag: Optional[int] = None
    p1_shield_strength: Optional[int] = None
    p2_shield_strength: Optional[int] = None


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
        p1_is_in_hitlag=idx("p1_is_in_hitlag"),
        p2_is_in_hitlag=idx("p2_is_in_hitlag"),
        p1_is_defender_in_hitlag=idx("p1_is_defender_in_hitlag"),
        p2_is_defender_in_hitlag=idx("p2_is_defender_in_hitlag"),
        p1_shield_strength=idx("p1_shield_strength"),
        p2_shield_strength=idx("p2_shield_strength"),
    )


def _compute_player_rewards(
    X: torch.Tensor,
    idx: RewardFeatureIdx,
    cfg,
    *,
    player: str,
) -> torch.Tensor:
    """Compute per-frame rewards from the perspective of a single player.

    Args:
        X: ``[B, L, F]`` feature tensor.
        idx: Cached feature indices.
        cfg: RL configuration namespace with reward weights.
        player: Either ``\"p1\"`` or ``\"p2\"`` indicating the ego player.

    Returns:
        ``[B, L]`` tensor containing that player's frame rewards.
    """
    if player not in ("p1", "p2"):
        raise ValueError(f"player must be 'p1' or 'p2', got {player!r}")

    opponent = "p2" if player == "p1" else "p1"

    B, L, _F = X.shape
    device = X.device
    dtype = X.dtype

    rewards = torch.zeros(B, L, device=device, dtype=dtype)

    if L > 1:
        # --- Damage deltas (current player deals damage to opponent/opponent to player) ---
        opp_percent_idx = getattr(idx, f"{opponent}_percent")

        d_opp = torch.diff(X[:, :, opp_percent_idx], dim=1)  # [B, L-1]
        rewards[:, 1:].add_(d_opp.mul_(100.0).mul_(cfg.reward_damage_dealt))


        # --- Stock changes ---
        opp_stock_idx = getattr(idx, f"{opponent}_stock")
        d_opp_stock = torch.diff(X[:, :, opp_stock_idx], dim=1)
        stock_taken = (-d_opp_stock).clamp_min_(0)
        rewards[:, 1:].add_(stock_taken.mul_(cfg.reward_stock_taken))

    # --- Hitlag rewards/penalties (per-frame) ---
    opp_hitlag_idx = getattr(idx, f"{opponent}_is_in_hitlag")
    opp_def_hitlag_idx = getattr(idx, f"{opponent}_is_defender_in_hitlag")
    opp_metric = X[:, :, opp_hitlag_idx] - X[:, :, opp_def_hitlag_idx]
    rewards.add_((opp_metric == 1).to(dtype).mul_(cfg.reward_hitlag_opponent))

    # --- Shield penalty (per-frame) ---
    shield_idx = getattr(idx, f"{player}_shield_strength")
    shield = X[:, :, shield_idx]
    penalty = (1.0 - 2.0 * shield).clamp_min_(0.0).clamp_max_(1.0)
    rewards.add_(penalty.mul_(cfg.reward_low_shield))

    return rewards


def compute_frame_rewards(
    X: torch.Tensor,
    colmap: ColumnMap,
    *,
    idx: Optional[RewardFeatureIdx] = None,
) -> torch.Tensor:
    """Compute zero-sum per-frame rewards as ego minus opponent reward.

    Example:
        The helper first computes rewards for the ego player ``p1`` (damage dealt, stocks taken,
        shield penalties, etc.). It then computes the same quantity from the opponent's perspective
        (treating ``p2`` as ego) and returns their difference. The resulting tensor is guaranteed to
        be zero-sum: swapping ``p1`` and ``p2`` negates the reward signal.

    Args:
        X: ``[B, L, F]`` input feature tensor.
        colmap: Column mapping describing feature positions.
        idx: Optional cached feature indices from :func:`build_reward_feature_index`.

    Returns:
        ``[B, L]`` tensor of per-frame rewards.
    """
    # TODO: Make this mandatory
    if idx is None:
        # Resolve on the fly (still cheap), or pass a cached `idx` from caller for max perf.
        idx = build_reward_feature_index(colmap)

    cfg = get_config().rl

    ego_rewards = _compute_player_rewards(X, idx, cfg, player="p1")
    opp_rewards = _compute_player_rewards(X, idx, cfg, player="p2")

    return ego_rewards - opp_rewards


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
    gamma: float,
    reward_idx: RewardFeatureIdx,
) -> torch.Tensor:
    """Compute discounted returns by summing future rewards with geometric decay.

    Example:
        Suppose ``compute_frame_rewards`` yields ``[[1.0, 2.0, 3.0]]`` with ``gamma=0.9``. The helper
        obtains gamma powers ``[1.0, 0.9, 0.81]`` and performs a reversed cumulative sum:

        * Weighted rewards become ``[[1.0, 1.8, 2.43]]``.
        * The reversed ``cumsum`` generates ``[[5.23, 4.23, 2.43]]``.
        * Dividing by the gamma powers recovers the discounted returns ``[[5.23, 4.7, 3.0]]``.

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
    gamma_powers = _get_gamma_powers(L, gamma, device, rewards.dtype)

    weighted = rewards * gamma_powers  # broadcast multiply
    discounted = torch.cumsum(weighted.flip(1), dim=1).flip(1)
    returns = discounted / gamma_powers.clamp_min(1e-12)

    return returns.unsqueeze(-1)  # [B, L, 1]
