"""RL/value head utilities for computing rewards and value targets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from column_map import ColumnMap
from config.reward_config import RewardConfig


@dataclass(frozen=True)
class RewardFeatureIdx:
    """Cached indices for reward computation features."""

    p1_action: Optional[int] = None
    p2_action: Optional[int] = None
    p1_percent: Optional[int] = None
    p2_percent: Optional[int] = None
    p1_is_in_hitlag: Optional[int] = None
    p2_is_in_hitlag: Optional[int] = None
    p1_is_defender_in_hitlag: Optional[int] = None
    p2_is_defender_in_hitlag: Optional[int] = None
    p1_shield_strength: Optional[int] = None
    p2_shield_strength: Optional[int] = None
    p1_is_in_hitstun: Optional[int] = None
    p2_is_in_hitstun: Optional[int] = None


def build_reward_feature_index(column_map: ColumnMap) -> RewardFeatureIdx:
    """Resolve frequently accessed feature indices from a :class:`ColumnMap` in one pass.

    Example:
        If ``colmap.feat_names`` equals ``["p1_action", "p2_action", "p1_percent"]``, calling
        ``build_reward_feature_index`` returns ``RewardFeatureIdx(p1_action=0, p2_action=1,
        p1_percent=2, ...)`` while any missing names (such as ``p2_percent``) remain ``None``. The
        example demonstrates how the helper searches each feature name and records the integer
        position so later reward computations can index into tensors without repeated list lookups.

    Args:
        column_map: Column mapping that lists feature names in order.

    Returns:
        :class:`RewardFeatureIdx` populated with index values where available.
    """
    names = column_map.feat_names

    def idx(name: str) -> Optional[int]:
        return names.index(name)

    return RewardFeatureIdx(
        p1_action=idx("p1_action"),
        p2_action=idx("p2_action"),
        p1_percent=idx("p1_percent"),
        p2_percent=idx("p2_percent"),
        p1_is_in_hitlag=idx("p1_is_in_hitlag"),
        p2_is_in_hitlag=idx("p2_is_in_hitlag"),
        p1_is_defender_in_hitlag=idx("p1_is_defender_in_hitlag"),
        p2_is_defender_in_hitlag=idx("p2_is_defender_in_hitlag"),
        p1_shield_strength=idx("p1_shield_strength"),
        p2_shield_strength=idx("p2_shield_strength"),
        p1_is_in_hitstun=idx("p1_is_in_hitstun"),
        p2_is_in_hitstun=idx("p2_is_in_hitstun"),
    )


def _compute_hitstun_streak_lengths(opponent_in_hitstun: torch.Tensor) -> torch.Tensor:
    """Compute the length of consecutive hitstun sequences at each position.

    For each frame, calculates how many consecutive frames of hitstun have occurred
    up to and including that frame. Resets to 0 when hitstun ends.

    Example:
        Given ``opponent_in_hitstun = [[False, True, True, True, False, True, True]]``,
        returns ``[[0, 1, 2, 3, 0, 1, 2]]``.

    Args:
        opponent_in_hitstun: ``[B, L]`` boolean tensor indicating hitstun status.

    Returns:
        ``[B, L]`` tensor where each position contains the current consecutive
        hitstun streak length (0 if not in hitstun).
    """
    B, L = opponent_in_hitstun.shape
    device = opponent_in_hitstun.device
    dtype = torch.float32

    # Convert boolean to float for computation
    hitstun_float = opponent_in_hitstun.to(dtype=dtype)

    # Initialize output tensor
    streak_lengths = torch.zeros((B, L), device=device, dtype=dtype)

    # Compute streak lengths sequentially
    # streak[t] = (streak[t-1] + 1) * hitstun[t]
    # This gives 0 when not in hitstun, and increments when in hitstun
    for i in range(L):
        if i == 0:
            streak_lengths[:, i] = hitstun_float[:, i]
        else:
            streak_lengths[:, i] = (streak_lengths[:, i - 1] + 1) * hitstun_float[:, i]

    return streak_lengths


def _compute_hitstun_curve_reward(
    streak_length: torch.Tensor,
    min_frames: int,
    peak_frames: int,
    max_frames: int,
    peak_reward: float,
) -> torch.Tensor:
    """Compute per-frame reward based on hitstun streak length using a piecewise linear curve.

    Implements a reward curve with three phases:
    1. ``[0, min_frames)``: No reward (filters out brief hits)
    2. ``[min_frames, peak_frames]``: Linear increase from 0 to peak_reward
    3. ``(peak_frames, max_frames]``: Linear decrease from peak_reward to 0
    4. ``(max_frames, inf)``: No reward (prevents infinite accumulation)

    Example:
        With ``min_frames=15``, ``peak_frames=400``, ``max_frames=600``, ``peak_reward=0.04``:
        - Streak of 10 frames: reward = 0.0
        - Streak of 200 frames: reward ≈ 0.019 (midway to peak)
        - Streak of 400 frames: reward = 0.04 (peak)
        - Streak of 500 frames: reward ≈ 0.02 (midway down)
        - Streak of 600+ frames: reward = 0.0

    Args:
        streak_length: ``[B, L]`` tensor of consecutive hitstun frame counts.
        min_frames: Minimum consecutive frames before reward starts.
        peak_frames: Frame count where reward reaches maximum.
        max_frames: Frame count where reward returns to zero.
        peak_reward: Maximum per-frame reward value at peak.

    Returns:
        ``[B, L]`` tensor of per-frame rewards.
    """
    reward = torch.zeros_like(streak_length)

    # Phase 1: Increasing phase [min_frames, peak_frames]
    increasing_mask = (streak_length >= min_frames) & (streak_length <= peak_frames)
    if increasing_mask.any():
        # Linear interpolation: progress from 0 to 1
        progress = (streak_length - min_frames) / max(1, peak_frames - min_frames)
        reward = torch.where(increasing_mask, progress * peak_reward, reward)

    # Phase 2: Decreasing phase (peak_frames, max_frames]
    decreasing_mask = (streak_length > peak_frames) & (streak_length <= max_frames)
    if decreasing_mask.any():
        # Linear interpolation: progress from 1 to 0
        progress = (max_frames - streak_length) / max(1, max_frames - peak_frames)
        reward = torch.where(decreasing_mask, progress * peak_reward, reward)

    return reward


def _compute_player_rewards(
    X: torch.Tensor,
    idx: RewardFeatureIdx,
    reward_cfg: RewardConfig,
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

    if L <= 1:
        return rewards

    prev_slice = slice(None, -1)  # indices 0 .. L-2
    curr_slice = slice(1, None)  # indices 1 .. L-1

    if player == "p1":
        # Current player deals damage to opponent
        opp_percent_idx = idx.p2_percent
        # Opponent dying
        opp_action_idx = idx.p2_action
        # Hitlag rewards/penalties
        opp_hitlag_idx = idx.p2_is_in_hitlag
        opp_def_hitlag_idx = idx.p2_is_defender_in_hitlag
        # Shield penalty
        shield_idx = idx.p1_shield_strength
    else:  # player == "p2"
        # Current player deals damage to opponent
        opp_percent_idx = idx.p1_percent
        # Opponent dying
        opp_action_idx = idx.p1_action
        # Hitlag rewards/penalties
        opp_hitlag_idx = idx.p1_is_in_hitlag
        opp_def_hitlag_idx = idx.p1_is_defender_in_hitlag
        # Shield penalty
        shield_idx = idx.p2_shield_strength

    # --- Damage deltas (current player deals damage to opponent/opponent to player) ---
    d_opp = torch.diff(X[:, :, opp_percent_idx], dim=1)  # [B, L-1]
    d_opp.clamp_min_(0.0)
    rewards[:, prev_slice].add_(d_opp.mul_(reward_cfg.reward_damage_dealt))

    # --- Death detection using action state (opponent dying)
    # Action state IDs 0-10 (0x00-0x0A) are death states
    opp_action_idx = getattr(idx, f"{opponent}_action")
    opp_action = X[:, :, opp_action_idx]  # [B, L]
    opp_is_dying = opp_action <= 0x0A  # [B, L]

    # Detect transitions from living to dying (stock loss)
    # deaths[t] = not dying at t-1 AND dying at t
    opp_deaths = torch.logical_and(
        torch.logical_not(opp_is_dying[:, :-1]),  # not dying at prev frame
        opp_is_dying[:, 1:],  # dying at current frame
    )  # [B, L-1]

    # Reward for taking opponent's stock
    stock_taken = opp_deaths.to(dtype)  # Convert bool to float
    rewards[:, prev_slice].add_(stock_taken.mul_(reward_cfg.reward_stock_taken))

    # --- Hitlag rewards/penalties (per-frame) ---
    opp_hitlag_idx = getattr(idx, f"{opponent}_is_in_hitlag")
    opp_def_hitlag_idx = getattr(idx, f"{opponent}_is_defender_in_hitlag")
    opp_metric = X[:, :, opp_hitlag_idx] - X[:, :, opp_def_hitlag_idx]
    hitlag_reward = (opp_metric == 1).to(dtype).mul_(reward_cfg.reward_hitlag_opponent)
    rewards[:, prev_slice].add_(hitlag_reward[:, curr_slice])

    # --- Shield penalty (per-frame) ---
    shield_idx = getattr(idx, f"{player}_shield_strength")
    shield = X[:, :, shield_idx]
    penalty = (1.0 - 2.0 * shield).clamp_min_(0.0).clamp_max_(1.0)
    rewards[:, prev_slice].add_(penalty[:, curr_slice].mul_(reward_cfg.reward_low_shield))

    # --- Hitstun combo reward (per-frame, with length-based curve) ---
    opp_hitstun_idx = getattr(idx, f"{opponent}_is_in_hitstun")
    if opp_hitstun_idx is not None:
        opp_in_hitstun = X[:, :, opp_hitstun_idx] > 0.5  # [B, L] boolean
        # Compute consecutive hitstun streak lengths
        streak_lengths = _compute_hitstun_streak_lengths(opp_in_hitstun)  # [B, L]
        # Apply configurable reward curve based on streak length
        hitstun_reward = _compute_hitstun_curve_reward(
            streak_lengths,
            min_frames=reward_cfg.reward_hitstun_min_frames,
            peak_frames=reward_cfg.reward_hitstun_peak_frames,
            max_frames=reward_cfg.reward_hitstun_max_frames,
            peak_reward=reward_cfg.reward_hitstun_peak_value,
        )
        # Add to rewards for all frames (not just prev_slice, since this is per-frame)
        rewards.add_(hitstun_reward.to(dtype=dtype))

    return rewards


def _compute_player_reward_components(
    X: torch.Tensor,
    idx: RewardFeatureIdx,
    reward_cfg: RewardConfig,
    *,
    player: str,
) -> Dict[str, torch.Tensor]:
    """Compute per-frame reward components from the perspective of a single player."""
    if player not in ("p1", "p2"):
        raise ValueError(f"player must be 'p1' or 'p2', got {player!r}")

    opponent = "p2" if player == "p1" else "p1"

    B, L, _F = X.shape
    device = X.device
    dtype = X.dtype

    components = {
        "damage": torch.zeros(B, L, device=device, dtype=dtype),
        "stock": torch.zeros(B, L, device=device, dtype=dtype),
        "hitlag": torch.zeros(B, L, device=device, dtype=dtype),
        "low_shield": torch.zeros(B, L, device=device, dtype=dtype),
        "hitstun": torch.zeros(B, L, device=device, dtype=dtype),
    }

    if L <= 1:
        return components

    prev_slice = slice(None, -1)  # indices 0 .. L-2
    curr_slice = slice(1, None)  # indices 1 .. L-1

    if player == "p1":
        opp_percent_idx = idx.p2_percent
        opp_action_idx = idx.p2_action
        opp_hitlag_idx = idx.p2_is_in_hitlag
        opp_def_hitlag_idx = idx.p2_is_defender_in_hitlag
        shield_idx = idx.p1_shield_strength
        opp_hitstun_idx = idx.p2_is_in_hitstun
    else:
        opp_percent_idx = idx.p1_percent
        opp_action_idx = idx.p1_action
        opp_hitlag_idx = idx.p1_is_in_hitlag
        opp_def_hitlag_idx = idx.p1_is_defender_in_hitlag
        shield_idx = idx.p2_shield_strength
        opp_hitstun_idx = idx.p1_is_in_hitstun

    d_opp = torch.diff(X[:, :, opp_percent_idx], dim=1)  # [B, L-1]
    d_opp.clamp_min_(0.0)
    components["damage"][:, prev_slice].add_(
        d_opp.mul_(reward_cfg.reward_damage_dealt)
    )

    opp_action = X[:, :, opp_action_idx]  # [B, L]
    opp_is_dying = opp_action <= 0x0A  # [B, L]
    opp_deaths = torch.logical_and(
        torch.logical_not(opp_is_dying[:, :-1]), opp_is_dying[:, 1:]
    )
    components["stock"][:, prev_slice].add_(
        opp_deaths.to(dtype).mul_(reward_cfg.reward_stock_taken)
    )

    opp_hitlag = X[:, :, opp_hitlag_idx]
    opp_def_hitlag = X[:, :, opp_def_hitlag_idx]
    opp_metric = opp_hitlag - opp_def_hitlag
    hitlag_reward = (opp_metric == 1).to(dtype).mul_(reward_cfg.reward_hitlag_opponent)
    components["hitlag"][:, prev_slice].add_(hitlag_reward[:, curr_slice])

    shield = X[:, :, shield_idx]
    penalty = (1.0 - 2.0 * shield).clamp_min_(0.0).clamp_max_(1.0)
    components["low_shield"][:, prev_slice].add_(
        penalty[:, curr_slice].mul_(reward_cfg.reward_low_shield)
    )

    if opp_hitstun_idx is not None:
        opp_in_hitstun = X[:, :, opp_hitstun_idx] > 0.5
        streak_lengths = _compute_hitstun_streak_lengths(opp_in_hitstun)
        hitstun_reward = _compute_hitstun_curve_reward(
            streak_lengths,
            min_frames=reward_cfg.reward_hitstun_min_frames,
            peak_frames=reward_cfg.reward_hitstun_peak_frames,
            max_frames=reward_cfg.reward_hitstun_max_frames,
            peak_reward=reward_cfg.reward_hitstun_peak_value,
        )
        components["hitstun"].add_(hitstun_reward.to(dtype=dtype))

    return components


def compute_frame_rewards(
    X: torch.Tensor, idx: RewardFeatureIdx, reward_cfg: RewardConfig
) -> torch.Tensor:
    """Compute zero-sum per-frame rewards as ego minus opponent reward.

    Example:
        The helper first computes rewards for the ego player ``p1`` (damage dealt, stocks taken,
        shield penalties, etc.). It then computes the same quantity from the opponent's perspective
        (treating ``p2`` as ego) and returns their difference. The resulting tensor is guaranteed to
        be zero-sum: swapping ``p1`` and ``p2`` negates the reward signal.

    Args:
        X: ``[B, L, F]`` input feature tensor.
        idx: Optional cached feature indices from :func:`build_reward_feature_index`.
        reward_cfg: Shared reward parameters (shaping weights and discount factor).

    Returns:
        ``[B, L]`` tensor of per-frame rewards.
    """
    ego_rewards = _compute_player_rewards(X, idx, reward_cfg, player="p1")
    opp_rewards = _compute_player_rewards(X, idx, reward_cfg, player="p2")

    return ego_rewards - opp_rewards


def compute_reward_components(
    X: torch.Tensor, idx: RewardFeatureIdx, reward_cfg: RewardConfig
) -> Dict[str, torch.Tensor]:
    """Compute zero-sum per-frame reward components.

    Returns per-component tensors with shape ``[B, L]`` and a ``total`` entry.
    """
    ego = _compute_player_reward_components(X, idx, reward_cfg, player="p1")
    opp = _compute_player_reward_components(X, idx, reward_cfg, player="p2")

    components: Dict[str, torch.Tensor] = {}
    for key in ego:
        components[key] = ego[key] - opp[key]

    components["total"] = sum(components.values())
    return components


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


# TODO: Separate this into two functions:
#   one for pulling the value from the dataset
#   one for pre-computing it to store in the dataset


def compute_value_targets(
    X: torch.Tensor,
    colmap: ColumnMap,
    reward_cfg: RewardConfig,
    reward_idx: Optional[int],
    *,
    reward_features: Optional[RewardFeatureIdx] = None,
) -> torch.Tensor:
    """Compute discounted returns by summing future rewards with geometric decay.

    Example:
        Suppose ``compute_frame_rewards`` yields ``[[1.0, 2.0, 3.0]]`` with ``reward_cfg.gamma=0.9``.
        The helper obtains gamma powers ``[1.0, 0.9, 0.81]`` and performs a reversed cumulative sum:

        * Weighted rewards become ``[[1.0, 1.8, 2.43]]``.
        * The reversed ``cumsum`` generates ``[[5.23, 4.23, 2.43]]``.
        * Dividing by the gamma powers recovers the discounted returns ``[[5.23, 4.7, 3.0]]``.

    Args:
        X: ``[B, L, F]`` input features.
        colmap: Column mapping describing feature positions.
        reward_cfg: Shared reward configuration (provides reward weights and ``gamma``).
        reward_idx: Column index of a precomputed discounted return (e.g., dataset-stored value
            targets). When ``None`` the helper recomputes per-frame rewards and discounts them.
        reward_features: Optional cached reward feature indices for the fallback path to avoid
            rebuilding them on every call.

    Returns:
        ``[B, L, 1]`` tensor of discounted returns.
    """
    B, L, _ = X.shape
    device = X.device
    dtype = X.dtype

    if reward_idx is not None:
        stored = X[..., reward_idx].unsqueeze(-1)
        return stored.to(device=device, dtype=dtype)

    reward_features = reward_features or build_reward_feature_index(colmap)
    rewards = compute_frame_rewards(X, idx=reward_features, reward_cfg=reward_cfg)
    gamma_powers = _get_gamma_powers(L, reward_cfg.gamma, device, rewards.dtype)

    weighted = rewards * gamma_powers  # broadcast multiply
    discounted = torch.cumsum(weighted.flip(1), dim=1).flip(1)
    returns = discounted / gamma_powers.clamp_min(1e-12)

    return returns.unsqueeze(-1)  # [B, L, 1]
