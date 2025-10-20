#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Optional

import torch
import torch.utils.benchmark as benchmark

from config import init_config
from train import compute_value_targets, RewardFeatureIdx, compute_frame_rewards


def compute_value_targets_baseline(
        X: torch.Tensor,
        colmap,
        gamma: float = 0.99,
        *,
        reward_idx: Optional[RewardFeatureIdx] = None,
) -> torch.Tensor:
    """Baseline implementation copied from train.py prior to optimizations."""
    B, L, _ = X.shape
    device = X.device
    dtype = X.dtype

    rewards = compute_frame_rewards(X, colmap, idx=reward_idx)  # [B, L]

    G = torch.empty((B, L), device=device, dtype=dtype)
    next_G = torch.zeros((B,), device=device, dtype=dtype)
    for t in range(L - 1, -1, -1):
        cur = rewards[:, t].add(next_G.mul(gamma))
        G[:, t] = cur
        next_G = cur

    terminal = 1.0
    pow_vec = torch.pow(
        torch.tensor(gamma, device=device, dtype=dtype),
        torch.arange(L - 1, -1, -1, device=device, dtype=dtype),
    )
    G = G + pow_vec * terminal
    return G.unsqueeze(-1)


def build_reward_feature_index_stub() -> RewardFeatureIdx:
    """Construct a reward index covering the fields touched by compute_frame_rewards."""
    return RewardFeatureIdx(
        p1_stock=0,
        p2_stock=1,
        p1_percent=2,
        p2_percent=3,
        p1_in_hitlag=4,
        p1_in_defender_hitlag=5,
        p2_in_hitlag=6,
        p2_in_defender_hitlag=7,
        p1_shield_strength=8,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark compute_value_targets before/after optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (B).")
    parser.add_argument("--seq-len", type=int, default=256, help="Sequence length (L).")
    parser.add_argument("--features", type=int, default=73, help="Number of feature columns (F). Must be ≥ 9.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to benchmark on, e.g. cpu or cuda.")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32"], help="Floating point dtype for X.")
    parser.add_argument("--min-seconds", type=float, default=10.0, help="Minimum wall-clock time per benchmark run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.features < 9:
        raise ValueError("features must be at least 9 to cover reward feature indices.")

    init_config()

    dtype = torch.float32  # restricted via parser choices
    device = torch.device(args.device)
    torch.manual_seed(0)

    X = torch.rand(args.batch_size, args.seq_len, args.features, device=device, dtype=dtype)
    reward_idx = build_reward_feature_index_stub()

    gamma = float(args.gamma)

    with torch.no_grad():
        baseline_out = compute_value_targets_baseline(X, None, gamma=gamma, reward_idx=reward_idx)
        optimized_out = compute_value_targets(X, None, gamma=gamma, reward_idx=reward_idx)
        max_diff = float((baseline_out - optimized_out).abs().max().item())

    if device.type == "cuda":
        torch.cuda.synchronize()

    timer_stmt = "fn(X, None, gamma=gamma, reward_idx=reward_idx)"

    timer_baseline = benchmark.Timer(
        stmt=timer_stmt,
        globals={
            "fn": compute_value_targets_baseline,
            "X": X,
            "gamma": gamma,
            "reward_idx": reward_idx,
        },
        num_threads=1,
    )
    timer_optimized = benchmark.Timer(
        stmt=timer_stmt,
        globals={
            "fn": compute_value_targets,
            "X": X,
            "gamma": gamma,
            "reward_idx": reward_idx,
        },
        num_threads=1,
    )

    baseline_result = timer_baseline.blocked_autorange(min_run_time=args.min_seconds)
    optimized_result = timer_optimized.blocked_autorange(min_run_time=args.min_seconds)

    baseline_us = baseline_result.mean * 1e6
    optimized_us = optimized_result.mean * 1e6
    baseline_median_us = baseline_result.median * 1e6
    optimized_median_us = optimized_result.median * 1e6
    speedup = baseline_result.mean / optimized_result.mean

    print(f"device={device.type}, dtype={dtype}, B={args.batch_size}, L={args.seq_len}, F={args.features}, gamma={gamma}")
    print(f"baseline   : mean {baseline_us:.2f} us | median {baseline_median_us:.2f} us")
    print(f"optimized  : mean {optimized_us:.2f} us | median {optimized_median_us:.2f} us")
    print(f"speedup    : {speedup:.2f}x faster")
    print(f"max |Δ|    : {max_diff:.3e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
