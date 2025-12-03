#!/usr/bin/env python3
"""
Main entry point for PPO reinforcement learning training.

This script launches the full hierarchical PPO training system with:
- 1 Coordinator (GPU) process for inference and training
- 12 ArenaShard (S8) processes managing environment workers
- 96 total Dolphin environments (8 per shard)

Usage:
    python train_ppo_rl.py --init-checkpoint checkpoints/latest.pt
    python train_ppo_rl.py --init-checkpoint checkpoints/latest.pt --num-shards 2 --envs-per-shard 4
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

from config import Config
from ppo.config import PPOConfig
from ppo.coordinator import Coordinator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Melee bot with PPO reinforcement learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model checkpoint
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        required=True,
        help="Initial policy checkpoint to load (from imitation learning)",
    )

    # Scale parameters
    parser.add_argument(
        "--num-shards",
        type=int,
        default=12,
        help="Number of ArenaShard (S8) processes",
    )
    parser.add_argument(
        "--envs-per-shard",
        type=int,
        default=8,
        help="Dolphin instances per shard (total envs = num_shards × envs_per_shard)",
    )

    # Dolphin paths
    parser.add_argument(
        "--dolphin-path",
        type=str,
        default="/Applications/Slippi Dolphin.app",
        help="Path to Slippi Dolphin executable",
    )
    parser.add_argument(
        "--iso-path",
        type=str,
        default="~/Documents/SSBM.iso",
        help="Path to SSBM ISO file",
    )

    # Game settings
    parser.add_argument(
        "--character",
        type=str,
        default="FOX",
        choices=["FOX", "FALCO", "MARTH", "SHEIK", "FALCON"],
        help="Character for both players",
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=["FD", "BF", "YS", "FoD", "PS", "DL"],
        help="Stage pool for randomization",
    )

    # Training hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--rollout-length",
        type=int,
        default=1024,
        help="Frames per rollout before PPO update",
    )
    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=4,
        help="Number of PPO epochs per update",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for PPO training",
    )

    # Opponent pool
    parser.add_argument(
        "--opponent-pool-size",
        type=int,
        default=20,
        help="Number of historical checkpoints to keep in pool",
    )
    parser.add_argument(
        "--opponent-sample-prob",
        type=float,
        default=0.8,
        help="Probability of using historical opponent (vs self-play)",
    )

    # Checkpointing
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("ppo_checkpoints"),
        help="Directory for saving PPO checkpoints",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10,
        help="Training steps between saving checkpoints",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 80)
    print("PPO Reinforcement Learning Training")
    print("=" * 80)
    print(f"Initial checkpoint: {args.init_checkpoint}")
    print(
        f"Scale: {args.num_shards} shards × {args.envs_per_shard} envs = {args.num_shards * args.envs_per_shard} total"
    )
    print(f"Character: {args.character}")
    print(f"Stages: {', '.join(args.stages)}")
    print(f"Learning rate: {args.lr}")
    print(
        f"Opponent pool: {args.opponent_pool_size} checkpoints, {args.opponent_sample_prob*100:.0f}% historical"
    )
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("=" * 80)

    # Verify checkpoint exists
    if not args.init_checkpoint.exists():
        print(f"ERROR: Checkpoint not found: {args.init_checkpoint}")
        print("Please provide a valid checkpoint from imitation learning")
        sys.exit(1)

    # Force spawn method (CUDA-safe)
    mp.set_start_method("spawn", force=True)

    # Load base config
    config = Config()

    # Create PPO config
    ppo_config = PPOConfig(
        num_shards=args.num_shards,
        envs_per_shard=args.envs_per_shard,
        dolphin_path=args.dolphin_path,
        iso_path=args.iso_path,
        character=args.character,
        stages=args.stages,
        lr=args.lr,
        rollout_length=args.rollout_length,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        opponent_pool_size=args.opponent_pool_size,
        opponent_sample_prob=args.opponent_sample_prob,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        init_checkpoint=args.init_checkpoint,
    )

    # Create checkpoint directory
    ppo_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[MAIN] Creating Coordinator...")
    try:
        coordinator = Coordinator(config, ppo_config)
    except Exception as e:
        print(f"[MAIN] ERROR: Failed to create coordinator: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print(f"[MAIN] Starting training loop...")
    print(f"[MAIN] Press Ctrl+C to stop gracefully\n")

    try:
        coordinator.run()
    except KeyboardInterrupt:
        print("\n[MAIN] Training interrupted by user")
    except Exception as e:
        print(f"\n[MAIN] ERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("[MAIN] Training complete!")


if __name__ == "__main__":
    main()
