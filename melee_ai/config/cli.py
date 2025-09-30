"""
Command-line interface utilities for configuration.

This module provides utilities for loading configuration from CLI arguments
and applying overrides to the Settings object.
"""

import argparse
from typing import Optional

from .settings import Settings, _apply_dict_overrides


def create_argument_parser() -> argparse.ArgumentParser:
    """Create argument parser for Melee AI configuration."""
    parser = argparse.ArgumentParser(description="Melee AI Training Configuration")

    # Data configuration
    parser.add_argument("--data-root", type=str, help="Root directory for training data")
    parser.add_argument("--replay-dir", type=str, help="Directory containing replay files")

    # Training configuration
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--mode", choices=["episode_linear", "random_windows"], help="Training mode")

    # Model configuration
    parser.add_argument("--block-size", type=int, help="Transformer block size")
    parser.add_argument("--n-embd", type=int, help="Embedding dimension")
    parser.add_argument("--n-layer", type=int, help="Number of transformer layers")
    parser.add_argument("--n-head", type=int, help="Number of attention heads")

    # Logging configuration
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")

    # Runtime configuration
    parser.add_argument("--device", type=str, help="Device to run on (auto, cpu, cuda, mps)")
    parser.add_argument("--seed", type=int, help="Random seed")

    # Configuration file
    parser.add_argument("--config", type=str, help="Path to configuration file")

    return parser


def load_settings_from_args(args: Optional[argparse.Namespace] = None) -> Settings:
    """
    Load settings from command line arguments.

    Args:
        args: Parsed command line arguments (if None, parses from sys.argv)

    Returns:
        Configured Settings object with CLI overrides applied
    """
    if args is None:
        parser = create_argument_parser()
        args = parser.parse_args()

    # Start with default settings
    settings = Settings()

    # Apply CLI overrides
    overrides = {}
    if args.data_root:
        overrides["data.data_root"] = args.data_root
    if args.replay_dir:
        overrides["data.replay_dir"] = args.replay_dir
    if args.batch_size:
        overrides["training.batch_size"] = args.batch_size
    if args.epochs:
        overrides["training.epochs"] = args.epochs
    if args.learning_rate:
        overrides["training.lr"] = args.learning_rate
    if args.mode:
        overrides["training.mode"] = args.mode
    if args.block_size:
        overrides["model.block_size"] = args.block_size
    if args.n_embd:
        overrides["model.n_embd"] = args.n_embd
    if args.n_layer:
        overrides["model.n_layer"] = args.n_layer
    if args.n_head:
        overrides["model.n_head"] = args.n_head
    if args.output_dir:
        overrides["logging.output_dir"] = args.output_dir
    if args.log_level:
        overrides["logging.log_level"] = args.log_level
    if args.device:
        overrides["runtime.device"] = args.device
    if args.seed:
        overrides["runtime.seed"] = args.seed

    if overrides:
        _apply_dict_overrides(settings, overrides)

    return settings


def print_settings(settings: Settings):
    """Print current settings in a readable format."""
    print("Melee AI Configuration:")
    print(f"  Data root: {settings.data.data_root}")
    print(f"  Batch size: {settings.training.batch_size}")
    print(f"  Learning rate: {settings.training.lr}")
    print(f"  Epochs: {settings.training.epochs}")
    print(f"  Mode: {settings.training.mode}")
    print(f"  Output dir: {settings.logging.output_dir}")
    print(f"  Device: {settings.runtime.device}")
    print(f"  Total steps: {settings.total_steps}")
