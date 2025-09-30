"""
Evaluation CLI entry point.

This module provides the command-line interface for evaluating
trained models.
"""

import argparse

from melee_ai.config import load_settings_from_args, Settings


def evaluate_command(args=None):
    """Main evaluation command."""
    # Load configuration
    settings = load_settings_from_args(args)

    print("Evaluation with configuration:")
    print(f"  Model checkpoint: {settings.logging.checkpoint_dir}")
    print("  (Evaluation not yet implemented)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Melee AI model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")

    args = parser.parse_args()
    evaluate_command(args)


if __name__ == "__main__":
    main()
