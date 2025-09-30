"""
Training CLI entry point.

This module provides the command-line interface for training models.
"""

import argparse
from pathlib import Path

from melee_ai.config import load_settings_from_args, Settings
from melee_ai.training.trainer import Trainer
from melee_ai.models import ModelRegistry
from melee_ai.data import ParquetDatasetProvider


def train_command(args=None):
    """Main training command."""
    # Load configuration
    settings = load_settings_from_args(args)

    print("Starting training with configuration:")
    print(f"  Data root: {settings.data.data_root}")
    print(f"  Batch size: {settings.training.batch_size}")
    print(f"  Learning rate: {settings.training.lr}")
    print(f"  Epochs: {settings.training.epochs}")

    # Create model
    model = ModelRegistry.get_model("gpt_v7", settings)

    # Create dataset provider
    dataset_provider = ParquetDatasetProvider(settings)

    # Create trainer
    trainer = Trainer(settings, model, dataset_provider)

    # Run training
    results = trainer.train()

    print("Training completed!")
    for epoch, result in results.items():
        print(f"Epoch {epoch}: loss={result.loss:.4f}, duration={result.duration:.2f}s")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Melee AI model")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data-root", type=str, help="Root directory for training data")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")

    args = parser.parse_args()
    train_command(args)


if __name__ == "__main__":
    main()
