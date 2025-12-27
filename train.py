"""Main training script for the Melee controller model."""

from __future__ import annotations

import argparse

from config import get_config, init_config
from model.nano_gpt import GPT
from train.loop import train_loop
from train.setup import parse_cli_overrides
from utils import print_model_diagram
from window_dataset import make_dataloader


def main() -> None:
    """Main script entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Disable wandb logging for local debugging runs.",
    )
    args, remaining = parser.parse_known_args()

    overrides = parse_cli_overrides(remaining)
    init_config(overrides=overrides)

    loader, ds, sampler = make_dataloader(get_config())

    config = get_config()

    model = GPT(config)
    print("\n=== Policy Model (GPT) Architecture ===")
    print_model_diagram(model)

    train_loop(model, loader, ds, sampler, debug=args.debug)


if __name__ == "__main__":
    main()
