#!/usr/bin/env python3
# train_mini_melee.py  –  now with per-epoch progress bars

from __future__ import annotations
import json, random
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Paths & hyper-parameters
# ---------------------------------------------------------------------
JSONL       = Path("frames.jsonl")          # your NDJSON
OBS_FEATS   = [
    "pos_x", "pos_y", "vel_x", "vel_y",
    "percent", "stocks", "action",
    "airborne", "jumps", "shield",
]
TARGETS     = ["joy_x", "joy_y"]            # predict left-stick only
BATCH       = 1_024
EPOCHS      = 3
LR          = 1e-3
TRAIN_SPLIT = 0.9
SEED        = 0
# ---------------------------------------------------------------------


class SlippiStream(IterableDataset):
    """Stream frames.jsonl and yield (obs, tgt) tensors."""
    def __init__(self, path: Path, train_split: float = 0.9) -> None:
        super().__init__()
        self.path = path
        self.train_split = train_split
        self.rng = random.Random(SEED)

    def __iter__(self):
        with self.path.open() as fh:
            for line in fh:
                row: Dict = json.loads(line)
                if self.rng.random() > self.train_split:
                    continue

                obs = torch.tensor([row[k] for k in OBS_FEATS], dtype=torch.float32)
                tgt = torch.tensor([row[k] for k in TARGETS],  dtype=torch.float32)
                yield obs, tgt


def build_dataloader(device: torch.device) -> DataLoader:
    workers = 0 if device.type == "mps" else 4
    pin     = False if device.type == "mps" else True
    return DataLoader(SlippiStream(JSONL, TRAIN_SPLIT),
                      batch_size=BATCH,
                      num_workers=workers,
                      pin_memory=pin)


def main() -> None:
    # -------------------- device --------------------
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("▶ using device:", device, "\n")

    # -------------------- data ----------------------
    loader = build_dataloader(device)

    # -------------------- model ---------------------
    model = nn.Sequential(
        nn.Linear(len(OBS_FEATS), 512),
        nn.ReLU(),
        nn.Linear(512, len(TARGETS)),
    ).to(device)

    opt      = optim.Adam(model.parameters(), lr=LR)
    loss_fn  = nn.MSELoss()

    # ---------------- training loop ----------------
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, n_seen = 0.0, 0

        progress = tqdm(loader, desc=f"Epoch {epoch}", leave=False)  # NEW  [oai_citation:1‡Stack Overflow](https://stackoverflow.com/questions/63426545/best-way-of-tqdm-for-data-loader?utm_source=chatgpt.com) [oai_citation:2‡GitHub](https://github.com/tqdm/tqdm?utm_source=chatgpt.com)
        for obs, tgt in progress:
            obs, tgt = obs.to(device), tgt.to(device)

            opt.zero_grad(set_to_none=True)
            pred  = model(obs)
            loss  = loss_fn(pred, tgt)
            loss.backward()
            opt.step()

            # live metrics
            running += loss.item() * obs.size(0)
            n_seen  += obs.size(0)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_mse = running / n_seen
        print(f"epoch {epoch}: MSE = {epoch_mse:.6f}")

    # ---------------- checkpoint -------------------
    torch.save(model.state_dict(), "mini_melee_linear.pt")
    print("\n✓ finished – model saved to mini_melee_linear.pt")


if __name__ == "__main__":
    torch.manual_seed(SEED); random.seed(SEED)
    main()
