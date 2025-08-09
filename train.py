from __future__ import annotations

import glob
import itertools
import os
from typing import Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from RecurrentNet import RecurrentConfig, RecurrentNet
from config import (
    TARGET_COLUMNS,
    SEQUENCE_LENGTH,
    CLASSIFICATION_TARGETS,
    REGRESSION_TARGETS,
    FEATURE_COLUMNS,
    BCE_INDICES,
    REG_INDICES,
    BATCH_SIZE,
)
from dataset import NumPySequenceDataset
from loss import SimpleHybridLoss

RECURRENT_CONFIG = RecurrentConfig(
    input_dim=len(FEATURE_COLUMNS),
    output_dim=len(TARGET_COLUMNS),
    hidden_size=1024,
    num_layers=4,
    dropout=0.0,
    bidirectional=False,  # start unidirectional for causal prediction
    kind="lstm",
    embed_dim=32,
)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,  # expects .components(preds, y) if available
    optimiser: optim.Optimizer,
    device: torch.device,
    steps_per_epoch: Optional[int] = None,
    log_every: int = 1,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_reg = 0.0
    n_batches = 0

    iterable = (
        loader if steps_per_epoch is None else itertools.islice(loader, steps_per_epoch)
    )
    with tqdm(desc="Training", leave=False, dynamic_ncols=True) as pbar:
        for step, (X, y) in enumerate(iterable, start=1):
            X, y = X.to(device), y.to(device)
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
            optimiser.zero_grad()
            preds = model(X, y)
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

            n_batches += 1
            step_total = float(loss.item())
            total_loss += step_total

            # If the criterion exposes components(), use it (it should be @torch.no_grad()).
            bce = reg = float("nan")
            if hasattr(criterion, "components"):
                comps = criterion.components(
                    preds, y
                )  # {"bce": ..., "reg": ..., "total": ...}
                bce = comps.get("bce", float("nan"))
                reg = comps.get("reg", float("nan"))
                step_total = comps.get("total", step_total)

            if not np.isnan(bce):
                total_bce += bce
            if not np.isnan(reg):
                total_reg += reg

            if step % log_every == 0:
                pbar.set_postfix(
                    step=f"{step_total:.4f}",
                    bce=f"{bce:.4f}" if not np.isnan(bce) else "nan",
                    reg=f"{reg:.4f}" if not np.isnan(reg) else "nan",
                    avg=f"{(total_loss / n_batches):.4f}",
                )
            pbar.update(1)

    return {
        "total": total_loss / max(n_batches, 1),
        "bce": (total_bce / n_batches) if n_batches else float("nan"),
        "reg": (total_reg / n_batches) if n_batches else float("nan"),
    }


if __name__ == "__main__":
    print(f"Num targets: {len(TARGET_COLUMNS)}")
    print(f"Num regression targets: {len(REGRESSION_TARGETS)}")
    print(f"Num classification targets: {len(CLASSIFICATION_TARGETS)}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Num features: {len(FEATURE_COLUMNS)}")
    tfrecord_glob = "/Users/eppie/melee-ai/shards/frames_chunk_000.npy"
    num_dataloader_workers = os.cpu_count() or 2

    files = sorted(
        f for f in glob.glob(tfrecord_glob) if not f.endswith(".columns.npy")
    )
    print(
        f"Initializing NumPy streaming dataset from {len(files)} shard(s) matching: {tfrecord_glob}"
    )
    if not files:
        print(f"\n--- ERROR: No .npy shards found for pattern '{tfrecord_glob}'. ---\n")
        exit()

    dataset = NumPySequenceDataset(
        files=files,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=num_dataloader_workers,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2 if (num_dataloader_workers or 0) > 0 else None,
    )
    print(f"DataLoader initialized with {num_dataloader_workers} workers.")

    device = torch.device("mps" if torch.mps.is_available() else "cpu")

    print(
        f"Model Config: Input Dim={RECURRENT_CONFIG.input_dim}, Output Dim={RECURRENT_CONFIG.output_dim}"
    )
    model = RecurrentNet(RECURRENT_CONFIG, feature_dim=len(FEATURE_COLUMNS)).to(device)
    print(f"Training on device: {device}")
    criterion = SimpleHybridLoss(
        bce_indices=BCE_INDICES,
        reg_indices=REG_INDICES,
        bce_weight=1.0,
        reg_weight=2.0,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)

    print("\nStarting training...")
    for epoch in range(30):
        stats = train_one_epoch(model, loader, criterion, optimizer, device)
        print(
            f"Epoch {epoch:02d} | total={stats['total']:.4f} | bce={stats['bce']:.4f} | reg={stats['reg']:.4f}"
        )
        model_ckpt_path = "trained_mlp_model.pt"
        torch.save(model.state_dict(), model_ckpt_path)
        print(f"Model saved to {model_ckpt_path}")
    print("Training finished.")
