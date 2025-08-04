import os
import time, itertools, torch

from torch.utils.data import DataLoader

from MLPModel import FeedForwardNet, MLPConfig, CombinedLoss
from dataset import MeleeIterableDataset


# ── 1A.  Measure raw dataloader throughput ────────────────────────────────
def bench_loader(loader, n_batches=200):
    it = iter(loader)
    t0 = time.perf_counter()
    for _ in range(n_batches):
        try:
            next(it)
        except StopIteration:
            break
    dt = time.perf_counter() - t0
    print(f"Avg. loader time / batch: {dt / max(1, n_batches):.4f}s")


if __name__ == "__main__":
    file_path = '/Users/eppie/PycharmProjects/melee-ai/shards/frames_chunk_000.parquet'
    SEQUENCE_LENGTH = 10
    REACTION_TIME = 18
    target_columns = [
        'p1_btn_a', 'p1_btn_b', 'p1_btn_x', 'p1_btn_l', 'p1_btn_z',
        'p1_btn_l_analog', 'p1_btn_r_analog', 'p1_pre_joystick_x',
        'p1_pre_joystick_y', 'p1_pre_cstick_x', 'p1_pre_cstick_y'
    ]
    classification_targets = ['p1_btn_a', 'p1_btn_b', 'p1_btn_x', 'p1_btn_l', 'p1_btn_z']
    regression_targets = [col for col in target_columns if col not in classification_targets]
    batch_size = 1024
    # Use multiple workers to parallelize data loading. Set to 0 to disable.
    num_dataloader_workers = os.cpu_count() or 2

    dataset = MeleeIterableDataset(
        file_path=file_path,
        target_columns=target_columns,
        sequence_length=SEQUENCE_LENGTH,
        reaction_time=REACTION_TIME,
    )
    loader = DataLoader(
        dataset,
        batch_size=1024,
        num_workers=os.cpu_count(),
        pin_memory=False,  # ← disable on M-series
        persistent_workers=True,  # keeps workers alive between epochs
        prefetch_factor=4,  # each worker pre-loads 4 batches
    )
    bench_loader(loader)  # <-- run before training starts

    # ── 1B.  Measure pure model step time with dummy data ────────────────────
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    num_features = len(dataset.feature_columns)
    cfg = MLPConfig(
        input_dim=SEQUENCE_LENGTH * num_features,
        output_dim=len(target_columns),
    )
    model = FeedForwardNet(cfg).to(device)
    dummy_x = torch.randn(batch_size, SEQUENCE_LENGTH, len(dataset.feature_columns), device=device)
    dummy_y = torch.randn(batch_size, len(target_columns), device=device)
    bce_indices = [target_columns.index(col) for col in classification_targets]
    mse_indices = [target_columns.index(col) for col in regression_targets]
    criterion = CombinedLoss(bce_indices=bce_indices, mse_indices=mse_indices)
    bench_steps = 200

    t0 = time.perf_counter()
    for _ in range(bench_steps):
        loss = criterion(model(dummy_x), dummy_y)
        loss.backward()
    device_dt = time.perf_counter() - t0
    print(f"Avg. forward+backward / batch: {device_dt / bench_steps:.4f}s on {device}")
