from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from itertools import islice
from typing import Iterator, Dict
from typing import List
from typing import Sequence, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch import optim
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from tqdm import tqdm


def _focal_bce_with_logits(
        logits: Tensor,
        targets: Tensor,
        *,
        pos_weight: Optional[Tensor] = None,  # shape [C] or None
        alpha: Optional[Tensor] = None,  # weight for positives per label in [0,1]; shape [C] or scalar
        gamma: float = 0.0,  # 0 -> plain BCE, >0 -> focal
) -> Tensor:
    """
    Numerically-stable BCE with logits + optional focal modulation and alpha balancing.
    Returns mean over batch and labels.
    """
    # Base BCE (per-element), with correct positive-class weighting
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )

    if gamma > 0.0:
        # p_t = p for y=1, 1-p for y=0
        p = torch.sigmoid(logits)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        mod = (1.0 - p_t).clamp_min(1e-4).pow(gamma)
        bce = bce * mod

    if alpha is not None:
        # α for positives; (1-α) for negatives. Broadcast to [B, C]
        alpha = torch.as_tensor(alpha, device=logits.device, dtype=logits.dtype)
        alpha_pos = alpha
        alpha_neg = 1.0 - alpha
        alpha_factor = targets * alpha_pos + (1.0 - targets) * alpha_neg
        bce = bce * alpha_factor

    return bce.mean()


class BalancedHybridLoss(nn.Module):
    """
    Hybrid classification+regression loss with:
      • BCE-with-logits (optionally focal) for classification, with per-label pos_weight.
      • Smooth L1 (Huber) for regression, with optional per-target scaling.
      • Group-wise averaging so BCE group and Regression group contribute comparably.

    Args:
        bce_indices: column indices of classification (binary) targets.
        reg_indices: column indices of regression targets.
        pos_weight:  per-label positive weight for BCE; usually neg/pos counts, shape [#bce] (optional).
        alpha:       per-label alpha in [0,1] for BCE (optional; leave None if you use pos_weight).
        focal_gamma: focal loss gamma >= 0 (0 disables focal).
        reg_beta:    Smooth L1 transition point.
        reg_scale:   per-regression weight (e.g., inverse std) to equalize magnitudes, shape [#reg] (optional).
        bce_group_weight, reg_group_weight: scalar weights for the two groups.
    """

    def __init__(
            self,
            bce_indices: Sequence[int],
            reg_indices: Sequence[int],
            *,
            pos_weight: Optional[Tensor] = None,
            alpha: Optional[Tensor] = None,
            focal_gamma: float = 0.0,
            reg_beta: float = 0.02,
            reg_scale: Optional[Tensor] = None,
            bce_group_weight: float = 1.0,
            reg_group_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight",
                             None if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32))
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))
        self.register_buffer("reg_scale",
                             None if reg_scale is None else torch.as_tensor(reg_scale, dtype=torch.float32))

        self.bce_idx = list(bce_indices)
        self.reg_idx = list(reg_indices)
        self.focal_gamma = float(focal_gamma)
        self.reg_beta = float(reg_beta)
        self.bce_group_weight = float(bce_group_weight)
        self.reg_group_weight = float(reg_group_weight)

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        loss_bce = preds.new_tensor(0.0)
        loss_reg = preds.new_tensor(0.0)

        if self.bce_idx:
            logits = preds[:, self.bce_idx]
            tgt = targets[:, self.bce_idx]
            loss_bce = _focal_bce_with_logits(
                logits, tgt, pos_weight=self.pos_weight, alpha=self.alpha, gamma=self.focal_gamma
            )

        if self.reg_idx:
            p = preds[:, self.reg_idx]
            t = targets[:, self.reg_idx]
            reg = F.smooth_l1_loss(p, t, beta=self.reg_beta, reduction="none")  # per-element
            if self.reg_scale is not None:
                # Broadcast scale [C] -> [B, C]
                reg = reg * self.reg_scale
            loss_reg = reg.mean()

        return self.bce_group_weight * loss_bce + self.reg_group_weight * loss_reg

    @torch.no_grad()
    def components(self, preds: Tensor, targets: Tensor) -> dict[str, float]:
        """Optional helper for logging the two parts separately."""
        out: dict[str, float] = {}
        if self.bce_idx:
            logits = preds[:, self.bce_idx]
            tgt = targets[:, self.bce_idx]
            out["bce"] = _focal_bce_with_logits(
                logits, tgt, pos_weight=self.pos_weight, alpha=self.alpha, gamma=self.focal_gamma
            ).item()
        if self.reg_idx:
            p = preds[:, self.reg_idx]
            t = targets[:, self.reg_idx]
            reg = F.smooth_l1_loss(p, t, beta=self.reg_beta, reduction="none")
            if self.reg_scale is not None:
                reg = reg * self.reg_scale
            out["reg"] = reg.mean().item()
        out["total"] = self.bce_group_weight * out.get("bce", 0.0) + self.reg_group_weight * out.get("reg", 0.0)
        return out


class NumPySequenceDataset(IterableDataset):
    """Stream sliding windows from sharded NumPy matrices written by preprocess.py.

    Each shard is a `.npy` float32 matrix of shape [N, K], with a companion
    `.columns.npy` containing the K column names (dtype=str). We construct
    inputs X by selecting all columns not in `target_columns`, and targets y
    by selecting the columns listed in `target_columns`.

    For a given sliding window of length `sequence_length`, the target vector
    is taken from the frame `reaction_time` steps AFTER the end of the window.
    Windows do not cross shard boundaries.
    """

    def __init__(
            self,
            files: list[str],
            target_columns: list[str],
            sequence_length: int,
            reaction_time: int,
    ) -> None:
        super().__init__()
        if not files:
            raise ValueError("No .npy shards provided.")
        self.files = files
        self.target_columns = list(target_columns)
        self.sequence_length = int(sequence_length)
        self.reaction_time = int(reaction_time)

        # Infer feature columns from the first shard's columns list
        first_cols = np.load(self._cols_path(files[0]))
        all_cols = [str(c) for c in list(first_cols.tolist())]
        self.feature_columns = [c for c in all_cols if c not in self.target_columns]

    @staticmethod
    def _cols_path(data_path: str) -> str:
        if data_path.endswith('.columns.npy'):
            return data_path
        if data_path.endswith('.npy'):
            return data_path[:-4] + '.columns.npy'
        return data_path + '.columns.npy'

    def _file_partition(self) -> list[str]:
        wi = get_worker_info()
        if wi is None:
            return self.files
        return self.files[wi.id:: wi.num_workers]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        seq_len = self.sequence_length
        rt = self.reaction_time

        emitted = 0  # samples yielded by this worker in this epoch

        part_files = self._file_partition()

        # Precompute per-file indices and sample counts (windows) for this worker
        per_file = []
        total_samples = 0
        for path in part_files:
            cols = np.load(self._cols_path(path))
            col_names = [str(c) for c in list(cols.tolist())]
            feat_idx = np.array([col_names.index(c) for c in self.feature_columns], dtype=np.int64)
            tgt_idx = np.array([col_names.index(c) for c in self.target_columns], dtype=np.int64)

            mat = np.load(path, mmap_mode='r')  # float32 [N, K]
            N = mat.shape[0]
            if N == 0 or len(feat_idx) == 0:
                file_samples = 0
                max_start = -1
            else:
                max_start = N - (seq_len + rt)
                file_samples = max(0, max_start + 1)
            per_file.append((path, mat, feat_idx, tgt_idx, max_start, file_samples))
            total_samples += file_samples

        if total_samples == 0:
            return  # nothing to yield

        # First pass: skip until rotation point
        file_idx = 0
        while file_idx < len(per_file):
            _, mat, feat_idx, tgt_idx, max_start, file_samples = per_file[file_idx]
            if file_samples == 0:
                file_idx += 1
                continue
            for s in range(8192*100):
                x_np = mat[s: s + seq_len, :][:, feat_idx]
                y_np = mat[s + seq_len - 1 + rt, :][tgt_idx]
                X = torch.from_numpy(np.ascontiguousarray(x_np))
                y = torch.from_numpy(np.ascontiguousarray(y_np))
                emitted += 1
                yield X, y
            file_idx += 1


@dataclass(slots=True)
class MLPConfig:
    """Hyperparameters for the feed-forward network."""
    input_dim: int
    output_dim: int
    hidden_dims: Sequence[int] = (8192, 2048, 8192)
    dropout: float = 0.1
    act: type[nn.Module] = nn.GELU
    weight_init: Optional[callable] = None


class FeedForwardNet(nn.Module):
    """Simple fully-connected network that flattens sequence input."""

    def __init__(self, cfg: MLPConfig) -> None:
        super().__init__()
        self.flatten = nn.Flatten()

        layers: List[nn.Module] = []
        dims = (cfg.input_dim, *cfg.hidden_dims, cfg.output_dim)

        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != cfg.output_dim:
                layers.append(cfg.act())
                if cfg.dropout > 0:
                    layers.append(nn.Dropout(cfg.dropout))

        self.net = nn.Sequential(*layers)

        if cfg.weight_init is not None:
            self.apply(cfg.weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, sequence_length, num_features)
        # Flatten to: (batch_size, sequence_length * num_features)
        x = self.flatten(x)
        return self.net(x)


# ─────────────────── Custom Loss for Hybrid Task ─────────────────────

class CombinedLoss(nn.Module):
    """A custom loss function for a hybrid classification and regression task."""

    def __init__(self, bce_indices: List[int], mse_indices: List[int], bce_weight: float = 1.0,
                 mse_weight: float = 1.0):
        super().__init__()
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.mse_loss_fn = nn.MSELoss()
        self.bce_indices = bce_indices
        self.mse_indices = mse_indices
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_bce = torch.tensor(0.0, device=preds.device)
        if self.bce_indices:
            preds_bce = preds[:, self.bce_indices]
            targets_bce = targets[:, self.bce_indices]
            loss_bce = self.bce_loss_fn(preds_bce, targets_bce)

        loss_mse = torch.tensor(0.0, device=preds.device)
        if self.mse_indices:
            preds_mse = preds[:, self.mse_indices]
            targets_mse = targets[:, self.mse_indices]
            loss_mse = self.mse_loss_fn(preds_mse, targets_mse)

        return (self.bce_weight * loss_bce) + (self.mse_weight * loss_mse)


# ─────────────────────────── Training Helper ───────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,          # expects .components(preds, y) if available
    optimiser: optim.Optimizer,
    device: torch.device,
    steps_per_epoch: Optional[int] = None,
    log_every: int = 1,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_bce = 0.0
    total_reg = 0.0
    n_batches = 0

    iterable = loader if steps_per_epoch is None else islice(loader, steps_per_epoch)
    with tqdm(desc="Training", leave=False, dynamic_ncols=True) as pbar:
        for step, (X, y) in enumerate(iterable, start=1):
            X, y = X.to(device), y.to(device)
            optimiser.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimiser.step()

            n_batches += 1
            step_total = float(loss.item())
            total_loss += step_total

            # If the criterion exposes components(), use it (it should be @torch.no_grad()).
            bce = reg = float("nan")
            if hasattr(criterion, "components"):
                comps = criterion.components(preds, y)  # {"bce": ..., "reg": ..., "total": ...}
                bce = comps.get("bce", float("nan"))
                reg = comps.get("reg", float("nan"))
                step_total = comps.get("total", step_total)

            if not np.isnan(bce): total_bce += bce
            if not np.isnan(reg): total_reg += reg

            if step % log_every == 0:
                pbar.set_postfix(
                    step=f"{step_total:.4f}",
                    bce=f"{bce:.4f}" if not np.isnan(bce) else "nan",
                    reg=f"{reg:.4f}" if not np.isnan(reg) else "nan",
                    avg=f"{(total_loss/n_batches):.4f}",
                )
            pbar.update(1)

    return {
        "total": total_loss / max(n_batches, 1),
        "bce": (total_bce / n_batches) if n_batches else float("nan"),
        "reg": (total_reg / n_batches) if n_batches else float("nan"),
    }


if __name__ == "__main__":
    tfrecord_glob = '/Users/eppie/PycharmProjects/melee-ai/shards/frames_chunk_000.npy'
    SEQUENCE_LENGTH = 60
    REACTION_TIME = 0
    target_columns = [
        'p1_btn_a', 'p1_btn_b', 'p1_btn_x', 'p1_btn_l', 'p1_btn_z',
        'p1_btn_l_analog', 'p1_btn_r_analog', 'p1_pre_joystick_x',
        'p1_pre_joystick_y', 'p1_pre_cstick_x', 'p1_pre_cstick_y', 'p1_btn_y', 'p1_btn_r'
    ]
    classification_targets = ['p1_btn_a', 'p1_btn_b', 'p1_btn_x', 'p1_btn_l', 'p1_btn_z', 'p1_btn_y', 'p1_btn_r']
    regression_targets = [col for col in target_columns if col not in classification_targets]
    batch_size = 1024 * 8
    num_dataloader_workers = os.cpu_count() or 2

    files = sorted(f for f in glob.glob(tfrecord_glob) if not f.endswith('.columns.npy'))
    print(f"Initializing NumPy streaming dataset from {len(files)} shard(s) matching: {tfrecord_glob}")
    if not files:
        print(f"\n--- ERROR: No .npy shards found for pattern '{tfrecord_glob}'. ---\n")
        exit()

    dataset = NumPySequenceDataset(
        files=files,
        target_columns=target_columns,
        sequence_length=SEQUENCE_LENGTH,
        reaction_time=REACTION_TIME,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_dataloader_workers,
        pin_memory=False,
        persistent_workers=True,
        prefetch_factor=2 if (num_dataloader_workers or 0) > 0 else None,
    )
    print(f"DataLoader initialized with {num_dataloader_workers} workers.")

    # --- 3. Configure and create the model ---
    num_features = len(dataset.feature_columns)
    cfg = MLPConfig(
        input_dim=SEQUENCE_LENGTH * num_features,
        output_dim=len(target_columns),
    )
    print(f"Model Config: Input Dim={cfg.input_dim}, Output Dim={cfg.output_dim}")

    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = FeedForwardNet(cfg).to(device)
    print(f"Training on device: {device}")

    # --- 4. Set up the combined loss function ---
    bce_indices = [target_columns.index(col) for col in classification_targets]
    mse_indices = [target_columns.index(col) for col in regression_targets]
    reg_indices = [target_columns.index(col) for col in regression_targets]
    import itertools

    with torch.no_grad():
        pos = torch.zeros(len(bce_indices))
        neg = torch.zeros(len(bce_indices))
        for _, y in itertools.islice(loader, 64):  # sample some batches
            t = y[:, bce_indices].float()
            pos += t.sum(0)
            neg += (1.0 - t).sum(0)
    pos_weight = (neg / pos.clamp_min(1.0))
    print(f"Positive weight: {pos_weight}")

    criterion = BalancedHybridLoss(
        bce_indices=bce_indices,
        reg_indices=reg_indices,
        pos_weight=pos_weight.to(device),  # or None if you prefer alpha/focal only
        focal_gamma=1.0,  # try 1–2 if classes are very imbalanced
        reg_beta=0.02,  # tune to your analog scale
        bce_group_weight=1.0,
        reg_group_weight=1.0,
    )
    # criterion = CombinedLoss(bce_indices=bce_indices, mse_indices=mse_indices)
    optimiser = optim.AdamW(model.parameters(), lr=3e-4)

    # --- 5. Run the training loop ---
    print("\nStarting training...")
    for epoch in range(10):
        stats = train_one_epoch(model, loader, criterion, optimiser, device)
        print(f"Epoch {epoch:02d} | total={stats['total']:.4f} | bce={stats['bce']:.4f} | reg={stats['reg']:.4f}")
    print("Training finished.")
    # ── 6. Save trained model ────────────────────────────────────────────
    model_ckpt_path = "trained_mlp_model.pt"
    torch.save(model.state_dict(), model_ckpt_path)
    print(f"Model saved to {model_ckpt_path}")
