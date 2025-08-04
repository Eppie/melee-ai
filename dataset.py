from __future__ import annotations

from typing import Iterator, Sequence, Optional, Tuple, List

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset


def _make_windows(
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int,
        reaction_time: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorised sliding-window view.

    features: (frames, n_feat)
    targets : (frames, n_tgt)

    Returns
    -------
    X : (num_samples, seq_len, n_feat)
    y : (num_samples, n_tgt)
    """
    # shape → (frames − seq_len + 1, 1, seq_len, n_feat)
    win = sliding_window_view(features, (seq_len, features.shape[1]))[:, 0]
    # drop the last `reaction_time` windows that don’t have a target look-ahead
    if reaction_time:
        win = win[:-reaction_time]
        y = targets[seq_len - 1 + reaction_time:]
    else:
        y = targets[seq_len - 1:]
    return win.astype(np.float32, copy=False), y.astype(np.float32, copy=False)


class MeleeIterableDataset(
    IterableDataset[Tuple[torch.Tensor, torch.Tensor]]
):
    """
    Streaming dataset – Arrow ➜ NumPy ➜ Torch, no Pandas.

    Parameters
    ----------
    file_path        : Parquet shard (must be sorted by file_hash, frame_id)
    target_columns   : columns to predict
    sequence_length  : timesteps per sample
    reaction_time    : look-ahead offset for target rows
    feature_columns  : override feature list (optional)
    """

    def __init__(
            self,
            file_path: str,
            target_columns: Sequence[str],
            sequence_length: int,
            reaction_time: int,
            feature_columns: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        self.file_path = file_path
        self.target_columns: List[str] = list(target_columns)
        self.sequence_length = sequence_length
        self.reaction_time = reaction_time

        pq_file = pq.ParquetFile(file_path)
        schema_arrow: pa.Schema = pq_file.schema_arrow

        if feature_columns is None:
            numeric_bool: List[str] = [
                f.name
                for f in schema_arrow
                if (
                        (pa.types.is_floating(f.type)
                         or pa.types.is_integer(f.type)
                         or pa.types.is_boolean(f.type))
                        and f.name not in self.target_columns
                        and f.name not in {"file_hash", "frame_id"}
                )
            ]
            self.feature_columns = numeric_bool
        else:
            self.feature_columns = list(feature_columns)

        # list of unique games (lazy-read)
        self.game_hashes: np.ndarray = (
            pq_file
            .read(columns=["file_hash"])  # Arrow Table
            .combine_chunks()  # merge record batches → 1 chunk
            .column("file_hash")  # Arrow Array
            .unique()  # still Arrow Array
            .to_numpy(zero_copy_only=False)  # allow copy if needed
        )

    # ─────── iterable API ────────────────────────────────────────────────
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            hashes = self.game_hashes
        else:
            # split games roughly evenly across workers
            per = int(np.ceil(len(self.game_hashes) / worker_info.num_workers))
            start = worker_info.id * per
            end = min(start + per, len(self.game_hashes))
            hashes = self.game_hashes[start:end]

        rng = np.random.default_rng()
        rng.shuffle(hashes)  # epoch-level shuffling

        cols_needed: List[str] = (
                list(self.feature_columns)
                + list(self.target_columns)
                + ["frame_id"]
        )

        for game_hash in hashes:
            table = pq.read_table(
                self.file_path,
                filters=[("file_hash", "==", game_hash)],
                columns=cols_needed,
            )

            n_frames = table.num_rows
            if n_frames < self.sequence_length + self.reaction_time:
                continue

            feats = np.column_stack(
                [
                    table[col].to_numpy(zero_copy_only=False)  # ← changed param
                    for col in self.feature_columns
                ]
            ).astype(np.float32, copy=False)

            tgts = np.column_stack(
                [
                    table[col].to_numpy(zero_copy_only=False)
                    for col in self.target_columns
                ]
            ).astype(np.float32, copy=False)

            X, y = _make_windows(
                feats, tgts, self.sequence_length, self.reaction_time
            )

            for xi, yi in zip(X, y, strict=False):
                # returns CPU tensors; DataLoader handles device transfer
                yield (
                    torch.from_numpy(xi.copy()),  # shape (seq_len, n_feat)
                    torch.from_numpy(yi.copy()),  # shape (n_tgt,)
                )
