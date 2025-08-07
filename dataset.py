from __future__ import annotations

from typing import Tuple, List, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info


class NumPySequenceDataset(IterableDataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Stream sliding windows from sharded NumPy matrices written by preprocess.py.

    Each shard is a `.npy` float32 matrix of shape [N, K], with a companion
    `.columns.npy` containing the K column names (dtype=str). Inputs X use all
    columns not in `target_columns`, and targets y use the columns in
    `target_columns`.

    For a sliding window of length `sequence_length`, the target vector is taken
    from the frame **immediately after** the window (i.e., predict t+1).

    Windows do not cross shard boundaries.

    Additional behavior:
      * Windows are yielded in a shuffled order each epoch/iteration.
      * Any window for which **any** `frame_id` in the window is < 0 is skipped.
    """

    def __init__(
            self,
            files: List[str],
            target_columns: List[str],
            feature_columns: List[str],
            sequence_length: int,
    ) -> None:
        super().__init__()
        if not files:
            raise ValueError("No .npy shards provided.")

        self.files: List[str] = files
        self.target_columns: List[str] = list(target_columns)
        self.sequence_length: int = int(sequence_length)

        # Infer feature columns from the first shard's columns list
        first_cols = np.load(self._cols_path(files[0]))
        all_cols = [str(c) for c in list(first_cols.tolist()) if str(c) != 'frame_id']
        self.feature_columns: List[str] = feature_columns

    @staticmethod
    def _cols_path(data_path: str) -> str:
        if data_path.endswith(".columns.npy"):
            return data_path
        if data_path.endswith(".npy"):
            return data_path[:-4] + ".columns.npy"
        return data_path + ".columns.npy"

    def _file_partition(self) -> List[str]:
        wi = get_worker_info()
        if wi is None:
            return self.files
        return self.files[wi.id:: wi.num_workers]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        seq_len = self.sequence_length
        part_files = self._file_partition()

        # Per-worker RNG so DataLoader(seed)/epoch reshuffles deterministically when desired
        seed = torch.initial_seed() % (2 ** 32 - 1)
        rng = np.random.default_rng(seed)

        per_file = []
        total_samples = 0
        for path in part_files:
            # --- Load columns; skip if missing or malformed ---
            try:
                cols = np.load(self._cols_path(path))
            except Exception as e:
                print(f"[WARN] Skipping shard {path}: couldn't load columns ({e})")
                continue

            col_names = [str(c) for c in list(cols.tolist())]

            # --- Ensure this shard has all required feature/target columns ---
            missing_feats = [c for c in self.feature_columns if c not in col_names]
            missing_tgts = [c for c in self.target_columns if c not in col_names]
            if missing_feats or missing_tgts:
                continue

            feat_idx = np.array([col_names.index(c) for c in self.feature_columns], dtype=np.int64)
            tgt_idx = np.array([col_names.index(c) for c in self.target_columns], dtype=np.int64)

            # `frame_id` is optional; if absent, disable frame-based filtering
            frame_idx = col_names.index("frame_id") if "frame_id" in col_names else -1

            # --- Load the data matrix and sanity-check shape ---
            try:
                mat = np.load(path, mmap_mode="r")  # float32 [N, K]
            except Exception as e:
                print(f"[WARN] Skipping shard {path}: couldn't load data ({e})")
                continue

            if mat.ndim != 2 or mat.shape[1] != len(col_names):
                print(
                    f"[WARN] Skipping shard {path}: bad shape {mat.shape}, expected [N,{len(col_names)}] given columns file"
                )
                continue

            N = int(mat.shape[0])

            # Valid starts s satisfy: s + seq_len < N  =>  s <= N - seq_len - 1
            # Count = N - seq_len (if positive).
            num_starts = max(0, N - seq_len)
            if num_starts <= 0:
                # No valid windows in this shard
                continue

            per_file.append((mat, feat_idx, tgt_idx, frame_idx, num_starts))
            total_samples += num_starts

        if total_samples == 0:
            return

        file_order = np.arange(len(per_file))
        rng.shuffle(file_order)

        for fi in file_order:
            mat, feat_idx, tgt_idx, frame_idx, num_starts = per_file[fi]
            if num_starts <= 0 or feat_idx.size == 0:
                continue

            starts = np.arange(num_starts, dtype=np.int64)  # [0 .. N-seq_len-1]
            rng.shuffle(starts)

            # print(starts)
            # print(len(starts))
            for s in starts:
                if frame_idx >= 0:
                    # Filter: any frame_id inside the window < 0 => skip
                    frame_col = mat[s: s + seq_len, frame_idx]
                    if np.any(frame_col < 0):
                        continue

                x_np = mat[s: s + seq_len, :][:, feat_idx]  # [seq_len, F]
                y_np = mat[s + seq_len, :][tgt_idx]  # predict t+1
                if not (np.isfinite(x_np).all() and np.isfinite(y_np).all()):
                    print('Infinite or NaN!')
                    continue
                X = torch.from_numpy(np.ascontiguousarray(x_np))  # float32
                y = torch.from_numpy(np.ascontiguousarray(y_np))  # float32

                yield X, y
