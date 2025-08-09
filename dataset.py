from __future__ import annotations

from typing import Tuple, List, Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from config import FEATURE_COLUMNS, TARGET_COLUMNS, SEQUENCE_LENGTH


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
    ) -> None:
        super().__init__()
        if not files:
            raise ValueError("No .npy shards provided.")

        self.files: List[str] = files

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
        return self.files[wi.id :: wi.num_workers]

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        part_files = self._file_partition()

        # Per-worker RNG so DataLoader(seed)/epoch reshuffles deterministically when desired
        # seed = torch.initial_seed() % (2 ** 32 - 1)
        # rng = np.random.default_rng(seed)

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
            missing_feats = [c for c in FEATURE_COLUMNS if c not in col_names]
            missing_tgts = [c for c in TARGET_COLUMNS if c not in col_names]
            if missing_feats or missing_tgts:
                continue

            feat_idx = np.array(
                [col_names.index(c) for c in FEATURE_COLUMNS], dtype=np.int64
            )
            tgt_idx = np.array(
                [col_names.index(c) for c in TARGET_COLUMNS], dtype=np.int64
            )

            # `frame_id` is optional; if absent, disable frame-based filtering
            frame_idx = col_names.index("frame_id") if "frame_id" in col_names else -1

            # Precompute local positions for forcing joystick Y to constant 1.0 (sanity check)
            try:
                p1_y_feat_pos = FEATURE_COLUMNS.index("p1_pre_joystick_y")
            except ValueError:
                p1_y_feat_pos = -1
            try:
                p2_y_feat_pos = FEATURE_COLUMNS.index("p2_pre_joystick_y")
            except ValueError:
                p2_y_feat_pos = -1
            try:
                p1_y_tgt_pos = TARGET_COLUMNS.index("p1_pre_joystick_y")
            except ValueError:
                p1_y_tgt_pos = -1

            # Additional positions for forcing joystick X and c-stick X/Y constants
            try:
                p1_x_feat_pos = FEATURE_COLUMNS.index("p1_pre_joystick_x")
            except ValueError:
                p1_x_feat_pos = -1
            try:
                p2_x_feat_pos = FEATURE_COLUMNS.index("p2_pre_joystick_x")
            except ValueError:
                p2_x_feat_pos = -1
            try:
                p1_cx_feat_pos = FEATURE_COLUMNS.index("p1_pre_cstick_x")
            except ValueError:
                p1_cx_feat_pos = -1
            try:
                p2_cx_feat_pos = FEATURE_COLUMNS.index("p2_pre_cstick_x")
            except ValueError:
                p2_cx_feat_pos = -1
            try:
                p1_cy_feat_pos = FEATURE_COLUMNS.index("p1_pre_cstick_y")
            except ValueError:
                p1_cy_feat_pos = -1
            try:
                p2_cy_feat_pos = FEATURE_COLUMNS.index("p2_pre_cstick_y")
            except ValueError:
                p2_cy_feat_pos = -1

            # Target positions (p1 only; targets do not include p2)
            try:
                p1_x_tgt_pos = TARGET_COLUMNS.index("p1_pre_joystick_x")
            except ValueError:
                p1_x_tgt_pos = -1
            try:
                p1_cx_tgt_pos = TARGET_COLUMNS.index("p1_pre_cstick_x")
            except ValueError:
                p1_cx_tgt_pos = -1
            try:
                p1_cy_tgt_pos = TARGET_COLUMNS.index("p1_pre_cstick_y")
            except ValueError:
                p1_cy_tgt_pos = -1

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

            # Valid starts s satisfy: s + SEQUENCE_LENGTH < N  =>  s <= N - SEQUENCE_LENGTH - 1
            # Count = N - SEQUENCE_LENGTH (if positive).
            num_starts = max(0, N - SEQUENCE_LENGTH)
            if num_starts <= 0:
                # No valid windows in this shard
                continue

            per_file.append(
                (
                    mat,
                    feat_idx,
                    tgt_idx,
                    frame_idx,
                    num_starts,
                    p1_y_feat_pos,
                    p2_y_feat_pos,
                    p1_x_feat_pos,
                    p2_x_feat_pos,
                    p1_cx_feat_pos,
                    p2_cx_feat_pos,
                    p1_cy_feat_pos,
                    p2_cy_feat_pos,
                    p1_y_tgt_pos,
                    p1_x_tgt_pos,
                    p1_cx_tgt_pos,
                    p1_cy_tgt_pos,
                )
            )
            total_samples += num_starts

        if total_samples == 0:
            return

        file_order = np.arange(len(per_file))
        # rng.shuffle(file_order)

        for fi in file_order:
            (
                mat,
                feat_idx,
                tgt_idx,
                frame_idx,
                num_starts,
                p1_y_feat_pos,
                p2_y_feat_pos,
                p1_x_feat_pos,
                p2_x_feat_pos,
                p1_cx_feat_pos,
                p2_cx_feat_pos,
                p1_cy_feat_pos,
                p2_cy_feat_pos,
                p1_y_tgt_pos,
                p1_x_tgt_pos,
                p1_cx_tgt_pos,
                p1_cy_tgt_pos,
            ) = per_file[fi]
            if num_starts <= 0 or feat_idx.size == 0:
                continue

            starts = np.arange(num_starts, dtype=np.int64)  # [0 .. N-SEQUENCE_LENGTH-1]
            # rng.shuffle(starts)

            # print(starts)
            # print(len(starts))
            for s in starts:
                if frame_idx >= 0:
                    # Filter: any frame_id inside the window < 0 => skip
                    frame_col = mat[s : s + SEQUENCE_LENGTH, frame_idx]
                    if np.any(frame_col < 0):
                        continue

                x_np = mat[s : s + SEQUENCE_LENGTH, :][
                    :, feat_idx
                ]  # [SEQUENCE_LENGTH, F]
                y_np = mat[s + SEQUENCE_LENGTH, :][tgt_idx]  # predict t+1

                # Determine whether to flip 0/1 stick constants based on the first frame_id parity
                flip_sticks = False
                if frame_idx >= 0:
                    # `frame_col` was computed above when filtering on frame_id; use its first value
                    first_frame_id = int(frame_col[0])
                    flip_sticks = (first_frame_id & 1) == 1  # True if odd

                # Force constants for sanity checking (conditionally flipped by first frame_id parity)
                # Features (both players)
                if p1_y_feat_pos >= 0:
                    x_np[:, p1_y_feat_pos] = 0.0 if flip_sticks else 1.0
                if p2_y_feat_pos >= 0:
                    x_np[:, p2_y_feat_pos] = 0.0 if flip_sticks else 1.0
                if p1_x_feat_pos >= 0:
                    x_np[:, p1_x_feat_pos] = 1.0 if flip_sticks else 0.0
                if p2_x_feat_pos >= 0:
                    x_np[:, p2_x_feat_pos] = 1.0 if flip_sticks else 0.0
                if p1_cx_feat_pos >= 0:
                    x_np[:, p1_cx_feat_pos] = 0.0 if flip_sticks else 1.0
                if p2_cx_feat_pos >= 0:
                    x_np[:, p2_cx_feat_pos] = 0.0 if flip_sticks else 1.0
                if p1_cy_feat_pos >= 0:
                    x_np[:, p1_cy_feat_pos] = 1.0 if flip_sticks else 0.0
                if p2_cy_feat_pos >= 0:
                    x_np[:, p2_cy_feat_pos] = 1.0 if flip_sticks else 0.0

                # Targets (p1 only)
                if p1_y_tgt_pos >= 0:
                    y_np[p1_y_tgt_pos] = np.float32(0.0 if flip_sticks else 1.0)
                if p1_x_tgt_pos >= 0:
                    y_np[p1_x_tgt_pos] = np.float32(1.0 if flip_sticks else 0.0)
                if p1_cx_tgt_pos >= 0:
                    y_np[p1_cx_tgt_pos] = np.float32(0.0 if flip_sticks else 1.0)
                if p1_cy_tgt_pos >= 0:
                    y_np[p1_cy_tgt_pos] = np.float32(1.0 if flip_sticks else 0.0)

                if not (np.isfinite(x_np).all() and np.isfinite(y_np).all()):
                    print("Infinite or NaN!")
                    continue
                X = torch.from_numpy(np.ascontiguousarray(x_np))  # float32
                y = torch.from_numpy(np.ascontiguousarray(y_np))  # float32

                yield X, y
