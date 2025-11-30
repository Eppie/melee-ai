from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, Sampler

from data_types import RawNumpyArray, ProcessedTorchTensor

FLOAT32_BYTES = np.dtype(np.float32).itemsize


@dataclass(frozen=True)
class EpisodeInfo:
    episode_id: int
    shard_id: int
    num_frames: int  # = X.shape[0] = Y.shape[0]
    num_windows: int  # = max(frames - seq_len + 1, 0)


class ZarrCorpusIndex:
    """
    Loads your dataset root (with shard_*.zarr, lengths.npy, wins_per_ep.npy, index.jsonl, meta.json).
    Provides O(1) mapping from global window index -> (episode_idx, start_offset).
    """

    def __init__(self, data_dir: str | Path) -> None:
        """Load metadata and map global-window lookups via a precomputed table.

        Example
        -------
        When ``data_dir`` contains ``meta.json``, ``lengths.npy`` and two episodes
        with window counts ``[3, 2]``, the constructor loads ``window_index.npy``
        and later ``window_to_episode(4)`` reads the precomputed row ``(1, 1)``,
        showing the fifth global window belongs to episode ``1`` at offset ``1``.
        """
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "meta.json"
        lengths_path = self.data_dir / "lengths.npy"
        wins_path = self.data_dir / "wins_per_ep.npy"
        index_path = self.data_dir / "index.jsonl"

        if not (
            meta_path.exists()
            and lengths_path.exists()
            and wins_path.exists()
            and index_path.exists()
        ):
            raise FileNotFoundError(
                f"Expected meta.json, lengths.npy, wins_per_ep.npy, index.jsonl in {self.data_dir}"
            )

        with meta_path.open("r") as f:
            self.meta = json.load(f)
        self.seq_len: int = int(self.meta["build_config"]["seq_len"])
        self.feature_names: List[str] = list(self.meta["schema"]["features"])
        self.target_names: List[str] = list(self.meta["schema"]["targets"])

        lengths = np.load(lengths_path)  # (E,) frames per episode
        windows = np.load(wins_path)  # (E,) windows per episode
        # index.jsonl: episode_id, shard_id, frames
        ep_rows: List[EpisodeInfo] = []
        with index_path.open("r") as f:
            for line in f:
                row = json.loads(line)
                ep_rows.append(
                    EpisodeInfo(
                        episode_id=int(row["episode_id"]),
                        shard_id=int(row["shard_id"]),
                        num_frames=int(row["frames"]),
                        num_windows=int(windows[len(ep_rows)]),  # aligned order
                    )
                )
        assert len(ep_rows) == len(lengths), "index.jsonl and lengths.npy out of sync"
        self.episodes: List[EpisodeInfo] = ep_rows

        # Episode start offsets (needed by episode_start_global_index)
        self._windows = windows.astype(np.int32)
        starts = np.zeros(len(self._windows), dtype=np.int32)
        total = 0
        for idx, count in enumerate(self._windows):
            starts[idx] = total
            total += int(count)
        self._episode_start_indices = starts
        self.total_windows = total

        self._window_index_path = self.data_dir / "window_index.npy"
        window_index = np.load(self._window_index_path, mmap_mode="r")
        if window_index.ndim != 2 or window_index.shape[1] != 2:
            raise ValueError(
                f"window_index.npy must have shape (N, 2), got {window_index.shape}"
            )
        if window_index.shape[0] != self.total_windows:
            raise ValueError(
                "window_index row count "
                f"{window_index.shape[0]} does not match total_windows={self.total_windows}"
            )
        self._window_index = window_index

        # shard paths
        self._shard_paths: Dict[int, Path] = {}
        for sdir in sorted(self.data_dir.glob("shard_*.zarr")):
            # shard_00012.zarr -> 12
            sid = int(sdir.stem.split("_")[1])
            self._shard_paths[sid] = sdir

    def _window_index_array(self) -> np.memmap:
        if self._window_index is None:
            self._window_index = np.load(self._window_index_path, mmap_mode="r")
        return self._window_index

    def __getstate__(self) -> Dict[str, object]:
        state = self.__dict__.copy()
        state["_window_index"] = None
        return state

    def __setstate__(self, state: Dict[str, object]) -> None:
        self.__dict__.update(state)

    def window_to_episode(self, global_win_idx: int) -> Tuple[int, int]:
        """Map ``global_win_idx`` to an episode index and start offset in O(1)."""
        window_index = self._window_index_array()
        row = window_index[global_win_idx]
        return int(row[0]), int(row[1])

    def episode_start_global_index(self, ep_idx: int) -> int:
        """Return the first global window index owned by ``ep_idx``."""
        return int(self._episode_start_indices[ep_idx])

    def open_episode_arrays(
        self, ep: EpisodeInfo
    ) -> Tuple[zarr.Array, Optional[zarr.Array]]:
        """Open the ``features``/``targets`` arrays for ``ep``.

        Example
        -------
        When ``ep`` describes ``episode_id=7`` in ``shard_00002.zarr``, the method
        locates the shard directory, opens ``root['ep_000007']``, and returns the
        ``features`` and ``targets`` arrays.
        """
        cache_key = (ep.shard_id, ep.episode_id)
        cached = getattr(self, "_episode_cache", None)
        if cached is not None and cache_key in cached:
            return cached[cache_key]

        shard_cache = getattr(self, "_shard_cache", None)
        if shard_cache is not None and ep.shard_id in shard_cache:
            root = shard_cache[ep.shard_id]
        else:
            shard_path = self._shard_paths.get(ep.shard_id)
            if shard_path is None:
                raise FileNotFoundError(
                    f"Shard path not found for shard_id={ep.shard_id}"
                )
            root = zarr.open_group(str(shard_path), mode="r", path=None)
            if shard_cache is not None:
                shard_cache[ep.shard_id] = root

        ep_name = f"ep_{ep.episode_id:06d}"
        epg = root[ep_name]
        features = epg["X"]  # shape (T, F), float32
        targets = epg["Y"]

        if cached is not None:
            cached[cache_key] = (features, targets)

        return features, targets


class WindowDataset(Dataset):
    """
    Map-style dataset over ALL valid windows in the corpus.

    __getitem__(i) returns:
        dict(
            X: FloatTensor [L, F],
            Y: FloatTensor [L, Yd] or empty (0-dim second axis) if no targets,
            episode_id: int,
            start: int,
        )
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        in_memory: bool = False,
        in_memory_shared: bool = False,
    ) -> None:
        """Prepare the dataset by indexing shards.

        ``in_memory`` loads episode arrays lazily into RAM on first access
        (per-process). ``in_memory_shared`` eagerly loads all episodes into
        torch shared memory so multiple workers can reuse the same backing
        buffers without duplicating memory.
        """
        super().__init__()
        self.index = ZarrCorpusIndex(data_dir)
        self.seq_len = self.index.seq_len
        self._feature_names = tuple(self.index.feature_names)
        self._target_names = tuple(self.index.target_names)
        self._feature_names_sel = list(self._feature_names)
        self._target_names_sel = list(self._target_names)
        self._shard_cache: Dict[int, zarr.Group] = {}
        self._episode_cache: Dict[Tuple[int, int], Tuple[zarr.Array, zarr.Array]] = {}
        self._shared_episode_cache: Dict[
            Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._episode_arrays_cache: Dict[
            Tuple[int, int], Tuple[np.ndarray, np.ndarray]
        ] = {}
        self._in_memory = bool(in_memory or in_memory_shared)
        self._in_memory_shared = bool(in_memory_shared)
        # Enforce preprocessed data presence
        preprocessed = self.index.meta.get("preprocessed", {})
        if not (preprocessed.get("features") and preprocessed.get("targets")):
            raise RuntimeError(
                "Dataset is not preprocessed. Regenerate dataset with preprocessing enabled."
            )
        required_targets = {
            "p1_main_stick_idx",
            "p1_c_stick_idx",
            "p1_shoulder_idx",
            "p1_button_a",
            "p1_button_b",
            "p1_button_xy",
            "p1_button_z",
            "p1_button_lr",
        }
        missing = sorted(required_targets.difference(self._target_names_sel))
        if missing:
            raise RuntimeError(
                f"Preprocessed dataset missing required target columns: {missing}"
            )

        if self._in_memory_shared:
            self._load_all_episodes_shared()

    def __len__(self) -> int:
        """Return the total number of sliding windows across the corpus."""
        return self.index.total_windows

    def estimate_batch_bytes(self, batch_size: int) -> int:
        """
        Approximate how many bytes a single batch occupies in host memory.
        Helps tune DataLoader prefetching so we avoid spawning too many inflight
        batches when using multiple workers.
        """
        if batch_size <= 0:
            return 0
        feat_cols = len(self._feature_names_sel)
        target_cols = len(self._target_names_sel)
        floats_per_window = self.seq_len * (feat_cols + target_cols)
        return batch_size * floats_per_window * FLOAT32_BYTES

    def __getitem__(self, i: int) -> Dict[str, object]:
        """Load window ``i`` from preprocessed storage."""
        ep_idx, offset = self.index.window_to_episode(i)
        ep = self.index.episodes[ep_idx]
        start = offset  # within episode, window starts at this index

        feature_array, target_array = self._get_episode_arrays(ep)
        # Slice contiguous window; arrays are (T, F) and (T, Yd)
        if isinstance(feature_array, torch.Tensor):
            features_out = feature_array[
                start : start + self.seq_len, :
            ]  # already float32 tensor
            targets_out = target_array[start : start + self.seq_len, :]
        else:
            feature_window = feature_array[
                start : start + self.seq_len, :
            ]  # (seq_len, num_features)
            target_window = target_array[
                start : start + self.seq_len, :
            ]  # (seq_len, num_targets)

            features_out: ProcessedTorchTensor = torch.from_numpy(
                np.ascontiguousarray(feature_window).astype(np.float32, copy=False)
            )
            targets_as_numpy: RawNumpyArray = np.ascontiguousarray(target_window)
            targets_out = torch.from_numpy(
                targets_as_numpy.astype(np.float32, copy=False)
            )

        return {
            "X": features_out,
            "Y": targets_out,
            "episode_id": ep.episode_id,
            "start": start,
        }

    def _get_episode_arrays(
        self, ep: EpisodeInfo
    ) -> Tuple[np.ndarray | zarr.Array | torch.Tensor, np.ndarray | zarr.Array | torch.Tensor]:
        """Return feature/target arrays for ``ep``, optionally keeping them in RAM."""
        cache_key = (ep.shard_id, ep.episode_id)

        if self._in_memory_shared:
            cached_shared = self._shared_episode_cache.get(cache_key)
            if cached_shared is not None:
                return cached_shared

        if self._in_memory:
            cached = self._episode_arrays_cache.get(cache_key)
            if cached is not None:
                return cached

        feature_array, target_array = self.index.open_episode_arrays(ep)
        if self._in_memory_shared:
            # Should only happen if a new episode is encountered after initial load;
            # fall back to on-the-fly load into shared memory.
            f_tensor = torch.from_numpy(
                np.ascontiguousarray(feature_array[:], dtype=np.float32)
            ).share_memory_()
            t_tensor = torch.from_numpy(
                np.ascontiguousarray(target_array[:], dtype=np.float32)
            ).share_memory_()
            self._shared_episode_cache[cache_key] = (f_tensor, t_tensor)
            return f_tensor, t_tensor

        if self._in_memory:
            features_np = np.ascontiguousarray(feature_array[:], dtype=np.float32)
            targets_np = np.ascontiguousarray(target_array[:], dtype=np.float32)
            self._episode_arrays_cache[cache_key] = (features_np, targets_np)
            return features_np, targets_np

        return feature_array, target_array

    def _load_all_episodes_shared(self) -> None:
        """Eagerly load all episodes into shared memory tensors (one copy for all workers)."""
        total_eps = len(self.index.episodes)
        print(f"[dataset] Preloading {total_eps} episodes into shared memory...")
        for idx, ep in enumerate(self.index.episodes):
            cache_key = (ep.shard_id, ep.episode_id)
            feature_array, target_array = self.index.open_episode_arrays(ep)
            f_tensor = torch.from_numpy(
                np.ascontiguousarray(feature_array[:], dtype=np.float32)
            ).share_memory_()
            t_tensor = torch.from_numpy(
                np.ascontiguousarray(target_array[:], dtype=np.float32)
            ).share_memory_()
            self._shared_episode_cache[cache_key] = (f_tensor, t_tensor)
            if (idx + 1) % 25 == 0 or (idx + 1) == total_eps:
                print(f"[dataset]   loaded {idx + 1}/{total_eps} episodes")


class RandomWindowSampler(Sampler[int]):
    """
    Random selection of windows honoring a global *stride* across episode-local window
    offsets.

    Stride semantics:
      - stride=1: sample over all windows (same behavior as before).
      - stride=s>1: in epoch `e` (0-based), include only windows whose episode-local
        start offset `t` satisfies `t % s == (e % s)`. Over `s` epochs you will cover
        all windows (subject to episode lengths not exactly divisible by `s`).

    Note: PyTorch 2.2+ Sampler API — do not pass or rely on `data_source` in the base.
    """

    def __init__(
        self,
        *,
        index: ZarrCorpusIndex,
        stride: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Create a sampler that enforces a stride across episode windows."""
        super().__init__()
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.index = index
        self.stride = int(stride)
        self.generator = generator
        self.epoch = 0
        self._start_offset = 0

    def set_epoch(self, epoch: int) -> None:
        """Record the epoch so future iterations honor ``epoch % stride``."""
        self.epoch = int(epoch)

    def set_start_offset(self, offset: int) -> None:
        """Skip the first ``offset`` samples the next time the sampler runs."""
        self._start_offset = max(0, int(offset))

    def _count_for_epoch(self, epoch: int) -> int:
        """Count how many windows satisfy the stride for ``epoch``."""
        s = self.stride
        m = epoch % s
        total = 0
        for ep in self.index.episodes:
            w = ep.num_windows
            if w <= m:
                continue
            total += ((w - 1 - m) // s) + 1
        return total

    def __len__(self) -> int:
        """Return the number of indices that ``__iter__`` will generate."""
        return self._count_for_epoch(self.epoch)

    def __iter__(self) -> Iterator[int]:
        """Yield global window indices for the configured stride with shuffling."""
        s = self.stride
        m = self.epoch % s

        # Build the list of global indices that satisfy the stride condition for this epoch.
        total = self._count_for_epoch(self.epoch)
        if total == 0:
            return
        buffer = np.empty(total, dtype=np.int32)
        cursor = 0
        for epi, ep in enumerate(self.index.episodes):
            w = ep.num_windows
            if w <= m:
                continue
            local_offsets = np.arange(m, w, s, dtype=np.int32)
            if local_offsets.size == 0:
                continue
            base = self.index.episode_start_global_index(epi)
            buffer[cursor : cursor + local_offsets.size] = base + local_offsets
            cursor += local_offsets.size
        if cursor != total:
            buffer = buffer[:cursor]
        if self._start_offset:
            start_offset = min(self._start_offset, buffer.size)
            buffer = buffer[start_offset:]
            self._start_offset = 0
        else:
            self._start_offset = 0
        if buffer.size == 0:
            return
        if buffer.size > 1:
            seed = (self.epoch * 0x9E3779B97F4A7C15 + 4242) % (2**63 - 1)
            rng = np.random.default_rng(seed)
            rng.shuffle(buffer)
        for value in buffer:
            yield int(value)
        return


def worker_init_fn(worker_id: int) -> None:
    """Seed NumPy and PyTorch for ``worker_id``."""
    # Same recipe as PyTorch DistributedSampler docs
    base_seed = torch.initial_seed() % 2**31
    np.random.seed(base_seed + worker_id)


def make_dataloader(
    config: "Config",
) -> Tuple[torch.utils.data.DataLoader, WindowDataset, Sampler[int]]:
    """Construct the dataset, sampler, and DataLoader."""
    ds = WindowDataset(
        config.zarr.out_root,
        in_memory_shared=getattr(config.train, "in_memory_shared", False),
    )

    stride = config.train.stride
    sampler = RandomWindowSampler(
        index=ds.index,
        stride=stride,
    )

    mp_ctx = None
    start_method = getattr(config.train, "worker_start_method", None)
    if config.train.num_workers and config.train.num_workers > 0 and start_method:
        try:
            mp_ctx = torch.multiprocessing.get_context(start_method)
        except RuntimeError as exc:
            print(
                f"[dataloader] Requested start method '{start_method}' unavailable "
                f"({exc}); falling back to PyTorch default."
            )
            mp_ctx = None

    pin_memory = config.train.pin_memory
    prefetch_factor = None
    if config.train.num_workers and config.train.num_workers > 0:
        prefetch_factor = config.train.prefetch_factor
        max_prefetch_mb = getattr(config.train, "max_loader_prefetch_mb", None)
        if max_prefetch_mb:
            batch_bytes = max(1, ds.estimate_batch_bytes(config.train.batch_size))
            max_prefetch_bytes = max_prefetch_mb * 1024 * 1024
            total_batches_budget = max_prefetch_bytes // batch_bytes
            budget_saturated = False
            if total_batches_budget == 0:
                total_batches_budget = 1
                budget_saturated = True
            allowed_per_worker = total_batches_budget // config.train.num_workers
            if allowed_per_worker == 0:
                allowed_per_worker = 1
                budget_saturated = True
            if allowed_per_worker < prefetch_factor:
                approx_batch_mb = batch_bytes / (1024**2)
                print(
                    "[dataloader] Reducing prefetch_factor from "
                    f"{prefetch_factor} to {allowed_per_worker} to honor "
                    f"{max_prefetch_mb} MiB prefetch budget (batch ≈ "
                    f"{approx_batch_mb:.2f} MiB)."
                )
                if budget_saturated:
                    print(
                        "[dataloader] Consider lowering train.num_workers or "
                        "batch_size, or increase train.max_loader_prefetch_mb "
                        "if you need more throughput."
                    )
                prefetch_factor = allowed_per_worker

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=config.train.batch_size,
        sampler=sampler,
        num_workers=config.train.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=(
            config.train.persistent_workers if config.train.num_workers > 0 else False
        ),
        worker_init_fn=worker_init_fn,
        drop_last=False,
        multiprocessing_context=mp_ctx,
    )
    return loader, ds, sampler
