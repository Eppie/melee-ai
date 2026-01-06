from __future__ import annotations

import ctypes
import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, IterableDataset, Sampler

from data_types import RawNumpyArray, ProcessedNumpyArray, ProcessedTorchTensor
from feature_transforms import apply_feature_transforms

FLOAT32_BYTES = np.dtype(np.float32).itemsize


@dataclass(frozen=True)
class EpisodeInfo:
    episode_id: int
    shard_id: int
    num_frames: int  # = X.shape[0] = Y.shape[0]
    num_windows: int  # = max(frames - seq_len + 1, 0)


@dataclass
class _MinimalIndex:
    """Minimal index-like object for SequentialEpisodeIterableDataset compatibility."""
    feature_names: List[str]
    target_names: List[str]


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
        # sequential_episodes: top-level key (v1+) or nested in build_config (fallback)
        self.sequential_episodes: bool = self.meta.get(
            "sequential_episodes",
            self.meta.get("build_config", {}).get("zarr", {}).get("sequential_episodes", False),
        )

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
    ) -> None:
        """Prepare the dataset by indexing shards."""
        super().__init__()
        self.index = ZarrCorpusIndex(data_dir)
        self.seq_len = self.index.seq_len
        self._feature_names = tuple(self.index.feature_names)
        self._target_names = tuple(self.index.target_names)
        self._feature_names_sel = list(self._feature_names)
        self._target_names_sel = list(self._target_names)
        self._shard_cache: Dict[int, zarr.Group] = {}
        self._episode_cache: Dict[Tuple[int, int], Tuple[zarr.Array, zarr.Array]] = {}

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
        """Load window ``i`` and apply feature transforms."""
        ep_idx, offset = self.index.window_to_episode(i)
        ep = self.index.episodes[ep_idx]
        start = offset  # within episode, window starts at this index

        feature_array, target_array = self.index.open_episode_arrays(ep)
        # Slice contiguous window; arrays are (T, F) and (T, Yd)
        feature_window = feature_array[
            start : start + self.seq_len, :
        ]  # (seq_len, num_features)
        target_window = target_array[
            start : start + self.seq_len, :
        ]  # (seq_len, num_targets)

        # Apply feature transforms
        feature_window: RawNumpyArray = np.ascontiguousarray(feature_window)
        feature_window: ProcessedNumpyArray = apply_feature_transforms(
            feature_window, self._feature_names
        )
        features_out: ProcessedTorchTensor = torch.from_numpy(
            feature_window.astype(np.float32, copy=False)
        )
        targets_as_numpy: RawNumpyArray = np.ascontiguousarray(target_window)
        targets_out = torch.from_numpy(targets_as_numpy.astype(np.float32, copy=False))

        return {
            "X": features_out,
            "Y": targets_out,
            "episode_id": ep.episode_id,
            "start": start,
        }


class SequentialEpisodeIterableDataset(IterableDataset):
    """
    IterableDataset that assigns whole episodes to workers for true episode-based batching.

    Each worker processes a disjoint subset of episodes. Episodes are loaded in groups
    of `interleave_episodes`, and windows from all loaded episodes are shuffled together
    before yielding. This provides:

      - Efficient I/O: each episode loaded as single chunk read
      - Good shuffle quality: windows shuffled across multiple episodes
      - Episode diversity: with 16 workers * 4 episodes = 64 episodes per batch cycle
      - No window_index.npy needed: iterates directly over episodes

    Usage:
        dataset = SequentialEpisodeIterableDataset(data_dir, stride=8, interleave_episodes=4)
        loader = DataLoader(dataset, batch_size=256, num_workers=16)
        for epoch in range(num_epochs):
            dataset.set_epoch(epoch)
            for batch in loader:
                ...

    Note: Uses shared memory for epoch synchronization with persistent_workers.
    """

    def __init__(
        self,
        data_dir: str | Path,
        stride: int = 1,
        interleave_episodes: int = 4,
    ) -> None:
        """Initialize the dataset by loading episode metadata.

        Args:
            data_dir: Path to the processed Zarr dataset.
            stride: Stride between consecutive windows (default 1).
            interleave_episodes: Number of episodes to load and shuffle together
                per worker. Higher values increase episode diversity per batch
                but use more memory. Default 4 means 16 workers * 4 = 64 episodes
                contribute to each batch cycle. Memory usage is approximately
                interleave_episodes * 3 MB per worker.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.stride = stride
        self.interleave_episodes = max(1, interleave_episodes)
        # Use shared memory for epoch so persistent workers see updates
        self._shared_epoch = multiprocessing.Value(ctypes.c_int64, 0)
        self._shared_start_offset = multiprocessing.Value(ctypes.c_int64, 0)

        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load episode metadata from the dataset directory."""
        meta_path = self.data_dir / "meta.json"
        index_path = self.data_dir / "index.jsonl"
        wins_path = self.data_dir / "wins_per_ep.npy"

        if not all(p.exists() for p in [meta_path, index_path, wins_path]):
            raise FileNotFoundError(
                f"Expected meta.json, index.jsonl, wins_per_ep.npy in {self.data_dir}"
            )

        with meta_path.open("r") as f:
            self.meta = json.load(f)
        self.seq_len: int = int(self.meta["build_config"]["seq_len"])
        self._feature_names: Tuple[str, ...] = tuple(self.meta["schema"]["features"])
        self._target_names: Tuple[str, ...] = tuple(self.meta["schema"]["targets"])

        # Compatibility attributes for ColumnMap.from_dataset
        self._feature_names_sel: List[str] = list(self._feature_names)
        self._target_names_sel: List[str] = list(self._target_names)

        # Create a minimal index-like object for compatibility with code that expects
        # dataset.index.feature_names / dataset.index.target_names
        self.index = _MinimalIndex(
            feature_names=list(self._feature_names),
            target_names=list(self._target_names),
        )

        windows = np.load(wins_path)

        # Load episode info from index.jsonl
        self.episodes: List[EpisodeInfo] = []
        with index_path.open("r") as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                self.episodes.append(
                    EpisodeInfo(
                        episode_id=int(row["episode_id"]),
                        shard_id=int(row["shard_id"]),
                        num_frames=int(row["frames"]),
                        num_windows=int(windows[i]),
                    )
                )

        # Shard paths
        self._shard_paths: Dict[int, Path] = {}
        for sdir in sorted(self.data_dir.glob("shard_*.zarr")):
            sid = int(sdir.stem.split("_")[1])
            self._shard_paths[sid] = sdir

    def set_epoch(self, epoch: int) -> None:
        """Set epoch for shuffling episodes. Call before each epoch.

        Uses shared memory so persistent workers see the update.
        """
        with self._shared_epoch.get_lock():
            self._shared_epoch.value = int(epoch)

    def set_start_offset(self, offset: int) -> None:
        """Skip the first ``offset`` windows when resuming. Applied once then reset.

        Uses shared memory so persistent workers see the update.
        """
        with self._shared_start_offset.get_lock():
            self._shared_start_offset.value = max(0, int(offset))

    def _count_windows_for_epoch(self, epoch: int) -> int:
        """Count total windows for this epoch respecting stride."""
        s = self.stride
        m = epoch % s
        total = 0
        for ep in self.episodes:
            w = ep.num_windows
            if w <= m:
                continue
            total += ((w - 1 - m) // s) + 1
        return total

    @property
    def epoch(self) -> int:
        """Current epoch (read from shared memory)."""
        return self._shared_epoch.value

    def __len__(self) -> int:
        """Return approximate number of windows for progress bars."""
        return self._count_windows_for_epoch(self.epoch)

    def _load_episode_data(
        self,
        ep: EpisodeInfo,
        shard_cache: Dict[int, zarr.Group],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load episode features and targets from Zarr storage.

        Args:
            ep: Episode metadata.
            shard_cache: Cache of open shard groups.

        Returns:
            Tuple of (features, targets) numpy arrays.
        """
        if ep.shard_id not in shard_cache:
            shard_path = self._shard_paths[ep.shard_id]
            shard_cache[ep.shard_id] = zarr.open_group(str(shard_path), mode="r")
        root = shard_cache[ep.shard_id]

        ep_name = f"ep_{ep.episode_id:06d}"
        epg = root[ep_name]
        features: np.ndarray = epg["X"][:]  # Load full episode (single chunk)
        targets: np.ndarray = epg["Y"][:]
        return features, targets

    def __iter__(self) -> Iterator[Dict[str, object]]:
        """Yield windows with multi-episode interleaving and shuffling.

        Each worker:
        1. Gets a disjoint subset of episodes (episode_order[worker_id::num_workers])
        2. Loads `interleave_episodes` episodes at a time
        3. Builds list of all (local_idx, offset) pairs for windows in loaded episodes
        4. Shuffles the list
        5. Yields windows in shuffled order
        6. Repeats with next group of episodes
        """
        worker_info = torch.utils.data.get_worker_info()

        # Read epoch from shared memory (so persistent workers see updates)
        epoch = self._shared_epoch.value

        # Global episode shuffle (same across all workers for determinism)
        num_episodes = len(self.episodes)
        episode_order = np.arange(num_episodes, dtype=np.int32)
        global_seed = (epoch * 0x9E3779B97F4A7C15 + 4243) % (2**63 - 1)
        global_rng = np.random.default_rng(global_seed)
        global_rng.shuffle(episode_order)

        # Worker assignment
        if worker_info is None:
            worker_id = 0
            num_workers = 1
            worker_episodes = episode_order
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            worker_episodes = episode_order[worker_id::num_workers]

        # Per-worker RNG for within-group shuffling (deterministic per worker per epoch)
        worker_seed = (epoch * 0x9E3779B97F4A7C15 + worker_id * 7919 + 12345) % (2**63 - 1)
        worker_rng = np.random.default_rng(worker_seed)

        s = self.stride
        m = epoch % s
        N = self.interleave_episodes

        # Read and reset start_offset from shared memory
        with self._shared_start_offset.get_lock():
            skip_remaining = self._shared_start_offset.value
            self._shared_start_offset.value = 0

        # Per-worker shard cache to avoid reopening zarr groups
        shard_cache: Dict[int, zarr.Group] = {}

        # Process episodes in groups of N
        ep_cursor = 0
        while ep_cursor < len(worker_episodes):
            # Load up to N episodes that have valid windows for this stride phase
            loaded_episodes: List[Tuple[EpisodeInfo, np.ndarray, np.ndarray]] = []
            while len(loaded_episodes) < N and ep_cursor < len(worker_episodes):
                ep_idx = worker_episodes[ep_cursor]
                ep_cursor += 1
                ep = self.episodes[ep_idx]

                # Skip episodes with insufficient windows for this stride phase
                if ep.num_windows <= m:
                    continue

                features, targets = self._load_episode_data(ep, shard_cache)
                loaded_episodes.append((ep, features, targets))

            if not loaded_episodes:
                continue

            # Build list of (local_episode_idx, offset) for all windows across loaded episodes
            window_indices: List[Tuple[int, int]] = []
            for local_idx, (ep, _, _) in enumerate(loaded_episodes):
                for offset in range(m, ep.num_windows, s):
                    window_indices.append((local_idx, offset))

            # Shuffle windows across all loaded episodes
            worker_rng.shuffle(window_indices)

            # Yield windows in shuffled order
            for local_idx, offset in window_indices:
                # Handle start_offset for resumption
                if skip_remaining > 0:
                    skip_remaining -= 1
                    continue

                ep, features, targets = loaded_episodes[local_idx]

                feature_window = features[offset : offset + self.seq_len, :]
                target_window = targets[offset : offset + self.seq_len, :]

                # Apply feature transforms
                feature_window = np.ascontiguousarray(feature_window)
                feature_window = apply_feature_transforms(
                    feature_window, self._feature_names
                )

                yield {
                    "X": torch.from_numpy(
                        feature_window.astype(np.float32, copy=False)
                    ),
                    "Y": torch.from_numpy(
                        np.ascontiguousarray(target_window).astype(np.float32, copy=False)
                    ),
                    "episode_id": ep.episode_id,
                    "start": offset,
                }


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


class SequentialEpisodeSampler(Sampler[int]):
    """
    Sequential sampling within episodes: shuffle episodes, then yield all windows
    within each episode in order.

    This sampler is designed for datasets built with ``sequential_episodes=True``,
    where each episode is stored as a single chunk. It improves temporal locality
    by keeping windows from the same episode together in the batch stream.

    Stride semantics match :class:`RandomWindowSampler`:
      - stride=1: sample all windows from each episode.
      - stride=s>1: in epoch `e`, include only windows where `offset % s == e % s`.
    """

    def __init__(
        self,
        *,
        index: ZarrCorpusIndex,
        stride: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Create a sampler that iterates episodes in shuffled order, windows sequentially."""
        super().__init__()
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.index = index
        self.stride = int(stride)
        self.generator = generator
        self.epoch = 0
        self._start_offset = 0

    def set_epoch(self, epoch: int) -> None:
        """Record the epoch for shuffling and stride phase."""
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
        """Yield global window indices: episodes shuffled, windows sequential within each."""
        s = self.stride
        m = self.epoch % s
        num_episodes = len(self.index.episodes)

        # Create shuffled episode order
        episode_order = np.arange(num_episodes, dtype=np.int32)
        seed = (self.epoch * 0x9E3779B97F4A7C15 + 4243) % (2**63 - 1)
        rng = np.random.default_rng(seed)
        rng.shuffle(episode_order)

        # Build list of (global_window_idx) in episode-sequential order
        total = self._count_for_epoch(self.epoch)
        if total == 0:
            return
        buffer = np.empty(total, dtype=np.int32)
        cursor = 0

        for epi in episode_order:
            ep = self.index.episodes[epi]
            w = ep.num_windows
            if w <= m:
                continue
            # Windows within episode that match stride, in sequential order
            local_offsets = np.arange(m, w, s, dtype=np.int32)
            if local_offsets.size == 0:
                continue
            base = self.index.episode_start_global_index(epi)
            buffer[cursor : cursor + local_offsets.size] = base + local_offsets
            cursor += local_offsets.size

        if cursor != total:
            buffer = buffer[:cursor]

        # Apply start offset (for resumption)
        if self._start_offset:
            start_offset = min(self._start_offset, buffer.size)
            buffer = buffer[start_offset:]
            self._start_offset = 0
        else:
            self._start_offset = 0

        if buffer.size == 0:
            return

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
) -> Tuple[torch.utils.data.DataLoader, Union[WindowDataset, SequentialEpisodeIterableDataset], Any]:
    """Construct the dataset, sampler, and DataLoader.

    The dataset type is automatically selected based on how the dataset was built:
      - ``sequential_episodes=True`` → :class:`SequentialEpisodeIterableDataset`
        (episode-based batching, workers handle whole episodes)
      - ``sequential_episodes=False`` (default) → :class:`WindowDataset` + :class:`RandomWindowSampler`
        (global window shuffling)

    Returns:
        Tuple of (DataLoader, dataset, epoch_setter) where epoch_setter has
        set_epoch() and set_start_offset() methods (either sampler or dataset).
    """
    stride = config.train.stride
    data_root = Path(config.zarr.out_root)

    # Check dataset build configuration to select appropriate loader type
    meta_path = data_root / "meta.json"
    with meta_path.open("r") as f:
        meta = json.load(f)
    sequential_episodes = meta.get(
        "sequential_episodes",
        meta.get("build_config", {}).get("zarr", {}).get("sequential_episodes", False),
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

    if sequential_episodes:
        # Episode-based batching: workers handle whole episodes with interleaving
        interleave = 4  # Load 4 episodes at a time, shuffle windows across them
        num_workers = config.train.num_workers or 1
        episodes_per_batch_cycle = interleave * num_workers
        print(
            f"[dataloader] Using SequentialEpisodeIterableDataset "
            f"(interleave={interleave} episodes/worker, {episodes_per_batch_cycle} episodes/batch cycle, stride={stride})"
        )
        ds = SequentialEpisodeIterableDataset(
            config.zarr.out_root,
            stride=stride,
            interleave_episodes=interleave,
        )

        # IterableDataset doesn't use a sampler - the dataset controls iteration
        # prefetch_factor must be specified for IterableDataset with num_workers > 0
        prefetch = config.train.prefetch_factor if config.train.num_workers > 0 else None

        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            pin_memory=config.train.pin_memory,
            prefetch_factor=prefetch,
            persistent_workers=(
                config.train.persistent_workers if config.train.num_workers > 0 else False
            ),
            worker_init_fn=worker_init_fn,
            drop_last=False,
            multiprocessing_context=mp_ctx,
        )
        # Return dataset as epoch_setter since it has set_epoch/set_start_offset
        return loader, ds, ds

    else:
        # Global window shuffling: traditional map-style dataset with sampler
        print(
            f"[dataloader] Using WindowDataset + RandomWindowSampler "
            f"(global shuffling, stride={stride})"
        )
        ds = WindowDataset(config.zarr.out_root)
        sampler = RandomWindowSampler(
            index=ds.index,
            stride=stride,
        )

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
