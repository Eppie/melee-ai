from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, Sampler

from config import get_config


@dataclass(frozen=True)
class EpisodeInfo:
    episode_id: int
    shard_id: int
    num_frames: int  # = X.shape[0] = Y.shape[0]
    num_windows: int  # = max(frames - seq_len + 1, 0)


class _LRUEpisodeCache:
    """Tiny per-worker cache for opened episode arrays to cut directory lookups."""

    def __init__(self, max_open: int = 8) -> None:
        self.max_open = max_open
        self._keys: List[Tuple[int, int]] = []  # (shard_id, episode_id)
        self._vals: List[Tuple[zarr.Array, Optional[zarr.Array]]] = []

    def get(self, key: Tuple[int, int]) -> Optional[Tuple[zarr.Array, Optional[zarr.Array]]]:
        try:
            i = self._keys.index(key)
        except ValueError:
            return None
        # LRU touch
        self._keys.append(self._keys.pop(i))
        self._vals.append(self._vals.pop(i))
        return self._vals[-1]

    def put(self, key: Tuple[int, int], value: Tuple[zarr.Array, Optional[zarr.Array]]) -> None:
        if key in self._keys:
            i = self._keys.index(key)
            self._keys.pop(i)
            self._vals.pop(i)
        self._keys.append(key)
        self._vals.append(value)
        if len(self._keys) > self.max_open:
            self._keys.pop(0)
            self._vals.pop(0)


class ZarrCorpusIndex:
    """
    Loads your dataset root (with shard_*.zarr, lengths.npy, wins_per_ep.npy, index.jsonl, meta.json).
    Provides O(log E) mapping from global window index -> (episode_idx, start_offset).
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        meta_path = self.data_dir / "meta.json"
        lengths_path = self.data_dir / "lengths.npy"
        wins_path = self.data_dir / "wins_per_ep.npy"
        index_path = self.data_dir / "index.jsonl"

        if not (meta_path.exists() and lengths_path.exists() and wins_path.exists() and index_path.exists()):
            raise FileNotFoundError("Expected meta.json, lengths.npy, wins_per_ep.npy, index.jsonl in data_dir")

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
                ep_rows.append(EpisodeInfo(
                    episode_id=int(row["episode_id"]),
                    shard_id=int(row["shard_id"]),
                    num_frames=int(row["frames"]),
                    num_windows=int(windows[len(ep_rows)]),  # aligned order
                ))
        assert len(ep_rows) == len(lengths), "index.jsonl and lengths.npy out of sync"
        self.episodes: List[EpisodeInfo] = ep_rows

        # prefix sums over windows for fast mapping
        self._windows = windows.astype(np.int64)
        self._cumulative_windows = np.cumsum(self._windows, dtype=np.int64)  # length E
        self.total_windows: int = int(self._cumulative_windows[-1]) if len(self._cumulative_windows) else 0

        # shard paths
        self._shard_paths: Dict[int, Path] = {}
        for sdir in sorted(self.data_dir.glob("shard_*.zarr")):
            # shard_00012.zarr -> 12
            sid = int(sdir.stem.split("_")[1])
            self._shard_paths[sid] = sdir

    # -------- mapping helpers --------

    def window_to_episode(self, global_win_idx: int) -> Tuple[int, int]:
        """
        Map global window index -> (episode_idx, start_offset).
        start_offset is in [0, wins_in_episode-1].
        """
        if not (0 <= global_win_idx < self.total_windows):
            raise IndexError(f"window index {global_win_idx} out of range 0..{self.total_windows - 1}")
        ep_idx = int(np.searchsorted(self._cumulative_windows, global_win_idx, side="right"))
        base = 0 if ep_idx == 0 else int(self._cumulative_windows[ep_idx - 1])
        offset = int(global_win_idx - base)
        return ep_idx, offset

    def episode_start_global_index(self, ep_idx: int) -> int:
        if ep_idx == 0:
            return 0
        return int(self._cumulative_windows[ep_idx - 1])

    # -------- zarr access --------

    def open_episode_arrays(
            self,
            ep: EpisodeInfo,
            *,
            cache: Optional[_LRUEpisodeCache] = None,
    ) -> Tuple[zarr.Array, Optional[zarr.Array]]:
        """
        Returns (X_array, Y_array|None) for the episode.
        """
        key = (ep.shard_id, ep.episode_id)
        if cache is not None:
            cached = cache.get(key)
            if cached is not None:
                return cached

        shard_path = self._shard_paths.get(ep.shard_id)
        if shard_path is None:
            raise FileNotFoundError(f"Shard path not found for shard_id={ep.shard_id}")

        # Open shard group (directory store). Consolidated metadata improves open latency.
        root = zarr.open_group(str(shard_path), mode="r", path=None)

        ep_name = f"ep_{ep.episode_id:06d}"
        epg = root[ep_name]
        X = epg["X"]  # shape (T, F), float32
        Y = epg.get("Y", None)  # shape (T, Yd) or missing

        if cache is not None:
            cache.put(key, (X, Y))
        return X, Y


# -------------------------------------------
# Feature transforms & column selection hooks
# -------------------------------------------

FeatureFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class FeatureTransformSpec:
    """
    Per-feature transform hook. Map feature name -> function operating on a (L,) or (L,1) view.
    Keep it vectorized; it will be applied column-wise.
    """
    by_name: Dict[str, FeatureFn]


def _apply_feature_transforms(
        X: np.ndarray,
        feature_names: Sequence[str],
        spec: Optional[FeatureTransformSpec]
) -> np.ndarray:
    if spec is None or not spec.by_name:
        return X
    # apply per-column (vectorized)
    out = X
    for j, name in enumerate(feature_names):
        fn = spec.by_name.get(name)
        if fn is not None:
            # view as (L,) vector for convenience
            col = out[:, j]
            out[:, j] = fn(col)
    return out


def _select_columns(
        X: np.ndarray,
        names: Sequence[str],
        keep: Optional[Sequence[str]]
) -> Tuple[np.ndarray, List[int], List[str]]:
    if not keep:
        return X, list(range(X.shape[1])), list(names)
    name2idx = {n: i for i, n in enumerate(names)}
    idxs: List[int] = []
    for n in keep:
        if n not in name2idx:
            raise KeyError(f"Requested feature '{n}' not found")
        idxs.append(name2idx[n])
    idxs_np = np.asarray(idxs, dtype=np.int64)
    return X[:, idxs_np], idxs, [names[i] for i in idxs]


class WindowDataset(Dataset):
    """
    Map-style dataset over ALL valid windows in the corpus.

    __getitem__(i) returns:
        dict(
            X: FloatTensor [L, F_sel],
            Y: FloatTensor [L, Y_sel] or empty (0-dim second axis) if no targets,
            episode_id: int,
            start: int,
        )
    """

    def __init__(
            self,
            data_dir: str | Path,
            *,
            feature_keep: Optional[Sequence[str]] = None,
            target_keep: Optional[Sequence[str]] = None,
            feature_transforms: Optional[FeatureTransformSpec] = None,
            ep_cache_size: int = 8,
            return_numpy: bool = False,
    ) -> None:
        super().__init__()
        self.index = ZarrCorpusIndex(data_dir)
        self.seq_len = self.index.seq_len
        self.feature_keep = list(feature_keep) if feature_keep else None
        self.target_keep = list(target_keep) if target_keep else None
        self.transforms = feature_transforms
        self._cache = _LRUEpisodeCache(max_open=ep_cache_size)
        self._return_numpy = return_numpy  # if True, return np.float32 arrays instead of torch tensors

        # Pre-compute column indices (validated lazily against first episode slice)
        self._feature_keep_idxs: Optional[List[int]] = None
        self._target_keep_idxs: Optional[List[int]] = None

    def __len__(self) -> int:
        return self.index.total_windows

    def _resolve_column_selections(
            self, X: np.ndarray, Y: Optional[np.ndarray]
    ) -> Tuple[List[int], Optional[List[int]], List[str], Optional[List[str]]]:
        # compute once
        if self._feature_keep_idxs is None:
            _, feat_idxs, feat_names = _select_columns(
                X[:1, :], self.index.feature_names, self.feature_keep
            )
            self._feature_keep_idxs = feat_idxs
            self._feature_names_sel = feat_names
        if Y is not None and self._target_keep_idxs is None:
            _, targ_idxs, targ_names = _select_columns(
                Y[:1, :], self.index.target_names, self.target_keep
            )
            self._target_keep_idxs = targ_idxs
            self._target_names_sel = targ_names
        return self._feature_keep_idxs, self._target_keep_idxs, self._feature_names_sel, getattr(self,
                                                                                                 "_target_names_sel",
                                                                                                 None)

    def __getitem__(self, i: int) -> Dict[str, object]:
        ep_idx, offset = self.index.window_to_episode(i)
        ep = self.index.episodes[ep_idx]
        start = offset  # within episode, window starts at this index
        L = self.seq_len

        Xa, Ya = self.index.open_episode_arrays(ep, cache=self._cache, )
        # Slice contiguous window; arrays are (T, F) and (T, Yd)
        Xw = Xa[start:start + L, :]  # (L, F)
        Yw = None if Ya is None else Ya[start:start + L, :]  # (L, Yd)

        # Select columns (if requested) - compute idxs on first call
        feat_keep_idxs, targ_keep_idxs, feat_names_sel, _ = self._resolve_column_selections(Xw, Yw)
        if feat_keep_idxs is not None:
            Xw = Xw[:, np.asarray(feat_keep_idxs, dtype=np.int64)]
        if Yw is not None and targ_keep_idxs is not None:
            Yw = Yw[:, np.asarray(targ_keep_idxs, dtype=np.int64)]

        # Apply per-feature transforms (in-place on view)
        Xw = np.ascontiguousarray(Xw)  # ensure contiguous for in-place ops
        Xw = _apply_feature_transforms(Xw, feat_names_sel, self.transforms)

        if self._return_numpy:
            X_out = Xw.astype(np.float32, copy=False)
            Y_out = (Yw.astype(np.float32, copy=False) if Yw is not None
                     else np.empty((L, 0), dtype=np.float32))
        else:
            X_out = torch.from_numpy(Xw.astype(np.float32, copy=False))
            Y_out = (torch.from_numpy(Yw.astype(np.float32, copy=False)) if Yw is not None
                     else torch.empty((L, 0), dtype=torch.float32))

        return {
            "X": X_out,
            "Y": Y_out,
            "episode_id": ep.episode_id,
            "start": start,
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
        num_samples: Optional[int] = None,
        stride: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.index = index
        self.num_samples = int(num_samples) if num_samples is not None else None
        self.stride = int(stride)
        self.generator = generator
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _count_for_epoch(self, epoch: int) -> int:
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
        if self.num_samples is not None:
            return self.num_samples
        return self._count_for_epoch(self.epoch)

    def __iter__(self) -> Iterator[int]:
        s = self.stride
        m = self.epoch % s

        # Build the list of global indices that satisfy the stride condition for this epoch.
        inds: List[int] = []
        for epi, ep in enumerate(self.index.episodes):
            base = self.index.episode_start_global_index(epi)
            w = ep.num_windows
            if m < w:
                inds.extend(base + t for t in range(m, w, s))

        # Deterministic per-epoch shuffle
        g = self.generator or torch.Generator()
        seed = (self.epoch * 0x9E3779B97F4A7C15 + 4242) % (2 ** 63 - 1)
        g.manual_seed(seed)
        if len(inds) > 1:
            perm = torch.randperm(len(inds), generator=g).tolist()
            inds = [inds[i] for i in perm]

        if self.num_samples is None:
            yield from inds
            return

        # If a fixed number of samples is requested, cap/extend accordingly
        k = self.num_samples
        if k <= len(inds):
            yield from inds[:k]
        else:
            out = list(inds)
            while len(out) < k:
                perm = torch.randperm(len(inds), generator=g).tolist()
                out.extend(inds[i] for i in perm)
            yield from out[:k]




def worker_init_fn(worker_id: int) -> None:
    """
    Set distinct NumPy / PyTorch seeds for each worker. Avoids identical shuffles per worker.
    """
    # Same recipe as PyTorch DistributedSampler docs
    base_seed = torch.initial_seed() % 2 ** 31
    np.random.seed(base_seed + worker_id)


def make_dataloader() -> Tuple[torch.utils.data.DataLoader, WindowDataset, Sampler[int]]:
    """
    Builds dataset + sampler + DataLoader with tuned defaults.
    """
    config = get_config()
    ds = WindowDataset(
        config.zarr.out_root,
        feature_keep=config.train.feature_keep,
        target_keep=config.train.target_keep,
        feature_transforms=None,
        return_numpy=False,
    )

    target_windows = config.train.windows_per_epoch
    if config.train.steps_per_epoch is not None:
        target_windows = config.train.steps_per_epoch * config.train.batch_size

    effective_num_samples = config.train.num_samples
    if target_windows is not None:
        effective_num_samples = target_windows

    stride = getattr(config.train, "stride", 1)
    sampler = RandomWindowSampler(
        index=ds.index,
        stride=stride,
        num_samples=effective_num_samples,
    )

    mp_ctx = None
    if config.train.num_workers and config.train.num_workers > 0:
        try:
            mp_ctx = torch.multiprocessing.get_context("spawn")
        except RuntimeError:
            mp_ctx = None

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=config.train.batch_size,
        sampler=sampler,
        num_workers=config.train.num_workers,
        pin_memory=config.train.pin_memory,
        prefetch_factor=config.train.prefetch_factor if config.train.num_workers > 0 else None,
        persistent_workers=config.train.persistent_workers if config.train.num_workers > 0 else False,
        worker_init_fn=worker_init_fn,
        drop_last=False,
        multiprocessing_context=mp_ctx,
    )
    return loader, ds, sampler
