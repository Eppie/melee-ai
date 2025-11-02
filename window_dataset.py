from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, Sampler

from config import FeatureConfig, get_config
from utils import _resolve_device
from feature_transforms import FeatureTransformSpec, feature_spec_from_config


@dataclass(frozen=True)
class EpisodeInfo:
    episode_id: int
    shard_id: int
    num_frames: int  # = X.shape[0] = Y.shape[0]
    num_windows: int  # = max(frames - seq_len + 1, 0)


# keys are randomly distributed, so hit rate is inversely proportional to the number of episodes
class _LRUEpisodeCache:
    """Tiny per-worker cache for opened episode arrays to cut directory lookups."""

    def __init__(self, max_open: int = 8) -> None:
        """Initialize the cache and show how capacity behaves in practice.

        Example
        -------
        ``cache = _LRUEpisodeCache(max_open=2)`` starts empty. After calling
        ``cache.put((0, 1), (X0, Y0))`` and ``cache.put((0, 2), (X1, Y1))`` both
        entries remain. Inserting ``cache.put((0, 3), (X2, Y2))`` pushes out the
        oldest ``(0, 1)`` pair so subsequent ``get`` calls only see the two most
        recently touched episodes.
        """
        self.max_open = max_open
        self._keys: List[Tuple[int, int]] = []  # (shard_id, episode_id)
        self._vals: List[Tuple[zarr.Array, Optional[zarr.Array]]] = []

    def get(
        self, key: Tuple[int, int]
    ) -> Optional[Tuple[zarr.Array, Optional[zarr.Array]]]:
        """Retrieve and mark an entry as recently used with a concrete trace.

        Example
        -------
        If ``cache`` already stored ``(0, 5) -> (X5, None)`` and ``(0, 6) -> (X6, Y6)``,
        calling ``cache.get((0, 5))`` returns ``(X5, None)`` and rotates that key to
        the end of ``_keys`` so the next eviction would remove ``(0, 6)`` instead.
        Requesting an unseen key yields ``None`` without mutating the cache.
        """
        try:
            i = self._keys.index(key)
        except ValueError:
            return None
        # LRU touch
        self._keys.append(self._keys.pop(i))
        self._vals.append(self._vals.pop(i))
        return self._vals[-1]

    def put(
        self, key: Tuple[int, int], value: Tuple[zarr.Array, Optional[zarr.Array]]
    ) -> None:
        """Insert ``key`` while maintaining the maximum cache size.

        Example
        -------
        Starting from ``[(0, 1), (0, 2)]`` in ``_keys`` with ``max_open=2``:

        1. ``put((0, 2), v)`` updates the value and moves ``(0, 2)`` to the end so
           it becomes the most recent entry.
        2. ``put((0, 3), v3)`` appends ``(0, 3)`` and pops ``(0, 1)`` from the
           front, ensuring only the two newest handles remain cached.
        """
        if key in self._keys:
            i = self._keys.index(key)
            self._keys.pop(i)
            self._vals.pop(i)
        self._keys.append(key)
        self._vals.append(value)
        if len(self._keys) > self.max_open:
            self._keys.pop(0)
            self._vals.pop(0)


# TODO: Can we do this with a generator? If not, can we compute it at dataset generation time?
# TODO: O(log E) is good, but can we get constant time?
class ZarrCorpusIndex:
    """
    Loads your dataset root (with shard_*.zarr, lengths.npy, wins_per_ep.npy, index.jsonl, meta.json).
    Provides O(log E) mapping from global window index -> (episode_idx, start_offset).
    """

    def __init__(self, data_dir: str | Path) -> None:
        """Load metadata and build prefix sums for global-window lookups.

        Example
        -------
        When ``data_dir`` contains ``meta.json``, ``lengths.npy`` and two episodes
        with window counts ``[3, 2]``, the constructor builds
        ``_cumulative_windows=[3, 5]``. Later ``window_to_episode(4)`` uses this
        array to return ``(1, 1)`` because the fifth global window belongs to
        episode index ``1`` starting at offset ``1``.
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
                "Expected meta.json, lengths.npy, wins_per_ep.npy, index.jsonl in data_dir"
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

        # prefix sums over windows for fast mapping
        self._windows = windows.astype(np.int64)
        self._cumulative_windows = np.cumsum(self._windows, dtype=np.int64)  # length E
        self.total_windows: int = (
            int(self._cumulative_windows[-1]) if len(self._cumulative_windows) else 0
        )

        # shard paths
        self._shard_paths: Dict[int, Path] = {}
        for sdir in sorted(self.data_dir.glob("shard_*.zarr")):
            # shard_00012.zarr -> 12
            sid = int(sdir.stem.split("_")[1])
            self._shard_paths[sid] = sdir

    def window_to_episode(self, global_win_idx: int) -> Tuple[int, int]:
        """Map ``global_win_idx`` to an episode index and start offset.

        Example
        -------
        With ``wins_per_ep = [3, 2]`` the cumulative windows are ``[3, 5]``. Calling
        ``window_to_episode(2)`` returns ``(0, 2)`` because the third window still
        falls within episode ``0`` at offset ``2``. Calling ``window_to_episode(3)``
        returns ``(1, 0)`` showing how the search jumps to the next episode when the
        index crosses a prefix boundary.
        """
        if not (0 <= global_win_idx < self.total_windows):
            raise IndexError(
                f"window index {global_win_idx} out of range 0..{self.total_windows - 1}"
            )
        ep_idx = int(
            np.searchsorted(self._cumulative_windows, global_win_idx, side="right")
        )
        base = 0 if ep_idx == 0 else int(self._cumulative_windows[ep_idx - 1])
        offset = int(global_win_idx - base)
        return ep_idx, offset

    def episode_start_global_index(self, ep_idx: int) -> int:
        """Return the first global window index owned by ``ep_idx``.

        Example
        -------
        Using the same ``wins_per_ep = [3, 2]`` example, ``ep_idx=0`` returns ``0``
        while ``ep_idx=1`` returns ``3`` so you can offset local window indices by
        this amount to obtain their global counterparts.
        """
        if ep_idx == 0:
            return 0
        return int(self._cumulative_windows[ep_idx - 1])

    def open_episode_arrays(
        self,
        ep: EpisodeInfo,
        *,
        cache: Optional[_LRUEpisodeCache] = None,
    ) -> Tuple[zarr.Array, Optional[zarr.Array]]:
        """Open and optionally cache the ``X``/``Y`` arrays for ``ep``.

        Example
        -------
        When ``ep`` describes ``episode_id=7`` in ``shard_00002.zarr``, the method
        locates the shard directory, opens ``root['ep_000007']``, and returns the
        ``X`` and ``Y`` arrays. Supplying a cache reuses the same handles when the
        worker revisits the episode later in the same epoch, saving filesystem
        round-trips.
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


def _resolve_feature_groups(
    feature_names: Sequence[str],
    requested: Sequence[str],
) -> List[Tuple[int, ...]]:
    """Expand requested feature names into column index groups with examples.

    Example
    -------
    Suppose ``feature_names`` contains ``('p1_main_stick_x', 'p1_main_stick_y',
    'p2_main_stick_x', 'p2_main_stick_y')`` and ``requested=('main_stick_x',
    'main_stick_y')``. The helper first tries the names verbatim (fails) and then
    matches both ``p1_`` and ``p2_`` prefixes, returning ``[(0, 1), (2, 3)]`` so
    transforms run on each player slice independently.
    """
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    if all(name in name_to_idx for name in requested):
        return [tuple(name_to_idx[name] for name in requested)]

    prefixes: set[str] = set()
    for name in feature_names:
        head, _, tail = name.partition("_")
        if tail and head.startswith("p") and head[1:].isdigit():
            prefixes.add(head)

    groups: List[Tuple[int, ...]] = []
    for prefix in sorted(prefixes):
        indices: List[int] = []
        found_all = True
        for feature in requested:
            col = f"{prefix}_{feature}"
            idx = name_to_idx.get(col)
            if idx is None:
                found_all = False
                break
            indices.append(idx)
        if found_all and indices:
            groups.append(tuple(indices))
    return groups


def _apply_feature_transforms(
    X: np.ndarray, feature_names: Sequence[str], spec: Optional[FeatureTransformSpec]
) -> np.ndarray:
    """Apply configured transforms to every requested column with a trace.

    Example
    -------
    ``spec`` contains two steps:

    1. ``('scale', features=('foo',), factor=0.5)``
    2. ``('offset', features=('foo', 'bar'), delta=1)``

    For input ``X = [[2., 5.], [4., 7.]]`` with feature names ``('foo', 'bar')`` we
    first scale ``foo`` to ``[1., 2.]`` then add ``1`` to both columns yielding
    ``[[2., 6.], [3., 8.]]``. The modified array is returned, demonstrating the
    in-place but staged nature of the transform pipeline.
    """
    if spec is None or not spec.steps:
        return X

    out = X
    for step in spec.steps:
        index_groups = _resolve_feature_groups(feature_names, step.features)
        if not index_groups:
            continue
        for group in index_groups:
            idxs = np.asarray(group, dtype=np.int64)
            if idxs.size == 1:
                col_idx = int(idxs[0])
                block = out[:, col_idx].copy()
                result = step.fn(block)
                if result is None:
                    result = block
                if result.shape != block.shape:
                    raise ValueError(
                        f"Transform '{step.transform}' expected output shape {block.shape}, got {result.shape}."
                    )
                out[:, col_idx] = result
            else:
                block = out[:, idxs].copy()
                result = step.fn(block)
                if result is None:
                    result = block
                if result.shape != block.shape:
                    raise ValueError(
                        f"Transform '{step.transform}' expected output shape {block.shape}, got {result.shape}."
                    )
                out[:, idxs] = result
    return out


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
        feature_transforms: Optional[FeatureTransformSpec] = None,
        ep_cache_size: int = 8,
        return_numpy: bool = False,
    ) -> None:
        """Prepare the dataset by indexing shards and wiring transforms.

        Example
        -------
        ``WindowDataset('dataset_root', ep_cache_size=1)`` loads corpus metadata,
        builds an LRU cache that keeps one episode open per worker, and stores the
        requested transform spec so future ``__getitem__`` calls transparently
        apply preprocessing before returning tensors.
        """
        super().__init__()
        # TODO: Can we build this index faster? generator?
        self.index = ZarrCorpusIndex(data_dir)
        self.seq_len = self.index.seq_len
        self.transforms = feature_transforms
        # TODO: How much does this actually help?
        self._cache = _LRUEpisodeCache(max_open=ep_cache_size)
        # TODO: Why do we have this?
        self._return_numpy = (
            return_numpy  # if True, return np.float32 arrays instead of torch tensors
        )

        self._feature_names = tuple(self.index.feature_names)
        self._target_names = tuple(self.index.target_names)
        self._feature_names_sel = list(self._feature_names)
        self._target_names_sel = list(self._target_names)

    def __len__(self) -> int:
        """Return the total number of sliding windows across the corpus.

        Example
        -------
        If the index reports ``total_windows=120_000`` this method simply returns
        that value, matching how PyTorch uses ``len(dataset)`` to size an epoch.
        """
        return self.index.total_windows

    def __getitem__(self, i: int) -> Dict[str, object]:
        """Load window ``i`` and show each intermediate tensor transformation.

        Example
        -------
        For ``seq_len=3`` and ``i=4``:

        1. ``window_to_episode`` might yield ``(ep_idx=1, offset=1)`` so we slice
           frames ``[1:4]`` from the episode arrays.
        2. After copying into contiguous buffers we run feature transforms such as
           scaling or palette snapping.
        3. If ``return_numpy`` is ``False`` the arrays are converted to
           ``torch.float32`` tensors and the target array defaults to shape
           ``(3, 0)`` when an episode lacks ``Y`` data.

        The method returns a dictionary containing the tensors alongside the
        ``episode_id`` and the local ``start`` offset, mirroring the exact payload
        consumed by the training loop.
        """
        ep_idx, offset = self.index.window_to_episode(i)
        ep = self.index.episodes[ep_idx]
        start = offset  # within episode, window starts at this index
        L = self.seq_len

        Xa, Ya = self.index.open_episode_arrays(
            ep,
            cache=self._cache,
        )
        # Slice contiguous window; arrays are (T, F) and (T, Yd)
        Xw = Xa[start : start + L, :]  # (L, F)
        Yw = None if Ya is None else Ya[start : start + L, :]  # (L, Yd)

        # Apply per-feature transforms (in-place on view)
        Xw = np.ascontiguousarray(Xw)  # ensure contiguous for in-place ops
        Xw = _apply_feature_transforms(Xw, self._feature_names, self.transforms)
        if Yw is not None:
            Yw = np.ascontiguousarray(Yw)
            Yw = _apply_feature_transforms(Yw, self._target_names, self.transforms)

        if self._return_numpy:
            X_out = Xw.astype(np.float32, copy=False)
            Y_out = (
                Yw.astype(np.float32, copy=False)
                if Yw is not None
                else np.empty((L, 0), dtype=np.float32)
            )
        else:
            X_out = torch.from_numpy(Xw.astype(np.float32, copy=False))
            Y_out = (
                torch.from_numpy(Yw.astype(np.float32, copy=False))
                if Yw is not None
                else torch.empty((L, 0), dtype=torch.float32)
            )

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
        stride: int = 1,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Create a sampler that enforces a stride across episode windows.

        Example
        -------
        With ``stride=2`` and two episodes having window counts ``[3, 4]`` the
        sampler's ``__iter__`` in epoch ``0`` yields offsets ``[0, 2, 0, 2]`` across
        episodes, while epoch ``1`` produces ``[1, 3, 1, 3]`` (where valid). The
        constructor stores the generator so shuffling remains reproducible.
        """
        super().__init__()
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.index = index
        self.stride = int(stride)
        self.generator = generator
        self.epoch = 0
        self._start_offset = 0

    def set_epoch(self, epoch: int) -> None:
        """Record the epoch so future iterations honor ``epoch % stride``.

        Example
        -------
        Calling ``set_epoch(3)`` with ``stride=2`` means ``__iter__`` will only
        visit windows whose local offsets satisfy ``t % 2 == 1`` because the epoch's
        modulo is ``1``.
        """
        self.epoch = int(epoch)

    def set_start_offset(self, offset: int) -> None:
        """Skip the first ``offset`` samples the next time the sampler runs.

        Example
        -------
        After drawing the indices ``[10, 20, 30]`` the sampler applies
        ``set_start_offset(1)`` so the very next ``__iter__`` call discards ``10``
        and starts yielding from ``20``. The internal counter resets to ``0`` after
        iteration so future epochs consume the full sequence again.
        """
        self._start_offset = max(0, int(offset))

    def _count_for_epoch(self, epoch: int) -> int:
        """Count how many windows satisfy the stride for ``epoch``.

        Example
        -------
        With ``stride=3`` and an episode containing ``5`` windows, epoch ``0``
        contributes ``2`` windows (offsets ``0`` and ``3``). Epoch ``1`` contributes
        offsets ``1`` and ``4`` (also ``2`` windows), while epoch ``2`` contributes
        just offset ``2``. Summing across episodes produces the number returned by
        ``__len__``.
        """
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
        """Return the number of indices that ``__iter__`` will generate.

        Example
        -------
        For ``stride=2`` with window counts ``[3, 4]`` and ``epoch=0`` the helper
        reports ``5`` because episode ``0`` contributes offsets ``0`` and ``2``
        while episode ``1`` contributes ``0``, ``2`` and ``4``.
        """
        return self._count_for_epoch(self.epoch)

    def __iter__(self) -> Iterator[int]:
        """Yield global window indices for the configured stride with shuffling.

        Example
        -------
        Continuing the ``stride=2`` scenario, ``__iter__`` first enumerates all
        valid offsets that satisfy ``t % 2 == epoch % 2``. It then applies
        ``torch.randperm`` when a generator is supplied, so two consecutive epochs
        with the same seed produce identical shuffled orders, ensuring reproducible
        training batches.
        """
        s = self.stride
        m = self.epoch % s

        # Build the list of global indices that satisfy the stride condition for this epoch.
        inds: List[int] = []
        for epi, ep in enumerate(self.index.episodes):
            base = self.index.episode_start_global_index(epi)
            w = ep.num_windows
            if m < w:
                inds.extend(base + t for t in range(m, w, s))

        start_offset = min(self._start_offset, len(inds)) if self._start_offset else 0
        self._start_offset = 0
        if start_offset:
            inds = inds[start_offset:]

        if not inds:
            return

        # Deterministic per-epoch shuffle
        g = self.generator or torch.Generator()
        seed = (self.epoch * 0x9E3779B97F4A7C15 + 4242) % (2**63 - 1)
        g.manual_seed(seed)
        if len(inds) > 1:
            perm = torch.randperm(len(inds), generator=g).tolist()
            inds = [inds[i] for i in perm]

        yield from inds
        return


def worker_init_fn(worker_id: int) -> None:
    """Seed NumPy and PyTorch for ``worker_id`` with a short computation trace.

    Example
    -------
    When PyTorch assigns base seed ``123`` to the worker, this helper computes
    ``base_seed = 123 % 2**31`` and seeds NumPy with ``base_seed + worker_id``. For
    worker ``2`` the resulting NumPy seed is ``125`` so each DataLoader worker
    shuffles batches differently.
    """
    # Same recipe as PyTorch DistributedSampler docs
    base_seed = torch.initial_seed() % 2**31
    np.random.seed(base_seed + worker_id)


def make_dataloader(
    config: 'Config',
) -> (
    Tuple[torch.utils.data.DataLoader, WindowDataset, Sampler[int]]
):
    """Construct the dataset, sampler, and DataLoader with an explicit example.

    Example
    -------
    When configuration specifies ``batch_size=8``, ``stride=4`` and ``num_workers=2``
    this function:

    1. Builds ``WindowDataset`` with feature transforms from the config.
    2. Instantiates :class:`RandomWindowSampler` using the dataset's index and the
       configured stride.
    3. Creates ``DataLoader`` with two workers, pinned memory (on CUDA), and
       ``worker_init_fn`` so each worker gets a unique seed.

    The three-tuple ``(loader, dataset, sampler)`` is returned so training scripts
    can iterate over ``loader`` while still accessing ``dataset`` metadata and the
    sampler to adjust epochs.
    """
    feature_spec = feature_spec_from_config(config.features)
    ds = WindowDataset(
        config.zarr.out_root,
        feature_transforms=feature_spec,
        return_numpy=False,
    )

    stride = config.train.stride
    sampler = RandomWindowSampler(
        index=ds.index,
        stride=stride,
    )

    mp_ctx = None
    if config.train.num_workers and config.train.num_workers > 0:
        try:
            mp_ctx = torch.multiprocessing.get_context("spawn")
        except RuntimeError:
            mp_ctx = None

    device = _resolve_device(None)
    pin_memory = config.train.pin_memory and device.type == "cuda"

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=config.train.batch_size,
        sampler=sampler,
        num_workers=config.train.num_workers,
        pin_memory=pin_memory,
        prefetch_factor=(
            config.train.prefetch_factor if config.train.num_workers > 0 else None
        ),
        persistent_workers=(
            config.train.persistent_workers if config.train.num_workers > 0 else False
        ),
        worker_init_fn=worker_init_fn,
        drop_last=False,
        multiprocessing_context=mp_ctx,
    )
    return loader, ds, sampler
