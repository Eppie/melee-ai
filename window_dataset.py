from __future__ import annotations

import json
import gc
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset, Sampler

from data_types import RawNumpyArray, ProcessedNumpyArray, ProcessedTorchTensor
from feature_transforms import FeatureTransformSpec, feature_spec_from_config

try:
    from pympler import asizeof as pympler_asizeof
except Exception:  # pragma: no cover - optional dependency
    pympler_asizeof = None

_PROFILE_LOCALS_ENABLED = os.environ.get("MELEE_PROFILE_WORKER_LOCALS", "0") == "1"
_PROFILE_LOCALS_MAX = int(os.environ.get("MELEE_PROFILE_WORKER_LOCALS_MAX", "20"))
_PROFILE_ATTRS_ENABLED = os.environ.get("MELEE_PROFILE_DATASET_ATTRS", "0") == "1"
_PROFILE_ATTRS_MAX = int(os.environ.get("MELEE_PROFILE_DATASET_ATTRS_MAX", "20"))
_PROFILED_WORKERS: Dict[int, bool] = {}
_PROFILE_GC_TYPES_ENABLED = os.environ.get("MELEE_PROFILE_GC_TYPES", "0") == "1"
_PROFILE_GC_TYPES_MAX = int(os.environ.get("MELEE_PROFILE_GC_TYPES_MAX", "15"))
_PROFILE_COLLATE_ENABLED = os.environ.get("MELEE_PROFILE_COLLATE", "0") == "1"
_COLLATE_PROFILE_MAX = int(os.environ.get("MELEE_PROFILE_COLLATE_MAX", "1"))
_COLLATE_PROFILED: Dict[int, int] = {}
_PROFILE_PYMPLER_ENABLED = os.environ.get("MELEE_PROFILE_PYMPLER_VARS", "0") == "1"


def _shallow_size(obj: object) -> int:
    """Approximate shallow size of ``obj`` including backing buffers."""
    if isinstance(obj, np.ndarray):
        return int(obj.nbytes)
    if isinstance(obj, torch.Tensor):
        return int(obj.element_size() * obj.nelement())
    try:
        return sys.getsizeof(obj)
    except TypeError:
        return 0


def _deep_size(obj: object, seen: Optional[set[int]] = None, depth: int = 0) -> int:
    """Compute rough deep size via referents, guarding against cycles."""
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    total = _shallow_size(obj)

    # Only descend into container-like objects; skip basic buffers to avoid double counting.
    if isinstance(obj, dict):
        for k, v in obj.items():
            total += _deep_size(k, seen, depth + 1)
            total += _deep_size(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for item in obj:
            total += _deep_size(item, seen, depth + 1)
    return total


def _maybe_profile_worker_locals(local_vars: Dict[str, object]) -> None:
    """When enabled, log the largest locals within a DataLoader worker."""
    if not _PROFILE_LOCALS_ENABLED:
        return
    worker_info = torch.utils.data.get_worker_info()
    worker_id = worker_info.id if worker_info is not None else -1
    if _PROFILED_WORKERS.get(worker_id):
        return

    rows: List[Tuple[str, int, str]] = []
    for name, value in local_vars.items():
        if name == "self":
            continue
        if isinstance(value, type(sys)):
            continue  # skip modules
        try:
            size = _deep_size(value)
        except Exception:
            size = _shallow_size(value)
        rows.append((name, size, type(value).__name__))

    rows.sort(key=lambda x: x[1], reverse=True)
    if _PROFILE_LOCALS_MAX > 0:
        rows = rows[:_PROFILE_LOCALS_MAX]

    lines = [f"[worker {worker_id}] local object sizes:"]
    for name, size, typ in rows:
        lines.append(f"  {name} ({typ}): {size / (1024**2):.2f} MB")

    dataset_obj = local_vars.get("self")
    if _PROFILE_ATTRS_ENABLED and dataset_obj is not None:
        dataset_attr_rows: List[Tuple[str, int, str]] = []
        for attr_name, attr_value in vars(dataset_obj).items():
            try:
                attr_size = _deep_size(attr_value)
            except Exception:
                attr_size = _shallow_size(attr_value)
            dataset_attr_rows.append((attr_name, attr_size, type(attr_value).__name__))
        dataset_attr_rows.sort(key=lambda x: x[1], reverse=True)
        if _PROFILE_ATTRS_MAX > 0:
            dataset_attr_rows = dataset_attr_rows[:_PROFILE_ATTRS_MAX]
        lines.append(f"[worker {worker_id}] dataset attribute sizes:")
        for name, size, typ in dataset_attr_rows:
            lines.append(f"  self.{name} ({typ}): {size / (1024**2):.2f} MB")

    if _PROFILE_GC_TYPES_ENABLED:
        type_totals: Dict[str, int] = {}
        for obj in gc.get_objects():
            try:
                type_name = type(obj).__name__
            except Exception:
                continue
            try:
                type_totals[type_name] = type_totals.get(type_name, 0) + _shallow_size(obj)
            except Exception:
                continue
        gc_rows = sorted(type_totals.items(), key=lambda x: x[1], reverse=True)
        if _PROFILE_GC_TYPES_MAX > 0:
            gc_rows = gc_rows[:_PROFILE_GC_TYPES_MAX]
        lines.append(f"[worker {worker_id}] GC-tracked types by shallow size:")
        for type_name, total_bytes in gc_rows:
            lines.append(f"  {type_name}: {total_bytes / (1024**2):.2f} MB")

    if _PROFILE_PYMPLER_ENABLED:
        if pympler_asizeof is None:
            lines.append(
                "[pympler disabled: package not available in environment]"
            )
        else:
            lines.append(f"[worker {worker_id}] pympler locals (deep sizes):")
            for name, value in local_vars.items():
                try:
                    size = pympler_asizeof.asizeof(value)
                except Exception:
                    size = _deep_size(value)
                lines.append(
                    f"  {name} ({type(value).__name__}): {size / (1024**2):.4f} MB"
                )
            if dataset_obj is not None:
                lines.append(f"[worker {worker_id}] pympler dataset attrs:")
                for attr_name, attr_value in vars(dataset_obj).items():
                    try:
                        attr_size = pympler_asizeof.asizeof(attr_value)
                    except Exception:
                        attr_size = _deep_size(attr_value)
                    lines.append(
                        f"  self.{attr_name} ({type(attr_value).__name__}): {attr_size / (1024**2):.4f} MB"
                    )

    print("\n".join(lines))
    _PROFILED_WORKERS[worker_id] = True


if _PROFILE_COLLATE_ENABLED:
    from torch.utils.data._utils import collate as _torch_collate

    _ORIG_DEFAULT_COLLATE = _torch_collate.default_collate

    def _profiling_default_collate(batch):
        result = _ORIG_DEFAULT_COLLATE(batch)
        pid = os.getpid()
        count = _COLLATE_PROFILED.get(pid, 0)
        if count >= _COLLATE_PROFILE_MAX:
            return result

        printable: List[Tuple[str, int, str]] = []
        if isinstance(result, dict):
            iterable = result.items()
        elif isinstance(result, (list, tuple)):
            iterable = enumerate(result)
        else:
            iterable = [("value", result)]
        for key, value in iterable:
            try:
                size = _deep_size(value)
            except Exception:
                size = _shallow_size(value)
            printable.append((str(key), size, type(value).__name__))
        printable.sort(key=lambda x: x[1], reverse=True)

        lines = [f"[worker-pid {pid}] collate batch object sizes:"]
        for name, size, typ in printable:
            lines.append(f"  batch[{name}] ({typ}): {size / (1024**2):.2f} MB")
        print("\n".join(lines))

        _COLLATE_PROFILED[pid] = count + 1
        return result

    _torch_collate.default_collate = _profiling_default_collate


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

        # Episode start offsets (needed by episode_start_global_index)
        self._windows = windows.astype(np.int64)
        starts = np.zeros(len(self._windows), dtype=np.int64)
        total = 0
        for idx, count in enumerate(self._windows):
            starts[idx] = total
            total += int(count)
        self._episode_start_indices = starts
        self.total_windows = total

        window_index_path = self.data_dir / "window_index.npy"
        window_index = np.load(window_index_path, mmap_mode="r")
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

    def window_to_episode(self, global_win_idx: int) -> Tuple[int, int]:
        """Map ``global_win_idx`` to an episode index and start offset in O(1)."""
        row = self._window_index[global_win_idx]
        return int(row[0]), int(row[1])

    def episode_start_global_index(self, ep_idx: int) -> int:
        """Return the first global window index owned by ``ep_idx``."""
        return int(self._episode_start_indices[ep_idx])

    def open_episode_arrays(
        self,
        ep: EpisodeInfo
    ) -> Tuple[zarr.Array, Optional[zarr.Array]]:
        """Open the ``X``/``Y`` arrays for ``ep``.

        Example
        -------
        When ``ep`` describes ``episode_id=7`` in ``shard_00002.zarr``, the method
        locates the shard directory, opens ``root['ep_000007']``, and returns the
        ``X`` and ``Y`` arrays.
        """
        key = (ep.shard_id, ep.episode_id)
        shard_path = self._shard_paths.get(ep.shard_id)
        if shard_path is None:
            raise FileNotFoundError(f"Shard path not found for shard_id={ep.shard_id}")

        # Open shard group (directory store). Consolidated metadata improves open latency.
        root = zarr.open_group(str(shard_path), mode="r", path=None)

        ep_name = f"ep_{ep.episode_id:06d}"
        epg = root[ep_name]
        X = epg["X"]  # shape (T, F), float32
        Y = epg.get("Y", None)  # shape (T, Yd) or missing

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
    X: RawNumpyArray, feature_names: Sequence[str], spec: Optional[FeatureTransformSpec]
) -> ProcessedNumpyArray:
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
        feature_transforms: Optional[FeatureTransformSpec] = None
    ) -> None:
        """Prepare the dataset by indexing shards and wiring transforms.

        Example
        -------
        ``WindowDataset('dataset_root')`` loads corpus metadata and stores the
        requested transform spec so future ``__getitem__`` calls transparently
        apply preprocessing before returning tensors.
        """
        super().__init__()
        # TODO: Can we build this index faster? generator?
        self.index = ZarrCorpusIndex(data_dir)
        self.seq_len = self.index.seq_len
        self.transforms = feature_transforms
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
        3. The arrays are converted to ``torch.float32`` tensors and the target array
           defaults to shape ``(3, 0)`` when an episode lacks ``Y`` data.

        The method returns a dictionary containing the tensors alongside the
        ``episode_id`` and the local ``start`` offset, mirroring the exact payload
        consumed by the training loop.
        """
        ep_idx, offset = self.index.window_to_episode(i)
        ep = self.index.episodes[ep_idx]
        start = offset  # within episode, window starts at this index
        L = self.seq_len

        feature_array, target_array = self.index.open_episode_arrays(ep)
        # Slice contiguous window; arrays are (T, F) and (T, Yd)
        feature_window = feature_array[start : start + L, :]  # (L, F)
        target_window = target_array[start : start + L, :]  # (L, Yd)

        # Apply per-feature transforms (in-place on view)
        feature_window: RawNumpyArray = np.ascontiguousarray(feature_window)  # ensure contiguous for in-place ops
        feature_window: ProcessedNumpyArray = _apply_feature_transforms(feature_window, self._feature_names, self.transforms)
        features_out: ProcessedTorchTensor = torch.from_numpy(feature_window.astype(np.float32, copy=False))
        targets_as_numpy: RawNumpyArray = np.ascontiguousarray(target_window)
        targets_out = torch.from_numpy(targets_as_numpy.astype(np.float32, copy=False))

        result = {
            "X": features_out,
            "Y": targets_out,
            "episode_id": ep.episode_id,
            "start": start,
        }
        _maybe_profile_worker_locals(dict(locals()))
        return result


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

        total = self._count_for_epoch(self.epoch)
        if total == 0:
            return

        inds = np.empty(total, dtype=np.int64)
        k = 0
        for epi, ep in enumerate(self.index.episodes):
            base = self.index.episode_start_global_index(epi)
            w = ep.num_windows
            if w <= m:
                continue
            count = ((w - 1 - m) // s) + 1
            stop = m + count * s
            inds[k : k + count] = base + np.arange(m, stop, s, dtype=np.int64)
            k += count

        if k != total:
            inds = inds[:k]

        start_offset = min(self._start_offset, inds.size) if self._start_offset else 0
        self._start_offset = 0
        if start_offset:
            inds = inds[start_offset:]

        if inds.size == 0:
            return

        seed = (self.epoch * 0x9E3779B97F4A7C15 + 4242) & ((1 << 63) - 1)
        rng = np.random.default_rng(seed=seed)
        if inds.size > 1:
            rng.shuffle(inds)

        for value in inds:
            yield int(value)
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
    config: "Config",
) -> Tuple[torch.utils.data.DataLoader, WindowDataset, Sampler[int]]:
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

    pin_memory = config.train.pin_memory

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
warnings.filterwarnings(
    "ignore",
    message="`torch.distributed.reduce_op` is deprecated",
    category=FutureWarning,
)
