import json
import math
import os
import shutil
import time
from concurrent.futures import (
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
    FIRST_COMPLETED,
)
from dataclasses import dataclass, fields
from operator import attrgetter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import tqdm
import zarr

from column_map import ColumnMap
from config import get_config, init_config
from constants import BUTTON_TARGET_NAMES
from data_types import RawNumpyArray
from controller_quantization import quantize_targets
from libmelee.melee.console import Console
from schema import (
    COMMON_SPEC,
    PLAYER_SPEC,
    Row,
    extract_row,
    get_feature_names,
    get_raw_target_names,
)
from feature_transforms import apply_feature_transforms
from train.value_head import (
    build_reward_feature_index,
    compute_value_targets,
)

ROW_FIELDS = tuple(fields(Row))
_ROW_FIELD_NAMES = [field.name for field in ROW_FIELDS]
_ROW_FIELD_INDEX = {name: idx for idx, name in enumerate(_ROW_FIELD_NAMES)}
_ROW_ATTR_GETTER = attrgetter(*_ROW_FIELD_NAMES)
_NUM_COMMON_FIELDS = len(COMMON_SPEC)
_NUM_PLAYER_FIELDS = len(PLAYER_SPEC)

DERIVED_FEATURES = ("value_target",)
# Each raw replay is stored twice: original and flipped player perspectives.
EPISODES_PER_REPLAY = 2


class _RawTargetColumnMap:
    """Minimal column map for raw controller targets used during dataset build."""

    def __init__(self, target_names: Sequence[str]) -> None:
        idx = {n: i for i, n in enumerate(target_names)}
        required = [
            "p1_main_stick_x",
            "p1_main_stick_y",
            "p1_c_stick_x",
            "p1_c_stick_y",
            "p1_shoulder_analog",
        ] + list(BUTTON_TARGET_NAMES)
        missing = [name for name in required if name not in idx]
        if missing:
            raise KeyError(
                f"Raw target layout missing columns; missing={missing}, available={target_names}"
            )
        self.y_main = (idx["p1_main_stick_x"], idx["p1_main_stick_y"])
        self.y_c = (idx["p1_c_stick_x"], idx["p1_c_stick_y"])
        self.y_shoulder = idx["p1_shoulder_analog"]
        self.y_buttons = [idx[name] for name in BUTTON_TARGET_NAMES]


def _ensure_config_initialized() -> None:
    """Ensure the global config singleton exists (needed inside worker processes)."""
    try:
        get_config()
    except RuntimeError:
        init_config()


def _swap_row_players(row: Row) -> Row:
    """Produce a new :class:`Row` with player 1/2 fields swapped.

    Example
    -------
    Given ``Row(p1_percent=10, p2_percent=20, stage=1)`` the helper returns a row
    where ``p1_percent=20`` and ``p2_percent=10`` while ``stage`` remains ``1``.
    This supports flipped-identity dataset variants.
    """
    values = _ROW_ATTR_GETTER(row)
    start_p1 = _NUM_COMMON_FIELDS
    start_p2 = start_p1 + _NUM_PLAYER_FIELDS
    swapped = (
        values[:_NUM_COMMON_FIELDS]
        + values[start_p2 : start_p2 + _NUM_PLAYER_FIELDS]
        + values[start_p1:start_p2]
    )
    return Row(*swapped)


def _flip_rows(rows: List[Row]) -> List[Row]:
    """Return a copy of ``rows`` with player 1/2 fields swapped frame-by-frame."""
    return [_swap_row_players(row) for row in rows]


@dataclass(frozen=True)
class Schema:
    features: List[str]
    targets: List[str]


@dataclass(frozen=True)
class ProcessedEpisode:
    features: RawNumpyArray
    targets: RawNumpyArray
    feat_dtypes: List[str]
    targ_dtypes: List[str]
    feature_names: List[str]
    target_names: List[str]


def _player_active(
    rows: List[Row], prefix: str, *, stick_eps: float = 0.05, min_frames: int = 10
) -> bool:
    """Return True if the specified player shows meaningful controller input."""
    stick_x = f"{prefix}main_stick_x"
    stick_y = f"{prefix}main_stick_y"
    c_x = f"{prefix}c_stick_x"
    c_y = f"{prefix}c_stick_y"
    shoulder = f"{prefix}shoulder_analog"
    btn_fields = [
        f"{prefix}button_a",
        f"{prefix}button_b",
        f"{prefix}button_xy",
        f"{prefix}button_z",
        f"{prefix}button_lr",
    ]

    active_frames = 0
    for row in rows:
        try:
            # TODO: Duplicate
            if abs(getattr(row, stick_x, 0.5) - 0.5) > stick_eps:
                active_frames += 1
                if active_frames >= min_frames:
                    return True
                continue
            if abs(getattr(row, stick_y, 0.5) - 0.5) > stick_eps:
                active_frames += 1
                if active_frames >= min_frames:
                    return True
                continue
            # TODO: Duplicate
            if abs(getattr(row, c_x, 0.5) - 0.5) > stick_eps:
                active_frames += 1
                if active_frames >= min_frames:
                    return True
                continue
            if abs(getattr(row, c_y, 0.5) - 0.5) > stick_eps:
                active_frames += 1
                if active_frames >= min_frames:
                    return True
                continue
            if getattr(row, shoulder, 0.0) > 0.0:
                active_frames += 1
                if active_frames >= min_frames:
                    return True
                continue
            if any(getattr(row, field, 0.0) > 0.0 for field in btn_fields):
                active_frames += 1
                if active_frames >= min_frames:
                    return True
        except AttributeError:
            continue
    return False


def process_one_episode(raw_path: str) -> List[Row]:
    """Convert an ``.slp`` replay into a list of schema rows with filtering steps.

    Example
    -------
    For a valid two-player match, the function iterates console frames, skips
    pre-game/invalid frames, extracts rows via :func:`extract`, and returns the
    list. Errors before any row is collected raise ``ValueError`` so the caller can
    discard the replay.
    """
    console = Console(is_dolphin=False, allow_old_version=True, path=raw_path)

    if not console.connect():
        raise ValueError(f"Failed to connect to SLP file: {raw_path}")

    rows = []

    try:
        while True:
            gamestate = console.step()

            if gamestate is None:
                break

            if len(gamestate.players) < 2:
                continue

            if gamestate.frame < 0:
                continue

            try:
                row = extract_row(gamestate)
                rows.append(row)
            except (ValueError, KeyError, AttributeError):
                continue
    except Exception as e:
        if rows:
            print(f"Error processing {raw_path} after {len(rows)} frames: {e}")
        else:
            raise ValueError(f"Failed to process {raw_path}: {e}")

    if not rows:
        raise ValueError(f"No valid frames found in {raw_path}")

    if not _player_active(rows, "p1_") or not _player_active(rows, "p2_"):
        raise ValueError(f"Replay {raw_path} discarded due to inactive player(s).")

    return rows


def _choose_chunk_t(num_features: int, elem_bytes: int) -> int:
    """Pick the temporal chunk size ``T`` for Zarr arrays given ``num_features`` features.

    Example
    -------
    With ``num_features=512`` and ``elem_bytes=4`` the helper approximates how many frames fit
    in ``config.zarr.target_chunk_mb`` megabytes, then rounds to a multiple of
    ``config.seq_len`` so sliding windows rarely straddle chunk boundaries.
    """
    config = get_config()
    chunk_frames = config.zarr.chunk_frames
    if chunk_frames and chunk_frames > 0:
        approx_t = int(chunk_frames)
    else:
        approx_t = int(
            (config.zarr.target_chunk_mb * (1024**2)) / (num_features * elem_bytes)
        )
    approx_t = max(config.seq_len, approx_t)
    # align to a multiple of seq len to minimize boundary splits
    if config.seq_len > 0:
        approx_t = (approx_t // config.seq_len) * config.seq_len or config.seq_len
    return approx_t


class EpisodeWriter:
    def __init__(self, schema: Schema, shard_path: str) -> None:
        """Initialize a shard writer that buffers data in a temporary directory.

        Example
        -------
        ``EpisodeWriter(schema, 'shard_00000.zarr')`` creates a temporary directory
        ``shard_00000.zarr.tmp``. Calls to :meth:`write_episode` populate this temp
        store until :meth:`finalize` atomically renames it into place.
        """
        self.schema = schema
        self.shard_path = Path(shard_path)
        # write rows to a temporary dir then rename atomically on finalize
        self.tmp_path = self.shard_path.with_suffix(".zarr.tmp")
        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path)
        self.root = zarr.open_group(
            str(self.tmp_path), mode="a"
        )  # FS-backed directory store

        self._chunk_t_cache: dict[int, int] = {}

    def _chunk_t(self, num_features: int, elem_bytes: int = 4) -> int:
        """Memoize the chunk length for feature dimension ``F``.

        Example
        -------
        The first call with ``F=512`` computes the chunk length via
        :func:`_choose_chunk_t` and caches it. Subsequent calls with the same ``F``
        reuse the cached value, avoiding repeated configuration math.
        """
        if num_features not in self._chunk_t_cache:
            ct = _choose_chunk_t(
                num_features=num_features,
                elem_bytes=elem_bytes,
            )
            self._chunk_t_cache[num_features] = ct
        return self._chunk_t_cache[num_features]

    def write_episode(
        self, episode_id: int, features: RawNumpyArray, targets: RawNumpyArray
    ) -> str:
        """Write ``features``/``targets`` arrays for ``episode_id`` into the shard.

        Example
        -------
        Given ``features`` with shape ``(300, num_features)`` and ``targets`` with ``(300, Yd)``, the method
        creates ``ep_000123/X`` and ``ep_000123/Y`` arrays (chunked along time),
        fills them with the provided data, and returns the episode group name.

        Arrays are stored in float16 format to reduce storage and transfer bandwidth by 50%.
        """
        config = get_config()
        assert features.dtype == np.float32 and targets.dtype == np.float32
        ep_name = f"ep_{episode_id:06d}"
        epg = self.root.require_group(ep_name)
        for name in ("X", "Y"):
            if name in epg:
                del epg[name]
        # Use elem_bytes=2 for fp16 chunk size calculation
        chunk_t = self._chunk_t(features.shape[1], elem_bytes=2)
        features_array = epg.create_array(
            "X",
            shape=features.shape,
            chunks=(min(chunk_t, features.shape[0]), features.shape[1]),
            compressors=[config.zarr.compressor],
            dtype="float16",
            overwrite=True,
        )
        features_array[:] = features.astype(np.float16)
        targets_array = epg.create_array(
            "Y",
            shape=targets.shape,
            chunks=(min(chunk_t, targets.shape[0]), targets.shape[1]),
            compressors=[config.zarr.compressor],
            dtype="float16",
            overwrite=True,
        )
        targets_array[:] = targets.astype(np.float16)
        return ep_name

    def finalize(self) -> None:
        """Commit the temporary shard directory by renaming it into place.

        Example
        -------
        After all episodes are written, :meth:`finalize` removes any existing shard
        directory and renames ``shard_00000.zarr.tmp`` to ``shard_00000.zarr`` so
        downstream readers see a consistent snapshot.
        """
        self.tmp_path.flush() if hasattr(self.tmp_path, "flush") else None
        if self.shard_path.exists():
            shutil.rmtree(self.shard_path)
        os.replace(self.tmp_path, self.shard_path)


@dataclass
class ShardResult:
    shard_id: int
    episode_ids: List[int]
    frames: List[int]
    feat_dtypes: List[str]
    targ_dtypes: List[str]
    feature_names: List[str]
    target_names: List[str]


def _rows_to_dense(
    rows: Sequence[object], schema: Schema
) -> Tuple[RawNumpyArray, RawNumpyArray, List[str], List[str], List[str], List[str]]:
    """Convert a list of :class:`Row` objects into feature/target matrices.

    Example
    -------
    For three rows ``r0, r1, r2`` the function builds ``X`` from ``r0`` and ``r1``
    while ``Y`` uses ``r1`` and ``r2`` (one-step lookahead). It returns
    ``float32`` arrays alongside the feature/target names and dtypes, matching the
    tensors written into the final Zarr shards.
    """
    num_frames = len(rows)
    if num_frames < 2:
        raise ValueError(
            f"Need at least 2 frames for temporal shifting, got {num_frames}"
        )
    base_feature_names = [
        name for name in schema.features if name not in DERIVED_FEATURES
    ]
    base_target_names = list(schema.targets)
    num_features = len(base_feature_names)
    num_targets = len(base_target_names)

    feat_dtypes: List[str] = []
    targ_dtypes: List[str] = []

    row_values = np.array([_ROW_ATTR_GETTER(row) for row in rows], dtype=object)
    feature_indices = [_ROW_FIELD_INDEX[name] for name in base_feature_names]
    target_indices = [_ROW_FIELD_INDEX[name] for name in base_target_names]

    def _dtype_label(value: object) -> str:
        if isinstance(value, (bool, np.bool_)):
            return "float32"
        return "int32" if isinstance(value, (int, np.integer)) else "float32"

    feat_dtypes.extend(_dtype_label(row_values[0, idx]) for idx in feature_indices)
    targ_dtypes.extend(_dtype_label(row_values[0, idx]) for idx in target_indices)

    X = np.asarray(row_values[:-1, feature_indices], dtype=np.float32)
    Y = np.asarray(row_values[1:, target_indices], dtype=np.float32)

    feature_names_out = list(base_feature_names)
    target_names_out = list(base_target_names)

    return X, Y, feat_dtypes, targ_dtypes, feature_names_out, target_names_out


def _rows_to_episode(rows: Sequence[Row], schema: Schema, config) -> ProcessedEpisode:
    """Convert raw :class:`Row` values into processed feature/target arrays."""
    (
        X,
        Y_raw,
        feat_dtypes,
        _targ_dtypes_raw,
        feature_names,
        target_names,
    ) = _rows_to_dense(rows, schema)

    # Apply feature transforms (quantization/scaling) in-place
    X = np.ascontiguousarray(X)
    X = apply_feature_transforms(X, feature_names)
    X = np.ascontiguousarray(X.astype(np.float32, copy=False))

    # Quantize targets to indices
    torch_Y = torch.from_numpy(Y_raw).unsqueeze(0)  # [1, T, Yd]
    raw_colmap = _RawTargetColumnMap(target_names)
    target_info = quantize_targets(torch_Y, raw_colmap, input_domain="unit01")
    main_idx = target_info["main_idx"].squeeze(0).cpu().numpy().astype(np.int32)
    c_idx = target_info["c_idx"].squeeze(0).cpu().numpy().astype(np.int32)
    shoulder_idx = target_info["shoulder_idx"].squeeze(0).cpu().numpy().astype(np.int32)
    buttons = target_info["buttons"].squeeze(0).cpu().numpy().astype(np.float32)

    # Build final target array
    Y = np.concatenate(
        [
            main_idx.reshape(-1, 1),
            c_idx.reshape(-1, 1),
            shoulder_idx.reshape(-1, 1),
            buttons,
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    targ_dtypes = ["float32"] * Y.shape[1]

    # Build target names
    base_target_names = [
        "p1_main_stick_idx",
        "p1_c_stick_idx",
        "p1_shoulder_idx",
        "p1_button_a",
        "p1_button_b",
        "p1_button_xy",
        "p1_button_z",
        "p1_button_lr",
    ]

    target_names = base_target_names

    derived_features = [name for name in schema.features if name in DERIVED_FEATURES]
    if derived_features:
        torch_X = torch.from_numpy(X).unsqueeze(0)  # [1, T, F]
        colmap = ColumnMap(feature_names, target_names)
        reward_features = build_reward_feature_index(colmap)

        for name in derived_features:
            if name != "value_target":
                raise ValueError(f"Unsupported derived feature '{name}'.")
            value_targets = (
                compute_value_targets(
                    torch_X,
                    colmap,
                    reward_cfg=config.reward,
                    reward_idx=None,
                    reward_features=reward_features,
                )
                .squeeze(0)
                .squeeze(-1)
            )
            value_column = value_targets.cpu().numpy().astype(np.float32, copy=False)
            X = np.concatenate(
                [X, value_column.reshape(value_column.shape[0], 1)], axis=1
            )
            feat_dtypes.append("float32")
            feature_names.append(name)

    return ProcessedEpisode(
        features=X,
        targets=Y,
        feat_dtypes=feat_dtypes,
        targ_dtypes=targ_dtypes,
        feature_names=feature_names,
        target_names=target_names,
    )


def _process_episode_task(
    raw_path: str,
    schema: Schema,
) -> List[ProcessedEpisode]:
    """Process a single episode path inside the multiprocessing pool.

    Example
    -------
    The worker calls :func:`process_one_episode` followed by :func:`_rows_to_episode`
    for both the original and flipped player perspectives, returning processed
    arrays and metadata for each variant.
    """
    _ensure_config_initialized()
    config = get_config()
    rows = process_one_episode(raw_path)
    flipped_rows = _flip_rows(rows)
    return [
        _rows_to_episode(rows, schema, config),
        _rows_to_episode(flipped_rows, schema, config),
    ]


def _merge_and_write_metadata(
    results: List[ShardResult],
    feature_names: Sequence[str],
    target_names: Sequence[str],
    out_root: str,
) -> None:
    """Write index files, lengths, and ``meta.json`` for the built dataset.

    Example
    -------
    Given shard results for two shards, the helper writes ``index.jsonl`` entries
    referencing each episode/shard pair, saves ``lengths.npy`` and
    ``wins_per_ep.npy``, and serializes ``meta.json`` with schema names and dtypes,
    mirroring the artifacts consumed by :class:`ZarrCorpusIndex`.
    """
    config = get_config()
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results:
        raise RuntimeError("No shard results available to write metadata.")

    # Index
    idx_path = out_dir / "index.jsonl"
    with idx_path.open("w") as f:
        for r in results:
            for ep_id, T in zip(r.episode_ids, r.frames):
                f.write(
                    json.dumps(
                        {"episode_id": ep_id, "shard_id": r.shard_id, "frames": int(T)}
                    )
                    + "\n"
                )

    # Order by episode_id
    all_eps: List[Tuple[int, int]] = []
    for r in results:
        all_eps.extend(zip(r.episode_ids, r.frames))
    all_eps.sort(key=lambda x: x[0])

    lengths = np.array([T for _, T in all_eps], dtype=np.int32)
    wins_per_ep = np.clip(lengths - config.seq_len + 1, a_min=0, a_max=None).astype(
        np.int32
    )
    np.save(out_dir / "lengths.npy", lengths)
    np.save(out_dir / "wins_per_ep.npy", wins_per_ep)

    # Precompute every global window's (episode_index, local_offset) for O(1) lookups.
    total_windows = int(wins_per_ep.sum())
    window_index = np.empty((total_windows, 2), dtype=np.int32)
    cursor = 0
    for ep_idx, num_windows in enumerate(wins_per_ep.tolist()):
        if num_windows <= 0:
            continue
        next_cursor = cursor + num_windows
        window_index[cursor:next_cursor, 0] = ep_idx
        window_index[cursor:next_cursor, 1] = np.arange(num_windows, dtype=np.int32)
        cursor = next_cursor
    np.save(out_dir / "window_index.npy", window_index)

    feat_dtypes = results[0].feat_dtypes
    targ_dtypes = results[0].targ_dtypes

    # Metadata with versioning
    meta = {
        "version": 1,
        "created_at_unix": int(time.time()),
        "build_config": config.model_dump(mode="json"),
        "schema": {"features": list(feature_names), "targets": list(target_names)},
        "preprocessed": {
            "features": True,
            "targets": True,
            "input_domain": "unit01",
            "target_layout": {
                "main": "p1_main_stick_idx",
                "c": "p1_c_stick_idx",
                "shoulder": "p1_shoulder_idx",
                "buttons": [
                    "p1_button_a",
                    "p1_button_b",
                    "p1_button_xy",
                    "p1_button_z",
                    "p1_button_lr",
                ],
            },
            "palette_sizes": {
                "main": len(target_names),  # placeholder; overridden below
            },
        },
        "feat_dtypes": feat_dtypes,
        "targ_dtypes": targ_dtypes,
    }

    # Palette sizes: recover from controller_quantization constants
    from constants import (
        _MAIN_STICK_PALETTE_CPU,
        _C_STICK_PALETTE_CPU,
        SHOULDER_QUANTIZED,
        CONTROLLER_KEY_GROUPS,
    )

    from constants import CONTROLLER_KEY_GROUPS

    meta["preprocessed"]["palette_sizes"] = {
        "main": int(_MAIN_STICK_PALETTE_CPU.shape[0]),
        "c": int(_C_STICK_PALETTE_CPU.shape[0]),
        "shoulder": int(len(SHOULDER_QUANTIZED)),
        "buttons": len(CONTROLLER_KEY_GROUPS["buttons"]),
    }

    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


def build_dataset(
    raw_episode_paths: Sequence[str], schema: Schema, out_root: str
) -> None:
    """Parallelize replay processing and assemble Zarr shards with metadata.

    Example
    -------
    When ``raw_episode_paths`` lists 10 files and the shard size is ``4`` episodes,
    the function spins up workers via :func:`_process_episode_task`, streams
    completed episodes (original + flipped) into :class:`EpisodeWriter` instances
    per shard, and finally writes ``index.jsonl``, ``lengths.npy``, and
    ``meta.json`` through :func:`_merge_and_write_metadata`.
    """
    config = get_config()
    N = len(raw_episode_paths)
    if N == 0:
        raise ValueError("No raw episodes provided.")

    replays_per_shard = max(1, config.zarr.shard_size // EPISODES_PER_REPLAY)
    num_shards = math.ceil(N / replays_per_shard)
    shards: List[List[str]] = []
    for s in range(num_shards):
        start = s * replays_per_shard
        end = min((s + 1) * replays_per_shard, N)
        shards.append(list(raw_episode_paths[start:end]))

    Path(out_root).mkdir(parents=True, exist_ok=True)

    max_workers = min(N, max(1, os.cpu_count() or 1))
    # For profiling, set ZARR_USE_THREADS=1 to avoid multiprocessing pickling issues
    use_threads = os.environ.get("ZARR_USE_THREADS", "").lower() in ("1", "true", "yes")
    if use_threads:
        max_workers = (
            1  # ThreadPoolExecutor doesn't benefit from many workers due to GIL
        )

    writers: Dict[int, EpisodeWriter] = {}
    shard_episode_entries: Dict[int, List[Tuple[int, int]]] = {
        i: [] for i in range(num_shards)
    }
    shard_feat_dtypes: Dict[int, List[str] | None] = {
        i: None for i in range(num_shards)
    }
    shard_targ_dtypes: Dict[int, List[str] | None] = {
        i: None for i in range(num_shards)
    }
    shard_feature_names: Dict[int, List[str] | None] = {
        i: None for i in range(num_shards)
    }
    shard_target_names: Dict[int, List[str] | None] = {
        i: None for i in range(num_shards)
    }
    shard_expected_episodes: Dict[int, int] = {
        i: len(shards[i]) * EPISODES_PER_REPLAY for i in range(num_shards)
    }
    shard_episode_base: Dict[int, int] = {}
    cumulative_episodes = 0
    for i in range(num_shards):
        shard_episode_base[i] = cumulative_episodes
        cumulative_episodes += shard_expected_episodes[i]

    final_feature_names: List[str] | None = None
    final_target_names: List[str] | None = None

    MAX_IN_FLIGHT = max(1, max_workers * 2)

    def _job_iter():
        for shard_idx, shard_paths in enumerate(shards):
            for local_idx, raw_path in enumerate(shard_paths):
                yield shard_idx, local_idx, raw_path

    jobs = _job_iter()
    futures: Dict[Future, Tuple[int, int]] = {}

    ExecutorClass = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    with ExecutorClass(max_workers=max_workers) as executor:

        def submit_next() -> bool:
            try:
                sidx, lidx, rpath = next(jobs)
            except StopIteration:
                return False
            fut = executor.submit(_process_episode_task, rpath, schema)
            futures[fut] = (sidx, lidx)
            return True

        # Prime the queue with a bounded number of tasks
        for _ in range(min(MAX_IN_FLIGHT, N)):
            if not submit_next():
                break

        with tqdm.tqdm(total=N, desc="Processing replays", unit="replay") as progress:
            while futures:
                done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    shard_idx, local_idx = futures.pop(future)
                    try:
                        episode_batch = future.result()
                    except Exception as exc:  # pragma: no cover
                        raise RuntimeError(
                            f"Episode processing failed for shard {shard_idx}, index {local_idx}: {exc}"
                        ) from exc

                    progress.update(1)

                    if not episode_batch:
                        raise RuntimeError(
                            f"No episodes produced for shard {shard_idx}, index {local_idx}"
                        )
                    anchor_episode = episode_batch[0]
                    if final_feature_names is None:
                        final_feature_names = list(anchor_episode.feature_names)
                    if final_target_names is None:
                        final_target_names = list(anchor_episode.target_names)

                    writer = writers.get(shard_idx)
                    if writer is None:
                        shard_path = Path(out_root) / f"shard_{shard_idx:05d}.zarr"
                        writer = EpisodeWriter(schema, str(shard_path))
                        writers[shard_idx] = writer

                    for episode in episode_batch:
                        episode_id = shard_episode_base[shard_idx] + len(
                            shard_episode_entries[shard_idx]
                        )
                        writer.write_episode(
                            episode_id, episode.features, episode.targets
                        )

                        if shard_feat_dtypes[shard_idx] is None:
                            shard_feat_dtypes[shard_idx] = episode.feat_dtypes
                            shard_targ_dtypes[shard_idx] = episode.targ_dtypes
                        if shard_feature_names[shard_idx] is None:
                            shard_feature_names[shard_idx] = list(episode.feature_names)
                        if shard_target_names[shard_idx] is None:
                            shard_target_names[shard_idx] = list(episode.target_names)

                        shard_episode_entries[shard_idx].append(
                            (episode_id, episode.features.shape[0])
                        )

                    # (B) Finalize this shard as soon as all its episodes are written
                    if (
                        len(shard_episode_entries[shard_idx])
                        == shard_expected_episodes[shard_idx]
                    ):
                        writer.finalize()
                        # Try to close underlying store to free resources
                        try:
                            store = writer.root.store
                            if hasattr(store, "close"):
                                store.close()
                        except Exception:
                            pass
                        # Remove from writers so it can be GC'd
                        if shard_idx in writers:
                            del writers[shard_idx]

                # Top up the in-flight queue
                while len(futures) < MAX_IN_FLIGHT and submit_next():
                    pass

    results: List[ShardResult] = []
    for shard_idx in range(num_shards):
        entries = shard_episode_entries[shard_idx]
        if not entries:
            continue

        entries.sort(key=lambda item: item[0])
        writer = writers.get(shard_idx)
        if writer is not None:
            writer.finalize()
            try:
                store = writer.root.store
                if hasattr(store, "close"):
                    store.close()
            except Exception:
                pass

        episode_ids = [ep for ep, _ in entries]
        frames = [frames for _, frames in entries]

        results.append(
            ShardResult(
                shard_id=shard_idx,
                episode_ids=episode_ids,
                frames=frames,
                feat_dtypes=shard_feat_dtypes[shard_idx] or [],
                targ_dtypes=shard_targ_dtypes[shard_idx] or [],
                feature_names=shard_feature_names[shard_idx] or [],
                target_names=shard_target_names[shard_idx] or [],
            )
        )

    results.sort(key=lambda r: r.shard_id)
    feature_names_out = final_feature_names or (
        results[0].feature_names if results else []
    )
    target_names_out = final_target_names or (
        results[0].target_names if results else []
    )
    _merge_and_write_metadata(results, feature_names_out, target_names_out, out_root)


def create_melee_schema() -> Schema:
    """Return a :class:`Schema` populated with default feature/target names.

    Example
    -------
    Calling this helper wraps :func:`get_feature_names` and
    :func:`get_raw_target_names`, producing a schema object ready for
    :func:`build_dataset`.
    """
    return Schema(
        features=get_feature_names(),
        targets=get_raw_target_names(),
    )


def main():
    """Entry point that builds validation and training datasets from ``.slp`` files.

    Example
    -------
    Running ``python zarr_storage.py`` discovers replay files under the configured
    input root, builds validation then training shards via :func:`build_dataset`,
    and prints summary statistics such as average frames per episode.
    """
    import glob
    import multiprocessing

    # Set multiprocessing start method to 'fork' for better pickling support on macOS
    try:
        multiprocessing.set_start_method("fork", force=False)
    except RuntimeError:
        pass  # Already set

    init_config()
    config = get_config()

    train_slp_files = sorted(
        glob.glob(os.path.join(config.zarr.input_root, "master-*.slp"))
    )[: config.zarr.episode_count]
    validation_slp_files = sorted(
        glob.glob(os.path.join(config.zarr.input_root, "master-*.slp"))
    )[
        config.zarr.episode_count : config.zarr.episode_count
        + config.zarr.validation_count
    ]

    if not train_slp_files:
        print(f"No .slp files found in {config.zarr.input_root}")
        return

    if not validation_slp_files:
        print(f"No .slp files found in {config.zarr.input_root}")
        return

    print(f"Found {len(train_slp_files)} .slp files in {config.zarr.input_root}")
    print(
        f"Found {len(validation_slp_files)} validation .slp files in {config.zarr.input_root}"
    )

    schema = create_melee_schema()

    print(f"Schema: {len(schema.features)} features, {len(schema.targets)} targets")
    print(f"Output directory: {config.zarr.out_root}")
    print(f"Validation directory: {config.zarr.validation_root}")
    print(
        f"Configuration: seq_len={config.seq_len}, shard_size={config.zarr.shard_size}"
    )

    try:
        build_dataset(validation_slp_files, schema, config.zarr.validation_root)
        print(f"Dataset built successfully in {config.zarr.validation_root}")

        # Print some statistics
        lengths = np.load(os.path.join(config.zarr.validation_root, "lengths.npy"))
        print(f"Total episodes: {len(lengths)}")
        print(f"Total frames: {lengths.sum()}")
        print(f"Average frames per episode: {lengths.mean():.1f}")
        print(f"Frame range: {lengths.min()} - {lengths.max()}")

    except Exception as e:
        print(f"Error building dataset: {e}")
        raise

    try:
        build_dataset(train_slp_files, schema, config.zarr.out_root)
        print(f"Dataset built successfully in {config.zarr.out_root}")

        # Print some statistics
        lengths = np.load(os.path.join(config.zarr.out_root, "lengths.npy"))
        print(f"Total episodes: {len(lengths)}")
        print(f"Total frames: {lengths.sum()}")
        print(f"Average frames per episode: {lengths.mean():.1f}")
        print(f"Frame range: {lengths.min()} - {lengths.max()}")

    except Exception as e:
        print(f"Error building dataset: {e}")
        raise


if __name__ == "__main__":
    main()
