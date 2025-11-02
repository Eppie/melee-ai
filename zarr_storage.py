import json
import math
import os
import shutil
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tqdm
import zarr

from config import get_config, init_config
from constants import _SHOULDER_PALETTE
from libmelee.melee.console import Console
from libmelee.melee.gamestate import GameState
from schema import Row, extract_row, get_feature_names, get_target_names


ROW_FIELDS = tuple(fields(Row))

# TODO: Rename this file
# TODO: exclude too-short and too-long replays


# TODO: this is implemented elsewhere
def _sticks01_to_unit11_np(xy01: np.ndarray) -> np.ndarray:
    """Rescale ``[0, 1]`` stick coordinates to ``[-1, 1]`` while clamping overflow.

    Example
    -------
    ``xy01 = [[0.0, 1.0], [0.75, 0.75]]`` becomes ``[[-0.7071, 0.7071], [0.5, 0.5]]``
    after scaling and unit-circle projection, matching the preprocessing used in
    dataset creation.
    """
    xy01_clipped = np.clip(xy01, 0.0, 1.0)
    xy11 = xy01_clipped * 2.0 - 1.0
    norms = np.linalg.norm(xy11, axis=1, keepdims=True)
    mask = norms > 1.0
    if np.any(mask):
        xy11[mask] /= norms[mask]
    return xy11


def _quantize_stick_block_np(
    values: np.ndarray,
    *,
    palette: np.ndarray,
    palette_norm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Snap ``values`` to the nearest stick palette entries with an example.

    Example
    -------
    With ``values=[[0.2, 0.9], [1.1, -0.4]]`` and a four-direction palette, the
    helper clamps the second row, computes squared distances to every palette
    vector, and returns the closest palette coordinates along with their indices.
    The result mirrors how controller targets are quantized during dataset build.
    """
    if values.shape[1] != 2:
        raise ValueError("Stick quantization expects two columns (x, y).")

    arr = values.astype(np.float32, copy=False)
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        xy11 = np.clip(arr, -1.0, 1.0).copy()
        norms = np.linalg.norm(xy11, axis=1, keepdims=True)
        mask = norms > 1.0
        if np.any(mask):
            xy11[mask] /= norms[mask]
    else:
        xy01 = np.clip(arr, 0.0, 1.0)
        xy11 = _sticks01_to_unit11_np(xy01.copy())

    dot = xy11 @ palette.T
    norm = np.sum(xy11**2, axis=1, keepdims=True)
    d2 = norm - 2.0 * dot + palette_norm.T
    idx = np.argmin(d2, axis=1)
    quantized = palette[idx]
    return quantized.astype(np.float32, copy=False), idx.astype(np.int32, copy=False)


def _quantize_shoulder_np(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize shoulder analog values to the predefined palette.

    Example
    -------
    ``values=[0.05, 0.9]`` yields palette values ``[0.0, 1.0]`` with indices
    ``[0, 2]`` when ``SHOULDER_QUANTIZED=[0.0, 0.5, 1.0]``, showing the nearest
    discrete shoulder levels stored alongside their indices.
    """
    if _SHOULDER_PALETTE is None:
        raise RuntimeError("Shoulder quantization requested but no palette is defined.")
    arr = values.astype(np.float32, copy=False).reshape(-1)
    diffs = np.abs(arr[:, None] - _SHOULDER_PALETTE[None, :])
    idx = np.argmin(diffs, axis=1)
    quantized = _SHOULDER_PALETTE[idx]
    return quantized.astype(np.float32, copy=False), idx.astype(np.int32, copy=False)


def extract(game_state: GameState) -> Row:
    """Extract a :class:`Row` of schema-aligned fields from ``game_state``.

    Example
    -------
    For a frame where player 1 is at ``30%`` and holding right, the resulting
    ``Row`` contains ``p1_percent=30`` and ``p1_main_stick_x≈1``. This matches the
    row objects consumed by :func:`process_one_episode`.
    """
    return extract_row(game_state)


def _row_to_winner_first(rows: List[Row]) -> List[Row]:
    """Reorder ``rows`` so the winner consistently appears as player 1.

    Example
    -------
    If the final row shows player 2 with more stocks, every row is swapped so the
    eventual winner becomes ``p1``. When stocks tie, percent is used as the
    tiebreaker, mirroring how training expects the protagonist to be indexed.
    """
    if not rows:
        return rows

    final_row = rows[-1]
    p1_stock = final_row.p1_stock
    p2_stock = final_row.p2_stock

    if p1_stock > p2_stock:
        return rows

    if p2_stock > p1_stock:
        return [_swap_row_players(row) for row in rows]

    # Stocks tied (likely timeout) – fall back to percent comparison.
    p1_percent = final_row.p1_percent
    p2_percent = final_row.p2_percent
    if p1_percent <= p2_percent:
        return rows

    return [_swap_row_players(row) for row in rows]


def _swap_row_players(row: Row) -> Row:
    """Produce a new :class:`Row` with player 1/2 fields swapped.

    Example
    -------
    Given ``Row(p1_percent=10, p2_percent=20, stage=1)`` the helper returns a row
    where ``p1_percent=20`` and ``p2_percent=10`` while ``stage`` remains ``1``.
    This is used by :func:`_row_to_winner_first` when flipping episode perspective.
    """
    swap_values: dict[str, object] = {}
    for field in ROW_FIELDS:
        name = field.name
        if name.startswith("p1_"):
            swap_values[name] = getattr(row, f"p2_{name[3:]}")
        elif name.startswith("p2_"):
            swap_values[name] = getattr(row, f"p1_{name[3:]}")
        else:
            swap_values[name] = getattr(row, name)
    return Row(**swap_values)


@dataclass(frozen=True)
class Schema:
    features: List[str]
    targets: List[str]


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
                row = extract(gamestate)
                rows.append(row)
            except (ValueError, KeyError, AttributeError):
                continue
    except Exception as e:
        if rows:
            print(f"Warning: Error processing {raw_path} after {len(rows)} frames: {e}")
        else:
            raise ValueError(f"Failed to process {raw_path}: {e}")

    if not rows:
        raise ValueError(f"No valid frames found in {raw_path}")

    return _row_to_winner_first(rows)


def _choose_chunk_t(F: int, elem_bytes: int) -> int:
    """Pick the temporal chunk size ``T`` for Zarr arrays given ``F`` features.

    Example
    -------
    With ``F=512`` and ``elem_bytes=4`` the helper approximates how many frames fit
    in ``config.zarr.target_chunk_mb`` megabytes, then rounds to a multiple of
    ``config.seq_len`` so sliding windows rarely straddle chunk boundaries.
    """
    config = get_config()
    approx_t = int((config.zarr.target_chunk_mb * (1024**2)) / (F * elem_bytes))
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

    def _chunk_t(self, F: int, elem_bytes: int = 4) -> int:
        """Memoize the chunk length for feature dimension ``F``.

        Example
        -------
        The first call with ``F=512`` computes the chunk length via
        :func:`_choose_chunk_t` and caches it. Subsequent calls with the same ``F``
        reuse the cached value, avoiding repeated configuration math.
        """
        if F not in self._chunk_t_cache:
            ct = _choose_chunk_t(
                F=F,
                elem_bytes=elem_bytes,
            )
            self._chunk_t_cache[F] = ct
        return self._chunk_t_cache[F]

    def write_episode(self, episode_id: int, X: np.ndarray, Y: np.ndarray) -> str:
        """Write ``X``/``Y`` arrays for ``episode_id`` into the shard.

        Example
        -------
        Given ``X`` with shape ``(300, F)`` and ``Y`` with ``(300, Yd)``, the method
        creates ``ep_000123/X`` and ``ep_000123/Y`` arrays (chunked along time),
        fills them with the provided data, and returns the episode group name. If
        ``Y`` is empty the feature array is still written while the target dataset
        is omitted.
        """
        config = get_config()
        assert X.dtype == np.float32 and (Y.size == 0 or Y.dtype == np.float32)
        ep_name = f"ep_{episode_id:06d}"
        epg = self.root.require_group(ep_name)
        for name in ("X", "Y"):
            if name in epg:
                del epg[name]
        chunk_t = self._chunk_t(X.shape[1], elem_bytes=4)
        arrX = epg.create_array(
            "X",
            shape=X.shape,
            chunks=(min(chunk_t, X.shape[0]), X.shape[1]),
            compressors=[config.zarr.compressor],
            dtype="float32",
            overwrite=True,
        )
        arrX[:] = X
        if Y.shape[1] > 0:
            arrY = epg.create_array(
                "Y",
                shape=Y.shape,
                chunks=(min(chunk_t, Y.shape[0]), Y.shape[1]),
                compressors=[config.zarr.compressor],
                dtype="float32",
                overwrite=True,
            )
            arrY[:] = Y
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
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str], List[str]]:
    """Convert a list of :class:`Row` objects into feature/target matrices.

    Example
    -------
    For three rows ``r0, r1, r2`` the function builds ``X`` from ``r0`` and ``r1``
    while ``Y`` uses ``r1`` and ``r2`` (one-step lookahead). It returns
    ``float32`` arrays alongside the feature/target names and dtypes, matching the
    tensors written into the final Zarr shards.
    """
    T = len(rows)
    if T < 2:
        raise ValueError(f"Need at least 2 frames for temporal shifting, got {T}")
    base_feature_names = list(schema.features)
    base_target_names = list(schema.targets)
    F = len(base_feature_names)
    Yd = len(base_target_names)

    feat_dtypes: List[str] = []
    targ_dtypes: List[str] = []

    T_out = T - 1
    X = np.empty((T_out, F), dtype=np.float32)
    Y = (
        np.empty((T_out, Yd), dtype=np.float32)
        if Yd
        else np.empty((T_out, 0), dtype=np.float32)
    )

    for name in base_feature_names:
        v0 = getattr(rows[0], name)
        feat_dtypes.append("int32" if isinstance(v0, (int, np.integer)) else "float32")
    for name in base_target_names:
        v0 = getattr(rows[0], name)
        targ_dtypes.append("int32" if isinstance(v0, (int, np.integer)) else "float32")

    for j, name in enumerate(base_feature_names):
        X[:, j] = np.fromiter(
            (getattr(rows[i], name) for i in range(T_out)),
            count=T_out,
            dtype=np.float32,
        )

    for j, name in enumerate(base_target_names):
        Y[:, j] = np.fromiter(
            (getattr(rows[i + 1], name) for i in range(T_out)),
            count=T_out,
            dtype=np.float32,
        )

    feature_names_out = list(base_feature_names)
    target_names_out = list(base_target_names)
    # TODO: Compute the quantized and preprocessed versions of everything,
    # while including the original features, while still getting the model shape/size right
    # feat_name_to_idx = {name: idx for idx, name in enumerate(base_feature_names)}
    # targ_name_to_idx = (
    #     {name: idx for idx, name in enumerate(base_target_names)} if Yd else {}
    # )
    #
    # quant_feature_blocks: List[np.ndarray] = []
    # quant_feature_names: List[str] = []
    #
    # for prefix in ("p1_", "p2_"):
    #     main_x = f"{prefix}main_stick_x"
    #     main_y = f"{prefix}main_stick_y"
    #     if main_x in feat_name_to_idx and main_y in feat_name_to_idx:
    #         idxs = [feat_name_to_idx[main_x], feat_name_to_idx[main_y]]
    #         main_quantized, _ = _quantize_stick_block_np(
    #             X[:, idxs],
    #             palette=_MAIN_STICK_PALETTE,
    #             palette_norm=_MAIN_STICK_PALETTE_NORM,
    #         )
    #         quant_feature_blocks.append(main_quantized)
    #         quant_feature_names.extend(
    #             [
    #                 f"{prefix}main_stick_quantized_x",
    #                 f"{prefix}main_stick_quantized_y",
    #             ]
    #         )
    #
    #     c_x = f"{prefix}c_stick_x"
    #     c_y = f"{prefix}c_stick_y"
    #     if c_x in feat_name_to_idx and c_y in feat_name_to_idx:
    #         idxs = [feat_name_to_idx[c_x], feat_name_to_idx[c_y]]
    #         c_quantized, _ = _quantize_stick_block_np(
    #             X[:, idxs],
    #             palette=_C_STICK_PALETTE,
    #             palette_norm=_C_STICK_PALETTE_NORM,
    #         )
    #         quant_feature_blocks.append(c_quantized)
    #         quant_feature_names.extend(
    #             [
    #                 f"{prefix}c_stick_quantized_x",
    #                 f"{prefix}c_stick_quantized_y",
    #             ]
    #         )
    #
    #     shoulder = f"{prefix}shoulder_analog"
    #     if _SHOULDER_PALETTE is not None and shoulder in feat_name_to_idx:
    #         col = feat_name_to_idx[shoulder]
    #         shoulder_quantized, _ = _quantize_shoulder_np(X[:, col])
    #         quant_feature_blocks.append(shoulder_quantized.reshape(-1, 1))
    #         quant_feature_names.append(f"{prefix}shoulder_quantized")
    #
    # if quant_feature_blocks:
    #     X = np.concatenate([X] + quant_feature_blocks, axis=1)
    #     added_cols = sum(block.shape[1] for block in quant_feature_blocks)
    #     feat_dtypes.extend(["float32"] * added_cols)
    #     feature_names_out.extend(quant_feature_names)
    #
    # quant_target_blocks: List[np.ndarray] = []
    # quant_target_names: List[str] = []
    #
    # if Yd > 0:
    #     main_x = "p1_main_stick_x"
    #     main_y = "p1_main_stick_y"
    #     if main_x in targ_name_to_idx and main_y in targ_name_to_idx:
    #         idxs = [targ_name_to_idx[main_x], targ_name_to_idx[main_y]]
    #         main_quantized, main_idx = _quantize_stick_block_np(
    #             Y[:, idxs],
    #             palette=_MAIN_STICK_PALETTE,
    #             palette_norm=_MAIN_STICK_PALETTE_NORM,
    #         )
    #         quant_target_blocks.append(main_quantized)
    #         quant_target_names.extend(
    #             [
    #                 "p1_main_stick_quantized_x",
    #                 "p1_main_stick_quantized_y",
    #             ]
    #         )
    #         quant_target_blocks.append(
    #             main_idx.astype(np.float32, copy=False).reshape(-1, 1)
    #         )
    #         quant_target_names.append("p1_main_stick_quantized_idx")
    #
    #     c_x = "p1_c_stick_x"
    #     c_y = "p1_c_stick_y"
    #     if c_x in targ_name_to_idx and c_y in targ_name_to_idx:
    #         idxs = [targ_name_to_idx[c_x], targ_name_to_idx[c_y]]
    #         c_quantized, c_idx = _quantize_stick_block_np(
    #             Y[:, idxs],
    #             palette=_C_STICK_PALETTE,
    #             palette_norm=_C_STICK_PALETTE_NORM,
    #         )
    #         quant_target_blocks.append(c_quantized)
    #         quant_target_names.extend(
    #             [
    #                 "p1_c_stick_quantized_x",
    #                 "p1_c_stick_quantized_y",
    #             ]
    #         )
    #         quant_target_blocks.append(
    #             c_idx.astype(np.float32, copy=False).reshape(-1, 1)
    #         )
    #         quant_target_names.append("p1_c_stick_quantized_idx")
    #
    #     shoulder = "p1_shoulder_analog"
    #     if _SHOULDER_PALETTE is not None and shoulder in targ_name_to_idx:
    #         col = targ_name_to_idx[shoulder]
    #         shoulder_quantized, shoulder_idx = _quantize_shoulder_np(Y[:, col])
    #         quant_target_blocks.append(shoulder_quantized.reshape(-1, 1))
    #         quant_target_names.append("p1_shoulder_quantized")
    #         quant_target_blocks.append(
    #             shoulder_idx.astype(np.float32, copy=False).reshape(-1, 1)
    #         )
    #         quant_target_names.append("p1_shoulder_quantized_idx")
    #
    # if quant_target_blocks:
    #     if Y.shape[1] == 0:
    #         Y = np.concatenate(quant_target_blocks, axis=1)
    #     else:
    #         Y = np.concatenate([Y] + quant_target_blocks, axis=1)
    #     added_cols = sum(block.shape[1] for block in quant_target_blocks)
    #     targ_dtypes.extend(["float32"] * added_cols)
    #     target_names_out.extend(quant_target_names)

    return X, Y, feat_dtypes, targ_dtypes, feature_names_out, target_names_out


def _process_episode_task(
    raw_path: str,
    schema: Schema,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str], List[str]]:
    """Process a single episode path inside the multiprocessing pool.

    Example
    -------
    The worker calls :func:`process_one_episode` followed by :func:`_rows_to_dense`
    and returns the resulting arrays and metadata, exactly as consumed by the main
    dataset builder loop.
    """
    rows = process_one_episode(raw_path)
    return _rows_to_dense(rows, schema)


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

    lengths = np.array([T for _, T in all_eps], dtype=np.int64)
    wins_per_ep = np.clip(lengths - config.seq_len + 1, a_min=0, a_max=None).astype(
        np.int64
    )
    np.save(out_dir / "lengths.npy", lengths)
    np.save(out_dir / "wins_per_ep.npy", wins_per_ep)

    feat_dtypes = results[0].feat_dtypes
    targ_dtypes = results[0].targ_dtypes

    # Metadata with versioning
    meta = {
        "version": 1,
        "created_at_unix": int(time.time()),
        "build_config": config.model_dump(mode="json"),
        "schema": {"features": list(feature_names), "targets": list(target_names)},
        "feat_dtypes": feat_dtypes,
        "targ_dtypes": targ_dtypes,
    }

    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


def build_dataset(
    raw_episode_paths: Sequence[str], schema: Schema, out_root: str
) -> None:
    """Parallelize replay processing and assemble Zarr shards with metadata.

    Example
    -------
    When ``raw_episode_paths`` lists 10 files and the shard size is ``4``, the
    function spins up workers via :func:`_process_episode_task`, streams completed
    episodes into :class:`EpisodeWriter` instances per shard, and finally writes
    ``index.jsonl``, ``lengths.npy``, and ``meta.json`` through
    :func:`_merge_and_write_metadata`.
    """
    config = get_config()
    N = len(raw_episode_paths)
    if N == 0:
        raise ValueError("No raw episodes provided.")

    num_shards = math.ceil(N / config.zarr.shard_size)
    shards: List[List[str]] = []
    for s in range(num_shards):
        start = s * config.zarr.shard_size
        end = min((s + 1) * config.zarr.shard_size, N)
        shards.append(list(raw_episode_paths[start:end]))

    Path(out_root).mkdir(parents=True, exist_ok=True)

    max_workers = min(N, max(1, os.cpu_count() or 1))
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

    final_feature_names: List[str] | None = None
    final_target_names: List[str] | None = None

    futures: Dict[Future, Tuple[int, int]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for shard_idx, shard_paths in enumerate(shards):
            for local_idx, raw_path in enumerate(shard_paths):
                future = executor.submit(_process_episode_task, raw_path, schema)
                futures[future] = (shard_idx, local_idx)

        with tqdm.tqdm(total=N, desc="Processing episodes", unit="episode") as progress:
            for future in as_completed(futures):
                shard_idx, local_idx = futures[future]
                try:
                    (
                        X,
                        Y,
                        feat_dtypes,
                        targ_dtypes,
                        feature_names,
                        target_names,
                    ) = future.result()
                except (
                    Exception
                ) as exc:  # pragma: no cover - include episode context when bubbling
                    raise RuntimeError(
                        f"Episode processing failed for shard {shard_idx}, index {local_idx}: {exc}"
                    ) from exc

                progress.update(1)

                if final_feature_names is None:
                    final_feature_names = list(feature_names)
                if final_target_names is None:
                    final_target_names = list(target_names)

                writer = writers.get(shard_idx)
                if writer is None:
                    shard_path = Path(out_root) / f"shard_{shard_idx:05d}.zarr"
                    writer = EpisodeWriter(schema, str(shard_path))
                    writers[shard_idx] = writer

                episode_id = shard_idx * config.zarr.shard_size + local_idx
                writer.write_episode(episode_id, X, Y)

                if shard_feat_dtypes[shard_idx] is None:
                    shard_feat_dtypes[shard_idx] = feat_dtypes
                    shard_targ_dtypes[shard_idx] = targ_dtypes
                if shard_feature_names[shard_idx] is None:
                    shard_feature_names[shard_idx] = list(feature_names)
                if shard_target_names[shard_idx] is None:
                    shard_target_names[shard_idx] = list(target_names)

                shard_episode_entries[shard_idx].append((episode_id, X.shape[0]))

    results: List[ShardResult] = []
    for shard_idx in range(num_shards):
        entries = shard_episode_entries[shard_idx]
        if not entries:
            continue

        entries.sort(key=lambda item: item[0])
        writer = writers.get(shard_idx)
        if writer is None:
            raise RuntimeError(
                f"Writer missing for shard {shard_idx} despite recorded entries"
            )
        writer.finalize()

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
    :func:`get_target_names`, producing a schema object ready for
    :func:`build_dataset`.
    """
    return Schema(
        features=get_feature_names(),
        targets=get_target_names(),
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

    init_config()
    config = get_config()

    train_slp_files = sorted(glob.glob(os.path.join(config.zarr.input_root, "*.slp")))[
        : config.zarr.episode_count
    ]
    validation_slp_files = sorted(
        glob.glob(os.path.join(config.zarr.input_root, "*.slp"))
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
