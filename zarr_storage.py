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

from config import init_config, get_config
from libmelee.melee.console import Console
from libmelee.melee.gamestate import GameState
from schema import Row, extract_row, get_feature_names, get_target_names

ROW_FIELDS = tuple(fields(Row))


def extract(game_state: GameState) -> Row:
    return extract_row(game_state)


def _row_to_winner_first(rows: List[Row]) -> List[Row]:
    """Ensure the winner is consistently treated as player 1."""
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
    config = get_config()
    approx_t = int((config.zarr.target_chunk_mb * (1024 ** 2)) / (F * elem_bytes))
    approx_t = max(config.seq_len, approx_t)
    # align to a multiple of seq len to minimize boundary splits
    if config.seq_len > 0:
        approx_t = (approx_t // config.seq_len) * config.seq_len or config.seq_len
    return approx_t


class EpisodeWriter:
    def __init__(self, schema: Schema, shard_path: str) -> None:
        self.schema = schema
        self.shard_path = Path(shard_path)
        # write rows to a temporary dir then rename atomically on finalize
        self.tmp_path = self.shard_path.with_suffix(".zarr.tmp")
        if self.tmp_path.exists():
            shutil.rmtree(self.tmp_path)
        self.root = zarr.open_group(str(self.tmp_path), mode="a")  # FS-backed directory store

        self._chunk_t_cache: dict[int, int] = {}

    def _chunk_t(self, F: int, elem_bytes: int = 4) -> int:
        if F not in self._chunk_t_cache:
            ct = _choose_chunk_t(
                F=F,
                elem_bytes=elem_bytes,
            )
            self._chunk_t_cache[F] = ct
        return self._chunk_t_cache[F]

    def write_episode(self, episode_id: int, X: np.ndarray, Y: np.ndarray) -> str:
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


def _rows_to_dense(rows: Sequence[object], schema: Schema) -> \
        Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    T = len(rows)
    if T < 2:
        raise ValueError(f"Need at least 2 frames for temporal shifting, got {T}")
    F = len(schema.features)
    Yd = len(schema.targets)

    feat_dtypes: List[str] = []
    targ_dtypes: List[str] = []

    T_out = T - 1
    X = np.empty((T_out, F), dtype=np.float32)
    Y = np.empty((T_out, Yd), dtype=np.float32) if Yd else np.empty((T_out, 0), dtype=np.float32)

    for name in schema.features:
        v0 = getattr(rows[0], name)
        feat_dtypes.append("int32" if isinstance(v0, (int, np.integer)) else "float32")
    for name in schema.targets:
        v0 = getattr(rows[0], name)
        targ_dtypes.append("int32" if isinstance(v0, (int, np.integer)) else "float32")

    for j, name in enumerate(schema.features):
        X[:, j] = np.fromiter((getattr(rows[i], name) for i in range(T_out)), count=T_out, dtype=np.float32)

    for j, name in enumerate(schema.targets):
        Y[:, j] = np.fromiter((getattr(rows[i + 1], name) for i in range(T_out)), count=T_out, dtype=np.float32)

    return X, Y, feat_dtypes, targ_dtypes


def _process_episode_task(raw_path: str, schema: Schema) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    rows = process_one_episode(raw_path)
    return _rows_to_dense(rows, schema)


def _merge_and_write_metadata(
        results: List[ShardResult],
        schema: Schema,
        out_root: str
) -> None:
    config = get_config()
    out_dir = Path(out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index
    idx_path = out_dir / "index.jsonl"
    with idx_path.open("w") as f:
        for r in results:
            for ep_id, T in zip(r.episode_ids, r.frames):
                f.write(json.dumps(
                    {"episode_id": ep_id, "shard_id": r.shard_id, "frames": int(T)}
                ) + "\n")

    # Order by episode_id
    all_eps: List[Tuple[int, int]] = []
    for r in results:
        all_eps.extend(zip(r.episode_ids, r.frames))
    all_eps.sort(key=lambda x: x[0])

    lengths = np.array([T for _, T in all_eps], dtype=np.int64)
    wins_per_ep = np.clip(lengths - config.seq_len + 1, a_min=0, a_max=None).astype(np.int64)
    np.save(out_dir / "lengths.npy", lengths)
    np.save(out_dir / "wins_per_ep.npy", wins_per_ep)

    feat_dtypes = results[0].feat_dtypes
    targ_dtypes = results[0].targ_dtypes

    # Metadata with versioning
    meta = {
        "version": 1,
        "created_at_unix": int(time.time()),
        "build_config": config.to_dict(),
        "schema": {"features": schema.features, "targets": schema.targets},
        "feat_dtypes": feat_dtypes,
        "targ_dtypes": targ_dtypes,
    }

    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


def build_dataset(raw_episode_paths: Sequence[str], schema: Schema, out_root: str) -> None:
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
    shard_episode_entries: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(num_shards)}
    shard_feat_dtypes: Dict[int, List[str] | None] = {i: None for i in range(num_shards)}
    shard_targ_dtypes: Dict[int, List[str] | None] = {i: None for i in range(num_shards)}

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
                    X, Y, feat_dtypes, targ_dtypes = future.result()
                except Exception as exc:  # pragma: no cover - include episode context when bubbling
                    raise RuntimeError(
                        f"Episode processing failed for shard {shard_idx}, index {local_idx}: {exc}"
                    ) from exc

                progress.update(1)

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

                shard_episode_entries[shard_idx].append((episode_id, X.shape[0]))

    results: List[ShardResult] = []
    for shard_idx in range(num_shards):
        entries = shard_episode_entries[shard_idx]
        if not entries:
            continue

        entries.sort(key=lambda item: item[0])
        writer = writers.get(shard_idx)
        if writer is None:
            raise RuntimeError(f"Writer missing for shard {shard_idx} despite recorded entries")
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
            )
        )

    results.sort(key=lambda r: r.shard_id)
    _merge_and_write_metadata(results, schema, out_root)


def create_melee_schema() -> Schema:
    return Schema(
        features=get_feature_names(),
        targets=get_target_names(),
    )


def main():
    import glob
    init_config()
    config = get_config()

    train_slp_files = sorted(glob.glob(os.path.join(config.zarr.input_root, "*.slp")))[:config.zarr.episode_count]
    validation_slp_files = sorted(glob.glob(os.path.join(config.zarr.input_root, "*.slp")))[
                           config.zarr.episode_count:config.zarr.episode_count + config.zarr.validation_count]

    if not train_slp_files:
        print(f"No .slp files found in {config.zarr.input_root}")
        return

    if not validation_slp_files:
        print(f"No .slp files found in {config.zarr.input_root}")
        return

    print(f"Found {len(train_slp_files)} .slp files in {config.zarr.input_root}")
    print(f"Found {len(validation_slp_files)} validation .slp files in {config.zarr.input_root}")

    schema = create_melee_schema()

    print(f"Schema: {len(schema.features)} features, {len(schema.targets)} targets")
    print(f"Output directory: {config.zarr.out_root}")
    print(f"Validation directory: {config.zarr.validation_root}")
    print(f"Configuration: seq_len={config.seq_len}, shard_size={config.zarr.shard_size}")

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
