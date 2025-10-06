import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import zarr

from config import init_config, get_config
from libmelee.melee import enums
from libmelee.melee.console import Console
from libmelee.melee.controller import ControllerState
from libmelee.melee.gamestate import GameState, PlayerState
from preprocess import _preprocess_stage, _preprocess_character, _preprocess_action, _preprocess_x_y_buttons, \
    _preprocess_l_r_buttons
from schema import Row


def extract(game_state: GameState) -> Row:
    # Common fields
    stage: np.int32 = _preprocess_stage(game_state.stage)

    # Get player ports (assume first two players)
    players: list[int] = sorted(game_state.players.keys())
    if len(players) < 2:
        raise ValueError(f"Need at least 2 players, got {len(players)}")

    p1_port: int = players[0]
    p2_port: int = players[1]

    def _extract_player_data(port: int) -> dict[str, np.int32 | np.float32 | bool]:
        """Extract all player data according to PLAYER_SPEC."""
        pl: PlayerState = game_state.players[port]
        cs: ControllerState = pl.controller_state

        # Buttons (strict access)
        b = cs.button

        return {
            "character": _preprocess_character(pl.character),
            "position_x": pl.position.x,
            "position_y": pl.position.y,
            "shield_strength": pl.shield_strength,
            "percent": np.int32(pl.percent),
            "action": _preprocess_action(pl.action),
            "stock": np.int32(pl.stock),
            "jumps_left": np.int32(pl.jumps_left),
            "facing": bool(pl.facing),
            "on_ground": np.float32(bool(pl.on_ground)),
            "is_invulnerable": np.float32(bool(pl.invulnerable)),
            "main_stick_x": cs.main_stick[0],
            "main_stick_y": cs.main_stick[1],
            "c_stick_x": cs.c_stick[0],
            "c_stick_y": cs.c_stick[1],
            "shoulder_analog": cs.l_shoulder,
            "button_a": np.float32(bool(b[enums.Button.BUTTON_A])),
            "button_b": np.float32(bool(b[enums.Button.BUTTON_B])),
            "button_xy": _preprocess_x_y_buttons(
                bool(b[enums.Button.BUTTON_X]),
                bool(b[enums.Button.BUTTON_Y])
            ),
            "button_z": np.float32(bool(b[enums.Button.BUTTON_Z])),
            "button_lr": _preprocess_l_r_buttons(
                bool(b[enums.Button.BUTTON_L]),
                bool(b[enums.Button.BUTTON_R])
            ),
        }

    # Extract data for both players
    p1_data = _extract_player_data(p1_port)
    p2_data = _extract_player_data(p2_port)

    # Build the complete field dictionary matching Row schema
    row_fields = {
        # Common fields
        "stage": stage,

        # Player 1 fields (prefixed with "p1_")
        **{f"p1_{key}": value for key, value in p1_data.items()},

        # Player 2 fields (prefixed with "p2_")
        **{f"p2_{key}": value for key, value in p2_data.items()},
    }

    return Row(**row_fields)


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

    return rows


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


def _process_shard(
        shard_id: int,
        raw_paths: Sequence[str],
        schema: Schema,
) -> ShardResult:
    config = get_config()
    shard_path = Path(config.zarr.out_root) / f"shard_{shard_id:05d}.zarr"
    writer = EpisodeWriter(schema, str(shard_path))

    feat_dtypes_first: List[str] | None = None
    targ_dtypes_first: List[str] | None = None

    episode_ids: List[int] = []
    frames: List[int] = []

    for local_idx, raw_path in enumerate(raw_paths):
        episode_id = shard_id * config.zarr.shard_size + local_idx
        rows = process_one_episode(raw_path)
        X, Y, feat_dtypes, targ_dtypes = _rows_to_dense(rows, schema)

        if feat_dtypes_first is None:
            feat_dtypes_first = feat_dtypes
            targ_dtypes_first = targ_dtypes

        writer.write_episode(episode_id, X, Y)

        episode_ids.append(episode_id)
        frames.append(X.shape[0])

    writer.finalize()

    return ShardResult(
        shard_id=shard_id,
        episode_ids=episode_ids,
        frames=frames,
        feat_dtypes=feat_dtypes_first or [],
        targ_dtypes=targ_dtypes_first or [],
    )


def _merge_and_write_metadata(
        results: List[ShardResult],
        schema: Schema,
) -> None:
    config = get_config()
    out_dir = Path(config.zarr.out_root)
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


def build_dataset(raw_episode_paths: Sequence[str], schema: Schema) -> None:
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

    results: List[ShardResult] = []
    Path(config.zarr.out_root).mkdir(parents=True, exist_ok=True)
    for s in range(num_shards):
        result = _process_shard(s, shards[s], schema)
        results.append(result)

    results.sort(key=lambda r: r.shard_id)
    _merge_and_write_metadata(results, schema)


def create_melee_schema() -> Schema:
    from schema import COMMON_SPEC, PLAYER_SPEC

    features = []
    targets = []

    controller_fields = {
        "main_stick_x", "main_stick_y", "c_stick_x", "c_stick_y",
        "shoulder_analog", "button_a", "button_b", "button_xy",
        "button_z", "button_lr"
    }

    for field_name, _ in COMMON_SPEC:
        features.append(field_name)

    for player_prefix in ["p1_", "p2_"]:
        for field_name, _ in PLAYER_SPEC:
            full_name = f"{player_prefix}{field_name}"

            # All gamestate fields go to features
            # Controller inputs also go to features (current frame inputs)
            features.append(full_name)

            # Only P1 (ego) controller inputs go to targets (next frame prediction)
            if player_prefix == "p1_" and field_name in controller_fields:
                targets.append(full_name)

    return Schema(features=features, targets=targets)


def main():
    import glob
    init_config()
    config = get_config()

    slp_files = sorted(glob.glob(os.path.join(config.zarr.input_root, "*.slp")))[:10]

    if not slp_files:
        print(f"No .slp files found in {config.zarr.input_root}")
        return

    print(f"Found {len(slp_files)} .slp files in {config.zarr.input_root}")
    schema = create_melee_schema()

    print(f"Schema: {len(schema.features)} features, {len(schema.targets)} targets")
    print(f"Output directory: {config.zarr.out_root}")
    print(f"Configuration: seq_len={config.seq_len}, shard_size={config.zarr.shard_size}")

    try:
        build_dataset(slp_files, schema)
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
