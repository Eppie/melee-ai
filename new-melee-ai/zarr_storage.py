import json
import math
import os
import shutil
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

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
class BuildConfig:
    out_root: str
    seq_len: int = 256
    shard_size: int = 1000
    target_chunk_mb: float = 4.0
    compressor: BloscCodec = field(
        default_factory=lambda: BloscCodec(cname="zstd", clevel=7, shuffle=BloscShuffle.bitshuffle))
    seed: int = 42
    use_consolidated_metadata: bool = False


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


def _choose_chunk_t(F: int, elem_bytes: int, target_chunk_mb: float, min_t: int, align_to: int) -> int:
    approx_t = int((target_chunk_mb * (1024 ** 2)) / (F * elem_bytes))
    approx_t = max(min_t, approx_t)
    # align to a multiple of seq len to minimize boundary splits
    if align_to > 0:
        approx_t = (approx_t // align_to) * align_to or align_to
    return approx_t


class EpisodeWriter:
    def __init__(self, cfg: BuildConfig, schema: Schema, shard_path: str) -> None:
        self.cfg = cfg
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
                target_chunk_mb=self.cfg.target_chunk_mb,
                min_t=self.cfg.seq_len,
                align_to=self.cfg.seq_len,
            )
            self._chunk_t_cache[F] = ct
        return self._chunk_t_cache[F]

    def write_episode(self, episode_id: int, X: np.ndarray, Y: np.ndarray) -> str:
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
            compressors=[self.cfg.compressor],
            dtype="float32",
            overwrite=True,
        )
        arrX[:] = X
        if Y.shape[1] > 0:
            arrY = epg.create_array(
                "Y",
                shape=Y.shape,
                chunks=(min(chunk_t, Y.shape[0]), Y.shape[1]),
                compressors=[self.cfg.compressor],
                dtype="float32",
                overwrite=True,
            )
            arrY[:] = Y
        return ep_name

    def finalize(self) -> None:
        self.tmp_path.flush() if hasattr(self.tmp_path, "flush") else None
        if self.cfg.use_consolidated_metadata:
            zarr.consolidate_metadata(str(self.tmp_path))
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
        cfg: BuildConfig,
        schema: Schema,
) -> ShardResult:
    shard_path = Path(cfg.out_root) / f"shard_{shard_id:05d}.zarr"
    writer = EpisodeWriter(cfg, schema, str(shard_path))

    feat_dtypes_first: List[str] | None = None
    targ_dtypes_first: List[str] | None = None

    episode_ids: List[int] = []
    frames: List[int] = []

    for local_idx, raw_path in enumerate(raw_paths):
        episode_id = shard_id * cfg.shard_size + local_idx
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
        cfg: BuildConfig,
        schema: Schema,
) -> None:
    out_dir = Path(cfg.out_root)
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
    wins_per_ep = np.clip(lengths - cfg.seq_len + 1, a_min=0, a_max=None).astype(np.int64)
    np.save(out_dir / "lengths.npy", lengths)
    np.save(out_dir / "wins_per_ep.npy", wins_per_ep)

    feat_dtypes = results[0].feat_dtypes
    targ_dtypes = results[0].targ_dtypes

    # Metadata with versioning
    meta = {
        "version": 1,
        "created_at_unix": int(time.time()),
        "build_config": {**asdict(cfg), "compressor": str(cfg.compressor)},
        "schema": {"features": schema.features, "targets": schema.targets},
        "feat_dtypes": feat_dtypes,
        "targ_dtypes": targ_dtypes,
        "seq_len": int(cfg.seq_len),
    }

    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)


def build_dataset(raw_episode_paths: Sequence[str], cfg: BuildConfig, schema: Schema) -> None:
    N = len(raw_episode_paths)
    if N == 0:
        raise ValueError("No raw episodes provided.")

    num_shards = math.ceil(N / cfg.shard_size)
    shards: List[List[str]] = []
    for s in range(num_shards):
        start = s * cfg.shard_size
        end = min((s + 1) * cfg.shard_size, N)
        shards.append(list(raw_episode_paths[start:end]))

    results: List[ShardResult] = []
    Path(cfg.out_root).mkdir(parents=True, exist_ok=True)
    for s in range(num_shards):
        result = _process_shard(s, shards[s], cfg, schema)
        results.append(result)

    results.sort(key=lambda r: r.shard_id)
    _merge_and_write_metadata(results, cfg, schema)


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
    import sys
    import glob

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = "/Users/eppie/Downloads/ALL_REPLAYS/FOX_vs_FOX"

    slp_files = sorted(glob.glob(os.path.join(input_path, "master-master*.slp")))[:1]

    if not slp_files:
        print(f"No .slp files found in {input_path}")
        return

    print(f"Found {len(slp_files)} .slp files in {input_path}")

    input_name = os.path.basename(input_path.rstrip(os.sep))
    output_path = f"dataset_{input_name}"

    config = BuildConfig(
        out_root=output_path,
        seq_len=256,  # 256 frames ~ 4.3 seconds at 60fps
        shard_size=100,  # 100 episodes per shard
        target_chunk_mb=8.0,  # 8MB chunks
        seed=42,
        use_consolidated_metadata=True,
    )

    schema = create_melee_schema()

    print(f"Schema: {len(schema.features)} features, {len(schema.targets)} targets")
    print(f"Output directory: {output_path}")
    print(f"Configuration: seq_len={config.seq_len}, shard_size={config.shard_size}")

    try:
        build_dataset(slp_files, config, schema)
        print(f"Dataset built successfully in {output_path}")

        # Print some statistics
        lengths = np.load(os.path.join(output_path, "lengths.npy"))
        print(f"Total episodes: {len(lengths)}")
        print(f"Total frames: {lengths.sum()}")
        print(f"Average frames per episode: {lengths.mean():.1f}")
        print(f"Frame range: {lengths.min()} - {lengths.max()}")

    except Exception as e:
        print(f"Error building dataset: {e}")
        raise


if __name__ == "__main__":
    main()
