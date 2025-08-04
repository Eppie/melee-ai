from __future__ import annotations

from typing import TypedDict
import pyarrow.compute as pc

import concurrent.futures
import hashlib
import os
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import numpy as np
import peppi_py
import pyarrow as pa
import pyarrow.types as pat
from pyarrow import parquet as pq

REPLAYS_DIR = Path("/Users/eppie/Downloads/replays_sorted/FOX_vs_FOX")
OUT_DIR = Path("shards")
OUT_DIR.mkdir(exist_ok=True)
COMBINED_OUT_PATH = OUT_DIR / "all_frames.parquet"

CHUNK_SIZE = 1000  # number of replays per output Parquet file


class Button(Enum):
    """A single button on a GCN controller

    Note:
        String values represent the Dolphin input string for that button"""
    BUTTON_A = "A"
    BUTTON_B = "B"
    BUTTON_X = "X"
    BUTTON_Y = "Y"
    BUTTON_Z = "Z"
    BUTTON_L = "L"
    BUTTON_R = "R"  # Control sticks considered "buttons" here
    BUTTON_MAIN = "MAIN"
    BUTTON_C = "C"


class Buttons(NamedTuple):
    A: np.bool_
    B: np.bool_
    X: np.bool_
    Y: np.bool_
    Z: np.bool_
    L: np.bool_
    R: np.bool_


LIBMELEE_BUTTONS = {name: Button(name) for name in Buttons._fields}

BUTTON_MASKS = {
    Button.BUTTON_A: 0x0100,
    Button.BUTTON_B: 0x0200,
    Button.BUTTON_X: 0x0400,
    Button.BUTTON_Y: 0x0800,
    Button.BUTTON_Z: 0x0010,
    Button.BUTTON_R: 0x0020,
    Button.BUTTON_L: 0x0040,
}


def get_buttons(button_bits: np.ndarray) -> Buttons:
    return Buttons(**{
        name: np.asarray(
            np.bitwise_and(button_bits, BUTTON_MASKS[button]),
            dtype=bool)
        for name, button in LIBMELEE_BUTTONS.items()
    })


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA‑256 digest of *path* as a hexadecimal string."""
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


class FlagBits(TypedDict):
    idx: int  # index within the 5-tuple of state-flag arrays
    mask: int  # bit-mask to test


# ────────────────────────────────────────────────────────────────────────────────
# Bit-definitions we care about ─ indices are *within* the state_flags tuple.
# The first element (index 0) is “State Bit Flags 1”, which we ignore.
# ────────────────────────────────────────────────────────────────────────────────
FLAG_DEFS: dict[str, FlagBits] = {
    # State Bit Flags 2  (state_flags[1])
    "has_temp_intang": {"idx": 1, "mask": 0x04},
    "is_fastfalling": {"idx": 1, "mask": 0x08},
    "defender_hitlag": {"idx": 1, "mask": 0x10},
    "in_hitlag": {"idx": 1, "mask": 0x20},

    # State Bit Flags 3  (state_flags[2])
    "is_grabbing": {"idx": 2, "mask": 0x04},
    "shield_active": {"idx": 2, "mask": 0x80},

    # State Bit Flags 4  (state_flags[3])
    "in_hitstun": {"idx": 3, "mask": 0x02},
    "shield_touch": {"idx": 3, "mask": 0x04},
    "powershield": {"idx": 3, "mask": 0x20},

    # State Bit Flags 5  (state_flags[4])
    "is_dead": {"idx": 4, "mask": 0x40},
    "offscreen": {"idx": 4, "mask": 0x80},
}


def _bit_is_set(arr: pa.UInt8Array, mask: int) -> pa.BooleanArray:
    """Return a BooleanArray with `True` where *mask* is asserted in *arr*."""
    return pc.not_equal(
        pc.bit_wise_and(arr, pa.scalar(mask, pa.uint8())),
        pa.scalar(0, pa.uint8()),
    )


def extract_state_flags(
        state_flags: tuple[pa.UInt8Array, pa.UInt8Array, pa.UInt8Array,
        pa.UInt8Array, pa.UInt8Array] | None,
) -> dict[str, pa.BooleanArray]:
    """
    Convert Slippi's packed `state_flags` into named BooleanArrays.

    If `state_flags` is None, this function returns an empty dict so callers can safely skip these columns.

    Returns
    -------
    Dict[str, pyarrow.BooleanArray]
        Keys are the human-readable field names defined in ``FLAG_DEFS``.
        Every returned BooleanArray has the same length as the original
        flag arrays (i.e. one element per frame).
    """
    if state_flags is None:
        # Gracefully handle missing state_flags by skipping derived columns.
        # Returning an empty dict allows callers to proceed without adding
        # these columns, and downstream concatenation with promote=True will
        # fill them as nulls where absent.
        return {}

    out: dict[str, pa.BooleanArray] = {}
    for human_name, spec in FLAG_DEFS.items():
        out[human_name] = _bit_is_set(state_flags[spec["idx"]], spec["mask"])

    return out


def process_slp(path: str) -> pa.Table | None:
    # print(f"Processing {path}")
    """Read one .slp file, return an Arrow Table with an added file_hash column."""
    tbl = table_from_slp(path)
    if tbl is None:
        return None

    digest = file_sha256(Path(path))
    hash_col = pa.array([digest] * tbl.num_rows, type=pa.string())
    return tbl.append_column("file_hash", hash_col)


def _table_to_numpy_matrix(table: pa.Table) -> tuple[np.ndarray, list[str]]:
    """Convert a pyarrow.Table to a dense float32 NumPy matrix and column names.

    Keeps only numeric/boolean columns. Booleans are cast to 0/1. Integers and
    floats are cast to float32. Non-numeric (e.g., strings) are dropped.
    Returns (matrix, column_names). Shape is [num_rows, num_selected_cols].
    """
    cols: list[np.ndarray] = []
    names: list[str] = []
    for name, chunked in zip(table.column_names, table.columns):
        typ = chunked.type
        if pat.is_boolean(typ):
            arr = chunked.combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        elif pat.is_integer(typ) or pat.is_floating(typ):
            arr = chunked.combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
        else:
            # Drop non-numeric columns (e.g., strings like file_hash, source_file)
            continue
        cols.append(arr)
        names.append(name)

    if not cols:
        # No numeric columns; return an empty (N, 0) matrix
        return np.empty((table.num_rows, 0), dtype=np.float32), []

    # Column-stack into [N, K]
    mat = np.column_stack(cols)
    return mat, names


def process_chunk_npy(paths: list[str], chunk_idx: int) -> tuple[int, str]:
    """Process a list of .slp paths and write one NumPy .npy shard (and columns).

    Writes:
      - shards/frames_chunk_{idx:03d}.npy          # float32 matrix [N, K]
      - shards/frames_chunk_{idx:03d}.columns.npy  # UTF-8 column names [K]
    Returns (rows_written, matrix_path).
    """
    tables: list[pa.Table] = []
    for p in paths:
        tbl = process_slp(p)
        if tbl is not None:
            tables.append(tbl)

    data_path = OUT_DIR / f"frames_chunk_{chunk_idx:03d}.npy"
    cols_path = OUT_DIR / f"frames_chunk_{chunk_idx:03d}.columns.npy"

    if not tables:
        # Create empty placeholders for consistency
        np.save(data_path, np.empty((0, 0), dtype=np.float32))
        np.save(cols_path, np.array([], dtype=np.str_))
        return 0, str(data_path)

    combined = pa.concat_tables(tables, promote_options="default")
    mat, names = _table_to_numpy_matrix(combined)

    # Save matrix and column names (unicode array, no pickle required)
    np.save(data_path, mat)
    np.save(cols_path, np.array(names, dtype=np.str_))
    return int(mat.shape[0]), str(data_path)



def table_from_slp(path: str) -> pa.Table | None:
    """
    Turn one .slp file into a flattened pyarrow.Table.
    """
    game = peppi_py.read_slippi(path)

    frames = game.frames
    port_data = frames.ports
    p1_pre, p1_post = port_data[0].leader.pre, port_data[0].leader.post
    p2_pre, p2_post = port_data[1].leader.pre, port_data[1].leader.post
    frame_count = len(p1_pre.random_seed)

    cols: dict[str, pa.Array] = {"frame_id": frames.id}

    def add(col: str, arr) -> None:
        cols[col] = (
            pa.nulls(frame_count) if arr is None
            else arr if isinstance(arr, pa.Array)
            else pa.array(arr)
        )

    # Helper to add every per‑player column without duplicating code
    def _add_player(prefix: str, pre, post) -> None:
        # State‑flag Booleans
        for flag_name, bool_arr in extract_state_flags(post.state_flags).items():
            add(f"{prefix}_{flag_name}", bool_arr)

        # Core numeric / categorical data
        add(f"{prefix}_pos_x", pre.position.x)
        add(f"{prefix}_pos_y", pre.position.y)
        add(f"{prefix}_action_state", post.state)
        add(f"{prefix}_percent", post.percent)
        add(f"{prefix}_stocks", post.stocks)
        add(f"{prefix}_character", post.character)
        add(f"{prefix}_btn_l_analog", pre.triggers_physical.l)
        add(f"{prefix}_btn_r_analog", pre.triggers_physical.r)

        # Button bit‑flags → BooleanArrays
        for btn_name, bool_arr in zip(
                Buttons._fields,
                get_buttons(pre.buttons_physical.to_numpy(zero_copy_only=False)),
        ):
            add(f"{prefix}_btn_{btn_name.lower()}", bool_arr)

        # Stick positions
        add(f"{prefix}_pre_joystick_x", pre.joystick.x)
        add(f"{prefix}_pre_joystick_y", pre.joystick.y)
        add(f"{prefix}_pre_cstick_x", pre.cstick.x)
        add(f"{prefix}_pre_cstick_y", pre.cstick.y)

    # Add data for both players using the helper
    for _prefix, _pre, _post in (
            ("p1", p1_pre, p1_post),
            ("p2", p2_pre, p2_post),
    ):
        _add_player(_prefix, _pre, _post)

    cols["source_file"] = pa.array(
        [Path(path).name] * frame_count, type=pa.string()
    )

    return pa.Table.from_pydict(cols)


def main() -> None:
    """Convert all .slp replays into Parquet in parallel, one file per 1,000 replays."""
    slp_files = sorted(REPLAYS_DIR.rglob("*.slp"))
    if not slp_files:
        print(f"No .slp files found in {REPLAYS_DIR}")
        return

    total_chunks = (len(slp_files) + CHUNK_SIZE - 1) // CHUNK_SIZE
    cpu_count = os.cpu_count() or 4

    print(f"Processing {len(slp_files)} replays in {total_chunks} chunks "
          f"using up to {cpu_count} processes...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as exe:
        futures: list[concurrent.futures.Future[tuple[int, str]]] = []
        for chunk_idx in range(total_chunks):
            start = chunk_idx * CHUNK_SIZE
            chunk_paths = [str(p) for p in slp_files[start:start + CHUNK_SIZE]]
            futures.append(exe.submit(process_chunk_npy, chunk_paths, chunk_idx))
        for fut in concurrent.futures.as_completed(futures):
            rows_written, out_path = fut.result()
            if rows_written == 0:
                print(f"{out_path}: no data, skipping.")
            else:
                print(f"Wrote {rows_written} rows → {out_path}")

    print("Conversion complete.")


if __name__ == "__main__":
    main()
