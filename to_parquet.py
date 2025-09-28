from __future__ import annotations

import dataclasses
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union, get_args, get_origin

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from process_replays import process_one_replay
# ---- your existing imports ----
# - Row: dynamic dataclass with replay_filename/replay_hash
# - process_one_replay: returns Optional[list[Row]]
from schema import Row

# ================= Arrow schema + helpers (np dtypes + str) =================

_NUMPY_TO_ARROW: dict[type, pa.DataType] = {
    np.int32: pa.int32(),
    np.uint32: pa.uint32(),
    np.float32: pa.float32(),
}
_NUMPY_TO_NPDTYPE: dict[type, np.dtype] = {
    np.int32: np.dtype(np.int32),
    np.uint32: np.dtype(np.uint32),
    np.float32: np.dtype(np.float32),
}
_PY_TO_ARROW: dict[type, pa.DataType] = {str: pa.string()}


def _unwrap_optional(tp: Any) -> Tuple[Any, bool]:
    origin = get_origin(tp)
    if origin is Union and type(None) in get_args(tp):
        inner = next(a for a in get_args(tp) if a is not type(None))
        return inner, True
    return tp, False


def _dtype_and_arrow(tp: Any) -> tuple[Optional[np.dtype], pa.DataType, bool]:
    inner, nullable = _unwrap_optional(tp)
    if inner in _NUMPY_TO_ARROW:
        return _NUMPY_TO_NPDTYPE[inner], _NUMPY_TO_ARROW[inner], nullable
    if inner in _PY_TO_ARROW:
        return None, _PY_TO_ARROW[inner], nullable
    raise TypeError(
        f"Unsupported field type {tp!r}. "
        "Supported: np.int32, np.uint32, np.float32, str, and Optional[...]"
    )


def _arrow_schema_from_row_cls(RowCls: type) -> tuple[pa.Schema, list[tuple[str, Optional[np.dtype], bool]]]:
    names: list[str] = []
    np_dtypes: list[Optional[np.dtype]] = []
    nulls: list[bool] = []
    fields: list[pa.Field] = []
    for f in dataclasses.fields(RowCls):
        np_dt, pa_dt, nullable = _dtype_and_arrow(f.type)
        names.append(f.name)
        np_dtypes.append(np_dt)
        nulls.append(nullable)
        fields.append(pa.field(f.name, pa_dt, nullable=nullable))
    return pa.schema(fields), list(zip(names, np_dtypes, nulls))


def _rows_to_table(rows: Sequence[Any], layout: list[tuple[str, Optional[np.dtype], bool]],
                   schema: pa.Schema) -> pa.Table:
    arrays: list[pa.Array] = []
    for (name, np_dt, nullable) in layout:
        pa_type = schema.field(name).type
        if np_dt is None:
            col = [getattr(r, name) for r in rows]
            arrays.append(pa.array(col, type=pa_type))
        else:
            cast = np_dt.type
            if nullable:
                col = [None if (v := getattr(r, name)) is None else cast(v) for r in rows]
            else:
                col = [cast(getattr(r, name)) for r in rows]
            arrays.append(pa.array(col, type=pa_type))
    return pa.Table.from_arrays(arrays, schema=schema)


# =========================== Worker: one replay → parquet ===========================

def _safe_unique_path(out_dir: Path, stem: str) -> Path:
    """Create a unique parquet path for this replay stem."""
    p = out_dir / f"{stem}.parquet"
    if not p.exists():
        return p
    i = 1
    while True:
        cand = out_dir / f"{stem}__{i}.parquet"
        if not cand.exists():
            return cand
        i += 1


def _write_replay_parquet(
        replay_path: str,
        out_dir: str,
        row_cls: type,
        *,
        row_group_size: int = 250_000,
        compression: str = "zstd",
) -> tuple[str, int]:
    """
    Process a single replay file and write one parquet file.
    Returns (output_filename, num_rows). Returns ("", 0) on skip/failure.
    """
    rows = process_one_replay(replay_path)
    if not rows:
        return "", 0

    schema, layout = _arrow_schema_from_row_cls(row_cls)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    stem = Path(replay_path).stem
    out_path = _safe_unique_path(out_dir_path, stem)

    # Stream in row groups for large replays
    writer = pq.ParquetWriter(
        where=str(out_path),
        schema=schema,
        compression=compression,
        use_dictionary=["replay_filename"],  # dictionary-encode filename
        write_statistics=True,
    )
    try:
        n = len(rows)
        if n <= row_group_size:
            tbl = _rows_to_table(rows, layout, schema)
            writer.write_table(tbl)
        else:
            for i in range(0, n, row_group_size):
                tbl = _rows_to_table(rows[i: i + row_group_size], layout, schema)
                writer.write_table(tbl)
    finally:
        writer.close()

    return out_path.name, len(rows)


# =========================== Driver: parallel over folder ===========================

def build_parquet_for_folder(
        in_dir: str | Path,
        out_dir: str | Path,
        *,
        max_workers: Optional[int] = None,
        row_group_size: int = 250_000,
) -> None:
    """
    Parallelize over all .slp files in `in_dir`, writing one parquet per replay into `out_dir`.
    Shows progress with tqdm.
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    slp_paths = sorted([str(p) for p in in_dir.glob("*.slp")])

    if not slp_paths:
        print(f"No .slp files found in {in_dir}")
        return

    total_rows = 0
    written = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(
                _write_replay_parquet,
                replay_path,
                str(out_dir),
                Row,  # your row class
                row_group_size=row_group_size,
                compression="zstd",
            )
            for replay_path in slp_paths
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Replays → Parquet", unit="file"):
            try:
                name, n = fut.result()
                if n > 0:
                    written += 1
                    total_rows += n
            except Exception as e:
                # Keep going if a replay fails
                # You can replace with logging if preferred
                print(f"[warn] worker failed: {e}")

    print(f"Wrote {written} parquet files to {out_dir} (total rows: {total_rows:,}).")


if __name__ == "__main__":
    in_dir = "/Users/eppie/Downloads/ALL_REPLAYS/FOX_vs_FOX"
    out_dir = "/Users/eppie/PycharmProjects/new-melee-ai/FOX_vs_FOX_parquet"

    # Use all CPUs by default; tune row_group_size if you expect very large replays
    build_parquet_for_folder(
        in_dir=in_dir,
        out_dir=out_dir,
        max_workers=os.cpu_count(),
        row_group_size=250_000,
    )

    # DuckDB usage:
    # import duckdb
    # con = duckdb.connect()
    # df = con.sql(f"SELECT COUNT(*) AS n FROM '{out_dir}/*.parquet'").df()
    # print(df)

#
# if __name__ == "__main__":
out_parquet = "/Users/eppie/PycharmProjects/new-melee-ai/FOX_vs_FOX.parquet"
#     import duckdb
#     con = duckdb.connect()
#     df = con.sql("SELECT stage, COUNT(*) AS n FROM 'melee_rows.parquet' GROUP BY stage ORDER BY n DESC").df()
#     print(df)
