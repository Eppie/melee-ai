from __future__ import annotations

import enum as py_enum
import math
import multiprocessing as mp
import random
from collections import Counter, defaultdict
from dataclasses import fields as dc_fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from loguru import logger
from tdigest import TDigest
from tqdm import tqdm

import zarr
from numcodecs import Blosc, VLenUTF8

from process_replays import process_one_replay


def _is_bool(v: Any) -> bool:
    return isinstance(v, (bool, np.bool_))


def _is_enum(v: Any) -> bool:
    return isinstance(v, py_enum.Enum)


def _is_number(v: Any) -> bool:
    if _is_bool(v):
        return False
    return isinstance(v, (int, float, np.number))


def _as_float(v: Any) -> float:
    return float(v)


def _near(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def _is_shoulder(fname: str) -> bool:
    return "shoulder" in fname


def _is_stick(fname: str) -> bool:
    return "stick" in fname

# -------------------------
# Zarr helpers
# -------------------------

def _infer_zarr_dtype(name: str, sample: Any) -> tuple[np.dtype, dict[str, Any]]:
    """Return (dtype, create_kwargs) for a column based on a sample value.
    create_kwargs may include object_codec for variable-length strings.
    """
    # Numeric types
    if isinstance(sample, (np.floating, float)):
        return np.dtype("<f4"), {}
    if isinstance(sample, (np.integer, int)):
        return np.dtype("<i4"), {}
    # Enums -> int32
    if _is_enum(sample):
        return np.dtype("<i4"), {}
    # Bools -> bool
    if _is_bool(sample):
        return np.dtype("?"), {}
    # Strings -> variable-length UTF-8 with VLenUTF8 codec
    if isinstance(sample, str):
        return np.dtype(object), {"object_codec": VLenUTF8()}
    # Fallback to JSON-encoded object strings if truly unknown
    return np.dtype(object), {"object_codec": VLenUTF8()}


def _row_to_primitive(row_obj: Any, field_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for fname in field_names:
        try:
            v = getattr(row_obj, fname)
        except Exception:
            v = None
        if v is None:
            out[fname] = None
            continue
        if _is_enum(v):
            # store the underlying numeric value if present, else hash of name
            val = getattr(v, "value", None)
            out[fname] = int(val) if isinstance(val, (int, np.integer)) else int(hash(getattr(v, "name", str(v))) & 0x7FFFFFFF)
        elif _is_bool(v):
            out[fname] = bool(v)
        elif _is_number(v):
            out[fname] = _as_float(v) if isinstance(v, (float, np.floating)) else int(v)
        elif isinstance(v, str):
            out[fname] = v
        else:
            # last resort string representation
            out[fname] = repr(v)
    return out


def _batch_to_columnar(rows: list[Any], field_names: list[str]) -> dict[str, np.ndarray]:
    """Convert a list of row objects into columnar numpy arrays with best-effort dtypes."""
    if not rows:
        return {fn: np.empty((0,), dtype="<f4") for fn in field_names}
    # First pass: gather primitives for all rows
    primals: list[dict[str, Any]] = [_row_to_primitive(r, field_names) for r in rows]
    # Infer dtype per field from first non-None value
    dtypes: dict[str, tuple[np.dtype, dict[str, Any]]] = {}
    for fn in field_names:
        sample = next((p[fn] for p in primals if p[fn] is not None), None)
        if sample is None:
            # default to float32 with NaN for missing-only columns
            dtypes[fn] = (np.dtype("<f4"), {})
        else:
            dtypes[fn] = _infer_zarr_dtype(fn, sample)
    # Build arrays
    cols: dict[str, np.ndarray] = {}
    for fn in field_names:
        dt, _kwargs = dtypes[fn]
        if dt == np.dtype("?"):
            arr = np.zeros(len(primals), dtype=dt)
            for i, p in enumerate(primals):
                arr[i] = bool(p[fn]) if p[fn] is not None else False
            cols[fn] = arr
        elif dt.kind in ("i",):
            arr = np.zeros(len(primals), dtype=dt)
            for i, p in enumerate(primals):
                val = p[fn]
                arr[i] = 0 if val is None else int(val)
            cols[fn] = arr
        elif dt.kind in ("f",):
            arr = np.empty(len(primals), dtype=dt)
            for i, p in enumerate(primals):
                val = p[fn]
                arr[i] = np.nan if val is None else float(val)
            cols[fn] = arr
        else:
            # object (strings)
            arr = np.empty(len(primals), dtype=object)
            for i, p in enumerate(primals):
                arr[i] = "" if p[fn] is None else p[fn]
            cols[fn] = arr
    return cols


def _ensure_zarr_columns(root: zarr.Group, field_names: list[str], sample_row: Any, *,
                         chunks: int, compressor: Any, zarr_version: int) -> dict[str, zarr.Array]:
    """Create or open resizable 1D arrays for each field under root['columns'].
    Returns a mapping name -> Array.
    """
    cols_grp = root.require_group("columns")
    arrays: dict[str, zarr.Array] = {}

    # Infer dtype settings from a sample
    sample_map = _row_to_primitive(sample_row, field_names)
    for fn in field_names:
        dt, extra = _infer_zarr_dtype(fn, sample_map[fn])
        if fn in cols_grp:
            arrays[fn] = cols_grp[fn]
            continue
        create_kwargs: dict[str, Any] = dict(shape=(0,), maxshape=(None,), chunks=(chunks,), dtype=dt, compressor=compressor)
        create_kwargs.update(extra)
        arr = cols_grp.create_array(fn, **create_kwargs)
        # Mark the single dimension name for xarray-compatibility
        try:
            arr.attrs["_ARRAY_DIMENSIONS"] = ["row"]
        except Exception:
            pass
        arrays[fn] = arr
    return arrays


def _append_to_zarr_columns(arrays: dict[str, zarr.Array], batch: dict[str, np.ndarray]) -> tuple[int, int]:
    """Append batch columns to the Zarr arrays. Returns (start, stop) row indices written."""
    # Determine current length from any array
    any_arr = next(iter(arrays.values()))
    cur = int(any_arr.shape[0])
    new = cur + int(next(iter(batch.values())).shape[0])
    for name, arr in arrays.items():
        arr.resize(new)
        arr[cur:new] = batch[name]
    return cur, new
def save_processed_replays_to_zarr(
    out_path: str | Path,
    *,
    base_dir: str | Path = "/Users/eppie/Downloads/replays_sorted",
    replays_per_subdir: int = 1,
    chunks: int = 65536,
    zarr_version: int = 2,
    compressor: Any | None = None,
    consolidate: bool = True,
) -> None:
    """
    Process .slp replays and save the resulting rows in a **columnar Zarr** store.

    Each field in the Row dataclass becomes a resizable 1D array under `columns/`.
    Also records a `replay_index` table with [start_row, stop_row] for each replay.

    Notes:
      - Uses a single writer (this process) to avoid concurrent-write issues to Zarr.
      - Sets `_ARRAY_DIMENSIONS=['row']` on each array for xarray compatibility.
      - Optionally consolidates metadata for faster future reads.
    """
    base = Path(base_dir)
    if not base.exists() or not base.is_dir():
        logger.error(f"Base directory not found or not a directory: {base}")
        return

    out_path = Path(out_path)

    if compressor is None:
        compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    # Build selected replay list (deterministic sampling)
    rng = random.Random(12345)
    selected: list[Path] = []
    for sub in sorted([p for p in base.iterdir() if p.is_dir()]):
        slp_files = [p for p in sub.iterdir() if p.is_file() and p.suffix.lower() == ".slp"]
        if not slp_files:
            continue
        k = min(replays_per_subdir, len(slp_files))
        selected.extend(rng.sample(slp_files, k=k))

    if not selected:
        logger.warning(f"No .slp files found under {base}")
        return

    # Open/create the root Zarr group
    root = zarr.open_group(str(out_path), mode="a", zarr_version=zarr_version, use_consolidated=False)
    root.attrs.setdefault("schema", "row_columnar_v1")
    root.attrs["source_dir"] = str(base)

    # Create replay_index dataset (2 columns: start, stop) if absent
    if "replay_index" not in root:
        root.create_array(
            "replay_index",
            shape=(0, 2),
            maxshape=(None, 2),
            chunks=(max(1, chunks // 32), 2),
            dtype="<i8",
            compressor=compressor,
        )
        try:
            root["replay_index"].attrs["_ARRAY_DIMENSIONS"] = ["replay", "bound"]
        except Exception:
            pass

    arrays: dict[str, zarr.Array] | None = None

    written = 0
    for rp in tqdm(selected, desc="Saving to Zarr", unit="replay"):
        try:
            rows = process_one_replay(str(rp))
        except Exception as e:
            logger.warning(f"Skipping {rp.name}: {e!r}")
            continue
        if not rows:
            continue

        # Lazily create arrays after first replay when we know the schema
        if arrays is None:
            field_names = [f.name for f in dc_fields(rows[0])]
            arrays = _ensure_zarr_columns(root, field_names, rows[0], chunks=chunks, compressor=compressor, zarr_version=zarr_version)
        assert arrays is not None
        field_names = list(arrays.keys())

        batch_cols = _batch_to_columnar(rows, field_names)
        start, stop = _append_to_zarr_columns(arrays, batch_cols)
        # Append to replay_index
        ri = root["replay_index"]
        ri.resize(ri.shape[0] + 1, 2)
        ri[-1, 0] = start
        ri[-1, 1] = stop
        written += (stop - start)

    # Optionally consolidate metadata for faster readers
    try:
        if consolidate:
            zarr.consolidate_metadata(str(out_path))
    except Exception as e:
        logger.warning(f"Consolidation skipped/failed: {e!r}")

    logger.info(f"Saved {written:,d} rows to Zarr store at {out_path}")


def _stats_shard_for_replay(replay_path: str, *, tdigest_delta: float = 0.01) -> dict:
    """
    Return a compact, picklable shard of stats for one replay:
      - For numeric fields: running moments, min/max, neg/zero/pos, shoulder/stick tallies,
        and a t-digest serialized via TDigest.to_dict() (merged later with update_from_dict()).
      - For bool/enum/str: counts (and up to top-10 items for enum/str)
    """
    try:
        rows = process_one_replay(replay_path)
    except Exception as e:
        return {"__error__": f"read_failed:{e!r}", "__file__": replay_path}

    if not rows:
        return {"__empty__": True, "__file__": replay_path}

    first = rows[0]
    if not is_dataclass(first):
        return {"__error__": f"not_dataclass:{type(first)!r}", "__file__": replay_path}

    field_names: List[str] = [f.name for f in dc_fields(first)]

    # Structures
    num: Dict[str, Dict[str, Any]] = {}
    boo: Dict[str, Dict[str, int]] = {}
    enu: Dict[str, Counter[str]] = {}
    stg: Dict[str, Dict[str, Any]] = {}  # strings (counts + len stats)
    oth: Dict[str, Counter[str]] = {}

    # We lazily classify a field on the first non-None value we see
    classified: Dict[str, str] = {}  # fname -> kind: "num" | "bool" | "enum" | "str" | "other"

    # Initialize a numeric field entry
    def _ensure_num(fname: str) -> Dict[str, Any]:
        if fname in num:
            return num[fname]
        num[fname] = {
            "n": 0,
            "missing": 0,
            "non_finite": 0,
            "min": math.inf,
            "max": -math.inf,
            "sum": 0.0,
            "sumsq": 0.0,
            "neg": 0,
            "zero": 0,
            "pos": 0,
            "shoulder_zero": 0,
            "shoulder_partial": 0,
            "shoulder_full": 0,
            "stick_neutral": 0,
            # serialized t-digest will be stored under 'td_digest' at shard end
        }
        # create a local tdigest with explicit K=25 (CamDavidsonPilon default)
        num[fname]["td"] = TDigest(delta=tdigest_delta, K=25)
        return num[fname]

    for row in rows:
        for fname in field_names:
            try:
                v = getattr(row, fname)
            except Exception:
                continue

            # classify
            kind = classified.get(fname)
            if kind is None and v is not None:
                if _is_enum(v):
                    classified[fname] = "enum"
                elif _is_bool(v):
                    classified[fname] = "bool"
                elif _is_number(v):
                    classified[fname] = "num"
                elif isinstance(v, str):
                    classified[fname] = "str"
                else:
                    classified[fname] = "other"
                kind = classified[fname]
            elif kind is None:
                # Defer classification until we see a non-None; we'll count the missing once classified.
                continue

            # If now classified, but value is None, count as missing for classified fields

            if kind == "num":
                stats = _ensure_num(fname)
                if v is None:
                    stats["missing"] += 1
                    continue
                x = _as_float(v)
                if math.isnan(x):
                    stats["missing"] += 1
                    continue
                if not math.isfinite(x):
                    stats["non_finite"] += 1
                    continue

                # update moments
                stats["n"] += 1
                if x < stats["min"]:
                    stats["min"] = x
                if x > stats["max"]:
                    stats["max"] = x
                stats["sum"] += x
                stats["sumsq"] += x * x
                if x < 0:
                    stats["neg"] += 1
                elif x == 0:
                    stats["zero"] += 1
                else:
                    stats["pos"] += 1

                # special buckets
                if _is_shoulder(fname):
                    if _near(x, 0.0):
                        stats["shoulder_zero"] += 1
                    elif _near(x, 1.0):
                        stats["shoulder_full"] += 1
                    elif 0.0 < x < 1.0:
                        stats["shoulder_partial"] += 1
                if _is_stick(fname) and _near(x, 0.5):
                    stats["stick_neutral"] += 1

                # tdigest
                stats["td"].update(x, 1.0)

            elif kind == "bool":
                d = boo.setdefault(fname, {"n": 0, "missing": 0, "true": 0, "false": 0})
                if v is None:
                    d["missing"] += 1
                else:
                    d["n"] += 1
                    if bool(v):
                        d["true"] += 1
                    else:
                        d["false"] += 1

            elif kind == "enum":
                c = enu.setdefault(fname, Counter())
                if v is None:
                    c["<None>"] += 1
                else:
                    if _is_enum(v):
                        key = f"{type(v).__name__}.{getattr(v, 'name', str(v))}"
                    else:
                        key = str(v)
                    c[key] += 1

            elif kind == "str":
                s = stg.setdefault(fname, {"n": 0, "missing": 0, "sum_len": 0, "min_len": math.inf, "max_len": 0,
                                           "counts": Counter()})
                if v is None:
                    s["missing"] += 1
                else:
                    vv = str(v)
                    L = len(vv)
                    s["n"] += 1
                    s["sum_len"] += L
                    if L < s["min_len"]:
                        s["min_len"] = L
                    if L > s["max_len"]:
                        s["max_len"] = L
                    s["counts"][vv] += 1

            else:
                c = oth.setdefault(fname, Counter())
                if v is None:
                    c["<None>"] += 1
                else:
                    try:
                        c[repr(v)] += 1
                    except Exception:
                        c["<unrepr>"] += 1

    # serialize digests
    for fname, stats in num.items():
        td: TDigest = stats.pop("td")
        # Use tdigest's built-in (stable) dict serialization to avoid accessing internals
        stats["td_digest"] = td.to_dict()

    # shrink enum/str to top10
    enum_shard: Dict[str, Dict[str, Any]] = {}
    for k, c in enu.items():
        total = sum(c.values())
        top10 = c.most_common(10)
        enum_shard[k] = {"count": total, "unique": len(c), "top10": top10}

    str_shard: Dict[str, Dict[str, Any]] = {}
    for k, s in stg.items():
        mean_len = (s["sum_len"] / s["n"]) if s["n"] else 0.0
        top10 = s["counts"].most_common(10)
        str_shard[k] = {
            "count": s["n"],
            "missing": s["missing"],
            "unique": len(s["counts"]),
            "min_len": None if s["n"] == 0 else int(s["min_len"]),
            "mean_len": float(mean_len),
            "max_len": None if s["n"] == 0 else int(s["max_len"]),
            "top10": top10,
        }

    return {
        "__file__": replay_path,
        "numeric": num,
        "bool": boo,
        "enum": enum_shard,
        "str": str_shard,
        "other": {k: dict(v) for k, v in oth.items()},
    }


def _merge_numeric_into(
        dst: Dict[str, Dict[str, Any]],
        src: Dict[str, Dict[str, Any]],
        *,
        tdigest_delta: float = 0.01,
) -> None:
    """Merge numeric shards, including t-digests."""
    for fname, s in src.items():
        d = dst.get(fname)
        if d is None:
            # clone
            d = {
                "n": 0, "missing": 0, "non_finite": 0,
                "min": math.inf, "max": -math.inf,
                "sum": 0.0, "sumsq": 0.0,
                "neg": 0, "zero": 0, "pos": 0,
                "shoulder_zero": 0, "shoulder_partial": 0, "shoulder_full": 0,
                "stick_neutral": 0,
                "td": TDigest(delta=tdigest_delta) if TDigest is not None else None,
            }
            dst[fname] = d

        # moments, counts
        d["n"] += s["n"]
        d["missing"] += s["missing"]
        d["non_finite"] += s["non_finite"]
        d["min"] = min(d["min"], s["min"])
        d["max"] = max(d["max"], s["max"])
        d["sum"] += s["sum"]
        d["sumsq"] += s["sumsq"]
        d["neg"] += s["neg"]
        d["zero"] += s["zero"]
        d["pos"] += s["pos"]
        d["shoulder_zero"] += s["shoulder_zero"]
        d["shoulder_partial"] += s["shoulder_partial"]
        d["shoulder_full"] += s["shoulder_full"]
        d["stick_neutral"] += s["stick_neutral"]

        # merge t-digest (CamDavidsonPilon API: use update_from_dict)
        if TDigest is not None:
            td: TDigest = d["td"]
            td_dict = s.get("td_digest")
            if td_dict:
                td.update_from_dict(td_dict)


def _quantiles_from_digest(td: "TDigest", percentiles: Sequence[float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for p in percentiles:
        val = td.percentile(p)  # CamDavidsonPilon API expects 0..100
        key = f"p{int(p)}" if float(int(p)) == float(p) else f"p{str(p).rstrip('0').rstrip('.')}"
        out[key] = float(val)
    return out


def process_replays_with_stats() -> None:
    """
    Parallel, bounded-memory stats using mergeable t-digests for numeric quantiles.
    """
    base_dir = Path("/Users/eppie/Downloads/replays_sorted")
    if not base_dir.exists() or not base_dir.is_dir():
        logger.error(f"Base directory not found or not a directory: {base_dir}")
        return
    if TDigest is None:
        logger.error("tdigest is not installed. Please `pip install tdigest` to enable parallel mergeable quantiles.")
        return

    # Select up to 10 random .slp per subfolder
    rng = random.Random(12345)
    selected_replays: list[Path] = []
    subfolders = [p for p in base_dir.iterdir() if p.is_dir()]
    for sub in sorted(subfolders):
        slp_files = [p for p in sub.iterdir() if p.is_file() and p.suffix.lower() == ".slp"]
        if not slp_files:
            continue
        k = min(1, len(slp_files))
        selected_replays.extend(rng.sample(slp_files, k=k))

    if not selected_replays:
        logger.warning(f"No .slp files found under {base_dir}")
        return

    logger.info(f"Selected {len(selected_replays)} replays from {len(subfolders)} subfolders")

    # Map in parallel to build shards
    shards: List[dict] = []
    procs = min(16, mp.cpu_count())
    with mp.Pool(processes=procs, maxtasksperchild=25) as pool:
        for shard in tqdm(
                pool.imap_unordered(_stats_shard_for_replay, map(str, selected_replays)),
                total=len(selected_replays),
                desc="Computing stats (parallel)",
                unit="replay",
        ):
            if "__error__" in shard or "__empty__" in shard:
                # Optionally log debug and skip
                continue
            shards.append(shard)

    if not shards:
        logger.warning("No stats shards produced.")
        return

    # Merge shards
    numeric: Dict[str, Dict[str, Any]] = {}
    bools: Dict[str, Dict[str, int]] = defaultdict(lambda: {"n": 0, "missing": 0, "true": 0, "false": 0})
    enums_acc: Dict[str, Counter[str]] = defaultdict(Counter)
    strs_acc: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "missing": 0, "unique": 0, "min_len": None, "mean_len": 0.0, "max_len": None, "top10": []})
    others_acc: Dict[str, Counter[str]] = defaultdict(Counter)

    for sh in shards:
        _merge_numeric_into(numeric, sh.get("numeric", {}))
        # bools
        for k, v in sh.get("bool", {}).items():
            b = bools[k]
            b["n"] += v.get("n", 0)
            b["missing"] += v.get("missing", 0)
            b["true"] += v.get("true", 0)
            b["false"] += v.get("false", 0)
        # enums (re-expand from top10 if you want full; here we just add counts we have)
        for k, v in sh.get("enum", {}).items():
            for val, cnt in v.get("top10", []):
                enums_acc[k][val] += cnt
        # strings
        for k, v in sh.get("str", {}).items():
            d = strs_acc[k]
            d["count"] += v.get("count", 0)
            d["missing"] += v.get("missing", 0)
            d["unique"] += v.get("unique", 0)  # note: this overcounts unique across shards; we still show top10
            d["min_len"] = v["min_len"] if d["min_len"] is None else (
                min(d["min_len"], v["min_len"]) if v["min_len"] is not None else d["min_len"])
            d["max_len"] = v["max_len"] if d["max_len"] is None else (
                max(d["max_len"], v["max_len"]) if v["max_len"] is not None else d["max_len"])
            # Weighted running mean for mean_len is messy without sum_len; we’ll recompute below from top10 only (approx).
            # If you want exact, return sum_len from shard. (Simple to add.)
            for val, cnt in v.get("top10", []):
                others_acc[f"__str_values__::{k}"][val] += cnt
        # others
        for k, v in sh.get("other", {}).items():
            others_acc[k].update(v)

    # Compute total rows from a robust numeric field (frame if present)
    total_rows = 0
    if "frame" in numeric:
        total_rows = numeric["frame"]["n"] + numeric["frame"]["missing"]

    # Emit
    lines: List[str] = ["=== Global Dataset Summary ===", f"rows: {total_rows:,d}"]

    # Need the field order; grab from any shard’s enum/num/bool keys union
    field_names = sorted(
        set(list(numeric.keys()) + list(bools.keys()) + list(enums_acc.keys()) + [k.split("::", 1)[-1] for k in
                                                                                  others_acc.keys() if k.startswith(
                "__str_values__::")] + list(others_acc.keys())))

    # Numeric
    for fname in sorted(numeric.keys()):
        d = numeric[fname]
        n = d["n"]
        mean = (d["sum"] / n) if n else float("nan")
        var = max(0.0, (d["sumsq"] / n) - mean * mean) if n else float("nan")
        std = math.sqrt(var) if n else float("nan")
        td: TDigest = d["td"]

        q = _quantiles_from_digest(td, [0.5, 1, 5, 25, 50, 75, 95, 99, 99.5])

        lines.append(f"- {fname}:")
        lines.append(
            f"    numeric | count={n:,d}, missing={d['missing']:,d}, non_finite={d['non_finite']:,d}, "
            f"min={None if n == 0 else float(d['min'])}, p50={q.get('p50')}, max={None if n == 0 else float(d['max'])}, "
            f"mean={None if n == 0 else float(mean)}, std={None if n == 0 else float(std)}, "
            f"neg/zero/pos={d['neg']}/{d['zero']}/{d['pos']}"
        )
        q_str = ", ".join(f"{k}={v}" for k, v in q.items())
        lines.append(f"    quantiles | {q_str}")

        if _is_shoulder(fname):
            total = max(1, n)
            z, p, f = d["shoulder_zero"], d["shoulder_partial"], d["shoulder_full"]
            lines.append(
                f"    shoulder | zero={z} ({z / total:.3%}), partial={p} ({p / total:.3%}), full={f} ({f / total:.3%})")
        if _is_stick(fname):
            total = max(1, n)
            lines.append(f"    neutral  | count={d['stick_neutral']} ({d['stick_neutral'] / total:.3%})")

    # Enums
    for fname in sorted(enums_acc.keys()):
        c = enums_acc[fname]
        total = sum(c.values())
        lines.append(f"- {fname}:")
        lines.append(f"    enum    | count={total:,d}, missing=0, unique≈{len(c):,d}")
        for val, cnt in c.most_common(10):
            lines.append(f"      - {val}: {cnt:,d}")

    # Bools
    for fname in sorted(bools.keys()):
        b = bools[fname]
        total = max(1, b["n"])
        lines.append(f"- {fname}:")
        lines.append(
            f"    bool    | count={b['n']:,d}, missing={b['missing']:,d}, true={b['true']:,d}, false={b['false']:,d}"
        )
        lines.append(f"    rates   | true={b['true'] / total:.3%}, false={b['false'] / total:.3%}")
        vc = Counter()
        if b["true"]:
            vc[True] = b["true"]
        if b["false"]:
            vc[False] = b["false"]
        if vc:
            lines.append(f"    values  | {vc}")

    # Strings (approx: top10 across shards)
    for tag, counts in others_acc.items():
        if not tag.startswith("__str_values__::"):
            continue
        fname = tag.split("::", 1)[-1]
        total = sum(counts.values())
        lines.append(f"- {fname}:")
        lines.append(f"    str     | count≈{total:,d}, missing=~, unique≈{len(counts):,d}, len[min/mean/max]=~")
        for val, cnt in counts.most_common(10):
            lines.append(f"      - {val!r}: {cnt:,d}")

    # Other
    for fname, counts in others_acc.items():
        if fname.startswith("__str_values__::"):
            continue
        total = sum(counts.values())
        lines.append(f"- {fname}:")
        lines.append(f"    other   | count≈{total:,d}, missing=~, unique≈{len(counts):,d}")
        for val, cnt in counts.most_common(10):
            lines.append(f"      - {val}: {cnt:,d}")

    logger.info("\n".join(lines))


if __name__ == "__main__":
    # Example: write processed rows to a Zarr store (uncomment to run)
    save_processed_replays_to_zarr(
        out_path="/Users/eppie/Downloads/processed_replays.zarr",
        base_dir="/Users/eppie/Downloads/replays_sorted",
        replays_per_subdir=1,
        chunks=65536,
        zarr_version=2,
        consolidate=True,
    )
    process_replays_with_stats()
