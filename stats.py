from __future__ import annotations

import enum as py_enum
import math
import random
from collections import Counter
from dataclasses import is_dataclass, fields as dc_fields
from typing import Any, Dict, List, Sequence, TypeVar

import numpy as np
from loguru import logger

R = TypeVar("R")


def log_all_stats(rows: Sequence[R], *, sample_cap: int = 28800,
                  quantiles: Sequence[float] = (0.5, 1, 5, 25, 50, 75, 95, 99, 99.5)) -> None:
    """
    Log per-field statistics for a homogeneous sequence of dataclass instances (e.g., list[Row]).
    Emits a single multi-line log entry to avoid repeated log prefixes from multiple calls.
    The function introspects the dataclass fields at runtime and adapts automatically when fields
    are added/removed, computing datatype-appropriate summaries.

    Supported categories & summaries
    --------------------------------
    - Numeric (int/float and NumPy numeric scalars):
        count, missing (None/NaN), non-finite, min, max, mean, std, neg/zero/pos counts,
        approximate percentiles from a bounded reservoir sample (configurable).
        - for `*_l_shoulder`/`*_r_shoulder`: bucket fractions for 0, (0,1), 1
        - for `*_main_stick_*`/`*_c_stick_*`: fraction of frames with neutral (0.5) input.
    - Enum (enum.Enum / IntEnum):
        total count, missing, top-K member frequencies (K=10).
    - Bool:
        total count, missing, true/false counts.
        also prints class-imbalance rates (true/false fractions).
    - Str:
        total count, missing, unique (up to a cap), min/mean/max length, top-K examples (by frequency).
    - Other hashable types:
        total count, missing, top-K values (K=10).
    - Fallback for non-hashable/unknown:
        logs presence and type information only.

    Performance
    -----------
    - Single pass over `rows` with O(#fields) aggregator updates per row.
    - Memory is bounded for numeric quantiles via reservoir sampling (cap = `sample_cap`).
      Quantiles are computed with `numpy.percentile` on the reservoir sample.

    Parameters
    ----------
    rows : Sequence[R]
        Sequence of dataclass instances (e.g., list[Row]). If empty, logs a warning and returns.
    sample_cap : int
        Maximum reservoir size for approximate quantiles of numeric fields.
    quantiles : Sequence[float]
        Percentiles (0..100) to report for numeric fields.

    Raises
    ------
    TypeError
        If the sequence is non-empty and elements are not dataclass instances.

    Notes
    -----
    - The function intentionally avoids storing all values for memory efficiency.
    - For extremely skewed or multimodal distributions, consider increasing `sample_cap`.
    """
    if not rows:
        logger.warning("log_all_stats: received 0 rows; nothing to summarize.")
        return

    first = rows[0]
    if not is_dataclass(first):
        raise TypeError(f"log_all_stats expects dataclass instances; got {type(first)!r}")

    def is_bool(v: Any) -> bool:
        return isinstance(v, (bool, np.bool_))

    def is_enum(v: Any) -> bool:
        return isinstance(v, py_enum.Enum)

    def is_number(v: Any) -> bool:
        # Treat NumPy numeric scalars as numbers; exclude booleans from numeric bucket.
        if is_bool(v):
            return False
        return isinstance(v, (int, float, np.number))

    def as_float(v: Any) -> float:
        # Convert numpy/python numerics to float for accumulation
        return float(v)  # safe for int/float/np.number

    def _near(a: float, b: float, tol: float = 1e-6) -> bool:
        return abs(a - b) <= tol

    class NumericAgg:
        __slots__ = (
            "n", "missing", "non_finite", "min", "max", "sum", "sumsq",
            "neg", "zero", "pos", "sample", "_rng", "_seen", "_cap",
            "value_counts", "_distinct_exceeded",
            "_fname", "_is_shoulder", "_is_stick",
            "shoulder_zero", "shoulder_partial", "shoulder_full", "stick_neutral"
        )

        def __init__(self, cap: int, fname: str) -> None:
            self.n: int = 0
            self.missing: int = 0
            self.non_finite: int = 0
            self.min: float = math.inf
            self.max: float = -math.inf
            self.sum: float = 0.0
            self.sumsq: float = 0.0
            self.neg: int = 0
            self.zero: int = 0
            self.pos: int = 0
            self.sample: List[float] = []
            self._rng = random.Random(0xC0FFEE)
            self._seen: int = 0
            self._cap: int = cap
            self.value_counts: Counter[float] = Counter()
            self._distinct_exceeded: bool = False
            self._fname = fname
            self._is_shoulder = fname.endswith(("_l_shoulder", "_r_shoulder"))
            self._is_stick = fname.endswith(("_main_stick_x", "_main_stick_y", "_c_stick_x", "_c_stick_y"))
            self.shoulder_zero = 0
            self.shoulder_partial = 0
            self.shoulder_full = 0
            self.stick_neutral = 0

        def add(self, v: Any) -> None:
            if v is None:
                self.missing += 1
                return
            x = as_float(v)
            if math.isnan(x):
                self.missing += 1
                return
            if not math.isfinite(x):
                self.non_finite += 1
                return

            # Special class-imbalance style tallies for shoulders and sticks
            if self._is_shoulder:
                if _near(x, 0.0):
                    self.shoulder_zero += 1
                elif _near(x, 1.0):
                    self.shoulder_full += 1
                elif 0.0 < x < 1.0:
                    self.shoulder_partial += 1
            if self._is_stick:
                if _near(x, 0.5):
                    self.stick_neutral += 1

            # Track exact value frequencies only while distinct values <= 10
            if not self._distinct_exceeded:
                self.value_counts[float(x)] += 1
                if len(self.value_counts) > 10:
                    self._distinct_exceeded = True
                    self.value_counts.clear()

            # Streaming stats
            self.n += 1
            self._seen += 1
            if x < self.min:
                self.min = x
            if x > self.max:
                self.max = x
            self.sum += x
            self.sumsq += x * x
            if x < 0:
                self.neg += 1
            elif x == 0:
                self.zero += 1
            else:
                self.pos += 1

            # Reservoir sampling for quantiles
            if len(self.sample) < self._cap:
                self.sample.append(x)
            else:
                j = self._rng.randrange(self._seen)
                if j < self._cap:
                    self.sample[j] = x

        def report(self, q: Sequence[float]) -> Dict[str, Any]:
            mean = (self.sum / self.n) if self.n else float("nan")
            var = max(0.0, (self.sumsq / self.n) - mean * mean) if self.n else float("nan")
            std = math.sqrt(var) if self.n else float("nan")
            qvals: Dict[str, float] = {}
            if self.sample:
                arr = np.asarray(self.sample, dtype=np.float64)
                try:
                    percs = np.percentile(arr, list(q))
                    qvals = {f"p{int(p)}": float(v) for p, v in zip(q, percs)}
                except Exception:
                    # Defensive: percentile can fail if sample has weird values (shouldn't with checks above)
                    qvals = {}
            return {
                "count": self.n,
                "missing": self.missing,
                "non_finite": self.non_finite,
                "min": float(self.min) if self.n else None,
                "max": float(self.max) if self.n else None,
                "mean": float(mean) if self.n else None,
                "std": float(std) if self.n else None,
                "neg": self.neg,
                "zero": self.zero,
                "pos": self.pos,
                "quantiles": qvals,
                "shoulder_zero": self.shoulder_zero,
                "shoulder_partial": self.shoulder_partial,
                "shoulder_full": self.shoulder_full,
                "stick_neutral": self.stick_neutral,
                "_is_shoulder": self._is_shoulder,
                "_is_stick": self._is_stick,
            }

    class EnumAgg:
        __slots__ = ("n", "missing", "counts")

        def __init__(self) -> None:
            self.n: int = 0
            self.missing: int = 0
            self.counts: Counter[str] = Counter()

        def add(self, v: Any) -> None:
            if v is None:
                self.missing += 1
                return
            if is_enum(v):
                key = f"{type(v).__name__}.{getattr(v, 'name', str(v))}"
            else:
                key = str(v)
            self.n += 1
            self.counts[key] += 1

        def report(self) -> Dict[str, Any]:
            top = self.counts.most_common(10)
            return {
                "count": self.n,
                "missing": self.missing,
                "unique": len(self.counts),
                "top10": [{"value": k, "count": c} for k, c in top],
            }

    class BoolAgg:
        __slots__ = ("n", "missing", "true", "false")

        def __init__(self) -> None:
            self.n: int = 0
            self.missing: int = 0
            self.true: int = 0
            self.false: int = 0

        def add(self, v: Any) -> None:
            if v is None:
                self.missing += 1
                return
            if bool(v):
                self.true += 1
            else:
                self.false += 1
            self.n += 1

        def report(self) -> Dict[str, Any]:
            return {"count": self.n, "missing": self.missing, "true": self.true, "false": self.false}

    class StrAgg:
        __slots__ = ("n", "missing", "min_len", "max_len", "sum_len", "counts")

        def __init__(self) -> None:
            self.n: int = 0
            self.missing: int = 0
            self.min_len: int = math.inf
            self.max_len: int = 0
            self.sum_len: int = 0
            self.counts: Counter[str] = Counter()

        def add(self, v: Any) -> None:
            if v is None:
                self.missing += 1
                return
            s = str(v)
            L = len(s)
            self.n += 1
            self.sum_len += L
            if L < self.min_len:
                self.min_len = L
            if L > self.max_len:
                self.max_len = L
            # Cap memory by only counting up to some diversity (optional). Here we keep counting.
            self.counts[s] += 1

        def report(self) -> Dict[str, Any]:
            mean_len = (self.sum_len / self.n) if self.n else 0.0
            top = self.counts.most_common(10)
            return {
                "count": self.n,
                "missing": self.missing,
                "unique": len(self.counts),
                "min_len": None if self.n == 0 else int(self.min_len),
                "mean_len": float(mean_len),
                "max_len": None if self.n == 0 else int(self.max_len),
                "top10": [{"value": k, "count": c} for k, c in top],
            }

    class FallbackAgg:
        __slots__ = ("n", "missing", "counts")

        def __init__(self) -> None:
            self.n: int = 0
            self.missing: int = 0
            self.counts: Counter[str] = Counter()

        def add(self, v: Any) -> None:
            if v is None:
                self.missing += 1
                return
            try:
                key = repr(v)
                self.counts[key] += 1
                self.n += 1
            except Exception:
                # Unhashable or repr failed
                self.n += 1

        def report(self) -> Dict[str, Any]:
            top = self.counts.most_common(10)
            return {
                "count": self.n,
                "missing": self.missing,
                "unique": len(self.counts),
                "top10": [{"value": k, "count": c} for k, c in top],
            }

    # -------------------------
    # Build aggregators per field
    # -------------------------
    field_names: List[str] = [f.name for f in dc_fields(first)]
    aggs: Dict[str, Any] = {}

    # Choose aggregator based on the first non-None value observed per field.
    # We defer creation until we find a concrete value (to avoid Optional[...] ambiguity).
    def get_or_create_agg(fname: str, v: Any) -> Any:
        if fname in aggs:
            return aggs[fname]
        if is_enum(v):
            aggs[fname] = EnumAgg()
        elif is_bool(v):
            aggs[fname] = BoolAgg()
        elif is_number(v):
            aggs[fname] = NumericAgg(sample_cap, fname)
        elif isinstance(v, str):
            aggs[fname] = StrAgg()
        else:
            aggs[fname] = FallbackAgg()
        return aggs[fname]

    # First pass: instantiate aggregators robustly and update
    for row in rows:
        # Defensive: allow subclasses / other dataclasses that share fields
        for fname in field_names:
            try:
                value = getattr(row, fname)
            except Exception:
                # Field missing? Skip safely.
                continue
            if fname not in aggs and value is None:
                # Defer until we see a non-None to classify type; but still count as missing later.
                aggs[fname] = None  # placeholder
            if aggs.get(fname) is None and value is not None:
                aggs[fname] = get_or_create_agg(fname, value)
            # If still None, create a fallback so we can count missings consistently
            if aggs.get(fname) is None:
                aggs[fname] = FallbackAgg()
            # Update
            try:
                aggs[fname].add(value)
            except Exception as e:
                logger.debug(f"Skipping update for field '{fname}' due to error: {e!r}")

    # -------------------------
    # Reporting
    # -------------------------
    total_rows = len(rows)
    lines: List[str] = []
    lines.append("=== Dataset Summary ===")
    lines.append(f"rows: {total_rows:,d}")
    lines.append(f"fields: {len(field_names)} -> {', '.join(field_names)}")

    for fname in field_names:
        agg = aggs.get(fname)
        if agg is None:
            lines.append(f"- {fname}: <no data>")
            continue

        lines.append(f"- {fname}:")

        # Dispatch by aggregator type
        if isinstance(agg, NumericAgg):
            rep = agg.report(quantiles)
            lines.append(
                f"    numeric | count={rep['count']:,d}, missing={rep['missing']:,d}, non_finite={rep['non_finite']:,d}, "
                f"min={rep['min']}, p50={rep['quantiles'].get('p50')}, max={rep['max']}, "
                f"mean={rep['mean']}, std={rep['std']}, neg/zero/pos={rep['neg']}/{rep['zero']}/{rep['pos']}"
            )
            if rep["quantiles"]:
                q_str = ", ".join(f"{k}={v}" for k, v in rep["quantiles"].items())
                lines.append(f"    quantiles | {q_str}")
            # If this numeric field only had a small number of distinct values, print exact counts
            if hasattr(agg, "value_counts") and not getattr(agg, "_distinct_exceeded", True) and agg.value_counts:
                lines.append(f"    values  | {agg.value_counts}")
            # Shoulder bucket fractions (0, (0,1), 1)
            if rep.get("_is_shoulder"):
                total = max(1, rep["count"])  # guard against div by zero
                z, p, f = rep["shoulder_zero"], rep["shoulder_partial"], rep["shoulder_full"]
                lines.append(
                    f"    shoulder | zero={z} ({z/total:.3%}), partial={p} ({p/total:.3%}), full={f} ({f/total:.3%})"
                )
            # Stick neutral (value == 0.5)
            if rep.get("_is_stick"):
                total = max(1, rep["count"])  # guard against div by zero
                n = rep["stick_neutral"]
                lines.append(f"    neutral  | count={n} ({n/total:.3%})")

        elif isinstance(agg, EnumAgg):
            rep = agg.report()
            lines.append(
                f"    enum    | count={rep['count']:,d}, missing={rep['missing']:,d}, unique={rep['unique']:,d}"
            )
            for val in rep["top10"]:
                lines.append(f"      - {val['value']}: {val['count']:,d}")

        elif isinstance(agg, BoolAgg):
            rep = agg.report()
            lines.append(
                f"    bool    | count={rep['count']:,d}, missing={rep['missing']:,d}, "
                f"true={rep['true']:,d}, false={rep['false']:,d}"
            )
            total = max(1, rep["count"])  # guard against div by zero
            lines.append(f"    rates   | true={rep['true']/total:.3%}, false={rep['false']/total:.3%}")
            vc = Counter()
            if rep["true"]:
                vc[True] = rep["true"]
            if rep["false"]:
                vc[False] = rep["false"]
            if vc:
                lines.append(f"    values  | {vc}")

        elif isinstance(agg, StrAgg):
            rep = agg.report()
            lines.append(
                f"    str     | count={rep['count']:,d}, missing={rep['missing']:,d}, unique={rep['unique']:,d}, "
                f"len[min/mean/max]={rep['min_len']}/{rep['mean_len']:.2f}/{rep['max_len']}"
            )
            for val in rep["top10"]:
                lines.append(f"      - {val['value']!r}: {val['count']:,d}")
            if len(agg.counts) <= 10 and agg.counts:
                lines.append(f"    values  | {agg.counts}")

        else:  # FallbackAgg or unknown
            rep = agg.report()
            lines.append(
                f"    other   | count={rep['count']:,d}, missing={rep['missing']:,d}, unique={rep['unique']:,d}"
            )
            for val in rep["top10"]:
                lines.append(f"      - {val['value']}: {val['count']:,d}")
            if len(agg.counts) <= 10 and agg.counts:
                lines.append(f"    values  | {agg.counts}")

    # Emit a single log entry to avoid repeated prefixes
    logger.info("\n".join(lines))
