# replay_inspector.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Iterable

import matplotlib.pyplot as plt
# Optional, but handy for interactive use
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from libmelee.melee.enums import Action, Character, Stage


PlayerKey = str  # "p0" or "p1"


# ---------- Core flattening / conversion ----------

def structarray_to_table(arr: pa.StructArray) -> pa.Table:
    """
    Recursively flatten a StructArray into a Table with dot-path column names.
    Ensures columns like 'p0.x', 'p0.y', 'p0.controller.main_stick.x', etc.
    This avoids naming surprises from Table.flatten().
    """
    if not pa.types.is_struct(arr.type):
        raise TypeError("structarray_to_table expects a StructArray")

    columns: list[pa.Array] = []
    names: list[str] = []

    def _recurse(a: pa.Array, prefix: str) -> None:
        t = a.type
        if pa.types.is_struct(t):
            struct_type: pa.StructType = t  # type: ignore[assignment]
            for i, field in enumerate(struct_type):
                # Prefer field.name when available; fall back to positional name
                fname = field.name if field.name is not None and field.name != "" else f"_{i}"
                _recurse(a.field(i), f"{prefix}{fname}.")
        else:
            name = prefix[:-1] if prefix.endswith(".") else prefix
            names.append(name)
            columns.append(a)

    _recurse(arr, "")
    return pa.Table.from_arrays(columns, names=names)


def table_to_pandas(tbl: pa.Table) -> pd.DataFrame:
    """Arrow Table -> pandas DataFrame with Arrow dtypes for fidelity."""
    return tbl.to_pandas(types_mapper=pd.ArrowDtype)


def result_to_pandas(result: pa.StructArray) -> pd.DataFrame:
    """One‑stop: StructArray -> flattened pandas DataFrame (one row per frame)."""
    return table_to_pandas(structarray_to_table(result))


# ---------- Quick introspection ----------

def frame_count(result: pa.StructArray) -> int:
    return len(result)


def get_frame_dict(result: pa.StructArray, idx: int) -> Dict[str, Any]:
    """
    Return a single frame as a plain dict (nested).
    Useful when you want to eyeball one frame without pandas.
    """
    if idx < 0 or idx >= len(result):
        raise IndexError(f"frame {idx} out of range [0, {len(result)-1}]")
    # Fast path to python object
    return result[idx].as_py()


def describe_frame(result: pa.StructArray, idx: int) -> str:
    """
    Human-readable snapshot of a frame with richer per-player details:
    character, stocks, shield, action frame, velocity (self+attack+ground),
    invulnerability/hitlag/hitstun, off-stage/fastfall flags, c-stick/shoulder,
    and button summary. Attempts to show enum names if available.

    Now also includes all per-player `computed_features`.
    """
    f = get_frame_dict(result, idx)

    # Try to access optional enums if they were imported at module scope.
    char_enum = globals().get("Character", None)
    stage_enum = globals().get("Stage", None)

    def enum_name(enum_obj: Any, val: Any) -> Optional[str]:
        if enum_obj is None or val is None:
            return None
        try:
            return enum_obj(int(val)).name
        except Exception:
            return None

    def _fnum(v: Any, fmt: str = "{:.2f}", dash: str = "-") -> str:
        if v is None:
            return dash
        try:
            return fmt.format(float(v))
        except Exception:
            return str(v)

    def _safe_sum(parts: Iterable[Optional[float]]) -> Optional[float]:
        total = 0.0
        any_present = False
        for v in parts:
            if v is not None:
                total += float(v)
                any_present = True
        return total if any_present else None

    def _fmt_cf(cf_dict: Any) -> str:
        """
        Compact formatter for the per-player computed features dict.
        Uses short labels; only shows keys that are present.
        Labels:
          dist, face_op, dz, act_in, stockΔ, edgeguard, cornerP, fsg, cool, jm
        """
        if not isinstance(cf_dict, dict):
            return ""
        def gv(k: str) -> Any:
            return cf_dict.get(k)
        parts_cf: list[str] = []
        # floats
        if gv("distance") is not None:
            parts_cf.append(f"dist={_fnum(gv('distance'))}")
        if gv("facing_opponent") is not None:
            parts_cf.append(f"face_op={gv('facing_opponent')}")
        if gv("distance_to_blastzones") is not None:
            parts_cf.append(f"dz={_fnum(gv('distance_to_blastzones'))}")
        if gv("actionable_in") is not None:
            parts_cf.append(f"act_in={gv('actionable_in')}")
        if gv("stock_delta_event") is not None:
            parts_cf.append(f"stockΔ={gv('stock_delta_event')}")
        if gv("edgeguard_situation") is not None:
            parts_cf.append(f"edgeguard={gv('edgeguard_situation')}")
        if gv("corner_pressure") is not None:
            parts_cf.append(f"cornerP={_fnum(gv('corner_pressure'))}")
        if gv("frames_since_grounded") is not None:
            parts_cf.append(f"fsg={gv('frames_since_grounded')}")
        if gv("cooldown_remaining") is not None:
            parts_cf.append(f"cool={gv('cooldown_remaining')}")
        if gv("jumps_max") is not None:
            parts_cf.append(f"jm={gv('jumps_max')}")
        return (" CF[" + " ".join(parts_cf) + "]") if parts_cf else ""

    def p_line(k: PlayerKey) -> str:
        p: Dict[str, Any] = f.get(k, {}) or {}

        # Basics
        x = p.get("x")
        y = p.get("y")
        facing = "R" if bool(p.get("facing", True)) else "L"
        on_ground = bool(p.get("on_ground", False))
        off_stage = p.get("off_stage")
        fastfall = p.get("is_fastfalling")
        jumps_left = p.get("jumps_left")
        stock = p.get("stock")
        percent = p.get("percent")
        shield = p.get("shield_strength")

        # Action + frame
        action_val = p.get("action")
        action_name = None
        if action_val is not None:
            try:
                action_name = Action(int(action_val)).name  # type: ignore[operator]
            except Exception:
                pass
        action_frame = p.get("action_frame")

        # Invuln / stun
        invul = p.get("invulnerable")
        invul_left = p.get("invulnerability_left")
        hitlag_left = p.get("hitlag_left")
        hitstun_left = p.get("hitstun_frames_left")

        # Velocity components (sum what exists)
        vx = _safe_sum([p.get("speed_air_x_self"), p.get("speed_x_attack"), p.get("speed_ground_x_self")])
        vy = _safe_sum([p.get("speed_y_self"), p.get("speed_y_attack")])

        # Controller
        ctrl = p.get("controller", {}) or {}
        main = ctrl.get("main_stick", {}) or {}
        cstk = ctrl.get("c_stick", {}) or {}
        shoulder = ctrl.get("shoulder")
        btns = ctrl.get("buttons", {}) or {}
        pressed = " ".join(b for b, v in btns.items() if v) or "-"

        # Character (if present)
        char_id = p.get("character") or p.get("character_selected")
        char_name = enum_name(char_enum, char_id)

        action_str = f"{action_val}" + (f" ({action_name})" if action_name else "")

        # Computed features (if present)
        cf = p.get("computed_features") or {}

        parts: list[str] = [
            f"{k}:",
            # (char_name or (f\"char={char_id}\" if char_id is not None else \"\")),
            f"pos=({_fnum(x)},{_fnum(y)})",
            f"v=({_fnum(vx)},{_fnum(vy)})",
            f"face={facing}",
            f"ground={on_ground}",
            (f"off={off_stage}" if off_stage is not None else ""),
            (f"fastfall={fastfall}" if fastfall is not None else ""),
            (f"jumps={jumps_left}" if jumps_left is not None else ""),
            (f"stock={stock}" if stock is not None else ""),
            (f"percent={percent}" if percent is not None else ""),
            (f"shield={_fnum(shield)}" if shield is not None else ""),
            f"action={action_str}",
            (f"a_f={action_frame}" if action_frame is not None else ""),
            (f"invul={invul}" if invul is not None else ""),
            (f"invlft={invul_left}" if invul_left is not None else ""),
            (f"hitlag={hitlag_left}" if hitlag_left is not None else ""),
            (f"hitstun={hitstun_left}" if hitstun_left is not None else ""),
            f"stick=({_fnum(main.get('x', 0.0), '{:.3f}')},{_fnum(main.get('y', 0.0), '{:.3f}')})",
            f"c=({_fnum(cstk.get('x', 0.0), '{:.3f}')},{_fnum(cstk.get('y', 0.0), '{:.3f}')})",
            (f"sh={_fnum(shoulder, '{:.3f}')}" if shoulder is not None else ""),
            f"btns=[{pressed}]",
            _fmt_cf(cf),
        ]

        return " ".join(s for s in parts if s)

    # Distance between players (if coords available)
    dist: Optional[float] = None
    try:
        p0, p1 = f.get("p0", {}), f.get("p1", {})
        if all(isinstance(p, dict) and {"x", "y"} <= set(p.keys()) for p in (p0, p1)):
            dx = float(p0["x"]) - float(p1["x"])
            dy = float(p0["y"]) - float(p1["y"])
            dist = math.hypot(dx, dy)
    except Exception:
        dist = None

    # Stage with optional enum name
    stage_val = f.get("stage")
    stage_name = enum_name(stage_enum, stage_val)

    header = f"frame {idx}"
    if dist is not None:
        header += f"  Δ={dist:.2f}"

    return f"{header}\n{p_line('p0')}\n{p_line('p1')}"

# ---------- Buttons & actions helpers (pandas) ----------

ALL_BUTTONS: Tuple[str, ...] = ("A", "B", "X", "Y", "Z", "L", "R", "D_UP")

def add_buttons_pressed_column(df: pd.DataFrame, player: PlayerKey) -> pd.DataFrame:
    """
    Adds a '{player}.buttons_pressed' string column summarizing pressed buttons per frame.
    Non‑destructive (returns df for chaining).
    """
    cols = [f"{player}.controller.buttons.{b}" for b in ALL_BUTTONS if f"{player}.controller.buttons.{b}" in df.columns]
    def _fmt(row: pd.Series) -> str:
        pressed = [b for b, col in zip(ALL_BUTTONS, cols) if bool(row.get(col, False))]
        return " ".join(pressed) if pressed else "-"
    df[f"{player}.buttons_pressed"] = df.apply(_fmt, axis=1)
    return df


def add_action_names(df: pd.DataFrame, player: PlayerKey) -> pd.DataFrame:
    """
    If libmelee Action enum is available, add a readable '{player}.action_name' column.
    Otherwise no‑op.
    """
    if Action is None:
        return df
    col = f"{player}.action"
    if col not in df.columns:
        return df
    def _name(v: Any) -> Optional[str]:
        try:
            return Action(int(v)).name  # type: ignore[operator]
        except Exception:
            return None
    df[f"{player}.action_name"] = df[col].map(_name)
    return df


# ---------- Sanity checks ----------

@dataclass
class SanityReport:
    n_frames: int
    ports_ok: bool
    no_nans: bool
    main_stick_in_range: bool
    c_stick_in_range: bool
    shoulder_in_range: bool

    def pretty(self) -> str:
        flags = {
            "ports_ok": self.ports_ok,
            "no_nans": self.no_nans,
            "main_stick_in_range": self.main_stick_in_range,
            "c_stick_in_range": self.c_stick_in_range,
            "shoulder_in_range": self.shoulder_in_range,
        }
        body = "\n".join(f"- {k}: {v}" for k, v in flags.items())
        return f"SanityReport(n_frames={self.n_frames})\n{body}"


def sanity_checks(result: pa.StructArray) -> SanityReport:
    """
    Quick integrity checks on stick ranges, NaNs, etc.
    Assumes normalized sticks (0..1) and trigger (0..1). Edit bounds if different.
    """
    tbl = structarray_to_table(result)

    def _in01(name: str) -> bool:
        if name not in tbl.column_names:
            return True  # ignore if missing
        col = tbl[name]
        # Ensure we operate on a single Array (not a ChunkedArray)
        if isinstance(col, pa.ChunkedArray):
            col = col.combine_chunks()
        # Compare directly against scalars to avoid Expression-only paths
        ok_lo = pc.greater_equal(col, 0.0)
        ok_hi = pc.less_equal(col, 1.0)
        ok = pc.and_(ok_lo, ok_hi)
        return bool(pc.all(ok).as_py())

    def _no_nans(cols: Sequence[str]) -> bool:
        for c in cols:
            if c in tbl.column_names:
                col = tbl[c]
                if isinstance(col, pa.ChunkedArray):
                    col = col.combine_chunks()
                if bool(pc.any(pc.is_nan(col)).as_py()):
                    return False
        return True

    n_frames = len(result)
    # Ports should be exactly two top-level player structs
    ports_ok = all(c in tbl.column_names for c in ("p0.percent", "p1.percent"))

    floats_to_check = [
        "p0.x", "p0.y", "p1.x", "p1.y",
        "p0.controller.main_stick.x", "p0.controller.main_stick.y",
        "p1.controller.main_stick.x", "p1.controller.main_stick.y",
        "p0.controller.c_stick.x", "p0.controller.c_stick.y",
        "p1.controller.c_stick.x", "p1.controller.c_stick.y",
        "p0.controller.shoulder", "p1.controller.shoulder",
        "p0.shield_strength", "p1.shield_strength",
    ]
    no_nans = _no_nans([c for c in floats_to_check if c in tbl.column_names])

    main_stick_in_range = _in01("p0.controller.main_stick.x") and _in01("p0.controller.main_stick.y") \
        and _in01("p1.controller.main_stick.x") and _in01("p1.controller.main_stick.y")
    c_stick_in_range = _in01("p0.controller.c_stick.x") and _in01("p0.controller.c_stick.y") \
        and _in01("p1.controller.c_stick.x") and _in01("p1.controller.c_stick.y")
    shoulder_in_range = _in01("p0.controller.shoulder") and _in01("p1.controller.shoulder")

    return SanityReport(
        n_frames=n_frames,
        ports_ok=ports_ok,
        no_nans=no_nans,
        main_stick_in_range=main_stick_in_range,
        c_stick_in_range=c_stick_in_range,
        shoulder_in_range=shoulder_in_range,
    )


# ---------- Simple plots (matplotlib) ----------

def plot_positions(df: pd.DataFrame) -> None:
    """
    Plot p0.x/y and p1.x/y vs frame index (one figure).
    """
    ax = df[["p0.x", "p0.y"]].plot(title="P0 position (x,y) vs frame", xlabel="frame")
    plt.show()
    ax = df[["p1.x", "p1.y"]].plot(title="P1 position (x,y) vs frame", xlabel="frame")
    plt.show()


def plot_percent(df: pd.DataFrame) -> None:
    ax = df[["p0.percent", "p1.percent"]].plot(title="Percent vs frame", xlabel="frame")
    plt.show()


def plot_sticks(df: pd.DataFrame, player: PlayerKey = "p0") -> None:
    """
    Plot main/c-stick X/Y vs frame (two charts).
    """
    cols_main = [f"{player}.controller.main_stick.x", f"{player}.controller.main_stick.y"]
    cols_c = [f"{player}.controller.c_stick.x", f"{player}.controller.c_stick.y"]
    ax = df[cols_main].plot(title=f"{player} main stick (x,y)", xlabel="frame")
    plt.show()
    ax = df[cols_c].plot(title=f"{player} c-stick (x,y)", xlabel="frame")
    plt.show()


# ---------- Convenience wrapper ----------

@dataclass
class ReplayView:
    result: pa.StructArray
    table: pa.Table
    df: pd.DataFrame

    @classmethod
    def from_result(cls, result: pa.StructArray) -> "ReplayView":
        tbl = structarray_to_table(result)
        df = table_to_pandas(tbl)
        df.index.name = "frame"
        return cls(result=result, table=tbl, df=df)

    def print_frame(self, idx: int) -> None:
        print(describe_frame(self.result, idx))

    # Typed helpers
    def add_decoded_columns(self, with_actions: bool = True) -> "ReplayView":
        for p in ("p0", "p1"):
            add_buttons_pressed_column(self.df, p)
            if with_actions:
                add_action_names(self.df, p)
        return self