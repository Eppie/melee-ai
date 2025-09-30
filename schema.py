import dataclasses
from typing import Optional

import numpy as np

# Base specs
COMMON_SPEC = [
    ("stage", np.int32),  # Stage enum
]

PLAYER_SPEC = [
    # Core categorical/ids (stored as ints post-preprocessing)
    ("action", np.int32),
    ("character", np.int32),

    # Geometry
    ("position_x", np.float32),
    ("position_y", np.float32),

    # Damage/stock & state bits
    ("percent", np.int32),
    ("stock", np.int32),
    ("facing", np.float32),
    ("on_ground", np.float32),

    # Buttons
    ("button_a", np.float32),
    ("button_b", np.float32),
    # Logical OR will be applied to buttons X and Y
    ("button_xy", np.float32),
    ("button_z", np.float32),
    # Logical OR will be applied to buttons L and R
    ("button_lr", np.float32),

    # Sticks / shoulders
    ("main_stick_x", np.float32),
    ("main_stick_y", np.float32),
    ("c_stick_x", np.float32),
    ("c_stick_y", np.float32),
    ("shoulder_analog", np.float32),  # game treats L/R shoulder identically

    # Additional state
    ("shield_strength", np.float32),
    # ("is_powershield", np.float32),
    # ("action_frame", np.int32),
    # ("is_reflect_active", np.float32),
    # ("is_subaction_invulnerable", np.float32),
    # ("is_fastfalling", np.float32),
    # ("is_defender_in_hitlag", np.float32),
    # ("is_in_hitlag", np.float32),
    # ("is_holding_character", np.float32),
    # ("is_shield_active", np.float32),
    # ("is_in_hitstun", np.float32),
    # ("is_dead", np.float32),
    # ("is_offscreen", np.float32),
    ("is_invulnerable", np.float32),
    # ("hitlag_left", np.int32),
    # ("hitstun_frames_left", np.int32),
    ("jumps_left", np.int32),
    # ("speed_air_x_self", np.float32),
    # ("speed_y_self", np.float32),
    # ("speed_x_attack", np.float32),
    # ("speed_y_attack", np.float32),
    # ("speed_ground_x_self", np.float32),
    # ("off_stage", np.float32),
    # ("l_cancel_status", np.int32),
]


def _prefixed(spec, prefix: str):
    # spec elements can be (name, type) or (name, type, default/field)
    out = []
    for item in spec:
        if len(item) == 2:
            name, typ = item
            out.append((f"{prefix}{name}", typ))
        else:
            name, typ, default_or_field = item
            out.append((f"{prefix}{name}", typ, default_or_field))
    return out


# Compose all dataclass fields in the final order
_ROW_FIELDS = (
        COMMON_SPEC
        + _prefixed(PLAYER_SPEC, "p1_")
        + _prefixed(PLAYER_SPEC, "p2_")
        + [("replay_hash", Optional[np.uint32], dataclasses.field(default=None))]
        + [("replay_filename", Optional[str], dataclasses.field(default=None))]

)


def get_feature_names() -> list[str]:
    """Canonical feature ordering for model input."""
    names = []
    for field_name, *_ in COMMON_SPEC:
        names.append(field_name)
    for prefix in ["p1_", "p2_"]:
        for field_name, *_ in PLAYER_SPEC:
            names.append(f"{prefix}{field_name}")
    return names


def get_target_names() -> list[str]:
    """Canonical target ordering for model output (P1 controller only)."""
    controller_fields = {
        "main_stick_x", "main_stick_y", "c_stick_x", "c_stick_y",
        "shoulder_analog", "button_a", "button_b", "button_xy",
        "button_z", "button_lr"
    }
    return [f"p1_{field}" for field, *_ in PLAYER_SPEC if field in controller_fields]


# Build the dataclass dynamically (flattened attributes), with slots for memory/perf
Row = dataclasses.make_dataclass("Row", _ROW_FIELDS, slots=True)

# Runtime assert to guarantee p1_/p2_ symmetry on import
_p1_names = [name for (name, *_rest) in _ROW_FIELDS if name.startswith("p1_")]
_p2_names = [name for (name, *_rest) in _ROW_FIELDS if name.startswith("p2_")]
assert [n[3:] for n in _p1_names] == [n[3:] for n in _p2_names], "p1/p2 spec mismatch"

# Convenience: export the authoritative specs, useful elsewhere (e.g., for dtype building)
__all__ = ["Row", "COMMON_SPEC", "PLAYER_SPEC"]

"""
Ideas for normalization:
FRAME (0 → 10,208 in this replay; up to ~28,923 worst-case)
Don’t feed the raw index. Instead provide small, well-behaved time features the model can use:
	•	Normalized progress: t_norm = frame / max_frames_seen_in_train (scalar in [0,1]).
	•	Periodic time signals: add 2–4 sin/cos pairs at different periods over t_norm (e.g., 1×, 2×, 4×).
	        This is the classic positional-encoding idea adapted to RNN/Transformer inputs.  ￼ ￼
	•	(Optional) Time2Vec: a learned multi-frequency time embedding; drop-in and often stronger than raw time.  ￼ ￼
No winsorization needed here; just keep a separate reset mask/token on stock/episode boundaries so the model knows
    when state should “forget.” (You already log action_frame etc.)
========================================================================================================================
STAGE (enum; constant in this episode but 6 legal values overall)
Feed as a learned embedding (e.g., 8–16 dims). This is standard for categorical variables with a small,
fixed vocabulary and lets the model learn stage-specific behavior without one-hot bloat.
Keep using stage-normalized geometry elsewhere (so positions/distances are comparable across stages).
========================================================================================================================
DISTANCE (heavy-tailed: p1≈2.94, p50≈35.28, p99≈244.19, max≈274.85)
This is exactly the kind of feature to treat robustly:
	1.	Stage-normalize first (if not already): e.g., divide by stage half-width so “1.0” ≈ half-stage.
	2.	Winsorize lightly to damp rare spikes. Given your quantiles, start at ~1% per tail → thresholds near
	    lo≈2.94 and hi≈244.19 (use train-split estimates). Keep a boolean distance_clipped flag.
	    Winsorization replaces values beyond chosen percentiles with the percentile values, reducing outlier leverage.  ￼ ￼
	3.	Robust scale the winsorized value using median / IQR (as in sklearn.preprocessing.RobustScaler):
        x_rs = (x_winsor − median) / IQR, where IQR = Q75 − Q25 computed on train.
        This centers at the median and scales by a robust spread that ignores tails.  ￼
	4.	(Optional) If you keep a raw-scale variant, try log1p(distance) then robust scale;
	    sometimes distance behaves nicer on a log axis in scramble/KO situations.
========================================================================================================================
ACTION (categorical; up to 0x18D codes overall)
	•	Use a learned embedding (e.g., 32–128 dims) rather than one-hot.
	        Neural nets learn useful geometry over actions and it scales well with cardinality.
	        Keep an <UNK> bucket for ultra-rare or unseen codes at inference.  ￼
	•	If you ever switch to linear models, fall back to OneHotEncoder for
	        low/medium cardinality; it’s the canonical baseline.  ￼
	•	If you someday explode cardinality (mods, items, etc.), the hashing trick is a lightweight
	        non-learning alternative to bound dimensionality. (Not needed now, but it’s the standard tool.)  ￼ ￼

Extras that help sequence models:
    (a) a tiny one-hot of coarse action family (move/attack/defense/ledge/etc.)
    (b) action_is_new = [action_t != action_{t-1}] to mark boundaries.
========================================================================================================================
CHARACTER (categorical; constant per episode, 26 possible)
	•	Feed a small learned embedding (8–32 dims) and broadcast it to all frames of the episode.
	    This captures character-specific physics/kit without ballooning dims.
	    Same reasoning as actions; one-hot is fine but wasteful.  ￼
========================================================================================================================
POS_X, POS_Y (continuous; long tails, stage-dependent)

Pipeline (do these in order):￼
	2.	Stage-normalize: divide both axes by a stage scale (e.g., half-width and platform/base height)
	    so units are comparable across stages.
	3.	Light winsorization (train-fit): ~1% per tail per axis
	    (replace values outside the 1st/99th percentiles with the cutoffs) to damp blast-zone excursions.
	    Keep a pos_x_clipped / pos_y_clipped flag.  ￼ ￼
	4.	Robust scale with median/IQR (train-fit): x’ = (x−median)/IQR; same for y.
	    This is less sensitive to aerial spikes than z-score.  ￼ ￼
Tip: also feed relative geometry (\\Delta x, \\Delta y to opponent) through the same steps;
many policies care more about where the opponent is than absolute stage coords.
========================================================================================================================
PERCENT (damage; right-skewed 0–110+)
	•	Apply log1p or a power transform (Yeo-Johnson works with zeros), then robust scale.
	    This tames heavy right tails while preserving ordering. If you prefer simple & explicit, log1p is great.  ￼
	•	Optionally add a few buckets (e.g., ≥50, ≥80, ≥100) as auxiliary binary features
	    if your loss benefits from phase hints; keep the main continuous channel too.
========================================================================================================================
STOCK (ordinal, 1–4; 0 appears transiently on KO frames)
	•	Treat as numeric with meaning: scale to [0,1] via stock/4 (train-fit mean/IQR optional).
	    Because order matters, avoid pure one-hot unless a linear model needs it.
	    (Ordinal encodings are appropriate when category order is meaningful.)  ￼
	•	Add event bits the model loves: lost_stock_this_frame, new_stock_spawned,
	    and maybe a tiny counter frames_since_stock_change (robust-scaled).
	    These make resets and invuln windows easy to learn.
========================================================================================================================
FACING (boolean 0/1)
	•	Keep one scalar channel as {-1, +1} (map False→-1, True→+1).
	    Use it to mirror x-like features as above; also feed it directly so the net can condition on facing when needed.
	    Egocentric/mirroring is a common way to exploit symmetry.  ￼
========================================================================================================================
ON_GROUND (boolean)
	•	Encode as float32 0/1. Optionally add ground_edge = on_ground_t & ~on_ground_{t-1} and
	    air_edge for transitions—cheap performance wins in sequence models.
	    (Binary inputs as 0/1 are standard practice.)  ￼
========================================================================================================================
BUTTON_[ABZXYLR])

Representation
	•	Keep each as a float32 0/1 channel (already boolean → numeric).

Derived event features (cheap wins for sequence models)
	•	Rising / falling edges: pressed_t & ~pressed_{t-1}, ~pressed_t & pressed_{t-1}.
	    Because inputs are sampled at frame start and affect the same frame,
	    edges at t align with the state change at t.
	•	Hold length: running counter since last rising edge (cap at, say, 60).
	    Scale the counter robustly (median/IQR) so rare long holds don’t dominate.  ￼
	•	Semantic ORs for equivalences:
	•	jump_pressed = x OR y (game treats X/Y identically).
	•	shield_pressed = l OR r (L/R identical in effect).
Keep the raw per-button bits too—players sometimes bind differently; the net can learn any subtle asymmetries.
Given your replay (buttons are sparse: e.g., Z only ~0.8%), edges and holds add signal without exploding dims.
========================================================================================================================
STICKS (processed to [0,1] with neutral at 0.5)

You want stable, symmetric channels that reflect intent and ignore tiny neutral jitter.

1) Center and scale to symmetric range
	•	Map to [-1, 1]: x̂ = 2*(x-0.5), ŷ = 2*(y-0.5); neutral → 0.

2) Ego-frame & mirroring
	•	Multiply X axes by facing_sign ∈ {−1,+1} so “forward” is +X for every frame.
	    (You already do this for positions; do the same for stick X.)

3) Radial soft deadzone
	•	Compute r = sqrt(x̂^2 + ŷ^2). Choose a radial deadzone δ (start with δ≈0.05–0.10).
	•	If r ≤ δ: set (x̂, ŷ) = (0,0).
	•	Else: rescale magnitude to fill range: r' = (r-δ)/(1-δ), then (x̂, ŷ) *= r'/r.
Radial deadzones are standard in controller APIs

4) Angle & magnitude (optional)
	•	Add mag = r' and (sin θ, cos θ) from the unit vector to give the model rotation-friendly signals
	    (no wraparound at 2π).

5) Activity flags
	•	stick_active = 1[r' > 0] and rising edge on activity.
	    Your c-stick is neutral ≥75% of the time (p25=p50=p75=0.5), so a c_stick_active bit is very informative.

6) No winsorization needed
	•	Sticks are hard-bounded after step (1). Robust/winsor steps are unnecessary; just clamp to [-1,1].
	    (Save robust scaling for unbounded counters.)

Choosing δ from data
Your main_stick_x has median 0.5 and wide spread (p25≈0.244, p75≈0.856).
Start with δ=0.08, then check what % of frames are neutral after dead-zoning
(target ~40–60% neutral on idle states like STANDING).
========================================================================================================================
SHOULDERS (l_shoulder, r_shoulder, in [0,1])
	•	Keep analog channels as-is (already bounded).
	•	As with buttons: edges + holds for the digital bit; robust-scale the hold counter.  ￼
========================================================================================================================
Quick checklist (train-time fit, inference-time apply)
	•	Buttons → float32 0/1 + edges + holds (+ jump/shield ORs).  ￼
	•	Sticks → center to [-1,1], mirror X by facing, radial deadzone (δ≈0.05–0.10), optional (mag, sin θ, cos θ), activity bit.  ￼
	•	Shoulders → keep analog; add digital with threshold (XInput-like), + edges/holds (robust-scaled).  ￼
	•	Any unbounded counters you add (e.g., holds) → RobustScaler (median/IQR) on the train split, then reuse stats at inference.  ￼
========================================================================================================================
"""
