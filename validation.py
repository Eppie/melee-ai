#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from column_map import ColumnMap
from config import get_config, init_config
from constants import CONTROLLER_KEY_GROUPS, _MAIN_STICK_LABELS, _BUTTON_PRETTY
from controller_quantization import quantize_targets
from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)
from libmelee.melee.enums import Action
from loss import compute_loss_components
from model.nano_gpt import GPT
from train import find_latest_checkpoint
from train.batch_utils import (
    SampleWeightRatios,
    build_model_inputs as build_inputs_for_gpt,
    compute_component_sample_weights,
)
from train.display import _print_table_block
from train.metrics import multilabel_prf
from train.value_head import (
    RewardFeatureIdx,
    build_reward_feature_index,
    compute_frame_rewards,
    compute_value_targets,
)
from utils import _resolve_device, match_state_dict_keys
from feature_transforms import apply_feature_transforms
from window_dataset import (
    WindowDataset,
    RandomWindowSampler,
    SequentialEpisodeSampler,
    worker_init_fn,
    EpisodeInfo,
)

_C_STICK_LABELS = [f"({float(x):.2f},{float(y):.2f})" for x, y in C_STICK_QUANTIZED]
_SHOULDER_LABELS = [f"{float(v):.2f}" for v in SHOULDER_QUANTIZED]
_DEFAULT_THRESHOLD_PATH = Path(__file__).resolve().with_name("button_thresholds.json")

_MAIN_PALETTE_T = torch.tensor(np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32))
_C_PALETTE_T = torch.tensor(np.asarray(C_STICK_QUANTIZED, dtype=np.float32))
_BTN_BIT_WEIGHTS = torch.tensor(
    [1 << idx for idx, _ in enumerate(CONTROLLER_KEY_GROUPS["buttons"])],
    dtype=torch.int64,
)
_BUTTON_NAME_TO_INDEX = {
    name: idx for idx, name in enumerate(CONTROLLER_KEY_GROUPS["buttons"])
}

_PALETTE_BASE = {"main": _MAIN_PALETTE_T, "c": _C_PALETTE_T}
_PALETTE_CACHE: Dict[Tuple[str, str], torch.Tensor] = {}


def _palette_on_device(name: str, device: torch.device) -> torch.Tensor:
    key = (name, str(device))
    tensor = _PALETTE_CACHE.get(key)
    if tensor is None:
        tensor = _PALETTE_BASE[name].to(device)
        _PALETTE_CACHE[key] = tensor
    return tensor


def _normalize_checkpoint_config(raw_config: Any) -> Optional[Dict[str, Any]]:
    """Return a plain dict from various config payload formats."""
    if raw_config is None:
        return None
    if hasattr(raw_config, "model_dump"):
        try:
            return raw_config.model_dump(mode="python")
        except TypeError:
            return raw_config.model_dump()
    if isinstance(raw_config, (bytes, bytearray)):
        raw_config = raw_config.decode("utf-8")
    if isinstance(raw_config, str):
        try:
            return json.loads(raw_config)
        except json.JSONDecodeError:
            print("Warning: checkpoint config string is not valid JSON; ignoring.")
            return None
    if isinstance(raw_config, dict):
        return dict(raw_config)
    return None


def _merge_config_section(target: Any, updates: Dict[str, Any]) -> bool:
    """Recursively apply nested dict updates onto a Pydantic model."""
    applied = False
    for key, value in updates.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if isinstance(value, dict) and hasattr(current, "model_fields"):
            if _merge_config_section(current, value):
                applied = True
        else:
            setattr(target, key, value)
            applied = True
    return applied


def _apply_checkpoint_config(config, raw_config: Any) -> None:
    """Merge checkpoint config metadata into the active Config instance."""
    cfg_dict = _normalize_checkpoint_config(raw_config)
    if not cfg_dict:
        return
    applied = _merge_config_section(config, cfg_dict)
    if not applied:
        _merge_config_section(config.train, cfg_dict)


def _patch_model_config_from_state_dict(
    config, state_dict: Optional[Dict[str, torch.Tensor]]
) -> None:
    """Infer critical model hyperparameters directly from checkpoint weights."""
    if not isinstance(state_dict, dict):
        return

    proj_weight = state_dict.get("projection_down.weight")
    if isinstance(proj_weight, torch.Tensor):
        config.model.n_embd = proj_weight.shape[0]

    block_indices = []
    for key in state_dict.keys():
        if not key.startswith("blocks."):
            continue
        parts = key.split(".")
        if len(parts) > 1 and parts[1].isdigit():
            block_indices.append(int(parts[1]))

    if block_indices:
        config.model.n_layer = max(block_indices) + 1


def _nearest_diffs_within_window(
    true_frames: Sequence[int],
    pred_frames: Sequence[int],
    window: int,
) -> np.ndarray:
    """Return pred-true latencies for each true frame using the nearest predicted
    frame within ±window. Inputs are assumed sorted ascending (they are by construction).
    Result is a 1D int32 numpy array of latencies (pred - true) for matches only.
    """
    if not true_frames or not pred_frames:
        return np.empty(0, dtype=np.int32)

    t = np.asarray(true_frames, dtype=np.int64)
    p = np.asarray(pred_frames, dtype=np.int64)

    idx = np.searchsorted(p, t, side="left")  # insertion positions for each t

    # Distances to the two nearest neighbors (hi: p[idx], lo: p[idx-1])
    big = np.iinfo(np.int64).max
    hi = np.full_like(t, big)
    lo = np.full_like(t, big)

    hi_mask = idx < p.size
    lo_mask = idx > 0

    if hi_mask.any():
        hi[hi_mask] = p[idx[hi_mask]] - t[hi_mask]
    if lo_mask.any():
        lo[lo_mask] = p[idx[lo_mask] - 1] - t[lo_mask]

    # Choose neighbor with smaller absolute distance (ties favor hi, matching prior behavior closely)
    use_hi = np.abs(hi) <= np.abs(lo)
    best = np.where(use_hi, hi, lo)

    keep = np.abs(best) <= window
    return best[keep].astype(np.int32)


def _pack_button_states(buttons: torch.Tensor) -> torch.Tensor:
    weights = _BTN_BIT_WEIGHTS.to(buttons.device)
    return (buttons.to(torch.int64) * weights).sum(dim=-1)


def _angle_diff(a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
    """Computes the shortest angle difference between two angles in radians."""
    return torch.atan2(torch.sin(a1 - a2), torch.cos(a1 - a2)).abs()


def _encode_state_codes(
    main_idx: torch.Tensor, c_idx: torch.Tensor, buttons: torch.Tensor
) -> torch.Tensor:
    btn_codes = _pack_button_states(buttons)
    return (main_idx.to(torch.int64) << 16) | (c_idx.to(torch.int64) << 8) | btn_codes


@dataclass(frozen=True)
class _CachedEpisode:
    """Materialized episode buffers that live entirely in memory."""

    episode_idx: int
    info: EpisodeInfo
    features: np.ndarray  # shape (T, F), float32
    targets: np.ndarray  # shape (T, Yd) or (T, 0)


def _format_bytes(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units[:-1]:
        if value < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} {units[-1]}"


class PreloadedWindowDataset(WindowDataset):
    """WindowDataset that eagerly loads every episode into RAM."""

    def __init__(
        self,
        data_dir: str | Path,
        progress: bool = True,
    ) -> None:
        super().__init__(data_dir)
        self._episode_cache: List[_CachedEpisode] = self._materialize_all_episodes(
            progress=progress
        )

    def _materialize_all_episodes(self, *, progress: bool) -> List[_CachedEpisode]:
        total_eps = len(self.index.episodes)
        total_frames = sum(ep.num_frames for ep in self.index.episodes)
        if progress:
            print(
                f"[preload] Loading {total_eps} episodes "
                f"({total_frames:,} frames, seq_len={self.seq_len}) into memory..."
            )
        cached: List[_CachedEpisode] = []
        frames_loaded = 0
        bytes_loaded = 0
        start_time = time.perf_counter()
        last_log = start_time

        for ep_idx, ep in enumerate(self.index.episodes):
            feat_arr, target_arr = self.index.open_episode_arrays(ep)
            features_np = np.asarray(feat_arr[:], dtype=np.float32, order="C")
            features_np = np.ascontiguousarray(features_np)
            features_np = apply_feature_transforms(features_np, self._feature_names)
            features_np = np.ascontiguousarray(
                features_np.astype(np.float32, copy=False)
            )

            if target_arr is None:
                targets_np = np.zeros((ep.num_frames, 0), dtype=np.float32)
            else:
                targets_np = np.asarray(target_arr[:], dtype=np.float32, order="C")
                targets_np = np.ascontiguousarray(targets_np)

            cached.append(
                _CachedEpisode(
                    episode_idx=ep_idx,
                    info=ep,
                    features=features_np,
                    targets=targets_np,
                )
            )
            frames_loaded += ep.num_frames
            bytes_loaded += features_np.nbytes + targets_np.nbytes

            if progress:
                now = time.perf_counter()
                if now - last_log >= 0.5 or ep_idx == total_eps - 1:
                    pct = (
                        (frames_loaded / total_frames) * 100 if total_frames else 100.0
                    )
                    elapsed = now - start_time
                    print(
                        f"[preload] {ep_idx + 1}/{total_eps} episodes | "
                        f"{frames_loaded:,}/{total_frames:,} frames ({pct:5.1f}%) | "
                        f"{_format_bytes(bytes_loaded)} resident | "
                        f"{elapsed:.1f}s elapsed",
                        end="\n" if ep_idx == total_eps - 1 else "\r",
                        flush=True,
                    )
                    last_log = now

        return cached

    def __getitem__(self, i: int) -> Dict[str, object]:
        ep_idx, offset = self.index.window_to_episode(i)
        cached = self._episode_cache[ep_idx]
        start = offset
        end = start + self.seq_len
        feature_window = cached.features[start:end, :]
        target_window = cached.targets[start:end, :]

        features_out = torch.from_numpy(feature_window)
        targets_out = torch.from_numpy(target_window)

        return {
            "X": features_out,
            "Y": targets_out,
            "episode_id": cached.info.episode_id,
            "start": start,
        }


@dataclass
class ChangeHoldStats:
    change_correct: float = 0.0
    change_total: float = 0.0
    hold_correct: float = 0.0
    hold_total: float = 0.0

    def update(
        self, correct: torch.Tensor, change_mask: torch.Tensor, hold_mask: torch.Tensor
    ) -> None:
        change_correct = (
            correct[change_mask].float().sum().item() if change_mask.any() else 0.0
        )
        hold_correct = (
            correct[hold_mask].float().sum().item() if hold_mask.any() else 0.0
        )
        self.change_correct += change_correct
        self.change_total += float(change_mask.sum().item())
        self.hold_correct += hold_correct
        self.hold_total += float(hold_mask.sum().item())

    def change_acc(self) -> float:
        return _safe_div(self.change_correct, self.change_total)

    def hold_acc(self) -> float:
        return _safe_div(self.hold_correct, self.hold_total)


@dataclass
class RunStats:
    """Streaming run-length tracker to avoid storing every state."""

    total_length: int = 0
    run_count: int = 0
    max_length: int = 0
    pending_value: Optional[int] = None
    pending_length: int = 0

    def _record_run(self, length: int) -> None:
        self.total_length += length
        self.run_count += 1
        if length > self.max_length:
            self.max_length = length

    def _close_pending(self) -> None:
        if self.pending_length:
            self._record_run(self.pending_length)
            self.pending_length = 0
            self.pending_value = None

    def observe_runs(self, codes: np.ndarray) -> None:
        if codes.size == 0:
            return
        idx = 0
        if self.pending_value is not None:
            while idx < codes.size and int(codes[idx]) == self.pending_value:
                self.pending_length += 1
                idx += 1
            if idx == codes.size:
                return
            self._close_pending()
            codes = codes[idx:]
        if codes.size == 0:
            return
        diffs = np.diff(codes)
        change_points = np.nonzero(diffs)[0] + 1
        starts = np.concatenate(([0], change_points))
        lengths = np.diff(np.concatenate((starts, [codes.size])))
        if lengths.size == 0:
            return
        # Keep the last run pending so it can merge with the next batch.
        for length in lengths[:-1]:
            self._record_run(int(length))
        self.pending_value = int(codes[starts[-1]])
        self.pending_length = int(lengths[-1])

    def finalize(self) -> None:
        self._close_pending()


@dataclass
class EnhancedMetrics:
    """Additional validation metrics beyond basic accuracy."""

    # Stick error metrics (Euclidean distance)
    total_main_stick_error: float = 0.0
    total_c_stick_error: float = 0.0
    total_main_stick_error_change: float = 0.0
    total_main_stick_error_hold: float = 0.0
    total_c_stick_error_change: float = 0.0
    total_c_stick_error_hold: float = 0.0

    # New: Stick error by component
    total_main_stick_magnitude_error: float = 0.0
    total_main_stick_angle_error: float = 0.0
    total_c_stick_magnitude_error: float = 0.0
    total_c_stick_angle_error: float = 0.0

    # Jitter metrics
    total_pred_main_jitter: float = 0.0
    total_true_main_jitter: float = 0.0
    total_pred_c_jitter: float = 0.0
    total_true_c_jitter: float = 0.0
    jitter_frames: int = 0

    # Entropy metrics
    total_main_entropy: float = 0.0
    total_c_entropy: float = 0.0
    total_shoulder_entropy: float = 0.0
    entropy_frames: int = 0

    # Per-state accuracy tracking
    state_correct: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    state_total: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    # New: Situational accuracy tracking
    situational_stats: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "recovery": defaultdict(float),
            "edgeguard": defaultdict(float),
            "neutral": defaultdict(float),
        }
    )

    # Action duration tracking (for "stuck" metric)
    pred_run_stats: RunStats = field(default_factory=RunStats)
    true_run_stats: RunStats = field(default_factory=RunStats)

    # Latency tracking (button change detection)
    lr_button_changes_true: List[int] = field(default_factory=list)  # frame indices
    lr_button_changes_pred: List[int] = field(default_factory=list)

    # New: L-Cancel tracking
    l_cancel_opportunities: int = 0
    l_cancel_true_success: int = 0
    l_cancel_pred_success: int = 0

    # New: Jump type tracking
    jumps_true_short: int = 0
    jumps_true_full: int = 0
    jumps_pred_short_correct: int = 0
    jumps_pred_full_correct: int = 0
    jumps_pred_full_when_short: int = 0
    jumps_pred_short_when_full: int = 0
    jumps_missed_short: int = 0
    jumps_missed_full: int = 0

    # Correlation matrix data
    all_preds_list: List[np.ndarray] = field(default_factory=list)
    all_labels_list: List[np.ndarray] = field(default_factory=list)

    # Value head metrics (RL)
    total_value_mse: float = 0.0
    total_value_mae: float = 0.0
    total_value_pred: float = 0.0
    total_value_target: float = 0.0
    value_pred_list: List[float] = field(default_factory=list)
    value_target_list: List[float] = field(default_factory=list)
    value_frames: int = 0

    # Frame-level tracking for value analysis (store tuples of (value_pred, value_target, reward, frame_features))
    value_frame_data: List[Tuple[float, float, float, np.ndarray]] = field(
        default_factory=list
    )

    total_frames: int = 0


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _format_value(value: object) -> str:
    """Format numeric values with up to 6 significant figures."""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _format_action(value: object) -> str:
    """Format action enum values to their names."""

    _ACTION_VALUE_TO_NAME = {action.value: action.name for action in Action}
    try:
        idx = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    return _ACTION_VALUE_TO_NAME.get(idx, str(idx))


def _compute_lagged_cross_correlation(
    preds: np.ndarray,
    targets: np.ndarray,
    *,
    max_lag: int = 1,
    window: Optional[int] = 4096,
) -> Tuple[Dict[int, float], int]:
    """Return normalized cross-correlation for lags in ``[-max_lag, max_lag]``.

    When ``window`` is ``None`` the computation spans the entire series;
    otherwise it is limited to the most recent ``window`` samples to keep the
    diagnostic local in time (helpful when hunting for phase shifts).
    """
    if preds.size == 0 or targets.size == 0:
        return {}, 0

    if window is None:
        usable = min(preds.size, targets.size)
        preds = preds[-usable:]
        targets = targets[-usable:]
    else:
        usable = min(window, preds.size, targets.size)
        preds = preds[-usable:]
        targets = targets[-usable:]

    results: Dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            x = preds[:lag]
            y = targets[-lag:]
        elif lag > 0:
            x = preds[lag:]
            y = targets[:-lag]
        else:
            x = preds
            y = targets

        if x.size < 2 or y.size < 2:
            results[lag] = float("nan")
            continue

        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denom = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
        if denom == 0.0:
            results[lag] = 0.0
        else:
            results[lag] = float(np.dot(x_centered, y_centered) / denom)

    return results, usable


def _print_extreme_value_frames(
    enhanced: EnhancedMetrics,
    colmap: ColumnMap,
    reward_features: Optional[RewardFeatureIdx],
    top_k: int = 1,
) -> None:
    """Print the top and bottom frames by predicted value, with context.

    Args:
        enhanced: Enhanced metrics containing frame data
        colmap: Column mapping for feature names
        top_k: Number of top/bottom frames to show
    """
    if not enhanced.value_frame_data:
        print("\nNo value frame data available for analysis.")
        return

    # Sort frames by predicted value
    sorted_frames = sorted(
        enhanced.value_frame_data, key=lambda x: x[0]
    )  # Sort by value_pred

    # Get top and bottom k
    bottom_frames = sorted_frames[:top_k]
    top_frames = sorted_frames[-top_k:][::-1]  # Reverse to show highest first

    reward_idx = reward_features or build_reward_feature_index(colmap)
    reward_columns: List[Tuple[str, int]] = []
    for field_info in fields(RewardFeatureIdx):
        col_idx = getattr(reward_idx, field_info.name)
        if col_idx is not None:
            reward_columns.append((field_info.name, col_idx))

    if reward_columns:
        selected_headers = [name for name, _ in reward_columns]
        selected_indices = [idx for _, idx in reward_columns]
    else:
        selected_headers = list(colmap.feat_names)
        selected_indices = list(range(len(colmap.feat_names)))

    # Helper to print a frame with context
    def print_frame_with_context(frame_idx: int, label: str):
        value_pred, value_target, reward, features = enhanced.value_frame_data[
            frame_idx
        ]

        print(f"\n{label}")
        print(f"  Predicted Value: {value_pred:.4f}")
        print(f"  Target Value:    {value_target:.4f}")
        print(f"  Frame Reward:    {reward:.4f}")
        print(f"  Frame Index:     {frame_idx}")

        # Get context (15 frames before and 15 frames after)
        context_start = max(0, frame_idx - 15)
        context_end = min(
            len(enhanced.value_frame_data), frame_idx + 16
        )  # include frame_idx and up to 15 after
        context_frames = []
        context_values = []
        context_targets = []
        context_rewards = []

        for i in range(context_start, context_end):
            if i < len(enhanced.value_frame_data):
                v_pred, v_targ, r, feat = enhanced.value_frame_data[i]
                context_frames.append(feat)
                context_values.append(v_pred)
                context_targets.append(v_targ)
                context_rewards.append(r)

        if context_frames:
            # Build data array for printing
            context_array = np.stack(
                [frame[selected_indices] for frame in context_frames], axis=0
            ).astype(np.float32)
            num_context = len(context_frames)

            # Add value and reward columns
            extended_array = np.zeros((num_context, context_array.shape[1] + 3))
            extended_array[:, :-3] = context_array
            extended_array[:, -3] = context_values
            extended_array[:, -2] = context_targets
            extended_array[:, -1] = context_rewards

            # Extended headers
            extended_headers = selected_headers + ["val_pred", "val_targ", "reward"]

            # Print the context frames
            _print_table_block(
                f"  Context frames ({context_start} to {context_end - 1})",
                extended_headers,
                extended_array,
                max_columns=len(extended_headers),
                formatters={},
            )

    # Print top frames (highest value predictions)
    print("\n" + "=" * 80)
    print("TOP FRAME BY PREDICTED VALUE (Highest)")
    print("=" * 80)

    for i, (v_pred, v_targ, reward, features) in enumerate(top_frames):
        # Find the index of this frame in the original data
        frame_idx = None
        for idx, (vp, vt, r, f) in enumerate(enhanced.value_frame_data):
            if (
                vp == v_pred
                and vt == v_targ
                and r == reward
                and np.array_equal(f, features)
            ):
                frame_idx = idx
                break

        if frame_idx is not None:
            print_frame_with_context(frame_idx, f"Top #{i + 1} Frame")

    # Print bottom frames (lowest value predictions)
    print("\n" + "=" * 80)
    print("BOTTOM FRAME BY PREDICTED VALUE (Lowest)")
    print("=" * 80)

    for i, (v_pred, v_targ, reward, features) in enumerate(bottom_frames):
        # Find the index of this frame in the original data
        frame_idx = None
        for idx, (vp, vt, r, f) in enumerate(enhanced.value_frame_data):
            if (
                vp == v_pred
                and vt == v_targ
                and r == reward
                and np.array_equal(f, features)
            ):
                frame_idx = idx
                break

        if frame_idx is not None:
            print_frame_with_context(frame_idx, f"Bottom #{i + 1} Frame")


# Action state categorization for per-state accuracy
_IDLE_STATES = {0x0E, 0x27, 0x28, 0x29}  # STANDING, CROUCH_START, CROUCHING, CROUCH_END
_HITSTUN_STATES = {
    0x26,  # TUMBLING
    0x4B,
    0x4C,
    0x4D,
    0x4E,
    0x4F,
    0x50,
    0x51,
    0x52,
    0x53,  # DAMAGE_HIGH/NEUTRAL/LOW
    0x54,
    0x55,
    0x56,  # DAMAGE_AIR
    0x57,
    0x58,
    0x59,
    0x5A,
    0x5B,  # DAMAGE_FLY variants
    0x9C,
    0x9D,  # DAMAGE_SCREW
    0xC1,  # DAMAGE_GROUND
}
_ATTACK_STATES = set(range(0x2C, 0x4B))  # Attack actions roughly in this range


def _categorize_action(action_value: int) -> str:
    """Categorize an action into idle, hitstun, attack, or other."""
    if action_value in _IDLE_STATES:
        return "idle"
    elif action_value in _HITSTUN_STATES:
        return "hitstun"
    elif action_value in _ATTACK_STATES:
        return "attack"
    else:
        return "other"


def _ensure_absolute(path: Path, anchor: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (anchor / path).resolve()


def _prepare_dataloader(
    data_root: Path,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: Optional[int],
    persistent_workers: bool,
    window_stride: int,
) -> Tuple[DataLoader, WindowDataset]:
    config = get_config()
    dataset = PreloadedWindowDataset(str(data_root))

    if num_workers and num_workers > 0:
        raise ValueError(
            "In-memory validation only supports --num-workers=0 to avoid duplicating the dataset cache."
        )

    # Select sampler based on dataset build configuration
    if dataset.index.sequential_episodes:
        sampler = SequentialEpisodeSampler(
            index=dataset.index,
            stride=window_stride,
        )
    else:
        sampler = RandomWindowSampler(
            index=dataset.index,
            stride=window_stride,
        )

    mp_ctx = None
    if num_workers and num_workers > 0:
        try:
            mp_ctx = torch.multiprocessing.get_context("spawn")
        except RuntimeError:
            mp_ctx = None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        worker_init_fn=worker_init_fn,
        drop_last=False,
        multiprocessing_context=mp_ctx,
    )
    return loader, dataset


def _frame_rewards_from_batch(
    X: torch.Tensor,
    colmap: ColumnMap,
    reward_features: Optional[RewardFeatureIdx],
) -> torch.Tensor:
    """Recompute per-frame rewards for logging."""
    features = reward_features or build_reward_feature_index(colmap)
    return compute_frame_rewards(X, idx=features)


def _build_sample_weight_ratios(loss_cfg) -> SampleWeightRatios:
    """Mirror the training-time ratios derived from LossConfig."""
    button_overrides = {
        "button_z": loss_cfg.button_z,
        "button_b": loss_cfg.button_b,
        "button_a": loss_cfg.button_a,
        "button_xy": loss_cfg.button_xy,
        "button_lr": loss_cfg.button_lr,
    }
    return SampleWeightRatios(
        main_change=loss_cfg.main_change,
        c_change=loss_cfg.c_change,
        shoulder_change=loss_cfg.shoulder_change,
        buttons_change_default=loss_cfg.buttons_change_default,
        buttons_change_per_key=button_overrides,
        hold_base=loss_cfg.hold_base,
        value_change=loss_cfg.value_change,
    )


def _decode_stick_coords(indices: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Map quantized stick indices to 2D coordinates."""
    B, L = indices.shape
    return palette.index_select(0, indices.reshape(-1)).reshape(B, L, 2)


def _change_hold_masks(sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return bool masks for change vs hold events (first frame treated as hold)."""
    change_mask = torch.zeros_like(sequence, dtype=torch.bool)
    if sequence.shape[1] > 1:
        change_mask[:, 1:] = sequence[:, 1:] != sequence[:, :-1]
    return change_mask, ~change_mask


def _accumulate_stick_errors(
    enhanced: EnhancedMetrics,
    errors: torch.Tensor,
    change_mask: torch.Tensor,
    hold_mask: torch.Tensor,
    prefix: str,
) -> None:
    """Update stick error totals for either main or c stick."""
    total_attr = f"total_{prefix}_stick_error"
    change_attr = f"total_{prefix}_stick_error_change"
    hold_attr = f"total_{prefix}_stick_error_hold"

    setattr(enhanced, total_attr, getattr(enhanced, total_attr) + errors.sum().item())
    if change_mask.any():
        setattr(
            enhanced,
            change_attr,
            getattr(enhanced, change_attr) + errors[change_mask].sum().item(),
        )
    if hold_mask.any():
        setattr(
            enhanced,
            hold_attr,
            getattr(enhanced, hold_attr) + errors[hold_mask].sum().item(),
        )


def _accumulate_jitter(
    enhanced: EnhancedMetrics,
    main_pred_coords: torch.Tensor,
    main_true_coords: torch.Tensor,
    c_pred_coords: torch.Tensor,
    c_true_coords: torch.Tensor,
    prev_pred_main_coords: Optional[torch.Tensor],
    prev_true_main_coords: Optional[torch.Tensor],
    prev_pred_c_coords: Optional[torch.Tensor],
    prev_true_c_coords: Optional[torch.Tensor],
) -> None:
    """Track cross-batch and intra-batch jitter statistics."""
    B, L, _ = main_pred_coords.shape

    def _pairwise_sum(current: torch.Tensor, previous: torch.Tensor) -> float:
        return torch.linalg.norm(current - previous, dim=-1).sum().item()

    if (
        prev_pred_main_coords is not None
        and prev_true_main_coords is not None
        and prev_pred_c_coords is not None
        and prev_true_c_coords is not None
        and prev_pred_main_coords.shape[0] == B
    ):
        enhanced.total_pred_main_jitter += _pairwise_sum(
            main_pred_coords[:, 0], prev_pred_main_coords
        )
        enhanced.total_true_main_jitter += _pairwise_sum(
            main_true_coords[:, 0], prev_true_main_coords
        )
        enhanced.total_pred_c_jitter += _pairwise_sum(
            c_pred_coords[:, 0], prev_pred_c_coords
        )
        enhanced.total_true_c_jitter += _pairwise_sum(
            c_true_coords[:, 0], prev_true_c_coords
        )
        enhanced.jitter_frames += B

    enhanced.total_pred_main_jitter += (
        torch.linalg.norm(main_pred_coords[:, 1:] - main_pred_coords[:, :-1], dim=-1)
        .sum()
        .item()
    )
    enhanced.total_true_main_jitter += (
        torch.linalg.norm(main_true_coords[:, 1:] - main_true_coords[:, :-1], dim=-1)
        .sum()
        .item()
    )
    enhanced.total_pred_c_jitter += (
        torch.linalg.norm(c_pred_coords[:, 1:] - c_pred_coords[:, :-1], dim=-1)
        .sum()
        .item()
    )
    enhanced.total_true_c_jitter += (
        torch.linalg.norm(c_true_coords[:, 1:] - c_true_coords[:, :-1], dim=-1)
        .sum()
        .item()
    )
    enhanced.jitter_frames += B * (L - 1)


def _entropy_sum(logits: torch.Tensor) -> float:
    """Return total entropy for a batch of logits."""
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
    return entropy.sum().item()


def _tally_action_categories(
    enhanced: EnhancedMetrics, actions: np.ndarray, correct_mask: np.ndarray
) -> None:
    """Accumulate per-action-state accuracy statistics."""
    for action_val, is_correct in zip(actions, correct_mask):
        category = _categorize_action(int(action_val))
        enhanced.state_total[category] += 1
        if is_correct:
            enhanced.state_correct[category] += 1


def _update_run_stats(
    enhanced: EnhancedMetrics,
    pred_main_idx: torch.Tensor,
    pred_c_idx: torch.Tensor,
    btn_pred: torch.Tensor,
    target_main: torch.Tensor,
    target_c: torch.Tensor,
    target_btn: torch.Tensor,
) -> None:
    """Track run-lengths for predicted and true controller states."""
    pred_state_codes = _encode_state_codes(pred_main_idx, pred_c_idx, btn_pred)
    true_state_codes = _encode_state_codes(target_main, target_c, target_btn)
    enhanced.pred_run_stats.observe_runs(
        pred_state_codes.reshape(-1).detach().cpu().numpy()
    )
    enhanced.true_run_stats.observe_runs(
        true_state_codes.reshape(-1).detach().cpu().numpy()
    )


def _record_lr_latency(
    enhanced: EnhancedMetrics, target_btn: torch.Tensor, btn_pred: torch.Tensor
) -> None:
    """Accumulate frame indices for L/R button press latency analysis."""
    B, L, _ = target_btn.shape
    lr_idx = _BUTTON_NAME_TO_INDEX["button_lr"]
    target_lr = target_btn[:, :, lr_idx]
    pred_lr = btn_pred[:, :, lr_idx]

    true_changes = (target_lr[:, :-1] == 0) & (target_lr[:, 1:] == 1)
    pred_changes = (pred_lr[:, :-1] == 0) & (pred_lr[:, 1:] == 1)

    batch_offsets = torch.arange(B, device=target_btn.device).unsqueeze(1) * L
    frame_offsets = torch.arange(1, L, device=target_btn.device).unsqueeze(0)
    frame_indices = batch_offsets + frame_offsets + enhanced.total_frames

    enhanced.lr_button_changes_true.extend(frame_indices[true_changes].cpu().tolist())
    enhanced.lr_button_changes_pred.extend(frame_indices[pred_changes].cpu().tolist())


def _tally_jump_types(
    enhanced: EnhancedMetrics,
    target_btn: torch.Tensor,
    btn_pred: torch.Tensor,
    grounded_mask: torch.Tensor,
) -> None:
    """Classify grounded X/Y presses as short (≤2f) or full (≥3f) hops."""
    if target_btn.numel() == 0:
        return

    xy_idx = _BUTTON_NAME_TO_INDEX["button_xy"]
    true_xy = target_btn[:, :, xy_idx].bool() & grounded_mask
    pred_xy = btn_pred[:, :, xy_idx].bool() & grounded_mask

    true_np = true_xy.detach().cpu().numpy()
    pred_np = pred_xy.detach().cpu().numpy()

    for seq_true, seq_pred in zip(true_np, pred_np):
        idx = 0
        length = seq_true.shape[0]
        while idx < length:
            if not seq_true[idx]:
                idx += 1
                continue
            end = idx + 1
            while end < length and seq_true[end]:
                end += 1
            run_len = end - idx

            pred_len = 0
            pred_cursor = idx
            while pred_cursor < length and seq_pred[pred_cursor]:
                pred_len += 1
                pred_cursor += 1

            if 1 <= run_len <= 2:
                enhanced.jumps_true_short += 1
                if pred_len == 0:
                    enhanced.jumps_missed_short += 1
                elif 1 <= pred_len <= 2:
                    enhanced.jumps_pred_short_correct += 1
                elif pred_len >= 3:
                    enhanced.jumps_pred_full_when_short += 1
            elif run_len >= 3:
                enhanced.jumps_true_full += 1
                if pred_len == 0:
                    enhanced.jumps_missed_full += 1
                elif 1 <= pred_len <= 2:
                    enhanced.jumps_pred_short_when_full += 1
                elif pred_len >= 3:
                    enhanced.jumps_pred_full_correct += 1

            idx = end


def _append_correlation_vectors(
    enhanced: EnhancedMetrics,
    main_pred_coords: torch.Tensor,
    main_true_coords: torch.Tensor,
    c_pred_coords: torch.Tensor,
    c_true_coords: torch.Tensor,
    btn_pred: torch.Tensor,
    target_btn: torch.Tensor,
) -> None:
    """Store flattened controller states for later correlation analysis."""
    pred_vecs = np.concatenate(
        [
            main_pred_coords.reshape(-1, 2).cpu().numpy(),
            c_pred_coords.reshape(-1, 2).cpu().numpy(),
            btn_pred.reshape(-1, btn_pred.shape[-1]).cpu().numpy().astype(np.float32),
        ],
        axis=1,
    )
    true_vecs = np.concatenate(
        [
            main_true_coords.reshape(-1, 2).cpu().numpy(),
            c_true_coords.reshape(-1, 2).cpu().numpy(),
            target_btn.reshape(-1, target_btn.shape[-1]).cpu().numpy(),
        ],
        axis=1,
    )
    enhanced.all_preds_list.append(pred_vecs)
    enhanced.all_labels_list.append(true_vecs)


def _accumulate_value_metrics(
    enhanced: EnhancedMetrics,
    X: torch.Tensor,
    colmap: ColumnMap,
    value_idx: Optional[int],
    reward_features: Optional[RewardFeatureIdx],
    value_pred: torch.Tensor,
) -> None:
    """Update all value-head related aggregates."""
    config = get_config()
    value_target = compute_value_targets(
        X,
        colmap,
        gamma=config.rl.gamma,
        reward_idx=value_idx,
        reward_features=reward_features,
    )
    frame_rewards = _frame_rewards_from_batch(X, colmap, reward_features)

    value_mse = ((value_pred - value_target) ** 2).mean().item()
    value_mae = (value_pred - value_target).abs().mean().item()

    frames = X.shape[0] * X.shape[1]
    enhanced.total_value_mse += value_mse * frames
    enhanced.total_value_mae += value_mae * frames
    enhanced.total_value_pred += value_pred.sum().item()
    enhanced.total_value_target += value_target.sum().item()

    value_pred_flat = value_pred.cpu().numpy().flatten().tolist()
    value_target_flat = value_target.cpu().numpy().flatten().tolist()
    enhanced.value_pred_list.extend(value_pred_flat)
    enhanced.value_target_list.extend(value_target_flat)
    enhanced.value_frames += frames

    value_pred_np = value_pred.squeeze(-1).cpu().numpy()
    value_target_np = value_target.squeeze(-1).cpu().numpy()
    frame_rewards_np = frame_rewards.cpu().numpy()
    X_np = X.cpu().numpy()

    B, L = value_pred_np.shape
    for b in range(B):
        for l in range(L):
            enhanced.value_frame_data.append(
                (
                    float(value_pred_np[b, l]),
                    float(value_target_np[b, l]),
                    float(frame_rewards_np[b, l]),
                    X_np[b, l, :].copy(),
                )
            )


def _update_enhanced_metrics(
    enhanced: EnhancedMetrics,
    X: torch.Tensor,
    target_info: Dict[str, torch.Tensor],
    pred_main_idx: torch.Tensor,
    pred_c_idx: torch.Tensor,
    btn_pred: torch.Tensor,
    logits_main: torch.Tensor,
    logits_c: torch.Tensor,
    logits_shoulder: torch.Tensor,
    colmap: ColumnMap,
    value_idx: Optional[int],
    reward_features: Optional[RewardFeatureIdx],
    prev_pred_main_coords: Optional[torch.Tensor],
    prev_pred_c_coords: Optional[torch.Tensor],
    prev_true_main_coords: Optional[torch.Tensor],
    prev_true_c_coords: Optional[torch.Tensor],
    value_pred: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Update enhanced metrics and return current stick coordinates for next iteration."""
    B, L = pred_main_idx.shape
    device = pred_main_idx.device

    target_main = target_info["main_idx"].view(B, L)
    target_c = target_info["c_idx"].view(B, L)
    target_btn = target_info["buttons"]
    feat_names = colmap.feat_names

    if value_pred is None:
        raise ValueError("value_pred must be provided for enhanced metrics")

    # Dequantize to get coordinates
    main_palette = _palette_on_device("main", device)
    c_palette = _palette_on_device("c", device)

    main_pred_coords = _decode_stick_coords(pred_main_idx, main_palette)
    main_true_coords = _decode_stick_coords(target_main, main_palette)
    c_pred_coords = _decode_stick_coords(pred_c_idx, c_palette)
    c_true_coords = _decode_stick_coords(target_c, c_palette)

    # 1. Mean Stick Error (Euclidean distance)
    main_errors = torch.linalg.norm(main_pred_coords - main_true_coords, dim=-1)
    c_errors = torch.linalg.norm(c_pred_coords - c_true_coords, dim=-1)
    main_change_mask, main_hold_mask = _change_hold_masks(target_main)
    c_change_mask, c_hold_mask = _change_hold_masks(target_c)

    _accumulate_stick_errors(
        enhanced, main_errors, main_change_mask, main_hold_mask, "main"
    )
    _accumulate_stick_errors(enhanced, c_errors, c_change_mask, c_hold_mask, "c")

    # New: Stick Error by Vector Components
    # Magnitudes
    main_pred_mag = torch.linalg.norm(main_pred_coords, dim=-1)
    main_true_mag = torch.linalg.norm(main_true_coords, dim=-1)
    c_pred_mag = torch.linalg.norm(c_pred_coords, dim=-1)
    c_true_mag = torch.linalg.norm(c_true_coords, dim=-1)
    enhanced.total_main_stick_magnitude_error += (
        (main_pred_mag - main_true_mag).abs().sum().item()
    )
    enhanced.total_c_stick_magnitude_error += (
        (c_pred_mag - c_true_mag).abs().sum().item()
    )

    # Angles (handle zero-magnitude vectors)
    main_pred_angle = torch.atan2(main_pred_coords[..., 1], main_pred_coords[..., 0])
    main_true_angle = torch.atan2(main_true_coords[..., 1], main_true_coords[..., 0])
    c_pred_angle = torch.atan2(c_pred_coords[..., 1], c_pred_coords[..., 0])
    c_true_angle = torch.atan2(c_true_coords[..., 1], c_true_coords[..., 0])

    # Only compute angle error for non-neutral sticks
    main_angle_mask = (main_true_mag > 0.1) & (main_pred_mag > 0.1)
    c_angle_mask = (c_true_mag > 0.1) & (c_pred_mag > 0.1)
    if main_angle_mask.any():
        enhanced.total_main_stick_angle_error += (
            _angle_diff(
                main_pred_angle[main_angle_mask], main_true_angle[main_angle_mask]
            )
            .sum()
            .item()
        )
    if c_angle_mask.any():
        enhanced.total_c_stick_angle_error += (
            _angle_diff(c_pred_angle[c_angle_mask], c_true_angle[c_angle_mask])
            .sum()
            .item()
        )

    # 2. Jitter (frame-to-frame distance)
    _accumulate_jitter(
        enhanced,
        main_pred_coords,
        main_true_coords,
        c_pred_coords,
        c_true_coords,
        prev_pred_main_coords,
        prev_true_main_coords,
        prev_pred_c_coords,
        prev_true_c_coords,
    )

    # 3. Entropy
    enhanced.total_main_entropy += _entropy_sum(logits_main)
    enhanced.total_c_entropy += _entropy_sum(logits_c)
    enhanced.total_shoulder_entropy += _entropy_sum(logits_shoulder)
    enhanced.entropy_frames += B * L

    # 4. Per-action-state accuracy (vectorized)
    p1_actions = X[..., colmap.ego_action_idx].cpu().numpy().flatten()
    main_correct_mask_flat = (pred_main_idx == target_main).cpu().numpy().flatten()
    _tally_action_categories(enhanced, p1_actions, main_correct_mask_flat)

    # New: Situational Accuracy
    p1_off_stage = X[..., feat_names.index("p1_off_stage")] > 0.5
    p2_off_stage = X[..., feat_names.index("p2_off_stage")] > 0.5
    recovery_mask = p1_off_stage
    edgeguard_mask = p2_off_stage & ~p1_off_stage
    neutral_mask = ~p1_off_stage & ~p2_off_stage
    situations = {
        "recovery": recovery_mask,
        "edgeguard": edgeguard_mask,
        "neutral": neutral_mask,
    }
    main_correct_mask = pred_main_idx == target_main
    btn_em_mask = (btn_pred == target_btn).all(dim=-1)

    for name, mask in situations.items():
        if mask.any():
            frames = mask.sum().item()
            enhanced.situational_stats[name]["frames"] += frames
            enhanced.situational_stats[name]["main_correct"] += (
                main_correct_mask[mask].sum().item()
            )
            enhanced.situational_stats[name]["main_total"] += frames
            enhanced.situational_stats[name]["btn_em_correct"] += (
                btn_em_mask[mask].sum().item()
            )
            enhanced.situational_stats[name]["btn_total"] += frames

    # 5. Controller state sequences for "stuck" duration
    _update_run_stats(
        enhanced, pred_main_idx, pred_c_idx, btn_pred, target_main, target_c, target_btn
    )

    # 6. Button change latency tracking (for L/R button)
    _record_lr_latency(enhanced, target_btn, btn_pred)

    # New: L-Cancel Tracking
    l_cancel_status = X[..., feat_names.index("p1_l_cancel_status")]
    # Opportunity is when a cancel was successful (1) or missed (2)
    opportunity_mask = (l_cancel_status > 0) & (l_cancel_status < 3)
    if opportunity_mask.any():
        enhanced.l_cancel_opportunities += opportunity_mask.sum().item()
        enhanced.l_cancel_true_success += (
            (l_cancel_status[opportunity_mask] == 1).sum().item()
        )
        # Check if model predicted L/R (shield) or Z.
        lr_idx = _BUTTON_NAME_TO_INDEX["button_lr"]
        z_idx = _BUTTON_NAME_TO_INDEX["button_z"]
        pred_any_cancel = btn_pred[..., lr_idx].bool() | btn_pred[..., z_idx].bool()
        enhanced.l_cancel_pred_success += pred_any_cancel[opportunity_mask].sum().item()

    # New: Short hop vs full hop accuracy
    grounded_mask = X[..., feat_names.index("p1_on_ground")] > 0.5
    _tally_jump_types(enhanced, target_btn, btn_pred, grounded_mask)

    # 7. Collect data for correlation matrix
    _append_correlation_vectors(
        enhanced,
        main_pred_coords,
        main_true_coords,
        c_pred_coords,
        c_true_coords,
        btn_pred,
        target_btn,
    )

    enhanced.total_frames += B * L

    # 8. Value head metrics
    _accumulate_value_metrics(
        enhanced, X, colmap, value_idx, reward_features, value_pred
    )

    # Return last coordinates for next batch
    last_pred_main = main_pred_coords[:, -1]
    last_pred_c = c_pred_coords[:, -1]
    last_true_main = main_true_coords[:, -1]
    last_true_c = c_true_coords[:, -1]

    return last_pred_main, last_pred_c, last_true_main, last_true_c


def _evaluate(
    model: GPT,
    loader: DataLoader,
    colmap: ColumnMap,
    device: torch.device,
    progress: bool,
    max_batches: Optional[int] = None,
) -> Dict[str, object]:
    config = get_config()
    value_col_idx = colmap.value_idx
    reward_features = (
        None if value_col_idx is not None else build_reward_feature_index(colmap)
    )
    ratios = _build_sample_weight_ratios(config.loss_weights)

    metrics = defaultdict(float)

    pred_counts_main = torch.zeros(
        config.model.target_shapes_by_head["main_stick"], dtype=torch.long
    )
    pred_counts_c = torch.zeros(
        config.model.target_shapes_by_head["c_stick"], dtype=torch.long
    )
    pred_counts_shoulder = torch.zeros(
        config.model.target_shapes_by_head["shoulder"], dtype=torch.long
    )
    pred_button_presses = torch.zeros(
        config.model.target_shapes_by_head["buttons"], dtype=torch.long
    )

    raw_counts_main = torch.zeros_like(pred_counts_main)
    raw_counts_c = torch.zeros_like(pred_counts_c)
    raw_counts_shoulder = torch.zeros_like(pred_counts_shoulder)

    change_stats_main = ChangeHoldStats()
    change_stats_c = ChangeHoldStats()
    change_stats_buttons = ChangeHoldStats()

    # Enhanced metrics
    enhanced = EnhancedMetrics()
    prev_pred_main_coords: Optional[torch.Tensor] = None
    prev_pred_c_coords: Optional[torch.Tensor] = None
    prev_true_main_coords: Optional[torch.Tensor] = None
    prev_true_c_coords: Optional[torch.Tensor] = None

    start_time = time.time()
    total_tokens = 0
    total_frames = 0

    loss_sums: Dict[str, float] = {
        key: 0.0 for key in ("total", "main", "c", "buttons", "shoulder")
    }

    model.eval()
    total_batches = len(loader)
    results: Dict[str, object] = {}

    with torch.inference_mode():
        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            X: torch.Tensor = batch["X"].to(device, non_blocking=True)
            Y: torch.Tensor = batch["Y"].to(device, non_blocking=True)

            # training=False ensures no P1 controller dropout during validation
            inputs_td = build_inputs_for_gpt(
                X, colmap, training=False,
                exclude_p1_controller=config.model.exclude_p1_controller,
            )
            target_info = quantize_targets(Y, colmap, input_domain="unit01")
            weights = compute_component_sample_weights(
                target_info,
                device,
                ratios=ratios,
                button_names=CONTROLLER_KEY_GROUPS["buttons"],
                change_scale=1.0,
            )

            pred = model(inputs_td)

            logits_main = pred["main_stick"]
            logits_c = pred["c_stick"]
            logits_btn = pred["buttons"]
            logits_shoulder = pred.get("shoulder")

            loss_components = compute_loss_components(
                pred,
                target_info,
                label_smoothing=config.train.label_smoothing,
                sample_weights=weights,
                loss_config=config.loss_weights,
            )

            for key, value in loss_components.items():
                loss_sums[key] += value.item()

            pred_main_idx = logits_main.argmax(dim=-1)
            pred_c_idx = logits_c.argmax(dim=-1)
            btn_probs = torch.sigmoid(logits_btn)
            btn_pred = torch.bernoulli(btn_probs).to(target_info["buttons"].dtype)

            # Update metrics directly
            B, L = pred_main_idx.shape
            target_main = target_info["main_idx"].view(B, L)
            target_c = target_info["c_idx"].view(B, L)
            target_btn = target_info["buttons"]

            main_correct_mask = pred_main_idx == target_main
            c_correct_mask = pred_c_idx == target_c
            main_correct = main_correct_mask.float().sum().item()
            c_correct = c_correct_mask.float().sum().item()
            metrics["main_correct"] += main_correct
            metrics["c_correct"] += c_correct
            metrics["main_total"] += B * L
            metrics["c_total"] += B * L

            em_b, p_b, r_b, f1_b, f1_macro_b = multilabel_prf(target_btn, btn_pred)
            metrics["btn_em_correct"] += em_b * B * L
            metrics["btn_total"] += B * L
            metrics["btn_f1_micro_sum"] += f1_b * B * L
            metrics["btn_f1_macro_sum"] += f1_macro_b * B * L

            btn_exact_match = (btn_pred == target_btn).all(dim=-1)

            # Change/hold masks (first frame treated as hold)
            main_change_mask = torch.zeros_like(target_main, dtype=torch.bool)
            main_change_mask[:, 1:] = target_main[:, 1:] != target_main[:, :-1]
            main_hold_mask = ~main_change_mask
            change_stats_main.update(
                main_correct_mask, main_change_mask, main_hold_mask
            )

            c_change_mask = torch.zeros_like(target_c, dtype=torch.bool)
            c_change_mask[:, 1:] = target_c[:, 1:] != target_c[:, :-1]
            c_hold_mask = ~c_change_mask
            change_stats_c.update(c_correct_mask, c_change_mask, c_hold_mask)

            btn_change_mask = torch.zeros_like(btn_exact_match, dtype=torch.bool)
            btn_change_mask[:, 1:] = torch.any(
                target_btn[:, 1:] != target_btn[:, :-1], dim=-1
            )
            btn_hold_mask = ~btn_change_mask
            change_stats_buttons.update(btn_exact_match, btn_change_mask, btn_hold_mask)

            # Extract value prediction if available
            value_pred = pred.get("value")  # [B, L, 1] or None

            # Update enhanced metrics
            (
                prev_pred_main_coords,
                prev_pred_c_coords,
                prev_true_main_coords,
                prev_true_c_coords,
            ) = _update_enhanced_metrics(
                enhanced,
                X,
                target_info,
                pred_main_idx,
                pred_c_idx,
                btn_pred,
                logits_main,
                logits_c,
                logits_shoulder,
                colmap,
                value_col_idx,
                reward_features,
                prev_pred_main_coords,
                prev_pred_c_coords,
                prev_true_main_coords,
                prev_true_c_coords,
                value_pred,
            )

            pred_counts_main += torch.bincount(
                pred_main_idx.reshape(-1).cpu(), minlength=pred_counts_main.shape[0]
            )
            pred_counts_c += torch.bincount(
                pred_c_idx.reshape(-1).cpu(), minlength=pred_counts_c.shape[0]
            )
            sh_pred = logits_shoulder.argmax(dim=-1)
            pred_counts_shoulder += torch.bincount(
                sh_pred.reshape(-1).cpu(), minlength=pred_counts_shoulder.shape[0]
            )

            pred_button_presses += btn_pred.sum(dim=(0, 1)).cpu().to(torch.long)

            raw_counts_main += torch.bincount(
                target_main.reshape(-1).detach().cpu(),
                minlength=raw_counts_main.shape[0],
            )

            raw_counts_c += torch.bincount(
                target_c.reshape(-1).detach().cpu(),
                minlength=raw_counts_c.shape[0],
            )

            shoulder_idx = target_info.get("shoulder_idx")
            raw_counts_shoulder += torch.bincount(
                shoulder_idx.reshape(-1).detach().cpu(),
                minlength=raw_counts_shoulder.shape[0],
            )

            total_tokens += int(X.numel())
            total_frames += int(X.shape[0] * X.shape[1])

            if progress:
                running_loss = loss_sums["total"] / batch_idx
                acc_main = metrics["main_correct"] / metrics["main_total"]
                acc_c = metrics["c_correct"] / metrics["c_total"]
                btn_em = metrics["btn_em_correct"] / metrics["btn_total"]

                pct = (
                    (batch_idx / total_batches) * 100.0
                    if total_batches
                    else float("nan")
                )
                line = (
                    f"[{batch_idx}/{total_batches if total_batches else '?'} | {pct:5.1f}%] "
                    f"loss {running_loss:.4f} | main acc {acc_main:.3f} "
                    f"(chg {change_stats_main.change_acc():.3f} hold {change_stats_main.hold_acc():.3f}) | "
                    f"c acc {acc_c:.3f} (chg {change_stats_c.change_acc():.3f} hold {change_stats_c.hold_acc():.3f}) | "
                    f"btn EM {btn_em:.3f} (chg {change_stats_buttons.change_acc():.3f} hold {change_stats_buttons.hold_acc():.3f})"
                )

                print(line, flush=True)

    elapsed = time.time() - start_time
    batches_processed = (
        min(total_batches, max_batches) if max_batches is not None else total_batches
    )

    results["loss_sums"] = loss_sums
    results["batches"] = batches_processed
    results["frames"] = total_frames
    results["tokens"] = total_tokens
    results["elapsed"] = elapsed
    results["metrics"] = metrics
    results["main_change_stats"] = change_stats_main
    results["c_change_stats"] = change_stats_c
    results["btn_change_stats"] = change_stats_buttons
    results["pred_counts_main"] = pred_counts_main
    results["pred_counts_c"] = pred_counts_c
    results["pred_counts_shoulder"] = pred_counts_shoulder
    results["pred_button_presses"] = pred_button_presses
    results["raw_counts_main"] = raw_counts_main
    results["raw_counts_c"] = raw_counts_c
    results["raw_counts_shoulder"] = raw_counts_shoulder

    results["enhanced"] = enhanced
    return results


def _format_distribution_comparison(
    pred_counts: np.ndarray,
    true_counts: np.ndarray,
    labels: Optional[Iterable[str]] = None,
    top_k: Optional[int] = 10,
) -> str:
    pred = np.asarray(pred_counts, dtype=np.float64)
    true = np.asarray(true_counts, dtype=np.float64)
    total_pred = pred.sum()
    total_true = true.sum()
    if total_pred <= 0 and total_true <= 0:
        return "  <no data>"
    indices = np.argsort(pred if total_pred > 0 else true)[::-1]
    if top_k is None:
        top_k = len(indices)
    lines = []
    for idx in indices[:top_k]:
        pred_count = pred[idx]
        true_count = true[idx]
        if pred_count <= 0 and true_count <= 0:
            continue
        label = labels[idx] if labels is not None else str(idx)
        pred_pct = (pred_count / total_pred * 100.0) if total_pred else 0.0
        true_pct = (true_count / total_true * 100.0) if total_true else 0.0
        lines.append(
            f"  {label:<12} pred {int(pred_count):,} ({pred_pct:5.2f}%) | true {int(true_count):,} ({true_pct:5.2f}%)"
        )
    remaining = len(indices) - top_k
    if remaining > 0:
        lines.append(f"  ... {remaining} additional bins suppressed")
    return "\n".join(lines)


def _print_button_press_distribution(
    pred_press_counts: np.ndarray,
    true_press_counts: np.ndarray,
    total_frames: int,
) -> None:
    if total_frames <= 0:
        print("  <no frames>")
        return
    button_names = CONTROLLER_KEY_GROUPS["buttons"]
    for idx, name in enumerate(button_names):
        label = _BUTTON_PRETTY.get(name, name.upper())
        pred_press = int(round(float(pred_press_counts[idx])))
        true_press = int(round(float(true_press_counts[idx])))
        pred_release = total_frames - pred_press
        true_release = total_frames - true_press
        pred_press_pct = pred_press / total_frames * 100.0
        pred_release_pct = pred_release / total_frames * 100.0
        true_press_pct = true_press / total_frames * 100.0
        true_release_pct = true_release / total_frames * 100.0
        print(
            f"  {label:<6} pred press {pred_press:,} ({pred_press_pct:5.2f}%) | true press {true_press:,} ({true_press_pct:5.2f}%)"
        )
        print(
            f"          pred release {pred_release:,} ({pred_release_pct:5.2f}%) | true release {true_release:,} ({true_release_pct:5.2f}%)"
        )


def _print_enhanced_metrics(enhanced: EnhancedMetrics) -> None:
    """Print all enhanced validation metrics."""
    print("\n===== Enhanced Metrics =====")

    # 1. Mean Stick Error
    print("\n1. Mean Stick Error:")
    avg_main_error = _safe_div(enhanced.total_main_stick_error, enhanced.total_frames)
    avg_c_error = _safe_div(enhanced.total_c_stick_error, enhanced.total_frames)
    print(f"  Main Stick (Euclidean): {avg_main_error:.4f}")
    print(f"  C-Stick (Euclidean):    {avg_c_error:.4f}")

    # New: Stick Error by component
    avg_main_mag_error = _safe_div(
        enhanced.total_main_stick_magnitude_error, enhanced.total_frames
    )
    avg_main_angle_error = _safe_div(
        enhanced.total_main_stick_angle_error, enhanced.total_frames
    )
    avg_c_mag_error = _safe_div(
        enhanced.total_c_stick_magnitude_error, enhanced.total_frames
    )
    avg_c_angle_error = _safe_div(
        enhanced.total_c_stick_angle_error, enhanced.total_frames
    )
    print(f"  Main Stick (Magnitude): {avg_main_mag_error:.4f}")
    print(
        f"  Main Stick (Angle rad): {avg_main_angle_error:.4f} (~{np.rad2deg(avg_main_angle_error):.2f}°)"
    )
    print(f"  C-Stick (Magnitude):    {avg_c_mag_error:.4f}")
    print(
        f"  C-Stick (Angle rad):    {avg_c_angle_error:.4f} (~{np.rad2deg(avg_c_angle_error):.2f}°)"
    )

    # 2. Jitter
    print("\n2. Stick Stability (Jitter - Avg Frame-to-Frame Distance):")
    avg_pred_main_jitter = _safe_div(
        enhanced.total_pred_main_jitter, enhanced.jitter_frames
    )
    avg_true_main_jitter = _safe_div(
        enhanced.total_true_main_jitter, enhanced.jitter_frames
    )
    avg_pred_c_jitter = _safe_div(enhanced.total_pred_c_jitter, enhanced.jitter_frames)
    avg_true_c_jitter = _safe_div(enhanced.total_true_c_jitter, enhanced.jitter_frames)
    print(f"  Main Stick:")
    print(f"    Predicted:     {avg_pred_main_jitter:.4f}")
    print(f"    Ground Truth:  {avg_true_main_jitter:.4f}")
    print(f"  C-Stick:")
    print(f"    Predicted:     {avg_pred_c_jitter:.4f}")
    print(f"    Ground Truth:  {avg_true_c_jitter:.4f}")

    # 3. Entropy
    print("\n3. Prediction Entropy (Uncertainty):")
    avg_main_entropy = _safe_div(enhanced.total_main_entropy, enhanced.entropy_frames)
    avg_c_entropy = _safe_div(enhanced.total_c_entropy, enhanced.entropy_frames)
    print(f"  Main Stick: {avg_main_entropy:.4f}")
    print(f"  C-Stick:    {avg_c_entropy:.4f}")
    if enhanced.total_shoulder_entropy > 0:
        avg_shoulder_entropy = _safe_div(
            enhanced.total_shoulder_entropy, enhanced.entropy_frames
        )
        print(f"  Shoulder:   {avg_shoulder_entropy:.4f}")

    # 4. Per-action-state accuracy
    print("\n4. Accuracy by Player State (Main Stick):")
    for state_type in ["idle", "hitstun", "attack", "other"]:
        if state_type in enhanced.state_total and enhanced.state_total[state_type] > 0:
            acc = _safe_div(
                enhanced.state_correct[state_type], enhanced.state_total[state_type]
            )
            count = int(enhanced.state_total[state_type])
            print(f"  {state_type.capitalize():<8}: {acc:.3f} ({count:,} frames)")

    # New: Situational Accuracy
    print("\n5. Accuracy by Game Situation:")
    for name, stats in enhanced.situational_stats.items():
        total = stats.get("main_total", 0)
        if total > 0:
            main_acc = _safe_div(stats.get("main_correct", 0), total)
            btn_em = _safe_div(
                stats.get("btn_em_correct", 0), stats.get("btn_total", 0)
            )
            print(
                f"  {name.capitalize():<10}: Main Acc {main_acc:.3f} | Btn EM {btn_em:.3f} ({int(total):,} frames)"
            )

    # 6. Stuck action duration
    print("\n6. 'Stuck' Action Duration (Consecutive Identical States):")
    enhanced.pred_run_stats.finalize()
    enhanced.true_run_stats.finalize()
    if enhanced.pred_run_stats.run_count:
        print(f"  Predicted:")
        print(
            f"    Mean: {_safe_div(enhanced.pred_run_stats.total_length, enhanced.pred_run_stats.run_count):.2f} frames"
        )
        print(f"    Max:  {enhanced.pred_run_stats.max_length} frames")
    if enhanced.true_run_stats.run_count:
        print(f"  Ground Truth:")
        print(
            f"    Mean: {_safe_div(enhanced.true_run_stats.total_length, enhanced.true_run_stats.run_count):.2f} frames"
        )
        print(f"    Max:  {enhanced.true_run_stats.max_length} frames")

    # 7. Action change latency
    print("\n7. L/R Button Press Latency:")
    if enhanced.lr_button_changes_true:
        window = 10  # Search ±10 frames
        latencies = _nearest_diffs_within_window(
            enhanced.lr_button_changes_true,
            enhanced.lr_button_changes_pred,
            window,
        )
        if latencies.size > 0:
            print(f"  Mean Latency: {latencies.mean():.2f} frames")
            print(f"  Std Dev:      {latencies.std():.2f} frames")
            print(
                f"  Matched:      {latencies.size}/{len(enhanced.lr_button_changes_true)}"
            )
        else:
            print(
                f"  No matched events (total true events: {len(enhanced.lr_button_changes_true)})"
            )
    else:
        print("  No L/R button press events detected")

    # New: L-Cancel Rate
    print("\n8. L-Cancel Success Rate:")
    if enhanced.l_cancel_opportunities > 0:
        true_rate = _safe_div(
            enhanced.l_cancel_true_success, enhanced.l_cancel_opportunities
        )
        pred_rate = _safe_div(
            enhanced.l_cancel_pred_success, enhanced.l_cancel_opportunities
        )
        print(f"  Opportunities: {enhanced.l_cancel_opportunities:,}")
        print(f"  Human Success Rate:   {true_rate:.3f}")
        print(f"  Model Attempt Rate:   {pred_rate:.3f} (predicts L/R/Z during window)")
    else:
        print("  No L-cancel opportunities found in validation set.")

    # New: Short Hop Accuracy
    print("\n9. Short Hop vs. Full Hop Accuracy:")
    short_total = int(enhanced.jumps_true_short)
    full_total = int(enhanced.jumps_true_full)
    short_correct = int(enhanced.jumps_pred_short_correct)
    full_correct = int(enhanced.jumps_pred_full_correct)
    short_full_instead = int(enhanced.jumps_pred_full_when_short)
    short_missed = int(enhanced.jumps_missed_short)
    full_short_instead = int(enhanced.jumps_pred_short_when_full)
    full_missed = int(enhanced.jumps_missed_full)
    if short_total > 0:
        short_acc = _safe_div(short_correct, short_total)
        print("  Short-hop opportunities (≤2f hold):")
        print(
            f"    Correct short hops: {short_correct}/{short_total} ({short_acc:.3f})"
        )
        print(f"    Full hopped instead: {short_full_instead}/{short_total}")
        print(f"    No jump: {short_missed}/{short_total}")
    else:
        print("  Short-hop opportunities (≤2f hold): No grounded attempts detected.")
    if full_total > 0:
        full_acc = _safe_div(full_correct, full_total)
        print("  Full-hop opportunities (≥3f hold):")
        print(f"    Correct full hops: {full_correct}/{full_total} ({full_acc:.3f})")
        print(f"    Short hopped instead: {full_short_instead}/{full_total}")
        print(f"    No jump: {full_missed}/{full_total}")
    else:
        print("  Full-hop opportunities (≥3f hold): No grounded attempts detected.")

    # 8. Correlation matrix
    print("\n10. Controller Input Correlation Matrix:")
    if enhanced.all_preds_list and enhanced.all_labels_list:
        # Concatenate all batches
        all_preds = np.vstack(enhanced.all_preds_list)  # [total_frames, features]
        all_labels = np.vstack(enhanced.all_labels_list)  # [total_frames, features]

        # Compute correlation matrices
        pred_corr = np.corrcoef(all_preds, rowvar=False)
        true_corr = np.corrcoef(all_labels, rowvar=False)

        # Compute Frobenius norm of difference
        corr_diff = np.linalg.norm(pred_corr - true_corr, ord="fro")
        print(f"  Frobenius norm of (Pred - True) correlation: {corr_diff:.4f}")
        print(
            f"  (Lower is better - means predicted correlations match true correlations)"
        )

    # 9. Value Head Metrics (RL)
    if enhanced.value_frames > 0:
        print("\n11. Value Head Metrics (RL Critic):")

        avg_value_mse = _safe_div(enhanced.total_value_mse, enhanced.value_frames)
        avg_value_mae = _safe_div(enhanced.total_value_mae, enhanced.value_frames)
        avg_value_pred = _safe_div(enhanced.total_value_pred, enhanced.value_frames)
        avg_value_target = _safe_div(enhanced.total_value_target, enhanced.value_frames)

        print(f"  Mean Squared Error (MSE):     {avg_value_mse:.6f}")
        print(f"  Mean Absolute Error (MAE):    {avg_value_mae:.6f}")
        print(f"  Root Mean Squared Error:      {np.sqrt(avg_value_mse):.6f}")
        print(f"  Average Predicted Value:      {avg_value_pred:.4f}")
        print(f"  Average Target Value:         {avg_value_target:.4f}")

        # Compute correlation
        if len(enhanced.value_pred_list) > 1 and len(enhanced.value_target_list) > 1:
            value_preds = np.array(enhanced.value_pred_list)
            value_targets = np.array(enhanced.value_target_list)

            # Pearson correlation
            correlation = np.corrcoef(value_preds, value_targets)[0, 1]
            print(f"  Pearson Correlation:          {correlation:.4f}")

            # R^2 score
            ss_res = ((value_preds - value_targets) ** 2).sum()
            ss_tot = ((value_targets - value_targets.mean()) ** 2).sum()
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            print(f"  R² Score:                     {r2_score:.4f}")

            lag_corrs_full, corr_total = _compute_lagged_cross_correlation(
                value_preds, value_targets, max_lag=1, window=None
            )
            if lag_corrs_full:
                print(
                    f"\n  Lagged Cross-Correlation (full run, lag=-1/0/+1, {corr_total} frames):"
                )
                for lag in sorted(lag_corrs_full.keys()):
                    label = f"{lag:+d}"
                    value = lag_corrs_full[lag]
                    print(f"    lag {label}: {value:.4f}")

            lag_corrs_recent, corr_window = _compute_lagged_cross_correlation(
                value_preds, value_targets, max_lag=1, window=4096
            )
            if lag_corrs_recent and corr_window < corr_total:
                print(
                    f"\n  Lagged Cross-Correlation (last {corr_window} frames, lag=-1/0/+1):"
                )
                for lag in sorted(lag_corrs_recent.keys()):
                    label = f"{lag:+d}"
                    value = lag_corrs_recent[lag]
                    print(f"    lag {label}: {value:.4f}")

            sample = min(16, value_preds.size)
            if sample > 0:
                print(f"\n  Recent {sample} frames (target → pred):")
                recent_targets = value_targets[-sample:]
                recent_preds = value_preds[-sample:]
                for idx in range(sample):
                    frame_offset = sample - idx
                    print(
                        f"    t-{frame_offset:>2}: {recent_targets[idx]:+.4f} → {recent_preds[idx]:+.4f}"
                    )

            # Distribution statistics
            print(f"\n  Prediction Distribution:")
            print(f"    Min:    {value_preds.min():.4f}")
            print(f"    25%:    {np.percentile(value_preds, 25):.4f}")
            print(f"    Median: {np.median(value_preds):.4f}")
            print(f"    75%:    {np.percentile(value_preds, 75):.4f}")
            print(f"    Max:    {value_preds.max():.4f}")
            print(f"    Std:    {value_preds.std():.4f}")

            print(f"\n  Target Distribution:")
            print(f"    Min:    {value_targets.min():.4f}")
            print(f"    25%:    {np.percentile(value_targets, 25):.4f}")
            print(f"    Median: {np.median(value_targets):.4f}")
            print(f"    75%:    {np.percentile(value_targets, 75):.4f}")
            print(f"    Max:    {value_targets.max():.4f}")
            print(f"    Std:    {value_targets.std():.4f}")

            # Error analysis by value range
            print(f"\n  Error Analysis by Target Value Range:")
            # Divide into quartiles
            quartiles = np.percentile(value_targets, [25, 50, 75])
            ranges = [
                (value_targets <= quartiles[0], f"Low (≤{quartiles[0]:.2f})"),
                (
                    (value_targets > quartiles[0]) & (value_targets <= quartiles[1]),
                    f"Mid-Low ({quartiles[0]:.2f}-{quartiles[1]:.2f})",
                ),
                (
                    (value_targets > quartiles[1]) & (value_targets <= quartiles[2]),
                    f"Mid-High ({quartiles[1]:.2f}-{quartiles[2]:.2f})",
                ),
                (value_targets > quartiles[2], f"High (>{quartiles[2]:.2f})"),
            ]

            for mask, label in ranges:
                if mask.sum() > 0:
                    range_mae = np.abs(value_preds[mask] - value_targets[mask]).mean()
                    range_mse = ((value_preds[mask] - value_targets[mask]) ** 2).mean()
                    count = mask.sum()
                    print(
                        f"    {label:<25} MAE: {range_mae:.4f}, MSE: {range_mse:.6f}, Count: {count:,}"
                    )

    print("===== End Enhanced Metrics =====")


def _print_enhanced_metrics_with_colmap(
    enhanced: EnhancedMetrics, colmap: ColumnMap
) -> None:
    """Print enhanced metrics and extreme value analysis."""
    _print_enhanced_metrics(enhanced)

    # Print extreme value frames if value head is enabled
    if enhanced.value_frame_data:
        reward_features = build_reward_feature_index(colmap)
        _print_extreme_value_frames(enhanced, colmap, reward_features, top_k=1)


def _print_final_summary(
    results: Dict[str, object],
    checkpoint: Path,
    data_root: Path,
    column_map: ColumnMap,
) -> None:
    batches = results["batches"]
    loss_sums: Dict[str, float] = results["loss_sums"]
    elapsed = results["elapsed"]
    total_frames = results["frames"]
    total_tokens = results["tokens"]
    metrics: Dict[str, float] = results["metrics"]
    main_stats: ChangeHoldStats = results["main_change_stats"]
    c_stats: ChangeHoldStats = results["c_change_stats"]
    btn_stats: ChangeHoldStats = results["btn_change_stats"]
    main_pred_counts = results["pred_counts_main"].cpu().numpy()
    c_pred_counts = results["pred_counts_c"].cpu().numpy()
    shoulder_pred_counts = results["pred_counts_shoulder"].cpu().numpy()

    main_true_counts = results["raw_counts_main"].cpu().numpy()
    c_true_counts = results["raw_counts_c"].cpu().numpy()
    shoulder_true_counts = results["raw_counts_shoulder"].cpu().numpy()

    avg_loss = {
        k: (v / batches if batches else float("nan")) for k, v in loss_sums.items()
    }
    per_frame_loss = {
        k: (v / total_frames if total_frames else float("nan"))
        for k, v in loss_sums.items()
    }

    acc_main = _safe_div(metrics["main_correct"], metrics["main_total"])
    acc_c = _safe_div(metrics["c_correct"], metrics["c_total"])
    btn_em = _safe_div(metrics["btn_em_correct"], metrics["btn_total"])
    btn_f1_micro = _safe_div(metrics["btn_f1_micro_sum"], metrics["btn_total"])
    btn_f1_macro = _safe_div(metrics["btn_f1_macro_sum"], metrics["btn_total"])

    print("\n===== Validation Summary =====")
    print(f"Checkpoint: {checkpoint}")
    print(f"Dataset:    {data_root}")
    print(f"Batches:    {batches}")
    print(f"Frames:     {total_frames:,}")
    print(f"Tokens:     {total_tokens:,}")
    print(
        f"Elapsed:    {elapsed:.2f}s | {total_tokens / max(elapsed, 1e-9):,.0f} tokens/s"
    )

    print("\nLoss (per batch):")
    for key in ("total", "main", "c", "buttons", "shoulder"):
        print(f"  {key:>8}: {avg_loss[key]:.6f}")

    print("\nLoss (per frame):")
    for key in ("total", "main", "c", "buttons", "shoulder"):
        print(f"  {key:>8}: {per_frame_loss[key]:.8f}")

    print("\nMain Stick:")
    print(f"  accuracy {acc_main:.3f}")
    print(
        f"  change   {main_stats.change_acc():.3f} | hold {main_stats.hold_acc():.3f}"
    )

    print("\nC-Stick:")
    print(f"  accuracy {acc_c:.3f}")
    print(f"  change   {c_stats.change_acc():.3f} | hold {c_stats.hold_acc():.3f}")

    print("\nButtons:")
    print(f"  EM {btn_em:.3f} | F1μ {btn_f1_micro:.3f} | F1_macro {btn_f1_macro:.3f}")
    print(f"  change {btn_stats.change_acc():.3f} | hold {btn_stats.hold_acc():.3f}")
    print("\nPrediction distributions:")
    print("  Main stick (top bins):")
    print(
        _format_distribution_comparison(
            main_pred_counts, main_true_counts, _MAIN_STICK_LABELS, top_k=10
        )
    )
    print("  C-stick:")
    print(
        _format_distribution_comparison(
            c_pred_counts, c_true_counts, _C_STICK_LABELS, top_k=None
        )
    )

    print("  Shoulder:")
    print(
        _format_distribution_comparison(
            shoulder_pred_counts, shoulder_true_counts, _SHOULDER_LABELS, top_k=None
        )
    )

    # Value head summary
    enhanced: EnhancedMetrics = results.get("enhanced")
    if enhanced and enhanced.value_frames > 0:
        avg_value_mse = _safe_div(enhanced.total_value_mse, enhanced.value_frames)
        avg_value_mae = _safe_div(enhanced.total_value_mae, enhanced.value_frames)
        avg_value_pred = _safe_div(enhanced.total_value_pred, enhanced.value_frames)
        avg_value_target = _safe_div(enhanced.total_value_target, enhanced.value_frames)

        if len(enhanced.value_pred_list) > 1 and len(enhanced.value_target_list) > 1:
            value_preds = np.array(enhanced.value_pred_list)
            value_targets = np.array(enhanced.value_target_list)
            correlation = np.corrcoef(value_preds, value_targets)[0, 1]
            ss_res = ((value_preds - value_targets) ** 2).sum()
            ss_tot = ((value_targets - value_targets.mean()) ** 2).sum()
            r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        else:
            correlation = 0.0
            r2_score = 0.0

        print("\nValue Head (RL Critic):")
        print(
            f"  MSE {avg_value_mse:.6f} | MAE {avg_value_mae:.6f} | RMSE {np.sqrt(avg_value_mse):.6f}"
        )
        print(f"  Correlation {correlation:.4f} | R² {r2_score:.4f}")
        print(f"  Avg Pred {avg_value_pred:.4f} | Avg Target {avg_value_target:.4f}")

    print("===== End Validation Summary =====")

    # Print enhanced metrics
    enhanced: EnhancedMetrics = results.get("enhanced")
    if enhanced:
        _print_enhanced_metrics_with_colmap(enhanced, column_map)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the controller model on a validation set."
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=Path,
        default=None,
        help="Path to a checkpoint; defaults to the latest in train.out_dir.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Evaluation batch size (defaults to 256).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (defaults to 0 for in-process loading).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Prefetch factor when num_workers>0 (defaults to train.prefetch_factor).",
    )
    parser.add_argument(
        "--device", default="auto", help="Torch device to run on (auto/cpu/cuda/mps)."
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=0,
        help="Every N batches, emit a richer interim summary (0 disables).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on the number of batches to evaluate.",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable per-batch progress output."
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=256,
        help="Stride between validation windows (defaults to seq_len for non-overlapping).",
    )

    return parser.parse_args()


def main() -> None:
    init_config()
    config = get_config()
    args = parse_args()

    project_root = Path(__file__).resolve().parent

    data_root = Path(config.zarr.validation_root)
    if not data_root.is_absolute():
        data_root = (project_root / data_root).resolve()
    data_root = data_root.expanduser()
    if not data_root.exists():
        print(f"ERROR: validation data root {data_root} not found", file=sys.stderr)
        sys.exit(1)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        ckpt_dir = Path(config.train.out_dir)
        ckpt_dir = _ensure_absolute(ckpt_dir, project_root)
        checkpoint_path = find_latest_checkpoint(ckpt_dir)
        if checkpoint_path is None:
            print(f"ERROR: no checkpoints found in {ckpt_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = _ensure_absolute(checkpoint_path, Path.cwd())

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    _apply_checkpoint_config(config, ckpt.get("config"))
    _patch_model_config_from_state_dict(config, ckpt.get("model"))

    device = _resolve_device()

    batch_size = args.batch_size or config.train.batch_size
    if args.num_workers is None:
        num_workers = config.train.num_workers
    else:
        num_workers = args.num_workers
    pin_memory = config.train.pin_memory
    persistent_workers = False
    window_stride = max(1, args.window_stride)
    prefetch_factor = (
        args.prefetch_factor
        if args.prefetch_factor is not None
        else config.train.prefetch_factor
    )

    loader, dataset = _prepare_dataloader(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        window_stride=window_stride,
    )

    colmap = ColumnMap.from_dataset(dataset)

    gamestate_dim = len(colmap.gamestate_idxs)
    controller_dim = len(colmap.controller_idxs)

    # Update the config with the dynamic dimensions
    config.model.input_size = (
        config.model.num_stages
        + config.model.num_characters * 2
        + config.model.num_actions * 2
        + gamestate_dim
        + controller_dim
    )

    model = GPT(config)
    # TODO: use loading from checkpoint.py
    model_state = match_state_dict_keys(ckpt["model"], model)
    model.load_state_dict(model_state)
    model.to(device)

    if "optimizer" in ckpt:
        del ckpt["optimizer"]
    if "scaler" in ckpt:
        del ckpt["scaler"]

    print(
        f"Evaluating on {len(dataset):,} windows with batch size {batch_size} (device={device})"
    )

    results = _evaluate(
        model,
        loader,
        colmap,
        device,
        progress=not args.no_progress,
        max_batches=args.max_batches,
    )

    _print_final_summary(results, checkpoint_path, data_root, colmap)


if __name__ == "__main__":
    main()
