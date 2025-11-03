#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

from column_map import ColumnMap
from constants import CONTROLLER_KEY_GROUPS, _MAIN_STICK_LABELS, _BUTTON_PRETTY
from config import get_config, init_config
from controller_quantization import quantize_targets
from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)
from feature_transforms import feature_spec_from_config
from loss import compute_loss_components
from model.nano_gpt import GPT
from train import (
    RunningMetrics,
    build_inputs_for_gpt,
    compute_value_targets,
)
from train.checkpoint import _latest_checkpoint
from utils import _resolve_device
from window_dataset import WindowDataset, worker_init_fn

_C_STICK_LABELS = [f"({float(x):.2f},{float(y):.2f})" for x, y in C_STICK_QUANTIZED]
_SHOULDER_LABELS = [f"{float(v):.2f}" for v in SHOULDER_QUANTIZED]
_DEFAULT_THRESHOLD_PATH = Path(__file__).resolve().with_name("button_thresholds.json")

_MAIN_PALETTE_T = torch.tensor(np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32))
_C_PALETTE_T = torch.tensor(np.asarray(C_STICK_QUANTIZED, dtype=np.float32))
_SHOULDER_PALETTE_T = torch.tensor(np.asarray(SHOULDER_QUANTIZED, dtype=np.float32))

from typing import Sequence


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


def _load_saved_button_thresholds(path: Path) -> Optional[List[float]]:
    try:
        with path.open("r") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as err:
        print(f"Warning: failed to load button thresholds from {path}: {err}")
        return None

    buttons = data.get("buttons")
    thresholds = data.get("thresholds")
    threshold_map = data.get("threshold_map")

    button_keys = list(CONTROLLER_KEY_GROUPS["buttons"])

    values: Optional[List[float]] = None
    if isinstance(thresholds, list) and len(thresholds) == len(button_keys):
        values = [float(x) for x in thresholds]
    elif isinstance(threshold_map, dict):
        order = (
            buttons
            if isinstance(buttons, list) and len(buttons) == len(button_keys)
            else button_keys
        )
        values = [float(threshold_map.get(key, 0.5)) for key in order]

    if values is not None and len(values) == len(button_keys):
        return values

    print(f"Warning: threshold file {path} missing expected fields; ignoring.")
    return None


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
class EnhancedMetrics:
    """Additional validation metrics beyond basic accuracy."""

    # Stick error metrics (Euclidean distance)
    total_main_stick_error: float = 0.0
    total_c_stick_error: float = 0.0
    total_main_stick_error_change: float = 0.0
    total_main_stick_error_hold: float = 0.0
    total_c_stick_error_change: float = 0.0
    total_c_stick_error_hold: float = 0.0

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

    # Action duration tracking (for "stuck" metric)
    pred_states: List[Tuple] = field(default_factory=list)
    true_states: List[Tuple] = field(default_factory=list)

    # Latency tracking (button change detection)
    lr_button_changes_true: List[int] = field(default_factory=list)  # frame indices
    lr_button_changes_pred: List[int] = field(default_factory=list)

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
    from libmelee.melee.enums import Action

    _ACTION_VALUE_TO_NAME = {action.value: action.name for action in Action}
    try:
        idx = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    return _ACTION_VALUE_TO_NAME.get(idx, str(idx))


def _print_table_block(
    title: str,
    headers: Sequence[str],
    data: np.ndarray,
    *,
    max_columns: int = 8,
    formatters: Optional[Dict[str, Callable[[object], str]]] = None,
) -> None:
    """Print a table of data with headers and formatted values."""
    if data.size == 0 or not len(headers):
        print(f"{title}: <empty>")
        return

    total_cols = len(headers)
    num_rows = data.shape[0]
    frame_label = "frame"
    formatters = formatters or {}
    frame_width = max(
        len(frame_label), len(str(num_rows - 1)) if num_rows else len(frame_label)
    )

    for start in range(0, total_cols, max_columns):
        cols = headers[start : start + max_columns]
        block = data[:, start : start + len(cols)]
        formatted_columns: List[List[str]] = []
        col_widths: List[int] = []
        for col_idx, col_name in enumerate(cols):
            formatter = formatters.get(col_name, _format_value)
            col_values: List[str] = []
            for row_idx in range(num_rows):
                value = block[row_idx, col_idx]
                col_values.append(str(formatter(value)))
            max_value_width = max((len(val) for val in col_values), default=0)
            col_width = max(len(col_name), max_value_width, 6)
            formatted_columns.append(col_values)
            col_widths.append(col_width)

        print(f"{title} (columns {start + 1}-{start + len(cols)} of {total_cols}):")
        widths = [frame_width + 2] + col_widths
        header_cells = [frame_label.rjust(widths[0])]
        header_cells.extend(col.rjust(width) for col, width in zip(cols, widths[1:]))
        print(" ".join(header_cells))

        for row_idx in range(num_rows):
            row_cells = [str(row_idx).rjust(widths[0])]
            for col_values, width in zip(formatted_columns, widths[1:]):
                row_cells.append(col_values[row_idx].rjust(width))
            print(" ".join(row_cells))
        print()


def _compute_frame_rewards(X: torch.Tensor, colmap: ColumnMap) -> torch.Tensor:
    """Compute per-frame rewards based on game state changes.

    Args:
        X: Input features [B, L, F]
        colmap: Column mapping

    Returns:
        Rewards [B, L] - reward for each frame based on state changes
    """
    B, L, F = X.shape
    device = X.device
    rewards = torch.zeros(B, L, device=device)

    # Get config for reward weights
    config = get_config()

    # Apply constant per-frame penalty to discourage stalling
    rewards += config.rl.reward_per_frame

    # Extract relevant features
    p1_stock_idx = (
        colmap.feat_names.index("p1_stock") if "p1_stock" in colmap.feat_names else None
    )
    p2_stock_idx = (
        colmap.feat_names.index("p2_stock") if "p2_stock" in colmap.feat_names else None
    )
    p1_percent_idx = (
        colmap.feat_names.index("p1_percent")
        if "p1_percent" in colmap.feat_names
        else None
    )
    p2_percent_idx = (
        colmap.feat_names.index("p2_percent")
        if "p2_percent" in colmap.feat_names
        else None
    )

    # Hitlag features
    p1_in_hitlag_idx = (
        colmap.feat_names.index("p1_in_hitlag")
        if "p1_in_hitlag" in colmap.feat_names
        else None
    )
    p2_in_hitlag_idx = (
        colmap.feat_names.index("p2_in_hitlag")
        if "p2_in_hitlag" in colmap.feat_names
        else None
    )
    p1_in_defender_hitlag_idx = (
        colmap.feat_names.index("p1_in_defender_hitlag")
        if "p1_in_defender_hitlag" in colmap.feat_names
        else None
    )
    p2_in_defender_hitlag_idx = (
        colmap.feat_names.index("p2_in_defender_hitlag")
        if "p2_in_defender_hitlag" in colmap.feat_names
        else None
    )

    # Shield strength features
    p1_shield_strength_idx = (
        colmap.feat_names.index("p1_shield_strength")
        if "p1_shield_strength" in colmap.feat_names
        else None
    )

    if L > 1:
        # Damage rewards (difference between consecutive frames)
        if p1_percent_idx is not None and p2_percent_idx is not None:
            p1_percent_curr = X[:, 1:, p1_percent_idx]  # [B, L-1]
            p1_percent_prev = X[:, :-1, p1_percent_idx]
            p2_percent_curr = X[:, 1:, p2_percent_idx]
            p2_percent_prev = X[:, :-1, p2_percent_idx]

            damage_dealt = (
                p2_percent_curr - p2_percent_prev
            ) * 100  # scale back to 0-100 range
            damage_taken = (p1_percent_curr - p1_percent_prev) * 100

            rewards[:, 1:] += damage_dealt * config.rl.reward_damage_dealt
            rewards[:, 1:] += damage_taken * config.rl.reward_damage_taken

        # Stock rewards (when stock changes)
        if p1_stock_idx is not None and p2_stock_idx is not None:
            p1_stock_curr = X[:, 1:, p1_stock_idx]  # [B, L-1]
            p1_stock_prev = X[:, :-1, p1_stock_idx]
            p2_stock_curr = X[:, 1:, p2_stock_idx]
            p2_stock_prev = X[:, :-1, p2_stock_idx]

            stock_taken = (p2_stock_prev - p2_stock_curr).clamp(
                min=0
            )  # opponent lost stock
            stock_lost = (p1_stock_prev - p1_stock_curr).clamp(min=0)  # we lost stock

            rewards[:, 1:] += stock_taken * config.rl.reward_stock_taken
            rewards[:, 1:] += stock_lost * config.rl.reward_stock_lost

    # Hitlag rewards/penalties (apply to all frames, not just differences)
    if (
        p1_in_hitlag_idx is not None
        and p1_in_defender_hitlag_idx is not None
        and p2_in_hitlag_idx is not None
        and p2_in_defender_hitlag_idx is not None
    ):
        # Compute hitlag metric for p1 (us): in_hitlag - in_defender_hitlag
        p1_hitlag_metric = (
            X[:, :, p1_in_hitlag_idx] - X[:, :, p1_in_defender_hitlag_idx]
        )  # [B, L]
        # Compute hitlag metric for p2 (opponent): in_hitlag - in_defender_hitlag
        p2_hitlag_metric = (
            X[:, :, p2_in_hitlag_idx] - X[:, :, p2_in_defender_hitlag_idx]
        )  # [B, L]

        # Penalty when we're in hitlag (being hit) - when metric = 1
        p1_in_bad_hitlag = (p1_hitlag_metric == 1.0).float()
        rewards += p1_in_bad_hitlag * config.rl.reward_hitlag_self  # negative reward

        # Reward when opponent is in hitlag (we're hitting them) - when metric = 1
        p2_in_bad_hitlag = (p2_hitlag_metric == 1.0).float()
        rewards += (
            p2_in_bad_hitlag * config.rl.reward_hitlag_opponent
        )  # positive reward

    # Shield strength penalty (apply to all frames)
    if p1_shield_strength_idx is not None:
        p1_shield = X[:, :, p1_shield_strength_idx]  # [B, L], range [0, 1]

        # Apply penalty when shield < 0.5
        # Magnify penalty as shield approaches 0: use (0.5 - shield) / 0.5 to get penalty multiplier
        # When shield = 0.5, penalty = 0
        # When shield = 0.25, penalty multiplier = 0.5
        # When shield = 0, penalty multiplier = 1.0
        low_shield_mask = (p1_shield < 0.5).float()  # [B, L]
        penalty_multiplier = ((0.5 - p1_shield) / 0.5).clamp(
            min=0, max=1
        )  # [B, L], 0 to 1
        shield_penalty = (
            low_shield_mask * penalty_multiplier * config.rl.reward_low_shield
        )  # negative
        rewards += shield_penalty

    return rewards


def _print_extreme_value_frames(
    enhanced: EnhancedMetrics, colmap: ColumnMap, top_k: int = 1
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

        # Get context (30 frames before and 30 frames after)
        context_start = max(0, frame_idx - 30)
        context_end = min(
            len(enhanced.value_frame_data), frame_idx + 31
        )  # +31 to include frame_idx and 30 after
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
            context_array = np.array(context_frames)  # [N, F]
            num_context = len(context_frames)

            # Add value and reward columns
            extended_array = np.zeros((num_context, context_array.shape[1] + 3))
            extended_array[:, :-3] = context_array
            extended_array[:, -3] = context_values
            extended_array[:, -2] = context_targets
            extended_array[:, -1] = context_rewards

            # Extended headers
            extended_headers = colmap.feat_names + ["val_pred", "val_targ", "reward"]

            # Formatters for actions
            formatters: Dict[str, Callable[[object], str]] = {}
            for key in colmap.feat_names:
                if key.endswith("_action"):
                    formatters[key] = _format_action

            # Print the context frames
            _print_table_block(
                f"  Context frames ({context_start} to {context_end - 1})",
                extended_headers,
                extended_array,
                max_columns=10,
                formatters=formatters,
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
            print_frame_with_context(frame_idx, f"Top #{i+1} Frame")

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
            print_frame_with_context(frame_idx, f"Bottom #{i+1} Frame")


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


def _get_run_lengths(sequence: List) -> List[int]:
    """Calculate run lengths of consecutive identical elements."""
    if not sequence:
        return []
    lengths = []
    current_run = 1
    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            current_run += 1
        else:
            lengths.append(current_run)
            current_run = 1
    lengths.append(current_run)
    return lengths


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
) -> Tuple[DataLoader, WindowDataset]:
    config = get_config()
    feature_spec = feature_spec_from_config(config.features)
    dataset = WindowDataset(
        data_dir=str(data_root),
        feature_transforms=feature_spec,
        return_numpy=False,
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
        sampler=SequentialSampler(dataset),
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        worker_init_fn=worker_init_fn,
        drop_last=False,
        multiprocessing_context=mp_ctx,
    )
    return loader, dataset


def _update_enhanced_metrics(
    enhanced: EnhancedMetrics,
    X: torch.Tensor,
    target_info: Dict[str, torch.Tensor],
    pred_main_idx: torch.Tensor,
    pred_c_idx: torch.Tensor,
    btn_pred: torch.Tensor,
    logits_main: torch.Tensor,
    logits_c: torch.Tensor,
    logits_shoulder: Optional[torch.Tensor],
    colmap: ColumnMap,
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

    # Dequantize to get coordinates
    # Need to flatten indices, index, then reshape
    main_pred_idx_flat = pred_main_idx.cpu().numpy().flatten()
    main_pred_coords = (
        torch.from_numpy(
            np.array(CONTROL_STICK_QUANTIZED, dtype=np.float32)[main_pred_idx_flat]
        )
        .to(device)
        .reshape(B, L, 2)
    )

    main_true_idx_flat = target_main.cpu().numpy().flatten()
    main_true_coords = (
        torch.from_numpy(
            np.array(CONTROL_STICK_QUANTIZED, dtype=np.float32)[main_true_idx_flat]
        )
        .to(device)
        .reshape(B, L, 2)
    )

    c_pred_idx_flat = pred_c_idx.cpu().numpy().flatten()
    c_pred_coords = (
        torch.from_numpy(np.array(C_STICK_QUANTIZED, dtype=np.float32)[c_pred_idx_flat])
        .to(device)
        .reshape(B, L, 2)
    )

    c_true_idx_flat = target_c.cpu().numpy().flatten()
    c_true_coords = (
        torch.from_numpy(np.array(C_STICK_QUANTIZED, dtype=np.float32)[c_true_idx_flat])
        .to(device)
        .reshape(B, L, 2)
    )

    # 1. Mean Stick Error (Euclidean distance)
    main_errors = torch.linalg.norm(main_pred_coords - main_true_coords, dim=-1)
    c_errors = torch.linalg.norm(c_pred_coords - c_true_coords, dim=-1)
    enhanced.total_main_stick_error += main_errors.sum().item()
    enhanced.total_c_stick_error += c_errors.sum().item()

    # Track error by change/hold
    main_change_mask = torch.zeros_like(target_main, dtype=torch.bool)
    main_change_mask[:, 1:] = target_main[:, 1:] != target_main[:, :-1]
    main_hold_mask = ~main_change_mask

    c_change_mask = torch.zeros_like(target_c, dtype=torch.bool)
    c_change_mask[:, 1:] = target_c[:, 1:] != target_c[:, :-1]
    c_hold_mask = ~c_change_mask

    if main_change_mask.any():
        enhanced.total_main_stick_error_change += (
            main_errors[main_change_mask].sum().item()
        )
    if main_hold_mask.any():
        enhanced.total_main_stick_error_hold += main_errors[main_hold_mask].sum().item()
    if c_change_mask.any():
        enhanced.total_c_stick_error_change += c_errors[c_change_mask].sum().item()
    if c_hold_mask.any():
        enhanced.total_c_stick_error_hold += c_errors[c_hold_mask].sum().item()

    # 2. Jitter (frame-to-frame distance)
    if prev_pred_main_coords is not None and prev_pred_main_coords.shape[0] == B:
        # Compute jitter across batch boundaries (only when batch sizes match)
        pred_jitter_main = (
            torch.linalg.norm(main_pred_coords[:, 0] - prev_pred_main_coords, dim=-1)
            .sum()
            .item()
        )
        true_jitter_main = (
            torch.linalg.norm(main_true_coords[:, 0] - prev_true_main_coords, dim=-1)
            .sum()
            .item()
        )
        enhanced.total_pred_main_jitter += pred_jitter_main
        enhanced.total_true_main_jitter += true_jitter_main

        pred_jitter_c = (
            torch.linalg.norm(c_pred_coords[:, 0] - prev_pred_c_coords, dim=-1)
            .sum()
            .item()
        )
        true_jitter_c = (
            torch.linalg.norm(c_true_coords[:, 0] - prev_true_c_coords, dim=-1)
            .sum()
            .item()
        )
        enhanced.total_pred_c_jitter += pred_jitter_c
        enhanced.total_true_c_jitter += true_jitter_c
        enhanced.jitter_frames += B

    # Within-batch jitter
    if L > 1:
        pred_jitter_main = (
            torch.linalg.norm(
                main_pred_coords[:, 1:] - main_pred_coords[:, :-1], dim=-1
            )
            .sum()
            .item()
        )
        true_jitter_main = (
            torch.linalg.norm(
                main_true_coords[:, 1:] - main_true_coords[:, :-1], dim=-1
            )
            .sum()
            .item()
        )
        enhanced.total_pred_main_jitter += pred_jitter_main
        enhanced.total_true_main_jitter += true_jitter_main

        pred_jitter_c = (
            torch.linalg.norm(c_pred_coords[:, 1:] - c_pred_coords[:, :-1], dim=-1)
            .sum()
            .item()
        )
        true_jitter_c = (
            torch.linalg.norm(c_true_coords[:, 1:] - c_true_coords[:, :-1], dim=-1)
            .sum()
            .item()
        )
        enhanced.total_pred_c_jitter += pred_jitter_c
        enhanced.total_true_c_jitter += true_jitter_c
        enhanced.jitter_frames += B * (L - 1)

    # 3. Entropy
    probs_main = torch.softmax(logits_main, dim=-1)
    probs_c = torch.softmax(logits_c, dim=-1)
    entropy_main = -torch.sum(probs_main * torch.log(probs_main + 1e-9), dim=-1)
    entropy_c = -torch.sum(probs_c * torch.log(probs_c + 1e-9), dim=-1)
    enhanced.total_main_entropy += entropy_main.sum().item()
    enhanced.total_c_entropy += entropy_c.sum().item()

    if logits_shoulder is not None:
        probs_shoulder = torch.softmax(logits_shoulder, dim=-1)
        entropy_shoulder = -torch.sum(
            probs_shoulder * torch.log(probs_shoulder + 1e-9), dim=-1
        )
        enhanced.total_shoulder_entropy += entropy_shoulder.sum().item()

    enhanced.entropy_frames += B * L

    # 4. Per-action-state accuracy (vectorized)
    p1_actions = X[..., colmap.ego_action_idx].cpu().numpy().flatten()  # [B*L]
    correct_main = (pred_main_idx == target_main).cpu().numpy().flatten()  # [B*L]

    # Vectorize the categorization
    for action_val, is_correct in zip(p1_actions, correct_main):
        category = _categorize_action(int(action_val))
        enhanced.state_total[category] += 1
        if is_correct:
            enhanced.state_correct[category] += 1

    # 5. Controller state sequences for "stuck" duration (vectorized tuple creation)
    pred_main_flat = pred_main_idx.cpu().numpy().flatten().tolist()
    pred_c_flat = pred_c_idx.cpu().numpy().flatten().tolist()
    target_main_flat = target_main.cpu().numpy().flatten().tolist()
    target_c_flat = target_c.cpu().numpy().flatten().tolist()
    btn_pred_flat = btn_pred.cpu().numpy().reshape(-1, btn_pred.shape[-1])
    target_btn_flat = target_btn.cpu().numpy().reshape(-1, target_btn.shape[-1])

    for i in range(B * L):
        pred_state = (
            pred_main_flat[i],
            pred_c_flat[i],
            tuple(btn_pred_flat[i].tolist()),
        )
        true_state = (
            target_main_flat[i],
            target_c_flat[i],
            tuple(target_btn_flat[i].tolist()),
        )
        enhanced.pred_states.append(pred_state)
        enhanced.true_states.append(true_state)

    # 6. Button change latency tracking (for L/R button) - vectorized
    lr_idx = 4
    if L > 1:
        # Reshape to [B, L] for easier slicing
        target_lr = target_btn[:, :, lr_idx]  # [B, L]
        pred_lr = btn_pred[:, :, lr_idx]  # [B, L]

        # Find button press events (0 -> 1 transitions)
        true_changes = (target_lr[:, :-1] == 0) & (target_lr[:, 1:] == 1)  # [B, L-1]
        pred_changes = (pred_lr[:, :-1] == 0) & (pred_lr[:, 1:] == 1)  # [B, L-1]

        # Get frame indices (accounting for the batch structure)
        batch_offsets = torch.arange(B, device=target_lr.device).unsqueeze(1) * L
        frame_offsets = torch.arange(1, L, device=target_lr.device).unsqueeze(0)
        frame_indices = batch_offsets + frame_offsets + enhanced.total_frames

        true_change_frames = frame_indices[true_changes].cpu().tolist()
        pred_change_frames = frame_indices[pred_changes].cpu().tolist()

        enhanced.lr_button_changes_true.extend(true_change_frames)
        enhanced.lr_button_changes_pred.extend(pred_change_frames)

    # 7. Collect data for correlation matrix (vectorized)
    # Reshape everything to [B*L, features] and concatenate
    main_pred_flat = main_pred_coords.reshape(-1, 2).cpu().numpy()
    main_true_flat = main_true_coords.reshape(-1, 2).cpu().numpy()
    c_pred_flat = c_pred_coords.reshape(-1, 2).cpu().numpy()
    c_true_flat = c_true_coords.reshape(-1, 2).cpu().numpy()
    btn_pred_np = (
        btn_pred.reshape(-1, btn_pred.shape[-1]).cpu().numpy().astype(np.float32)
    )
    btn_true_np = target_btn.reshape(-1, target_btn.shape[-1]).cpu().numpy()

    pred_vecs = np.concatenate(
        [main_pred_flat, c_pred_flat, btn_pred_np], axis=1
    )  # [B*L, 2+2+5]
    true_vecs = np.concatenate(
        [main_true_flat, c_true_flat, btn_true_np], axis=1
    )  # [B*L, 2+2+5]

    enhanced.all_preds_list.append(pred_vecs)
    enhanced.all_labels_list.append(true_vecs)

    enhanced.total_frames += B * L

    # 8. Value head metrics (if available)
    if value_pred is not None:
        config = get_config()
        value_target = compute_value_targets(
            X, colmap, gamma=config.rl.gamma
        )  # [B, L, 1]
        frame_rewards = _compute_frame_rewards(X, colmap)  # [B, L]

        # MSE and MAE
        value_mse = ((value_pred - value_target) ** 2).mean().item()
        value_mae = (value_pred - value_target).abs().mean().item()

        enhanced.total_value_mse += value_mse * (B * L)
        enhanced.total_value_mae += value_mae * (B * L)
        enhanced.total_value_pred += value_pred.sum().item()
        enhanced.total_value_target += value_target.sum().item()

        # Store individual predictions and targets for correlation/distribution analysis
        value_pred_flat = value_pred.cpu().numpy().flatten().tolist()
        value_target_flat = value_target.cpu().numpy().flatten().tolist()
        enhanced.value_pred_list.extend(value_pred_flat)
        enhanced.value_target_list.extend(value_target_flat)
        enhanced.value_frames += B * L

        # Store frame-level data for detailed analysis (with context)
        # For each frame, store (value_pred, value_target, reward, frame_features)
        value_pred_np = value_pred.squeeze(-1).cpu().numpy()  # [B, L]
        value_target_np = value_target.squeeze(-1).cpu().numpy()  # [B, L]
        frame_rewards_np = frame_rewards.cpu().numpy()  # [B, L]
        X_np = X.cpu().numpy()  # [B, L, F]

        for b in range(B):
            for l in range(L):
                enhanced.value_frame_data.append(
                    (
                        float(value_pred_np[b, l]),
                        float(value_target_np[b, l]),
                        float(frame_rewards_np[b, l]),
                        X_np[b, l, :].copy(),  # Store full frame features
                    )
                )

    # Return last coordinates for next batch
    last_pred_main = main_pred_coords[:, -1]
    last_pred_c = c_pred_coords[:, -1]
    last_true_main = main_true_coords[:, -1]
    last_true_c = c_true_coords[:, -1]

    return last_pred_main, last_pred_c, last_true_main, last_true_c


def _update_running_metrics(
    metrics: RunningMetrics,
    target_info: Dict[str, torch.Tensor],
    pred_main_idx: torch.Tensor,
    pred_c_idx: torch.Tensor,
    btn_pred: torch.Tensor,
    btn_probs: torch.Tensor,
    logits_btn: torch.Tensor,
    logits_shoulder: Optional[torch.Tensor],
    change_stats_main: ChangeHoldStats,
    change_stats_c: ChangeHoldStats,
    change_stats_buttons: ChangeHoldStats,
) -> None:
    B, L = pred_main_idx.shape

    target_main = target_info["main_idx"].view(B, L)
    target_c = target_info["c_idx"].view(B, L)
    target_btn = target_info["buttons"]

    main_change_mask = torch.zeros_like(target_main, dtype=torch.bool)
    main_change_mask[:, 1:] = target_main[:, 1:] != target_main[:, :-1]
    main_hold_mask = ~main_change_mask
    main_hold_mask[:, 0] = True

    c_change_mask = torch.zeros_like(target_c, dtype=torch.bool)
    c_change_mask[:, 1:] = target_c[:, 1:] != target_c[:, :-1]
    c_hold_mask = ~c_change_mask
    c_hold_mask[:, 0] = True

    btn_change_mask = torch.zeros((B, L), dtype=torch.bool, device=target_btn.device)
    btn_change_mask[:, 1:] = torch.any(target_btn[:, 1:] != target_btn[:, :-1], dim=-1)
    btn_hold_mask = ~btn_change_mask
    btn_hold_mask[:, 0] = True

    main_major = metrics._majority_label(metrics.main_label_counts)
    c_major = metrics._majority_label(metrics.c_label_counts)

    rep_mask = torch.ones((B, L), dtype=torch.bool, device=target_main.device)
    rep_mask[:, 0] = False
    main_rep = torch.zeros_like(target_main)
    c_rep = torch.zeros_like(target_c)
    if L > 1:
        main_rep[:, 1:] = target_main[:, :-1]
        c_rep[:, 1:] = target_c[:, :-1]

    metrics.update_main(
        pred_main_idx.reshape(-1),
        target_main.reshape(-1),
        main_major,
        main_rep.reshape(-1),
        rep_mask.reshape(-1),
    )
    metrics.update_c(
        pred_c_idx.reshape(-1),
        target_c.reshape(-1),
        c_major,
        c_rep.reshape(-1),
        rep_mask.reshape(-1),
    )
    metrics.update_buttons(logits_btn, target_btn, btn_probs)

    if logits_shoulder is not None and target_info.get("shoulder_idx") is not None:
        sh_major = (
            metrics._majority_label(metrics.shoulder_label_counts)
            if metrics.K_shoulder
            else None
        )
        metrics.update_shoulder(logits_shoulder, target_info["shoulder_idx"], sh_major)

    correct_main = pred_main_idx == target_main
    correct_c = pred_c_idx == target_c
    correct_btn_em = (btn_pred == target_btn).all(dim=-1)

    change_stats_main.update(correct_main, main_change_mask, main_hold_mask)
    change_stats_c.update(correct_c, c_change_mask, c_hold_mask)
    change_stats_buttons.update(correct_btn_em, btn_change_mask, btn_hold_mask)


def _evaluate(
    model: GPT,
    loader: DataLoader,
    colmap: ColumnMap,
    device: torch.device,
    button_thresholds: torch.Tensor,
    progress: bool,
    report_every: int,
    max_batches: Optional[int] = None,
) -> Dict[str, object]:
    config = get_config()
    metrics = RunningMetrics(
        K_main=config.model.target_shapes_by_head["main_stick"],
        K_c=config.model.target_shapes_by_head["c_stick"],
        K_buttons=config.model.target_shapes_by_head["buttons"],
        K_shoulder=config.model.target_shapes_by_head["shoulder"],
        device=device,
    )

    pred_counts_main = torch.zeros(
        config.model.target_shapes_by_head["main_stick"], dtype=torch.long
    )
    pred_counts_c = torch.zeros(
        config.model.target_shapes_by_head["c_stick"], dtype=torch.long
    )
    pred_counts_shoulder = (
        torch.zeros(config.model.target_shapes_by_head["shoulder"], dtype=torch.long)
        if config.model.target_shapes_by_head["shoulder"] > 0
        else None
    )
    pred_button_presses = torch.zeros(
        config.model.target_shapes_by_head["buttons"], dtype=torch.long
    )

    raw_counts_main = torch.zeros_like(pred_counts_main)
    raw_counts_c = torch.zeros_like(pred_counts_c)
    raw_counts_shoulder = (
        torch.zeros_like(pred_counts_shoulder)
        if pred_counts_shoulder is not None
        else None
    )

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
        threshold_vector = button_thresholds.to(device=device, dtype=torch.float32)
        threshold_view = threshold_vector.view(1, 1, -1)

        for batch_idx, batch in enumerate(loader, start=1):
            if max_batches is not None and batch_idx > max_batches:
                break

            X: torch.Tensor = batch["X"].to(device, non_blocking=True)
            Y: torch.Tensor = batch["Y"].to(device, non_blocking=True)

            inputs_td = build_inputs_for_gpt(X, colmap)
            target_info = quantize_targets(Y, colmap, input_domain="unit11")

            pred = model(inputs_td)

            logits_main = pred["main_stick"]
            logits_c = pred["c_stick"]
            logits_btn = pred["buttons"]
            logits_shoulder = pred.get("shoulder")

            loss_components = compute_loss_components(
                pred,
                target_info,
                label_smoothing=config.train.label_smoothing,
                sample_weights=None,
            )

            for key, value in loss_components.items():
                loss_sums[key] += value.item()

            pred_main_idx = logits_main.argmax(dim=-1)
            pred_c_idx = logits_c.argmax(dim=-1)
            btn_probs = torch.sigmoid(logits_btn)
            btn_pred = (btn_probs > threshold_view).to(target_info["buttons"].dtype)

            _update_running_metrics(
                metrics,
                target_info,
                pred_main_idx,
                pred_c_idx,
                btn_pred,
                btn_probs,
                logits_btn,
                logits_shoulder,
                change_stats_main,
                change_stats_c,
                change_stats_buttons,
            )

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
            if pred_counts_shoulder is not None and logits_shoulder is not None:
                sh_pred = logits_shoulder.argmax(dim=-1)
                pred_counts_shoulder += torch.bincount(
                    sh_pred.reshape(-1).cpu(), minlength=pred_counts_shoulder.shape[0]
                )

            pred_button_presses += btn_pred.sum(dim=(0, 1)).cpu().to(torch.long)

            Y_cpu = batch["Y"].detach().cpu()
            main_vals = Y_cpu[..., colmap.y_main]
            main_idx_raw = _assign_to_palette(main_vals.reshape(-1, 2), _MAIN_PALETTE_T)
            raw_counts_main += torch.bincount(
                main_idx_raw.cpu(), minlength=raw_counts_main.shape[0]
            )

            c_vals = Y_cpu[..., colmap.y_c]
            c_idx_raw = _assign_to_palette(c_vals.reshape(-1, 2), _C_PALETTE_T)
            raw_counts_c += torch.bincount(
                c_idx_raw.cpu(), minlength=raw_counts_c.shape[0]
            )

            if raw_counts_shoulder is not None and colmap.y_shoulder is not None:
                s_vals = torch.clamp(
                    Y_cpu[..., colmap.y_shoulder].to(torch.float32), 0.0, 1.0
                )
                s_vals = s_vals.reshape(-1)
                s_idx = torch.searchsorted(
                    _SHOULDER_PALETTE_T, s_vals, right=True
                ) - 1
                s_idx = torch.clamp(s_idx, min=0, max=_SHOULDER_PALETTE_T.shape[0] - 1)
                raw_counts_shoulder += torch.bincount(
                    s_idx.cpu(), minlength=raw_counts_shoulder.shape[0]
                )

            total_tokens += int(X.numel())
            total_frames += int(X.shape[0] * X.shape[1])

            if progress:
                running_loss = loss_sums["total"] / batch_idx
                summary = metrics.summary()
                pct = (
                    (batch_idx / total_batches) * 100.0
                    if total_batches
                    else float("nan")
                )
                line = (
                    f"[{batch_idx}/{total_batches if total_batches else '?'} | {pct:5.1f}%] "
                    f"loss {running_loss:.4f} | main acc {summary['acc_main']:.3f} "
                    f"(chg {change_stats_main.change_acc():.3f} hold {change_stats_main.hold_acc():.3f}) | "
                    f"c acc {summary['acc_c']:.3f} (chg {change_stats_c.change_acc():.3f} hold {change_stats_c.hold_acc():.3f}) | "
                    f"btn EM {summary['btn_em']:.3f} (chg {change_stats_buttons.change_acc():.3f} hold {change_stats_buttons.hold_acc():.3f})"
                )
                acc_shoulder = summary.get("acc_shoulder")
                if acc_shoulder is not None:
                    line += f" | shoulder acc {acc_shoulder:.3f}"
                print(line, flush=True)

                if report_every and batch_idx % report_every == 0:
                    _print_intermediate_summary(
                        metrics, change_stats_main, change_stats_c, change_stats_buttons
                    )

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
    results["button_thresholds"] = threshold_vector.cpu().tolist()
    results["enhanced"] = enhanced
    return results


def _print_intermediate_summary(
    metrics: RunningMetrics,
    main_stats: ChangeHoldStats,
    c_stats: ChangeHoldStats,
    btn_stats: ChangeHoldStats,
) -> None:
    summary = metrics.summary()
    print("Interim summary:")
    print(
        f"  Main acc {summary['acc_main']:.3f} | maj {summary['acc_main_maj']:.3f} | rep {summary['acc_main_rep']:.3f}"
    )
    print(
        f"    change {main_stats.change_acc():.3f} | hold {main_stats.hold_acc():.3f}"
    )
    print(
        f"  C acc {summary['acc_c']:.3f} | maj {summary['acc_c_maj']:.3f} | rep {summary['acc_c_rep']:.3f}"
    )
    print(f"    change {c_stats.change_acc():.3f} | hold {c_stats.hold_acc():.3f}")
    print(
        f"  Buttons EM {summary['btn_em']:.3f} | F1μ {summary['btn_f1_micro']:.3f} | F1_macro {summary['btn_f1_macro']:.3f}"
    )
    print(
        f"    change {btn_stats.change_acc():.3f} | hold {btn_stats.hold_acc():.3f}",
        flush=True,
    )


def _render_button_metrics(metrics: RunningMetrics) -> str:
    total_frames = (
        float(metrics.btn_total.item())
        if isinstance(metrics.btn_total, torch.Tensor)
        else float(metrics.btn_total)
    )
    tp = metrics.btn_tp.detach().cpu().numpy()
    fp = metrics.btn_fp.detach().cpu().numpy()
    fn = metrics.btn_fn.detach().cpu().numpy()
    pos_counts = metrics.btn_pos_counts.detach().cpu().numpy()
    neg_counts = total_frames - pos_counts
    tn = np.maximum(neg_counts - fp, 0.0)

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    accuracy = np.divide(
        tp + tn, total_frames, out=np.zeros_like(tp), where=total_frames > 0
    )
    pos_rate = np.divide(
        pos_counts, total_frames, out=np.zeros_like(tp), where=total_frames > 0
    )

    lines = ["Per-button metrics:"]
    button_names = CONTROLLER_KEY_GROUPS["buttons"]
    for idx, name in enumerate(button_names):
        label = _BUTTON_PRETTY.get(name, name.upper())
        lines.append(
            f"  {label:<6} acc {accuracy[idx]:.3f} | prec {precision[idx]:.3f} | rec {recall[idx]:.3f} | f1 {f1[idx]:.3f} | pos_rate {pos_rate[idx]:.3f}"
        )
    return "\n".join(lines)


def _assign_to_palette(
    values: torch.Tensor, palette: torch.Tensor, *, from_unit_square: bool = False
) -> torch.Tensor:
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=values.device)
    vals = values.to(torch.float32)
    if from_unit_square:
        vals = vals * 2.0 - 1.0
    vals = torch.clamp(vals, -1.0, 1.0)
    radius = torch.linalg.norm(vals, dim=-1, keepdim=True)
    vals = vals / torch.clamp(radius, min=1.0)
    norms = (vals**2).sum(dim=-1, keepdim=True)
    palette = palette.to(vals.device)
    palette_norm = (palette**2).sum(dim=-1).unsqueeze(0)
    dot = vals @ palette.t()
    d2 = norms - 2.0 * dot + palette_norm
    return d2.argmin(dim=-1)


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
        lines.append(f"  … {remaining} additional bins suppressed")
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
    print("\n1. Mean Stick Error (Euclidean Distance):")
    avg_main_error = _safe_div(enhanced.total_main_stick_error, enhanced.total_frames)
    avg_c_error = _safe_div(enhanced.total_c_stick_error, enhanced.total_frames)
    print(f"  Main Stick: {avg_main_error:.4f}")
    print(f"  C-Stick:    {avg_c_error:.4f}")

    # Error by change/hold
    change_frames_main = enhanced.total_main_stick_error_change
    hold_frames_main = enhanced.total_main_stick_error_hold
    # We need to track the count of change/hold frames for proper averaging
    # For now, use the total as denominator (this is an approximation)
    print(
        f"  Main Stick (change frames): {_safe_div(enhanced.total_main_stick_error_change, enhanced.total_frames):.4f}"
    )
    print(
        f"  Main Stick (hold frames):   {_safe_div(enhanced.total_main_stick_error_hold, enhanced.total_frames):.4f}"
    )
    print(
        f"  C-Stick (change frames):    {_safe_div(enhanced.total_c_stick_error_change, enhanced.total_frames):.4f}"
    )
    print(
        f"  C-Stick (hold frames):      {_safe_div(enhanced.total_c_stick_error_hold, enhanced.total_frames):.4f}"
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
    print("\n4. Accuracy by Player State:")
    for state_type in ["idle", "hitstun", "attack", "other"]:
        if state_type in enhanced.state_total and enhanced.state_total[state_type] > 0:
            acc = _safe_div(
                enhanced.state_correct[state_type], enhanced.state_total[state_type]
            )
            count = int(enhanced.state_total[state_type])
            print(f"  {state_type.capitalize():<8}: {acc:.3f} ({count:,} frames)")

    # 5. Stuck action duration
    print("\n5. 'Stuck' Action Duration (Consecutive Identical States):")
    if enhanced.pred_states and enhanced.true_states:
        pred_runs = _get_run_lengths(enhanced.pred_states)
        true_runs = _get_run_lengths(enhanced.true_states)
        if pred_runs and true_runs:
            print(f"  Predicted:")
            print(f"    Mean: {np.mean(pred_runs):.2f} frames")
            print(f"    Max:  {np.max(pred_runs)} frames")
            print(f"  Ground Truth:")
            print(f"    Mean: {np.mean(true_runs):.2f} frames")
            print(f"    Max:  {np.max(true_runs)} frames")

    # 6. Action change latency
    print("\n6. L/R Button Press Latency:")
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

    # 7. Correlation matrix
    print("\n7. Controller Input Correlation Matrix:")
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

        # You could optionally save the matrices for visualization
        # import matplotlib.pyplot as plt
        # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # axes[0].imshow(pred_corr, cmap='coolwarm', vmin=-1, vmax=1)
        # axes[0].set_title('Predicted Correlations')
        # axes[1].imshow(true_corr, cmap='coolwarm', vmin=-1, vmax=1)
        # axes[1].set_title('True Correlations')
        # plt.savefig('correlation_matrices.png')

    # 8. Value Head Metrics (RL)
    if enhanced.value_frames > 0:
        print("\n8. Value Head Metrics (RL Critic):")

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
        _print_extreme_value_frames(enhanced, colmap, top_k=1)


def _print_final_summary(
    results: Dict[str, object],
    checkpoint: Path,
    data_root: Path,
    colmap: ColumnMap,
) -> None:
    batches = results["batches"]
    loss_sums: Dict[str, float] = results["loss_sums"]
    elapsed = results["elapsed"]
    total_frames = results["frames"]
    total_tokens = results["tokens"]
    metrics: RunningMetrics = results["metrics"]
    main_stats: ChangeHoldStats = results["main_change_stats"]
    c_stats: ChangeHoldStats = results["c_change_stats"]
    btn_stats: ChangeHoldStats = results["btn_change_stats"]
    main_pred_counts = results["pred_counts_main"].cpu().numpy()
    c_pred_counts = results["pred_counts_c"].cpu().numpy()
    btn_press_counts = results["pred_button_presses"].cpu().numpy()
    shoulder_pred_counts = None
    if results.get("pred_counts_shoulder") is not None:
        shoulder_pred_counts = results["pred_counts_shoulder"].cpu().numpy()

    main_true_counts = results["raw_counts_main"].cpu().numpy()
    c_true_counts = results["raw_counts_c"].cpu().numpy()
    shoulder_true_counts = (
        results["raw_counts_shoulder"].cpu().numpy()
        if results.get("raw_counts_shoulder") is not None
        else None
    )
    btn_true_press_counts = metrics.btn_pos_counts.detach().cpu().numpy()
    button_thresholds = results.get("button_thresholds", [])

    avg_loss = {
        k: (v / batches if batches else float("nan")) for k, v in loss_sums.items()
    }
    per_frame_loss = {
        k: (v / total_frames if total_frames else float("nan"))
        for k, v in loss_sums.items()
    }

    summary = metrics.summary()

    print("\n===== Validation Summary =====")
    print(f"Checkpoint: {checkpoint}")
    print(f"Dataset:    {data_root}")
    print(f"Batches:    {batches}")
    print(f"Frames:     {total_frames:,}")
    print(f"Tokens:     {total_tokens:,}")
    print(
        f"Elapsed:    {elapsed:.2f}s | {total_tokens / max(elapsed, 1e-9):,.0f} tokens/s"
    )
    if button_thresholds:
        formatted_thr = ", ".join(f"{thr:.3f}" for thr in button_thresholds)
        print(f"Button thresholds: [{formatted_thr}]")

    print("\nLoss (per batch):")
    for key in ("total", "main", "c", "buttons", "shoulder"):
        print(f"  {key:>8}: {avg_loss[key]:.6f}")

    print("\nLoss (per frame):")
    for key in ("total", "main", "c", "buttons", "shoulder"):
        print(f"  {key:>8}: {per_frame_loss[key]:.8f}")

    print("\nMain Stick:")
    print(
        f"  accuracy {summary['acc_main']:.3f} | majority {summary['acc_main_maj']:.3f} | repeat {summary['acc_main_rep']:.3f}"
    )
    print(
        f"  change   {main_stats.change_acc():.3f} | hold {main_stats.hold_acc():.3f}"
    )

    print("\nC-Stick:")
    print(
        f"  accuracy {summary['acc_c']:.3f} | majority {summary['acc_c_maj']:.3f} | repeat {summary['acc_c_rep']:.3f}"
    )
    print(f"  change   {c_stats.change_acc():.3f} | hold {c_stats.hold_acc():.3f}")

    print("\nButtons:")
    print(
        f"  EM {summary['btn_em']:.3f} | F1μ {summary['btn_f1_micro']:.3f} | F1_macro {summary['btn_f1_macro']:.3f} | "
        f"maj_EM {summary['btn_em_maj']:.3f} | rep_EM {summary['btn_em_rep']:.3f}"
    )
    print(f"  change {btn_stats.change_acc():.3f} | hold {btn_stats.hold_acc():.3f}")
    print(_render_button_metrics(metrics))

    acc_shoulder = summary.get("acc_shoulder")
    if acc_shoulder is not None:
        print("\nShoulder:")
        print(
            f"  accuracy {acc_shoulder:.3f} | majority {summary['acc_shoulder_maj']:.3f}"
        )

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

    if shoulder_pred_counts is not None and shoulder_true_counts is not None:
        print("  Shoulder:")
        print(
            _format_distribution_comparison(
                shoulder_pred_counts, shoulder_true_counts, _SHOULDER_LABELS, top_k=None
            )
        )

    print("  Button press rates:")
    _print_button_press_distribution(
        btn_press_counts, btn_true_press_counts, total_frames
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
        _print_enhanced_metrics_with_colmap(enhanced, colmap)


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
        "--data-root",
        type=Path,
        default=None,
        help="Root directory of the validation Zarr dataset (defaults to validation_set under project root).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation batch size (defaults to train.batch_size).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader workers (defaults to train.num_workers).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=None,
        help="Prefetch factor when num_workers>0 (defaults to train.prefetch_factor).",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable DataLoader pin_memory (enabled by default).",
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
        "--thresholds-path",
        type=Path,
        default=None,
        help="Path to per-button threshold JSON (defaults to button_thresholds.json if present).",
    )
    return parser.parse_args()


def main() -> None:
    init_config()
    config = get_config()
    args = parse_args()

    project_root = Path(__file__).resolve().parent

    data_root = args.data_root
    if data_root is None:
        data_root = project_root / "validation_set"
    data_root = _ensure_absolute(data_root, project_root)
    if not data_root.exists():
        print(f"ERROR: validation data root {data_root} not found", file=sys.stderr)
        sys.exit(1)

    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        ckpt_dir = Path(config.train.out_dir)
        ckpt_dir = _ensure_absolute(ckpt_dir, project_root)
        checkpoint_path = _latest_checkpoint(ckpt_dir)
        if checkpoint_path is None:
            print(f"ERROR: no checkpoints found in {ckpt_dir}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = _ensure_absolute(checkpoint_path, Path.cwd())

    device = _resolve_device()

    batch_size = args.batch_size or config.train.batch_size
    num_workers = (
        args.num_workers if args.num_workers is not None else config.train.num_workers
    )
    prefetch_factor = (
        args.prefetch_factor
        if args.prefetch_factor is not None
        else config.train.prefetch_factor
    )
    pin_memory = config.train.pin_memory and not args.no_pin_memory
    persistent_workers = config.train.persistent_workers and num_workers > 0

    loader, dataset = _prepare_dataloader(
        data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
    )

    colmap = ColumnMap.from_dataset(dataset)

    model = GPT()
    # TODO: use loading from checkpoint.py
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)

    if "optimizer" in ckpt:
        del ckpt["optimizer"]
    if "scaler" in ckpt:
        del ckpt["scaler"]

    threshold_path = args.thresholds_path or _DEFAULT_THRESHOLD_PATH
    threshold_path = _ensure_absolute(threshold_path, project_root)
    threshold_values = _load_saved_button_thresholds(threshold_path)
    if threshold_values is None:
        threshold_values = [0.5] * len(CONTROLLER_KEY_GROUPS["buttons"])
        print("No threshold file found; using 0.50 for all buttons")
    else:
        print(f"Loaded button thresholds from {threshold_path}")

    button_thresholds = torch.tensor(threshold_values, dtype=torch.float32)

    print(
        f"Evaluating on {len(dataset):,} windows with batch size {batch_size} (device={device})"
    )

    results = _evaluate(
        model,
        loader,
        colmap,
        device,
        button_thresholds,
        progress=not args.no_progress,
        report_every=args.report_every,
        max_batches=args.max_batches,
    )

    _print_final_summary(results, checkpoint_path, data_root, colmap)


if __name__ == "__main__":
    main()
