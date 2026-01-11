#!/usr/bin/env python3
"""
Analyze SAE features to find interpretable concepts.

For each SAE feature, computes:
1. Input feature correlations - what game state activates it?
2. Action state analysis - mean activation per action (categorical, not correlation)
3. Temporal patterns - what happens before/after feature fires?
4. Max-activating examples - for manual inspection

Usage:
    # Analyze all features in a trained SAE
    python scripts/analyze_sae_features.py \
        --sae sae_checkpoints/block_output_L4/sae_final.pt \
        --checkpoint checkpoints/model.pt

    # Focus on specific features
    python scripts/analyze_sae_features.py \
        --sae sae_checkpoints/block_output_L4/sae_final.pt \
        --checkpoint checkpoints/model.pt \
        --features 42 17 93

    # Export detailed report
    python scripts/analyze_sae_features.py \
        --sae sae_checkpoints/block_output_L4/sae_final.pt \
        --checkpoint checkpoints/model.pt \
        --output-dir feature_reports/
"""

from __future__ import annotations

# Suppress noisy warnings before any imports
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
warnings.filterwarnings("ignore", message=".*pin_memory.*MPS.*")

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def _format_time(seconds: float) -> str:
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        return str(timedelta(seconds=int(seconds)))


def _log(msg: str, flush: bool = True) -> None:
    """Print with timestamp and flush."""
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=flush)

from column_map import ColumnMap
from config import Config, init_config_from_checkpoint
from interp import HookPoint, HookPointType, TopKSparseAutoencoder
from interp.hooks import HookManager
from model.nano_gpt import GPT
from train.batch_utils import build_model_inputs
from utils import _resolve_device, match_state_dict_keys
from window_dataset import make_dataloader

# Global action state name lookup (loaded from library/action_state.json)
_ACTION_STATE_NAMES: Dict[int, str] = {}


def _load_action_state_names() -> Dict[int, str]:
    """Load action state names from library/action_state.json."""
    global _ACTION_STATE_NAMES
    if _ACTION_STATE_NAMES:
        return _ACTION_STATE_NAMES

    # Try to load from library
    library_path = Path(__file__).parent.parent / "library" / "action_state.json"
    if library_path.exists():
        with open(library_path) as f:
            data = json.load(f)

        # Load Common states
        if "Common" in data and "known_values" in data["Common"]:
            for action_id, info in data["Common"]["known_values"].items():
                _ACTION_STATE_NAMES[int(action_id)] = info.get("ident", f"Action_{action_id}")

        # Load Fox-specific states (since most training data is Fox)
        if "Fox" in data and "known_values" in data["Fox"]:
            for action_id, info in data["Fox"]["known_values"].items():
                aid = int(action_id)
                # Only add if not already present (Common takes precedence for shared IDs)
                if aid not in _ACTION_STATE_NAMES:
                    _ACTION_STATE_NAMES[aid] = f"Fox_{info.get('ident', action_id)}"

        print(f"Loaded {len(_ACTION_STATE_NAMES)} action state names from library")
    else:
        # Fallback to stats/utils/melee_constants
        try:
            from stats.utils.melee_constants import ACTION_STATE_NAMES
            _ACTION_STATE_NAMES = dict(ACTION_STATE_NAMES)
            print(f"Loaded {len(_ACTION_STATE_NAMES)} action state names from melee_constants")
        except ImportError:
            print("Warning: Could not load action state names")

    return _ACTION_STATE_NAMES


@dataclass
class ActionAnalysis:
    """Analysis of feature activation by action state."""
    # Mean activation when in this action (action_id -> mean_activation)
    ego_action_means: Dict[int, float] = field(default_factory=dict)
    opp_action_means: Dict[int, float] = field(default_factory=dict)
    # Top actions by mean activation
    top_ego_actions: List[Tuple[int, str, float]] = field(default_factory=list)  # (id, name, mean)
    top_opp_actions: List[Tuple[int, str, float]] = field(default_factory=list)


@dataclass
class TemporalPattern:
    """Temporal patterns around feature activation."""
    # Actions that commonly occur at each offset (negative = before, positive = after)
    # offset -> [(action_id, name, prob)]
    actions_at_offset: Dict[int, List[Tuple[int, str, float]]] = field(default_factory=dict)

    # Input features that are significantly different at each offset compared to baseline
    # offset -> [(feature_name, mean_when_firing, baseline_mean, z_score)]
    features_at_offset: Dict[int, List[Tuple[str, float, float, float]]] = field(default_factory=dict)

    # Legacy compatibility
    @property
    def actions_before(self) -> Dict[int, List[Tuple[int, str, float]]]:
        return {k: v for k, v in self.actions_at_offset.items() if k < 0}

    @property
    def actions_after(self) -> Dict[int, List[Tuple[int, str, float]]]:
        return {k: v for k, v in self.actions_at_offset.items() if k > 0}


@dataclass
class FeatureStats:
    """Statistics for a single SAE feature."""

    feature_idx: int

    # Activation statistics
    activation_frequency: float  # How often does it fire?
    mean_activation: float  # Mean value when active
    max_activation: float  # Maximum activation seen

    # Top correlated input features (excluding action which is categorical)
    top_input_correlations: List[Tuple[str, float]]  # (feature_name, correlation)

    # Action state analysis (proper categorical handling)
    action_analysis: ActionAnalysis = field(default_factory=ActionAnalysis)

    # Temporal patterns
    temporal_pattern: Optional[TemporalPattern] = None

    # Max activating frame indices (for inspection)
    max_activating_indices: List[int] = field(default_factory=list)


@dataclass
class FeatureAnalysisReport:
    """Full analysis report for multiple features."""

    sae_path: str
    hook_point: str
    n_features_total: int
    n_features_analyzed: int
    n_dead_features: int

    # Per-feature stats
    feature_stats: Dict[int, FeatureStats] = field(default_factory=dict)

    # Summary statistics
    mean_activation_frequency: float = 0.0
    features_by_frequency: List[int] = field(default_factory=list)  # Sorted by freq


def load_sae_and_model(
    sae_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    data_dir: Optional[Path] = None,
) -> Tuple[TopKSparseAutoencoder, GPT, Config, HookPoint]:
    """Load SAE, model, and determine hook point from metadata."""
    # Load SAE
    sae = TopKSparseAutoencoder.load(str(sae_path), device=device)

    # Load metadata to get hook point
    metadata_path = sae_path.parent / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        hook_type = HookPointType(metadata["hook_type"])
        layer_idx = metadata.get("layer_idx")
        hook_point = HookPoint(hook_type, layer_idx)
    else:
        # Default to block output at middle layer
        print("Warning: No metadata.json found, defaulting to block_output at layer 4")
        hook_point = HookPoint(HookPointType.BLOCK_OUTPUT, layer_idx=4)

    # Load model and config
    overrides = {}
    if data_dir is not None:
        # Parse data_dir to extract episode count suffix (e.g., processed_data_1 -> 1)
        import re
        data_dir_str = str(data_dir)
        match = re.search(r"_(\d+)$", data_dir_str)
        if match:
            episode_count = int(match.group(1))
            base_path = data_dir_str[: match.start()]
            overrides["zarr.out_root"] = base_path
            overrides["zarr.episode_count"] = str(episode_count)
        else:
            overrides["zarr.out_root"] = data_dir_str

    config = init_config_from_checkpoint(checkpoint_path, overrides=overrides if overrides else None)

    model = GPT(config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # Handle different checkpoint structures
    state_dict = ckpt.get("model") or ckpt.get("model_state_dict")
    if state_dict is None:
        raise ValueError(f"Checkpoint has no 'model' or 'model_state_dict' key: {list(ckpt.keys())}")

    # Filter to only transformer layers (skip output heads which may have different architecture)
    # We only need the transformer layers for SAE analysis
    # Handle _orig_mod. prefix from torch.compile
    transformer_prefixes = ("blocks.", "ln_f.", "embed.", "wte.", "wpe.", "input_proj", "projection_down.")
    transformer_keys = {}
    for k, v in state_dict.items():
        # Strip _orig_mod. prefix if present
        clean_key = k.replace("_orig_mod.", "")
        if clean_key.startswith(transformer_prefixes):
            transformer_keys[k] = v
    _log(f"Loading {len(transformer_keys)} transformer layer parameters (skipping output heads)")

    model_state = match_state_dict_keys(transformer_keys, model)
    # Use strict=False since we're only loading transformer layers
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    head_keys = [k for k in missing if "head" in k.lower()]
    non_head_missing = [k for k in missing if "head" not in k.lower()]
    if non_head_missing:
        _log(f"Warning: Missing {len(non_head_missing)} non-head keys: {non_head_missing[:5]}")
    model = model.to(device)
    model.eval()

    return sae, model, config, hook_point


def get_action_name(action_id: int) -> str:
    """Get human-readable name for an action state."""
    names = _load_action_state_names()
    return names.get(action_id, f"Action_{action_id}")


def collect_feature_activations(
    model: GPT,
    sae: TopKSparseAutoencoder,
    hook_point: HookPoint,
    dataloader,
    colmap: ColumnMap,
    device: torch.device,
    max_samples: int = 50000,
    batch_size: int = 32,
    seq_len: int = 256,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Collect SAE feature activations along with input features.

    All computations done on GPU for speed.

    Returns:
        feature_activations: [N, hidden_dim] SAE feature activations (GPU)
        input_features: [N, input_dim] Raw input features (GPU)
        batch_seq_info: [N, 2] batch_idx and seq_pos for temporal analysis (GPU)
    """
    hook_manager = HookManager(model)
    hook_manager.install_hooks([hook_point])

    # Pre-allocate on GPU
    n_batches_needed = (max_samples // (batch_size * seq_len)) + 1

    all_activations = []
    all_inputs = []
    all_batch_seq = []
    total_samples = 0
    batch_idx = 0

    _log(f"Collecting SAE activations on {device} (target: {max_samples:,})...")
    start_time = time.time()
    last_log_time = start_time

    exclude_p1 = getattr(model.config.model, 'exclude_p1_controller', False)

    try:
        with torch.no_grad():
            for batch in dataloader:
                X = batch["X"].to(device)
                B, L, _ = X.shape

                # Build model inputs and run forward pass
                inputs_td = build_model_inputs(X, colmap, exclude_p1_controller=exclude_p1)
                _ = model(inputs_td)

                # Get activations from hook
                acts = hook_manager.get_single(hook_point)
                # Flatten batch and sequence dimensions
                acts_flat = acts.view(-1, acts.shape[-1])  # [B*L, d_model]

                # Get SAE feature activations (keep on GPU)
                sae_acts, _, _ = sae.encode(acts_flat)  # [B*L, hidden_dim]

                # Store on GPU
                all_activations.append(sae_acts)
                all_inputs.append(X.view(-1, X.shape[-1]))

                # Track batch/sequence position for temporal analysis
                batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
                seq_positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
                batch_seq = torch.stack([batch_indices.flatten() + batch_idx * B,
                                        seq_positions.flatten()], dim=1)
                all_batch_seq.append(batch_seq)

                hook_manager.clear()
                n_frames = B * L
                total_samples += n_frames
                batch_idx += 1

                # Log every 5 seconds or every 10 batches
                now = time.time()
                if now - last_log_time >= 5.0 or batch_idx % 10 == 0:
                    elapsed = now - start_time
                    rate = total_samples / elapsed if elapsed > 0 else 0
                    remaining = (max_samples - total_samples) / rate if rate > 0 else 0
                    pct = 100.0 * total_samples / max_samples
                    _log(f"  Collected {total_samples:,}/{max_samples:,} ({pct:.1f}%) - "
                         f"{rate:.0f} samples/s - ETA: {_format_time(remaining)}")
                    last_log_time = now

                if total_samples >= max_samples:
                    break

    finally:
        hook_manager.remove_hooks()

    feature_activations = torch.cat(all_activations, dim=0)[:max_samples]
    input_features = torch.cat(all_inputs, dim=0)[:max_samples]
    batch_seq_info = torch.cat(all_batch_seq, dim=0)[:max_samples]

    elapsed = time.time() - start_time
    _log(f"  Collection complete: {len(feature_activations):,} samples in {_format_time(elapsed)}")

    return feature_activations, input_features, batch_seq_info


def precompute_action_means(
    feature_activations: Tensor,  # [N, hidden_dim]
    ego_actions: Tensor,          # [N]
    opp_actions: Tensor,          # [N]
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Precompute mean activation per action for ALL features at once.

    Returns:
        ego_unique: unique ego action IDs
        ego_means: [n_ego_actions, hidden_dim] mean activation per ego action per feature
        ego_counts: [n_ego_actions] count per ego action
        opp_unique, opp_means, opp_counts: same for opponent
    """
    device = feature_activations.device
    hidden_dim = feature_activations.shape[1]

    # Ego actions
    ego_unique = ego_actions.unique()
    n_ego = len(ego_unique)
    ego_means = torch.zeros(n_ego, hidden_dim, device=device)
    ego_counts = torch.zeros(n_ego, device=device)

    for i, action_id in enumerate(ego_unique):
        mask = ego_actions == action_id
        count = mask.sum()
        ego_counts[i] = count
        if count > 0:
            ego_means[i] = feature_activations[mask].mean(dim=0)

    # Opp actions
    opp_unique = opp_actions.unique()
    n_opp = len(opp_unique)
    opp_means = torch.zeros(n_opp, hidden_dim, device=device)
    opp_counts = torch.zeros(n_opp, device=device)

    for i, action_id in enumerate(opp_unique):
        mask = opp_actions == action_id
        count = mask.sum()
        opp_counts[i] = count
        if count > 0:
            opp_means[i] = feature_activations[mask].mean(dim=0)

    return ego_unique, ego_means, ego_counts, opp_unique, opp_means, opp_counts


def precompute_correlations(
    feature_activations: Tensor,  # [N, hidden_dim]
    input_features: Tensor,       # [N, input_dim]
    feature_names: List[str],
) -> Tensor:
    """
    Precompute correlation of all SAE features with all input features.

    Returns:
        correlations: [hidden_dim, input_dim] correlation matrix
    """
    # Skip categorical features
    skip_patterns = ['action', 'character', 'stage']
    valid_cols = []
    for i, name in enumerate(feature_names):
        if i >= input_features.shape[1]:
            break
        if not any(pat in name.lower() for pat in skip_patterns):
            valid_cols.append(i)

    valid_cols = torch.tensor(valid_cols, device=input_features.device)
    input_subset = input_features[:, valid_cols]  # [N, n_valid]

    # Standardize
    feat_mean = feature_activations.mean(dim=0, keepdim=True)
    feat_std = feature_activations.std(dim=0, keepdim=True).clamp(min=1e-8)
    feat_norm = (feature_activations - feat_mean) / feat_std

    inp_mean = input_subset.mean(dim=0, keepdim=True)
    inp_std = input_subset.std(dim=0, keepdim=True).clamp(min=1e-8)
    inp_norm = (input_subset - inp_mean) / inp_std

    # Correlation = mean of element-wise product of standardized values
    # [hidden_dim, n_valid] = [hidden_dim, N] @ [N, n_valid] / N
    correlations = (feat_norm.T @ inp_norm) / feature_activations.shape[0]

    return correlations, valid_cols


def analyze_action_states_fast(
    feature_idx: int,
    ego_unique: Tensor,
    ego_means: Tensor,      # [n_ego, hidden_dim]
    ego_counts: Tensor,
    opp_unique: Tensor,
    opp_means: Tensor,      # [n_opp, hidden_dim]
    opp_counts: Tensor,
    baseline: float,
    n_top: int = 10,
    min_count: int = 10,
) -> ActionAnalysis:
    """Fast action analysis using precomputed means."""
    analysis = ActionAnalysis()

    # Get means for this feature
    ego_feat_means = ego_means[:, feature_idx]
    opp_feat_means = opp_means[:, feature_idx]

    # Filter by count and build dicts
    ego_valid = ego_counts > min_count
    opp_valid = opp_counts > min_count

    ego_action_means = {
        int(ego_unique[i]): ego_feat_means[i].item()
        for i in range(len(ego_unique)) if ego_valid[i]
    }
    opp_action_means = {
        int(opp_unique[i]): opp_feat_means[i].item()
        for i in range(len(opp_unique)) if opp_valid[i]
    }

    analysis.ego_action_means = ego_action_means
    analysis.opp_action_means = opp_action_means

    # Top actions
    ego_sorted = sorted(ego_action_means.items(), key=lambda x: x[1], reverse=True)
    analysis.top_ego_actions = [
        (aid, get_action_name(aid), mean)
        for aid, mean in ego_sorted[:n_top]
        if mean > baseline * 1.2
    ]

    opp_sorted = sorted(opp_action_means.items(), key=lambda x: x[1], reverse=True)
    analysis.top_opp_actions = [
        (aid, get_action_name(aid), mean)
        for aid, mean in opp_sorted[:n_top]
        if mean > baseline * 1.2
    ]

    return analysis


def analyze_temporal_patterns(
    feature_acts: Tensor,      # [N] activations for one feature
    ego_actions: Tensor,       # [N] ego action IDs
    opp_actions: Tensor,       # [N] opponent action IDs
    batch_seq_info: Tensor,    # [N, 2] batch_idx and seq_pos
    input_features: Optional[Tensor] = None,  # [N, input_dim] all input features
    feature_names: Optional[List[str]] = None,
    threshold_percentile: float = 90,
    offsets: List[int] = [-60, -30, -15, -5, 5, 15, 30, 60],
    n_top_actions: int = 5,
    n_top_features: int = 8,
) -> TemporalPattern:
    """
    Analyze what happens before and after a feature fires.

    This finds frames where the feature fires strongly, then looks at
    what actions and input features occur at various temporal offsets.
    """
    # Compute threshold here (slow path, for backwards compatibility)
    threshold = torch.quantile(feature_acts, threshold_percentile / 100.0)
    if threshold <= 0:
        threshold = feature_acts.max() * 0.5
    seq_len = int(batch_seq_info[:, 1].max().item()) + 1

    return analyze_temporal_patterns_fast(
        feature_acts, ego_actions, opp_actions, batch_seq_info,
        input_features=input_features,
        feature_names=feature_names,
        threshold=threshold.item() if isinstance(threshold, Tensor) else threshold,
        seq_len=seq_len,
        offsets=offsets,
        n_top_actions=n_top_actions,
        n_top_features=n_top_features,
    )


def analyze_temporal_patterns_fast(
    feature_acts: Tensor,      # [N] activations for one feature
    ego_actions: Tensor,       # [N] ego action IDs
    opp_actions: Tensor,       # [N] opponent action IDs
    batch_seq_info: Tensor,    # [N, 2] batch_idx and seq_pos
    threshold: float,          # Pre-computed threshold
    seq_len: int,              # Pre-computed sequence length
    input_features: Optional[Tensor] = None,  # [N, input_dim] all input features
    feature_names: Optional[List[str]] = None,
    baseline_means: Optional[Tensor] = None,  # [input_dim] precomputed baseline means
    baseline_stds: Optional[Tensor] = None,   # [input_dim] precomputed baseline stds
    offsets: List[int] = [-60, -30, -15, -5, 5, 15, 30, 60],
    n_top_actions: int = 5,
    n_top_features: int = 8,
) -> TemporalPattern:
    """
    Fast temporal pattern analysis with pre-computed threshold and seq_len.

    Analyzes both action states and all input features at each offset.
    """
    pattern = TemporalPattern()

    # Handle edge case
    if threshold <= 0:
        threshold = feature_acts.max().item() * 0.5

    # Find frames where feature fires strongly
    strong_mask = feature_acts >= threshold
    strong_indices = strong_mask.nonzero().squeeze(-1)

    if len(strong_indices) < 50:
        return pattern  # Not enough data

    # Get sequence positions for strong activations
    strong_batch = batch_seq_info[strong_indices, 0]
    strong_pos = batch_seq_info[strong_indices, 1]

    # Precompute baseline stats for input features if available
    analyze_features = input_features is not None and feature_names is not None
    if analyze_features and baseline_means is None:
        baseline_means = input_features.mean(dim=0)
        baseline_stds = input_features.std(dim=0).clamp(min=1e-6)

    # For each offset, analyze what actions and features occur
    for offset in offsets:
        target_pos = strong_pos + offset

        # Filter to valid positions (within sequence bounds)
        valid_mask = (target_pos >= 0) & (target_pos < seq_len)

        if valid_mask.sum() < 20:
            continue

        # Get the global indices for the offset positions
        offset_indices = strong_indices[valid_mask] + offset

        # Clamp to valid range
        offset_indices = offset_indices.clamp(0, len(ego_actions) - 1)

        # Check that we're still in the same sequence (batch)
        same_batch = (batch_seq_info[offset_indices, 0] == strong_batch[valid_mask])
        offset_indices = offset_indices[same_batch]

        if len(offset_indices) < 20:
            continue

        # === Action state analysis ===
        ego_at_offset = ego_actions[offset_indices]
        unique, counts = ego_at_offset.unique(return_counts=True)
        total = counts.sum().item()
        probs = counts.float() / total
        sorted_idx = probs.argsort(descending=True)
        top_actions = []
        for idx in sorted_idx[:n_top_actions]:
            aid = int(unique[idx])
            prob = probs[idx].item()
            top_actions.append((aid, get_action_name(aid), prob))
        pattern.actions_at_offset[offset] = top_actions

        # === Input feature analysis ===
        if analyze_features:
            # Get input features at this offset
            feats_at_offset = input_features[offset_indices]  # [n_samples, input_dim]
            means_at_offset = feats_at_offset.mean(dim=0)  # [input_dim]

            # Compute z-scores (how different from baseline)
            z_scores = (means_at_offset - baseline_means) / baseline_stds

            # Find top features by absolute z-score
            abs_z = z_scores.abs()
            top_k = min(n_top_features, len(z_scores))
            top_z_vals, top_z_idx = abs_z.topk(top_k)

            top_features = []
            for i in range(top_k):
                idx = top_z_idx[i].item()
                if idx < len(feature_names):
                    name = feature_names[idx]
                    mean_val = means_at_offset[idx].item()
                    baseline_val = baseline_means[idx].item()
                    z_val = z_scores[idx].item()
                    # Only include if z-score is meaningful
                    if abs(z_val) > 0.5:
                        top_features.append((name, mean_val, baseline_val, z_val))

            if top_features:
                pattern.features_at_offset[offset] = top_features

    return pattern


@dataclass
class PrecomputedData:
    """Precomputed data for fast feature analysis."""
    # Basic stats per feature
    activation_freq: Tensor      # [hidden_dim]
    mean_when_active: Tensor     # [hidden_dim]
    max_activation: Tensor       # [hidden_dim]
    overall_mean: Tensor         # [hidden_dim]

    # Correlations
    correlations: Tensor         # [hidden_dim, n_valid_inputs]
    valid_input_cols: Tensor     # [n_valid_inputs]
    feature_names: List[str]

    # Action means
    ego_unique: Tensor
    ego_means: Tensor            # [n_ego, hidden_dim]
    ego_counts: Tensor
    opp_unique: Tensor
    opp_means: Tensor            # [n_opp, hidden_dim]
    opp_counts: Tensor

    # For temporal analysis
    ego_actions: Tensor          # [N]
    opp_actions: Tensor          # [N]


def precompute_all(
    feature_activations: Tensor,  # [N, hidden_dim]
    input_features: Tensor,       # [N, input_dim]
    batch_seq_info: Tensor,       # [N, 2]
    colmap: ColumnMap,
    feature_names: List[str],
) -> PrecomputedData:
    """Precompute everything needed for fast analysis."""
    N, hidden_dim = feature_activations.shape
    device = feature_activations.device

    _log("Precomputing basic stats...")
    # Basic stats - vectorized
    active_mask = feature_activations > 0  # [N, hidden_dim]
    activation_freq = active_mask.float().mean(dim=0)  # [hidden_dim]

    # Mean when active - need to handle zeros
    sum_when_active = (feature_activations * active_mask).sum(dim=0)
    count_active = active_mask.sum(dim=0).clamp(min=1)
    mean_when_active = sum_when_active / count_active

    max_activation = feature_activations.max(dim=0).values
    overall_mean = feature_activations.mean(dim=0)

    _log("Precomputing correlations...")
    correlations, valid_cols = precompute_correlations(
        feature_activations, input_features, feature_names
    )

    _log("Precomputing action state means...")
    ego_action_idx = colmap.ego_action_idx
    opp_action_idx = colmap.opp_action_idx
    ego_actions = input_features[:, ego_action_idx].long()
    opp_actions = input_features[:, opp_action_idx].long()

    ego_unique, ego_means, ego_counts, opp_unique, opp_means, opp_counts = \
        precompute_action_means(feature_activations, ego_actions, opp_actions)

    _log("Precomputation complete")

    return PrecomputedData(
        activation_freq=activation_freq,
        mean_when_active=mean_when_active,
        max_activation=max_activation,
        overall_mean=overall_mean,
        correlations=correlations,
        valid_input_cols=valid_cols,
        feature_names=feature_names,
        ego_unique=ego_unique,
        ego_means=ego_means,
        ego_counts=ego_counts,
        opp_unique=opp_unique,
        opp_means=opp_means,
        opp_counts=opp_counts,
        ego_actions=ego_actions,
        opp_actions=opp_actions,
    )


def analyze_all_features_batched(
    feature_indices: List[int],
    precomputed: PrecomputedData,
    feature_activations: Tensor,  # [N, hidden_dim]
    batch_seq_info: Tensor,
    input_features: Optional[Tensor] = None,  # [N, input_dim] for temporal feature analysis
    n_top_correlations: int = 10,
    n_max_activating: int = 20,
    analyze_temporal: bool = True,
) -> Dict[int, FeatureStats]:
    """Analyze all features in batched operations to minimize GPU syncs."""
    device = feature_activations.device
    n_features = len(feature_indices)
    feat_idx_tensor = torch.tensor(feature_indices, device=device)

    _log(f"Batched analysis of {n_features} features...")

    # 1. Transfer all needed scalar data to CPU in one go
    _log("  Transferring precomputed stats to CPU...")
    activation_freq_cpu = precomputed.activation_freq[feat_idx_tensor].cpu().numpy()
    mean_when_active_cpu = precomputed.mean_when_active[feat_idx_tensor].cpu().numpy()
    max_activation_cpu = precomputed.max_activation[feat_idx_tensor].cpu().numpy()
    overall_mean_cpu = precomputed.overall_mean[feat_idx_tensor].cpu().numpy()

    # 2. Get top correlations for all features at once
    _log("  Computing top correlations...")
    corr_subset = precomputed.correlations[feat_idx_tensor]  # [n_features, n_valid]
    abs_corr = corr_subset.abs()
    top_k = min(n_top_correlations, corr_subset.shape[1])
    _, top_corr_idx = abs_corr.topk(top_k, dim=1)  # [n_features, top_k]
    # Gather actual correlation values
    top_corr_vals = torch.gather(corr_subset, 1, top_corr_idx)  # [n_features, top_k]
    top_corr_idx_cpu = top_corr_idx.cpu().numpy()
    top_corr_vals_cpu = top_corr_vals.cpu().numpy()
    valid_cols_cpu = precomputed.valid_input_cols.cpu().numpy()

    # 3. Get top activating indices for all features
    _log("  Computing max activating indices...")
    acts_subset = feature_activations[:, feat_idx_tensor].T  # [n_features, N]
    top_k_act = min(n_max_activating, acts_subset.shape[1])
    _, max_act_idx = acts_subset.topk(top_k_act, dim=1)  # [n_features, top_k_act]
    max_act_idx_cpu = max_act_idx.cpu().numpy()

    # 4. Precompute action analysis data on CPU
    _log("  Preparing action analysis data...")
    ego_unique_cpu = precomputed.ego_unique.cpu().numpy()
    opp_unique_cpu = precomputed.opp_unique.cpu().numpy()
    ego_means_cpu = precomputed.ego_means[:, feat_idx_tensor].cpu().numpy()  # [n_ego, n_features]
    opp_means_cpu = precomputed.opp_means[:, feat_idx_tensor].cpu().numpy()  # [n_opp, n_features]
    ego_counts_cpu = precomputed.ego_counts.cpu().numpy()
    opp_counts_cpu = precomputed.opp_counts.cpu().numpy()

    # 5. Build results on CPU (no more GPU syncs)
    _log("  Building feature stats...")
    results = {}
    min_count = 10

    for i, feat_idx in enumerate(feature_indices):
        activation_frequency = float(activation_freq_cpu[i])
        mean_activation = float(mean_when_active_cpu[i])
        max_activation_val = float(max_activation_cpu[i])
        baseline = float(overall_mean_cpu[i])

        # Build correlation list
        top_input_correlations = []
        for j in range(top_k):
            col_idx = int(valid_cols_cpu[top_corr_idx_cpu[i, j]])
            name = precomputed.feature_names[col_idx]
            corr_val = float(top_corr_vals_cpu[i, j])
            top_input_correlations.append((name, corr_val))

        # Build action analysis
        ego_action_means = {}
        opp_action_means = {}
        for k in range(len(ego_unique_cpu)):
            if ego_counts_cpu[k] > min_count:
                ego_action_means[int(ego_unique_cpu[k])] = float(ego_means_cpu[k, i])
        for k in range(len(opp_unique_cpu)):
            if opp_counts_cpu[k] > min_count:
                opp_action_means[int(opp_unique_cpu[k])] = float(opp_means_cpu[k, i])

        # Top actions
        ego_sorted = sorted(ego_action_means.items(), key=lambda x: x[1], reverse=True)
        top_ego_actions = [
            (aid, get_action_name(aid), mean)
            for aid, mean in ego_sorted[:10]
            if mean > baseline * 1.2
        ]
        opp_sorted = sorted(opp_action_means.items(), key=lambda x: x[1], reverse=True)
        top_opp_actions = [
            (aid, get_action_name(aid), mean)
            for aid, mean in opp_sorted[:10]
            if mean > baseline * 1.2
        ]

        action_analysis = ActionAnalysis(
            ego_action_means=ego_action_means,
            opp_action_means=opp_action_means,
            top_ego_actions=top_ego_actions,
            top_opp_actions=top_opp_actions,
        )

        # Max activating indices
        max_activating_indices = max_act_idx_cpu[i].tolist()

        results[feat_idx] = FeatureStats(
            feature_idx=feat_idx,
            activation_frequency=activation_frequency,
            mean_activation=mean_activation,
            max_activation=max_activation_val,
            top_input_correlations=top_input_correlations,
            action_analysis=action_analysis,
            temporal_pattern=None,  # Filled in below for active features
            max_activating_indices=max_activating_indices,
        )

        # Progress every 1000 features
        if (i + 1) % 1000 == 0:
            _log(f"    Built stats for {i+1}/{n_features} features")

    # Temporal analysis for features with >1% activation
    if analyze_temporal:
        active_for_temporal = [idx for idx in feature_indices
                               if results[idx].activation_frequency > 0.01]
        if active_for_temporal:
            n_temporal = len(active_for_temporal)
            _log(f"  Computing temporal patterns for {n_temporal} features...")

            # Batch compute thresholds (90th percentile) for all features at once - on GPU
            feat_idx_tensor_temporal = torch.tensor(active_for_temporal, device=feature_activations.device)
            acts_subset_gpu = feature_activations[:, feat_idx_tensor_temporal]  # [N, n_temporal_features]
            thresholds = torch.quantile(acts_subset_gpu, 0.90, dim=0)  # [n_temporal_features]

            # Move everything to CPU to avoid per-feature GPU syncs
            _log(f"    Transferring {n_temporal} feature columns to CPU...")
            acts_subset_cpu = acts_subset_gpu.cpu()  # [N, n_temporal]
            thresholds_cpu = thresholds.cpu()
            ego_actions_cpu = precomputed.ego_actions.cpu() if isinstance(precomputed.ego_actions, Tensor) \
                else torch.tensor(precomputed.ego_actions)
            opp_actions_cpu = precomputed.opp_actions.cpu() if isinstance(precomputed.opp_actions, Tensor) \
                else torch.tensor(precomputed.opp_actions)
            batch_seq_cpu = batch_seq_info.cpu()

            # Also transfer input features for full temporal analysis
            input_features_cpu = None
            feature_names = None
            baseline_means_cpu = None
            baseline_stds_cpu = None
            if input_features is not None:
                _log(f"    Transferring input features to CPU for temporal analysis...")
                input_features_cpu = input_features.cpu()
                feature_names = precomputed.feature_names
                # Precompute baseline stats once
                baseline_means_cpu = input_features_cpu.mean(dim=0)
                baseline_stds_cpu = input_features_cpu.std(dim=0).clamp(min=1e-6)

            # Precompute seq_len once
            seq_len = int(batch_seq_cpu[:, 1].max().item()) + 1

            _log(f"    Running temporal analysis on CPU (expanded offsets + input features)...")
            start_temporal = time.time()
            for i, feat_idx in enumerate(active_for_temporal):
                # Use pre-extracted CPU data - no GPU syncs
                acts = acts_subset_cpu[:, i]
                threshold = thresholds_cpu[i].item()
                temporal = analyze_temporal_patterns_fast(
                    acts, ego_actions_cpu, opp_actions_cpu, batch_seq_cpu,
                    threshold=threshold, seq_len=seq_len,
                    input_features=input_features_cpu,
                    feature_names=feature_names,
                    baseline_means=baseline_means_cpu,
                    baseline_stds=baseline_stds_cpu,
                )
                results[feat_idx].temporal_pattern = temporal

                if (i + 1) % 500 == 0:
                    elapsed = time.time() - start_temporal
                    rate = (i + 1) / elapsed
                    remaining = (n_temporal - i - 1) / rate if rate > 0 else 0
                    _log(f"    Temporal: {i+1}/{n_temporal} ({rate:.1f}/s, ETA: {_format_time(remaining)})")

    return results


def analyze_feature_fast(
    feature_idx: int,
    precomputed: PrecomputedData,
    feature_activations: Tensor,  # [N, hidden_dim] - needed for temporal
    batch_seq_info: Tensor,
    n_top_correlations: int = 10,
    n_max_activating: int = 20,
    analyze_temporal: bool = True,
) -> FeatureStats:
    """Fast feature analysis using precomputed data (single feature version)."""
    # Basic stats - just index into precomputed
    activation_frequency = precomputed.activation_freq[feature_idx].item()
    mean_activation = precomputed.mean_when_active[feature_idx].item()
    max_activation_val = precomputed.max_activation[feature_idx].item()
    baseline = precomputed.overall_mean[feature_idx].item()

    # Correlations - index into precomputed matrix
    corr_row = precomputed.correlations[feature_idx]  # [n_valid]
    abs_corr = corr_row.abs()
    top_k = min(n_top_correlations, len(corr_row))
    top_vals, top_idx = abs_corr.topk(top_k)

    # Build correlation list
    top_input_correlations = []
    for i in range(top_k):
        col_idx = precomputed.valid_input_cols[top_idx[i]].item()
        name = precomputed.feature_names[col_idx]
        corr_val = corr_row[top_idx[i]].item()
        top_input_correlations.append((name, corr_val))

    # Action analysis - use precomputed means
    action_analysis = analyze_action_states_fast(
        feature_idx,
        precomputed.ego_unique,
        precomputed.ego_means,
        precomputed.ego_counts,
        precomputed.opp_unique,
        precomputed.opp_means,
        precomputed.opp_counts,
        baseline,
    )

    # Temporal analysis - still need to compute per-feature (but only for active features)
    temporal_pattern = None
    if analyze_temporal and activation_frequency > 0.01:
        acts = feature_activations[:, feature_idx]
        temporal_pattern = analyze_temporal_patterns(
            acts, precomputed.ego_actions, precomputed.opp_actions, batch_seq_info
        )

    # Max activating frames
    acts = feature_activations[:, feature_idx]
    top_activations = acts.topk(min(n_max_activating, len(acts)))
    max_activating_indices = top_activations.indices.cpu().tolist()

    return FeatureStats(
        feature_idx=feat_idx,
        activation_frequency=activation_frequency,
        mean_activation=mean_activation,
        max_activation=max_activation_val,
        top_input_correlations=top_input_correlations,
        action_analysis=action_analysis,
        temporal_pattern=temporal_pattern,
        max_activating_indices=max_activating_indices,
    )


def suggest_feature_label(stats: FeatureStats) -> str:
    """Generate a suggested label for a feature based on analysis."""
    labels = []

    # Check action analysis (proper categorical)
    for action_id, name, mean in stats.action_analysis.top_ego_actions[:2]:
        if mean > stats.mean_activation * 1.5:  # Significantly above average
            labels.append(f"ego:{name}")

    for action_id, name, mean in stats.action_analysis.top_opp_actions[:2]:
        if mean > stats.mean_activation * 1.5:
            labels.append(f"opp:{name}")

    # Check input correlations
    for name, corr in stats.top_input_correlations[:4]:
        if abs(corr) > 0.2:
            sign = "+" if corr > 0 else "-"
            # Simplify name
            simple_name = name.replace("p1_", "").replace("p2_", "opp_")
            labels.append(f"{sign}{simple_name}")

    # Check temporal patterns
    if stats.temporal_pattern:
        # Look for predictive actions (what happens after feature fires)
        for offset, actions in stats.temporal_pattern.actions_after.items():
            if actions and offset <= 15:  # Within ~0.25 seconds
                top_action = actions[0]
                if top_action[2] > 0.3:  # >30% probability
                    labels.append(f"->({offset}f){top_action[1]}")

    if labels:
        return " | ".join(labels[:4])
    elif stats.activation_frequency < 0.01:
        return "rare (possibly dead)"
    else:
        return "unknown"


def generate_report(
    sae_path: Path,
    feature_stats: Dict[int, FeatureStats],
    n_features_total: int,
    hook_point: HookPoint,
) -> FeatureAnalysisReport:
    """Generate analysis report."""
    n_dead = sum(1 for s in feature_stats.values() if s.activation_frequency < 0.001)

    mean_freq = (
        sum(s.activation_frequency for s in feature_stats.values()) / len(feature_stats)
        if feature_stats
        else 0.0
    )

    # Sort features by activation frequency
    features_by_freq = sorted(
        feature_stats.keys(),
        key=lambda idx: feature_stats[idx].activation_frequency,
        reverse=True,
    )

    return FeatureAnalysisReport(
        sae_path=str(sae_path),
        hook_point=str(hook_point),
        n_features_total=n_features_total,
        n_features_analyzed=len(feature_stats),
        n_dead_features=n_dead,
        feature_stats=feature_stats,
        mean_activation_frequency=mean_freq,
        features_by_frequency=features_by_freq,
    )


def export_report(report: FeatureAnalysisReport, output_dir: Path) -> None:
    """Export report to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON
    summary = {
        "sae_path": report.sae_path,
        "hook_point": report.hook_point,
        "n_features_total": report.n_features_total,
        "n_features_analyzed": report.n_features_analyzed,
        "n_dead_features": report.n_dead_features,
        "mean_activation_frequency": report.mean_activation_frequency,
        "top_features_by_frequency": report.features_by_frequency[:50],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Per-feature details
    features_data = {}
    for idx, stats in report.feature_stats.items():
        suggested_label = suggest_feature_label(stats)

        # Serialize action analysis
        action_data = {
            "top_ego_actions": [
                {"id": aid, "name": name, "mean_activation": mean}
                for aid, name, mean in stats.action_analysis.top_ego_actions
            ],
            "top_opp_actions": [
                {"id": aid, "name": name, "mean_activation": mean}
                for aid, name, mean in stats.action_analysis.top_opp_actions
            ],
        }

        # Serialize temporal patterns
        temporal_data = None
        if stats.temporal_pattern:
            temporal_data = {
                "actions_at_offset": {
                    str(k): [{"id": a[0], "name": a[1], "prob": a[2]} for a in v]
                    for k, v in stats.temporal_pattern.actions_at_offset.items()
                },
                "features_at_offset": {
                    str(k): [{"name": f[0], "mean": f[1], "baseline": f[2], "z_score": f[3]} for f in v]
                    for k, v in stats.temporal_pattern.features_at_offset.items()
                },
                # Legacy format for compatibility
                "actions_before": {
                    str(k): [{"id": a[0], "name": a[1], "prob": a[2]} for a in v]
                    for k, v in stats.temporal_pattern.actions_before.items()
                },
                "actions_after": {
                    str(k): [{"id": a[0], "name": a[1], "prob": a[2]} for a in v]
                    for k, v in stats.temporal_pattern.actions_after.items()
                },
            }

        features_data[str(idx)] = {
            "suggested_label": suggested_label,
            "activation_frequency": stats.activation_frequency,
            "mean_activation": stats.mean_activation,
            "max_activation": stats.max_activation,
            "top_input_correlations": stats.top_input_correlations,
            "action_analysis": action_data,
            "temporal_patterns": temporal_data,
            "max_activating_indices": stats.max_activating_indices,
        }
    with open(output_dir / "features.json", "w") as f:
        json.dump(features_data, f, indent=2)

    # Human-readable summary
    with open(output_dir / "features.md", "w") as f:
        f.write(f"# SAE Feature Analysis\n\n")
        f.write(f"**SAE Path:** {report.sae_path}\n")
        f.write(f"**Hook Point:** {report.hook_point}\n")
        f.write(f"**Features Analyzed:** {report.n_features_analyzed} / {report.n_features_total}\n")
        f.write(f"**Dead Features:** {report.n_dead_features}\n")
        f.write(f"**Mean Activation Frequency:** {report.mean_activation_frequency:.4f}\n\n")

        f.write("## Top Features by Activation Frequency\n\n")
        for i, feat_idx in enumerate(report.features_by_frequency[:30]):
            stats = report.feature_stats[feat_idx]
            label = suggest_feature_label(stats)
            f.write(f"### Feature {feat_idx}\n\n")
            f.write(f"- **Suggested Label:** {label}\n")
            f.write(f"- **Activation Frequency:** {stats.activation_frequency:.4f}\n")
            f.write(f"- **Mean Activation:** {stats.mean_activation:.4f}\n\n")

            if stats.top_input_correlations:
                f.write("**Top Input Correlations:**\n")
                for name, corr in stats.top_input_correlations[:5]:
                    f.write(f"- {name}: {corr:+.3f}\n")
                f.write("\n")

            if stats.action_analysis.top_ego_actions:
                f.write("**Top Ego Actions (by mean activation):**\n")
                for action_id, name, mean in stats.action_analysis.top_ego_actions[:5]:
                    f.write(f"- {name} (id={action_id}): {mean:.3f}\n")
                f.write("\n")

            if stats.action_analysis.top_opp_actions:
                f.write("**Top Opponent Actions (by mean activation):**\n")
                for action_id, name, mean in stats.action_analysis.top_opp_actions[:5]:
                    f.write(f"- {name} (id={action_id}): {mean:.3f}\n")
                f.write("\n")

            if stats.temporal_pattern:
                if stats.temporal_pattern.actions_at_offset:
                    f.write("**Temporal Action Patterns:**\n")
                    f.write("| Offset | Top Action | Prob |\n")
                    f.write("|--------|------------|------|\n")
                    for offset in sorted(stats.temporal_pattern.actions_at_offset.keys()):
                        actions = stats.temporal_pattern.actions_at_offset[offset]
                        if actions:
                            top = actions[0]
                            label = f"{offset:+d}f"
                            f.write(f"| {label} | {top[1]} | {top[2]*100:.1f}% |\n")
                    f.write("\n")

                if stats.temporal_pattern.features_at_offset:
                    f.write("**Temporal Input Feature Changes (z-score from baseline):**\n")
                    f.write("| Offset | Top Features (z-score) |\n")
                    f.write("|--------|------------------------|\n")
                    for offset in sorted(stats.temporal_pattern.features_at_offset.keys()):
                        features = stats.temporal_pattern.features_at_offset[offset]
                        if features:
                            label = f"{offset:+d}f"
                            feat_strs = [f"{name}({z:+.1f})" for name, _, _, z in features[:4]]
                            f.write(f"| {label} | {', '.join(feat_strs)} |\n")
                    f.write("\n")

    print(f"\nReport exported to: {output_dir}")


def print_feature_summary(stats: FeatureStats) -> None:
    """Print a summary for a single feature."""
    label = suggest_feature_label(stats)
    print(f"\nFeature {stats.feature_idx}:")
    print(f"  Suggested label: {label}")
    print(f"  Activation frequency: {stats.activation_frequency:.4f}")
    print(f"  Mean activation: {stats.mean_activation:.4f}")
    print(f"  Max activation: {stats.max_activation:.4f}")

    if stats.top_input_correlations:
        print("  Top input correlations:")
        for name, corr in stats.top_input_correlations[:5]:
            print(f"    {name}: {corr:+.3f}")

    if stats.action_analysis.top_ego_actions:
        print("  Top ego actions (mean activation when in action):")
        for action_id, name, mean in stats.action_analysis.top_ego_actions[:5]:
            print(f"    {name} [{action_id}]: {mean:.3f}")

    if stats.action_analysis.top_opp_actions:
        print("  Top opponent actions:")
        for action_id, name, mean in stats.action_analysis.top_opp_actions[:3]:
            print(f"    {name} [{action_id}]: {mean:.3f}")

    if stats.temporal_pattern:
        # Show actions at key offsets
        if stats.temporal_pattern.actions_at_offset:
            print("  Temporal action patterns:")
            for offset in sorted(stats.temporal_pattern.actions_at_offset.keys()):
                actions = stats.temporal_pattern.actions_at_offset[offset]
                if actions:
                    top = actions[0]
                    label = f"{offset:+d}f" if offset != 0 else "0f"
                    print(f"    {label}: {top[1]} ({top[2]*100:.1f}%)")

        # Show input features that change significantly
        if stats.temporal_pattern.features_at_offset:
            print("  Temporal input feature changes (z-score):")
            for offset in sorted(stats.temporal_pattern.features_at_offset.keys()):
                features = stats.temporal_pattern.features_at_offset[offset]
                if features:
                    label = f"{offset:+d}f" if offset != 0 else "0f"
                    top_feats = [f"{name}({z:+.1f})" for name, _, _, z in features[:3]]
                    print(f"    {label}: {', '.join(top_feats)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze SAE features")

    parser.add_argument(
        "--sae",
        type=Path,
        required=True,
        help="Path to trained SAE checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--features",
        type=int,
        nargs="*",
        default=None,
        help="Specific feature indices to analyze (default: all)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100000,
        help="Maximum activation samples to collect (default: 100000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to export report (optional)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top features to display",
    )
    parser.add_argument(
        "--no-temporal",
        action="store_true",
        help="Skip temporal analysis (faster)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data directory (default: use checkpoint config)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overall_start = time.time()

    device = _resolve_device(None)
    _log(f"Using device: {device}")

    # Load action state names from library
    _load_action_state_names()

    # Load SAE and model
    _log(f"Loading SAE from: {args.sae}")
    _log(f"Loading model from: {args.checkpoint}")
    sae, model, config, hook_point = load_sae_and_model(
        args.sae, args.checkpoint, device, data_dir=args.data_dir
    )
    _log(f"Hook point: {hook_point}")
    _log(f"SAE: {sae.input_dim} -> {sae.hidden_dim} features (k={sae.k})")

    # Create dataloader
    _log("Creating dataloader...")
    loader, ds, _ = make_dataloader(config)
    colmap = ColumnMap.from_dataset(ds)
    feature_names = ds._feature_names_sel
    _log(f"Dataloader ready: {len(feature_names)} input features")

    # Collect activations (on GPU)
    feature_acts, input_feats, batch_seq_info = collect_feature_activations(
        model, sae, hook_point, loader, colmap, device, max_samples=args.max_samples
    )

    # Precompute everything for fast analysis
    _log("Precomputing analysis data (this is the slow part)...")
    precompute_start = time.time()
    precomputed = precompute_all(
        feature_acts,
        input_feats,
        batch_seq_info,
        colmap,
        feature_names,
    )
    precompute_elapsed = time.time() - precompute_start
    _log(f"Precomputation complete in {_format_time(precompute_elapsed)}")

    # Determine which features to analyze
    if args.features:
        feature_indices = args.features
    else:
        # Analyze all non-dead features (with some activity)
        _log("Identifying active features...")
        active_features = (precomputed.activation_freq > 0.001).nonzero().squeeze(-1).tolist()
        dead_count = sae.hidden_dim - len(active_features)
        feature_indices = active_features
        _log(f"Found {len(feature_indices):,} active features ({dead_count:,} dead)")

    # Analyze features using batched operations (minimizes GPU syncs)
    analysis_start = time.time()
    feature_stats = analyze_all_features_batched(
        feature_indices,
        precomputed,
        feature_acts,
        batch_seq_info,
        input_features=input_feats,  # Pass for expanded temporal analysis
        analyze_temporal=not args.no_temporal,
    )
    analysis_elapsed = time.time() - analysis_start
    _log(f"Feature analysis complete in {_format_time(analysis_elapsed)}")

    # Generate report
    _log("Generating report...")
    report = generate_report(args.sae, feature_stats, sae.hidden_dim, hook_point)

    # Print top features
    print(f"\n{'=' * 60}")
    print(f"Top {args.top_n} Features by Activation Frequency")
    print(f"{'=' * 60}")

    for feat_idx in report.features_by_frequency[:args.top_n]:
        print_feature_summary(report.feature_stats[feat_idx])

    # Summary stats
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"Total features: {report.n_features_total}")
    print(f"Analyzed: {report.n_features_analyzed}")
    print(f"Dead features (<0.1% activation): {report.n_dead_features}")
    print(f"Mean activation frequency: {report.mean_activation_frequency:.4f}")

    # Export if requested
    if args.output_dir:
        _log(f"Exporting report to: {args.output_dir}")
        export_report(report, args.output_dir)

    total_time = time.time() - overall_start
    _log(f"Total analysis time: {_format_time(total_time)}")
    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
