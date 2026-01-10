#!/usr/bin/env python3
"""Comprehensive checkpoint analysis for detecting model issues.

This script analyzes model checkpoints to identify potential training problems:
- Weight/bias distribution anomalies
- Logit magnitude issues (overconfidence)
- Input projection feature concentration
- Layer-wise statistics
- Training metrics analysis (overfitting detection)
- Numerical stability issues

Usage:
    python scripts/analyze_checkpoint.py                    # Analyze latest checkpoint
    python scripts/analyze_checkpoint.py --checkpoint path/to/model.pt
    python scripts/analyze_checkpoint.py --all              # Run all analyses
    python scripts/analyze_checkpoint.py --weights --logits # Run specific analyses
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from column_map import ColumnMap
from config.config import Config
from schema import get_feature_names, get_target_names


class Severity(Enum):
    """Issue severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class Issue:
    """A detected issue in the checkpoint."""
    severity: Severity
    category: str
    message: str
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        prefix = {
            Severity.INFO: "\033[94m[INFO]\033[0m",
            Severity.WARNING: "\033[93m[WARNING]\033[0m",
            Severity.ERROR: "\033[91m[ERROR]\033[0m",
        }[self.severity]
        return f"{prefix} [{self.category}] {self.message}"


@dataclass
class AnalysisReport:
    """Container for analysis results."""
    checkpoint_path: str
    issues: List[Issue] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    def add_issue(
        self,
        severity: Severity,
        category: str,
        message: str,
        details: Optional[Dict] = None,
    ) -> None:
        self.issues.append(Issue(severity, category, message, details))

    def add_stat(self, category: str, name: str, value: Any) -> None:
        if category not in self.statistics:
            self.statistics[category] = {}
        self.statistics[category][name] = value

    def summary(self) -> str:
        """Generate a summary of the analysis."""
        lines = [
            "=" * 80,
            f"Checkpoint Analysis Report: {self.checkpoint_path}",
            "=" * 80,
        ]

        # Count issues by severity
        counts = {sev: 0 for sev in Severity}
        for issue in self.issues:
            counts[issue.severity] += 1

        lines.append(
            f"\nIssues: {counts[Severity.ERROR]} errors, "
            f"{counts[Severity.WARNING]} warnings, {counts[Severity.INFO]} info"
        )

        if self.issues:
            lines.append("\n" + "-" * 40)
            lines.append("Issues Found:")
            lines.append("-" * 40)
            for issue in sorted(self.issues, key=lambda x: x.severity.value):
                lines.append(str(issue))
                if issue.details:
                    for k, v in issue.details.items():
                        if isinstance(v, float):
                            lines.append(f"    {k}: {v:.6g}")
                        else:
                            lines.append(f"    {k}: {v}")

        return "\n".join(lines)


class CheckpointAnalyzer:
    """Comprehensive checkpoint analyzer."""

    # Thresholds for detecting issues
    WEIGHT_EXTREME_THRESHOLD = 10.0  # Weights beyond this are concerning
    WEIGHT_VERY_EXTREME_THRESHOLD = 50.0  # Weights beyond this are errors
    BIAS_EXTREME_THRESHOLD = 20.0
    LOGIT_EXTREME_THRESHOLD = 100.0  # Logits beyond this indicate overconfidence
    LOGIT_VERY_EXTREME_THRESHOLD = 500.0
    ENTROPY_LOW_THRESHOLD = 0.1  # Very low entropy = overconfident
    DEAD_NEURON_THRESHOLD = 1e-7  # Weights below this considered dead
    GRADIENT_VANISH_THRESHOLD = 1e-7
    GRADIENT_EXPLODE_THRESHOLD = 100.0

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = "cpu",
        verbose: bool = True,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.verbose = verbose
        self.report = AnalysisReport(str(checkpoint_path))

        # Load checkpoint
        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load checkpoint and extract components."""
        if self.verbose:
            print(f"Loading checkpoint: {self.checkpoint_path}")

        self.checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )

        # Extract model state
        self.model_state = self.checkpoint.get("model", {})
        if not self.model_state:
            self.report.add_issue(
                Severity.ERROR, "checkpoint", "No model state found in checkpoint"
            )

        # Extract config
        self.config_dict = self.checkpoint.get("config", {})
        if self.config_dict:
            try:
                self.config = Config.model_validate(self.config_dict)
            except Exception as e:
                self.config = None
                self.report.add_issue(
                    Severity.WARNING,
                    "checkpoint",
                    f"Could not parse config: {e}",
                )
        else:
            self.config = None
            self.report.add_issue(
                Severity.WARNING, "checkpoint", "No config found in checkpoint"
            )

        # Extract metadata
        self.epoch = self.checkpoint.get("epoch", "unknown")
        self.global_step = self.checkpoint.get("global_step", "unknown")
        self.git_commit = self.checkpoint.get("git_commit", "unknown")

        self.report.add_stat("metadata", "epoch", self.epoch)
        self.report.add_stat("metadata", "global_step", self.global_step)
        self.report.add_stat("metadata", "git_commit", self.git_commit)

        if self.verbose:
            print(f"  Epoch: {self.epoch}, Step: {self.global_step}")

    def _get_tensor_stats(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive statistics for a tensor."""
        t = tensor.float().flatten()
        return {
            "mean": float(t.mean()),
            "std": float(t.std()),
            "min": float(t.min()),
            "max": float(t.max()),
            "abs_mean": float(t.abs().mean()),
            "abs_max": float(t.abs().max()),
            "p01": float(torch.quantile(t, 0.01)),
            "p05": float(torch.quantile(t, 0.05)),
            "p50": float(torch.quantile(t, 0.50)),
            "p95": float(torch.quantile(t, 0.95)),
            "p99": float(torch.quantile(t, 0.99)),
            "num_elements": int(t.numel()),
            "num_zeros": int((t == 0).sum()),
            "num_nan": int(torch.isnan(t).sum()),
            "num_inf": int(torch.isinf(t).sum()),
        }

    def analyze_weights(self) -> None:
        """Analyze weight distributions across all layers."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("WEIGHT ANALYSIS")
            print("=" * 60)

        weight_stats = {}
        extreme_weights = []

        for name, param in self.model_state.items():
            if "weight" not in name:
                continue

            stats = self._get_tensor_stats(param)
            weight_stats[name] = stats

            # Check for numerical issues
            if stats["num_nan"] > 0:
                self.report.add_issue(
                    Severity.ERROR,
                    "weights",
                    f"NaN values in {name}",
                    {"num_nan": stats["num_nan"]},
                )

            if stats["num_inf"] > 0:
                self.report.add_issue(
                    Severity.ERROR,
                    "weights",
                    f"Inf values in {name}",
                    {"num_inf": stats["num_inf"]},
                )

            # Check for extreme weights
            if stats["abs_max"] > self.WEIGHT_VERY_EXTREME_THRESHOLD:
                self.report.add_issue(
                    Severity.ERROR,
                    "weights",
                    f"Very extreme weights in {name}",
                    {"abs_max": stats["abs_max"], "std": stats["std"]},
                )
                extreme_weights.append((name, stats["abs_max"]))
            elif stats["abs_max"] > self.WEIGHT_EXTREME_THRESHOLD:
                self.report.add_issue(
                    Severity.WARNING,
                    "weights",
                    f"Extreme weights in {name}",
                    {"abs_max": stats["abs_max"], "std": stats["std"]},
                )
                extreme_weights.append((name, stats["abs_max"]))

            # Check for dead neurons (all zeros or near-zeros)
            if "weight" in name and param.dim() >= 2:
                row_norms = param.float().norm(dim=-1)
                dead_count = (row_norms < self.DEAD_NEURON_THRESHOLD).sum().item()
                if dead_count > 0:
                    pct = 100 * dead_count / row_norms.numel()
                    if pct > 10:
                        self.report.add_issue(
                            Severity.WARNING,
                            "weights",
                            f"Dead neurons in {name}",
                            {"dead_count": dead_count, "percentage": f"{pct:.1f}%"},
                        )

            if self.verbose:
                print(f"\n{name}:")
                print(f"  Shape: {list(param.shape)}")
                print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
                print(f"  Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
                print(f"  |Max|: {stats['abs_max']:.6f}")

        self.report.add_stat("weights", "layer_stats", weight_stats)

        # Report top extreme weight layers
        if extreme_weights:
            extreme_weights.sort(key=lambda x: x[1], reverse=True)
            self.report.add_stat(
                "weights",
                "most_extreme_layers",
                [(n, f"{v:.2f}") for n, v in extreme_weights[:5]],
            )

    def analyze_biases(self) -> None:
        """Analyze bias distributions."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("BIAS ANALYSIS")
            print("=" * 60)

        bias_stats = {}

        for name, param in self.model_state.items():
            if "bias" not in name:
                continue

            stats = self._get_tensor_stats(param)
            bias_stats[name] = stats

            if stats["num_nan"] > 0 or stats["num_inf"] > 0:
                self.report.add_issue(
                    Severity.ERROR,
                    "biases",
                    f"NaN/Inf in {name}",
                    {"num_nan": stats["num_nan"], "num_inf": stats["num_inf"]},
                )

            if stats["abs_max"] > self.BIAS_EXTREME_THRESHOLD:
                self.report.add_issue(
                    Severity.WARNING,
                    "biases",
                    f"Extreme bias values in {name}",
                    {"abs_max": stats["abs_max"]},
                )

            if self.verbose:
                print(f"\n{name}:")
                print(f"  Shape: {list(param.shape)}")
                print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
                print(f"  Range: [{stats['min']:.6f}, {stats['max']:.6f}]")

        self.report.add_stat("biases", "layer_stats", bias_stats)

    def analyze_input_projection(self) -> None:
        """Analyze input projection weights to identify concentrated features."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("INPUT PROJECTION ANALYSIS")
            print("=" * 60)

        # Find projection_down weight
        proj_weight = None
        for name, param in self.model_state.items():
            if "projection_down.weight" in name:
                proj_weight = param
                break

        if proj_weight is None:
            self.report.add_issue(
                Severity.INFO,
                "input_projection",
                "No projection_down layer found",
            )
            return

        # Compute L2 norm of weights for each input feature
        # projection_down.weight has shape [embedding_dim, input_size]
        feature_norms = proj_weight.float().norm(dim=0)  # [input_size]

        # Get feature names mapping
        try:
            feature_names = get_feature_names()
            target_names = get_target_names()
            colmap = ColumnMap(feature_names, target_names)
        except Exception as e:
            self.report.add_issue(
                Severity.WARNING,
                "input_projection",
                f"Could not load feature mapping: {e}",
            )
            colmap = None

        # Build input feature index mapping
        input_feature_map = self._build_input_feature_map(colmap)

        # Find top features by weight norm
        top_k = 20
        top_values, top_indices = torch.topk(feature_norms, min(top_k, len(feature_norms)))

        if self.verbose:
            print(f"\nTop {top_k} input features by weight norm:")

        top_features = []
        for i, (idx, val) in enumerate(zip(top_indices.tolist(), top_values.tolist())):
            feature_name = input_feature_map.get(idx, f"index_{idx}")
            top_features.append((feature_name, idx, val))
            if self.verbose:
                print(f"  {i+1:2d}. {feature_name:40s} (idx={idx:3d}): {val:.4f}")

        self.report.add_stat("input_projection", "top_features_by_norm", top_features)

        # Check for highly concentrated weights
        norm_std = float(feature_norms.std())
        norm_mean = float(feature_norms.mean())
        concentration_ratio = float(top_values[0] / (norm_mean + 1e-8))

        if concentration_ratio > 10:
            self.report.add_issue(
                Severity.WARNING,
                "input_projection",
                f"Highly concentrated input weights (top feature {concentration_ratio:.1f}x mean)",
                {"top_feature": top_features[0][0], "concentration_ratio": concentration_ratio},
            )

        self.report.add_stat("input_projection", "norm_mean", norm_mean)
        self.report.add_stat("input_projection", "norm_std", norm_std)
        self.report.add_stat("input_projection", "concentration_ratio", concentration_ratio)

    def _build_input_feature_map(self, colmap: Optional[ColumnMap]) -> Dict[int, str]:
        """Build mapping from input tensor index to feature name.

        The input tensor is constructed in _embed_inputs as:
        - stage one-hot (num_stages)
        - ego_character one-hot (num_characters)
        - opponent_character one-hot (num_characters)
        - ego_action one-hot (num_actions)
        - opponent_action one-hot (num_actions)
        - gamestate features
        - controller features
        """
        feature_map = {}

        if self.config is None:
            return feature_map

        mc = self.config.model
        idx = 0

        # Stage one-hot
        for i in range(mc.num_stages):
            feature_map[idx] = f"stage[{i}]"
            idx += 1

        # Character mappings (Fox is index 1 in the preprocessed enum)
        character_names = {
            0: "MARIO", 1: "FOX", 2: "CFALCON", 3: "DK", 4: "KIRBY",
            5: "BOWSER", 6: "LINK", 7: "SHEIK", 8: "NESS", 9: "PEACH",
            10: "POPO", 11: "NANA", 12: "PIKACHU", 13: "SAMUS", 14: "YOSHI",
            15: "JIGGLYPUFF", 16: "MEWTWO", 17: "LUIGI", 18: "MARTH", 19: "ZELDA",
            20: "YLINK", 21: "DOC", 22: "FALCO", 23: "PICHU", 24: "GAMEWATCH",
            25: "GANONDORF", 26: "ROY",
        }

        # Ego character one-hot
        for i in range(mc.num_characters):
            char_name = character_names.get(i, f"CHAR_{i}")
            feature_map[idx] = f"ego_character[{char_name}]"
            idx += 1

        # Opponent character one-hot
        for i in range(mc.num_characters):
            char_name = character_names.get(i, f"CHAR_{i}")
            feature_map[idx] = f"opponent_character[{char_name}]"
            idx += 1

        # Action names (common important ones)
        action_names = {
            0: "DEAD_DOWN", 1: "DEAD_LEFT", 2: "DEAD_RIGHT", 3: "DEAD_UP",
            4: "DEAD_UP_STAR", 5: "DEAD_UP_FALL", 6: "SLEEP", 7: "REBIRTH",
            8: "REBIRTH_WAIT", 14: "STANDING", 15: "WALK_SLOW", 16: "WALK_MIDDLE",
            17: "WALK_FAST", 20: "WAIT", 23: "DASH", 24: "RUN", 25: "STOP_RUN",
            26: "STOP_TURN", 27: "STOP_TURN_RUN", 28: "KNEE_BEND", 29: "JUMP_F",
            30: "JUMP_B", 31: "JUMP_AERIAL_F", 32: "JUMP_AERIAL_B", 37: "LANDING",
            40: "CROUCH_START", 41: "CROUCH", 42: "CROUCH_END", 43: "ATTACK_11",
            44: "ATTACK_12", 46: "ATTACK_100_START", 50: "ATTACK_S3_HI",
            51: "ATTACK_S3_HI_S", 52: "ATTACK_S3_S", 53: "ATTACK_S3_LW_S",
            54: "ATTACK_S3_LW", 55: "ATTACK_HI3", 56: "ATTACK_LW3", 57: "ATTACK_S4_HI",
            59: "ATTACK_S4_S", 61: "ATTACK_S4_LW", 63: "ATTACK_HI4", 64: "ATTACK_LW4",
            65: "ATTACK_AIR_N", 66: "ATTACK_AIR_F", 67: "ATTACK_AIR_B",
            68: "ATTACK_AIR_HI", 69: "ATTACK_AIR_LW", 70: "LANDING_AIR_N",
            178: "GUARD_ON", 179: "GUARD", 180: "GUARD_OFF", 181: "GUARD_SET_OFF",
            182: "GUARD_REFLECT", 183: "SHIELD_STUN", 186: "TECH_MISS_DOWN",
            212: "ESCAPE_F", 213: "ESCAPE_B", 233: "DOWN_BOUND_U", 234: "DOWN_WAIT_U",
            241: "DOWN_BOUND_D", 242: "DOWN_WAIT_D", 291: "SHINE_START",
        }

        # Ego action one-hot
        for i in range(mc.num_actions):
            action_name = action_names.get(i, f"ACTION_{i}")
            feature_map[idx] = f"ego_action[{action_name}]"
            idx += 1

        # Opponent action one-hot
        for i in range(mc.num_actions):
            action_name = action_names.get(i, f"ACTION_{i}")
            feature_map[idx] = f"opponent_action[{action_name}]"
            idx += 1

        # Gamestate features (from colmap)
        if colmap is not None:
            try:
                feature_names = colmap.feat_names
                for gs_idx in colmap.gamestate_idxs:
                    feature_map[idx] = f"gamestate:{feature_names[gs_idx]}"
                    idx += 1

                for ctrl_idx in colmap.controller_idxs:
                    feature_map[idx] = f"controller:{feature_names[ctrl_idx]}"
                    idx += 1
            except Exception:
                pass

        return feature_map

    def analyze_output_heads(self) -> None:
        """Analyze output head weights and biases."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("OUTPUT HEAD ANALYSIS")
            print("=" * 60)

        head_names = ["button_head", "main_stick_head", "c_stick_head", "shoulder_head", "value_head"]

        for head_name in head_names:
            head_weights = {}
            head_biases = {}

            for name, param in self.model_state.items():
                if head_name in name:
                    if "weight" in name:
                        head_weights[name] = self._get_tensor_stats(param)
                    elif "bias" in name:
                        head_biases[name] = self._get_tensor_stats(param)

            if head_weights:
                if self.verbose:
                    print(f"\n{head_name}:")

                for name, stats in head_weights.items():
                    if stats["abs_max"] > self.WEIGHT_EXTREME_THRESHOLD:
                        self.report.add_issue(
                            Severity.WARNING,
                            "output_heads",
                            f"Extreme weights in {name}",
                            {"abs_max": stats["abs_max"]},
                        )

                    if self.verbose:
                        print(f"  {name}: |max|={stats['abs_max']:.4f}, std={stats['std']:.4f}")

                self.report.add_stat(f"output_heads.{head_name}", "weights", head_weights)
                self.report.add_stat(f"output_heads.{head_name}", "biases", head_biases)

    def analyze_attention_layers(self) -> None:
        """Analyze attention layer statistics."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("ATTENTION LAYER ANALYSIS")
            print("=" * 60)

        # Group by block number
        block_stats = defaultdict(dict)

        for name, param in self.model_state.items():
            if "blocks." not in name or "attention" not in name:
                continue

            # Extract block number (handle _orig_mod. prefix from torch.compile)
            # Format: _orig_mod.blocks.0.attention.query_projection.weight
            # or: blocks.0.attention.query_projection.weight
            parts = name.split(".")
            block_idx_pos = parts.index("blocks") + 1
            block_idx = int(parts[block_idx_pos])

            stats = self._get_tensor_stats(param)
            # Extract layer type (everything after block index)
            layer_type = ".".join(parts[block_idx_pos + 1:])
            block_stats[block_idx][layer_type] = stats

            # Check for issues
            if stats["abs_max"] > self.WEIGHT_EXTREME_THRESHOLD:
                self.report.add_issue(
                    Severity.WARNING,
                    "attention",
                    f"Extreme values in block {block_idx} {layer_type}",
                    {"abs_max": stats["abs_max"]},
                )

        if self.verbose:
            for block_idx in sorted(block_stats.keys()):
                print(f"\nBlock {block_idx}:")
                for layer_type, stats in block_stats[block_idx].items():
                    print(f"  {layer_type}: |max|={stats['abs_max']:.4f}, std={stats['std']:.4f}")

        self.report.add_stat("attention", "block_stats", dict(block_stats))

    def analyze_mlp_layers(self) -> None:
        """Analyze MLP layer statistics."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("MLP LAYER ANALYSIS")
            print("=" * 60)

        block_stats = defaultdict(dict)

        for name, param in self.model_state.items():
            if "blocks." not in name or "mlp" not in name:
                continue

            # Extract block number (handle _orig_mod. prefix from torch.compile)
            parts = name.split(".")
            block_idx_pos = parts.index("blocks") + 1
            block_idx = int(parts[block_idx_pos])

            stats = self._get_tensor_stats(param)
            layer_type = ".".join(parts[block_idx_pos + 1:])
            block_stats[block_idx][layer_type] = stats

            # The output_projection of MLP is known to sometimes have extreme values
            if "output_projection" in name and stats["abs_max"] > self.WEIGHT_EXTREME_THRESHOLD:
                self.report.add_issue(
                    Severity.WARNING,
                    "mlp",
                    f"Extreme values in block {block_idx} MLP output_projection",
                    {"abs_max": stats["abs_max"]},
                )

        if self.verbose:
            for block_idx in sorted(block_stats.keys()):
                print(f"\nBlock {block_idx}:")
                for layer_type, stats in block_stats[block_idx].items():
                    print(f"  {layer_type}: |max|={stats['abs_max']:.4f}, std={stats['std']:.4f}")

        self.report.add_stat("mlp", "block_stats", dict(block_stats))

    def analyze_optimizer_state(self) -> None:
        """Analyze optimizer state for potential issues."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("OPTIMIZER STATE ANALYSIS")
            print("=" * 60)

        opt_state = self.checkpoint.get("optimizer", {})
        if not opt_state:
            self.report.add_issue(
                Severity.INFO, "optimizer", "No optimizer state found"
            )
            return

        state = opt_state.get("state", {})
        if not state:
            return

        # Analyze momentum buffers and variance estimates
        extreme_momentum = []
        extreme_variance = []

        for param_id, param_state in state.items():
            # AdamW has exp_avg (momentum) and exp_avg_sq (variance)
            if "exp_avg" in param_state:
                momentum = param_state["exp_avg"]
                m_stats = self._get_tensor_stats(momentum)
                if m_stats["abs_max"] > 10.0:
                    extreme_momentum.append((param_id, m_stats["abs_max"]))

            if "exp_avg_sq" in param_state:
                variance = param_state["exp_avg_sq"]
                v_stats = self._get_tensor_stats(variance)
                if v_stats["max"] > 100.0:
                    extreme_variance.append((param_id, v_stats["max"]))

        if extreme_momentum:
            self.report.add_issue(
                Severity.WARNING,
                "optimizer",
                f"Large momentum buffers in {len(extreme_momentum)} parameters",
                {"count": len(extreme_momentum)},
            )

        if extreme_variance:
            self.report.add_issue(
                Severity.WARNING,
                "optimizer",
                f"Large variance estimates in {len(extreme_variance)} parameters",
                {"count": len(extreme_variance)},
            )

        if self.verbose:
            print(f"  Parameters with state: {len(state)}")
            print(f"  Extreme momentum buffers: {len(extreme_momentum)}")
            print(f"  Extreme variance estimates: {len(extreme_variance)}")

    def analyze_training_metrics(self, metrics_path: Optional[Path] = None) -> None:
        """Analyze training metrics for overfitting and other issues."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("TRAINING METRICS ANALYSIS")
            print("=" * 60)

        # Try to find metrics file
        if metrics_path is None:
            checkpoint_dir = self.checkpoint_path.parent
            metrics_path = checkpoint_dir / "training_metrics.jsonl"

        if not metrics_path.exists():
            self.report.add_issue(
                Severity.INFO,
                "training_metrics",
                f"No training metrics file found at {metrics_path}",
            )
            return

        # Load metrics
        metrics = []
        with open(metrics_path, "r") as f:
            for line in f:
                try:
                    metrics.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        if not metrics:
            return

        if self.verbose:
            print(f"  Loaded {len(metrics)} metric entries")

        # Analyze key metrics
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        effective_batch_fractions = []

        for m in metrics:
            if "train_loss" in m:
                train_losses.append(m["train_loss"])
            if "val_loss" in m:
                val_losses.append(m["val_loss"])
            if "train_acc" in m or "train_main_acc" in m:
                train_accs.append(m.get("train_acc", m.get("train_main_acc", 0)))
            if "val_acc" in m or "val_main_acc" in m:
                val_accs.append(m.get("val_acc", m.get("val_main_acc", 0)))
            if "effective_batch_fraction" in m:
                effective_batch_fractions.append(m["effective_batch_fraction"])

        # Check for overfitting
        if train_losses and val_losses:
            # Find minimum validation loss and when it occurred
            min_val_idx = np.argmin(val_losses)
            min_val_loss = val_losses[min_val_idx]
            final_val_loss = val_losses[-1]
            final_train_loss = train_losses[-1] if train_losses else None

            self.report.add_stat("training_metrics", "min_val_loss", min_val_loss)
            self.report.add_stat("training_metrics", "min_val_loss_step", min_val_idx)
            self.report.add_stat("training_metrics", "final_val_loss", final_val_loss)
            self.report.add_stat("training_metrics", "final_train_loss", final_train_loss)

            # Overfitting: val loss increased significantly from minimum
            if final_val_loss > min_val_loss * 1.2:
                self.report.add_issue(
                    Severity.WARNING,
                    "training_metrics",
                    f"Potential overfitting: val_loss increased {(final_val_loss/min_val_loss - 1)*100:.1f}% from minimum",
                    {
                        "min_val_loss": min_val_loss,
                        "final_val_loss": final_val_loss,
                        "min_occurred_at_step": min_val_idx,
                    },
                )

            # Train/val gap
            if final_train_loss and final_val_loss:
                gap = final_val_loss - final_train_loss
                gap_ratio = final_val_loss / (final_train_loss + 1e-8)
                if gap_ratio > 1.5:
                    self.report.add_issue(
                        Severity.WARNING,
                        "training_metrics",
                        f"Large train/val loss gap (ratio: {gap_ratio:.2f}x)",
                        {"train_loss": final_train_loss, "val_loss": final_val_loss},
                    )

            if self.verbose:
                print(f"  Final train loss: {final_train_loss:.4f}" if final_train_loss else "")
                print(f"  Final val loss: {final_val_loss:.4f}")
                print(f"  Min val loss: {min_val_loss:.4f} (at step {min_val_idx})")

        # Check accuracy gap
        if train_accs and val_accs:
            final_train_acc = train_accs[-1]
            final_val_acc = val_accs[-1]
            acc_gap = final_train_acc - final_val_acc

            self.report.add_stat("training_metrics", "final_train_acc", final_train_acc)
            self.report.add_stat("training_metrics", "final_val_acc", final_val_acc)

            if acc_gap > 0.15:  # 15% gap
                self.report.add_issue(
                    Severity.WARNING,
                    "training_metrics",
                    f"Large train/val accuracy gap: {acc_gap*100:.1f}%",
                    {"train_acc": final_train_acc, "val_acc": final_val_acc},
                )

            if self.verbose:
                print(f"  Final train accuracy: {final_train_acc*100:.1f}%")
                print(f"  Final val accuracy: {final_val_acc*100:.1f}%")

        # Check effective batch fraction
        if effective_batch_fractions:
            avg_ebf = np.mean(effective_batch_fractions)
            self.report.add_stat("training_metrics", "avg_effective_batch_fraction", avg_ebf)

            if avg_ebf < 0.5:
                self.report.add_issue(
                    Severity.WARNING,
                    "training_metrics",
                    f"Low effective batch fraction: {avg_ebf*100:.1f}%",
                    {"expected": "~85%", "actual": f"{avg_ebf*100:.1f}%"},
                )

            if self.verbose:
                print(f"  Avg effective batch fraction: {avg_ebf*100:.1f}%")

    def analyze_numerical_stability(self) -> None:
        """Check for numerical stability issues across all parameters."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("NUMERICAL STABILITY CHECK")
            print("=" * 60)

        total_nan = 0
        total_inf = 0
        very_large_params = []

        for name, param in self.model_state.items():
            nan_count = torch.isnan(param).sum().item()
            inf_count = torch.isinf(param).sum().item()

            total_nan += nan_count
            total_inf += inf_count

            abs_max = param.float().abs().max().item()
            if abs_max > 1000:
                very_large_params.append((name, abs_max))

        if total_nan > 0:
            self.report.add_issue(
                Severity.ERROR,
                "numerical_stability",
                f"Total NaN values across all parameters: {total_nan}",
            )

        if total_inf > 0:
            self.report.add_issue(
                Severity.ERROR,
                "numerical_stability",
                f"Total Inf values across all parameters: {total_inf}",
            )

        if very_large_params:
            self.report.add_issue(
                Severity.WARNING,
                "numerical_stability",
                f"Parameters with values > 1000: {len(very_large_params)}",
                {"parameters": [(n, f"{v:.1f}") for n, v in very_large_params[:5]]},
            )

        if self.verbose:
            print(f"  Total NaN values: {total_nan}")
            print(f"  Total Inf values: {total_inf}")
            print(f"  Parameters with |values| > 1000: {len(very_large_params)}")

        self.report.add_stat("numerical_stability", "total_nan", total_nan)
        self.report.add_stat("numerical_stability", "total_inf", total_inf)
        self.report.add_stat("numerical_stability", "very_large_param_count", len(very_large_params))

    def analyze_layer_norms(self) -> None:
        """Analyze layer/RMS norm statistics if present."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("NORMALIZATION LAYER ANALYSIS")
            print("=" * 60)

        norm_params = {}
        for name, param in self.model_state.items():
            if "norm" in name.lower() or "ln" in name.lower():
                norm_params[name] = self._get_tensor_stats(param)

        if not norm_params:
            if self.verbose:
                print("  No normalization layers found (using RMSNorm without learnable params)")
            return

        for name, stats in norm_params.items():
            if self.verbose:
                print(f"  {name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

    def analyze_gradient_scaler(self) -> None:
        """Analyze gradient scaler state for AMP training issues."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("GRADIENT SCALER ANALYSIS")
            print("=" * 60)

        scaler_state = self.checkpoint.get("scaler", {})
        if not scaler_state:
            if self.verbose:
                print("  No gradient scaler state (not using AMP or using bfloat16)")
            return

        scale = scaler_state.get("scale", "unknown")
        growth_factor = scaler_state.get("growth_factor", "unknown")
        backoff_factor = scaler_state.get("backoff_factor", "unknown")
        growth_interval = scaler_state.get("growth_interval", "unknown")

        self.report.add_stat("scaler", "scale", scale)

        if isinstance(scale, (int, float)) and scale < 1:
            self.report.add_issue(
                Severity.WARNING,
                "scaler",
                f"Very low gradient scale: {scale}",
                {"scale": scale},
            )

        if self.verbose:
            print(f"  Scale: {scale}")
            print(f"  Growth factor: {growth_factor}")
            print(f"  Backoff factor: {backoff_factor}")
            print(f"  Growth interval: {growth_interval}")

    def analyze_config(self) -> None:
        """Analyze config settings for potential issues."""
        if self.verbose:
            print("\n" + "=" * 60)
            print("CONFIG ANALYSIS")
            print("=" * 60)

        if self.config is None:
            return

        # Check learning rate
        lr = self.config.train.lr
        if lr > 1e-3:
            self.report.add_issue(
                Severity.INFO,
                "config",
                f"High learning rate: {lr}",
            )

        # Check weight decay
        wd = self.config.train.weight_decay
        if wd < 0.001:
            self.report.add_issue(
                Severity.INFO,
                "config",
                f"Low weight decay: {wd} (may contribute to overfitting)",
            )

        # Check dropout
        dropout = self.config.model.dropout
        if dropout < 0.05:
            self.report.add_issue(
                Severity.INFO,
                "config",
                f"Low dropout: {dropout} (may contribute to overfitting)",
            )

        # Check label smoothing
        ls = self.config.train.label_smoothing
        if ls < 0.01:
            self.report.add_issue(
                Severity.INFO,
                "config",
                f"Low label smoothing: {ls} (may contribute to overconfidence)",
            )

        if self.verbose:
            print(f"  Learning rate: {lr}")
            print(f"  Weight decay: {wd}")
            print(f"  Dropout: {dropout}")
            print(f"  Label smoothing: {ls}")
            print(f"  Grad clip: {self.config.train.grad_clip}")
            print(f"  Batch size: {self.config.train.batch_size}")

    def run_all(self) -> AnalysisReport:
        """Run all analyses."""
        analyses = [
            self.analyze_numerical_stability,
            self.analyze_weights,
            self.analyze_biases,
            self.analyze_input_projection,
            self.analyze_output_heads,
            self.analyze_attention_layers,
            self.analyze_mlp_layers,
            self.analyze_layer_norms,
            self.analyze_optimizer_state,
            self.analyze_gradient_scaler,
            self.analyze_training_metrics,
            self.analyze_config,
        ]

        for analysis in analyses:
            try:
                analysis()
            except Exception as e:
                self.report.add_issue(
                    Severity.ERROR,
                    "analysis",
                    f"Analysis {analysis.__name__} failed: {e}",
                )

        return self.report


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the most recent checkpoint in a directory."""
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive checkpoint analysis for detecting model issues"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint file (default: latest in ../checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints",
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all analyses (default)",
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Analyze weight distributions",
    )
    parser.add_argument(
        "--biases",
        action="store_true",
        help="Analyze bias distributions",
    )
    parser.add_argument(
        "--logits",
        action="store_true",
        help="Analyze output logits (requires sample data)",
    )
    parser.add_argument(
        "--input-projection",
        action="store_true",
        help="Analyze input projection weights",
    )
    parser.add_argument(
        "--training-metrics",
        action="store_true",
        help="Analyze training metrics for overfitting",
    )
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=None,
        help="Path to training_metrics.jsonl file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary and issues",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    # Find checkpoint
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path is None:
            print(f"No checkpoints found in {args.checkpoint_dir}")
            sys.exit(1)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Create analyzer
    analyzer = CheckpointAnalyzer(
        checkpoint_path=checkpoint_path,
        verbose=not args.quiet,
    )

    # Determine which analyses to run
    run_specific = any([
        args.weights,
        args.biases,
        args.logits,
        args.input_projection,
        args.training_metrics,
    ])

    if args.all or not run_specific:
        report = analyzer.run_all()
    else:
        if args.weights:
            analyzer.analyze_weights()
        if args.biases:
            analyzer.analyze_biases()
        if args.input_projection:
            analyzer.analyze_input_projection()
        if args.training_metrics:
            analyzer.analyze_training_metrics(args.metrics_file)
        report = analyzer.report

    # Output results
    if args.json:
        output = {
            "checkpoint_path": str(report.checkpoint_path),
            "issues": [
                {
                    "severity": i.severity.value,
                    "category": i.category,
                    "message": i.message,
                    "details": i.details,
                }
                for i in report.issues
            ],
            "statistics": report.statistics,
        }
        print(json.dumps(output, indent=2, default=str))
    else:
        print(report.summary())

    # Exit with error code if there are errors
    error_count = sum(1 for i in report.issues if i.severity == Severity.ERROR)
    sys.exit(1 if error_count > 0 else 0)


if __name__ == "__main__":
    main()
