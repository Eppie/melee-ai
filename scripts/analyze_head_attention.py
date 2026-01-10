#!/usr/bin/env python3
"""Analyze how much each output head attends to prior heads' outputs.

This script examines the fc1 weight matrices of each output head to determine
how strongly each head weights its various input sources (hidden_states,
button logits, stick logits, etc.).

Usage:
    python scripts/analyze_head_attention.py                    # Analyze latest checkpoint
    python scripts/analyze_head_attention.py --checkpoint path/to/model.pt
    python scripts/analyze_head_attention.py --json             # Output as JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from constants import CONTROLLER_KEY_GROUPS

# Button names in order (matches model)
BUTTON_NAMES = CONTROLLER_KEY_GROUPS["buttons"]
BUTTON_DISPLAY = ["A", "B", "X/Y", "Z", "L/R"]


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find the most recent checkpoint in a directory."""
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)


def analyze_head_weights(
    weight: torch.Tensor,
    regions: List[Tuple[str, int, int]],
    individual_cols: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
    """
    Analyze a head's attention to different input regions.

    Args:
        weight: The fc1 weight matrix [hidden_dim, input_dim]
        regions: List of (region_name, start_idx, end_idx) tuples
        individual_cols: Optional dict of {region_name: [col_names]} for detailed breakdown

    Returns:
        Dict with analysis results
    """
    total_norm = weight.norm().item()
    hs_norm_per_dim = None

    results = {
        "weight_shape": list(weight.shape),
        "total_norm": total_norm,
        "regions": [],
    }

    for region_name, start, end in regions:
        region_weights = weight[:, start:end]
        region_norm = region_weights.norm().item()
        region_dim = end - start
        norm_per_dim = region_norm / region_dim if region_dim > 0 else 0

        region_data = {
            "name": region_name,
            "columns": f"[{start}:{end}]",
            "dims": region_dim,
            "l2_norm": region_norm,
            "norm_per_dim": norm_per_dim,
        }

        if region_name == "hidden_states":
            hs_norm_per_dim = norm_per_dim
            region_data["vs_hidden"] = 1.0
        elif hs_norm_per_dim and hs_norm_per_dim > 0:
            region_data["vs_hidden"] = norm_per_dim / hs_norm_per_dim

        # Individual column breakdown if requested
        if individual_cols and region_name in individual_cols:
            col_names = individual_cols[region_name]
            region_data["individual"] = []
            for i, col_name in enumerate(col_names):
                if start + i < end:
                    col_norm = weight[:, start + i].norm().item()
                    vs_hs = col_norm / hs_norm_per_dim if hs_norm_per_dim else 0
                    region_data["individual"].append({
                        "name": col_name,
                        "norm": col_norm,
                        "vs_hidden": vs_hs,
                    })

        results["regions"].append(region_data)

    results["hidden_norm_per_dim"] = hs_norm_per_dim
    return results


def print_head_analysis(name: str, analysis: Dict[str, Any], verbose: bool = True) -> None:
    """Print formatted analysis for a single head."""
    print(f"{'─' * 90}")
    print(f"  {name}")
    shape = analysis["weight_shape"]
    print(f"  Weight shape: {shape} → [{shape[0]} hidden] x [{shape[1]} input]")
    print(f"{'─' * 90}")

    print(f"  {'Input Source':<25} │ {'Cols':<12} │ {'Dims':>5} │ {'L2 Norm':>9} │ {'Norm/Dim':>9} │ {'vs Hidden':>10}")
    print(f"  {'─' * 25}─┼─{'─' * 12}─┼─{'─' * 5}─┼─{'─' * 9}─┼─{'─' * 9}─┼─{'─' * 10}")

    for region in analysis["regions"]:
        vs_hidden = region.get("vs_hidden", 0)
        if region["name"] == "hidden_states":
            vs_str = "baseline"
        else:
            vs_str = f"{vs_hidden:.2f}x"

        print(
            f"  {region['name']:<25} │ {region['columns']:<12} │ "
            f"{region['dims']:>5} │ {region['l2_norm']:>9.4f} │ "
            f"{region['norm_per_dim']:>9.4f} │ {vs_str:>10}"
        )

    # Print individual columns if present
    if verbose:
        for region in analysis["regions"]:
            if "individual" in region:
                print()
                print(f"  Individual {region['name']}:")
                for col in region["individual"]:
                    print(f"    {col['name']:<20}: {col['norm']:.4f} ({col['vs_hidden']:.2f}x hidden)")

    print()


def analyze_checkpoint(checkpoint_path: Path, verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze head-to-head attention patterns in a checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        verbose: Whether to print detailed output

    Returns:
        Dict with all analysis results
    """
    if verbose:
        print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", {})

    if not state:
        raise ValueError("No model state found in checkpoint")

    # Check if using separate button heads
    has_separate_buttons = any("button_heads" in k for k in state.keys())

    # Check if passing button hidden to stick (by checking main_stick input size)
    main_stick_key = None
    for k in state.keys():
        if "main_stick_head.fc1.weight" in k:
            main_stick_key = k
            break

    pass_button_hidden = False
    if main_stick_key:
        main_stick_input_size = state[main_stick_key].shape[1]
        # If input size > 517 (512 + 5), then button hidden is being passed
        pass_button_hidden = main_stick_input_size > 520

    results = {
        "checkpoint_path": str(checkpoint_path),
        "has_separate_button_heads": has_separate_buttons,
        "pass_button_hidden_to_stick": pass_button_hidden,
        "heads": {},
    }

    if verbose:
        print()
        print("=" * 90)
        print("COMPREHENSIVE HEAD-TO-HEAD ATTENTION ANALYSIS")
        print("=" * 90)
        print()
        print("For each head, we measure how much weight (L2 norm) it places on each input source.")
        print("'Norm/Dim' normalizes by dimension count - higher = more attention per feature.")
        print("'vs Hidden' shows attention relative to hidden_states (baseline = 1.0x).")
        print()
        print(f"Config: separate_button_heads={has_separate_buttons}, pass_button_hidden_to_stick={pass_button_hidden}")

    # ========================================================================
    # BUTTON HEADS
    # ========================================================================
    if has_separate_buttons:
        if verbose:
            print()
            print("=" * 90)
            print("BUTTON HEADS (autoregressive: A → B → X/Y → Z → L/R)")
            print("=" * 90)

        results["heads"]["buttons"] = {}

        for i, (btn, display) in enumerate(zip(BUTTON_NAMES, BUTTON_DISPLAY)):
            key = f"_orig_mod.button_heads.{btn}.fc1.weight"
            if key not in state:
                key = f"button_heads.{btn}.fc1.weight"
            if key not in state:
                continue

            weight = state[key]
            regions = [("hidden_states", 0, 512)]
            individual = {}

            if i > 0:
                prev_names = BUTTON_DISPLAY[:i]
                regions.append((f"prev_buttons ({','.join(prev_names)})", 512, 512 + i))
                individual[f"prev_buttons ({','.join(prev_names)})"] = prev_names

            analysis = analyze_head_weights(weight, regions, individual if i > 0 else None)
            results["heads"]["buttons"][display] = analysis

            if verbose:
                print_head_analysis(f"Button {display} Head", analysis, verbose=True)

    else:
        # Unified button head
        if verbose:
            print()
            print("=" * 90)
            print("BUTTON HEAD (unified)")
            print("=" * 90)

        key = "_orig_mod.button_head.fc1.weight"
        if key not in state:
            key = "button_head.fc1.weight"
        if key in state:
            weight = state[key]
            regions = [("hidden_states", 0, 512)]
            analysis = analyze_head_weights(weight, regions)
            results["heads"]["buttons"] = {"unified": analysis}
            if verbose:
                print_head_analysis("Button Head (unified)", analysis)

    # ========================================================================
    # MAIN STICK HEAD
    # ========================================================================
    if verbose:
        print()
        print("=" * 90)
        print("MAIN STICK HEAD")
        print("=" * 90)

    key = "_orig_mod.main_stick_head.fc1.weight"
    if key not in state:
        key = "main_stick_head.fc1.weight"
    if key in state:
        weight = state[key]
        input_size = weight.shape[1]

        regions = [
            ("hidden_states", 0, 512),
            ("button_logits (all)", 512, 517),
        ]

        # Check if button hidden is included
        if input_size > 520:
            regions.append(("button_hidden (5x128)", 517, 517 + 640))

        individual = {"button_logits (all)": BUTTON_DISPLAY}

        analysis = analyze_head_weights(weight, regions, individual)
        results["heads"]["main_stick"] = analysis

        if verbose:
            print_head_analysis("Main Stick Head", analysis, verbose=True)

    # ========================================================================
    # C-STICK HEAD
    # ========================================================================
    if verbose:
        print()
        print("=" * 90)
        print("C-STICK HEAD")
        print("=" * 90)

    key = "_orig_mod.c_stick_head.fc1.weight"
    if key not in state:
        key = "c_stick_head.fc1.weight"
    if key in state:
        weight = state[key]
        regions = [
            ("hidden_states", 0, 512),
            ("button_logits (all)", 512, 517),
            ("main_stick_logits (64)", 517, 581),
        ]
        individual = {"button_logits (all)": BUTTON_DISPLAY}

        analysis = analyze_head_weights(weight, regions, individual)
        results["heads"]["c_stick"] = analysis

        if verbose:
            print_head_analysis("C-Stick Head", analysis, verbose=True)

    # ========================================================================
    # SHOULDER HEAD
    # ========================================================================
    if verbose:
        print()
        print("=" * 90)
        print("SHOULDER HEAD")
        print("=" * 90)

    key = "_orig_mod.shoulder_head.fc1.weight"
    if key not in state:
        key = "shoulder_head.fc1.weight"
    if key in state:
        weight = state[key]
        regions = [
            ("hidden_states", 0, 512),
            ("button_logits (all)", 512, 517),
            ("main_stick_logits (64)", 517, 581),
            ("c_stick_logits (9)", 581, 590),
        ]
        individual = {"button_logits (all)": BUTTON_DISPLAY}

        analysis = analyze_head_weights(weight, regions, individual)
        results["heads"]["shoulder"] = analysis

        if verbose:
            print_head_analysis("Shoulder Head", analysis, verbose=True)

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================
    if verbose:
        print_summary_table(results)

    return results


def print_summary_table(results: Dict[str, Any]) -> None:
    """Print a summary table of head-to-head attention."""
    print()
    print("=" * 90)
    print("SUMMARY: HEAD-TO-HEAD ATTENTION (vs hidden_states baseline)")
    print("=" * 90)
    print()

    # Build summary data
    heads_data = results.get("heads", {})

    print("┌─────────────────┬────────────────────────────────────────────────────────────────────────┐")
    print("│                 │                        INPUT SOURCE ATTENTION                          │")
    print("│  OUTPUT HEAD    ├──────────┬────────┬────────┬────────┬────────┬────────┬───────┬───────┤")
    print("│                 │ hidden   │ btn_A  │ btn_B  │ btn_XY │ btn_Z  │ btn_LR │ main  │ c_stk │")
    print("│                 │ states   │        │        │        │        │        │ stick │       │")
    print("├─────────────────┼──────────┼────────┼────────┼────────┼────────┼────────┼───────┼───────┤")

    def get_vs_hidden(analysis: Dict, region_name: str, col_idx: Optional[int] = None) -> str:
        """Get vs_hidden value for a region or individual column."""
        for region in analysis.get("regions", []):
            if region["name"].startswith(region_name) or region_name in region["name"]:
                if col_idx is not None and "individual" in region:
                    if col_idx < len(region["individual"]):
                        val = region["individual"][col_idx].get("vs_hidden", 0)
                        return f"{val:.1f}x" if val >= 0.1 else "<0.1x"
                else:
                    val = region.get("vs_hidden", 0)
                    return f"{val:.1f}x" if val >= 0.1 else "<0.1x"
        return "-"

    # Button heads
    if "buttons" in heads_data and isinstance(heads_data["buttons"], dict):
        for display in BUTTON_DISPLAY:
            if display in heads_data["buttons"]:
                analysis = heads_data["buttons"][display]
                row = f"│ Button {display:<9} │ baseline │"

                # Previous button columns
                for j, prev in enumerate(BUTTON_DISPLAY):
                    if j < BUTTON_DISPLAY.index(display):
                        val = get_vs_hidden(analysis, "prev_buttons", j)
                        row += f" {val:>6} │"
                    else:
                        row += "    -   │"

                row += "   -   │   -   │"
                print(row)

    print("├─────────────────┼──────────┼────────┼────────┼────────┼────────┼────────┼───────┼───────┤")

    # Main stick head
    if "main_stick" in heads_data:
        analysis = heads_data["main_stick"]
        row = "│ MAIN STICK      │ baseline │"
        for j in range(5):
            val = get_vs_hidden(analysis, "button_logits", j)
            row += f" {val:>6} │"
        row += "   -   │   -   │"
        print(row)

    print("├─────────────────┼──────────┼────────┼────────┼────────┼────────┼────────┼───────┼───────┤")

    # C-stick head
    if "c_stick" in heads_data:
        analysis = heads_data["c_stick"]
        row = "│ C-Stick         │ baseline │"
        for j in range(5):
            val = get_vs_hidden(analysis, "button_logits", j)
            row += f" {val:>6} │"
        val = get_vs_hidden(analysis, "main_stick")
        row += f" {val:>5} │   -   │"
        print(row)

    print("├─────────────────┼──────────┼────────┼────────┼────────┼────────┼────────┼───────┼───────┤")

    # Shoulder head
    if "shoulder" in heads_data:
        analysis = heads_data["shoulder"]
        row = "│ Shoulder        │ baseline │"
        for j in range(5):
            val = get_vs_hidden(analysis, "button_logits", j)
            row += f" {val:>6} │"
        val_main = get_vs_hidden(analysis, "main_stick")
        val_c = get_vs_hidden(analysis, "c_stick")
        row += f" {val_main:>5} │ {val_c:>5} │"
        print(row)

    print("└─────────────────┴──────────┴────────┴────────┴────────┴────────┴────────┴───────┴───────┘")
    print()

    # Key findings
    print("KEY FINDINGS:")
    print()

    # Check main_stick attention to L/R
    if "main_stick" in heads_data:
        analysis = heads_data["main_stick"]
        for region in analysis.get("regions", []):
            if "button_logits" in region["name"] and "individual" in region:
                lr_val = region["individual"][4].get("vs_hidden", 0) if len(region["individual"]) > 4 else 0
                if lr_val < 5:
                    print(f"  ⚠ Main stick head has LOW attention to L/R ({lr_val:.1f}x hidden)")
                    print(f"    This may cause coordination issues (e.g., airdodges instead of wavedash)")
                else:
                    print(f"  ✓ Main stick head has good attention to L/R ({lr_val:.1f}x hidden)")

    # Check shoulder attention to L/R
    if "shoulder" in heads_data:
        analysis = heads_data["shoulder"]
        for region in analysis.get("regions", []):
            if "button_logits" in region["name"] and "individual" in region:
                lr_val = region["individual"][4].get("vs_hidden", 0) if len(region["individual"]) > 4 else 0
                print(f"  ✓ Shoulder head has strong attention to L/R ({lr_val:.1f}x hidden) - expected for shield")

    # Check if button hidden is being passed
    if results.get("pass_button_hidden_to_stick"):
        print()
        print("  ℹ Button hidden activations (640 dim) are being passed to main_stick_head")
        if "main_stick" in heads_data:
            analysis = heads_data["main_stick"]
            for region in analysis.get("regions", []):
                if "button_hidden" in region["name"]:
                    val = region.get("vs_hidden", 0)
                    print(f"    Attention to button_hidden: {val:.1f}x hidden")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze head-to-head attention patterns in model checkpoints"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint file (default: latest in ~/checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path.home() / "checkpoints",
        help="Directory containing checkpoints",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only print summary table",
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

    # Run analysis
    results = analyze_checkpoint(checkpoint_path, verbose=not args.json)

    if args.json:
        # Convert any non-serializable types
        def make_serializable(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(v) for v in obj]
            return obj

        print(json.dumps(make_serializable(results), indent=2))


if __name__ == "__main__":
    main()
