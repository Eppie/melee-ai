#!/usr/bin/env python3
"""
Training Run Analysis Script

Analyzes training metrics from jsonl log files and checkpoint files.
Detects configuration changes, parameter resets, and tracks key metrics over time.
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import torch


def load_jsonl(path: Path) -> list[dict]:
    """Load all entries from a jsonl file."""
    entries = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def fmt(value, spec: str = ".4f") -> str:
    """Safely format a value, returning 'N/A' if not a number."""
    if value is None or value == "N/A":
        return "N/A"
    try:
        return f"{value:{spec}}"
    except (ValueError, TypeError):
        return str(value)


def separate_by_type(entries: list[dict]) -> dict[str, list[dict]]:
    """Separate entries by type (train, val, gradient)."""
    by_type = defaultdict(list)
    for entry in entries:
        entry_type = entry.get('type', 'unknown')
        by_type[entry_type].append(entry)
    return dict(by_type)


def detect_lr_changes(train_entries: list[dict], threshold: float = 0.1) -> list[dict]:
    """Detect significant learning rate changes (beyond normal scheduling)."""
    changes = []
    prev_lr = None
    prev_step = None

    for entry in train_entries:
        lr = entry.get('lr')
        step = entry.get('global_step', entry.get('step'))

        if lr is None or step is None:
            continue

        if prev_lr is not None:
            # Calculate expected lr change from scheduling (should be small)
            # A large jump indicates manual intervention
            ratio = lr / prev_lr if prev_lr > 0 else float('inf')

            # Detect significant jumps (either direction)
            if ratio > 1.5 or ratio < 0.67:
                changes.append({
                    'step': step,
                    'prev_step': prev_step,
                    'prev_lr': prev_lr,
                    'new_lr': lr,
                    'ratio': ratio,
                    'timestamp': entry.get('timestamp')
                })

        prev_lr = lr
        prev_step = step

    return changes


def detect_param_norm_discontinuities(train_entries: list[dict], threshold: float = 0.05) -> list[dict]:
    """Detect discontinuities in parameter norm (indicating resets or interventions)."""
    discontinuities = []
    prev_norm = None
    prev_step = None

    for entry in train_entries:
        norm = entry.get('params/total_norm')
        step = entry.get('global_step', entry.get('step'))

        if norm is None or step is None:
            continue

        if prev_norm is not None:
            # Calculate relative change
            rel_change = abs(norm - prev_norm) / prev_norm if prev_norm > 0 else 0

            # Detect significant changes (parameter norm should change smoothly)
            if rel_change > threshold:
                discontinuities.append({
                    'step': step,
                    'prev_step': prev_step,
                    'prev_norm': prev_norm,
                    'new_norm': norm,
                    'rel_change': rel_change,
                    'timestamp': entry.get('timestamp')
                })

        prev_norm = norm
        prev_step = step

    return discontinuities


def extract_metric_series(entries: list[dict], metric_key: str) -> tuple[list[int], list[float]]:
    """Extract a metric series from entries."""
    steps = []
    values = []

    for entry in entries:
        step = entry.get('global_step', entry.get('step'))
        value = entry.get(metric_key)

        if step is not None and value is not None:
            steps.append(step)
            values.append(value)

    return steps, values


def compute_stats_in_windows(steps: list[int], values: list[float], window_size: int = 1000) -> list[dict]:
    """Compute statistics in windows."""
    if not steps:
        return []

    windows = []
    current_window_start = steps[0]
    current_values = []

    for step, value in zip(steps, values):
        if step >= current_window_start + window_size:
            if current_values:
                windows.append({
                    'start_step': current_window_start,
                    'end_step': step,
                    'mean': sum(current_values) / len(current_values),
                    'min': min(current_values),
                    'max': max(current_values),
                    'count': len(current_values)
                })
            current_window_start = step
            current_values = []
        current_values.append(value)

    # Final window
    if current_values:
        windows.append({
            'start_step': current_window_start,
            'end_step': steps[-1],
            'mean': sum(current_values) / len(current_values),
            'min': min(current_values),
            'max': max(current_values),
            'count': len(current_values)
        })

    return windows


def analyze_checkpoint(checkpoint_path: Path) -> dict:
    """Analyze a single checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    analysis = {
        'path': str(checkpoint_path),
        'keys': list(checkpoint.keys()),
    }

    # Extract config if present
    if 'config' in checkpoint:
        config = checkpoint['config']
        if isinstance(config, dict):
            analysis['config'] = config
        else:
            # Pydantic model - try to convert
            try:
                analysis['config'] = config.model_dump() if hasattr(config, 'model_dump') else str(config)
            except:
                analysis['config'] = str(config)

    # Extract optimizer state info
    if 'optimizer' in checkpoint:
        opt_state = checkpoint['optimizer']
        if 'param_groups' in opt_state:
            param_groups = opt_state['param_groups']
            analysis['optimizer_param_groups'] = []
            for i, pg in enumerate(param_groups):
                pg_info = {k: v for k, v in pg.items() if k != 'params'}
                analysis['optimizer_param_groups'].append(pg_info)

    # Compute model parameter statistics
    if 'model' in checkpoint:
        model_state = checkpoint['model']
        total_params = 0
        total_norm = 0.0
        param_stats = {}

        for name, param in model_state.items():
            if isinstance(param, torch.Tensor):
                numel = param.numel()
                norm = param.float().norm().item()
                total_params += numel
                total_norm += norm ** 2

                # Track large parameters
                if norm > 1.0:
                    param_stats[name] = {
                        'shape': list(param.shape),
                        'norm': norm,
                        'mean': param.float().mean().item(),
                        'std': param.float().std().item(),
                        'max_abs': param.float().abs().max().item()
                    }

        analysis['total_params'] = total_params
        analysis['total_norm'] = total_norm ** 0.5
        analysis['large_params'] = param_stats

    # Training info
    for key in ['epoch', 'step', 'global_step', 'best_val_loss']:
        if key in checkpoint:
            analysis[key] = checkpoint[key]

    return analysis


def analyze_checkpoints(checkpoint_dir: Path) -> list[dict]:
    """Analyze all checkpoints in a directory."""
    checkpoint_files = sorted(checkpoint_dir.glob('model_*.pt'))
    analyses = []

    for ckpt_path in checkpoint_files:
        print(f"Analyzing {ckpt_path.name}...")
        try:
            analysis = analyze_checkpoint(ckpt_path)
            analyses.append(analysis)
        except Exception as e:
            print(f"  Error: {e}")

    return analyses


def compare_checkpoints(analyses: list[dict]) -> list[dict]:
    """Compare consecutive checkpoints to detect changes."""
    comparisons = []

    for i in range(1, len(analyses)):
        prev = analyses[i-1]
        curr = analyses[i]

        comparison = {
            'from': prev['path'],
            'to': curr['path'],
            'changes': []
        }

        # Compare optimizer param groups
        if 'optimizer_param_groups' in prev and 'optimizer_param_groups' in curr:
            for j, (prev_pg, curr_pg) in enumerate(zip(
                prev['optimizer_param_groups'],
                curr['optimizer_param_groups']
            )):
                for key in set(prev_pg.keys()) | set(curr_pg.keys()):
                    prev_val = prev_pg.get(key)
                    curr_val = curr_pg.get(key)
                    if prev_val != curr_val:
                        comparison['changes'].append({
                            'type': 'optimizer',
                            'param_group': j,
                            'key': key,
                            'old': prev_val,
                            'new': curr_val
                        })

        # Compare parameter norms
        if 'total_norm' in prev and 'total_norm' in curr:
            prev_norm = prev['total_norm']
            curr_norm = curr['total_norm']
            rel_change = abs(curr_norm - prev_norm) / prev_norm if prev_norm > 0 else 0

            if rel_change > 0.05:  # 5% threshold
                comparison['changes'].append({
                    'type': 'param_norm',
                    'old_norm': prev_norm,
                    'new_norm': curr_norm,
                    'rel_change': rel_change
                })

        if comparison['changes']:
            comparisons.append(comparison)

    return comparisons


def summarize_training_progress(train_entries: list[dict], val_entries: list[dict]) -> dict:
    """Summarize overall training progress."""
    if not train_entries:
        return {}

    first = train_entries[0]
    last = train_entries[-1]

    summary = {
        'total_steps': last.get('global_step', last.get('step', 0)),
        'total_epochs': last.get('epoch', 0),
        'start_time': first.get('timestamp'),
        'end_time': last.get('timestamp'),
    }

    # Initial and final metrics - map from log keys to summary keys
    metrics_mapping = {
        'loss/total': 'loss_total',
        'loss/main': 'loss_main',
        'metrics/acc_main_batch': 'acc_main',
        'metrics/acc_main_change': 'acc_main_change',
        'gradients/param_total_norm': 'params_total_norm',
    }

    for log_key, summary_key in metrics_mapping.items():
        if log_key in first:
            summary[f'initial_{summary_key}'] = first[log_key]
        if log_key in last:
            summary[f'final_{summary_key}'] = last[log_key]

    # Validation metrics
    if val_entries:
        last_val = val_entries[-1]
        summary['last_val_step'] = last_val.get('step')
        summary['last_val_loss'] = last_val.get('val/loss')
        summary['last_val_acc_main'] = last_val.get('val/acc_main')

    return summary


def detect_interventions(train_entries: list[dict]) -> list[dict]:
    """Detect all manual interventions (LR jumps, param resets, etc.)."""
    interventions = []

    prev_lr = None
    prev_norm = None
    prev_loss = None
    prev_step = None

    for entry in train_entries:
        step = entry.get('global_step', entry.get('step'))
        lr = entry.get('lr')
        norm = entry.get('params/total_norm')
        loss = entry.get('loss/total')

        if step and prev_step:
            # Check for LR jump (beyond normal scheduling)
            if lr and prev_lr:
                ratio = lr / prev_lr if prev_lr > 0 else 1
                if ratio > 1.3 or ratio < 0.77:
                    interventions.append({
                        'type': 'lr_change',
                        'step': step,
                        'prev_step': prev_step,
                        'prev_value': prev_lr,
                        'new_value': lr,
                        'ratio': ratio,
                        'loss_before': prev_loss,
                        'loss_after': loss
                    })

            # Check for param norm discontinuity
            if norm and prev_norm:
                rel_change = abs(norm - prev_norm) / prev_norm
                if rel_change > 0.005:  # 0.5% threshold
                    interventions.append({
                        'type': 'param_reset',
                        'step': step,
                        'prev_step': prev_step,
                        'prev_value': prev_norm,
                        'new_value': norm,
                        'rel_change': rel_change,
                        'loss_before': prev_loss,
                        'loss_after': loss
                    })

        prev_lr = lr
        prev_norm = norm
        prev_loss = loss
        prev_step = step

    return interventions


def analyze_projection_bias(checkpoint_path: Path) -> dict:
    """Analyze projection_down bias for potential explosion issues.

    Returns dict with bias analysis including:
    - Largest bias dimensions
    - Weight correlations for those dimensions
    - Input features driving them
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = checkpoint.get('model', checkpoint)
    config = checkpoint.get('config', {})

    # Handle _orig_mod prefix from torch.compile
    weight_key = '_orig_mod.projection_down.weight'
    bias_key = '_orig_mod.projection_down.bias'

    if weight_key not in state:
        weight_key = 'projection_down.weight'
        bias_key = 'projection_down.bias'

    if weight_key not in state or bias_key not in state:
        return {'error': 'projection_down not found'}

    weight = state[weight_key]
    bias = state[bias_key]

    # Find top 5 biases by magnitude
    bias_mags = bias.abs()
    top_k = min(5, len(bias))
    top_indices = torch.topk(bias_mags, k=top_k).indices.tolist()

    result = {
        'checkpoint': str(checkpoint_path.name),
        'bias_stats': {
            'mean': bias.mean().item(),
            'std': bias.std().item(),
            'max_abs': bias.abs().max().item(),
        },
        'top_biases': [],
        'mirror_pairs': [],
    }

    # Analyze top bias dimensions
    for idx in top_indices:
        result['top_biases'].append({
            'dim': idx,
            'bias': bias[idx].item(),
            'weight_norm': weight[idx].norm().item(),
        })

    # Check if top dims are mirror images (highly anti-correlated)
    if len(top_indices) >= 2:
        for i in range(len(top_indices)):
            for j in range(i+1, len(top_indices)):
                d1, d2 = top_indices[i], top_indices[j]
                w1, w2 = weight[d1], weight[d2]
                corr = torch.corrcoef(torch.stack([w1, w2]))[0, 1].item()
                if corr < -0.9:  # Highly anti-correlated
                    result['mirror_pairs'].append({
                        'dims': (d1, d2),
                        'correlation': corr,
                        'biases': (bias[d1].item(), bias[d2].item()),
                    })

    # Identify input features with largest weights for the problematic dimension
    if top_indices:
        worst_dim = top_indices[0]
        w = weight[worst_dim]
        top_inputs = torch.topk(w.abs(), k=10)
        result['top_input_weights'] = [
            {'input_idx': idx.item(), 'weight': w[idx].item()}
            for idx in top_inputs.indices
        ]

    return result


def analyze_logits(train_entries: list[dict]) -> dict:
    """Analyze logit statistics over training for potential issues.

    Checks for:
    - Logit saturation (values too extreme)
    - Std collapse (model becoming overconfident uniformly)
    - Std explosion (model becoming unstable)

    Returns dict with analysis results and any detected issues.
    """
    import math

    if not train_entries:
        return {"error": "No training entries"}

    # Collect logit stats for each head
    heads = ["main", "c", "buttons", "shoulder"]
    stats = {head: {"mins": [], "maxs": [], "means": [], "stds": []} for head in heads}
    steps = []

    for entry in train_entries:
        step = entry.get("step", entry.get("global_step"))
        if step is None:
            continue
        steps.append(step)

        for head in heads:
            min_key = f"logits/{head}_min"
            max_key = f"logits/{head}_max"
            mean_key = f"logits/{head}_mean"
            std_key = f"logits/{head}_std"

            if min_key in entry:
                stats[head]["mins"].append(entry[min_key])
                stats[head]["maxs"].append(entry[max_key])
                stats[head]["means"].append(entry[mean_key])
                stats[head]["stds"].append(entry[std_key])

    if not steps:
        return {"error": "No logit data found"}

    # Compute ranges and detect issues
    issues = []
    results = {"heads": {}, "issues": issues}

    for head in heads:
        if not stats[head]["mins"]:
            continue

        mins = stats[head]["mins"]
        maxs = stats[head]["maxs"]
        stds = stats[head]["stds"]

        head_result = {
            "min_range": [min(mins), max(mins)],
            "max_range": [min(maxs), max(maxs)],
            "std_range": [min(stds), max(stds)],
            "final_min": mins[-1],
            "final_max": maxs[-1],
            "final_std": stds[-1],
        }
        results["heads"][head] = head_result

        # Check for saturation (logits too extreme)
        # For sigmoid (buttons): |logit| > 20 means sigmoid saturates to 0 or 1
        # For softmax (others): extreme values cause numerical issues
        saturation_threshold = 20 if head == "buttons" else 15

        extreme_min = min(mins)
        extreme_max = max(maxs)
        if extreme_min < -saturation_threshold or extreme_max > saturation_threshold:
            # Check sigmoid/softmax values at extremes
            if head == "buttons":
                # Buttons use sigmoid
                sat_val = 1 / (1 + math.exp(min(-extreme_min, 700)))  # clamp to avoid overflow
                if sat_val < 1e-15:
                    issues.append(f"{head}: logit min {extreme_min:.1f} may cause gradient vanishing (sigmoid → 0)")
            else:
                # Others use softmax - extreme values can cause overflow
                if extreme_max > 30:
                    issues.append(f"{head}: logit max {extreme_max:.1f} approaching softmax overflow risk")

        # Check for std collapse (excluding first few steps which are initialization)
        # Skip first 10% of entries for this check
        skip_n = max(1, len(stds) // 10)
        recent_stds = stds[skip_n:]
        if recent_stds and min(recent_stds) < 0.3:
            issues.append(f"{head}: logit std collapsed to {min(recent_stds):.2f} (overconfident?)")

        # Check for std explosion
        if recent_stds and max(recent_stds) > 10:
            issues.append(f"{head}: logit std exploded to {max(recent_stds):.2f} (unstable?)")

        # Check trend - is min getting more negative over time? (saturation trend)
        # NOTE: For buttons, min trending negative is EXPECTED behavior.
        # Buttons are usually NOT pressed, so the model correctly learns to output
        # negative logits (which map to sigmoid → 0). Values like -16.5 are healthy
        # and indicate the model is confidently predicting "not pressed".
        if len(mins) >= 10:
            first_avg = sum(mins[:5]) / 5
            last_avg = sum(mins[-5:]) / 5
            if last_avg < first_avg - 10:
                # Only flag if it's actually getting extreme (and not buttons)
                if last_avg < -15 and head != "buttons":
                    head_result["trend_warning"] = f"min trending more negative: {first_avg:.1f} → {last_avg:.1f}"

    return results


def print_logit_analysis(train_entries: list[dict]) -> None:
    """Print logit analysis section of the report."""
    print("\n## Logit Statistics")

    analysis = analyze_logits(train_entries)

    if "error" in analysis:
        print(f"  {analysis['error']}")
        return

    # Print per-head summary
    print("\n  Per-head logit ranges (across all training steps):")
    print(f"  {'Head':<10} {'Min Range':>18} {'Max Range':>18} {'Std Range':>18}")
    print("  " + "-" * 66)

    for head, stats in analysis.get("heads", {}).items():
        min_r = stats["min_range"]
        max_r = stats["max_range"]
        std_r = stats["std_range"]
        print(f"  {head:<10} [{min_r[0]:>6.1f}, {min_r[1]:>6.1f}] [{max_r[0]:>6.1f}, {max_r[1]:>6.1f}] [{std_r[0]:>6.2f}, {std_r[1]:>6.2f}]")

        if "trend_warning" in stats:
            print(f"    ⚠️  {stats['trend_warning']}")

    # Print issues
    issues = analysis.get("issues", [])
    print("\n  Issues detected:")
    if issues:
        for issue in issues:
            print(f"    ⚠️  {issue}")
    else:
        print("    ✅ None - logits look healthy")


def print_analysis_report(
    train_entries: list[dict],
    val_entries: list[dict],
    gradient_entries: list[dict],
    checkpoint_analyses: list[dict],
    checkpoint_comparisons: list[dict]
):
    """Print a comprehensive analysis report."""

    print("=" * 80)
    print("TRAINING RUN ANALYSIS REPORT")
    print("=" * 80)

    # Summary
    summary = summarize_training_progress(train_entries, val_entries)
    print("\n## Summary")
    print(f"  Total steps: {summary.get('total_steps', 'N/A'):,}")
    print(f"  Total epochs: {summary.get('total_epochs', 'N/A')}")
    print(f"  Start time: {summary.get('start_time', 'N/A')}")
    print(f"  End time: {summary.get('end_time', 'N/A')}")

    print("\n## Initial vs Final Metrics")
    print(f"  Loss: {fmt(summary.get('initial_loss_total'))} -> {fmt(summary.get('final_loss_total'))}")
    print(f"  Main stick accuracy: {fmt(summary.get('initial_acc_main'))} -> {fmt(summary.get('final_acc_main'))}")
    print(f"  Change accuracy: {fmt(summary.get('initial_acc_main_change'))} -> {fmt(summary.get('final_acc_main_change'))}")
    print(f"  Param norm: {fmt(summary.get('initial_params_total_norm'), '.2f')} -> {fmt(summary.get('final_params_total_norm'), '.2f')}")

    if summary.get('last_val_loss'):
        print(f"\n  Last validation loss: {fmt(summary['last_val_loss'])}")
        print(f"  Last validation acc_main: {fmt(summary.get('last_val_acc_main'))}")

    # Logit analysis
    print_logit_analysis(train_entries)

    # Detect and report all interventions
    interventions = detect_interventions(train_entries)

    print("\n## Manual Interventions Detected")
    if interventions:
        for i, intv in enumerate(interventions, 1):
            if intv['type'] == 'lr_change':
                print(f"\n  [{i}] LR CHANGE at step {intv['step']:,}")
                print(f"      LR: {intv['prev_value']:.2e} → {intv['new_value']:.2e} ({intv['ratio']:.2f}x)")
                if intv['loss_before'] and intv['loss_after']:
                    print(f"      Loss: {intv['loss_before']:.4f} → {intv['loss_after']:.4f}")
            elif intv['type'] == 'param_reset':
                direction = "↓" if intv['new_value'] < intv['prev_value'] else "↑"
                print(f"\n  [{i}] PARAMETER RESET at step {intv['step']:,}")
                print(f"      Norm: {intv['prev_value']:.2f} {direction} {intv['new_value']:.2f} ({intv['rel_change']*100:.2f}% change)")
                if intv['loss_before'] and intv['loss_after']:
                    loss_jump = intv['loss_after'] - intv['loss_before']
                    print(f"      Loss spike: {intv['loss_before']:.4f} → {intv['loss_after']:.4f} (+{loss_jump:.4f})")
    else:
        print("  No manual interventions detected")

    # Learning rate changes
    print("\n## Learning Rate Changes")
    lr_changes = detect_lr_changes(train_entries)
    if lr_changes:
        for change in lr_changes:
            print(f"  Step {change['prev_step']:,} -> {change['step']:,}: "
                  f"LR {change['prev_lr']:.2e} -> {change['new_lr']:.2e} "
                  f"(ratio: {change['ratio']:.2f})")
    else:
        print("  No significant LR changes detected")

    # Parameter norm discontinuities
    print("\n## Parameter Norm Discontinuities")
    discontinuities = detect_param_norm_discontinuities(train_entries)
    if discontinuities:
        for disc in discontinuities:
            direction = "↑" if disc['new_norm'] > disc['prev_norm'] else "↓"
            print(f"  Step {disc['prev_step']:,} -> {disc['step']:,}: "
                  f"Norm {disc['prev_norm']:.2f} {direction} {disc['new_norm']:.2f} "
                  f"({disc['rel_change']*100:.1f}% change)")
    else:
        print("  No discontinuities detected (threshold: 5%)")

    # Checkpoint comparisons
    if checkpoint_comparisons:
        print("\n## Checkpoint Config Changes")
        for comp in checkpoint_comparisons:
            from_name = Path(comp['from']).name
            to_name = Path(comp['to']).name
            print(f"\n  {from_name} -> {to_name}:")
            for change in comp['changes']:
                if change['type'] == 'optimizer':
                    print(f"    Optimizer group {change['param_group']}: "
                          f"{change['key']} = {change['old']} -> {change['new']}")
                elif change['type'] == 'param_norm':
                    print(f"    Param norm: {change['old_norm']:.2f} -> {change['new_norm']:.2f} "
                          f"({change['rel_change']*100:.1f}% change)")

    # Checkpoint details
    if checkpoint_analyses:
        print("\n## Checkpoint Details")
        for analysis in checkpoint_analyses:
            name = Path(analysis['path']).name
            print(f"\n  {name}:")
            print(f"    Step: {analysis.get('global_step', analysis.get('step', 'N/A'))}")
            print(f"    Param norm: {fmt(analysis.get('total_norm'), '.2f')}")

            if 'optimizer_param_groups' in analysis:
                for i, pg in enumerate(analysis['optimizer_param_groups']):
                    lr = pg.get('lr')
                    wd = pg.get('weight_decay')
                    lr_str = f"{lr:.2e}" if lr is not None else "N/A"
                    wd_str = f"{wd}" if wd is not None else "N/A"
                    print(f"    Optimizer group {i}: lr={lr_str}, weight_decay={wd_str}")

    # Gradient statistics
    if gradient_entries:
        print("\n## Gradient Statistics (last 10 entries)")
        for entry in gradient_entries[-10:]:
            step = entry.get('step', 'N/A')
            # Handle both naming conventions
            pre_clip = entry.get('gradients/total_norm_pre_clip', entry.get('grad/total_norm_pre_clip'))
            post_clip = entry.get('gradients/total_norm_post_clip', entry.get('grad/total_norm_post_clip'))
            if pre_clip is not None and post_clip is not None:
                clipped = "CLIPPED" if pre_clip != post_clip else ""
                print(f"  Step {step:,}: pre={pre_clip:.4f}, post={post_clip:.4f} {clipped}")
            else:
                print(f"  Step {step}: gradient norms not available")

    # Projection bias analysis (for detecting bias explosion)
    if checkpoint_analyses:
        print("\n## Projection Bias Analysis")
        latest_ckpt = Path(checkpoint_analyses[-1]['path'])
        bias_analysis = analyze_projection_bias(latest_ckpt)

        if 'error' in bias_analysis:
            print(f"  Error: {bias_analysis['error']}")
        else:
            stats = bias_analysis['bias_stats']
            print(f"  Bias statistics: mean={stats['mean']:.4f}, std={stats['std']:.4f}, max_abs={stats['max_abs']:.4f}")

            if stats['max_abs'] > 5.0:
                print(f"  ⚠️  WARNING: Large bias detected (|bias| > 5.0)")

            print("\n  Top biases by magnitude:")
            for tb in bias_analysis['top_biases']:
                flag = " ⚠️" if abs(tb['bias']) > 5 else ""
                print(f"    Dim {tb['dim']}: bias={tb['bias']:+.4f}, weight_norm={tb['weight_norm']:.4f}{flag}")

            if bias_analysis['mirror_pairs']:
                print("\n  Mirror pairs detected (anti-correlated dimensions):")
                for mp in bias_analysis['mirror_pairs']:
                    print(f"    Dims {mp['dims']}: correlation={mp['correlation']:.4f}")
                    print(f"      biases: {mp['biases'][0]:+.4f} vs {mp['biases'][1]:+.4f}")

            if 'top_input_weights' in bias_analysis:
                print(f"\n  Top input weights for worst dimension (dim {bias_analysis['top_biases'][0]['dim']}):")
                for iw in bias_analysis['top_input_weights'][:5]:
                    print(f"    Input {iw['input_idx']}: weight={iw['weight']:+.4f}")

    # Training curve summary (sample every N steps)
    print("\n## Training Curve (sampled)")
    steps, losses = extract_metric_series(train_entries, 'loss/total')
    if steps:
        # Sample ~20 points
        sample_interval = max(1, len(steps) // 20)
        print("  Step      Loss    Acc_main  Acc_change  Btn_EM")
        print("  " + "-" * 55)

        for i in range(0, len(train_entries), sample_interval):
            entry = train_entries[i]
            step = entry.get('global_step', entry.get('step', 0))
            loss = entry.get('loss/total', 0)
            acc_main = entry.get('metrics/acc_main_batch', 0)
            acc_change = entry.get('metrics/acc_main_change', 0)
            btn_em = entry.get('metrics/buttons_em_batch', 0)
            print(f"  {step:>8,}  {loss:>6.3f}  {acc_main:>8.2%}  {acc_change:>10.2%}  {btn_em:>6.2%}")


def main():
    parser = argparse.ArgumentParser(description='Analyze training run')
    parser.add_argument('--log', type=Path, default=Path('/home/eppie/checkpoints/training_metrics.jsonl'),
                        help='Path to training metrics jsonl file')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('/home/eppie/checkpoints'),
                        help='Directory containing checkpoint files')
    parser.add_argument('--no-checkpoints', action='store_true',
                        help='Skip checkpoint analysis')
    args = parser.parse_args()

    # Load log file
    print(f"Loading {args.log}...")
    entries = load_jsonl(args.log)
    print(f"  Loaded {len(entries):,} entries")

    # Separate by type
    by_type = separate_by_type(entries)
    train_entries = by_type.get('train', [])
    val_entries = by_type.get('val', [])
    gradient_entries = by_type.get('gradient', [])

    print(f"  Train entries: {len(train_entries):,}")
    print(f"  Val entries: {len(val_entries):,}")
    print(f"  Gradient entries: {len(gradient_entries):,}")

    # Analyze checkpoints
    checkpoint_analyses = []
    checkpoint_comparisons = []

    if not args.no_checkpoints and args.checkpoint_dir.exists():
        print(f"\nAnalyzing checkpoints in {args.checkpoint_dir}...")
        checkpoint_analyses = analyze_checkpoints(args.checkpoint_dir)
        checkpoint_comparisons = compare_checkpoints(checkpoint_analyses)

    # Print report
    print_analysis_report(
        train_entries,
        val_entries,
        gradient_entries,
        checkpoint_analyses,
        checkpoint_comparisons
    )


if __name__ == '__main__':
    main()
