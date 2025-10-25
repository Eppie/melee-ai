"""Logging, formatting, and display utilities."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from libmelee.melee.enums import Action

# Action enum mapping for formatting
_ACTION_VALUE_TO_NAME = {action.value: action.name for action in Action}


def _format_value(value: object) -> str:
    """Format numeric values with up to 6 significant figures."""
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _format_action(value: object) -> str:
    """Format action enum values to their names."""
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


def print_batch_preview(
    batch: Dict[str, torch.Tensor],
    feature_names: Sequence[str],
    target_names: Sequence[str],
    *,
    max_frames: int = 10,
) -> None:
    """Pretty-print the first sequence from the first batch for manual inspection."""
    X = batch["X"].detach().cpu()
    Y = batch["Y"].detach().cpu() if batch["Y"].numel() else None

    first_seq = X[0]
    num_frames = min(max_frames, first_seq.shape[0])
    feat_slice = first_seq[:num_frames].numpy()

    print(
        "=== First batch preview (sequence 0, first {num_frames} frames) ===".format(
            num_frames=num_frames
        )
    )
    formatters: Dict[str, Callable[[object], str]] = {}
    for key in feature_names:
        if key.endswith("_action"):
            formatters[key] = _format_action

    _print_table_block(
        "Feature preview",
        feature_names,
        feat_slice,
        formatters=formatters,
    )

    if Y is not None and Y.shape[-1] > 0:
        target_slice = Y[0, :num_frames].numpy()
        _print_table_block(
            "Target preview",
            target_names,
            target_slice,
        )


def _top_confusions(cm: torch.Tensor, k: int = 8) -> List[Tuple[int, int, int, float]]:
    """Return top-k off-diagonal confusions as (true, pred, count, pct_of_offdiag)."""
    cm_np = cm.numpy()
    off = cm_np.copy()
    np.fill_diagonal(off, 0)
    total_off = off.sum()
    if total_off <= 0:
        return []
    flat = off.ravel()
    idx = np.argpartition(flat, -k)[-k:]
    idx = idx[np.argsort(flat[idx])[::-1]]
    out: List[Tuple[int, int, int, float]] = []
    W = cm_np.shape[1]
    for f in idx:
        cnt = int(flat[f])
        if cnt <= 0:
            continue
        i = int(f // W)
        j = int(f % W)
        pct = 100.0 * cnt / float(total_off)
        out.append((i, j, cnt, pct))
    return out


def format_confusion_matrix(
    cm: torch.Tensor,
    max_size: int = 12,
    title: str = "confusion",
    labels: Optional[Sequence[str]] = None,
) -> str:
    """Render a confusion matrix or its top confusions in a compact string.

    Args:
        cm: [K, K] confusion matrix
        max_size: Maximum size to show full matrix (otherwise show top confusions)
        title: Title for the output
        labels: Optional label names for rows/columns

    Returns:
        Formatted string representation
    """
    K = cm.shape[0]

    if labels is not None:
        if len(labels) != K:
            raise ValueError("labels length must match confusion matrix dimensions")
        label_list: Sequence[str] = [str(lbl) for lbl in labels]
    else:
        label_list = [f"{i:02d}" for i in range(K)]

    if K > max_size:
        tops = _top_confusions(cm, k=10)
        if not tops:
            return f"{title}: (no confusions)"
        lines = [f"{title}: top confusions (true->pred: count, %offdiag)"]
        lines += [
            f"  {label_list[t]}->{label_list[p]}: {c} ({pct:.1f}%)"
            for t, p, c, pct in tops
        ]
        return "\n".join(lines)

    arr = cm.numpy()
    row_sums = arr.sum(axis=1)
    diag_vals = np.diag(arr).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        diag_pct = np.divide(
            diag_vals * 100.0,
            row_sums,
            out=np.zeros_like(diag_vals, dtype=float),
            where=row_sums > 0,
        )

    cell_width = max(4, max(len(lbl) for lbl in label_list))
    header = (
        " " * (cell_width + 1)
        + " ".join(lbl.rjust(cell_width) for lbl in label_list)
        + " | sum"
    )
    lines = [f"{title}: full {K}x{K}", header]
    for i in range(K):
        row = " ".join(f"{int(v):>{cell_width}d}" for v in arr[i])
        lines.append(
            f"{label_list[i].rjust(cell_width)}: {row} | {int(row_sums[i]):>{cell_width}d}"
        )
    lines.append("diag% per row: " + " ".join(f"{p:>5.1f}" for p in diag_pct))
    return "\n".join(lines)


def format_metrics_dict(metrics: Dict[str, float], precision: int = 3) -> str:
    """Format a metrics dictionary as a compact string.

    Args:
        metrics: Dictionary of metric names to values
        precision: Number of decimal places

    Returns:
        Formatted string like "acc=0.950, f1=0.823"
    """
    parts = [f"{key}={value:.{precision}f}" for key, value in sorted(metrics.items())]
    return ", ".join(parts)


def format_loss_summary(losses: Dict[str, float]) -> str:
    """Format a loss dictionary as a compact string.

    Args:
        losses: Dictionary of loss component names to values

    Returns:
        Formatted string like "total=1.234 (main=0.5, c=0.3, btn=0.4)"
    """
    total = losses.get("total", sum(v for k, v in losses.items() if k != "total"))
    components = [f"{k}={v:.4f}" for k, v in sorted(losses.items()) if k != "total"]
    if components:
        return f"total={total:.4f} ({', '.join(components)})"
    else:
        return f"total={total:.4f}"


def format_training_progress(
    epoch: int,
    total_epochs: int,
    step: int,
    total_steps: int,
    loss: float,
    metrics: Dict[str, float],
) -> str:
    """Format a training progress message.

    Args:
        epoch: Current epoch (0-indexed)
        total_epochs: Total number of epochs
        step: Current step within epoch
        total_steps: Total steps per epoch
        loss: Current loss value
        metrics: Dictionary of metrics to display

    Returns:
        Formatted progress string
    """
    metric_str = format_metrics_dict(metrics)
    return f"[Epoch {epoch+1}/{total_epochs}, Step {step}/{total_steps}] loss={loss:.4f} | {metric_str}"
