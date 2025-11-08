"""Logging, formatting, and display utilities."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from libmelee.melee.enums import Action

# Action enum mapping for formatting
_ACTION_VALUE_TO_NAME = {action.value: action.name for action in Action}


def _format_value(value: object) -> str:
    """Format numbers to six significant figures while leaving other objects untouched.

    Example:
        Calling ``_format_value(0.123456789)`` casts the input to ``float`` and formats it as the
        string ``"0.123457"``. Passing ``_format_value("noop")`` skips numeric formatting and
        simply returns ``"noop"``. The example demonstrates how the helper branches between numeric
        and non-numeric inputs.

    Args:
        value: Value to format, typically a scalar extracted from a tensor or array.

    Returns:
        String representation with limited significant figures for numeric inputs.
    """
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return str(value)


def _format_action(value: object) -> str:
    """Convert action enumeration indices back into descriptive names.

    Example:
        Suppose ``value`` is ``4`` and ``Action(4).name`` equals ``"JUMP"``. The helper casts the
        value to ``int`` and looks it up in ``_ACTION_VALUE_TO_NAME``, producing ``"JUMP"``. If the
        value were ``99`` (not a valid key), it would fall back to ``"99"``. This illustrates how the
        function either resolves human-friendly labels or echoes the numeric input.

    Args:
        value: Numeric or string representation of the action index.

    Returns:
        Action name string if available, otherwise the original value as a string.
    """
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
    """Render a slice of ``data`` as a human-readable table, respecting ``max_columns``.

    Example:
        Given ``headers = ["x", "y"]`` and ``data = np.array([[1.0, 2.0], [3.0, 4.0]])``, the
        helper prints::

            Example (columns 1-2 of 2):
                 frame      x      y
                 0          1      2
                 1          3      4

        It iterates column blocks of size ``max_columns``, formats each cell (using custom
        ``formatters`` when provided), right-justifies the headers, and prints each row preceded by
        the frame index. This step-by-step process mirrors how batches are previewed in logs.

    Args:
        title: Label printed above the table block.
        headers: Column names corresponding to the last dimension of ``data``.
        data: Two-dimensional slice to display (``num_rows`` by ``num_cols``).
        max_columns: Maximum number of columns to display per block.
        formatters: Optional mapping from column name to custom formatter function.
    """
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
    """Preview the first sequence of a batch, showing how tensors map to readable tables.

    Example:
        Imagine ``batch["X"]`` is shaped ``[2, 4, 3]`` with the first sequence::

            [[0., 10., 0.1],
             [1., 11., 0.2],
             [2., 12., 0.3],
             [3., 13., 0.4]]

        and ``feature_names = ["frame", "action", "value"]``. ``print_batch_preview`` slices the
        first ``max_frames`` rows (``4`` here), converts them to ``numpy``, and passes them to
        :func:`_print_table_block`, resulting in a console table where the ``action`` column is
        formatted via :func:`_format_action` if applicable. The example demonstrates how tensor data
        becomes a readable summary for quick debugging.

    Args:
        batch: Dictionary containing ``"X"`` features and optionally ``"Y"`` targets.
        feature_names: Names corresponding to the feature columns.
        target_names: Names corresponding to the target columns.
        max_frames: Maximum number of frames to display.
    """
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
    """Extract the ``k`` most common misclassifications from a confusion matrix.

    Example:
        For a ``3x3`` matrix::

            [[5, 2, 0],
             [1, 7, 0],
             [0, 3, 4]]

        The off-diagonal counts sum to ``6``. Calling ``_top_confusions(cm, k=2)`` returns
        ``[(2, 1, 3, 50.0), (0, 1, 2, 33.3)]`` meaning "class 2 predicted as 1" occurred three times
        (50% of off-diagonal mistakes) and "class 0 predicted as 1" occurred twice (33.3%). The
        walkthrough shows how counts are flattened, sorted, and converted back into coordinates.

    Args:
        cm: ``[K, K]`` confusion matrix.
        k: Number of top misclassifications to return.

    Returns:
        List of tuples ``(true_idx, pred_idx, count, percent_of_off_diag)``.
    """
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
    """Format a confusion matrix as text, falling back to top errors for large matrices.

    Example:
        With the ``3x3`` matrix from :func:`_top_confusions`, ``format_confusion_matrix`` first sees
        that ``K=3`` is smaller than ``max_size``. It therefore prints the full table::

            confusion: full 3x3
                     | 00 01 02 | sum
                 00:   5  2  0 |  7
                 01:   1  7  0 |  8
                 02:   0  3  4 |  7
            diag% per row: 71.4 87.5 57.1

        If ``K`` exceeded ``max_size``, it would call :func:`_top_confusions` and list the top
        misclassifications instead. This demonstrates both code paths.

    Args:
        cm: ``[K, K]`` confusion matrix.
        max_size: Largest matrix dimension that still prints in full.
        title: Title prefix for the formatted output.
        labels: Optional human-readable labels for rows and columns.

    Returns:
        Multi-line string summarizing the confusion matrix or its top confusions.
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
    """Serialize metric key-value pairs into a comma-separated string.

    Example:
        For ``metrics = {"loss": 1.23456, "acc": 0.98765}`` and ``precision=2``, the helper sorts
        the keys alphabetically, formats each number to two decimals, and joins them to produce
        ``"acc=0.99, loss=1.23"``. This shows how each metric passes through formatting before being
        returned.

    Args:
        metrics: Dictionary of metric names to values.
        precision: Number of decimal places to display.

    Returns:
        Comma-separated ``key=value`` pairs sorted by key.
    """
    parts = [f"{key}={value:.{precision}f}" for key, value in sorted(metrics.items())]
    return ", ".join(parts)


def format_loss_summary(losses: Dict[str, float]) -> str:
    """Summarize loss components, highlighting the total and per-component values.

    Example:
        Given ``{"total": 1.2, "main": 0.5, "btn": 0.7}``, the helper builds the string
        ``"total=1.2000 (btn=0.7000, main=0.5000)"`` by separating the total from the other keys and
        formatting each to four decimals. If ``total`` were absent it would sum the components first.
        The example traces how inputs are reordered and combined into the final message.

    Args:
        losses: Dictionary of loss component names to values.

    Returns:
        Readable summary string emphasizing the total loss.
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
    """Compose a progress string combining epoch, step, loss, and metrics.

    Example:
        With ``epoch=1``, ``total_epochs=5``, ``step=32``, ``total_steps=100``, ``loss=0.4321``, and
        ``metrics={"acc": 0.98}``, the helper formats the metrics via :func:`format_metrics_dict`
        (yielding ``"acc=0.980"``) and returns the message ``"[Epoch 2/5, Step 32/100] loss=0.4321 |
        acc=0.980"``. The example highlights how values are combined and 0-indexed epochs are
        converted to human-friendly 1-indexed numbers.

    Args:
        epoch: Current 0-indexed epoch.
        total_epochs: Total number of epochs that will run.
        step: Current step within the epoch.
        total_steps: Total steps per epoch.
        loss: Current loss value.
        metrics: Dictionary of metrics to display.

    Returns:
        Formatted progress string suitable for logging.
    """
    metric_str = format_metrics_dict(metrics)
    return f"[Epoch {epoch+1}/{total_epochs}, Step {step}/{total_steps}] loss={loss:.4f} | {metric_str}"
