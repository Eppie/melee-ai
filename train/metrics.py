"""Metrics computation and accumulation for training."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


class StickMetrics:
    def __init__(self, K: int, device: torch.device):
        self.correct = torch.tensor(0, dtype=torch.long, device=device)
        self.total = torch.tensor(0, dtype=torch.long, device=device)
        self.label_counts = torch.zeros(K, dtype=torch.long, device=device)
        self.maj_correct = torch.tensor(0, dtype=torch.long, device=device)

    def update(self, pred_idx: torch.Tensor, true_idx: torch.Tensor, majority_baseline: int):
        self.correct += (pred_idx == true_idx).sum()
        self.total += true_idx.numel()
        bincount = torch.bincount(true_idx, minlength=self.label_counts.shape[0])
        self.label_counts += bincount[: self.label_counts.shape[0]]
        self.maj_correct += (true_idx == majority_baseline).sum()

    def reset(self):
        self.correct.zero_()
        self.total.zero_()
        self.label_counts.zero_()
        self.maj_correct.zero_()

class PositionMetrics:
    def __init__(self, K: int, device: torch.device):
        self.correct = torch.tensor(0, dtype=torch.long, device=device)
        self.total = torch.tensor(0, dtype=torch.long, device=device)
        self.label_counts = torch.zeros(K, dtype=torch.long, device=device)
        self.maj_correct = torch.tensor(0, dtype=torch.long, device=device)
        self.spatial_error_sum = torch.tensor(0.0, dtype=torch.float32, device=device)

    def update(self, pred_idx: torch.Tensor, true_idx: torch.Tensor, majority_baseline: int, spatial_error_sum: torch.Tensor):
        self.correct += (pred_idx == true_idx).sum()
        self.total += true_idx.numel()
        bincount = torch.bincount(true_idx, minlength=self.label_counts.shape[0])
        self.label_counts += bincount[: self.label_counts.shape[0]]
        self.maj_correct += (true_idx == majority_baseline).sum()
        self.spatial_error_sum += spatial_error_sum

    def reset(self):
        self.correct.zero_()
        self.total.zero_()
        self.label_counts.zero_()
        self.maj_correct.zero_()
        self.spatial_error_sum.zero_()


# TODO: Do we really need this? Check sweep.py.
class MetricsAccumulator:
    """Stateful metrics tracker for training/validation.

    Tracks stick, button, and shoulder metrics with running tallies.
    Replaces the old RunningMetrics pattern with cleaner implementation.
    """

    device: torch.device
    K_main: int
    K_c: int
    K_buttons: int
    K_shoulder: int

    # Main stick metrics
    main_correct: torch.Tensor
    main_total: torch.Tensor
    main_label_counts: torch.Tensor
    main_maj_correct: torch.Tensor

    # C-stick metrics
    c_correct: torch.Tensor
    c_total: torch.Tensor
    c_label_counts: torch.Tensor
    c_maj_correct: torch.Tensor

    # Button metrics
    btn_true_positives: torch.Tensor
    btn_false_positives: torch.Tensor
    btn_false_negatives: torch.Tensor
    btn_pos_counts: torch.Tensor
    btn_total: torch.Tensor
    btn_em_correct: torch.Tensor
    btn_maj_em_correct: torch.Tensor

    # Shoulder metrics
    shoulder_correct: torch.Tensor
    shoulder_total: torch.Tensor
    shoulder_label_counts: torch.Tensor
    shoulder_maj_correct: torch.Tensor

    def __init__(
        self,
        K_main: int,
        K_c: int,
        K_buttons: int,
        K_shoulder: int,
        device: torch.device,
    ):
        """Initialize tensors that accumulate accuracy-style statistics.

        Example:
            Instantiating ``MetricsAccumulator(3, 2, 4, 0, torch.device("cpu"))`` allocates zeroed
            tensors such as ``self.main_correct`` and ``self.btn_tp`` with shapes derived from the
            supplied ``K_*`` values. After initialization you can call
            :meth:`update_stick_metrics` with predicted vs. true indices and the counters increment
            accordingly, demonstrating how the constructor merely prepares storage while deferring
            computation to the update methods.

        Args:
            K_main: Number of main stick quantization bins.
            K_c: Number of C-stick quantization bins.
            K_buttons: Number of button outputs.
            K_shoulder: Number of shoulder quantization bins.
            device: Device on which running totals should be stored.
        """
        self.device = device
        self.K_main = K_main
        self.K_c = K_c
        self.K_buttons = K_buttons
        self.K_shoulder = K_shoulder

        self.main = StickMetrics(K_main, device)
        self.c = StickMetrics(K_c, device)
        self.shoulder = StickMetrics(K_shoulder, device)

        # Button metrics
        self.btn_true_positives = torch.zeros(
            K_buttons, dtype=torch.float32, device=device
        )
        self.btn_false_positives = torch.zeros(
            K_buttons, dtype=torch.float32, device=device
        )
        self.btn_false_negatives = torch.zeros(
            K_buttons, dtype=torch.float32, device=device
        )
        self.btn_pos_counts = torch.zeros(K_buttons, dtype=torch.float32, device=device)
        self.btn_total = torch.tensor(0, dtype=torch.long, device=device)
        self.btn_em_correct = torch.tensor(0, dtype=torch.long, device=device)
        self.btn_maj_em_correct = torch.tensor(0, dtype=torch.long, device=device)

    def _majority_label(self, label_counts: torch.Tensor) -> int:
        """Return the index of the largest count, defaulting to zero for empty tensors.

        Example:
            If ``label_counts`` equals ``tensor([2, 5, 3])`` the method returns ``1`` because index 1
            has the highest count ``5``. For an empty tensor ``tensor([])`` it immediately returns
            ``0``. The example mirrors how the helper uses :func:`torch.argmax` to select the most
            frequent label.

        Args:
            label_counts: Histogram of observed labels.

        Returns:
            Integer index of the majority label or ``0`` when ``label_counts`` has no elements.
        """
        if label_counts.numel() == 0:
            return 0
        return int(torch.argmax(label_counts).item())

    def update_stick_metrics(
        self,
        pred_idx: torch.Tensor,
        true_idx: torch.Tensor,
        stick_type: str,
        majority_baseline: int,
        repeat_baseline: torch.Tensor,
        repeat_mask: torch.Tensor,
    ) -> None:
        """Update running accuracy counters for either the main stick or C-stick outputs.

        Example:
            Suppose the true indices are ``tensor([0, 1, 2])`` and the predictions are
            ``tensor([0, 0, 2])`` for ``stick_type="main"``. The method increments
            ``self.main_correct`` by ``2`` (frames 0 and 2 match) and ``self.main_total`` by ``3``.
            If ``majority_baseline=0`` it also adds ``1`` to ``self.main_maj_correct`` because only
            the first frame matches the baseline. Passing ``repeat_baseline=tensor([0, 1, 1])`` with a
            ``repeat_mask`` that marks frames ``[False, True, True]`` adds one extra correct repeat to
            ``self.main_rep_correct`` (frame 2) and increases ``self.main_rep_total`` by ``2``. The
            walk-through demonstrates how each argument influences the tracked counters.

        Args:
            pred_idx: Predicted indices ``[N]``.
            true_idx: True indices ``[N]``.
            stick_type: ``"main"`` or ``"c"`` selecting which counters to update.
            majority_baseline: Majority label accuracy baseline.
            repeat_baseline: Repeat prediction vector ``[N]``.
            repeat_mask: Boolean mask ``[N]`` for frames eligible for repeat accuracy.
        """
        if stick_type == "main":
            self.main.update(pred_idx, true_idx, majority_baseline)
        elif stick_type == "c":
            self.c.update(pred_idx, true_idx, majority_baseline)
        else:
            raise ValueError(f"Unknown stick type: {stick_type}")

    def update_button_metrics(
        self,
        pred_buttons: torch.Tensor,
        true_buttons: torch.Tensor,
    ) -> None:
        """Update precision/recall-style counters and exact match for button predictions.

        Example:
            For a batch with ``batch_size=1``, ``sequence_length=2``, ``num_buttons=2`` where ``true_buttons`` equals
            ``[[[1, 0], [0, 1]]]`` and ``pred_buttons`` equals ``[[[1, 0], [1, 1]]]``:

            * The method flattens both tensors to ``[[1, 0], [0, 1]]`` vs. ``[[1, 0], [1, 1]]``.
            * True positives become ``[1, 1]`` (buttons 0 and 1 each correct once).
            * False positives become ``[1, 0]`` (button 0 predicted on frame 1 where it should be 0).
            * Exact match counts only the first frame, so ``self.btn_em_correct`` increases by ``1``
              while ``self.btn_total`` increases by ``2``.

            The example shows how each accumulation is derived step by step.

        Args:
            pred_buttons: Predicted button states ``[batch_size, sequence_length, num_buttons]``.
            true_buttons: True button states ``[batch_size, sequence_length, num_buttons]``.
        """
        batch_size, sequence_length, num_buttons = pred_buttons.shape

        # Flatten
        pred_flat = pred_buttons.reshape(-1, num_buttons).float()
        true_flat = true_buttons.reshape(-1, num_buttons).float()

        # TP, FP, FN
        self.btn_true_positives += (pred_flat * true_flat).sum(dim=0)
        self.btn_false_positives += (pred_flat * (1 - true_flat)).sum(dim=0)
        self.btn_false_negatives += ((1 - pred_flat) * true_flat).sum(dim=0)

        # Positive counts
        self.btn_pos_counts += true_flat.sum(dim=0)

        # Total frames
        self.btn_total += batch_size * sequence_length

        # Exact match
        em_correct = (pred_buttons == true_buttons).all(dim=-1).sum()
        self.btn_em_correct += em_correct

        # Majority baseline (all zeros or all ones depending on majority)
        pos_rate = true_flat.mean(dim=0)
        maj_pred = (pos_rate >= 0.5).float().unsqueeze(0).expand_as(pred_flat)
        maj_em = (
            (maj_pred.reshape(batch_size, sequence_length, num_buttons) == true_buttons)
            .all(dim=-1)
            .sum()
        )
        self.btn_maj_em_correct += maj_em

    def update_shoulder_metrics(
        self,
        pred_idx: torch.Tensor,
        true_idx: torch.Tensor,
        majority_baseline: int,
    ) -> None:
        """Track accuracy for shoulder trigger quantization bins.

        Example:
            If ``self.K_shoulder`` is ``3`` and ``true_idx`` equals ``[[0, 1]]`` while
            ``pred_idx`` equals ``[[0, 2]]``, the method increments ``self.shoulder_correct`` by ``1``
            (only the first frame matches) and ``self.shoulder_total`` by ``2``. When
            ``majority_baseline=0`` it adds another ``1`` to ``self.shoulder_maj_correct`` because the
            baseline matches frame 0. The tensor ``self.shoulder_label_counts`` also updates to
            ``[1, 1, 0]`` to reflect how often each class appears. This example makes the counter
            updates explicit.

        Args:
            pred_idx: Predicted indices ``[B, L]``.
            true_idx: True indices ``[B, L]``.
            majority_baseline: Majority label for baseline comparisons.
        """
        pred_flat = pred_idx.reshape(-1)
        true_flat = true_idx.reshape(-1)

        self.shoulder.update(pred_flat, true_flat, majority_baseline)

    def get_summary(self) -> Dict[str, float]:
        """Convert accumulated counters into scalar metrics ready for logging.

        Example:
            After calling :meth:`update_button_metrics` for the example above, where
            ``self.btn_em_correct == 1`` and ``self.btn_total == 2``, ``get_summary()`` computes
            ``btn_em = 1 / 2 = 0.5``. Similarly, if ``self.main_correct == 8`` and ``self.main_total == 10``
            it reports ``acc_main = 0.8``. The method performs these divisions for every component and
            bundles the results into a dictionary, demonstrating the final aggregation step.

        Returns:
            Dictionary of scalar metrics such as ``acc_main`` and ``btn_f1_micro``.
        """
        # Batch ALL scalar transfers into a single GPU->CPU operation
        # This replaces 16+ individual .item() calls with ONE transfer
        scalar_tensors = [
            self.main.correct,
            self.main.total,
            self.main.maj_correct,
            self.c.correct,
            self.c.total,
            self.c.maj_correct,
            self.btn_em_correct,
            self.btn_total,
            self.btn_maj_em_correct,
        ]

        # Add shoulder metrics if present
        if self.K_shoulder > 0:
            scalar_tensors.extend([
                self.shoulder.correct,
                self.shoulder.total,
                self.shoulder.maj_correct,
            ])

        # Single batched transfer (ONE sync instead of 16+)
        scalars_stacked = torch.stack([t.float() for t in scalar_tensors])
        scalars_cpu = scalars_stacked.cpu().tolist()

        # Unpack values (no syncs!)
        idx = 0
        main_correct = scalars_cpu[idx]; idx += 1
        main_total = scalars_cpu[idx]; idx += 1
        main_maj_correct = scalars_cpu[idx]; idx += 1
        c_correct = scalars_cpu[idx]; idx += 1
        c_total = scalars_cpu[idx]; idx += 1
        c_maj_correct = scalars_cpu[idx]; idx += 1
        btn_em_correct = scalars_cpu[idx]; idx += 1
        btn_total = scalars_cpu[idx]; idx += 1
        btn_maj_em_correct = scalars_cpu[idx]; idx += 1

        if self.K_shoulder > 0:
            shoulder_correct = scalars_cpu[idx]; idx += 1
            shoulder_total = scalars_cpu[idx]; idx += 1
            shoulder_maj_correct = scalars_cpu[idx]; idx += 1

        # Main stick metrics
        summary = {
            "acc_main": main_correct / max(1.0, main_total),
            "acc_main_maj": main_maj_correct / max(1.0, main_total),
        }

        # C-stick metrics
        summary["acc_c"] = c_correct / max(1.0, c_total)
        summary["acc_c_maj"] = c_maj_correct / max(1.0, c_total)

        # Button metrics - batch array transfers too
        # Stack all button arrays and transfer once (3 syncs -> 1 sync)
        btn_arrays_stacked = torch.stack([
            self.btn_true_positives,
            self.btn_false_positives,
            self.btn_false_negatives,
        ])
        btn_arrays_cpu = btn_arrays_stacked.cpu().numpy()

        tp = btn_arrays_cpu[0]
        fp = btn_arrays_cpu[1]
        fn = btn_arrays_cpu[2]

        # Micro-averaged metrics (computed on CPU after single transfer)
        tp_sum = tp.sum()
        fp_sum = fp.sum()
        fn_sum = fn.sum()
        prec_micro = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        rec_micro = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        f1_micro = (
            2 * prec_micro * rec_micro / (prec_micro + rec_micro)
            if (prec_micro + rec_micro) > 0
            else 0.0
        )

        # Macro-averaged metrics
        prec_per_button = np.divide(
            tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0
        )
        rec_per_button = np.divide(
            tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0
        )
        f1_per_button = np.divide(
            2 * prec_per_button * rec_per_button,
            prec_per_button + rec_per_button,
            out=np.zeros_like(tp),
            where=(prec_per_button + rec_per_button) > 0,
        )
        f1_macro = f1_per_button.mean()

        summary["btn_em"] = btn_em_correct / max(1.0, btn_total)
        summary["btn_prec_micro"] = float(prec_micro)
        summary["btn_rec_micro"] = float(rec_micro)
        summary["btn_f1_micro"] = float(f1_micro)
        summary["btn_f1_macro"] = float(f1_macro)
        summary["btn_em_maj"] = btn_maj_em_correct / max(1.0, btn_total)

        # Shoulder metrics
        if self.K_shoulder > 0:
            summary["acc_shoulder"] = shoulder_correct / max(1.0, shoulder_total)
            summary["acc_shoulder_maj"] = shoulder_maj_correct / max(1.0, shoulder_total)

        return summary

    def reset(self) -> None:
        """Zero out every counter so the accumulator can be reused for a new epoch.

        Example:
            After multiple updates the tensor ``self.main_correct`` might equal ``tensor(42)``.
            Calling ``reset()`` sets it back to ``tensor(0)`` using in-place ``zero_()`` operations on
            every field, including nested tensors such as ``self.btn_tp``. The example highlights how
            the method prepares the accumulator for the next round of tracking.
        """
        self.main.reset()
        self.c.reset()
        self.shoulder.reset()

        # Buttons
        self.btn_true_positives.zero_()
        self.btn_false_positives.zero_()
        self.btn_false_negatives.zero_()
        self.btn_pos_counts.zero_()
        self.btn_total.zero_()
        self.btn_em_correct.zero_()
        self.btn_maj_em_correct.zero_()


def compute_confusion_matrix(
    true_flat: torch.Tensor, pred_flat: torch.Tensor, K: int
) -> torch.Tensor:
    """Build a confusion matrix by counting ``(true, pred)`` index pairs.

    Example:
        With ``true_flat = tensor([0, 1, 1, 2])``, ``pred_flat = tensor([0, 2, 1, 2])`` and ``K = 3``,
        the function forms the combined indices ``true * K + pred`` → ``[0, 5, 4, 8]``. The
        ``torch.bincount`` call counts each occurrence, reshaping to::

            [[1, 0, 0],
             [0, 1, 1],
             [0, 0, 1]]

        representing how often each class was predicted. The example precisely traces the tensor
        operations used to populate the matrix.

    Args:
        true_flat: True labels ``[N]``.
        pred_flat: Predicted labels ``[N]``.
        K: Number of classes.

    Returns:
        ``[K, K]`` confusion matrix on CPU.
    """
    tf = true_flat.to(torch.int64)
    pf = pred_flat.to(torch.int64)
    cm = torch.bincount(tf * K + pf, minlength=K * K).view(K, K)
    return cm.cpu()


def compute_binary_rates(
    true_positives: torch.Tensor,
    false_positives: torch.Tensor,
    false_negatives: torch.Tensor,
    total_count: float | torch.Tensor,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute TPR/TNR/FPR/FNR and true negatives from confusion counts.

    Args:
        true_positives: Per-class true positive counts.
        false_positives: Per-class false positive counts.
        false_negatives: Per-class false negative counts.
        total_count: Total number of examples per class (scalar).
        eps: Small constant to avoid division by zero.

    Returns:
        Tuple of tensors ``(tpr, tnr, fpr, fnr, tn)`` each shaped like the inputs.
    """
    tp = true_positives.float()
    fp = false_positives.float()
    fn = false_negatives.float()
    total = (
        total_count.to(tp)
        if isinstance(total_count, torch.Tensor)
        else tp.new_tensor(float(total_count))
    )

    positives = tp + fn
    negatives = total - positives
    tn = torch.clamp(negatives - fp, min=0.0)

    pos_den = positives + eps
    neg_den = tn + fp + eps

    tpr = tp / pos_den
    fnr = fn / pos_den
    tnr = tn / neg_den
    fpr = fp / neg_den

    return tpr, tnr, fpr, fnr, tn


# TODO: Optimize? are there more useful metrics? expose precision and recall here too?
# TODO: What is the difference between micro and macro?
def multilabel_prf(
    true_labels: torch.Tensor, pred_labels: torch.Tensor
) -> Tuple[float, float, float, float, float]:
    """Calculate multi-label exact match, micro precision/recall/F1, and macro F1.

    Example:
        With ``true_labels = tensor([[[1, 0], [0, 1]]])`` and
        ``pred_labels = tensor([[[1, 1], [0, 1]]])``:

        * Flattening yields two frames with predictions ``[[1, 1], [0, 1]]`` and truths
          ``[[1, 0], [0, 1]]``.
        * True positives per class are ``[1, 1]``, false positives ``[0, 1]``, and false negatives
          ``[0, 0]`` giving micro precision ``2/3`` and micro recall ``2/2 = 1`` → micro F1 ``0.8``.
        * Macro F1 averages the per-class F1 scores: button 0 has precision ``1`` and recall ``1``
          (F1 ``1``), button 1 has precision ``0.5`` and recall ``1`` (F1 ``2/3``), averaging to
          ``0.8333``.
        * Exact match checks each frame: the first differs because of button 1, so the final value is
          ``0.5``.

        The example exposes every intermediate count used in the calculations.

    Args:
        true_labels: True labels ``[B, L, K]`` or ``[N, K]``.
        pred_labels: Predicted labels ``[B, L, K]`` or ``[N, K]``.

    Returns:
        Tuple ``(exact_match, precision_micro, recall_micro, f1_micro, f1_macro)``.
    """
    # Ensure 3D
    if true_labels.dim() == 2:
        true_labels = true_labels.unsqueeze(0)
        pred_labels = pred_labels.unsqueeze(0)

    t = true_labels.bool()
    p = pred_labels.bool()

    # Compute all metrics on GPU first
    tp_micro = (t & p).sum()
    fp_micro = ((~t) & p).sum()
    fn_micro = (t & (~p)).sum()
    tp_per_class = (t & p).sum(dim=(0, 1))
    fp_per_class = ((~t) & p).sum(dim=(0, 1))
    fn_per_class = (t & (~p)).sum(dim=(0, 1))
    em_gpu = (pred_labels == true_labels).all(dim=-1).float().mean()

    # Batch ALL transfers: scalars + per-class arrays in one go
    # Stack scalars and transfer
    scalars = torch.stack([tp_micro, fp_micro, fn_micro, em_gpu]).float()
    scalars_cpu = scalars.cpu().tolist()
    tp = scalars_cpu[0]
    fp = scalars_cpu[1]
    fn = scalars_cpu[2]
    em = scalars_cpu[3]

    # Batch per-class arrays
    per_class_stacked = torch.stack([tp_per_class, fp_per_class, fn_per_class])
    per_class_cpu = per_class_stacked.cpu().numpy()
    tp_c = per_class_cpu[0]
    fp_c = per_class_cpu[1]
    fn_c = per_class_cpu[2]

    # Micro metrics (computed on CPU after single transfer)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Macro metrics (per class)
    f1_c = []
    for a, b, c in zip(tp_c, fp_c, fn_c):
        pr = a / (a + b) if (a + b) > 0 else 0.0
        rc = a / (a + c) if (a + c) > 0 else 0.0
        f1_c.append(2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0)
    f1_macro = float(np.mean(f1_c) if len(f1_c) else 0.0)

    return em, float(prec), float(rec), float(f1), f1_macro
