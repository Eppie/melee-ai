"""Metrics computation and accumulation for training."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

# TODO: Do we really need this? Check sweep.py.
class MetricsAccumulator:
    """Stateful metrics tracker for training/validation.

    Tracks stick, button, and shoulder metrics with running tallies.
    Replaces the old RunningMetrics pattern with cleaner implementation.
    """

    def __init__(
        self,
        K_main: int,
        K_c: int,
        K_buttons: int,
        K_shoulder: int,
        device: torch.device,
    ):
        """Initialize metrics accumulator.

        Args:
            K_main: Number of main stick quantization bins
            K_c: Number of C-stick quantization bins
            K_buttons: Number of button outputs
            K_shoulder: Number of shoulder quantization bins
            device: Device for tensors
        """
        self.device = device
        self.K_main = K_main
        self.K_c = K_c
        self.K_buttons = K_buttons
        self.K_shoulder = K_shoulder

        # Main stick metrics
        self.main_correct = torch.tensor(0, dtype=torch.long, device=device)
        self.main_total = torch.tensor(0, dtype=torch.long, device=device)
        self.main_label_counts = torch.zeros(K_main, dtype=torch.long, device=device)
        self.main_maj_correct = torch.tensor(0, dtype=torch.long, device=device)
        self.main_rep_correct = torch.tensor(0, dtype=torch.long, device=device)
        self.main_rep_total = torch.tensor(0, dtype=torch.long, device=device)

        # C-stick metrics
        self.c_correct = torch.tensor(0, dtype=torch.long, device=device)
        self.c_total = torch.tensor(0, dtype=torch.long, device=device)
        self.c_label_counts = torch.zeros(K_c, dtype=torch.long, device=device)
        self.c_maj_correct = torch.tensor(0, dtype=torch.long, device=device)
        self.c_rep_correct = torch.tensor(0, dtype=torch.long, device=device)
        self.c_rep_total = torch.tensor(0, dtype=torch.long, device=device)

        # Button metrics
        self.btn_tp = torch.zeros(K_buttons, dtype=torch.float32, device=device)
        self.btn_fp = torch.zeros(K_buttons, dtype=torch.float32, device=device)
        self.btn_fn = torch.zeros(K_buttons, dtype=torch.float32, device=device)
        self.btn_pos_counts = torch.zeros(K_buttons, dtype=torch.float32, device=device)
        self.btn_total = torch.tensor(0, dtype=torch.long, device=device)
        self.btn_em_correct = torch.tensor(0, dtype=torch.long, device=device)
        self.btn_maj_em_correct = torch.tensor(0, dtype=torch.long, device=device)
        self.btn_rep_em_correct = torch.tensor(0, dtype=torch.long, device=device)
        self.btn_rep_total = torch.tensor(0, dtype=torch.long, device=device)

        # Shoulder metrics
        self.shoulder_correct = torch.tensor(0, dtype=torch.long, device=device)
        self.shoulder_total = torch.tensor(0, dtype=torch.long, device=device)
        self.shoulder_label_counts = (
            torch.zeros(K_shoulder, dtype=torch.long, device=device)
            if K_shoulder > 0
            else None
        )
        self.shoulder_maj_correct = torch.tensor(0, dtype=torch.long, device=device)

    def _majority_label(self, label_counts: torch.Tensor) -> int:
        """Get the majority label from counts."""
        if label_counts.numel() == 0:
            return 0
        return int(torch.argmax(label_counts).item())

    def update_stick_metrics(
        self,
        pred_idx: torch.Tensor,
        true_idx: torch.Tensor,
        stick_type: str,
        majority_baseline: Optional[int] = None,
        repeat_baseline: Optional[torch.Tensor] = None,
        repeat_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Update stick (main or C-stick) metrics.

        Args:
            pred_idx: Predicted indices [N]
            true_idx: True indices [N]
            stick_type: "main" or "c"
            majority_baseline: Optional majority label for baseline
            repeat_baseline: Optional repeat prediction [N]
            repeat_mask: Optional mask for repeat frames [N]
        """
        if stick_type == "main":
            correct_attr = "main_correct"
            total_attr = "main_total"
            counts_attr = "main_label_counts"
            maj_attr = "main_maj_correct"
            rep_correct_attr = "main_rep_correct"
            rep_total_attr = "main_rep_total"
        elif stick_type == "c":
            correct_attr = "c_correct"
            total_attr = "c_total"
            counts_attr = "c_label_counts"
            maj_attr = "c_maj_correct"
            rep_correct_attr = "c_rep_correct"
            rep_total_attr = "c_rep_total"
        else:
            raise ValueError(f"Unknown stick type: {stick_type}")

        # Accuracy
        correct = (pred_idx == true_idx).sum()
        setattr(self, correct_attr, getattr(self, correct_attr) + correct)
        setattr(self, total_attr, getattr(self, total_attr) + true_idx.numel())

        # Label counts
        label_counts = getattr(self, counts_attr)
        bincount = torch.bincount(true_idx, minlength=label_counts.shape[0])
        setattr(self, counts_attr, label_counts + bincount[: label_counts.shape[0]])

        # Majority baseline
        if majority_baseline is not None:
            maj_correct = (true_idx == majority_baseline).sum()
            setattr(self, maj_attr, getattr(self, maj_attr) + maj_correct)

        # Repeat baseline
        if repeat_baseline is not None and repeat_mask is not None:
            if repeat_mask.any():
                rep_correct = (
                    repeat_baseline[repeat_mask] == true_idx[repeat_mask]
                ).sum()
                setattr(
                    self,
                    rep_correct_attr,
                    getattr(self, rep_correct_attr) + rep_correct,
                )
                setattr(
                    self,
                    rep_total_attr,
                    getattr(self, rep_total_attr) + repeat_mask.sum(),
                )

    def update_button_metrics(
        self,
        pred_buttons: torch.Tensor,
        true_buttons: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        """Update button metrics.

        Args:
            pred_buttons: Predicted button states [B, L, K]
            true_buttons: True button states [B, L, K]
            logits: Button logits [B, L, K]
        """
        B, L, K = pred_buttons.shape

        # Flatten
        pred_flat = pred_buttons.reshape(-1, K).float()
        true_flat = true_buttons.reshape(-1, K).float()

        # TP, FP, FN
        self.btn_tp += (pred_flat * true_flat).sum(dim=0)
        self.btn_fp += (pred_flat * (1 - true_flat)).sum(dim=0)
        self.btn_fn += ((1 - pred_flat) * true_flat).sum(dim=0)

        # Positive counts
        self.btn_pos_counts += true_flat.sum(dim=0)

        # Total frames
        self.btn_total += B * L

        # Exact match
        em_correct = (pred_buttons == true_buttons).all(dim=-1).sum()
        self.btn_em_correct += em_correct

        # Majority baseline (all zeros or all ones depending on majority)
        pos_rate = true_flat.mean(dim=0)
        maj_pred = (pos_rate >= 0.5).float().unsqueeze(0).expand_as(pred_flat)
        maj_em = (maj_pred.reshape(B, L, K) == true_buttons).all(dim=-1).sum()
        self.btn_maj_em_correct += maj_em

    def update_shoulder_metrics(
        self,
        pred_idx: torch.Tensor,
        true_idx: torch.Tensor,
        majority_baseline: Optional[int] = None,
    ) -> None:
        """Update shoulder metrics.

        Args:
            pred_idx: Predicted indices [B, L]
            true_idx: True indices [B, L]
            majority_baseline: Optional majority label for baseline
        """
        if self.K_shoulder <= 0:
            return

        pred_flat = pred_idx.reshape(-1)
        true_flat = true_idx.reshape(-1)

        # Accuracy
        correct = (pred_flat == true_flat).sum()
        self.shoulder_correct += correct
        self.shoulder_total += true_flat.numel()

        # Label counts
        if self.shoulder_label_counts is not None:
            bincount = torch.bincount(true_flat, minlength=self.K_shoulder)
            self.shoulder_label_counts += bincount[: self.K_shoulder]

        # Majority baseline
        if majority_baseline is not None:
            maj_correct = (true_flat == majority_baseline).sum()
            self.shoulder_maj_correct += maj_correct

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all metrics.

        Returns:
            Dictionary of metric names to values
        """
        summary = {}

        # Main stick
        summary["acc_main"] = float(self.main_correct.item()) / max(
            1, float(self.main_total.item())
        )
        summary["acc_main_maj"] = float(self.main_maj_correct.item()) / max(
            1, float(self.main_total.item())
        )
        if self.main_rep_total.item() > 0:
            summary["acc_main_rep"] = float(self.main_rep_correct.item()) / float(
                self.main_rep_total.item()
            )
        else:
            summary["acc_main_rep"] = 0.0

        # C-stick
        summary["acc_c"] = float(self.c_correct.item()) / max(
            1, float(self.c_total.item())
        )
        summary["acc_c_maj"] = float(self.c_maj_correct.item()) / max(
            1, float(self.c_total.item())
        )
        if self.c_rep_total.item() > 0:
            summary["acc_c_rep"] = float(self.c_rep_correct.item()) / float(
                self.c_rep_total.item()
            )
        else:
            summary["acc_c_rep"] = 0.0

        # Buttons
        tp = self.btn_tp.cpu().numpy()
        fp = self.btn_fp.cpu().numpy()
        fn = self.btn_fn.cpu().numpy()

        # Micro-averaged
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

        # Macro-averaged
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

        summary["btn_em"] = float(self.btn_em_correct.item()) / max(
            1, float(self.btn_total.item())
        )
        summary["btn_prec_micro"] = float(prec_micro)
        summary["btn_rec_micro"] = float(rec_micro)
        summary["btn_f1_micro"] = float(f1_micro)
        summary["btn_f1_macro"] = float(f1_macro)
        summary["btn_em_maj"] = float(self.btn_maj_em_correct.item()) / max(
            1, float(self.btn_total.item())
        )
        if self.btn_rep_total.item() > 0:
            summary["btn_em_rep"] = float(self.btn_rep_em_correct.item()) / float(
                self.btn_rep_total.item()
            )
        else:
            summary["btn_em_rep"] = 0.0

        # Shoulder
        if self.K_shoulder > 0:
            summary["acc_shoulder"] = float(self.shoulder_correct.item()) / max(
                1, float(self.shoulder_total.item())
            )
            summary["acc_shoulder_maj"] = float(self.shoulder_maj_correct.item()) / max(
                1, float(self.shoulder_total.item())
            )

        return summary

    def reset(self) -> None:
        """Reset all metrics to zero."""
        # Main stick
        self.main_correct.zero_()
        self.main_total.zero_()
        self.main_label_counts.zero_()
        self.main_maj_correct.zero_()
        self.main_rep_correct.zero_()
        self.main_rep_total.zero_()

        # C-stick
        self.c_correct.zero_()
        self.c_total.zero_()
        self.c_label_counts.zero_()
        self.c_maj_correct.zero_()
        self.c_rep_correct.zero_()
        self.c_rep_total.zero_()

        # Buttons
        self.btn_tp.zero_()
        self.btn_fp.zero_()
        self.btn_fn.zero_()
        self.btn_pos_counts.zero_()
        self.btn_total.zero_()
        self.btn_em_correct.zero_()
        self.btn_maj_em_correct.zero_()
        self.btn_rep_em_correct.zero_()
        self.btn_rep_total.zero_()

        # Shoulder
        self.shoulder_correct.zero_()
        self.shoulder_total.zero_()
        if self.shoulder_label_counts is not None:
            self.shoulder_label_counts.zero_()
        self.shoulder_maj_correct.zero_()


def compute_confusion_matrix(
    true_flat: torch.Tensor, pred_flat: torch.Tensor, K: int
) -> torch.Tensor:
    """Compute confusion matrix from flattened integer labels.

    Args:
        true_flat: True labels [N]
        pred_flat: Predicted labels [N]
        K: Number of classes

    Returns:
        [K, K] confusion matrix on CPU
    """
    tf = true_flat.to(torch.int64)
    pf = pred_flat.to(torch.int64)
    cm = torch.bincount(tf * K + pf, minlength=K * K).view(K, K)
    return cm.cpu()


# TODO: why is this unused?
def compute_change_hold_accuracy(
    pred: torch.Tensor,
    true: torch.Tensor,
    change_mask: torch.Tensor,
    hold_mask: torch.Tensor,
) -> Tuple[float, float]:
    """Compute accuracy separately for change and hold frames.

    Args:
        pred: Predictions [B, L]
        true: True labels [B, L]
        change_mask: Boolean mask for change frames [B, L]
        hold_mask: Boolean mask for hold frames [B, L]

    Returns:
        (change_accuracy, hold_accuracy)
    """
    correct = pred == true

    change_acc = 0.0
    if change_mask.any():
        change_acc = float(correct[change_mask].float().mean().item())

    hold_acc = 0.0
    if hold_mask.any():
        hold_acc = float(correct[hold_mask].float().mean().item())

    return change_acc, hold_acc


# TODO: Optimize? are there more useful metrics? expose precision and recall here too?
# TODO: What is the difference between micro and macro?
def multilabel_prf(
    true_labels: torch.Tensor, pred_labels: torch.Tensor
) -> Tuple[float, float, float, float, float]:
    """Compute multi-label precision, recall, F1.

    Args:
        true_labels: True labels [B, L, K] or [N, K]
        pred_labels: Predicted labels [B, L, K] or [N, K]

    Returns:
        (exact_match, precision_micro, recall_micro, f1_micro, f1_macro)
    """
    # Ensure 3D
    if true_labels.dim() == 2:
        true_labels = true_labels.unsqueeze(0)
        pred_labels = pred_labels.unsqueeze(0)

    t = true_labels.bool()
    p = pred_labels.bool()

    # Micro metrics
    tp = (t & p).sum().item()
    fp = ((~t) & p).sum().item()
    fn = (t & (~p)).sum().item()

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Macro metrics (per class)
    tp_c = (t & p).sum(dim=(0, 1)).cpu().numpy()
    fp_c = ((~t) & p).sum(dim=(0, 1)).cpu().numpy()
    fn_c = (t & (~p)).sum(dim=(0, 1)).cpu().numpy()
    f1_c = []
    for a, b, c in zip(tp_c, fp_c, fn_c):
        pr = a / (a + b) if (a + b) > 0 else 0.0
        rc = a / (a + c) if (a + c) > 0 else 0.0
        f1_c.append(2 * pr * rc / (pr + rc) if (pr + rc) > 0 else 0.0)
    f1_macro = float(np.mean(f1_c) if len(f1_c) else 0.0)

    # Exact match
    em = float((pred_labels == true_labels).all(dim=-1).float().mean().item())

    return em, float(prec), float(rec), float(f1), f1_macro
