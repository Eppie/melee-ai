"""
Training metrics computation for Melee AI.

This module handles computation and tracking of training metrics
including accuracy, loss, and baseline comparisons.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from melee_ai.config import Settings


class TrainingMetrics:
    """Container for training metrics."""

    def __init__(self):
        self.epoch: int = 0
        self.loss: float = 0.0
        self.accuracy_main: float = 0.0
        self.accuracy_c_stick: float = 0.0
        self.accuracy_buttons: float = 0.0
        self.duration: float = 0.0


class MetricAggregator:
    """Aggregates and computes training metrics."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.main_correct = 0
        self.main_total = 0
        self.c_correct = 0
        self.c_total = 0
        self.buttons_tp = 0
        self.buttons_fp = 0
        self.buttons_fn = 0
        self.buttons_total = 0

    def update_batch_metrics(
        self,
        pred_main: torch.Tensor,
        true_main: torch.Tensor,
        pred_c: torch.Tensor,
        true_c: torch.Tensor,
        pred_buttons: torch.Tensor,
        true_buttons: torch.Tensor
    ):
        """Update metrics with batch predictions."""
        # Main stick accuracy
        main_correct = (pred_main.argmax(dim=-1) == true_main).sum().item()
        self.main_correct += main_correct
        self.main_total += true_main.numel()

        # C-stick accuracy
        c_correct = (pred_c.argmax(dim=-1) == true_c).sum().item()
        self.c_correct += c_correct
        self.c_total += true_c.numel()

        # Button metrics (multi-label)
        pred_buttons_bool = (pred_buttons > 0.5).float()
        true_buttons_bool = true_buttons.float()

        tp = (pred_buttons_bool * true_buttons_bool).sum().item()
        fp = (pred_buttons_bool * (1 - true_buttons_bool)).sum().item()
        fn = ((1 - pred_buttons_bool) * true_buttons_bool).sum().item()

        self.buttons_tp += tp
        self.buttons_fp += fp
        self.buttons_fn += fn
        self.buttons_total += true_buttons.numel()

    def compute_epoch_metrics(self, epoch: int, loss: float, duration: float) -> TrainingMetrics:
        """Compute metrics for completed epoch."""
        metrics = TrainingMetrics()
        metrics.epoch = epoch
        metrics.loss = loss
        metrics.duration = duration

        if self.main_total > 0:
            metrics.accuracy_main = self.main_correct / self.main_total

        if self.c_total > 0:
            metrics.accuracy_c_stick = self.c_correct / self.c_total

        if self.buttons_total > 0:
            precision = self.buttons_tp / (self.buttons_tp + self.buttons_fp) if (self.buttons_tp + self.buttons_fp) > 0 else 0.0
            recall = self.buttons_tp / (self.buttons_tp + self.buttons_fn) if (self.buttons_tp + self.buttons_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            metrics.accuracy_buttons = f1

        return metrics

    def compute_baselines(self) -> Dict[str, float]:
        """Compute baseline metrics for comparison."""
        baselines = {}

        # Random baseline
        if self.main_total > 0:
            main_classes = len(self.settings.preprocessing.fox_stick_palette_size)
            baselines["main_random"] = 1.0 / main_classes

        if self.c_total > 0:
            c_classes = len(self.settings.preprocessing.c_stick_palette_size)
            baselines["c_random"] = 1.0 / c_classes

        if self.buttons_total > 0:
            baselines["buttons_random"] = 0.5  # Random binary classification

        return baselines
