"""Comprehensive unit tests for metrics calculations."""

from __future__ import annotations

import pytest
import torch

from train.metrics import (
    MetricsAccumulator,
    compute_binary_rates,
    compute_confusion_matrix,
    multilabel_prf,
)


class TestMetricsAccumulator:
    """Test suite for MetricsAccumulator class."""

    @pytest.fixture
    def device(self):
        """Return CPU device for testing."""
        return torch.device("cpu")

    @pytest.fixture
    def accumulator(self, device):
        """Create a standard accumulator for testing."""
        return MetricsAccumulator(
            K_main=64,
            K_c=9,
            K_buttons=5,
            K_shoulder=5,
            device=device,
        )

    def test_initialization(self, device):
        """Test that accumulator initializes with correct shapes and zero values."""
        acc = MetricsAccumulator(
            K_main=10,
            K_c=5,
            K_buttons=3,
            K_shoulder=4,
            device=device,
        )

        # Check device and dimensions
        assert acc.device == device
        assert acc.K_main == 10
        assert acc.K_c == 5
        assert acc.K_buttons == 3
        assert acc.K_shoulder == 4

        # Check main stick metrics initialized to zero
        assert acc.main_correct.item() == 0
        assert acc.main_total.item() == 0
        assert acc.main_label_counts.shape == (10,)
        assert acc.main_label_counts.sum().item() == 0
        assert acc.main_maj_correct.item() == 0

        # Check c-stick metrics initialized to zero
        assert acc.c_correct.item() == 0
        assert acc.c_total.item() == 0
        assert acc.c_label_counts.shape == (5,)
        assert acc.c_label_counts.sum().item() == 0
        assert acc.c_maj_correct.item() == 0

        # Check button metrics initialized to zero
        assert acc.btn_true_positives.shape == (3,)
        assert acc.btn_false_positives.shape == (3,)
        assert acc.btn_false_negatives.shape == (3,)
        assert acc.btn_pos_counts.shape == (3,)
        assert acc.btn_true_positives.sum().item() == 0
        assert acc.btn_false_positives.sum().item() == 0
        assert acc.btn_false_negatives.sum().item() == 0
        assert acc.btn_pos_counts.sum().item() == 0
        assert acc.btn_total.item() == 0
        assert acc.btn_em_correct.item() == 0
        assert acc.btn_maj_em_correct.item() == 0

        # Check shoulder metrics initialized to zero
        assert acc.shoulder_correct.item() == 0
        assert acc.shoulder_total.item() == 0
        assert acc.shoulder_label_counts.shape == (4,)
        assert acc.shoulder_label_counts.sum().item() == 0
        assert acc.shoulder_maj_correct.item() == 0

    def test_majority_label_normal(self, accumulator):
        """Test _majority_label returns the argmax of counts."""
        counts = torch.tensor([2, 5, 3, 1])
        assert accumulator._majority_label(counts) == 1

        counts = torch.tensor([10, 3, 7])
        assert accumulator._majority_label(counts) == 0

        counts = torch.tensor([1, 1, 10])
        assert accumulator._majority_label(counts) == 2

    def test_majority_label_empty(self, accumulator):
        """Test _majority_label returns 0 for empty tensor."""
        counts = torch.tensor([])
        assert accumulator._majority_label(counts) == 0

    def test_majority_label_ties(self, accumulator):
        """Test _majority_label behavior on ties (argmax returns first occurrence)."""
        counts = torch.tensor([5, 5, 3])
        assert accumulator._majority_label(counts) == 0

    def test_update_stick_metrics_perfect_prediction(self, accumulator):
        """Test stick metrics with perfect predictions."""
        pred_idx = torch.tensor([0, 1, 2, 3, 4])
        true_idx = torch.tensor([0, 1, 2, 3, 4])
        majority_baseline = 0
        repeat_baseline = torch.tensor([0, 0, 2, 3, 4])
        repeat_mask = torch.tensor([False, False, True, True, True])

        accumulator.update_stick_metrics(
            pred_idx=pred_idx,
            true_idx=true_idx,
            stick_type="main",
            majority_baseline=majority_baseline,
            repeat_baseline=repeat_baseline,
            repeat_mask=repeat_mask,
        )

        # All predictions correct
        assert accumulator.main_correct.item() == 5
        assert accumulator.main_total.item() == 5

        # Label counts should match true_idx
        expected_counts = torch.zeros(64, dtype=torch.long)
        expected_counts[0] = 1
        expected_counts[1] = 1
        expected_counts[2] = 1
        expected_counts[3] = 1
        expected_counts[4] = 1
        assert torch.equal(accumulator.main_label_counts, expected_counts)

        # Majority baseline (only first element matches baseline of 0)
        assert accumulator.main_maj_correct.item() == 1

    def test_update_stick_metrics_partial_prediction(self, accumulator):
        """Test stick metrics with some incorrect predictions."""
        pred_idx = torch.tensor([0, 2, 2, 0, 4])
        true_idx = torch.tensor([0, 1, 2, 3, 4])
        majority_baseline = 0
        repeat_baseline = torch.tensor([0, 0, 2, 3, 4])
        repeat_mask = torch.tensor([False, False, True, True, True])

        accumulator.update_stick_metrics(
            pred_idx=pred_idx,
            true_idx=true_idx,
            stick_type="main",
            majority_baseline=majority_baseline,
            repeat_baseline=repeat_baseline,
            repeat_mask=repeat_mask,
        )

        # Only indices 0, 2, 4 correct (3 out of 5)
        assert accumulator.main_correct.item() == 3
        assert accumulator.main_total.item() == 5

        # Majority baseline
        assert accumulator.main_maj_correct.item() == 1

    def test_update_stick_metrics_multiple_batches(self, accumulator):
        """Test that stick metrics accumulate across multiple updates."""
        # First batch
        pred_idx = torch.tensor([0, 1, 2])
        true_idx = torch.tensor([0, 1, 2])
        majority_baseline = 0
        repeat_baseline = torch.tensor([0, 1, 2])
        repeat_mask = torch.tensor([True, True, True])

        accumulator.update_stick_metrics(
            pred_idx=pred_idx,
            true_idx=true_idx,
            stick_type="main",
            majority_baseline=majority_baseline,
            repeat_baseline=repeat_baseline,
            repeat_mask=repeat_mask,
        )

        # Second batch
        pred_idx = torch.tensor([3, 4])
        true_idx = torch.tensor([3, 3])
        repeat_baseline = torch.tensor([3, 3])
        repeat_mask = torch.tensor([False, True])

        accumulator.update_stick_metrics(
            pred_idx=pred_idx,
            true_idx=true_idx,
            stick_type="main",
            majority_baseline=majority_baseline,
            repeat_baseline=repeat_baseline,
            repeat_mask=repeat_mask,
        )

        # 3 correct from first batch + 1 correct from second = 4 total
        assert accumulator.main_correct.item() == 4
        assert accumulator.main_total.item() == 5

    def test_update_button_metrics_perfect_prediction(self, accumulator):
        """Test button metrics with perfect predictions."""
        # Shape: [batch_size=2, sequence_length=3, num_buttons=5]
        pred_buttons = torch.tensor(
            [
                [[1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 1, 0, 0, 1]],
                [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 1, 1, 0, 0]],
            ]
        )
        true_buttons = pred_buttons.clone()

        accumulator.update_button_metrics(pred_buttons, true_buttons)

        # All predictions perfect
        # Count positive instances per button across all frames
        expected_tp = torch.tensor([3.0, 3.0, 2.0, 1.0, 1.0])  # Sum across all frames
        assert torch.allclose(accumulator.btn_true_positives, expected_tp)
        assert torch.allclose(accumulator.btn_false_positives, torch.zeros(5))
        assert torch.allclose(accumulator.btn_false_negatives, torch.zeros(5))

        # Exact match: all 6 frames match
        assert accumulator.btn_em_correct.item() == 6
        assert accumulator.btn_total.item() == 6

    def test_update_button_metrics_partial_prediction(self, accumulator):
        """Test button metrics with some incorrect predictions."""
        # Shape: [batch_size=1, sequence_length=2, num_buttons=5]
        true_buttons = torch.tensor([[[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]]])
        pred_buttons = torch.tensor(
            [
                [
                    [1, 1, 1, 0, 0],
                    [0, 1, 0, 0, 1],
                ]  # Frame 0: FP on button 1; Frame 1: FN on button 3, FP on button 4
            ]
        )

        accumulator.update_button_metrics(pred_buttons, true_buttons)

        # Button 0: 1 TP, 0 FP, 0 FN
        # Button 1: 1 TP, 1 FP, 0 FN
        # Button 2: 1 TP, 0 FP, 0 FN
        # Button 3: 0 TP, 0 FP, 1 FN
        # Button 4: 0 TP, 1 FP, 0 FN
        expected_tp = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0])
        expected_fp = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0])
        expected_fn = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])

        assert torch.allclose(accumulator.btn_true_positives, expected_tp)
        assert torch.allclose(accumulator.btn_false_positives, expected_fp)
        assert torch.allclose(accumulator.btn_false_negatives, expected_fn)

        # Exact match: 0 frames match (both have errors)
        assert accumulator.btn_em_correct.item() == 0
        assert accumulator.btn_total.item() == 2

    def test_update_button_metrics_all_zeros(self, accumulator):
        """Test button metrics when all buttons are zero."""
        pred_buttons = torch.zeros(1, 3, 5)
        true_buttons = torch.zeros(1, 3, 5)

        accumulator.update_button_metrics(pred_buttons, true_buttons)

        # No positive predictions or true positives
        assert torch.allclose(accumulator.btn_true_positives, torch.zeros(5))
        assert torch.allclose(accumulator.btn_false_positives, torch.zeros(5))
        assert torch.allclose(accumulator.btn_false_negatives, torch.zeros(5))

        # All frames match (exact match)
        assert accumulator.btn_em_correct.item() == 3
        assert accumulator.btn_total.item() == 3

    def test_update_button_metrics_accumulation(self, accumulator):
        """Test that button metrics accumulate across batches."""
        # First batch
        pred_buttons = torch.tensor([[[1, 0, 0, 0, 0]]])
        true_buttons = torch.tensor([[[1, 0, 0, 0, 0]]])
        accumulator.update_button_metrics(pred_buttons, true_buttons)

        assert accumulator.btn_true_positives[0].item() == 1
        assert accumulator.btn_total.item() == 1
        assert accumulator.btn_em_correct.item() == 1

        # Second batch
        pred_buttons = torch.tensor([[[1, 1, 0, 0, 0]]])
        true_buttons = torch.tensor([[[1, 0, 0, 0, 0]]])
        accumulator.update_button_metrics(pred_buttons, true_buttons)

        # Button 0: 2 TP total
        # Button 1: 1 FP total
        assert accumulator.btn_true_positives[0].item() == 2
        assert accumulator.btn_false_positives[1].item() == 1
        assert accumulator.btn_total.item() == 2
        assert (
            accumulator.btn_em_correct.item() == 1
        )  # Only first frame was exact match

    def test_update_shoulder_metrics_perfect(self, accumulator):
        """Test shoulder metrics with perfect predictions."""
        pred_idx = torch.tensor([[0, 1, 2, 3, 4]])
        true_idx = torch.tensor([[0, 1, 2, 3, 4]])
        majority_baseline = 0

        accumulator.update_shoulder_metrics(pred_idx, true_idx, majority_baseline)

        assert accumulator.shoulder_correct.item() == 5
        assert accumulator.shoulder_total.item() == 5
        assert (
            accumulator.shoulder_maj_correct.item() == 1
        )  # Only first element equals 0

        # Label counts
        expected_counts = torch.tensor([1, 1, 1, 1, 1], dtype=torch.long)
        assert torch.equal(accumulator.shoulder_label_counts, expected_counts)

    def test_update_shoulder_metrics_partial(self, accumulator):
        """Test shoulder metrics with some errors."""
        pred_idx = torch.tensor([[0, 2, 2, 0, 4]])
        true_idx = torch.tensor([[0, 1, 2, 3, 4]])
        majority_baseline = 0

        accumulator.update_shoulder_metrics(pred_idx, true_idx, majority_baseline)

        # Correct: indices 0, 2, 4 (3 out of 5)
        assert accumulator.shoulder_correct.item() == 3
        assert accumulator.shoulder_total.item() == 5
        assert accumulator.shoulder_maj_correct.item() == 1

    def test_update_shoulder_metrics_accumulation(self, accumulator):
        """Test shoulder metrics accumulate across batches."""
        # First batch
        pred_idx = torch.tensor([[0, 1, 2]])
        true_idx = torch.tensor([[0, 1, 2]])
        majority_baseline = 0

        accumulator.update_shoulder_metrics(pred_idx, true_idx, majority_baseline)

        assert accumulator.shoulder_correct.item() == 3
        assert accumulator.shoulder_total.item() == 3

        # Second batch
        pred_idx = torch.tensor([[3, 3]])
        true_idx = torch.tensor([[3, 4]])

        accumulator.update_shoulder_metrics(pred_idx, true_idx, majority_baseline)

        # 3 from first + 1 from second = 4 correct
        assert accumulator.shoulder_correct.item() == 4
        assert accumulator.shoulder_total.item() == 5

    def test_get_summary_perfect_predictions(self, accumulator):
        """Test get_summary with perfect predictions on all components."""
        # Main stick
        pred_main = torch.tensor([0, 1, 2, 3, 4])
        true_main = torch.tensor([0, 1, 2, 3, 4])
        accumulator.update_stick_metrics(
            pred_main,
            true_main,
            "main",
            0,
            torch.zeros_like(pred_main),
            torch.zeros_like(pred_main, dtype=torch.bool),
        )

        # Buttons
        pred_buttons = torch.ones(1, 5, 5)
        true_buttons = torch.ones(1, 5, 5)
        accumulator.update_button_metrics(pred_buttons, true_buttons)

        # Shoulder
        pred_shoulder = torch.tensor([[0, 1, 2, 3, 4]])
        true_shoulder = torch.tensor([[0, 1, 2, 3, 4]])
        accumulator.update_shoulder_metrics(pred_shoulder, true_shoulder, 0)

        summary = accumulator.get_summary()

        # All accuracies should be 1.0
        assert summary["acc_main"] == 1.0
        assert summary["btn_em"] == 1.0
        assert summary["btn_prec_micro"] == 1.0
        assert summary["btn_rec_micro"] == 1.0
        assert summary["btn_f1_micro"] == 1.0
        assert summary["btn_f1_macro"] == 1.0
        assert summary["acc_shoulder"] == 1.0

    def test_get_summary_zero_division_protection(self, accumulator):
        """Test that get_summary handles empty accumulators without division by zero."""
        summary = accumulator.get_summary()

        # Should return 0.0 for all metrics when no data
        assert summary["acc_main"] == 0.0
        assert summary["acc_main_maj"] == 0.0
        assert summary["acc_c"] == 0.0
        assert summary["acc_c_maj"] == 0.0
        assert summary["btn_em"] == 0.0
        assert summary["btn_prec_micro"] == 0.0
        assert summary["btn_rec_micro"] == 0.0
        assert summary["btn_f1_micro"] == 0.0
        assert summary["btn_f1_macro"] == 0.0
        assert summary["btn_em_maj"] == 0.0
        assert summary["acc_shoulder"] == 0.0
        assert summary["acc_shoulder_maj"] == 0.0

    def test_get_summary_realistic_values(self, accumulator):
        """Test get_summary with realistic imperfect predictions."""
        # Main stick: 80% accuracy (8 correct out of 10)
        # Correct indices: 0, 1, 2, 5, 6, 7, 8, 9
        pred_main = torch.tensor([0, 1, 2, 0, 0, 5, 6, 7, 8, 9])
        true_main = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        accumulator.update_stick_metrics(
            pred_main,
            true_main,
            "main",
            0,
            torch.zeros_like(pred_main),
            torch.zeros_like(pred_main, dtype=torch.bool),
        )

        # Buttons: some errors
        true_buttons = torch.tensor([[[1, 0, 1, 0, 0], [0, 1, 0, 1, 0]]])
        pred_buttons = torch.tensor([[[1, 1, 1, 0, 0], [0, 1, 0, 0, 1]]])
        accumulator.update_button_metrics(pred_buttons, true_buttons)

        # Shoulder: 80% accuracy (4 correct out of 5)
        # Correct indices: 0, 1, 2, 4
        pred_shoulder = torch.tensor([[0, 1, 2, 0, 4]])
        true_shoulder = torch.tensor([[0, 1, 2, 3, 4]])
        accumulator.update_shoulder_metrics(pred_shoulder, true_shoulder, 0)

        summary = accumulator.get_summary()

        # Main stick: 8/10 = 0.8
        assert summary["acc_main"] == 0.8

        # Shoulder: 4/5 = 0.8
        assert summary["acc_shoulder"] == 0.8

        # Button exact match: 0/2 = 0.0
        assert summary["btn_em"] == 0.0

        # Verify metrics are in valid ranges
        assert 0.0 <= summary["btn_prec_micro"] <= 1.0
        assert 0.0 <= summary["btn_rec_micro"] <= 1.0
        assert 0.0 <= summary["btn_f1_micro"] <= 1.0
        assert 0.0 <= summary["btn_f1_macro"] <= 1.0

    def test_reset(self, accumulator):
        """Test that reset zeros out all metrics."""
        # Add some data
        pred_main = torch.tensor([0, 1, 2])
        true_main = torch.tensor([0, 1, 2])
        accumulator.update_stick_metrics(
            pred_main,
            true_main,
            "main",
            0,
            torch.zeros_like(pred_main),
            torch.zeros_like(pred_main, dtype=torch.bool),
        )

        pred_buttons = torch.ones(1, 3, 5)
        true_buttons = torch.ones(1, 3, 5)
        accumulator.update_button_metrics(pred_buttons, true_buttons)

        pred_shoulder = torch.tensor([[0, 1, 2]])
        true_shoulder = torch.tensor([[0, 1, 2]])
        accumulator.update_shoulder_metrics(pred_shoulder, true_shoulder, 0)

        # Verify data was added
        assert accumulator.main_total.item() > 0
        assert accumulator.btn_total.item() > 0
        assert accumulator.shoulder_total.item() > 0

        # Reset
        accumulator.reset()

        # Verify everything is zero
        assert accumulator.main_correct.item() == 0
        assert accumulator.main_total.item() == 0
        assert accumulator.main_label_counts.sum().item() == 0
        assert accumulator.main_maj_correct.item() == 0

        assert accumulator.c_correct.item() == 0
        assert accumulator.c_total.item() == 0
        assert accumulator.c_label_counts.sum().item() == 0
        assert accumulator.c_maj_correct.item() == 0

        assert accumulator.btn_true_positives.sum().item() == 0
        assert accumulator.btn_false_positives.sum().item() == 0
        assert accumulator.btn_false_negatives.sum().item() == 0
        assert accumulator.btn_pos_counts.sum().item() == 0
        assert accumulator.btn_total.item() == 0
        assert accumulator.btn_em_correct.item() == 0
        assert accumulator.btn_maj_em_correct.item() == 0

        assert accumulator.shoulder_correct.item() == 0
        assert accumulator.shoulder_total.item() == 0
        assert accumulator.shoulder_label_counts.sum().item() == 0
        assert accumulator.shoulder_maj_correct.item() == 0

    def test_button_majority_baseline(self, device):
        """Test that button majority baseline is computed correctly."""
        # Create accumulator with 2 buttons for this test
        acc = MetricsAccumulator(
            K_main=64,
            K_c=9,
            K_buttons=2,
            K_shoulder=5,
            device=device,
        )

        # Create a dataset where button 0 is mostly 1, button 1 is mostly 0
        # Button 0: [1, 1, 1, 1, 0] -> 4 ones, 1 zero -> majority is 1 (pos_rate = 0.8 >= 0.5)
        # Button 1: [0, 0, 1, 0, 0] -> 1 one, 4 zeros -> majority is 0 (pos_rate = 0.2 < 0.5)
        true_buttons = torch.tensor([[[1, 0], [1, 0], [1, 1], [1, 0], [0, 0]]])
        pred_buttons = torch.zeros_like(true_buttons)  # Predict all zeros

        acc.update_button_metrics(pred_buttons, true_buttons)

        # Majority baseline should predict [1, 0] for all frames
        # Frame 0: [1, 0] == [1, 0] ✓
        # Frame 1: [1, 0] == [1, 0] ✓
        # Frame 2: [1, 1] != [1, 0] ✗
        # Frame 3: [1, 0] == [1, 0] ✓
        # Frame 4: [0, 0] != [1, 0] ✗
        # Total matches: 3 out of 5
        assert acc.btn_maj_em_correct.item() == 3


class TestBinaryRates:
    """Test suite for compute_binary_rates helper."""

    def test_binary_rate_computation(self):
        """Check TPR/TNR/FPR/FNR calculations for mixed outcomes."""
        tp = torch.tensor([5.0, 0.0])
        fp = torch.tensor([1.0, 2.0])
        fn = torch.tensor([1.0, 0.0])

        tpr, tnr, fpr, fnr, tn = compute_binary_rates(tp, fp, fn, total_count=10.0)

        # Class 0: pos=6, neg=4 → tn=3
        assert torch.allclose(tn, torch.tensor([3.0, 8.0]))
        assert tpr.tolist()[0] == pytest.approx(5 / 6, rel=1e-5)
        assert tnr.tolist()[0] == pytest.approx(3 / 4, rel=1e-5)
        assert fpr.tolist()[0] == pytest.approx(1 / 4, rel=1e-5)
        assert fnr.tolist()[0] == pytest.approx(1 / 6, rel=1e-5)

        # Class 1: no positives, 2 false positives against 10 total
        assert tpr.tolist()[1] == 0.0
        assert fnr.tolist()[1] == 0.0
        assert tnr.tolist()[1] == pytest.approx(0.8, rel=1e-5)
        assert fpr.tolist()[1] == pytest.approx(0.2, rel=1e-5)


class TestConfusionMatrix:
    """Test suite for compute_confusion_matrix function."""

    def test_perfect_predictions(self):
        """Test confusion matrix with perfect predictions."""
        true_flat = torch.tensor([0, 1, 2, 0, 1, 2])
        pred_flat = torch.tensor([0, 1, 2, 0, 1, 2])
        K = 3

        cm = compute_confusion_matrix(true_flat, pred_flat, K)

        # Should have diagonal matrix
        expected = torch.tensor([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
        assert torch.equal(cm, expected)

    def test_all_wrong_predictions(self):
        """Test confusion matrix when all predictions are wrong."""
        true_flat = torch.tensor([0, 0, 0])
        pred_flat = torch.tensor([1, 1, 1])
        K = 3

        cm = compute_confusion_matrix(true_flat, pred_flat, K)

        # All predictions in [0, 1] cell
        expected = torch.tensor([[0, 3, 0], [0, 0, 0], [0, 0, 0]])
        assert torch.equal(cm, expected)

    def test_mixed_predictions(self):
        """Test confusion matrix with realistic mixed predictions."""
        # True: [0, 1, 1, 2]
        # Pred: [0, 2, 1, 2]
        true_flat = torch.tensor([0, 1, 1, 2])
        pred_flat = torch.tensor([0, 2, 1, 2])
        K = 3

        cm = compute_confusion_matrix(true_flat, pred_flat, K)

        # Class 0: predicted as 0 (1 time)
        # Class 1: predicted as 2 (1 time), predicted as 1 (1 time)
        # Class 2: predicted as 2 (1 time)
        expected = torch.tensor(
            [
                [1, 0, 0],  # True class 0
                [0, 1, 1],  # True class 1
                [0, 0, 1],  # True class 2
            ]
        )
        assert torch.equal(cm, expected)

    def test_single_class(self):
        """Test confusion matrix with only one class."""
        true_flat = torch.tensor([2, 2, 2, 2])
        pred_flat = torch.tensor([2, 2, 2, 2])
        K = 5

        cm = compute_confusion_matrix(true_flat, pred_flat, K)

        # Should be all zeros except cm[2, 2] = 4
        assert cm.sum().item() == 4
        assert cm[2, 2].item() == 4

    def test_returns_cpu_tensor(self):
        """Test that confusion matrix is returned on CPU."""
        if torch.cuda.is_available():
            true_flat = torch.tensor([0, 1, 2], device="cuda")
            pred_flat = torch.tensor([0, 1, 2], device="cuda")
            K = 3

            cm = compute_confusion_matrix(true_flat, pred_flat, K)

            assert cm.device.type == "cpu"

    def test_dtype_conversion(self):
        """Test that confusion matrix handles different dtypes."""
        true_flat = torch.tensor([0, 1, 2], dtype=torch.float32)
        pred_flat = torch.tensor([0, 1, 2], dtype=torch.float32)
        K = 3

        cm = compute_confusion_matrix(true_flat, pred_flat, K)

        # Should still work and produce correct result
        expected = torch.eye(3, dtype=torch.int64)
        assert torch.equal(cm, expected)


class TestMultilabelPRF:
    """Test suite for multilabel_prf function."""

    def test_perfect_predictions(self):
        """Test with perfect multi-label predictions."""
        true_labels = torch.tensor([[[1, 0, 1], [0, 1, 0]]])
        pred_labels = torch.tensor([[[1, 0, 1], [0, 1, 0]]])

        em, prec, rec, f1_micro, f1_macro = multilabel_prf(true_labels, pred_labels)

        assert em == 1.0
        assert prec == 1.0
        assert rec == 1.0
        assert f1_micro == 1.0
        assert f1_macro == 1.0

    def test_all_wrong_predictions(self):
        """Test when all predictions are wrong."""
        true_labels = torch.tensor([[[1, 0, 1], [0, 1, 0]]])
        pred_labels = torch.tensor([[[0, 1, 0], [1, 0, 1]]])

        em, prec, rec, f1_micro, f1_macro = multilabel_prf(true_labels, pred_labels)

        assert em == 0.0
        # All predictions are wrong: TP=0, so precision=0, recall=0, f1=0
        assert prec == 0.0
        assert rec == 0.0
        assert f1_micro == 0.0
        assert f1_macro == 0.0

    def test_partial_predictions(self):
        """Test with realistic partial predictions."""
        # Frame 0: true=[1,0], pred=[1,1] -> 1 TP, 1 FP, 0 FN (not exact match)
        # Frame 1: true=[0,1], pred=[0,1] -> 1 TP, 0 FP, 0 FN (exact match)
        true_labels = torch.tensor([[[1, 0], [0, 1]]])
        pred_labels = torch.tensor([[[1, 1], [0, 1]]])

        em, prec, rec, f1_micro, f1_macro = multilabel_prf(true_labels, pred_labels)

        # Exact match: 1/2 = 0.5
        assert em == 0.5

        # Micro metrics:
        # TP = 2 (button 0 frame 0, button 1 frame 1)
        # FP = 1 (button 1 frame 0)
        # FN = 0
        # Precision = 2/(2+1) = 2/3
        # Recall = 2/(2+0) = 1.0
        # F1 = 2 * (2/3) * 1 / (2/3 + 1) = 2 * 2/3 / 5/3 = 4/5 = 0.8
        assert prec == pytest.approx(2 / 3, abs=1e-6)
        assert rec == 1.0
        assert f1_micro == pytest.approx(0.8, abs=1e-6)

        # Macro metrics:
        # Button 0: TP=1, FP=0, FN=0 -> P=1, R=1, F1=1
        # Button 1: TP=1, FP=1, FN=0 -> P=0.5, R=1, F1=2/3
        # Macro F1 = (1 + 2/3) / 2 = 5/6
        assert f1_macro == pytest.approx(5 / 6, abs=1e-6)

    def test_all_zeros(self):
        """Test with all zero predictions and labels."""
        true_labels = torch.zeros(1, 3, 4)
        pred_labels = torch.zeros(1, 3, 4)

        em, prec, rec, f1_micro, f1_macro = multilabel_prf(true_labels, pred_labels)

        # All frames match (all zeros)
        assert em == 1.0

        # No positive instances, so TP=FP=FN=0
        # This should handle division by zero gracefully
        assert prec == 0.0
        assert rec == 0.0
        assert f1_micro == 0.0
        assert f1_macro == 0.0

    def test_all_ones(self):
        """Test with all one predictions and labels."""
        true_labels = torch.ones(1, 3, 4)
        pred_labels = torch.ones(1, 3, 4)

        em, prec, rec, f1_micro, f1_macro = multilabel_prf(true_labels, pred_labels)

        assert em == 1.0
        assert prec == 1.0
        assert rec == 1.0
        assert f1_micro == 1.0
        assert f1_macro == 1.0

    def test_2d_input(self):
        """Test that function handles 2D input by expanding to 3D."""
        true_labels = torch.tensor([[1, 0, 1], [0, 1, 0]])
        pred_labels = torch.tensor([[1, 0, 1], [0, 1, 0]])

        em, prec, rec, f1_micro, f1_macro = multilabel_prf(true_labels, pred_labels)

        assert em == 1.0
        assert prec == 1.0
        assert rec == 1.0
        assert f1_micro == 1.0
        assert f1_macro == 1.0

    def test_single_class_all_positive(self):
        """Test with a single class that's always positive."""
        true_labels = torch.tensor([[[1], [1], [1]]])
        pred_labels = torch.tensor([[[1], [1], [1]]])

        em, prec, rec, f1_micro, f1_macro = multilabel_prf(true_labels, pred_labels)

        assert em == 1.0
        assert prec == 1.0
        assert rec == 1.0
        assert f1_micro == 1.0
        assert f1_macro == 1.0

    def test_false_negatives_only(self):
        """Test case with only false negatives (predictions all zero, truth has ones)."""
        true_labels = torch.tensor([[[1, 1, 1]]])
        pred_labels = torch.tensor([[[0, 0, 0]]])

        em, prec, rec, f1_micro, f1_macro = multilabel_prf(true_labels, pred_labels)

        # No exact match
        assert em == 0.0

        # TP=0, FP=0, FN=3
        # Precision = 0/0 -> 0 (by convention)
        # Recall = 0/3 = 0
        assert prec == 0.0
        assert rec == 0.0
        assert f1_micro == 0.0
        assert f1_macro == 0.0

    def test_false_positives_only(self):
        """Test case with only false positives (predictions all one, truth has zeros)."""
        true_labels = torch.tensor([[[0, 0, 0]]])
        pred_labels = torch.tensor([[[1, 1, 1]]])

        em, prec, rec, f1_micro, f1_macro = multilabel_prf(true_labels, pred_labels)

        # No exact match
        assert em == 0.0

        # TP=0, FP=3, FN=0
        # Precision = 0/3 = 0
        # Recall = 0/0 -> 0 (by convention)
        assert prec == 0.0
        assert rec == 0.0
        assert f1_micro == 0.0
        assert f1_macro == 0.0

    def test_batch_dimension(self):
        """Test with multiple batch elements."""
        true_labels = torch.tensor([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])
        pred_labels = torch.tensor([[[1, 0], [0, 1]], [[1, 1], [0, 0]]])

        em, prec, rec, f1_micro, f1_macro = multilabel_prf(true_labels, pred_labels)

        # All frames match
        assert em == 1.0
        assert prec == 1.0
        assert rec == 1.0
        assert f1_micro == 1.0
        assert f1_macro == 1.0

    def test_imbalanced_classes(self):
        """Test with highly imbalanced classes."""
        # Class 0: appears 10 times, Class 1: appears 1 time
        true_labels = torch.tensor([[[1, 0]] * 9 + [[1, 1]]])
        pred_labels = torch.tensor([[[1, 0]] * 10])

        em, prec, rec, f1_micro, f1_macro = multilabel_prf(true_labels, pred_labels)

        # 9/10 frames match exactly
        assert em == pytest.approx(0.9, abs=1e-6)

        # Micro: TP=10 (class 0) + 0 (class 1) = 10, FP=0, FN=1
        # Precision = 10/10 = 1.0
        # Recall = 10/11
        assert prec == 1.0
        assert rec == pytest.approx(10 / 11, abs=1e-6)
        assert f1_micro == pytest.approx(
            2 * 1.0 * (10 / 11) / (1.0 + 10 / 11), abs=1e-6
        )

        # Macro: average F1 across both classes
        # Class 0: P=1, R=1, F1=1
        # Class 1: P=0, R=0, F1=0
        # Macro F1 = 0.5
        assert f1_macro == pytest.approx(0.5, abs=1e-6)
