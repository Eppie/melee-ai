from __future__ import annotations

import sys
from io import StringIO
import re

import numpy as np
import pytest

# This is a private function, so we have to do this.
from validation import (
    EnhancedMetrics,
    RunStats,
    StickErrorMetrics,
    _print_enhanced_metrics,
)


def run_print_and_capture(enhanced: EnhancedMetrics) -> str:
    """Helper function to run _print_enhanced_metrics and capture stdout."""
    captured_output = StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured_output
    try:
        _print_enhanced_metrics(enhanced)
    finally:
        sys.stdout = original_stdout
    return captured_output.getvalue()


class TestPrintEnhancedMetrics:
    def test_print_empty_metrics(self):
        """Test printing with a default/empty EnhancedMetrics object."""
        enhanced = EnhancedMetrics()
        output = run_print_and_capture(enhanced)

        assert "===== Enhanced Metrics =====" in output
        assert "1. Mean Stick Error" in output
        assert "Main Stick (Euclidean): 0.0000" in output
        assert "C-Stick (Euclidean):    0.0000" in output
        assert "2. Stick Stability (Jitter - Avg Frame-to-Frame Distance):" in output
        assert "3. Prediction Entropy (Uncertainty):" in output
        assert "4. Accuracy by Player State (Main Stick):" in output
        assert "6. 'Stuck' Action Duration (Consecutive Identical States):" in output
        assert "7. L/R Button Press Latency:" in output
        assert "No L/R button press events detected" in output
        assert "10. Controller Input Correlation Matrix:" in output
        assert "Value Head Metrics (RL Critic)" not in output

    def test_print_full_metrics(self):
        """Test printing with a fully populated EnhancedMetrics object."""
        enhanced = EnhancedMetrics(
            main_stick_error=StickErrorMetrics(
                total_error=10.0,
                error_change=4.0,
                error_hold=6.0,
            ),
            c_stick_error=StickErrorMetrics(
                total_error=5.0,
                error_change=2.0,
                error_hold=3.0,
            ),
            total_pred_main_jitter=20.0,
            total_true_main_jitter=18.0,
            total_pred_c_jitter=15.0,
            total_true_c_jitter=12.0,
            jitter_frames=100,
            total_main_entropy=50.0,
            total_c_entropy=40.0,
            total_shoulder_entropy=30.0,
            entropy_frames=100,
            state_correct={"idle": 80, "attack": 70, "other": 0, "hitstun": 0},
            state_total={"idle": 100, "attack": 100, "other": 50, "hitstun": 0},
            pred_run_stats=RunStats(total_length=200, run_count=50, max_length=10),
            true_run_stats=RunStats(total_length=220, run_count=45, max_length=12),
            lr_button_changes_true=[10, 20, 30],
            lr_button_changes_pred=[11, 22, 28],
            all_preds_list=[np.random.rand(10, 9)],
            all_labels_list=[np.random.rand(10, 9)],
            total_value_mse=0.1,
            total_value_mae=0.2,
            total_value_pred=100.0,
            total_value_target=110.0,
            value_pred_list=list(np.random.rand(100)),
            value_target_list=list(np.random.rand(100)),
            value_frames=100,
            total_frames=250,
        )

        output = run_print_and_capture(enhanced)

        assert "Mean Stick Error" in output
        assert "Main Stick (Euclidean): 0.0400" in output
        assert "Stick Stability (Jitter - Avg Frame-to-Frame Distance):" in output
        assert "Predicted:     0.2000" in output
        assert "Prediction Entropy" in output
        assert "Main Stick: 0.5000" in output
        assert "Shoulder:   0.3000" in output
        assert "Accuracy by Player State" in output
        assert re.search(r"Idle\s*:\s*0.800\s*\(100 frames\)", output)
        assert "'Stuck' Action Duration" in output
        assert "Mean: 4.00 frames" in output
        assert "L/R Button Press Latency" in output
        assert "Mean Latency:" in output
        assert "Controller Input Correlation Matrix" in output
        assert "Frobenius norm" in output
        assert "Value Head Metrics" in output
        assert "Mean Squared Error (MSE):     0.001000" in output

    def test_zero_denominator_safe_div(self):
        """Test safe division by zero cases."""
        enhanced = EnhancedMetrics(total_frames=0, jitter_frames=0, entropy_frames=0)
        output = run_print_and_capture(enhanced)
        assert "Main Stick: 0.0000" in output
        assert "Predicted:     0.0000" in output
        assert "Main Stick: 0.0000" in output

    def test_latency_no_matched_events(self):
        """Test L/R button press latency with no matched events."""
        enhanced = EnhancedMetrics(
            lr_button_changes_true=[10, 20, 30],
            lr_button_changes_pred=[100, 200, 300],  # No matches within window
        )
        output = run_print_and_capture(enhanced)
        assert "No matched events" in output

    def test_value_head_metrics_edge_cases(self):
        """Test edge cases in value head metrics printing."""
        # Test case where ss_tot is zero (all target values are the same)
        enhanced = EnhancedMetrics(
            value_frames=10,
            value_pred_list=[0.1, 0.2, 0.3],
            value_target_list=[0.5, 0.5, 0.5],
        )
        output = run_print_and_capture(enhanced)
        assert "R² Score:                     0.0000" in output

        # Test case with single value
        enhanced = EnhancedMetrics(
            value_frames=1,
            value_pred_list=[0.1],
            value_target_list=[0.5],
        )
        output = run_print_and_capture(enhanced)
        assert "Pearson Correlation:" not in output
        assert "R² Score:" not in output

    def test_stuck_action_one_sided(self):
        """Test stuck action duration with only one side having data."""
        enhanced_pred = EnhancedMetrics(
            pred_run_stats=RunStats(total_length=200, run_count=50, max_length=10),
        )
        output_pred = run_print_and_capture(enhanced_pred)
        assert "Predicted:" in output_pred
        # Check that the "Ground Truth" header for stuck action is not there
        assert "Ground Truth:" not in output_pred.split("6. 'Stuck' Action Duration")[1]

        enhanced_true = EnhancedMetrics(
            true_run_stats=RunStats(total_length=220, run_count=45, max_length=12),
        )
        output_true = run_print_and_capture(enhanced_true)
        assert "Predicted:" not in output_true.split("6. 'Stuck' Action Duration")[1]
        assert "Ground Truth:" in output_true

    def test_value_range_error_analysis(self):
        """Test the error analysis by value range section."""
        value_preds = [0.1, 0.2, 0.8, 0.9]
        value_targets = [0.0, 0.25, 0.75, 1.0]
        enhanced = EnhancedMetrics(
            value_frames=4,
            value_pred_list=value_preds,
            value_target_list=value_targets,
        )
        output = run_print_and_capture(enhanced)
        assert "Error Analysis by Target Value Range" in output
        assert re.search(r"Low \(≤\d+\.\d+\)", output)
        assert re.search(r"Mid-Low \(\d+\.\d+-\d+\.\d+\)", output)
        assert re.search(r"Mid-High \(\d+\.\d+-\d+\.\d+\)", output)
        assert re.search(r"High \(>\d+\.\d+\)", output)

    def test_correlation_matrix_empty(self):
        """Test correlation matrix with empty lists."""
        enhanced = EnhancedMetrics(all_preds_list=[], all_labels_list=[])
        output = run_print_and_capture(enhanced)
        assert "Controller Input Correlation Matrix" in output
        assert "Frobenius norm" not in output

        enhanced = EnhancedMetrics(
            all_preds_list=[np.random.rand(10, 9)], all_labels_list=[]
        )
        output = run_print_and_capture(enhanced)
        assert "Controller Input Correlation Matrix" in output
        assert "Frobenius norm" not in output

    def test_value_range_no_data_in_quartile(self):
        """Test value range analysis when a quartile has no data."""
        value_preds = [0.1, 0.9]
        value_targets = [0.0, 1.0]
        enhanced = EnhancedMetrics(
            value_frames=2,
            value_pred_list=value_preds,
            value_target_list=value_targets,
        )
        output = run_print_and_capture(enhanced)
        assert "Error Analysis by Target Value Range" in output
        assert re.search(r"Low \(≤\d+\.\d+\)", output)
        assert not re.search(r"Mid-Low", output)
        assert not re.search(r"Mid-High", output)
        assert re.search(r"High \(>\d+\.\d+\)", output)
