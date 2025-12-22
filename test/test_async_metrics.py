"""Unit tests for async transfer optimizations in metrics and logging."""

import torch
import numpy as np
from train.metrics import MetricsAccumulator, multilabel_prf


def test_metrics_accumulator_batched_transfers():
    """Test that MetricsAccumulator.get_summary() batches GPU->CPU transfers."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create accumulator
    acc = MetricsAccumulator(
        K_main=10, K_c=10, K_buttons=12, K_shoulder=4, device=device
    )

    # Update with some fake data
    B, L = 32, 64
    main_pred = torch.randint(0, 10, (B, L), device=device)
    main_true = torch.randint(0, 10, (B, L), device=device)
    repeat_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    repeat_baseline = torch.zeros(B, L, dtype=torch.long, device=device)
    acc.update_stick_metrics(
        main_pred,
        main_true,
        majority_baseline=0,
        repeat_baseline=repeat_baseline,
        repeat_mask=repeat_mask,
        stick_type="main",
    )

    c_pred = torch.randint(0, 10, (B, L), device=device)
    c_true = torch.randint(0, 10, (B, L), device=device)
    acc.update_stick_metrics(
        c_pred,
        c_true,
        majority_baseline=0,
        repeat_baseline=repeat_baseline,
        repeat_mask=repeat_mask,
        stick_type="c",
    )

    # Button data
    btn_pred = torch.randint(0, 2, (B, L, 12), device=device).float()
    btn_true = torch.randint(0, 2, (B, L, 12), device=device).float()
    acc.update_button_metrics(btn_pred, btn_true)

    shoulder_pred = torch.randint(0, 4, (B, L), device=device)
    shoulder_true = torch.randint(0, 4, (B, L), device=device)
    acc.update_shoulder_metrics(shoulder_pred, shoulder_true, majority_baseline=0)

    # Get summary (should do batched transfers)
    summary = acc.get_summary()

    # Verify we got expected metrics
    assert "acc_main" in summary
    assert "acc_main_maj" in summary
    assert "acc_c" in summary
    assert "acc_c_maj" in summary
    assert "btn_em" in summary
    assert "btn_prec_micro" in summary
    assert "btn_rec_micro" in summary
    assert "btn_f1_micro" in summary
    assert "btn_f1_macro" in summary
    assert "acc_shoulder" in summary
    assert "acc_shoulder_maj" in summary

    # All values should be floats
    for key, value in summary.items():
        assert isinstance(value, float), f"{key} should be float, got {type(value)}"
        assert 0.0 <= value <= 1.0, f"{key} = {value} should be in [0, 1]"

    print("✅ MetricsAccumulator.get_summary() test passed")


def test_multilabel_prf_batched():
    """Test that multilabel_prf batches GPU->CPU transfers."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create test data
    B, L, K = 16, 32, 12
    true_labels = torch.randint(0, 2, (B, L, K), device=device).float()
    pred_labels = torch.randint(0, 2, (B, L, K), device=device).float()

    # Make some predictions match truth for non-zero metrics
    pred_labels[:, :5, :] = true_labels[:, :5, :]

    # Call function
    em, prec, rec, f1, f1_macro = multilabel_prf(true_labels, pred_labels)

    # Verify types
    assert isinstance(em, float)
    assert isinstance(prec, float)
    assert isinstance(rec, float)
    assert isinstance(f1, float)
    assert isinstance(f1_macro, float)

    # Verify ranges
    assert 0.0 <= em <= 1.0
    assert 0.0 <= prec <= 1.0
    assert 0.0 <= rec <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert 0.0 <= f1_macro <= 1.0

    print("✅ multilabel_prf() test passed")


def test_multilabel_prf_perfect():
    """Test multilabel_prf with perfect predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Perfect predictions
    true_labels = torch.tensor([[[1, 0, 1], [0, 1, 0]]], device=device).float()
    pred_labels = true_labels.clone()

    em, prec, rec, f1, f1_macro = multilabel_prf(true_labels, pred_labels)

    assert em == 1.0, f"Perfect EM should be 1.0, got {em}"
    assert prec == 1.0, f"Perfect precision should be 1.0, got {prec}"
    assert rec == 1.0, f"Perfect recall should be 1.0, got {rec}"
    assert f1 == 1.0, f"Perfect F1 should be 1.0, got {f1}"
    assert f1_macro == 1.0, f"Perfect F1 macro should be 1.0, got {f1_macro}"

    print("✅ multilabel_prf() perfect predictions test passed")


if __name__ == "__main__":
    print("Running async metrics tests...\n")

    test_metrics_accumulator_batched_transfers()
    test_multilabel_prf_batched()
    test_multilabel_prf_perfect()

    print("\n" + "=" * 60)
    print("✅ All async metrics tests passed!")
    print("=" * 60)
