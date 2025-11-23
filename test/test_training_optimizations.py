"""Tests for training loop optimizations (GPU/CPU transfer minimization)."""

from __future__ import annotations

import torch
import torch.nn as nn
from pytest import MonkeyPatch

from train.components import EpochContext
from train.gradients import collect_gradient_diagnostics


class TestEpochContext:
    """Tests for EpochContext loss accumulation optimization."""

    def test_add_loss_accumulates_without_item(self, monkeypatch: MonkeyPatch) -> None:
        """Test that add_loss doesn't call .item() during accumulation."""
        call_count = {"value": 0}
        original_item = torch.Tensor.item

        def spy(self):
            call_count["value"] += 1
            return original_item(self)

        monkeypatch.setattr(torch.Tensor, "item", spy, raising=False)

        ctx = EpochContext()
        losses = [torch.tensor(0.5), torch.tensor(0.3), torch.tensor(0.2)]

        for loss in losses:
            ctx.add_loss(loss)
            ctx.iters_processed += 1

        # No .item() calls during accumulation
        assert call_count["value"] == 0

        # Only call .item() when getting the average
        avg = ctx.get_avg_loss()
        assert call_count["value"] == 1
        assert abs(avg - (0.5 + 0.3 + 0.2) / 3) < 1e-6

    def test_add_loss_correct_accumulation(self) -> None:
        """Test that losses are accumulated correctly."""
        ctx = EpochContext()

        ctx.add_loss(torch.tensor(1.0))
        ctx.iters_processed += 1
        assert abs(ctx.get_avg_loss() - 1.0) < 1e-6

        ctx.add_loss(torch.tensor(3.0))
        ctx.iters_processed += 1
        assert abs(ctx.get_avg_loss() - 2.0) < 1e-6  # (1 + 3) / 2

        ctx.add_loss(torch.tensor(2.0))
        ctx.iters_processed += 1
        assert abs(ctx.get_avg_loss() - 2.0) < 1e-6  # (1 + 3 + 2) / 3

    def test_get_avg_loss_empty(self) -> None:
        """Test that get_avg_loss returns 0 when no losses added."""
        ctx = EpochContext()
        assert ctx.get_avg_loss() == 0.0

    def test_add_loss_preserves_device(self) -> None:
        """Test that accumulated loss stays on the same device as input."""
        ctx = EpochContext()
        loss = torch.tensor(1.0)
        ctx.add_loss(loss)

        assert ctx.epoch_loss_sum is not None
        assert ctx.epoch_loss_sum.device == loss.device


class TestGradientDiagnostics:
    """Tests for gradient diagnostics optimization."""

    def test_collect_gradient_diagnostics_basic(self) -> None:
        """Test basic gradient statistics computation."""
        model = nn.Linear(10, 5)
        # Create some gradients
        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()

        stats = collect_gradient_diagnostics(model)

        # Check all expected keys are present
        expected_keys = [
            "total_norm", "mean_abs", "mean", "std", "max_abs",
            "zero_fraction", "zero_count", "nan_count", "inf_count",
            "param_total_norm", "param_max_abs", "grad_param_ratio_mean",
            "grad_param_ratio_max", "grad_param_ratio_min", "grad_to_param_norm_ratio",
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"
            assert isinstance(stats[key], float), f"Key {key} should be float"

        # Basic sanity checks
        assert stats["total_norm"] > 0
        assert stats["mean_abs"] >= 0
        assert stats["max_abs"] >= stats["mean_abs"]
        assert 0 <= stats["zero_fraction"] <= 1
        assert stats["nan_count"] == 0
        assert stats["inf_count"] == 0

    def test_collect_gradient_diagnostics_no_gradients(self) -> None:
        """Test with model that has no gradients."""
        model = nn.Linear(10, 5)
        # Don't compute gradients

        stats = collect_gradient_diagnostics(model)

        # Should return zeros for all stats
        assert stats["total_norm"] == 0.0
        assert stats["mean_abs"] == 0.0
        assert stats["nan_count"] == 0.0

    def test_collect_gradient_diagnostics_minimizes_item_calls(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Test that gradient diagnostics minimizes .item() calls."""
        model = nn.Linear(10, 5)
        x = torch.randn(2, 10)
        y = model(x).sum()
        y.backward()

        call_count = {"value": 0}
        original_item = torch.Tensor.item

        def spy(self):
            call_count["value"] += 1
            return original_item(self)

        monkeypatch.setattr(torch.Tensor, "item", spy, raising=False)

        stats = collect_gradient_diagnostics(model)

        # Should have minimal .item() calls:
        # - 0 for main stats (uses .cpu().tolist())
        # - At most a few for per-parameter ratio computation
        # With 2 parameters (weight + bias), expect at most 2 transfers for ratios
        # (each does .cpu().tolist() not .item())
        assert call_count["value"] == 0, (
            f"Expected 0 .item() calls, got {call_count['value']}"
        )

    def test_collect_gradient_diagnostics_accuracy(self) -> None:
        """Test that computed statistics are mathematically correct."""
        # Create a simple model with known gradients
        model = nn.Linear(4, 2, bias=False)

        # Set specific gradient values
        model.weight.grad = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ])

        stats = collect_gradient_diagnostics(model)

        # Manual calculations
        grads = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
        expected_sq_sum = (grads ** 2).sum().item()
        expected_norm = (expected_sq_sum) ** 0.5
        expected_mean_abs = grads.abs().mean().item()
        expected_mean = grads.mean().item()
        expected_max_abs = grads.abs().max().item()

        assert abs(stats["total_norm"] - expected_norm) < 1e-5
        assert abs(stats["mean_abs"] - expected_mean_abs) < 1e-5
        assert abs(stats["mean"] - expected_mean) < 1e-5
        assert abs(stats["max_abs"] - expected_max_abs) < 1e-5
        assert stats["zero_count"] == 0
        assert stats["nan_count"] == 0
        assert stats["inf_count"] == 0


class TestIntegration:
    """Integration tests for the training optimizations."""

    def test_epoch_context_with_gpu_tensors(self) -> None:
        """Test EpochContext works correctly (simulating GPU behavior on CPU)."""
        ctx = EpochContext()

        # Simulate a mini training loop
        for i in range(10):
            loss = torch.tensor(float(i) * 0.1)
            ctx.add_loss(loss)
            ctx.iters_processed += 1

        # Expected: sum(0.0, 0.1, 0.2, ..., 0.9) / 10 = 4.5 / 10 = 0.45
        expected_avg = sum(i * 0.1 for i in range(10)) / 10
        assert abs(ctx.get_avg_loss() - expected_avg) < 1e-6
