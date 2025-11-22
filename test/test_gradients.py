from __future__ import annotations

import math

import torch

from train.gradients import collect_gradient_diagnostics


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        self.w2 = torch.nn.Parameter(torch.tensor([0.0, 4.0, -4.0]))


def test_collect_gradient_diagnostics_basic_metrics() -> None:
    model = _TinyModel()
    model.w1.grad = torch.tensor([1.0, -2.0])
    model.w2.grad = torch.tensor([0.0, 2.0, -2.0])

    stats = collect_gradient_diagnostics(model, eps=1e-12)

    assert math.isclose(stats["total_norm"], math.sqrt(13.0), rel_tol=1e-6)
    assert stats["zero_count"] == 1.0
    assert stats["nan_count"] == 0.0
    assert stats["inf_count"] == 0.0
    assert math.isclose(stats["zero_fraction"], 1 / 5)
    assert math.isclose(stats["param_max_abs"], 4.0)
    assert math.isclose(stats["grad_param_ratio_mean"], 0.75, rel_tol=1e-6)
    assert math.isclose(stats["grad_param_ratio_max"], 1.0, rel_tol=1e-6)
    assert math.isclose(stats["grad_param_ratio_min"], 0.5, rel_tol=1e-6)
    assert math.isclose(
        stats["grad_to_param_norm_ratio"], math.sqrt(13.0 / 37.0), rel_tol=1e-6
    )


def test_collect_gradient_diagnostics_handles_nonfinite_values() -> None:
    model = _TinyModel()
    model.w1.grad = torch.tensor([float("nan"), float("inf")])
    model.w2.grad = torch.tensor([1.0, 0.0, -1.0])

    stats = collect_gradient_diagnostics(model, eps=1e-12)

    assert stats["nan_count"] == 1.0
    assert stats["inf_count"] == 1.0
    assert stats["zero_count"] == 1.0
    assert math.isfinite(stats["total_norm"])
    assert math.isfinite(stats["mean_abs"])
