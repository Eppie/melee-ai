from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from pytest import MonkeyPatch
import torch

from train.components import AMPContext, TrainingComponents
from train.step import perform_backward_pass


def build_training_components() -> TrainingComponents:
    optimizer = mock.MagicMock()
    scaler = mock.MagicMock()
    scaler.is_enabled.return_value = False

    model = mock.MagicMock()
    model.parameters.return_value = [torch.ones(1, requires_grad=True)]

    return TrainingComponents(
        config=SimpleNamespace(train=SimpleNamespace(grad_clip=1.0)),
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        logger=mock.MagicMock(),
        local_logger=mock.MagicMock(),
        device=torch.device("cpu"),
        amp=AMPContext(enabled=False, device_type="cpu", dtype=torch.float16),
        ratios=None,
        column_map=None,
        value_idx=0,
        loader=None,
        sampler=None,
        total_steps=0,
        out_dir=Path("."),
        last_step_file=Path("last-step.txt"),
        debug=False,
    )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_perform_backward_pass_skips_non_finite_loss(bad_value: float) -> None:
    components = build_training_components()
    loss = torch.tensor(bad_value, requires_grad=True)

    grad_stats = perform_backward_pass(
        components,
        loss,
        collect_grad_stats=False,
    )

    assert grad_stats == {}
    components.optimizer.zero_grad.assert_not_called()
    components.scaler.scale.assert_not_called()
    components.scaler.step.assert_not_called()


def test_perform_backward_pass_clips_gradients_without_stats(
    monkeypatch: MonkeyPatch,
) -> None:
    components = build_training_components()
    loss = torch.tensor(1.0, requires_grad=True)
    clip_mock = mock.MagicMock(return_value=torch.tensor(0.0))
    monkeypatch.setattr("train.step.clip_grad_norm_", clip_mock)
    grad_diag_mock = mock.MagicMock(return_value={})
    monkeypatch.setattr("train.step.collect_gradient_diagnostics", grad_diag_mock)

    perform_backward_pass(
        components,
        loss,
        collect_grad_stats=False,
    )

    clip_mock.assert_called_once_with(
        components.model.parameters(),
        components.config.train.grad_clip,
    )
    grad_diag_mock.assert_not_called()
