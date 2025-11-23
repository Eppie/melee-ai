from __future__ import annotations

from typing import Dict, List

import torch
from pytest import MonkeyPatch

from train.logging import _compute_tensor_stats_batch


def test_compute_tensor_stats_batch_avoids_item(monkeypatch: MonkeyPatch) -> None:
    """Test that _compute_tensor_stats_batch computes correct stats without using .item()."""
    tensor1 = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    tensor2 = torch.ones(5, dtype=torch.float32) * 3.0

    expected = [
        [float(torch.amin(tensor1)), float(torch.amax(tensor1)),
         float(tensor1.mean()), float(tensor1.std(unbiased=False))],
        [3.0, 3.0, 3.0, 0.0],  # tensor2: all same value, so std=0
    ]

    call_count = {"value": 0}
    original_item = torch.Tensor.item

    def spy(self):
        call_count["value"] += 1
        return original_item(self)

    monkeypatch.setattr(torch.Tensor, "item", spy, raising=False)

    result = _compute_tensor_stats_batch([tensor1, tensor2])
    result_list = result.tolist()

    # Check values are correct
    for i in range(2):
        for j in range(4):
            assert abs(result_list[i][j] - expected[i][j]) < 1e-5, \
                f"Mismatch at [{i}][{j}]: {result_list[i][j]} vs {expected[i][j]}"

    # Check no .item() calls during computation
    assert call_count["value"] == 0


def test_compute_tensor_stats_batch_single_element() -> None:
    """Test that single-element tensors get std=0."""
    tensor = torch.tensor([5.0])
    result = _compute_tensor_stats_batch([tensor])
    result_list = result.tolist()[0]

    assert result_list[0] == 5.0  # min
    assert result_list[1] == 5.0  # max
    assert result_list[2] == 5.0  # mean
    assert result_list[3] == 0.0  # std (single element)


def test_compute_tensor_stats_batch_integer_tensor() -> None:
    """Test that integer tensors are handled correctly (converted to float)."""
    tensor = torch.arange(10, dtype=torch.int64)
    result = _compute_tensor_stats_batch([tensor])
    result_list = result.tolist()[0]

    assert result_list[0] == 0.0  # min
    assert result_list[1] == 9.0  # max
    assert abs(result_list[2] - 4.5) < 1e-5  # mean
    assert result_list[3] > 0  # std should be non-zero
