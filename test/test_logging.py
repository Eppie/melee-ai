from __future__ import annotations

from typing import Dict

import torch
from pytest import MonkeyPatch

from train.logging import append_tensor_stats


def test_append_tensor_stats_avoids_item(monkeypatch: MonkeyPatch) -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    expected: Dict[str, float] = {
        "foo_min": float(torch.amin(tensor).item()),
        "foo_max": float(torch.amax(tensor).item()),
        "foo_mean": float(tensor.mean().item()),
        "foo_std": float(tensor.std(unbiased=False).item()),
    }

    call_count = {"value": 0}
    original_item = torch.Tensor.item

    def spy(self):
        call_count["value"] += 1
        return original_item(self)

    monkeypatch.setattr(torch.Tensor, "item", spy, raising=False)

    out: Dict[str, float] = {}
    append_tensor_stats("foo", tensor, out)

    assert out == expected
    assert call_count["value"] == 0
