from bisect import bisect_right

import torch
import numpy as np
import pytest

from controller_quantization import _quantize_stick
from controller_utils import CONTROL_STICK_QUANTIZED, SHOULDER_QUANTIZED


@pytest.fixture
def palette_data():
    """Prepares the palette and its squared norm for tests."""
    palette = torch.tensor(np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32))
    palette_norm_sq = (palette * palette).sum(dim=1)
    return palette, palette_norm_sq


def test_quantize_stick_unit01(palette_data):
    """Tests the 'unit01' input domain for stick quantization."""
    palette, palette_norm_sq = palette_data
    B, L = 1, 2
    # Input in [0, 1] range
    xy = torch.tensor([[[0.5, 0.5], [1.0, 0.5]]], dtype=torch.float32)

    # Expected indices
    # For [0.5, 0.5], corresponding to [0, 0] in unit11, which is neutral
    neutral_idx = np.where(
        (np.array(CONTROL_STICK_QUANTIZED) == [0.0, 0.0]).all(axis=1)
    )[0][0]
    # For [1.0, 0.5], corresponding to [1, 0] in unit11
    right_idx = np.where((np.array(CONTROL_STICK_QUANTIZED) == [1.0, 0.0]).all(axis=1))[
        0
    ][0]

    expected = torch.tensor([[neutral_idx, right_idx]], dtype=torch.long)
    result = _quantize_stick(xy, palette, palette_norm_sq, "unit01", B, L)
    assert torch.equal(result, expected)


def test_quantize_stick_unit11(palette_data):
    """Tests the 'unit11' input domain for stick quantization."""
    palette, palette_norm_sq = palette_data
    B, L = 1, 2
    # Input in [-1, 1] range
    xy = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=torch.float32)

    # Expected indices
    neutral_idx = np.where(
        (np.array(CONTROL_STICK_QUANTIZED) == [0.0, 0.0]).all(axis=1)
    )[0][0]
    right_idx = np.where((np.array(CONTROL_STICK_QUANTIZED) == [1.0, 0.0]).all(axis=1))[
        0
    ][0]

    expected = torch.tensor([[neutral_idx, right_idx]], dtype=torch.long)
    result = _quantize_stick(xy, palette, palette_norm_sq, "unit11", B, L)
    assert torch.equal(result, expected)


def test_quantize_stick_unit01_rejects_out_of_range(palette_data):
    """unit01 domain should assert if inputs leave [0, 1]."""
    palette, palette_norm_sq = palette_data
    B, L = 1, 1
    xy = torch.tensor([[[1.1, 0.5]]], dtype=torch.float32)

    with pytest.raises(ValueError):
        _quantize_stick(xy, palette, palette_norm_sq, "unit01", B, L)


def test_quantize_stick_unit11_rejects_out_of_range(palette_data):
    """unit11 domain should assert if inputs leave [-1, 1]."""
    palette, palette_norm_sq = palette_data
    B, L = 1, 1
    xy = torch.tensor([[[1.1, 0.5]]], dtype=torch.float32)

    with pytest.raises(ValueError):
        _quantize_stick(xy, palette, palette_norm_sq, "unit11", B, L)


from controller_quantization import (
    sticks01_to_unit11,
    _clamp_unit_circle,
    _device_cache_key,
    _palette_for_device,
    quantize_targets,
)


def test_sticks01_to_unit11():
    """Tests the sticks01_to_unit11 function."""

    # Test case 1: values within [0, 1]

    xy01 = torch.tensor([[[0.5, 0.5], [0.0, 1.0]]])

    expected = torch.tensor([[[0.0, 0.0], [-1.0, 1.0]]])

    # Corrected expected value after _clamp_unit_circle

    expected = torch.tensor([[[0.0, 0.0], [-0.7071, 0.7071]]])

    result = sticks01_to_unit11(xy01)

    assert torch.allclose(result, expected, atol=1e-4)

    # Test case 2: values outside [0, 1]

    xy01 = torch.tensor([[[1.1, -0.1]]])

    with pytest.raises(ValueError):
        sticks01_to_unit11(xy01)


def test_clamp_unit_circle():
    """Tests the _clamp_unit_circle function."""

    # Test case 1: values inside the unit circle

    xy11 = torch.tensor([[[0.5, 0.5]]])

    expected = torch.tensor([[[0.5, 0.5]]])

    result = _clamp_unit_circle(xy11)

    assert torch.allclose(result, expected)

    # Test case 2: values outside the unit circle

    xy11 = torch.tensor([[[1.0, 1.0]]])

    expected = torch.tensor([[[1 / np.sqrt(2), 1 / np.sqrt(2)]]], dtype=torch.float32)

    result = _clamp_unit_circle(xy11)

    assert torch.allclose(result, expected, atol=1e-4)


def test_device_cache_key_cpu():
    """Tests the _device_cache_key function for CPU."""
    device = torch.device("cpu")
    result = _device_cache_key(device)
    assert result == ("cpu", None)


def test_palette_for_device_cpu():
    """Tests the _palette_for_device function for CPU."""
    cpu_palette = torch.tensor([[0.0, 0.0]])
    cache = {}
    device = torch.device("cpu")
    result = _palette_for_device(cpu_palette, cache, device)
    assert torch.equal(result, cpu_palette)
    assert not cache  # cache should not be populated for cpu


class MockColumnMap:
    def __init__(self):
        self.y_main = (0, 1)
        self.y_c = (2, 3)
        self.y_buttons = (4, 5)
        self.y_shoulder = 6


def test_quantize_targets():
    """Tests the quantize_targets function."""
    colmap = MockColumnMap()
    B, L = 1, 1
    # main_x, main_y, c_x, c_y, button_a, button_b, shoulder
    batch_Y = torch.tensor([[[0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.5]]], dtype=torch.float32)

    result = quantize_targets(batch_Y, colmap, input_domain="unit01")

    assert "main_idx" in result
    assert "c_idx" in result
    assert "buttons" in result
    assert "shoulder_idx" in result
    assert "main_K" in result
    assert "c_K" in result
    assert "buttons_K" in result
    assert "shoulder_K" in result

    assert result["main_idx"].shape == (B, L)
    assert result["c_idx"].shape == (B, L)
    assert result["buttons"].shape == (B, L, 2)
    assert result["shoulder_idx"].shape == (B, L)


def test_quantize_targets_shoulders_floor_palette():
    """Shoulder quantization should choose the largest palette value <= raw input."""
    colmap = MockColumnMap()
    B, L = 1, 3
    shoulder_vals = [0.15, 0.31, 0.65]
    batch_Y = torch.tensor(
        [
            [
                [0.5, 0.5, 0.5, 0.5, 1.0, 0.0, shoulder_vals[0]],
                [0.5, 0.5, 0.5, 0.5, 1.0, 0.0, shoulder_vals[1]],
                [0.5, 0.5, 0.5, 0.5, 1.0, 0.0, shoulder_vals[2]],
            ]
        ],
        dtype=torch.float32,
    )

    result = quantize_targets(batch_Y, colmap, input_domain="unit01")
    palette = SHOULDER_QUANTIZED
    expected_indices = torch.tensor(
        [
            [
                max(
                    0,
                    min(len(palette) - 1, bisect_right(palette, shoulder_vals[0]) - 1),
                ),
                max(
                    0,
                    min(len(palette) - 1, bisect_right(palette, shoulder_vals[1]) - 1),
                ),
                max(
                    0,
                    min(len(palette) - 1, bisect_right(palette, shoulder_vals[2]) - 1),
                ),
            ]
        ],
        dtype=torch.long,
    )

    assert torch.equal(result["shoulder_idx"], expected_indices)


def test_quantize_targets_no_shoulder(monkeypatch):
    """Tests quantize_targets when SHOULDER_QUANTIZED is empty."""
    colmap = MockColumnMap()
    batch_Y = torch.tensor([[[0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 0.5]]], dtype=torch.float32)

    monkeypatch.setattr("controller_quantization._SHOULDER_PALETTE_CPU", None)

    with pytest.raises(AttributeError):
        quantize_targets(batch_Y, colmap, input_domain="unit01")


def test_quantize_stick_unit01_out_of_range(palette_data):
    """Tests the 'unit01' input domain with out-of-range values."""
    palette, palette_norm_sq = palette_data
    B, L = 1, 1
    xy = torch.tensor([[[1.1, 0.5]]], dtype=torch.float32)

    with pytest.raises(ValueError):
        _quantize_stick(xy, palette, palette_norm_sq, "unit01", B, L)
