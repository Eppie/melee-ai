import numpy as np
import pytest

from controller_utils import CONTROL_STICK_QUANTIZED
from feature_transforms import (
    _transform_scale,
    _transform_offset,
    _sticks01_to_unit11_np,
    _stick_palette_apply,
)


def test_transform_scale():
    column = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    factor = 0.1
    result = _transform_scale(column, factor=factor)
    expected = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    assert np.allclose(result, expected)


def test_transform_offset():
    column = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    delta = 5.0
    result = _transform_offset(column, delta=delta)
    expected = np.array([4.0, 5.0, 7.0], dtype=np.float32)
    assert np.allclose(result, expected)


def test_sticks01_to_unit11_np():
    # Test case 1: values within [0, 1]
    xy01 = np.array([[0.5, 0.5], [0.0, 1.0]])
    expected = np.array([[0.0, 0.0], [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]])
    result = _sticks01_to_unit11_np(xy01)
    assert np.allclose(result, expected, atol=1e-4)

    # Test case 2: values outside [0, 1]
    xy01 = np.array([[1.1, -0.1]])
    # clipped to [[1.0, 0.0]]
    # scaled to [[1.0, -1.0]]
    # clamped to [[1/sqrt(2), -1/sqrt(2)]]
    expected = np.array([[1.0 / np.sqrt(2), -1.0 / np.sqrt(2)]])
    result = _sticks01_to_unit11_np(xy01)
    assert np.allclose(result, expected, atol=1e-4)


@pytest.fixture
def stick_palette_data():
    palette = np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32)
    palette_norm = np.sum(palette**2, axis=1, keepdims=True)
    return palette, palette_norm


def test_stick_palette_apply_unit01(stick_palette_data):
    palette, palette_norm = stick_palette_data
    block = np.array([[0.5, 0.5], [1.0, 0.5]], dtype=np.float32)

    result = _stick_palette_apply(
        block.copy(), palette=palette, palette_norm=palette_norm
    )

    expected_0 = np.array([0.0, 0.0])
    expected_1 = np.array([1.0, 0.0])

    assert np.allclose(result[0], expected_0)
    assert np.allclose(result[1], expected_1)


def test_stick_palette_apply_out_of_range(stick_palette_data):
    palette, palette_norm = stick_palette_data
    block = np.array([[-0.5, 0.5], [1.1, 0.0]], dtype=np.float32)

    result = _stick_palette_apply(
        block.copy(), palette=palette, palette_norm=palette_norm
    )

    # Find closest for [-0.5, 0.5]
    points = np.asarray(CONTROL_STICK_QUANTIZED)
    target0 = np.array([-0.5, 0.5])
    distances0 = np.sum((points - target0) ** 2, axis=1)
    closest_idx0 = np.argmin(distances0)
    expected_0 = points[closest_idx0]

    # Find closest for [1.0, 0.0]
    target1 = np.array([1.0, 0.0])
    distances1 = np.sum((points - target1) ** 2, axis=1)
    closest_idx1 = np.argmin(distances1)
    expected_1 = points[closest_idx1]

    assert np.allclose(result[0], expected_0)
    assert np.allclose(result[1], expected_1)


def test_stick_palette_apply_invalid_shape(stick_palette_data):
    palette, palette_norm = stick_palette_data
    block = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        _stick_palette_apply(block, palette=palette, palette_norm=palette_norm)
