import numpy as np
import pytest

from controller_utils import CONTROL_STICK_QUANTIZED, C_STICK_QUANTIZED
from feature_transforms import (
    MAIN_PALETTE,
    C_PALETTE,
    apply_feature_transforms,
    apply_feature_transforms_dict,
    _quantize_stick,
    _sticks01_to_unit11,
)


def test_sticks01_to_unit11():
    # Test case 1: values within [0, 1]
    xy01 = np.array([[0.5, 0.5], [0.0, 1.0]])
    expected = np.array([[0.0, 0.0], [-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)]])
    result = _sticks01_to_unit11(xy01)
    assert np.allclose(result, expected, atol=1e-4)

    # Test case 2: values outside [0, 1]
    xy01 = np.array([[1.1, -0.1]])
    # clipped to [[1.0, 0.0]]
    # scaled to [[1.0, -1.0]]
    # clamped to [[1/sqrt(2), -1/sqrt(2)]]
    expected = np.array([[1.0 / np.sqrt(2), -1.0 / np.sqrt(2)]])
    result = _sticks01_to_unit11(xy01)
    assert np.allclose(result, expected, atol=1e-4)


def test_quantize_stick_unit01():
    """Test stick quantization with [0,1] input domain."""
    palette_norm = np.sum(MAIN_PALETTE**2, axis=1, keepdims=True)
    block = np.array([[0.5, 0.5], [1.0, 0.5]], dtype=np.float32)

    result = _quantize_stick(block.copy(), MAIN_PALETTE, palette_norm)

    expected_0 = np.array([0.0, 0.0])  # neutral
    expected_1 = np.array([1.0, 0.0])  # right

    assert np.allclose(result[0], expected_0)
    assert np.allclose(result[1], expected_1)


def test_quantize_stick_out_of_range():
    """Test stick quantization with out-of-range values."""
    palette_norm = np.sum(MAIN_PALETTE**2, axis=1, keepdims=True)
    block = np.array([[-0.5, 0.5], [1.1, 0.0]], dtype=np.float32)

    result = _quantize_stick(block.copy(), MAIN_PALETTE, palette_norm)

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


def test_apply_feature_transforms_basic():
    """Test that apply_feature_transforms works on basic input."""
    # Create a minimal feature array with known columns
    feature_names = [
        "p1_main_stick_x",
        "p1_main_stick_y",
        "p1_c_stick_x",
        "p1_c_stick_y",
        "p1_facing",
        "p1_percent",
        "p1_position_x",
        "p1_position_y",
        "p1_jumps_left",
    ]

    # Create test data: 2 frames, 9 features
    features = np.array(
        [
            [0.5, 0.5, 0.5, 0.5, 1.0, 50.0, 20.0, 10.0, 6.0],
            [1.0, 0.5, 0.5, 0.5, 0.0, 100.0, 0.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )

    result = apply_feature_transforms(features.copy(), feature_names)

    # Check main stick quantization (neutral stick should be [0, 0])
    assert np.allclose(result[0, 0:2], [0.0, 0.0], atol=1e-4)

    # Check c-stick quantization (neutral stick should be [0, 0])
    assert np.allclose(result[0, 2:4], [0.0, 0.0], atol=1e-4)

    # Check facing: 1.0 * 2.0 - 1.0 = 1.0
    assert np.allclose(result[0, 4], 1.0)
    # facing: 0.0 * 2.0 - 1.0 = -1.0
    assert np.allclose(result[1, 4], -1.0)

    # Check percent: 50.0 / 100 = 0.5
    assert np.allclose(result[0, 5], 0.5)

    # Check position_x: 20.0 / 20 = 1.0
    assert np.allclose(result[0, 6], 1.0)

    # Check jumps_left: 6.0 / 6 = 1.0
    assert np.allclose(result[0, 8], 1.0)


def test_apply_feature_transforms_dict_basic():
    """Test that apply_feature_transforms_dict works on basic input."""
    features = {
        "p1_main_stick_x": 0.5,
        "p1_main_stick_y": 0.5,
        "p1_c_stick_x": 0.5,
        "p1_c_stick_y": 0.5,
        "p1_facing": 1.0,
        "p1_percent": 50.0,
        "p1_position_x": 20.0,
        "p1_position_y": 10.0,
        "p1_jumps_left": 6.0,
    }

    result = apply_feature_transforms_dict(features)

    # Check main stick quantization (neutral stick should be [0, 0])
    assert np.allclose(result["p1_main_stick_x"], 0.0, atol=1e-4)
    assert np.allclose(result["p1_main_stick_y"], 0.0, atol=1e-4)

    # Check facing: 1.0 * 2.0 - 1.0 = 1.0
    assert np.allclose(result["p1_facing"], 1.0)

    # Check percent: 50.0 / 100 = 0.5
    assert np.allclose(result["p1_percent"], 0.5)


def test_train_inference_parity():
    """Verify that array and dict transforms produce identical results."""
    feature_names = [
        "p1_main_stick_x",
        "p1_main_stick_y",
        "p1_c_stick_x",
        "p1_c_stick_y",
        "p1_facing",
        "p1_percent",
        "p1_position_x",
        "p1_position_y",
        "p1_jumps_left",
    ]

    # Test with random values
    np.random.seed(42)
    features_array = np.random.rand(1, 9).astype(np.float32)
    features_array[0, 4] = 1.0  # facing in [0, 1]
    features_array[0, 5] = 75.0  # percent
    features_array[0, 6] = 15.0  # position_x
    features_array[0, 7] = -5.0  # position_y
    features_array[0, 8] = 2.0  # jumps_left

    # Create dict version
    features_dict = {
        name: float(features_array[0, i]) for i, name in enumerate(feature_names)
    }

    # Apply both transforms
    result_array = apply_feature_transforms(features_array.copy(), feature_names)
    result_dict = apply_feature_transforms_dict(features_dict)

    # Compare results
    for i, name in enumerate(feature_names):
        assert np.allclose(
            result_array[0, i], result_dict[name], atol=1e-5
        ), f"Mismatch for {name}: array={result_array[0, i]}, dict={result_dict[name]}"
