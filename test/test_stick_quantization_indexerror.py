"""Test for controller_quantization_shared.py IndexError in boolean indexing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from controller_quantization_shared import (
    _sticks01_to_unit11,
    quantize_stick_indices,
)
from feature_transforms import MAIN_PALETTE, _MAIN_PALETTE_NORM


def test_sticks01_to_unit11_with_batch_dimension():
    """Test that _sticks01_to_unit11 handles batched inputs with values outside unit circle.

    When converting from [0,1] to [-1,1], some coordinates may end up outside
    the unit circle and need to be clamped. This test ensures the boolean
    indexing works correctly for multidimensional arrays.
    """
    # Create a batch of stick coordinates in [0,1] range
    # These values, when converted to [-1,1], will be (1.0, 1.0) which is outside the unit circle
    # sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.414 > 1.0
    batch_size = 5
    sticks = np.array([[1.0, 1.0], [0.5, 0.5], [1.0, 0.5], [0.75, 0.75], [1.0, 1.0]])
    assert sticks.shape == (batch_size, 2)

    # This should not raise IndexError
    result = _sticks01_to_unit11(sticks)

    # Check that results are within or on the unit circle
    norms = np.linalg.norm(result, axis=-1)
    assert np.all(norms <= 1.0 + 1e-6), "All results should be within unit circle"


def test_sticks01_to_unit11_single_coordinate_outside_circle():
    """Test single coordinate pair that exceeds unit circle after conversion."""
    # (1.0, 1.0) in [0,1] becomes (1.0, 1.0) in [-1,1], which is outside the circle
    stick = np.array([1.0, 1.0])

    result = _sticks01_to_unit11(stick)

    # Should be clamped to unit circle
    norm = np.linalg.norm(result)
    assert norm <= 1.0 + 1e-6, f"Result norm {norm} should be <= 1.0"


def test_quantize_stick_indices_with_batch():
    """Test quantize_stick_indices with batched inputs that trigger the clamping path."""
    # Create stick coordinates that will exceed unit circle after conversion
    sticks = np.array(
        [
            [1.0, 1.0],  # Will be outside circle
            [0.5, 0.5],  # Will be inside circle
            [1.0, 0.0],  # On circle edge
        ]
    )

    # This should not raise IndexError
    indices = quantize_stick_indices(
        sticks, MAIN_PALETTE, _MAIN_PALETTE_NORM, input_domain="unit01"
    )

    # Verify we got valid indices
    assert indices.shape == (3,)
    assert np.all(indices >= 0)
    assert np.all(indices < len(MAIN_PALETTE))


def test_quantize_stick_indices_3d_input():
    """Test with 3D input (e.g., batch of sequences)."""
    # Shape: (batch=2, sequence=3, coords=2)
    sticks = np.array(
        [
            [[1.0, 1.0], [0.5, 0.5], [0.0, 0.0]],
            [[1.0, 0.0], [0.0, 1.0], [0.75, 0.75]],
        ]
    )

    # This should not raise IndexError
    indices = quantize_stick_indices(
        sticks, MAIN_PALETTE, _MAIN_PALETTE_NORM, input_domain="unit01"
    )

    # Verify output shape
    assert indices.shape == (2, 3)
    assert np.all(indices >= 0)
    assert np.all(indices < len(MAIN_PALETTE))


def test_sticks01_to_unit11_torch_consistency():
    """Ensure torch and numpy implementations produce consistent results."""
    sticks_np = np.array([[1.0, 1.0], [0.5, 0.5], [0.0, 1.0]])
    sticks_torch = torch.from_numpy(sticks_np).float()

    result_np = _sticks01_to_unit11(sticks_np)
    result_torch = _sticks01_to_unit11(sticks_torch).numpy()

    # Results should be very close (accounting for floating point differences)
    np.testing.assert_allclose(result_np, result_torch, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "shape",
    [
        (2,),  # Single coordinate
        (5, 2),  # Batch
        (3, 4, 2),  # 3D tensor
        (2, 3, 4, 2),  # 4D tensor
    ],
)
def test_sticks01_to_unit11_various_shapes(shape):
    """Test _sticks01_to_unit11 with various input shapes."""
    # Create array with values that will exceed unit circle
    arr = np.ones(shape, dtype=np.float32)

    # Should not raise IndexError
    result = _sticks01_to_unit11(arr)

    # Check shape is preserved
    assert result.shape == shape

    # Check all norms are within unit circle
    norms = np.linalg.norm(result, axis=-1)
    assert np.all(norms <= 1.0 + 1e-6)


def test_clamp_with_auto_domain():
    """Test the auto domain detection with values that need clamping."""
    # Values in [0,1] range that will exceed unit circle
    sticks = np.array([[1.0, 1.0], [0.9, 0.9]])

    # Auto domain should detect [0,1] range and convert properly
    indices = quantize_stick_indices(
        sticks, MAIN_PALETTE, _MAIN_PALETTE_NORM, input_domain="auto"
    )

    assert indices.shape == (2,)
    assert np.all(indices >= 0)
    assert np.all(indices < len(MAIN_PALETTE))
