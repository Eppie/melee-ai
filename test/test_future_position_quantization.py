"""Test future position quantization and dequantization logic.

This test verifies that position values are correctly bucketed and reconstructed,
with special attention to negative Y coordinates where bias has been observed.
"""

import torch
import numpy as np
from controller_quantization import (
    quantize_future_position,
    dequantize_future_position,
    FUTURE_X_BUCKETS,
    FUTURE_Y_BUCKETS,
)


def test_quantize_dequantize_roundtrip():
    """Test that quantize -> dequantize gives reasonable reconstruction."""
    print("=" * 80)
    print("TEST: Quantize-Dequantize Roundtrip")
    print("=" * 80)

    # Test key Y positions (especially negative values)
    test_y_values = torch.tensor([
        -150.0,  # Blastzone bottom
        -100.0,  # Below stage
        -50.0,   # Below stage
        -20.0,   # Boundary value
        -10.0,   # Negative near stage
        -5.0,    # Small negative
        0.0,     # Stage level
        5.0,     # Small positive
        10.0,    # Low platform
        30.0,    # Stage level
        60.0,    # Platform height
        100.0,   # Air
        260.0,   # Blastzone top
    ])

    print(f"\nTesting {len(test_y_values)} Y positions:")
    print("-" * 80)

    errors = []
    max_error = 0.0
    worst_value = None

    for i, y_val in enumerate(test_y_values):
        y_tensor = torch.tensor([y_val])

        # Quantize
        bucket_idx = quantize_future_position(y_tensor, 'y')

        # Dequantize
        reconstructed = dequantize_future_position(bucket_idx, 'y')

        error = (reconstructed - y_val).abs().item()
        errors.append(error)

        if error > max_error:
            max_error = error
            worst_value = y_val.item()

        # Find bucket boundaries
        bucket = bucket_idx.item()
        if bucket == 0:
            lower_bound = "< " + str(FUTURE_Y_BUCKETS[0].item())
            upper_bound = str(FUTURE_Y_BUCKETS[0].item())
        elif bucket >= len(FUTURE_Y_BUCKETS):
            lower_bound = str(FUTURE_Y_BUCKETS[-1].item())
            upper_bound = "> " + str(FUTURE_Y_BUCKETS[-1].item())
        else:
            lower_bound = str(FUTURE_Y_BUCKETS[bucket - 1].item()) if bucket > 0 else "< " + str(FUTURE_Y_BUCKETS[0].item())
            upper_bound = str(FUTURE_Y_BUCKETS[bucket].item()) if bucket < len(FUTURE_Y_BUCKETS) else "> " + str(FUTURE_Y_BUCKETS[-1].item())

        status = "❌" if error > 10.0 else "⚠️" if error > 5.0 else "✓"
        print(f"{status} y={y_val.item():7.1f} → bucket[{bucket:2d}] → {reconstructed.item():7.1f} "
              f"(error: {error:5.1f}, bounds: [{lower_bound}, {upper_bound}])")

    print("-" * 80)
    print(f"Mean error: {np.mean(errors):.2f}")
    print(f"Max error:  {max_error:.2f} (at y={worst_value})")
    print()

    return errors, max_error


def test_boundary_values():
    """Test positions exactly on bucket boundaries."""
    print("=" * 80)
    print("TEST: Boundary Values")
    print("=" * 80)

    print("\nY-axis boundaries:")
    print("-" * 80)

    for i, boundary in enumerate(FUTURE_Y_BUCKETS):
        # Test value exactly on boundary
        y_tensor = torch.tensor([boundary.item()])
        bucket_idx = quantize_future_position(y_tensor, 'y')
        reconstructed = dequantize_future_position(bucket_idx, 'y')
        error = (reconstructed - boundary).abs().item()

        print(f"Boundary {i:2d}: y={boundary.item():7.1f} → bucket[{bucket_idx.item():2d}] "
              f"→ {reconstructed.item():7.1f} (error: {error:5.1f})")

    print()


def test_negative_y_bias():
    """Specifically test for bias in negative Y coordinate predictions."""
    print("=" * 80)
    print("TEST: Negative Y Coordinate Bias")
    print("=" * 80)

    # Focus on the range where user reported issues: negative Y near stage level
    test_range = torch.linspace(-30.0, 10.0, 41)  # -30 to 10 in steps of 1

    print(f"\nTesting {len(test_range)} values from -30.0 to 10.0:")
    print("-" * 80)

    negative_errors = []
    positive_errors = []

    for y_val in test_range:
        y_tensor = torch.tensor([y_val])
        bucket_idx = quantize_future_position(y_tensor, 'y')
        reconstructed = dequantize_future_position(bucket_idx, 'y')
        error = reconstructed.item() - y_val.item()  # Signed error

        if y_val < 0:
            negative_errors.append(error)
        else:
            positive_errors.append(error)

        if abs(error) > 3.0:  # Only print large errors
            sign = "↓" if error < 0 else "↑"
            print(f"  y={y_val.item():6.1f} → {reconstructed.item():6.1f} "
                  f"({sign} error: {error:+6.2f})")

    print("-" * 80)
    print(f"Negative Y values (y < 0):")
    print(f"  Mean error:   {np.mean(negative_errors):+.2f} (negative = underestimate)")
    print(f"  Median error: {np.median(negative_errors):+.2f}")
    print(f"  Std error:    {np.std(negative_errors):.2f}")
    print()
    print(f"Positive Y values (y >= 0):")
    print(f"  Mean error:   {np.mean(positive_errors):+.2f}")
    print(f"  Median error: {np.median(positive_errors):+.2f}")
    print(f"  Std error:    {np.std(positive_errors):.2f}")
    print()

    # Check for systematic bias
    if np.mean(negative_errors) < -1.0:
        print("⚠️  WARNING: Systematic underestimation bias detected for negative Y!")
        print(f"   Models predicting negative Y will consistently output values that are")
        print(f"   {-np.mean(negative_errors):.1f} units too low on average.")

    print()


def test_bucket_midpoint_calculation():
    """Verify the midpoint calculation logic is correct."""
    print("=" * 80)
    print("TEST: Bucket Midpoint Calculation")
    print("=" * 80)

    from controller_quantization import _compute_midpoints

    midpoints_y = _compute_midpoints(FUTURE_Y_BUCKETS, torch.device('cpu'))

    print("\nY-axis bucket midpoints:")
    print("-" * 80)
    print(f"{'Bucket':<8} {'Lower':<10} {'Upper':<10} {'Midpoint':<12} {'Width':<8}")
    print("-" * 80)

    for i in range(32):
        if i == 0:
            lower = f"<{FUTURE_Y_BUCKETS[0].item()}"
            upper = f"{FUTURE_Y_BUCKETS[0].item()}"
            width = "extrapolated"
        elif i < len(FUTURE_Y_BUCKETS):
            lower = f"{FUTURE_Y_BUCKETS[i-1].item():.1f}"
            upper = f"{FUTURE_Y_BUCKETS[i].item():.1f}"
            width = f"{FUTURE_Y_BUCKETS[i].item() - FUTURE_Y_BUCKETS[i-1].item():.1f}"
        else:
            lower = f"{FUTURE_Y_BUCKETS[-1].item():.1f}"
            upper = f">{FUTURE_Y_BUCKETS[-1].item()}"
            width = "extrapolated"

        midpoint = f"{midpoints_y[i].item():.1f}"
        print(f"{i:<8} {lower:<10} {upper:<10} {midpoint:<12} {width:<8}")

    print()


def test_x_axis_quantization():
    """Test X-axis quantization for completeness."""
    print("=" * 80)
    print("TEST: X-axis Quantization")
    print("=" * 80)

    test_x_values = torch.tensor([
        -240.0,  # Blastzone left
        -95.0,   # Stage left edge
        -50.0,   # Left stage
        0.0,     # Center
        50.0,    # Right stage
        95.0,    # Stage right edge
        240.0,   # Blastzone right
    ])

    print(f"\nTesting {len(test_x_values)} X positions:")
    print("-" * 80)

    for x_val in test_x_values:
        x_tensor = torch.tensor([x_val])
        bucket_idx = quantize_future_position(x_tensor, 'x')
        reconstructed = dequantize_future_position(bucket_idx, 'x')
        error = (reconstructed - x_val).abs().item()

        status = "✓" if error < 10.0 else "⚠️"
        print(f"{status} x={x_val.item():7.1f} → bucket[{bucket_idx.item():2d}] → "
              f"{reconstructed.item():7.1f} (error: {error:5.1f})")

    print()


def test_batch_processing():
    """Test that batched quantization works correctly."""
    print("=" * 80)
    print("TEST: Batch Processing")
    print("=" * 80)

    # Create a batch of positions
    batch_y = torch.tensor([
        [-20.0, -10.0, 0.0, 10.0],
        [30.0, 60.0, 100.0, -50.0],
    ])  # Shape: [2, 4]

    print(f"\nInput batch shape: {batch_y.shape}")
    print(f"Input values:\n{batch_y}")

    # Quantize
    bucket_indices = quantize_future_position(batch_y, 'y')
    print(f"\nBucket indices:\n{bucket_indices}")

    # Dequantize
    reconstructed = dequantize_future_position(bucket_indices, 'y')
    print(f"\nReconstructed values:\n{reconstructed}")

    # Check errors
    errors = (reconstructed - batch_y).abs()
    print(f"\nAbsolute errors:\n{errors}")
    print(f"Max error: {errors.max().item():.2f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("FUTURE POSITION QUANTIZATION TEST SUITE")
    print("=" * 80)
    print()

    # Run all tests
    test_quantize_dequantize_roundtrip()
    test_boundary_values()
    test_negative_y_bias()
    test_bucket_midpoint_calculation()
    test_x_axis_quantization()
    test_batch_processing()

    print("=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)
