"""Test optimized augment_batch_with_horizons."""

import torch
from train.batch_utils import augment_batch_with_horizons

# Mock column_map
class MockColumnMap:
    def __init__(self):
        self.y_future_x_keyframes = list(range(30, 39))
        self.y_future_y_keyframes = list(range(39, 48))
        self.y_future_valid_mask = list(range(48, 57))

B, L, F = 128, 256, 50
X = torch.randn(B, L, F, device='cuda')
Y = torch.randn(B, L, 60, device='cuda')

colmap = MockColumnMap()

print("Testing optimized implementation...")
print(f"Input: X={X.shape}, Y={Y.shape}")

# Run multiple times to test caching
for i in range(3):
    X_aug, fx, fy, valid = augment_batch_with_horizons(X, Y, colmap)
    print(f"\nRun {i+1}:")
    print(f"  X_aug: {X_aug.shape}")
    print(f"  future_x: {fx.shape}")
    print(f"  future_y: {fy.shape}")
    print(f"  valid: {valid.shape}")

    # Verify shapes
    assert X_aug.shape == (B, L, F+1), f"X_aug shape mismatch"
    assert fx.shape == (B, L), f"future_x shape mismatch"
    assert fy.shape == (B, L), f"future_y shape mismatch"
    assert valid.shape == (B, L), f"valid shape mismatch"

print("\n✓ All tests passed!")
