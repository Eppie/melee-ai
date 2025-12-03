"""Tests for ring buffer and RoPE index computation."""

import numpy as np
import pytest
import torch

from ppo.shared_memory import compute_rope_indices


class TestRoPEIndices:
    """Test RoPE index computation for ring buffer wraparound."""

    def test_no_wraparound(self):
        """Test case where ring hasn't wrapped (t_mod < seq_len - 1)."""
        # If t_mod = 0 (just wrote to position 0), oldest is at position 1
        # But if we've only filled [0], then oldest is actually at 0
        # This test assumes ring is full

        # t_mod = 0 means we just wrote to position 0
        # Oldest frame is at position 1
        # Ring order: [1, 2, 3, ..., 255, 0]
        # RoPE order: [0, 1, 2, ..., 254, 255]
        indices = compute_rope_indices(t_mod=0, seq_len=256)

        expected = torch.arange(256)
        assert torch.equal(indices, expected)

    def test_wraparound_mid(self):
        """Test wraparound in middle of sequence."""
        # t_mod = 10 means we just wrote to position 10
        # Oldest frame is at position 11
        # Ring order: [11, 12, ..., 255, 0, 1, ..., 10]
        # RoPE order: [0, 1, ..., 244, 245, 246, ..., 255]
        indices = compute_rope_indices(t_mod=10, seq_len=256)

        expected = torch.cat([torch.arange(11, 256), torch.arange(0, 11)])
        assert torch.equal(indices, expected)

    def test_wraparound_end(self):
        """Test wraparound at end of sequence."""
        # t_mod = 255 means we just wrote to position 255
        # Oldest frame is at position 0 (wraps around)
        # Ring order: [0, 1, 2, ..., 255]
        # RoPE order: [0, 1, 2, ..., 255]
        indices = compute_rope_indices(t_mod=255, seq_len=256)

        expected = torch.arange(256)
        assert torch.equal(indices, expected)

    def test_various_positions(self):
        """Test various t_mod positions."""
        seq_len = 256

        for t_mod in [0, 1, 10, 50, 100, 127, 128, 200, 254, 255]:
            indices = compute_rope_indices(t_mod=t_mod, seq_len=seq_len)

            # Indices should be a permutation of [0, 1, ..., 255]
            assert len(indices) == seq_len
            assert set(indices.tolist()) == set(range(seq_len))

            # First index should correspond to oldest frame
            oldest_pos = (t_mod + 1) % seq_len
            if oldest_pos == 0:
                assert indices[0].item() == 0
            else:
                assert indices[0].item() == oldest_pos

    def test_device_placement(self):
        """Test that indices are placed on correct device."""
        # CPU
        indices_cpu = compute_rope_indices(t_mod=10, device=torch.device("cpu"))
        assert indices_cpu.device.type == "cpu"

        # CUDA (if available)
        if torch.cuda.is_available():
            indices_gpu = compute_rope_indices(t_mod=10, device=torch.device("cuda:0"))
            assert indices_gpu.device.type == "cuda"

    def test_small_sequence(self):
        """Test with smaller sequence length."""
        # seq_len = 16, t_mod = 5
        # Oldest at position 6
        # Ring: [6, 7, ..., 15, 0, 1, ..., 5]
        # RoPE: [0, 1, ..., 9, 10, 11, ..., 15]
        indices = compute_rope_indices(t_mod=5, seq_len=16)

        expected = torch.cat([torch.arange(6, 16), torch.arange(0, 6)])
        assert torch.equal(indices, expected)


class TestRingBufferLogic:
    """Test ring buffer update logic."""

    def test_incremental_update(self):
        """Test incremental ring buffer updates."""
        seq_len = 256
        t_mod = 0

        # Simulate 1000 updates
        for step in range(1000):
            # Current position
            assert t_mod == step % seq_len

            # Compute RoPE indices
            indices = compute_rope_indices(t_mod=t_mod, seq_len=seq_len)

            # Verify indices form valid sequence
            assert len(indices) == seq_len
            assert indices[0].item() == (t_mod + 1) % seq_len

            # Update t_mod
            t_mod = (t_mod + 1) % seq_len

    def test_feature_ring_access(self):
        """Test feature ring access pattern."""
        # Simulate ring buffer writes
        ring = np.zeros((96, 256, 908), dtype=np.float32)
        t_mod = 0

        # Write 512 frames (2 full cycles)
        for step in range(512):
            # Write frame to current position
            ring[:, t_mod, :] = step

            # Read oldest frame
            oldest_pos = (t_mod + 1) % 256
            oldest_frame = ring[:, oldest_pos, :]

            # After 256 steps, oldest frame should be overwritten
            if step >= 256:
                # Oldest frame is from (step - 255) steps ago
                expected_step = step - 255
                assert np.allclose(oldest_frame[0, 0], expected_step)

            # Update position
            t_mod = (t_mod + 1) % 256

    def test_rope_temporal_consistency(self):
        """Test that RoPE indices maintain temporal order."""
        seq_len = 256

        for t_mod in range(seq_len):
            indices = compute_rope_indices(t_mod=t_mod, seq_len=seq_len)

            # Check that indices form a contiguous temporal sequence
            # (allowing for wraparound)
            for i in range(len(indices) - 1):
                current = indices[i].item()
                next_val = indices[i + 1].item()

                # Next should be current + 1, wrapping at seq_len
                expected_next = (current + 1) % seq_len
                assert next_val == expected_next


class TestBatchedRingBufferAccess:
    """Test batched access patterns for GPU ring buffer."""

    def test_column_copy(self):
        """Test copying single column from ring buffer."""
        # GPU ring buffer
        ring_gpu = torch.zeros((96, 256, 908), dtype=torch.bfloat16)

        # Staging buffer
        staging = torch.randn((96, 1, 908), dtype=torch.float32)

        # Copy to ring at position t_mod
        t_mod = 42
        ring_gpu[:, t_mod, :] = staging[:, 0, :].to(dtype=torch.bfloat16)

        # Verify copy
        assert ring_gpu[:, t_mod, :].shape == (96, 908)

    def test_incremental_h2d_pattern(self):
        """Simulate incremental H2D copy pattern."""
        ring_gpu = torch.zeros((96, 256, 908), dtype=torch.bfloat16)
        t_mod = 0

        # Simulate 1000 frames
        for step in range(1000):
            # Create new frame data
            frame_data = torch.full((96, 1, 908), float(step), dtype=torch.float32)

            # Copy only new column
            ring_gpu[:, t_mod, :] = frame_data[:, 0, :].to(dtype=torch.bfloat16)

            # Verify write
            assert torch.allclose(
                ring_gpu[:, t_mod, :].float(),
                torch.full((96, 908), float(step)),
                atol=1e-2,  # bfloat16 precision
            )

            # Update position
            t_mod = (t_mod + 1) % 256

    def test_full_context_read(self):
        """Test reading full context from ring buffer."""
        seq_len = 256
        ring_gpu = torch.zeros((96, seq_len, 908), dtype=torch.bfloat16)

        # Fill ring with incremental values
        for t in range(seq_len):
            ring_gpu[:, t, :] = float(t)

        # Read at different t_mod positions
        for t_mod in [0, 10, 127, 255]:
            # Compute RoPE indices
            rope_indices = compute_rope_indices(t_mod=t_mod, seq_len=seq_len)

            # Read ring in temporal order
            temporal_ring = ring_gpu[:, rope_indices, :]

            # Verify temporal order
            oldest_pos = (t_mod + 1) % seq_len
            for i in range(seq_len):
                expected_value = (oldest_pos + i) % seq_len
                actual_value = temporal_ring[0, i, 0].item()
                assert np.isclose(actual_value, expected_value, atol=1e-2)
