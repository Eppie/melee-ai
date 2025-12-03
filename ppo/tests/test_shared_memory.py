"""Tests for shared memory infrastructure."""

import numpy as np
import pytest
import torch

from ppo.shared_memory import (
    ActionData,
    ActionData_dtype,
    FrameData,
    PinnedStagingBuffer,
    SharedMemorySlab,
)


# Helper to check if CUDA is available for pinned memory tests
HAS_CUDA = torch.cuda.is_available()


@pytest.fixture
def cleanup_shm():
    """Cleanup shared memory segments after tests."""
    created_slabs = {}  # Track by shard_id to avoid duplicates

    def register(slab):
        # If shard already exists, clean it up first
        if slab.shard_id in created_slabs:
            old_slab = created_slabs[slab.shard_id]
            try:
                old_slab.close()
                old_slab.unlink()
            except Exception:
                pass

        created_slabs[slab.shard_id] = slab
        return slab

    yield register

    # Cleanup all slabs
    for slab in created_slabs.values():
        try:
            slab.close()
            slab.unlink()
        except Exception:
            pass


class TestActionData:
    """Test ActionData dataclass and numpy conversion."""

    def test_to_numpy_struct(self):
        """Test conversion to numpy structured array."""
        action = ActionData(
            main_idx=32,
            c_idx=4,
            shoulder_idx=2,
            buttons=np.array([True, False, True, False, True]),
            logp=-1.5,
            value=0.75,
        )

        arr = action.to_numpy_struct()

        assert arr["main_idx"] == 32
        assert arr["c_idx"] == 4
        assert arr["shoulder_idx"] == 2
        assert np.array_equal(arr["buttons"], np.array([1, 0, 1, 0, 1], dtype=np.uint8))
        assert np.isclose(arr["logp"], -1.5)
        assert np.isclose(arr["value"], 0.75)

    def test_from_numpy_struct(self):
        """Test creation from numpy structured array."""
        arr = np.zeros(1, dtype=ActionData_dtype)[0]
        arr["main_idx"] = 16
        arr["c_idx"] = 7
        arr["shoulder_idx"] = 3
        arr["buttons"] = np.array([0, 1, 0, 1, 0], dtype=np.uint8)
        arr["logp"] = -2.3
        arr["value"] = 0.42

        action = ActionData.from_numpy_struct(arr)

        assert action.main_idx == 16
        assert action.c_idx == 7
        assert action.shoulder_idx == 3
        assert np.array_equal(action.buttons, np.array([False, True, False, True, False]))
        assert np.isclose(action.logp, -2.3)
        assert np.isclose(action.value, 0.42)

    def test_roundtrip(self):
        """Test roundtrip conversion."""
        original = ActionData(
            main_idx=50,
            c_idx=8,
            shoulder_idx=4,
            buttons=np.array([True, True, False, False, True]),
            logp=-0.8,
            value=0.9,
        )

        arr = original.to_numpy_struct()
        recovered = ActionData.from_numpy_struct(arr)

        assert recovered.main_idx == original.main_idx
        assert recovered.c_idx == original.c_idx
        assert recovered.shoulder_idx == original.shoulder_idx
        assert np.array_equal(recovered.buttons, original.buttons)
        assert np.isclose(recovered.logp, original.logp)
        assert np.isclose(recovered.value, original.value)


class TestFrameData:
    """Test FrameData dataclass."""

    def test_valid_frame_data(self):
        """Test valid FrameData creation."""
        features = np.random.randn(908).astype(np.float32)
        frame = FrameData(features=features, reward=1.5, done=False, mask=True)

        assert frame.features.shape == (908,)
        assert frame.features.dtype == np.float32
        assert frame.reward == 1.5
        assert not frame.done
        assert frame.mask

    def test_invalid_shape(self):
        """Test that invalid shape raises error."""
        features = np.random.randn(100).astype(np.float32)

        with pytest.raises(AssertionError):
            FrameData(features=features, reward=0.0, done=False, mask=True)

    def test_invalid_dtype(self):
        """Test that invalid dtype raises error."""
        features = np.random.randn(908).astype(np.float64)

        with pytest.raises(AssertionError):
            FrameData(features=features, reward=0.0, done=False, mask=True)


class TestSharedMemorySlab:
    """Test SharedMemorySlab class."""

    def test_create(self, cleanup_shm):
        """Test creating shared memory slab."""
        # Create slab
        slab = cleanup_shm(SharedMemorySlab(shard_id=0, envs_per_shard=8, create=True))

        # Verify shapes
        assert slab.features.shape == (8, 256, 908)
        assert slab.ego_actions.shape == (8,)
        assert slab.opp_actions.shape == (8,)
        assert slab.ready_flags.shape == (8,)

        # Verify dtypes
        assert slab.features.dtype == np.float32
        assert slab.ego_actions.dtype == ActionData_dtype
        assert slab.opp_actions.dtype == ActionData_dtype
        assert slab.ready_flags.dtype == np.uint8

    def test_feature_ring_write_read(self, cleanup_shm):
        """Test writing and reading features from ring buffer."""
        slab = cleanup_shm(SharedMemorySlab(shard_id=1, envs_per_shard=4, create=True))

        # Write features at different positions
        for env_id in range(4):
            for t_mod in range(10):
                features = np.full(908, env_id * 1000 + t_mod, dtype=np.float32)
                slab.features[env_id, t_mod, :] = features

        # Read back and verify
        for env_id in range(4):
            for t_mod in range(10):
                features = slab.features[env_id, t_mod, :]
                expected = env_id * 1000 + t_mod
                assert np.allclose(features, expected)

    def test_action_slots(self, cleanup_shm):
        """Test writing and reading actions."""
        slab = cleanup_shm(SharedMemorySlab(shard_id=2, envs_per_shard=8, create=True))

        # Write actions
        for env_id in range(8):
            action = ActionData(
                main_idx=env_id,
                c_idx=env_id % 9,
                shoulder_idx=env_id % 5,
                buttons=np.array([env_id % 2 == 0] * 5),
                logp=-float(env_id),
                value=float(env_id) / 10,
            )
            slab.write_action(env_id, action)

        # Read back and verify
        for env_id in range(8):
            action = slab.read_action(env_id)
            assert action.main_idx == env_id
            assert action.c_idx == env_id % 9
            assert action.shoulder_idx == env_id % 5
            assert np.isclose(action.logp, -float(env_id))
            assert np.isclose(action.value, float(env_id) / 10)

    def test_ready_flags(self, cleanup_shm):
        """Test ready flag signaling."""
        slab = cleanup_shm(SharedMemorySlab(shard_id=3, envs_per_shard=8, create=True))

        # Initially all zeros
        assert np.all(slab.ready_flags == 0)

        # Set some flags
        slab.ready_flags[0] = 1
        slab.ready_flags[3] = 1
        slab.ready_flags[7] = 1

        # Verify
        assert slab.ready_flags[0] == 1
        assert slab.ready_flags[1] == 0
        assert slab.ready_flags[3] == 1
        assert slab.ready_flags[7] == 1

        # Clear all
        slab.ready_flags[:] = 0
        assert np.all(slab.ready_flags == 0)

    def test_metadata(self, cleanup_shm):
        """Test step_id and t_mod metadata."""
        slab = cleanup_shm(SharedMemorySlab(shard_id=4, envs_per_shard=8, create=True))

        # Test step_id
        slab.step_id = 12345
        assert slab.step_id == 12345

        # Test t_mod
        slab.t_mod = 127
        assert slab.t_mod == 127

        # Test wraparound
        slab.t_mod = 255
        assert slab.t_mod == 255

        slab.t_mod = 0
        assert slab.t_mod == 0


class TestPinnedStagingBuffer:
    """Test PinnedStagingBuffer class."""

    def test_allocation(self):
        """Test buffer allocation."""
        buffer = PinnedStagingBuffer(num_envs=96, feature_dim=908)

        assert buffer.staging.shape == (96, 1, 908)
        assert buffer.staging.dtype == torch.float32
        # Only check pinning on CUDA systems
        if HAS_CUDA:
            assert buffer.staging.is_pinned()

    def test_copy_from_slabs(self, cleanup_shm):
        """Test gathering features from multiple slabs."""
        # Create 3 slabs with 4 envs each (12 total)
        # Use unique shard IDs to avoid conflicts
        slabs = []
        for i in range(3):
            shard_id = 10 + i  # Start at 10 to avoid conflicts with other tests
            slab = cleanup_shm(SharedMemorySlab(shard_id=shard_id, envs_per_shard=4, create=True))
            slabs.append(slab)

        # Write distinct features to each slab
        t_mod = 42
        for i, slab in enumerate(slabs):
            for env_id in range(4):
                value = i * 1000 + env_id
                slab.features[env_id, t_mod, :] = value

        # Gather into staging buffer
        buffer = PinnedStagingBuffer(num_envs=12, feature_dim=908)
        buffer.copy_from_slabs(slabs, t_mod)

        # Verify gathered data
        for i in range(3):
            for env_id in range(4):
                global_env_id = i * 4 + env_id
                expected = i * 1000 + env_id
                actual = buffer.staging[global_env_id, 0, 0].item()
                assert np.isclose(actual, expected)

    def test_get_tensor(self):
        """Test getting tensor for H2D transfer."""
        buffer = PinnedStagingBuffer(num_envs=96)
        tensor = buffer.get_tensor()

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (96, 1, 908)
        # Only check pinning on CUDA systems
        if HAS_CUDA:
            assert tensor.is_pinned()
