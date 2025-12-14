"""Shared memory infrastructure for zero-copy IPC between CRD and S8 shards."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Optional

import numpy as np
import torch
from schema import get_feature_names


# ActionData structured numpy dtype
# main_idx: int32, c_idx: int32, shoulder_idx: int32, buttons: 5×uint8, logp: float32, value: float32
ActionData_dtype = np.dtype(
    [
        ("main_idx", np.int32),
        ("c_idx", np.int32),
        ("shoulder_idx", np.int32),
        ("buttons", np.uint8, (5,)),
        ("logp", np.float32),
        ("value", np.float32),
    ]
)


@dataclass
class FrameData:
    """Single frame worth of data for one environment."""

    features: np.ndarray  # [feature_dim] float32
    reward: float
    done: bool
    mask: bool  # False if warmup, True if valid for training
    feature_dim: Optional[int] = None

    def __post_init__(self):
        expected_dim = self.feature_dim or self.features.shape[0]
        assert self.features.shape == (
            expected_dim,
        ), f"Expected ({expected_dim},), got {self.features.shape}"
        assert (
            self.features.ndim == 1
        ), f"Expected 1D features, got {self.features.ndim}D"
        assert (
            self.features.dtype == np.float32
        ), f"Expected float32, got {self.features.dtype}"
        self.feature_dim = expected_dim


@dataclass
class ActionData:
    """Quantized action + log probability + value estimate."""

    main_idx: int  # 0-63
    c_idx: int  # 0-8
    shoulder_idx: int  # 0-4
    buttons: np.ndarray  # [5] bool
    logp: float  # Log probability of action
    value: float  # Value estimate V(s)

    def to_numpy_struct(self) -> np.ndarray:
        """Convert to structured numpy array for shared memory."""
        arr = np.zeros(1, dtype=ActionData_dtype)[0]
        arr["main_idx"] = self.main_idx
        arr["c_idx"] = self.c_idx
        arr["shoulder_idx"] = self.shoulder_idx
        arr["buttons"] = self.buttons.astype(np.uint8)
        arr["logp"] = self.logp
        arr["value"] = self.value
        return arr

    @classmethod
    def from_numpy_struct(cls, arr: np.ndarray) -> ActionData:
        """Create from structured numpy array."""
        return cls(
            main_idx=int(arr["main_idx"]),
            c_idx=int(arr["c_idx"]),
            shoulder_idx=int(arr["shoulder_idx"]),
            buttons=arr["buttons"].astype(bool),
            logp=float(arr["logp"]),
            value=float(arr["value"]),
        )


class SharedMemorySlab:
    """
    Single contiguous shared memory slab for zero-copy S8→CRD communication.

    Memory layout:
    - Feature ring: [envs_per_shard, context_length, feature_dim] float32
    - Action slots: [envs_per_shard] ActionData structs
    - Ready flags: [envs_per_shard] uint8
    - Metadata: step_id (uint64), t_mod (uint16)
    """

    def __init__(
        self,
        shard_id: int,
        envs_per_shard: int = 8,
        context_length: int = 256,
        feature_dim: Optional[int] = None,
        create: bool = True,
    ):
        self.shard_id = shard_id
        self.envs_per_shard = envs_per_shard
        self.context_length = context_length
        self.feature_dim = feature_dim or len(get_feature_names())
        self.name = f"ppo_shard_{shard_id}"

        # Calculate sizes
        self.feature_ring_size = (
            envs_per_shard * context_length * feature_dim * 4
        )  # float32
        self.ego_action_slots_size = envs_per_shard * ActionData_dtype.itemsize
        self.opp_action_slots_size = envs_per_shard * ActionData_dtype.itemsize
        self.control_size = envs_per_shard  # ready flags (uint8)
        self.metadata_size = 8 + 2  # step_id (uint64) + t_mod (uint16)
        self.total_size = (
            self.feature_ring_size
            + self.ego_action_slots_size
            + self.opp_action_slots_size
            + self.control_size
            + self.metadata_size
        )

        # Create or attach to shared memory
        if create:
            self.shm = shared_memory.SharedMemory(
                create=True,
                size=self.total_size,
                name=self.name,
            )
            # Zero-initialize the memory using numpy
            np.ndarray((self.total_size,), dtype=np.uint8, buffer=self.shm.buf)[:] = 0
        else:
            self.shm = shared_memory.SharedMemory(
                name=self.name,
            )

        # Create numpy views (no copies)
        offset = 0

        # Feature ring: [envs_per_shard, context_length, feature_dim]
        self.features = np.ndarray(
            (envs_per_shard, context_length, feature_dim),
            dtype=np.float32,
            buffer=self.shm.buf[offset : offset + self.feature_ring_size],
        )
        offset += self.feature_ring_size

        # Ego action slots: [envs_per_shard] structured array
        self.ego_actions = np.ndarray(
            (envs_per_shard,),
            dtype=ActionData_dtype,
            buffer=self.shm.buf[offset : offset + self.ego_action_slots_size],
        )
        offset += self.ego_action_slots_size

        # Opponent action slots: [envs_per_shard] structured array
        self.opp_actions = np.ndarray(
            (envs_per_shard,),
            dtype=ActionData_dtype,
            buffer=self.shm.buf[offset : offset + self.opp_action_slots_size],
        )
        offset += self.opp_action_slots_size

        # Ready flags: [envs_per_shard] uint8
        self.ready_flags = np.ndarray(
            (envs_per_shard,),
            dtype=np.uint8,
            buffer=self.shm.buf[offset : offset + self.control_size],
        )
        offset += self.control_size

        # Metadata: step_id (uint64), t_mod (uint16)
        self.metadata_buf = self.shm.buf[offset : offset + self.metadata_size]

    @property
    def step_id(self) -> int:
        """Current step ID (set by CRD)."""
        return struct.unpack("<Q", self.metadata_buf[0:8])[0]

    @step_id.setter
    def step_id(self, value: int):
        struct.pack_into("<Q", self.metadata_buf, 0, value)

    @property
    def t_mod(self) -> int:
        """Current ring position (0-255)."""
        return struct.unpack("<H", self.metadata_buf[8:10])[0]

    @t_mod.setter
    def t_mod(self, value: int):
        struct.pack_into("<H", self.metadata_buf, 8, value)

    def write_action(self, env_id: int, action: ActionData, is_ego: bool = True):
        """
        Write action to shared slot for environment.

        Args:
            env_id: Environment ID (0-7 within shard)
            action: Action data to write
            is_ego: If True, write to ego slot; else opponent slot
        """
        if is_ego:
            self.ego_actions[env_id] = action.to_numpy_struct()
        else:
            self.opp_actions[env_id] = action.to_numpy_struct()

    def read_action(self, env_id: int, is_ego: bool = True) -> ActionData:
        """
        Read action from shared slot for environment.

        Args:
            env_id: Environment ID (0-7 within shard)
            is_ego: If True, read from ego slot; else opponent slot

        Returns:
            ActionData for the environment
        """
        if is_ego:
            return ActionData.from_numpy_struct(self.ego_actions[env_id])
        else:
            return ActionData.from_numpy_struct(self.opp_actions[env_id])

    def close(self):
        """Close shared memory handle."""
        # Delete numpy array views first to release buffer references
        del self.features
        del self.ego_actions
        del self.opp_actions
        del self.ready_flags
        self.shm.close()

    def unlink(self):
        """Unlink (delete) shared memory segment."""
        try:
            self.shm.unlink()
        except FileNotFoundError:
            pass  # Already unlinked


class PinnedStagingBuffer:
    """
    Pinned host memory staging buffer for async H2D transfers.

    CRD gathers features from all S8 slabs into this buffer, then
    performs a single async H2D copy to GPU.
    """

    def __init__(
        self,
        num_envs: int = 96,
        feature_dim: Optional[int] = None,
    ):
        self.num_envs = num_envs
        self.feature_dim = feature_dim or len(get_feature_names())

        # Allocate pinned memory for one column of the ring
        # Shape: [num_envs, 1, feature_dim]
        # Note: Use empty() instead of zeros() for MPS compatibility
        #       Data will be overwritten before use anyway
        HAS_CUDA = torch.cuda.is_available()
        self.staging = torch.empty(
            (num_envs, 1, feature_dim),
            dtype=torch.float32,
            pin_memory=HAS_CUDA,  # Only pin on CUDA, not MPS
        )

    def copy_from_slabs(
        self,
        slabs: list[SharedMemorySlab],
        t_mod: int,
    ):
        """
        Zero-copy gather from all S8 slabs into staging buffer.

        Args:
            slabs: List of SharedMemorySlab objects (one per shard)
            t_mod: Current ring position (0-255)
        """
        for shard_idx, slab in enumerate(slabs):
            offset = shard_idx * slab.envs_per_shard
            end = offset + slab.envs_per_shard

            # View into shard's feature ring at position t_mod
            # Shape: [envs_per_shard, feature_dim]
            features = slab.features[:, t_mod, :]

            # Copy into staging buffer
            # torch.from_numpy creates a view (no copy)
            self.staging[offset:end, 0, :] = torch.from_numpy(features)

    def get_tensor(self) -> torch.Tensor:
        """Get staging buffer as tensor for H2D transfer."""
        return self.staging


def compute_rope_indices(
    t_mod: int,
    seq_len: int = 256,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Compute RoPE position indices accounting for ring wraparound.

    If ring is at position t_mod=10 (just wrote to position 10),
    the oldest frame is at position 11 (next to be overwritten).

    Ring positions: [11, 12, ..., 255, 0, 1, ..., 10]
    RoPE indices:   [0,  1,  ..., 244, 245, 246, ..., 255]

    Args:
        t_mod: Current ring position (0-255)
        seq_len: Sequence length (default 256)
        device: Target device for tensor

    Returns:
        Tensor of shape [seq_len] with RoPE position indices
    """
    device = device or torch.device("cpu")
    oldest_pos = (t_mod + 1) % seq_len

    if oldest_pos == 0:
        # No wraparound: indices are just [0, 1, ..., seq_len-1]
        return torch.arange(seq_len, device=device)
    else:
        # Wraparound: [oldest_pos, ..., seq_len-1, 0, ..., oldest_pos-1]
        return torch.cat(
            [
                torch.arange(oldest_pos, seq_len, device=device),
                torch.arange(0, oldest_pos, device=device),
            ]
        )
