"""IPC primitives for control messages between CRD and S8 shards."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Pipe
from multiprocessing.connection import Connection
from typing import Any, Optional, Tuple


class MessageType(Enum):
    """Message types for CRD ↔ S8 communication."""

    # S8 → CRD
    READY = "READY"  # Shard ready for inference
    ROLLOUT_COMPLETE = "ROLLOUT_COMPLETE"  # Rollout buffer ready
    HEALTH = "HEALTH"  # Health status report
    ERROR = "ERROR"  # Error occurred

    # CRD → S8
    ACTIONS_READY = "ACTIONS_READY"  # Actions computed, proceed
    CHECKPOINT_UPDATE = "CHECKPOINT_UPDATE"  # New policy checkpoint
    SHUTDOWN = "SHUTDOWN"  # Graceful shutdown request


@dataclass
class Message:
    """Structured message for IPC."""

    msg_type: MessageType
    shard_id: int
    payload: Optional[dict[str, Any]] = None

    def to_tuple(self) -> tuple:
        """Serialize to tuple for Pipe transmission."""
        return (self.msg_type.value, self.shard_id, self.payload)

    @classmethod
    def from_tuple(cls, data: tuple) -> Message:
        """Deserialize from tuple."""
        msg_type_str, shard_id, payload = data
        return cls(
            msg_type=MessageType(msg_type_str),
            shard_id=shard_id,
            payload=payload,
        )


class ShardPipe:
    """
    Wrapper around multiprocessing.Pipe for S8 ↔ CRD communication.

    Provides structured message passing with timeouts and error handling.
    """

    def __init__(self, connection: Connection, shard_id: int):
        self.conn = connection
        self.shard_id = shard_id

    def send(self, msg_type: MessageType, payload: Optional[dict] = None):
        """Send structured message."""
        msg = Message(msg_type=msg_type, shard_id=self.shard_id, payload=payload)
        self.conn.send(msg.to_tuple())

    def recv(self, timeout: Optional[float] = None) -> Message:
        """
        Receive structured message with optional timeout.

        Args:
            timeout: Timeout in seconds (None = blocking)

        Returns:
            Message object

        Raises:
            TimeoutError: If timeout exceeded
        """
        if timeout is not None:
            if not self.conn.poll(timeout):
                raise TimeoutError(f"Timeout waiting for message from shard {self.shard_id}")

        data = self.conn.recv()
        return Message.from_tuple(data)

    def poll(self, timeout: float = 0.0) -> bool:
        """Check if message is available (non-blocking)."""
        return self.conn.poll(timeout)

    def close(self):
        """Close connection."""
        self.conn.close()


def create_shard_pipe_pair(shard_id: int) -> Tuple[ShardPipe, ShardPipe]:
    """
    Create a pair of ShardPipe objects for bidirectional communication.

    Returns:
        (coordinator_pipe, shard_pipe): Pipes for CRD and S8 respectively
    """
    parent_conn, child_conn = Pipe()
    coordinator_pipe = ShardPipe(parent_conn, shard_id)
    shard_pipe = ShardPipe(child_conn, shard_id)
    return coordinator_pipe, shard_pipe


class Barrier:
    """
    Barrier for synchronizing CRD with all S8 shards.

    CRD waits for all shards to signal READY before proceeding.
    """

    def __init__(self, pipes: list[ShardPipe], timeout: float = 10.0):
        self.pipes = pipes
        self.num_shards = len(pipes)
        self.timeout = timeout

    def wait_all_ready(self, step_id: int) -> dict[int, Message]:
        """
        Wait for all shards to signal READY.

        Args:
            step_id: Current step ID (for verification)

        Returns:
            Dict mapping shard_id → Message

        Raises:
            TimeoutError: If any shard times out
            ValueError: If received unexpected message type
        """
        ready_messages = {}
        start_time = time.time()

        for pipe in self.pipes:
            elapsed = time.time() - start_time
            remaining = max(0.0, self.timeout - elapsed)

            try:
                msg = pipe.recv(timeout=remaining)
            except TimeoutError:
                raise TimeoutError(
                    f"Shard {pipe.shard_id} timed out waiting for READY (step {step_id})"
                )

            if msg.msg_type != MessageType.READY:
                raise ValueError(
                    f"Shard {pipe.shard_id} sent {msg.msg_type}, expected READY"
                )

            # Verify step_id matches (optional sanity check)
            if msg.payload and msg.payload.get("step_id") != step_id:
                print(
                    f"[WARN] Shard {pipe.shard_id} step_id mismatch: "
                    f"expected {step_id}, got {msg.payload.get('step_id')}"
                )

            ready_messages[pipe.shard_id] = msg

        return ready_messages

    def signal_all_actions_ready(self, step_id: int):
        """Signal all shards that actions are ready."""
        for pipe in self.pipes:
            pipe.send(MessageType.ACTIONS_READY, payload={"step_id": step_id})


class HealthMonitor:
    """
    Monitor health of S8 shards.

    Tracks heartbeats and detects stalled/crashed shards.
    """

    def __init__(self, pipes: list[ShardPipe], heartbeat_interval: float = 5.0):
        self.pipes = pipes
        self.heartbeat_interval = heartbeat_interval
        self.last_heartbeat: dict[int, float] = {
            pipe.shard_id: time.time() for pipe in pipes
        }

    def check_health(self) -> dict[int, bool]:
        """
        Check health of all shards.

        Returns:
            Dict mapping shard_id → is_healthy
        """
        current_time = time.time()
        health_status = {}

        for pipe in self.pipes:
            shard_id = pipe.shard_id
            elapsed = current_time - self.last_heartbeat[shard_id]
            health_status[shard_id] = elapsed < (self.heartbeat_interval * 2)

        return health_status

    def update_heartbeat(self, shard_id: int):
        """Update last heartbeat time for shard."""
        self.last_heartbeat[shard_id] = time.time()
