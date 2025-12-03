"""ArenaShard: Process managing 8 EnvWorker threads."""

from __future__ import annotations

import time
from threading import Thread
from typing import List

from column_map import ColumnMap

from .config import PPOConfig
from .env_worker import EnvWorker
from .ipc import MessageType, ShardPipe
from .shared_memory import SharedMemorySlab


class ArenaShard:
    """
    Manages 8 EnvWorker threads and coordinates with CRD.

    Responsibilities:
    - Spawn 8 ENV threads
    - Allocate shared memory slab
    - Coordinate ready/done signaling with CRD
    - Lightweight orchestration (no heavy compute)
    """

    def __init__(
        self,
        shard_id: int,
        coordinator_pipe: ShardPipe,
        config: PPOConfig,
        column_map: ColumnMap,
    ):
        self.shard_id = shard_id
        self.pipe = coordinator_pipe
        self.config = config
        self.column_map = column_map

        # Allocate shared memory slab
        print(f"[S8-{shard_id}] Allocating shared memory slab")
        self.slab = SharedMemorySlab(
            shard_id=shard_id,
            envs_per_shard=config.envs_per_shard,
            context_length=config.context_length,
            create=True,
        )

        # Spawn ENV workers
        self.workers: List[Thread] = []
        self._spawn_workers()

        print(f"[S8-{shard_id}] ArenaShard initialized with {len(self.workers)} workers")

    def _spawn_workers(self):
        """Spawn EnvWorker threads."""
        for env_id in range(self.config.envs_per_shard):
            worker = EnvWorker(
                shard_id=self.shard_id,
                env_id=env_id,
                shared_slab=self.slab,
                config=self.config,
                column_map=self.column_map,
            )

            # Create daemon thread
            thread = Thread(
                target=worker.run,
                name=f"ENV-{self.shard_id}-{env_id}",
                daemon=True,
            )
            thread.start()
            self.workers.append(thread)

        print(f"[S8-{self.shard_id}] Spawned {len(self.workers)} ENV threads")

    def _wait_all_ready(self):
        """Spin until all 8 ENVs set ready flag."""
        while True:
            # Check if all ready
            if (self.slab.ready_flags == 1).all():
                break
            time.sleep(0.0001)  # Brief yield

    def run(self):
        """Main S8 coordination loop."""
        step_id = 0

        print(f"[S8-{self.shard_id}] Entering main coordination loop")

        try:
            while True:
                # 1. Wait for all 8 ENVs to mark ready
                self._wait_all_ready()

                # 2. Notify CRD that this shard is ready
                self.pipe.send(
                    MessageType.READY,
                    payload={"step_id": step_id},
                )

                # 3. Wait for CRD to signal actions are ready
                msg = self.pipe.recv(timeout=10.0)  # 10s timeout
                if msg.msg_type != MessageType.ACTIONS_READY:
                    print(
                        f"[S8-{self.shard_id}] WARNING: "
                        f"Expected ACTIONS_READY, got {msg.msg_type}"
                    )

                # 4. Clear ready flags to release ENVs
                self.slab.ready_flags[:] = 0

                step_id += 1

                # Periodic health check
                if step_id % 1000 == 0:
                    alive = sum(1 for w in self.workers if w.is_alive())
                    print(
                        f"[S8-{self.shard_id}] Step {step_id}, "
                        f"{alive}/{len(self.workers)} threads alive"
                    )

        except KeyboardInterrupt:
            print(f"[S8-{self.shard_id}] Interrupted, shutting down")
        except Exception as e:
            print(f"[S8-{self.shard_id}] Error in coordination loop: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self._cleanup()

    def _cleanup(self):
        """Cleanup resources."""
        print(f"[S8-{self.shard_id}] Cleaning up")

        # Wait for threads to finish (with timeout)
        for worker in self.workers:
            worker.join(timeout=1.0)

        # Close shared memory
        try:
            self.slab.close()
            self.slab.unlink()
        except Exception as e:
            print(f"[S8-{self.shard_id}] Error cleaning up slab: {e}")

        # Close pipe
        self.pipe.close()

        print(f"[S8-{self.shard_id}] Cleanup complete")


def shard_main(
    shard_id: int,
    pipe: ShardPipe,
    config: PPOConfig,
    column_map: ColumnMap,
):
    """
    Entry point for ArenaShard process.

    Called by CRD via multiprocessing.spawn().
    """
    print(f"[S8-{shard_id}] Shard process starting (PID: {__import__('os').getpid()})")

    try:
        shard = ArenaShard(
            shard_id=shard_id,
            coordinator_pipe=pipe,
            config=config,
            column_map=column_map,
        )
        shard.run()
    except Exception as e:
        print(f"[S8-{shard_id}] Fatal error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print(f"[S8-{shard_id}] Shard process exiting")
