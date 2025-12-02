from __future__ import annotations
import multiprocessing as mp
import numpy as np
import traceback
import signal
from config.config import init_config
from new_ppo.ppo_types import PPOConfig, InferenceRequest, InferenceResponse
from new_ppo.batched_env import BatchedDolphinEnv

def run_worker(
    worker_id: int,
    config: PPOConfig,
    state_queue: mp.Queue[InferenceRequest],
    action_queue: mp.Queue[InferenceResponse]
) -> None:
    """
    Entry point for the worker process.
    """
    # Ignore SIGINT in worker so parent can handle it
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # Initialize global config in this process
    init_config()
    
    worker = BatchedWorker(worker_id, config, state_queue, action_queue)
    
    try:
        worker.run()
    except Exception as e:
        print(f"CRITICAL: Worker {worker_id} crashed: {e}")
        traceback.print_exc()
    finally:
        worker.close()

class BatchedWorker:
    def __init__(
        self,
        worker_id: int,
        config: PPOConfig,
        state_queue: mp.Queue[InferenceRequest],
        action_queue: mp.Queue[InferenceResponse]
    ):
        self.worker_id = worker_id
        self.config = config
        self.state_queue = state_queue
        self.action_queue = action_queue
        
        # Initialize Environment
        # We assume dolphin_path and iso_path are set in config
        if not config.dolphin_path or not config.iso_path:
            raise ValueError("Dolphin Path and ISO Path must be provided in config.")
            
        self.env = BatchedDolphinEnv(
            num_envs=config.envs_per_worker,
            worker_id=worker_id,
            dolphin_path=config.dolphin_path,
            iso_path=config.iso_path
        )
        
        # Initial Actions (Neutral)
        self.next_p1_actions = self._get_neutral_actions(config.envs_per_worker)
        self.next_p2_actions = self._get_neutral_actions(config.envs_per_worker)

    def run(self) -> None:
        print(f"Worker {self.worker_id} entering main loop.")
        while True:
            # 1. Step Environment
            obs, rewards, dones = self.env.step(self.next_p1_actions, self.next_p2_actions)
            
            # 2. Send Request
            req = InferenceRequest(
                worker_id=self.worker_id,
                obs=obs,
                rewards=rewards,
                dones=dones
            )
            self.state_queue.put(req)
            
            # 3. Wait for Response
            # This blocks until the InferenceServer processes the batch
            # In a robust system, we might add a timeout here to detect server hangs
            try:
                resp: InferenceResponse = self.action_queue.get()
            except (EOFError, KeyboardInterrupt):
                print(f"Worker {self.worker_id} received stop signal (queue closed).")
                break
                
            self.next_p1_actions = resp.p1_actions
            self.next_p2_actions = resp.p2_actions

    def close(self) -> None:
        print(f"Worker {self.worker_id} shutting down.")
        self.env.close()

    def _get_neutral_actions(self, n: int):
        """Returns a batch of neutral actions for initialization."""
        return {
            'main_stick': np.zeros(n, dtype=int),
            'c_stick': np.zeros(n, dtype=int),
            'buttons': np.zeros((n, 5), dtype=bool),
            'shoulder': np.zeros(n, dtype=int)
        }
