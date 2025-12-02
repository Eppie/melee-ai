import unittest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from new_ppo.inference import InferenceServer
from new_ppo.ppo_types import PPOConfig, InferenceRequest

class TestInferenceServer(unittest.TestCase):
    def setUp(self):
        # Patch GPT
        self.patcher_gpt = patch("new_ppo.inference.GPT")
        self.mock_gpt = self.patcher_gpt.start()
        self.addCleanup(self.patcher_gpt.stop)
        
        # Patch get_config
        self.patcher_cfg = patch("new_ppo.inference.get_config")
        self.mock_get_cfg = self.patcher_cfg.start()
        self.addCleanup(self.patcher_cfg.stop)
        
        # Patch swap_player_features - THIS IS CRITICAL
        self.patcher_swap = patch("new_ppo.inference.swap_player_features")
        self.mock_swap = self.patcher_swap.start()
        self.mock_swap.side_effect = lambda x, y: x # Identity
        self.addCleanup(self.patcher_swap.stop)
        
        self.num_workers = 2
        self.envs_per_worker = 2
        self.total_envs = 4
        self.feature_dim = 10
        
        self.config = PPOConfig(
            num_workers=self.num_workers,
            envs_per_worker=self.envs_per_worker,
            feature_dim=self.feature_dim,
            rollout_length=2, # Short rollout
            device="cpu"
        )
        
        # Mock Queues
        self.state_queue = MagicMock()
        self.action_queues = {i: MagicMock() for i in range(self.num_workers)}
        
        # Mock Model Output
        self.mock_model = MagicMock()
        # Ensure .to() returns self so chaining works
        self.mock_model.to.return_value = self.mock_model
        
        # Output needs to be dict of tensors [B, 1, Logits]
        def model_forward(x):
            B = x.shape[0]
            return {
                'main_stick': torch.zeros(B, 1, 5),
                'c_stick': torch.zeros(B, 1, 5),
                'buttons': torch.zeros(B, 1, 5),
                'shoulder': torch.zeros(B, 1, 5),
                'value': torch.zeros(B, 1, 1)
            }
        self.mock_model.side_effect = model_forward
        self.mock_gpt.return_value = self.mock_model
        
        self.server = InferenceServer(self.config, self.state_queue, self.action_queues)

    def test_collect_rollout_logic(self):
        """Verify data aggregation and dispatch logic."""
        
        # Setup Input Data
        # We need (rollout_length + 1) * num_workers inputs in the queue
        # 1 for init, L for the loop
        requests = []
        for _ in range(self.config.rollout_length + 1):
            for w in range(self.num_workers):
                req = InferenceRequest(
                    worker_id=w,
                    obs=np.zeros((self.envs_per_worker, self.feature_dim), dtype=np.float32),
                    rewards=np.zeros(self.envs_per_worker),
                    dones=np.zeros(self.envs_per_worker)
                )
                requests.append(req)
        
        self.state_queue.get.side_effect = requests
        
        # Run
        self.server.collect_rollout()
        
        # Verifications
        
        # 1. Buffer should be full (rollout_length steps)
        self.assertEqual(self.server.buffer.step, 0) # Pointer resets after compute_returns
        
        # 2. Action Queues should have received responses
        # Total responses = rollout_length * num_workers (Init doesn't trigger response, loop does)
        expected_puts = self.config.rollout_length
        for q in self.action_queues.values():
            self.assertEqual(q.put.call_count, expected_puts)
            
        # 3. Check Persistent State
        self.assertIsNotNone(self.server.next_obs)
        self.assertIsNotNone(self.server.next_dones)

    def test_slice_dict(self):
        d = {'a': np.array([0, 1, 2, 3])}
        sliced = self.server._slice_dict(d, 1, 3)
        self.assertTrue(np.array_equal(sliced['a'], np.array([1, 2])))

if __name__ == "__main__":
    unittest.main()
