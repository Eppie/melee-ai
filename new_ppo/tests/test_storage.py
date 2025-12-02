import unittest
import torch
from new_ppo.storage import RolloutBuffer

class TestRolloutBuffer(unittest.TestCase):
    def setUp(self):
        self.steps = 4
        self.envs = 2
        self.dim = 3
        self.buffer = RolloutBuffer(self.steps, self.envs, self.dim, device="cpu")

    def test_insert_and_shape(self):
        """Verify data insertion and flattening shapes."""
        
        # Insert dummy data for full rollout
        for i in range(self.steps):
            obs = torch.full((self.envs, self.dim), float(i))
            actions = {
                'main_stick': torch.zeros((self.envs), dtype=torch.long),
                'c_stick': torch.zeros((self.envs), dtype=torch.long),
                'buttons': torch.zeros((self.envs, 5), dtype=torch.bool),
                'shoulder': torch.zeros((self.envs), dtype=torch.long),
            }
            rewards = torch.ones(self.envs)
            dones = torch.zeros(self.envs)
            values = torch.zeros(self.envs)
            log_probs = torch.zeros(self.envs)
            
            self.buffer.insert(obs, actions, rewards, dones, values, log_probs)
            
        self.assertEqual(self.buffer.step, self.steps)
        
        # Compute returns (dummy)
        self.buffer.compute_returns_and_finish(torch.zeros(self.envs), 0.99, 0.95)
        
        # Check pointer reset
        self.assertEqual(self.buffer.step, 0)
        
        # Check Batch
        batch = self.buffer.get_batch()
        
        expected_rows = self.steps * self.envs
        self.assertEqual(batch.obs.shape, (expected_rows, self.dim))
        self.assertEqual(batch.rewards.shape, (expected_rows,))
        
        # Check content logic (Flatten is Time-Major or Batch-Major?)
        # Implementation: self.obs.flatten(0, 1).
        # obs shape was [Steps, Envs, F]
        # Flatten(0, 1) merges Steps and Envs.
        # Sequence: Step0_Env0, Step0_Env1, Step1_Env0...
        
        # Step 0 should be all 0.0s
        self.assertEqual(batch.obs[0, 0].item(), 0.0)
        # Step 1 (index = 1 * Envs + 0 = 2) should be 1.0s
        self.assertEqual(batch.obs[2, 0].item(), 1.0)

if __name__ == "__main__":
    unittest.main()
