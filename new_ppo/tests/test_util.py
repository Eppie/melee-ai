import unittest
import torch
import numpy as np
from unittest.mock import patch

from new_ppo.util import compute_gae, get_feature_swap_indices, swap_player_features

class TestUtil(unittest.TestCase):
    def test_compute_gae_simple(self):
        """Test GAE with simple deterministic values."""
        # Setup: T=3, N=1
        rewards = torch.tensor([[1.0], [1.0], [1.0]])
        values = torch.tensor([[1.0], [1.0], [1.0]])
        dones = torch.tensor([[0.0], [0.0], [0.0]])
        next_val = torch.tensor([1.0])
        
        gamma = 0.9
        lam = 0.5
        
        # Calc manually:
        # t=2: delta = r + g*V_next - V = 1 + 0.9*1 - 1 = 0.9
        #      adv = delta = 0.9
        # t=1: delta = 1 + 0.9*1 - 1 = 0.9
        #      adv = delta + g*lam*next_adv = 0.9 + 0.9*0.5*0.9 = 0.9 + 0.405 = 1.305
        # t=0: delta = 0.9
        #      adv = 0.9 + 0.45 * 1.305 = 0.9 + 0.58725 = 1.48725
        
        adv = compute_gae(rewards, values, dones, next_val, gamma, lam)
        
        self.assertTrue(torch.allclose(adv[2], torch.tensor([0.9])))
        self.assertTrue(torch.allclose(adv[1], torch.tensor([1.305])))
        self.assertTrue(torch.allclose(adv[0], torch.tensor([1.48725])))

    def test_compute_gae_dones(self):
        """Test GAE reset on done."""
        # Setup: T=3, N=1. Step 1 is DONE.
        rewards = torch.tensor([[1.0], [1.0], [1.0]])
        values = torch.tensor([[1.0], [1.0], [1.0]])
        dones = torch.tensor([[0.0], [1.0], [0.0]]) # t=1 is terminal
        next_val = torch.tensor([1.0])
        
        gamma = 0.9
        lam = 0.5
        
        adv = compute_gae(rewards, values, dones, next_val, gamma, lam)
        
        # t=1 is done. next_non_terminal = 0.
        # t=1: delta = r + g*V_next*0 - V = 1 + 0 - 1 = 0.0
        #      adv = delta = 0.0 (Bootstrap cut)
        self.assertTrue(torch.allclose(adv[1], torch.tensor([0.0])))
        
        # t=0: delta = 1 + 0.9*1*1 - 1 = 0.9
        #      adv = delta + g*l*next_adv = 0.9 + 0.45 * 0.0 = 0.9
        self.assertTrue(torch.allclose(adv[0], torch.tensor([0.9])))

    @patch("new_ppo.util.get_feature_names")
    def test_feature_swap(self, mock_get_names):
        """Test that p1_x swaps with p2_x."""
        mock_get_names.return_value = ["p1_pos_x", "p1_pos_y", "p2_pos_x", "p2_pos_y", "stage_id"]
        
        # Indices should be:
        # p1_pos_x (0) <-> p2_pos_x (2)
        # p1_pos_y (1) <-> p2_pos_y (3)
        # stage_id (4) <-> stage_id (4)
        expected_indices = [2, 3, 0, 1, 4]
        
        indices = get_feature_swap_indices()
        self.assertEqual(indices, expected_indices)
        
        # Test Tensor Swap
        t = torch.tensor([10, 20, 30, 40, 50])
        swapped = swap_player_features(t, torch.tensor(indices))
        
        self.assertTrue(torch.equal(swapped, torch.tensor([30, 40, 10, 20, 50])))

if __name__ == "__main__":
    unittest.main()
