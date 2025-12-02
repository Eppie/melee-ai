import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from new_ppo.batched_env import BatchedDolphinEnv
from libmelee.melee.enums import Menu

class TestBatchedEnv(unittest.TestCase):
    @patch("new_ppo.batched_env.Console")
    @patch("new_ppo.batched_env.Controller")
    @patch("new_ppo.batched_env.MenuHelper")
    def setUp(self, mock_menu, mock_ctl, mock_console):
        self.num_envs = 2
        self.feature_dim = 45
        
        # Mock the console instance
        self.mock_console_inst = MagicMock()
        self.mock_console_inst.connect.return_value = True
        mock_console.return_value = self.mock_console_inst
        
        # Mock controller
        self.mock_ctl_inst = MagicMock()
        self.mock_ctl_inst.connect.return_value = True
        mock_ctl.return_value = self.mock_ctl_inst
        
        # Mock Menu
        mock_menu.return_value = MagicMock()

        with patch("new_ppo.batched_env.collect_raw_inputs_from_gamestate") as mock_collect:
             # Mock feature collection to return dummy dict
             mock_collect.return_value.transformed = {f"p1_x": 0.5}
             
             self.env = BatchedDolphinEnv(
                 num_envs=self.num_envs,
                 worker_id=0,
                 dolphin_path="mock",
                 iso_path="mock"
             )

    def test_init(self):
        self.assertEqual(len(self.env.consoles), self.num_envs)
        self.assertEqual(len(self.env.controllers), self.num_envs)

    @patch("new_ppo.batched_env.collect_raw_inputs_from_gamestate")
    def test_step(self, mock_collect):
        # Setup Mocks for Step
        mock_gamestate = MagicMock()
        mock_gamestate.menu_state = Menu.IN_GAME
        
        # Configure console.step to return this gamestate
        # We have self.env.consoles which are mocks
        for c in self.env.consoles:
            c.step.return_value = mock_gamestate
            
        # Configure feature extractor
        # We need it to return different values per call if we want to verify stacking,
        # but for now constant is fine to check shape.
        mock_collect.return_value.transformed = {"p1_action_state": 1.0}
        
        # Mock Actions
        p1_actions = {
            'main_stick': np.zeros(self.num_envs, dtype=int),
            'c_stick': np.zeros(self.num_envs, dtype=int),
            'buttons': np.zeros((self.num_envs, 5), dtype=bool),
            'shoulder': np.zeros(self.num_envs, dtype=int),
        }
        p2_actions = p1_actions.copy()
        
        # Run Step
        obs, rewards, dones = self.env.step(p1_actions, p2_actions)
        
        # Verify Shapes
        self.assertEqual(obs.shape, (self.num_envs, self.env.feature_dim))
        self.assertEqual(rewards.shape, (self.num_envs,))
        self.assertEqual(dones.shape, (self.num_envs,))
        
        # Verify calls
        # collect_raw_inputs should be called num_envs times
        self.assertEqual(mock_collect.call_count, self.num_envs)

if __name__ == "__main__":
    unittest.main()
