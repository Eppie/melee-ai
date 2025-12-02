from __future__ import annotations
import torch
import torch.nn as nn
import multiprocessing as mp
import numpy as np
import time
from typing import List, Dict, Tuple, Optional

from new_ppo.ppo_types import PPOConfig, InferenceRequest, InferenceResponse
from new_ppo.storage import RolloutBuffer
from new_ppo.util import get_feature_swap_indices, swap_player_features

# Assuming the Model interface matches what we need (inputs->outputs)
# We'll need to adapt the GPT import or wrap it.
from model.nano_gpt import GPT
from config.config import get_config # Need global config for model init

class InferenceServer:
    """
    Central coordinator. 
    - Collects observations from all workers.
    - Batches them.
    - Runs Inference (P1 & P2).
    - Dispatches actions.
    - Stores P1 experiences.
    """
    def __init__(
        self,
        config: PPOConfig,
        state_queue: mp.Queue[InferenceRequest],
        action_queues: Dict[int, mp.Queue[InferenceResponse]]
    ):
        self.config = config
        self.state_queue = state_queue
        self.action_queues = action_queues
        self.device = torch.device(config.device)
        
        # Initialize Models
        # We need to load the global config to initialize GPT correctly
        global_cfg = get_config()
        
        print(f"InferenceServer: Loading Learner Model on {self.device}...")
        self.learner = GPT(global_cfg).to(self.device)
        self.learner.train() # Train mode for dropout/etc if applicable
        
        print(f"InferenceServer: Loading Opponent Model...")
        self.opponent = GPT(global_cfg).to(self.device)
        if config.opponent_path:
            ckpt = torch.load(config.opponent_path, map_location=self.device)
            self.opponent.load_state_dict(ckpt['model'])
            print(f"InferenceServer: Loaded opponent from {config.opponent_path}")
        else:
            # Self-Play: Share weights (reference copy or state_dict copy?)
            # For true self-play (training against itself), we usually want 
            # a snapshot or the live model. 
            # Let's use the live model reference for now (shared parameter update).
            self.opponent = self.learner 
            print("InferenceServer: Self-play mode (Opponent is Learner).")
            
        self.opponent.eval() # Opponent usually in eval mode
        
        # Helpers
        self.swap_indices = torch.tensor(get_feature_swap_indices(), device=self.device)
        self.total_envs = config.num_workers * config.envs_per_worker
        
        # Storage
        self.buffer = RolloutBuffer(
            num_steps=config.rollout_length,
            num_envs=self.total_envs,
            feature_dim=config.feature_dim,
            device=config.device
        )
        
        # Persistent state across rollouts for bootstrapping
        self.next_obs: Optional[torch.Tensor] = None
        self.next_dones: Optional[torch.Tensor] = None

    def collect_rollout(self):
        """
        Runs the environment interaction loop for `rollout_length` steps.
        Fills `self.buffer`.
        """
        print(f"InferenceServer: Starting Rollout Collection ({self.config.rollout_length} steps)...")
        start_time = time.time()
        
        # 0. Initialize Loop (First time only)
        if self.next_obs is None:
            # Collect first batch from ALL workers (neutral actions result)
            self.next_obs, self.next_dones = self._receive_batch()

        for step in range(self.config.rollout_length):
            # 1. Use current observation (self.next_obs)
            obs_t = self.next_obs
            dones_t = self.next_dones
            
            # 2. Inference
            # P1 (Learner)
            obs_in = obs_t.unsqueeze(1) 
            
            with torch.no_grad(): 
                p1_out = self.learner(obs_in)
                
                # Opponent (P2)
                obs_p2 = swap_player_features(obs_t, self.swap_indices).unsqueeze(1)
                p2_out = self.opponent(obs_p2)
                
            # 3. Sample Actions
            p1_actions, p1_log_probs, p1_vals = self._sample_actions(p1_out)
            p2_actions, _, _ = self._sample_actions(p2_out)
            
            # 4. Dispatch Actions (Trigger Worker Step)
            self._dispatch_actions(p1_actions, p2_actions)
            
            # 5. Receive NEXT Observation (Result of Action)
            # We need to receive the full batch including rewards from the transition
            obs_next_t, rewards_t, dones_next_t = self._receive_batch_full()

            # 6. Store Transition (Step t)
            # We store: obs_t, action_t, reward_t (from next), value_t, log_prob_t.
            # done_t? dones in buffer usually means "step t was terminal".
            # dones_next_t tells us if we landed in terminal state.
            # PPO buffer usually stores `dones` corresponding to `obs`.
            # Or `masks`.
            # `compute_gae`: next_non_terminal = 1.0 - dones[t].
            # If `dones[t]` is True, it means `obs[t]` was the last frame.
            # So we should use `dones_next_t`.
            
            self.buffer.insert(
                obs=obs_t,
                actions=p1_actions,
                rewards=rewards_t.to(self.device), # Reward for taking action_t
                dones=dones_next_t.to(self.device), # Did action_t lead to terminal?
                values=p1_vals,
                log_probs=p1_log_probs
            )
            
            # 7. Advance State
            self.next_obs = obs_next_t
            self.next_dones = dones_next_t
                
        # End of Rollout: Compute Returns
        # We use self.next_obs (which is obs_{L}) for bootstrapping
        with torch.no_grad():
             obs_in = self.next_obs.unsqueeze(1)
             val_out = self.learner(obs_in)
             next_value = val_out['value'].squeeze(1).squeeze(-1)
        
        self.buffer.compute_returns_and_finish(next_value, self.config.gamma, self.config.gae_lambda)
        
        fps = (self.config.rollout_length * self.total_envs) / (time.time() - start_time)
        print(f"Rollout Complete. FPS: {fps:.2f}")

    def _receive_batch_full(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper to collect batch from all workers."""
        requests: List[InferenceRequest] = []
        for _ in range(self.config.num_workers):
            req = self.state_queue.get() 
            requests.append(req)
        requests.sort(key=lambda r: r.worker_id)
        
        obs_np = np.concatenate([r.obs for r in requests], axis=0)
        rew_np = np.concatenate([r.rewards for r in requests], axis=0)
        done_np = np.concatenate([r.dones for r in requests], axis=0)
        
        return (
            torch.from_numpy(obs_np).to(self.device),
            torch.from_numpy(rew_np).to(self.device), # keep float
            torch.from_numpy(done_np).to(self.device) # keep float/bool
        )

    def _receive_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy helper for init (ignore rewards)."""
        obs, _, dones = self._receive_batch_full()
        return obs, dones

    def _dispatch_actions(self, p1_actions, p2_actions):
        p1_cpu = self._to_cpu_dict(p1_actions)
        p2_cpu = self._to_cpu_dict(p2_actions)
        chunk = self.config.envs_per_worker
        
        for i in range(self.config.num_workers):
            start = i * chunk
            end = start + chunk
            resp = InferenceResponse(
                p1_actions=self._slice_dict(p1_cpu, start, end),
                p2_actions=self._slice_dict(p2_cpu, start, end)
            )
            self.action_queues[i].put(resp)

    def _sample_actions(self, model_out: Dict[str, torch.Tensor]) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """
        Extracts actions, log_probs, and values from model output.
        Assuming model_out has keys: 'main_stick', 'c_stick', 'buttons', 'shoulder', 'value'
        """
        # Outputs are [B, T, logits] -> squeeze T=1 -> [B, logits]
        main_logits = model_out['main_stick'].squeeze(1)
        c_logits = model_out['c_stick'].squeeze(1)
        but_logits = model_out['buttons'].squeeze(1)
        sh_logits = model_out['shoulder'].squeeze(1)
        val = model_out['value'].squeeze(1).squeeze(-1) # [B]
        
        # Categorical Sampling (Sticks/Shoulder)
        # For prototype: Greedy/Argmax or Categorical? PPO needs stochasticity.
        # Let's use Categorical.
        from torch.distributions import Categorical, Bernoulli
        
        d_main = Categorical(logits=main_logits)
        d_c = Categorical(logits=c_logits)
        d_sh = Categorical(logits=sh_logits)
        d_but = Bernoulli(logits=but_logits)
        
        a_main = d_main.sample()
        a_c = d_c.sample()
        a_sh = d_sh.sample()
        a_but = d_but.sample().bool() # [B, 5]
        
        # Log Probs
        log_prob = (
            d_main.log_prob(a_main) + 
            d_c.log_prob(a_c) + 
            d_sh.log_prob(a_sh) + 
            d_but.log_prob(a_but.float()).sum(dim=-1)
        )
        
        actions = {
            'main_stick': a_main,
            'c_stick': a_c,
            'shoulder': a_sh,
            'buttons': a_but
        }
        return actions, log_prob, val

    def _to_cpu_dict(self, d: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        return {k: v.cpu().numpy() for k, v in d.items()}

    def _slice_dict(self, d: Dict[str, np.ndarray], start: int, end: int) -> Dict[str, np.ndarray]:
        return {k: v[start:end] for k, v in d.items()}