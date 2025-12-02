import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Bernoulli
from new_ppo.ppo_types import PPOConfig, RolloutBatch
from new_ppo.storage import RolloutBuffer
from model.nano_gpt import GPT

class PPOTrainer:
    def __init__(self, config: PPOConfig, model: GPT):
        self.config = config
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, eps=1e-5)
        self.mse = nn.MSELoss()

    def train_epoch(self, buffer: RolloutBuffer):
        """
        Updates the model using the data in the buffer.
        """
        self.model.train()
        full_batch = buffer.get_batch()
        
        # Basic shuffling and mini-batching
        N = full_batch.obs.shape[0]
        indices = torch.randperm(N)
        
        avg_loss = 0.0
        batches = 0
        
        for start in range(0, N, self.config.batch_size):
            end = start + self.config.batch_size
            if end > N: break
            
            idx = indices[start:end]
            self._update_step(full_batch, idx)
            batches += 1
            
        print(f"Trainer: Update Complete. Steps={N}, Batches={batches}")

    def _update_step(self, batch: RolloutBatch, idx: torch.Tensor):
        # 1. Re-run model on batch
        # Obs: [B, F] -> [B, 1, F]
        obs = batch.obs[idx].unsqueeze(1)
        output = self.model(obs)
        
        # 2. Evaluate Actions (Get new log_probs and values)
        # Squeeze time dim [B, 1, ...] -> [B, ...]
        main_logits = output['main_stick'].squeeze(1)
        c_logits = output['c_stick'].squeeze(1)
        sh_logits = output['shoulder'].squeeze(1)
        but_logits = output['buttons'].squeeze(1)
        values = output['value'].squeeze(1).squeeze(-1)
        
        # Re-build distributions
        d_main = Categorical(logits=main_logits)
        d_c = Categorical(logits=c_logits)
        d_sh = Categorical(logits=sh_logits)
        d_but = Bernoulli(logits=but_logits)
        
        # Calculate Log Probs of TAKEN actions
        act_main = batch.actions['main_stick'][idx]
        act_c = batch.actions['c_stick'][idx]
        act_sh = batch.actions['shoulder'][idx]
        act_but = batch.actions['buttons'][idx].float() # Bernoulli needs float target? No, usually works. 
        # Wait, Bernoulli.log_prob takes value (0 or 1).
        
        new_log_prob = (
            d_main.log_prob(act_main) +
            d_c.log_prob(act_c) +
            d_sh.log_prob(act_sh) +
            d_but.log_prob(act_but).sum(dim=-1)
        )
        
        # Entropy
        entropy = (
            d_main.entropy() + 
            d_c.entropy() + 
            d_sh.entropy() + 
            d_but.entropy().sum(dim=-1)
        ).mean()
        
        # 3. PPO Loss
        old_log_prob = batch.log_probs[idx]
        advantages = batch.advantages[idx]
        returns = batch.returns[idx]
        
        # Normalize Adv
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        ratio = torch.exp(new_log_prob - old_log_prob)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        value_loss = self.mse(values, returns)
        
        loss = policy_loss + self.config.vf_coef * value_loss - self.config.ent_coef * entropy
        
        # 4. Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
