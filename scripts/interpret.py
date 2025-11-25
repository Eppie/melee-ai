"""
interpret.py

Implementation of Anthropic's Mechanistic Interpretability techniques
(Sparse Autoencoders) for the Nano Melee GPT model.

Techniques applied:
1. Activation Extraction (Hooking residual streams)
2. Sparse Dictionary Learning (Training an SAE)
3. Feature Interpretation (Correlating latent features with GameState)
4. Dead Neuron Resampling (Fixing feature collapse)
"""

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

# Import your codebase
from config import init_config, get_config
from model.nano_gpt import GPT
from train.setup import parse_cli_overrides
from train import find_latest_checkpoint
from utils import _resolve_device
from column_map import ColumnMap

# Imports for Data Loading
from window_dataset import RandomWindowSampler, worker_init_fn
from validation import PreloadedWindowDataset

# -----------------------------------------------------------------------------
# 1. The Sparse Autoencoder (SAE)
# -----------------------------------------------------------------------------


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, expansion_factor: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = input_dim * expansion_factor

        # Encoder: W_enc (projection) + b_enc (bias)
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(input_dim, self.feature_dim))
        )
        self.b_enc = nn.Parameter(torch.zeros(self.feature_dim))

        # Decoder: W_dec (reconstruction) + b_dec (bias)
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.feature_dim, input_dim))
        )
        self.b_dec = nn.Parameter(torch.zeros(input_dim))

        # Initialize bias to 0.0 (since data is centered)
        self.b_dec.data.zero_()

        # Initialize encoder bias to negative to encourage sparsity at start
        self.b_enc.data.fill_(-0.1)

        # Tie weights initially (optional, helps stability)
        with torch.no_grad():
            self.W_enc.data = self.W_dec.data.T.clone()

        self.norm_decoder_weights()

    def norm_decoder_weights(self):
        """Anthropic technique: Keep decoder weight columns normalized to unit length."""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(
        self, x: torch.Tensor, l1_coeff: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        recons = self.decode(acts)

        # Losses
        loss_recons = F.mse_loss(recons, x)
        loss_l1 = l1_coeff * acts.sum()
        loss_total = loss_recons + loss_l1

        return loss_total, recons, acts, loss_recons


# -----------------------------------------------------------------------------
# 2. Activation Buffer
# -----------------------------------------------------------------------------


class ActivationBuffer:
    def __init__(
        self, model: GPT, layer_idx: int, buffer_size: int = 100_000, device="cpu"
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.buffer_size = buffer_size
        self.device = device
        self.activations = []
        self.ptr = 0
        self.hook_handle = model.blocks[layer_idx].register_forward_hook(self._hook_fn)
        self.storage = None
        self.mean = None
        self.std = None

    def _hook_fn(self, module, input, output):
        flat_act = output.detach().reshape(-1, output.shape[-1])
        self.activations.append(flat_act)

    def fill_buffer(self, dataloader, colmap: ColumnMap):
        self.activations = []
        self.model.eval()

        print(f"Filling buffer (target: {self.buffer_size})...")
        with torch.no_grad():
            for batch in dataloader:
                X = batch["X"].to(self.device)
                from train.batch_utils import build_model_inputs

                inputs_td = build_model_inputs(X, colmap)
                self.model(inputs_td)
                if sum(a.shape[0] for a in self.activations) >= self.buffer_size:
                    break

        full_data = torch.cat(self.activations, dim=0)

        # Normalize
        self.mean = full_data.mean(dim=0)
        self.std = full_data.std(dim=0) + 1e-6
        full_data = (full_data - self.mean) / self.std

        perm = torch.randperm(full_data.shape[0])
        self.storage = full_data[perm][: self.buffer_size]
        self.activations = []

    def get_batch(self, batch_size: int) -> torch.Tensor:
        if self.storage is None:
            raise ValueError("Buffer empty.")
        if self.ptr + batch_size > self.storage.shape[0]:
            self.ptr = 0
            # Reshuffle
            perm = torch.randperm(self.storage.shape[0])
            self.storage = self.storage[perm]

        batch = self.storage[self.ptr : self.ptr + batch_size]
        self.ptr += batch_size
        return batch

    def close(self):
        self.hook_handle.remove()


# -----------------------------------------------------------------------------
# 3. Analysis & Helper Functions
# -----------------------------------------------------------------------------


def resample_dead_neurons(sae, optimizer, batch_acts, feature_acts):
    """
    Anthropic Trick: If neurons are dead, reset them to point at
    data examples that are currently poorly reconstructed.
    """
    with torch.no_grad():
        # Identify dead neurons (those that didn't fire in this batch)
        # In a real impl, we track this over time, but per-batch is a decent proxy for "very dead"
        fired_mask = (feature_acts > 0).any(dim=0)
        dead_indices = (~fired_mask).nonzero(as_tuple=True)[0]

        if len(dead_indices) == 0:
            return 0

        # Calculate residuals (error)
        recons = sae.decode(feature_acts)
        residuals = batch_acts - recons  # [B, InputDim]

        # Pick random residuals to re-initialize dead features
        # We calculate the squared error of each sample
        errors = residuals.pow(2).sum(dim=1)
        # Probability of picking a sample is proportional to its error
        probs = errors / errors.sum()

        # Sample indices
        new_directions_idx = torch.multinomial(
            probs, num_samples=len(dead_indices), replacement=True
        )
        new_directions = residuals[new_directions_idx]  # [N_dead, InputDim]

        # Normalize
        new_directions = F.normalize(new_directions, dim=1)

        # Reset Decoder Weights
        sae.W_dec.data[dead_indices] = new_directions

        # Reset Encoder Weights (normalized transpose)
        sae.W_enc.data[:, dead_indices] = new_directions.T * 0.2  # Small scale

        # Reset Encoder Bias
        sae.b_enc.data[dead_indices] = 0.0

        # Reset optimizer state for these parameters (important for Adam)
        # This is complex in PyTorch, so we simply zero the gradient for now
        if sae.W_enc.grad is not None:
            sae.W_enc.grad[:, dead_indices] = 0
        if sae.W_dec.grad is not None:
            sae.W_dec.grad[dead_indices] = 0

        return len(dead_indices)


def analyze_features(sae, buffer, colmap, dataloader, top_k_features=100):
    print(f"\n--- Interpreting Layer {buffer.layer_idx} ---")
    sae_acts_list = []
    input_feats_list = []
    buffer.activations = []

    batch_limit = 20
    from train.batch_utils import build_model_inputs

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            print(f"\nBatch {i}")
            if i >= batch_limit:
                break
            X = batch["X"].to(buffer.device)
            inputs_td = build_model_inputs(X, colmap)
            buffer.model(inputs_td)
            resid_stream = buffer.activations[-1]

            # IMPORTANT: Use the same normalization as training
            resid_stream = (
                resid_stream - buffer.mean.to(buffer.device)
            ) / buffer.std.to(buffer.device)

            feature_acts = sae.encode(resid_stream)
            sae_acts_list.append(feature_acts.cpu())
            flat_inputs = X.reshape(-1, X.shape[-1])
            input_feats_list.append(flat_inputs.cpu())
            buffer.activations = []

    all_sae_acts = torch.cat(sae_acts_list, dim=0)
    all_inputs = torch.cat(input_feats_list, dim=0)

    # Filter for features that are actually active
    max_acts, _ = all_sae_acts.max(dim=0)
    active_mask = max_acts > 0.5  # Filter out noise
    active_indices = active_mask.nonzero(as_tuple=True)[0]

    if len(active_indices) == 0:
        print("No features active above noise threshold.")
        return

    # Sort active features by activity
    sorted_vals, sort_idx = torch.sort(max_acts[active_indices], descending=True)
    top_indices = active_indices[sort_idx][:top_k_features]

    feature_names = colmap.feat_names
    print(f"{'SAE Feat':<10} | {'Max Act':<10} | {'Top Correlated Game Inputs'}")
    print("-" * 80)

    for feat_idx in top_indices:
        feat_idx = feat_idx.item()
        f_acts = all_sae_acts[:, feat_idx]
        f_centered = f_acts - f_acts.mean()
        f_std = f_acts.std() + 1e-6

        correlations = []
        for inp_idx in range(all_inputs.shape[1]):
            inp_col = all_inputs[:, inp_idx]
            # Only check correlation if input has variance
            if inp_col.std() < 1e-6:
                continue

            i_centered = inp_col - inp_col.mean()
            i_std = inp_col.std() + 1e-6
            corr = (f_centered @ i_centered) / (f_centered.shape[0] * f_std * i_std)
            correlations.append((feature_names[inp_idx], corr.item()))

        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        # Filter for meaningful correlations (> 0.1)
        meaningful = [c for c in correlations[:3] if abs(c[1]) > 0.1]
        if not meaningful:
            top_str = "(Uncorrelated / Abstract)"
        else:
            top_str = ", ".join([f"{n}: {c:.2f}" for n, c in meaningful])

        print(f"{feat_idx:<10} | {max_acts[feat_idx]:<10.4f} | {top_str}")


# -----------------------------------------------------------------------------
# 4. Main
# -----------------------------------------------------------------------------


def main():
    init_config()
    config = get_config()
    device = _resolve_device()

    print("Loading Dataset...")
    ds = PreloadedWindowDataset(
        config.zarr.out_root,
        progress=True,
    )
    loader = DataLoader(
        ds,
        batch_size=config.train.batch_size,
        sampler=RandomWindowSampler(index=ds.index),
        num_workers=0,
        worker_init_fn=worker_init_fn,
    )
    colmap = ColumnMap.from_dataset(ds)

    print("Loading Model...")
    ckpt_path = find_latest_checkpoint(Path(config.train.out_dir))
    model = GPT(config).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    model.eval()

    target_layer = config.model.n_layer // 2
    embedding_dim = config.model.n_embd
    print(f"Layer: {target_layer}, Dim: {embedding_dim}")

    # Hyperparams
    sae = SparseAutoencoder(input_dim=embedding_dim, expansion_factor=8).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=3e-4)  # Lower LR for stability

    buffer = ActivationBuffer(model, target_layer, buffer_size=100000, device=device)
    buffer.fill_buffer(loader, colmap)

    print("Training SAE with L1 Warmup & Resampling...")
    total_steps = 5000
    warmup_steps = 2000  # No L1 for first 2000 steps
    final_l1 = 1e-5  # Target L1 (slightly higher than 1e-5 to enforce sparsity)

    start_t = time.time()
    for step in range(total_steps):
        batch_acts = buffer.get_batch(4096)

        # L1 Scheduler: Linear ramp from 0 to final_l1
        if step < warmup_steps:
            l1_coeff = 0.0
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            l1_coeff = final_l1 * min(1.0, progress)

        optimizer.zero_grad()
        loss, recons, feature_acts, mse = sae(batch_acts, l1_coeff)
        loss.backward()
        optimizer.step()
        sae.norm_decoder_weights()

        # Dead Neuron Resampling (Every 1000 steps, after warmup)
        resurrected = 0
        if step > warmup_steps and step % 1000 == 0:
            resurrected = resample_dead_neurons(
                sae, optimizer, batch_acts, feature_acts
            )

        if step % 500 == 0:
            l0 = (
                (feature_acts > 0.01).float().sum(dim=1).mean().item()
            )  # Threshold > 0 for stability
            print(
                f"Step {step}: L1_Coeff {l1_coeff:.1e} | MSE {mse.item():.4f} | L0 {l0:.1f} | Resurrected {resurrected}"
            )

    print(f"SAE Trained in {time.time() - start_t:.1f}s")
    analyze_features(sae, buffer, colmap, loader)
    buffer.close()


if __name__ == "__main__":
    main()
