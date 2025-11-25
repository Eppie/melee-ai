"""
interpret_multilayer.py

1. Loops through Layers 0 (Input), Middle (Abstract), End (Output).
2. Prints 'State Snapshots' to explain features.
3. Includes detailed progress logging during SAE training.
"""

import math
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

from config import init_config, get_config
from model.nano_gpt import GPT
from train import find_latest_checkpoint
from utils import _resolve_device, match_state_dict_keys
from column_map import ColumnMap
from window_dataset import RandomWindowSampler, worker_init_fn
from validation import PreloadedWindowDataset


# -----------------------------------------------------------------------------
# SAE Implementation
# -----------------------------------------------------------------------------
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, expansion_factor: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = input_dim * expansion_factor
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(input_dim, self.feature_dim))
        )
        self.b_enc = nn.Parameter(torch.zeros(self.feature_dim))
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.feature_dim, input_dim))
        )
        self.b_dec = nn.Parameter(torch.zeros(input_dim))
        self.b_dec.data.zero_()
        self.b_enc.data.fill_(-0.1)
        with torch.no_grad():
            self.W_enc.data = self.W_dec.data.T.clone()
        self.norm_decoder_weights()

    def norm_decoder_weights(self):
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor, l1_coeff: float):
        acts = self.encode(x)
        recons = self.decode(acts)
        loss = F.mse_loss(recons, x) + l1_coeff * acts.sum()
        return loss, recons, acts, F.mse_loss(recons, x)


# -----------------------------------------------------------------------------
# Buffer
# -----------------------------------------------------------------------------
class ActivationBuffer:
    def __init__(
        self, model: GPT, layer_idx: int, buffer_size: int = 50000, device="cpu"
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.buffer_size = buffer_size
        self.device = device
        self.activations = []
        self.inputs = []  # Keep inputs for context printing
        self.ptr = 0

        # Hook the specific block
        if layer_idx < len(model.blocks):
            block = model.blocks[layer_idx]
            self.hook_handle = block.register_forward_hook(self._hook_fn)
        else:
            # Hook final norm if layer index is out of bounds of blocks
            # This captures the state right before the output heads
            self.hook_handle = model.norm_final.register_forward_hook(self._hook_fn)

        self.storage = None
        self.input_storage = None
        self.mean = None
        self.std = None

    def _hook_fn(self, module, input, output):
        # Output is [Batch, Seq, Dim]
        self.activations.append(output.detach().reshape(-1, output.shape[-1]))

    def fill_buffer(self, dataloader, colmap: ColumnMap):
        self.activations = []
        self.inputs = []
        self.model.eval()

        print(f"  Collecting activations (Target: {self.buffer_size})...")
        with torch.no_grad():
            for batch in dataloader:
                X = batch["X"].to(self.device)
                from train.batch_utils import build_model_inputs

                inputs_td = build_model_inputs(X, colmap)
                self.model(inputs_td)

                # Store the raw inputs associated with these activations
                # X is [B, L, F]. Flatten to [B*L, F]
                self.inputs.append(X.reshape(-1, X.shape[-1]))

                if sum(a.shape[0] for a in self.activations) >= self.buffer_size:
                    break

        full_acts = torch.cat(self.activations, dim=0)
        full_inputs = torch.cat(self.inputs, dim=0)

        # Normalize Acts
        self.mean = full_acts.mean(dim=0)
        self.std = full_acts.std(dim=0) + 1e-6
        full_acts = (full_acts - self.mean) / self.std

        # Shuffle both in unison
        perm = torch.randperm(full_acts.shape[0])
        self.storage = full_acts[perm][: self.buffer_size]
        self.input_storage = full_inputs[perm][: self.buffer_size]

        self.activations = []
        self.inputs = []

    def get_batch(self, batch_size: int):
        if self.ptr + batch_size > self.storage.shape[0]:
            self.ptr = 0
            perm = torch.randperm(self.storage.shape[0])
            self.storage = self.storage[perm]
            self.input_storage = self.input_storage[perm]  # Keep aligned

        batch_acts = self.storage[self.ptr : self.ptr + batch_size]
        batch_inputs = self.input_storage[self.ptr : self.ptr + batch_size]
        self.ptr += batch_size
        return batch_acts, batch_inputs

    def close(self):
        self.hook_handle.remove()


def resample_dead_neurons(sae, batch_acts, feature_acts):
    with torch.no_grad():
        dead_indices = (~(feature_acts > 0).any(dim=0)).nonzero(as_tuple=True)[0]
        if len(dead_indices) == 0:
            return 0

        recons = sae.decode(feature_acts)
        residuals = batch_acts - recons
        errors = residuals.pow(2).sum(dim=1)
        probs = errors / errors.sum()

        new_idx = torch.multinomial(
            probs, num_samples=len(dead_indices), replacement=True
        )
        new_dirs = F.normalize(residuals[new_idx], dim=1)

        sae.W_dec.data[dead_indices] = new_dirs
        sae.W_enc.data[:, dead_indices] = new_dirs.T * 0.2
        sae.b_enc.data[dead_indices] = 0.0

        # Zero grad to prevent momentum issues
        if sae.W_enc.grad is not None:
            sae.W_enc.grad[:, dead_indices] = 0
        return len(dead_indices)


def print_feature_context(feat_idx, max_val, input_vec, colmap):
    """Prints the game state when the feature fired hardest."""
    feat_names = colmap.feat_names

    # 1. Get Top 5 active raw features at this exact moment
    # We care about values that are distinct from 0 (or means)
    sorted_idxs = torch.argsort(input_vec.abs(), descending=True)

    context_str = []
    count = 0
    for idx in sorted_idxs:
        idx = idx.item()
        val = input_vec[idx].item()
        name = feat_names[idx]

        # Simple filter to ignore boring zeros in raw data
        if abs(val) < 0.01:
            continue

        context_str.append(f"{name}={val:.2f}")
        count += 1
        if count >= 4:
            break

    print(
        f"  Feature {feat_idx:<4} (Act: {max_val:.2f}) Context: {', '.join(context_str)}"
    )


# -----------------------------------------------------------------------------
# Main Analysis Loop
# -----------------------------------------------------------------------------


def run_layer_analysis(layer_idx, model, loader, colmap, device, config):
    print(f"\n{'='*60}")
    print(f"ANALYZING LAYER {layer_idx}")
    print(f"{'='*60}")

    buffer = ActivationBuffer(model, layer_idx, buffer_size=40000, device=device)
    buffer.fill_buffer(loader, colmap)

    embedding_dim = config.model.n_embd
    sae = SparseAutoencoder(input_dim=embedding_dim, expansion_factor=8).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=5e-4)

    steps = 5000
    warmup = 1000
    final_l1 = 5e-6

    print(f"  Training SAE (Steps: {steps})...")
    start_t = time.time()

    for step in range(steps):
        acts, _ = buffer.get_batch(4096)

        l1 = (
            0.0
            if step < warmup
            else final_l1 * min(1.0, (step - warmup) / (steps - warmup))
        )

        optimizer.zero_grad()
        loss, _, feats, mse = sae(acts, l1)
        loss.backward()
        optimizer.step()
        sae.norm_decoder_weights()

        resurrected = 0
        if step > warmup and step % 1000 == 0:
            resurrected = resample_dead_neurons(sae, acts, feats)

        if step % 500 == 0:
            l0 = (feats > 0.01).float().sum(dim=1).mean().item()
            print(
                f"    Step {step}: L1 {l1:.1e} | Loss {loss.item():.4f} | MSE {mse.item():.4f} | L0 {l0:.1f} | Resurrected {resurrected}"
            )

    print(f"  SAE Trained in {time.time() - start_t:.1f}s")

    # Interpretation Phase
    print("\n  --- Top 5 Active Features (State Snapshots) ---")

    with torch.no_grad():
        acts = buffer.storage
        inputs = buffer.input_storage

        feature_acts = sae.encode(acts)  # [N, Feats]

        # Find top 5 most active features
        max_acts, max_indices = feature_acts.max(dim=0)
        top_feats = torch.argsort(max_acts, descending=True)[:5]

        for feat_idx in top_feats:
            feat_idx = feat_idx.item()
            max_val = max_acts[feat_idx].item()
            batch_idx = max_indices[feat_idx].item()
            raw_state = inputs[batch_idx]
            print_feature_context(feat_idx, max_val, raw_state, colmap)

    buffer.close()


def main():
    init_config()
    config = get_config()
    device = _resolve_device()

    print("Loading Data...")
    ds = PreloadedWindowDataset(
        config.zarr.out_root,
        progress=False,
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
    model = GPT(config).to(device)

    # Patch to fix potential missing keys in simple GPT config
    if not hasattr(model, "norm_final"):
        # If your GPT uses a different name for the final norm, adjust here
        # The provided nano_gpt.py doesn't seem to have a self.norm_final,
        # it applies norm() in forward(). We might need to rely on blocks[-1] for now.
        pass

    ckpt_state = torch.load(
        find_latest_checkpoint(Path(config.train.out_dir)), map_location="cpu"
    )["model"]
    model_state = match_state_dict_keys(ckpt_state, model)
    model.load_state_dict(model_state)
    model.eval()

    # Scan Input, Middle, and Output
    # Note: Layer -1 is technically the last block.
    layers_to_scan = [0, config.model.n_layer // 2, config.model.n_layer - 1]

    for layer in layers_to_scan:
        run_layer_analysis(layer, model, loader, colmap, device, config)


if __name__ == "__main__":
    main()
