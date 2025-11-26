"""Centralized GPU inference coordinator for distributed PPO training.

This module implements a central inference server that handles batched GPU inference
for multiple parallel simulation workers. All model inference is concentrated here
to maximize GPU throughput.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tensordict import TensorDict

from column_map import ColumnMap
from config.config import get_config
from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)
from model.nano_gpt import GPT
from ppo.ppo_loss import compute_log_probs
from schema import get_feature_names, get_target_names
from train.batch_utils import build_model_inputs
from utils import match_state_dict_keys


@dataclass
class InferenceTimings:
    """Detailed timing breakdown for inference operations."""

    buffer_update: float = 0.0  # CPU: Updating buffers and preparing workers
    batch_prep: float = 0.0  # CPU: Stacking tensors and moving to device
    input_building: float = 0.0  # CPU: build_model_inputs transformation
    learner_forward: float = 0.0  # GPU: Learner model forward pass
    learner_sampling: float = 0.0  # CPU/GPU: Sampling actions from learner outputs
    opponent_batch_prep: float = 0.0  # CPU: Opponent batch preparation
    opponent_forward: float = 0.0  # GPU: Opponent model forward pass
    opponent_sampling: float = 0.0  # CPU/GPU: Sampling actions from opponent outputs
    recording: float = 0.0  # CPU: Recording steps and building results
    total: float = 0.0  # Total time for process_states call


@dataclass
class WorkerState:
    """State maintained for each simulation worker."""

    worker_id: int
    # Feature buffers for P1 (learner) and P2 (opponent) perspectives
    learner_buffer: deque = field(default_factory=lambda: deque(maxlen=256))
    opponent_buffer: deque = field(default_factory=lambda: deque(maxlen=256))
    # Track if this worker is past warmup
    frames_collected: int = 0


@dataclass
class StepRecord:
    """Single step record for trajectory collection."""

    worker_id: int
    state: torch.Tensor  # [F] raw features
    action_taken: Dict[str, torch.Tensor]  # sampled actions
    log_prob: torch.Tensor  # log probability of action
    value: torch.Tensor  # value estimate
    reward: float = 0.0


class InferenceCoordinator:
    """Centralized GPU inference coordinator for parallel PPO training.

    This class manages:
    - Per-worker feature buffers for sequence context
    - Batched inference for both learner and opponent models
    - Trajectory recording for PPO training
    - Opponent model loading and swapping
    """

    def __init__(
        self,
        learner_model: GPT,
        device: torch.device,
        num_workers: int,
        seq_len: int = 256,
        warmup_frames: int = 128,
    ):
        """Initialize the inference coordinator.

        Args:
            learner_model: The model being trained (handles P1 inference)
            device: GPU device for inference
            num_workers: Number of parallel simulation workers
            seq_len: Sequence length for transformer context
            warmup_frames: Frames to buffer before predictions
        """
        self.learner_model = learner_model
        self.device = device
        self.num_workers = num_workers
        self.seq_len = seq_len
        self.warmup_frames = warmup_frames

        # Feature configuration
        self.feature_names = get_feature_names()
        self.target_names = get_target_names()
        self.feature_dim = len(self.feature_names)
        self.colmap = ColumnMap(self.feature_names, self.target_names)

        # Precompute feature swap indices for P1/P2 perspective switching
        self.swap_indices = self._compute_swap_indices()

        # Per-worker state (feature buffers, frame counts)
        self.worker_states: Dict[int, WorkerState] = {}
        for worker_id in range(num_workers):
            self.worker_states[worker_id] = WorkerState(
                worker_id=worker_id,
                learner_buffer=deque(maxlen=seq_len),
                opponent_buffer=deque(maxlen=seq_len),
            )

        # Opponent model (single model on GPU, swapped as needed)
        self.opponent_model: Optional[GPT] = None

        # Trajectory recording - continuous stream per worker
        self.step_records: Dict[int, List[StepRecord]] = {
            i: [] for i in range(num_workers)
        }

        # Action palettes for decoding
        self._main_stick_palette = np.asarray(CONTROL_STICK_QUANTIZED, dtype=np.float32)
        self._c_stick_palette = np.asarray(C_STICK_QUANTIZED, dtype=np.float32)
        self._shoulder_centers = SHOULDER_QUANTIZED

    def load_opponent(self, checkpoint_path) -> None:
        """Load an opponent model from checkpoint.

        Args:
            checkpoint_path: Path to opponent checkpoint file
        """
        if self.opponent_model is None:
            config = get_config()
            self.opponent_model = GPT(config).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model_state = match_state_dict_keys(ckpt["model"], self.opponent_model)
        self.opponent_model.load_state_dict(model_state)
        self.opponent_model.eval()
        print(f"Loaded opponent model from {checkpoint_path}")

    def reset_worker(self, worker_id: int) -> None:
        """Reset a worker's state (buffers and frame count).

        Called when a match ends/restarts for that worker.

        Args:
            worker_id: ID of the worker to reset
        """
        state = self.worker_states[worker_id]
        state.learner_buffer.clear()
        state.opponent_buffer.clear()
        state.frames_collected = 0

    def _compute_swap_indices(self) -> torch.Tensor:
        """Precompute indices for swapping P1/P2 features.

        This is called once during initialization to avoid string operations
        in the hot loop. Returns a permutation tensor where applying it to
        features swaps P1 and P2 perspectives.

        Returns:
            [F] tensor of indices for feature swapping
        """
        indices = torch.arange(len(self.feature_names), dtype=torch.long)

        for idx, name in enumerate(self.feature_names):
            if name.startswith("p1_"):
                p2_name = "p2_" + name[3:]
                if p2_name in self.feature_names:
                    p2_idx = self.feature_names.index(p2_name)
                    # Swap indices for both p1 and p2 features
                    indices[idx] = p2_idx
                    indices[p2_idx] = idx

        return indices

    def _swap_player_features(self, features: torch.Tensor) -> torch.Tensor:
        """Swap P1 and P2 features to get opponent's perspective.

        Uses precomputed swap indices for O(1) operation instead of
        iterating through feature names.

        Args:
            features: [F] tensor of features from P1's perspective

        Returns:
            [F] tensor with P1/P2 swapped for P2's perspective
        """
        return features[self.swap_indices]

    def process_states(
        self,
        worker_states_batch: List[Tuple[int, torch.Tensor, float, bool]],
    ) -> Tuple[
        Dict[int, Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
        InferenceTimings,
    ]:
        """Process a batch of states from workers and return actions for both players.

        This is the main inference method called each frame. It:
        1. Updates feature buffers for each worker
        2. Batches all ready workers together
        3. Runs inference for learner (P1) and opponent (P2)
        4. Records steps for trajectory collection
        5. Returns actions for both players

        Args:
            worker_states_batch: List of (worker_id, features, reward, done) tuples
                - features: [F] tensor of raw features from P1's perspective
                - reward: Immediate reward for the previous step
                - done: Whether the match ended for this worker

        Returns:
            Tuple of (actions_dict, timings):
                - actions_dict: Dict mapping worker_id to (p1_actions, p2_actions) tuples
                  Actions are dicts with keys: main_stick, c_stick, buttons, shoulder
                - timings: InferenceTimings object with detailed timing breakdown
        """
        start_total = time.perf_counter()
        timings = InferenceTimings()

        # Update buffers and collect workers ready for inference
        buffer_start = time.perf_counter()
        ready_learner_inputs = []  # (worker_id, inputs_tensor)
        ready_opponent_inputs = []  # (worker_id, inputs_tensor)
        worker_order = []  # Track which workers in which order

        for worker_id, features, reward, done in worker_states_batch:
            state = self.worker_states[worker_id]

            # Record reward for previous step
            if len(self.step_records[worker_id]) > 0:
                self.step_records[worker_id][-1].reward = reward

            # Handle episode boundaries
            if done:
                self.reset_worker(worker_id)
                continue

            # Add to buffers
            state.learner_buffer.append(features)
            opponent_features = self._swap_player_features(features)
            state.opponent_buffer.append(opponent_features)
            state.frames_collected += 1

            # Check if past warmup
            if state.frames_collected >= self.warmup_frames:
                # Build input tensors - always use exactly seq_len frames
                buffer_list = list(state.learner_buffer)
                if len(buffer_list) != self.seq_len:
                    raise RuntimeError(
                        f"Buffer length {len(buffer_list)} != seq_len {self.seq_len}. "
                        f"Ensure warmup_frames == seq_len for fixed input shapes."
                    )
                learner_frames = torch.stack(buffer_list, dim=0)
                opponent_frames = torch.stack(list(state.opponent_buffer), dim=0)

                ready_learner_inputs.append((worker_id, learner_frames))
                ready_opponent_inputs.append((worker_id, opponent_frames))
                worker_order.append(worker_id)

        timings.buffer_update = time.perf_counter() - buffer_start

        # If no workers ready, return empty actions
        if len(worker_order) == 0:
            timings.total = time.perf_counter() - start_total
            return (
                {
                    wid: (self._neutral_actions(), self._neutral_actions())
                    for wid, _, _, _ in worker_states_batch
                },
                timings,
            )

        # Batch inference for learner model - prepare batch
        batch_prep_start = time.perf_counter()
        learner_batch = torch.stack([x[1] for x in ready_learner_inputs], dim=0).to(
            self.device
        )  # [B, T, F]
        timings.batch_prep = time.perf_counter() - batch_prep_start

        # Build model inputs
        input_build_start = time.perf_counter()
        learner_inputs = build_model_inputs(learner_batch, self.colmap)
        timings.input_building = time.perf_counter() - input_build_start

        # Learner forward pass (GPU)
        learner_forward_start = time.perf_counter()
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
            learner_outputs = self.learner_model(learner_inputs)
        # Synchronize to get accurate GPU timing
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        timings.learner_forward = time.perf_counter() - learner_forward_start

        # Extract per-worker learner actions
        learner_sampling_start = time.perf_counter()
        learner_gpu_actions, learner_actions_cpu, learner_log_probs_cpu, learner_values_cpu = (
            self._sample_actions_batch(learner_outputs)
        )
        timings.learner_sampling = time.perf_counter() - learner_sampling_start

        # Batch inference for opponent model
        if self.opponent_model is not None:
            # Opponent batch preparation
            opp_batch_prep_start = time.perf_counter()
            opponent_batch = torch.stack(
                [x[1] for x in ready_opponent_inputs], dim=0
            ).to(self.device)
            opponent_inputs = build_model_inputs(opponent_batch, self.colmap)
            timings.opponent_batch_prep = time.perf_counter() - opp_batch_prep_start

            # Opponent forward pass (GPU)
            opp_forward_start = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
                opponent_outputs = self.opponent_model(opponent_inputs)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            timings.opponent_forward = time.perf_counter() - opp_forward_start

            # Opponent action sampling
            opp_sampling_start = time.perf_counter()
            opponent_gpu_actions, _, _, _ = self._sample_actions_batch(opponent_outputs)
            timings.opponent_sampling = time.perf_counter() - opp_sampling_start
        else:
            # Self-play: opponent uses same model
            opp_sampling_start = time.perf_counter()
            opponent_gpu_actions, _, _, _ = self._sample_actions_batch(learner_outputs)
            timings.opponent_sampling = time.perf_counter() - opp_sampling_start

        # **BATCH CPU TRANSFER** - Move all states to CPU at once
        recording_start = time.perf_counter()
        states_gpu = torch.stack([
            list(self.worker_states[wid].learner_buffer)[-1]
            for wid in worker_order
        ], dim=0)  # [B, F]
        states_cpu = states_gpu.cpu()  # Single batch transfer

        # Record steps and build result
        results = {}
        for batch_idx, worker_id in enumerate(worker_order):
            # Extract GPU actions for game control
            p1_actions_gpu, _ = learner_gpu_actions[batch_idx]
            p2_actions_gpu, _ = opponent_gpu_actions[batch_idx]

            # Record step using CPU data (already transferred)
            step = StepRecord(
                worker_id=worker_id,
                state=states_cpu[batch_idx],
                action_taken={k: v[batch_idx] for k, v in learner_actions_cpu.items()},
                log_prob=learner_log_probs_cpu[batch_idx],
                value=learner_values_cpu[batch_idx],
            )
            self.step_records[worker_id].append(step)

            results[worker_id] = (p1_actions_gpu, p2_actions_gpu)

        # Fill in neutral actions for workers not ready
        for worker_id, _, _, done in worker_states_batch:
            if worker_id not in results and not done:
                results[worker_id] = (self._neutral_actions(), self._neutral_actions())

        timings.recording = time.perf_counter() - recording_start
        timings.total = time.perf_counter() - start_total

        return results, timings

    def _sample_actions_batch(
        self,
        outputs: TensorDict,
    ) -> Tuple[
        List[Tuple[Dict, Dict]],  # GPU actions for game control
        Dict[str, torch.Tensor],  # CPU actions for recording
        torch.Tensor,  # CPU log_probs [B]
        torch.Tensor,  # CPU values [B]
    ]:
        """Sample actions from batched model outputs (vectorized).

        Uses the same sampling strategy as model_interface.py:
        - Sticks (main, c) and shoulder: argmax (deterministic)
        - Buttons: bernoulli sampling (stochastic)

        Args:
            outputs: Model output TensorDict with shape [B, T, ...]

        Returns:
            Tuple of:
            - gpu_actions: List of (p1_actions_gpu, action_logits_gpu) per batch element (for game control)
            - actions_cpu: Dict of action tensors on CPU with shape [B, ...]
            - log_probs_cpu: Log probabilities on CPU [B]
            - values_cpu: Value estimates on CPU [B]
        """
        batch_size = outputs["main_stick"].shape[0]

        # Extract logits at last timestep - [B, num_classes]
        main_logits = outputs["main_stick"][:, -1]
        c_logits = outputs["c_stick"][:, -1]
        button_logits = outputs["buttons"][:, -1]
        shoulder_logits = outputs["shoulder"][:, -1]
        values = outputs.get("value", torch.zeros(batch_size, 1, 1, device=self.device))
        values = values[:, -1, 0]  # [B]

        # Action logits for all heads
        action_logits_batch = {
            "main_stick": main_logits,
            "c_stick": c_logits,
            "buttons": button_logits,
            "shoulder": shoulder_logits,
        }

        # Sample actions - argmax for sticks/shoulder, bernoulli for buttons
        main_actions = torch.argmax(main_logits, dim=-1)  # [B]
        c_actions = torch.argmax(c_logits, dim=-1)  # [B]
        shoulder_actions = torch.argmax(shoulder_logits, dim=-1)  # [B]

        # Buttons use stochastic bernoulli sampling
        button_probs = torch.sigmoid(button_logits)  # [B, num_buttons]
        button_actions = torch.bernoulli(button_probs).bool()  # [B, num_buttons]

        actions_batch = {
            "main_stick": main_actions,
            "c_stick": c_actions,
            "shoulder": shoulder_actions,
            "buttons": button_actions,
        }

        # Compute log probabilities for entire batch at once
        log_probs = compute_log_probs(action_logits_batch, actions_batch)  # [B]

        # **BATCH CPU TRANSFER** - Move all data to CPU at once (1 sync per tensor)
        actions_cpu = {k: v.cpu() for k, v in actions_batch.items()}
        log_probs_cpu = log_probs.cpu()
        values_cpu = values.cpu()

        # Build GPU results list for game control (still on GPU for immediate use)
        gpu_actions = []
        for b in range(batch_size):
            action_logits = {k: v[b] for k, v in action_logits_batch.items()}
            actions = {k: v[b] for k, v in actions_batch.items()}
            gpu_actions.append((actions, action_logits))

        return gpu_actions, actions_cpu, log_probs_cpu, values_cpu

    def _compute_log_prob(
        self,
        action_logits: Dict[str, torch.Tensor],
        actions_taken: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute log probability of actions."""
        logits_batched = {k: v.unsqueeze(0) for k, v in action_logits.items()}
        actions_batched = {k: v.unsqueeze(0) for k, v in actions_taken.items()}
        log_prob = compute_log_probs(logits_batched, actions_batched)
        return log_prob.squeeze(0)

    def _neutral_actions(self) -> Dict[str, torch.Tensor]:
        """Return neutral controller actions (for warmup period)."""
        return {
            "main_stick": torch.tensor(0, device=self.device),  # Neutral
            "c_stick": torch.tensor(0, device=self.device),
            "buttons": torch.zeros(5, dtype=torch.bool, device=self.device),
            "shoulder": torch.tensor(0, device=self.device),
        }

    def get_step_records(self, worker_id: int) -> List[StepRecord]:
        """Get recorded steps for a worker."""
        return self.step_records[worker_id]

    def get_all_step_records(self) -> Dict[int, List[StepRecord]]:
        """Get all recorded steps from all workers."""
        return self.step_records

    def clear_step_records(self) -> None:
        """Clear all recorded steps (after training)."""
        for worker_id in self.step_records:
            self.step_records[worker_id] = []

    def get_total_steps(self) -> int:
        """Get total number of recorded steps across all workers."""
        return sum(len(records) for records in self.step_records.values())

    def bootstrap_values(self) -> Dict[int, torch.Tensor]:
        """Compute bootstrap values for the current state of all workers.

        Called at the end of a rollout to estimate V(s_T) for incomplete episodes.

        Returns:
            Dict mapping worker_id to bootstrap value tensor
        """
        bootstrap_values = {}

        # Collect workers with data in their buffers
        ready_workers = []
        ready_inputs = []

        for worker_id, state in self.worker_states.items():
            if state.frames_collected >= self.warmup_frames:
                buffer_list = list(state.learner_buffer)
                assert len(buffer_list) == self.seq_len, \
                    f"Buffer length {len(buffer_list)} != seq_len {self.seq_len}"
                frames = torch.stack(buffer_list, dim=0)
                ready_inputs.append(frames)
                ready_workers.append(worker_id)

        if len(ready_workers) == 0:
            return {wid: torch.tensor(0.0) for wid in self.worker_states}

        # Batch inference
        batch = torch.stack(ready_inputs, dim=0).to(self.device)
        inputs = build_model_inputs(batch, self.colmap)

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
            outputs = self.learner_model(inputs)

        values = outputs.get("value", torch.zeros(len(ready_workers), 1, 1))
        values = values[:, -1, 0].cpu()  # [B]

        for idx, worker_id in enumerate(ready_workers):
            bootstrap_values[worker_id] = values[idx]

        # Zero for workers not ready
        for worker_id in self.worker_states:
            if worker_id not in bootstrap_values:
                bootstrap_values[worker_id] = torch.tensor(0.0)

        return bootstrap_values
