"""Centralized GPU inference coordinator for distributed PPO training.

This module implements a central inference server that handles batched GPU inference
for multiple parallel simulation workers. All model inference is concentrated here
to maximize GPU throughput.
"""

from __future__ import annotations

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
    action_logits: Dict[str, torch.Tensor]  # logits for each head
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
        self.opponent_model.load_state_dict(ckpt["model"])
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

    def _swap_player_features(self, features: torch.Tensor) -> torch.Tensor:
        """Swap P1 and P2 features to get opponent's perspective.

        Args:
            features: [F] tensor of features from P1's perspective

        Returns:
            [F] tensor with P1/P2 swapped for P2's perspective
        """
        swapped = features.clone()
        for idx, name in enumerate(self.feature_names):
            if name.startswith("p1_"):
                p2_name = "p2_" + name[3:]
                if p2_name in self.feature_names:
                    p2_idx = self.feature_names.index(p2_name)
                    swapped[idx] = features[p2_idx]
                    swapped[p2_idx] = features[idx]
        return swapped

    def process_states(
        self,
        worker_states_batch: List[Tuple[int, torch.Tensor, float, bool]],
    ) -> Dict[int, Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
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
            Dict mapping worker_id to (p1_actions, p2_actions) tuples
            Actions are dicts with keys: main_stick, c_stick, buttons, shoulder
        """
        # Update buffers and collect workers ready for inference
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
                # Build input tensors
                learner_frames = torch.stack(list(state.learner_buffer), dim=0)
                opponent_frames = torch.stack(list(state.opponent_buffer), dim=0)

                ready_learner_inputs.append((worker_id, learner_frames))
                ready_opponent_inputs.append((worker_id, opponent_frames))
                worker_order.append(worker_id)

        # If no workers ready, return empty actions
        if len(worker_order) == 0:
            return {wid: (self._neutral_actions(), self._neutral_actions())
                    for wid, _, _, _ in worker_states_batch}

        # Batch inference for learner model
        learner_batch = torch.stack(
            [x[1] for x in ready_learner_inputs], dim=0
        ).to(self.device)  # [B, T, F]
        learner_inputs = build_model_inputs(learner_batch, self.colmap)

        with torch.no_grad():
            learner_outputs = self.learner_model(learner_inputs)

        # Extract per-worker learner actions (stochastic)
        learner_actions_batch = self._sample_actions_batch(
            learner_outputs, exploration=True
        )

        # Batch inference for opponent model
        opponent_actions_batch = {}
        if self.opponent_model is not None:
            opponent_batch = torch.stack(
                [x[1] for x in ready_opponent_inputs], dim=0
            ).to(self.device)
            opponent_inputs = build_model_inputs(opponent_batch, self.colmap)

            with torch.no_grad():
                opponent_outputs = self.opponent_model(opponent_inputs)

            opponent_actions_batch = self._sample_actions_batch(
                opponent_outputs, exploration=False  # Deterministic for opponent
            )
        else:
            # Self-play: opponent uses same model but deterministic
            opponent_actions_batch = self._sample_actions_batch(
                learner_outputs, exploration=False
            )

        # Record steps and build result
        results = {}
        for batch_idx, worker_id in enumerate(worker_order):
            # Extract this worker's results
            p1_logits, p1_actions, p1_log_prob, p1_value = learner_actions_batch[batch_idx]
            p2_logits, p2_actions, _, _ = opponent_actions_batch[batch_idx]

            # Record step for trajectory
            state = self.worker_states[worker_id]
            # Use the last frame added to buffer as state
            step_state = list(state.learner_buffer)[-1]

            step = StepRecord(
                worker_id=worker_id,
                state=step_state.cpu(),
                action_logits={k: v.cpu() for k, v in p1_logits.items()},
                action_taken={k: v.cpu() for k, v in p1_actions.items()},
                log_prob=p1_log_prob.cpu(),
                value=p1_value.cpu(),
            )
            self.step_records[worker_id].append(step)

            results[worker_id] = (p1_actions, p2_actions)

        # Fill in neutral actions for workers not ready
        for worker_id, _, _, done in worker_states_batch:
            if worker_id not in results and not done:
                results[worker_id] = (self._neutral_actions(), self._neutral_actions())

        return results

    def _sample_actions_batch(
        self,
        outputs: TensorDict,
        exploration: bool = True,
    ) -> List[Tuple[Dict, Dict, torch.Tensor, torch.Tensor]]:
        """Sample actions from batched model outputs.

        Args:
            outputs: Model output TensorDict with shape [B, T, ...]
            exploration: If True, sample stochastically. If False, use argmax.

        Returns:
            List of (action_logits, actions_taken, log_prob, value) per batch element
        """
        batch_size = outputs["main_stick"].shape[0]
        results = []

        # Extract logits at last timestep
        main_logits = outputs["main_stick"][:, -1]  # [B, num_bins]
        c_logits = outputs["c_stick"][:, -1]
        button_logits = outputs["buttons"][:, -1]  # [B, num_buttons]
        shoulder_logits = outputs.get("shoulder")
        if shoulder_logits is not None:
            shoulder_logits = shoulder_logits[:, -1]
        values = outputs.get("value", torch.zeros(batch_size, 1, 1, device=self.device))
        values = values[:, -1, 0]  # [B]

        for b in range(batch_size):
            action_logits = {
                "main_stick": main_logits[b],
                "c_stick": c_logits[b],
                "buttons": button_logits[b],
            }
            if shoulder_logits is not None:
                action_logits["shoulder"] = shoulder_logits[b]

            actions = {}
            if exploration:
                # Sample from distributions
                main_probs = torch.softmax(main_logits[b], dim=-1)
                actions["main_stick"] = torch.multinomial(main_probs, 1).squeeze(-1)

                c_probs = torch.softmax(c_logits[b], dim=-1)
                actions["c_stick"] = torch.multinomial(c_probs, 1).squeeze(-1)

                if shoulder_logits is not None:
                    shoulder_probs = torch.softmax(shoulder_logits[b], dim=-1)
                    actions["shoulder"] = torch.multinomial(shoulder_probs, 1).squeeze(-1)
                else:
                    actions["shoulder"] = torch.tensor(0, device=self.device)

                button_probs = torch.sigmoid(button_logits[b])
                actions["buttons"] = torch.bernoulli(button_probs).bool()
            else:
                # Argmax (deterministic)
                actions["main_stick"] = torch.argmax(main_logits[b], dim=-1)
                actions["c_stick"] = torch.argmax(c_logits[b], dim=-1)
                if shoulder_logits is not None:
                    actions["shoulder"] = torch.argmax(shoulder_logits[b], dim=-1)
                else:
                    actions["shoulder"] = torch.tensor(0, device=self.device)
                actions["buttons"] = (torch.sigmoid(button_logits[b]) > 0.5).bool()

            # Compute log probability
            log_prob = self._compute_log_prob(action_logits, actions)

            results.append((action_logits, actions, log_prob, values[b]))

        return results

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
                frames = torch.stack(list(state.learner_buffer), dim=0)
                ready_inputs.append(frames)
                ready_workers.append(worker_id)

        if len(ready_workers) == 0:
            return {wid: torch.tensor(0.0) for wid in self.worker_states}

        # Batch inference
        batch = torch.stack(ready_inputs, dim=0).to(self.device)
        inputs = build_model_inputs(batch, self.colmap)

        with torch.no_grad():
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
