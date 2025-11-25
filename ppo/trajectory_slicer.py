"""Trajectory slicing for continuous PPO training.

This module handles converting continuous streams of experience into fixed-length
windows suitable for PPO training. Unlike episode-based training, we collect
fixed-length rollouts and bootstrap the final state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from ppo.inference_coordinator import InferenceCoordinator, StepRecord
from ppo.trajectory import Trajectory, Step


@dataclass
class RolloutSlice:
    """A fixed-length slice of experience for training.

    Contains steps from all workers collected during a rollout period,
    with bootstrap values for incomplete episodes.
    """

    # Steps organized by worker
    worker_steps: Dict[int, List[Step]]
    # Bootstrap values for the final state of each worker
    bootstrap_values: Dict[int, torch.Tensor]
    # Total frames collected
    total_frames: int


class TrajectorySlicer:
    """Manages trajectory collection and slicing for continuous PPO training.

    Instead of waiting for episodes to end, we:
    1. Collect a fixed number of frames (rollout_length)
    2. Bootstrap the value of the final state
    3. Compute advantages/returns on the slice
    4. Train on the slice
    5. Continue from where we left off (preserving context)
    """

    def __init__(
        self,
        coordinator: InferenceCoordinator,
        rollout_length: int = 5000,
        gamma: float = 0.995,
        gae_lambda: float = 0.95,
        normalize_advantages: bool = True,
    ):
        """Initialize the trajectory slicer.

        Args:
            coordinator: The inference coordinator managing workers
            rollout_length: Number of frames to collect per rollout
            gamma: Discount factor for rewards
            gae_lambda: Lambda for GAE computation
            normalize_advantages: Whether to normalize advantages
        """
        self.coordinator = coordinator
        self.rollout_length = rollout_length
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def _actions_to_primitives(self, actions: Dict[str, torch.Tensor]) -> Dict[str, any]:
        """Convert action tensors to Python primitives to avoid file descriptor leaks.

        PyTorch tensors use file descriptors when sent through multiprocessing queues,
        which can cause "too many open files" errors. Converting to primitives fixes this.
        """
        primitives = {}
        for key, value in actions.items():
            if key == "buttons":
                # Convert boolean tensor to list of bools
                primitives[key] = value.cpu().tolist()
            else:
                # Convert scalar tensors to Python int
                primitives[key] = int(value.cpu().item())
        return primitives

    def collect_rollout(
        self,
        state_queue,
        action_queues: Dict[int, object],
    ) -> RolloutSlice:
        """Collect a fixed-length rollout from all workers.

        This method runs the simulation loop for rollout_length frames,
        collecting experience from all workers.

        Args:
            state_queue: Queue receiving states from workers
            action_queues: Dict mapping worker_id to action queues

        Returns:
            RolloutSlice containing collected experience
        """
        frames_collected = 0
        num_workers = self.coordinator.num_workers
        start_time = time.time()
        last_print_time = start_time
        last_print_frames = 0

        # Accumulate timing stats
        timing_sums = {
            'buffer_update': 0.0,
            'batch_prep': 0.0,
            'input_building': 0.0,
            'learner_forward': 0.0,
            'learner_sampling': 0.0,
            'opponent_batch_prep': 0.0,
            'opponent_forward': 0.0,
            'opponent_sampling': 0.0,
            'recording': 0.0,
            'total': 0.0,
        }
        timing_counts = 0

        print(f"Collecting rollout of {self.rollout_length} frames...")

        while frames_collected < self.rollout_length:
            batch_start = time.time()

            # Collect states from all workers
            collect_start = time.time()
            worker_states = []
            for _ in range(num_workers):
                try:
                    worker_id, features_np, reward, done = state_queue.get(timeout=30.0)
                    # Convert numpy array back to tensor (much faster than from list)
                    features = torch.from_numpy(features_np)
                    worker_states.append((worker_id, features, reward, done))
                except Exception as e:
                    print(f"Warning: Timeout waiting for worker state: {e}")
                    continue
            collect_time = time.time() - collect_start

            if not worker_states:
                continue

            # Process batch and get actions
            actions, timings = self.coordinator.process_states(worker_states)

            # Accumulate timing stats
            timing_sums['buffer_update'] += timings.buffer_update
            timing_sums['batch_prep'] += timings.batch_prep
            timing_sums['input_building'] += timings.input_building
            timing_sums['learner_forward'] += timings.learner_forward
            timing_sums['learner_sampling'] += timings.learner_sampling
            timing_sums['opponent_batch_prep'] += timings.opponent_batch_prep
            timing_sums['opponent_forward'] += timings.opponent_forward
            timing_sums['opponent_sampling'] += timings.opponent_sampling
            timing_sums['recording'] += timings.recording
            timing_sums['total'] += timings.total
            timing_counts += 1

            # Send actions to workers (convert to primitives to avoid file descriptor leaks)
            send_start = time.time()
            conversion_time = 0.0
            queue_put_time = 0.0

            for worker_id, (p1_actions, p2_actions) in actions.items():
                if worker_id in action_queues:
                    # Convert tensors to Python primitives to avoid file descriptor leaks
                    conv_start = time.time()
                    p1_primitives = self._actions_to_primitives(p1_actions)
                    p2_primitives = self._actions_to_primitives(p2_actions)
                    conversion_time += time.time() - conv_start

                    # Queue put
                    put_start = time.time()
                    action_queues[worker_id].put((p1_primitives, p2_primitives))
                    queue_put_time += time.time() - put_start
            send_time = time.time() - send_start

            batch_time = time.time() - batch_start
            frames_collected += len(worker_states)

            # Calculate unaccounted time
            accounted_time = collect_time + timings.total + send_time
            overhead_time = batch_time - accounted_time

            # Print detailed timing breakdown every 50 batches
            batch_count = frames_collected // len(worker_states)
            if batch_count % 50 == 0:
                # Compute averages
                avg_timings = {k: v / timing_counts * 1000 for k, v in timing_sums.items()}

                # Calculate GPU vs CPU time
                gpu_time = avg_timings['learner_forward'] + avg_timings['opponent_forward']
                cpu_time = (avg_timings['buffer_update'] + avg_timings['batch_prep'] +
                           avg_timings['input_building'] + avg_timings['learner_sampling'] +
                           avg_timings['opponent_batch_prep'] + avg_timings['opponent_sampling'] +
                           avg_timings['recording'])

                print(
                    f"  [TIMING] Batch: {batch_time*1000:.1f}ms | "
                    f"Collect: {collect_time*1000:.1f}ms | "
                    f"Inference: {timings.total*1000:.1f}ms | "
                    f"Send: {send_time*1000:.1f}ms | "
                    f"Overhead: {overhead_time*1000:.1f}ms"
                )
                print(
                    f"    Send breakdown: Conversion={conversion_time*1000:.1f}ms | "
                    f"QueuePut={queue_put_time*1000:.1f}ms"
                )
                print(
                    f"    Current Inference: GPU={timings.learner_forward*1000 + timings.opponent_forward*1000:.1f}ms | "
                    f"CPU={timings.buffer_update*1000 + timings.batch_prep*1000 + timings.input_building*1000 + timings.learner_sampling*1000 + timings.opponent_sampling*1000 + timings.recording*1000:.1f}ms"
                )
                print(
                    f"      CPU: Buffer={timings.buffer_update*1000:.2f}ms | "
                    f"BatchPrep={timings.batch_prep*1000:.2f}ms | "
                    f"InputBuild={timings.input_building*1000:.2f}ms | "
                    f"LearnerSample={timings.learner_sampling*1000:.2f}ms | "
                    f"OpponentSample={timings.opponent_sampling*1000:.2f}ms | "
                    f"Record={timings.recording*1000:.2f}ms"
                )
                print(
                    f"      GPU: Learner={timings.learner_forward*1000:.2f}ms | "
                    f"Opponent={timings.opponent_forward*1000:.2f}ms"
                )

            # Print progress with FPS every 500 frames or at completion
            if frames_collected % 500 == 0 or frames_collected == self.rollout_length:
                current_time = time.time()
                elapsed = current_time - last_print_time
                frames_since_last = frames_collected - last_print_frames

                if elapsed > 0:
                    current_fps = frames_since_last / elapsed
                    overall_fps = frames_collected / (current_time - start_time)
                    print(
                        f"  Collected {frames_collected}/{self.rollout_length} frames | "
                        f"FPS: {current_fps:.1f} (avg: {overall_fps:.1f})"
                    )
                else:
                    print(f"  Collected {frames_collected}/{self.rollout_length} frames")

                last_print_time = current_time
                last_print_frames = frames_collected

        # Get bootstrap values for all workers
        bootstrap_values = self.coordinator.bootstrap_values()

        # Convert step records to Step objects
        worker_steps = {}
        for worker_id, records in self.coordinator.get_all_step_records().items():
            worker_steps[worker_id] = [
                Step(
                    state=r.state,
                    action_taken=r.action_taken,
                    log_prob=r.log_prob,
                    value=r.value,
                    reward=r.reward,
                    done=False,  # We don't track done in continuous mode
                )
                for r in records
            ]

        # Print rollout summary with timing stats
        total_time = time.time() - start_time
        avg_fps = frames_collected / total_time if total_time > 0 else 0
        total_steps = sum(len(steps) for steps in worker_steps.values())
        print(
            f"\nRollout complete: {total_steps} steps from {num_workers} workers | "
            f"Time: {total_time:.1f}s | Avg FPS: {avg_fps:.1f}"
        )

        # Print final timing summary
        if timing_counts > 0:
            avg_timings = {k: v / timing_counts * 1000 for k, v in timing_sums.items()}
            gpu_time = avg_timings['learner_forward'] + avg_timings['opponent_forward']
            cpu_time = (avg_timings['buffer_update'] + avg_timings['batch_prep'] +
                       avg_timings['input_building'] + avg_timings['learner_sampling'] +
                       avg_timings['opponent_batch_prep'] + avg_timings['opponent_sampling'] +
                       avg_timings['recording'])

            print("\n=== Inference Performance Summary ===")
            print(f"Average inference time: {avg_timings['total']:.2f}ms")
            print(f"  GPU time: {gpu_time:.2f}ms ({gpu_time/avg_timings['total']*100:.1f}%)")
            print(f"  CPU time: {cpu_time:.2f}ms ({cpu_time/avg_timings['total']*100:.1f}%)")
            print(f"\nGPU breakdown:")
            print(f"  Learner forward:  {avg_timings['learner_forward']:.2f}ms")
            print(f"  Opponent forward: {avg_timings['opponent_forward']:.2f}ms")
            print(f"\nCPU breakdown:")
            print(f"  Buffer update:    {avg_timings['buffer_update']:.2f}ms")
            print(f"  Batch prep:       {avg_timings['batch_prep']:.2f}ms")
            print(f"  Input building:   {avg_timings['input_building']:.2f}ms")
            print(f"  Learner sampling: {avg_timings['learner_sampling']:.2f}ms")
            print(f"  Opponent sampling:{avg_timings['opponent_sampling']:.2f}ms")
            print(f"  Recording:        {avg_timings['recording']:.2f}ms")
            print("=" * 37)

        return RolloutSlice(
            worker_steps=worker_steps,
            bootstrap_values=bootstrap_values,
            total_frames=frames_collected,
        )

    def prepare_training_data(
        self,
        rollout: RolloutSlice,
        seq_len: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Prepare training data from a rollout slice.

        Converts the rollout into overlapping sequence windows with
        computed advantages and returns.

        Args:
            rollout: The collected rollout slice
            seq_len: Sequence length for transformer
            device: Device to place tensors on

        Returns:
            Dictionary of batched tensors ready for training
        """
        # Convert to trajectories with GAE computed
        trajectories = []

        for worker_id, steps in rollout.worker_steps.items():
            if len(steps) == 0:
                continue

            # Create trajectory
            traj = Trajectory(steps=steps)

            # Compute returns with bootstrap value
            bootstrap_val = rollout.bootstrap_values.get(worker_id, torch.tensor(0.0))
            self._compute_gae_with_bootstrap(traj, bootstrap_val)

            trajectories.append(traj)

        if len(trajectories) == 0:
            return {}

        # Build sequence windows
        stride = max(1, seq_len // 4)  # 75% overlap
        windows = self._build_sequence_windows(trajectories, seq_len, stride, device)

        return windows

    def _compute_gae_with_bootstrap(
        self,
        trajectory: Trajectory,
        bootstrap_value: torch.Tensor,
    ) -> None:
        """Compute GAE for a trajectory with bootstrap value for final state.

        Args:
            trajectory: The trajectory to compute GAE for
            bootstrap_value: Value estimate for the state after the last step
        """
        T = len(trajectory.steps)
        if T == 0:
            trajectory.advantages = torch.tensor([])
            trajectory.returns = torch.tensor([])
            return

        # Extract values and rewards
        values = torch.stack([step.value for step in trajectory.steps]).squeeze(-1)
        rewards = torch.tensor(
            [step.reward for step in trajectory.steps], dtype=values.dtype
        )

        # Append bootstrap value for computing TD errors
        next_values = torch.cat([values[1:], bootstrap_value.unsqueeze(0)])

        # TD errors
        deltas = rewards + self.gamma * next_values - values

        # GAE computation (backward pass)
        advantages = torch.zeros(T, dtype=values.dtype)
        gae = 0.0
        for t in reversed(range(T)):
            gae = deltas[t] + self.gamma * self.gae_lambda * gae
            advantages[t] = gae

        # Normalize advantages
        if self.normalize_advantages and T > 1:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - adv_mean) / (adv_std + 1e-8)
            else:
                advantages = advantages - adv_mean

        trajectory.advantages = advantages
        trajectory.returns = advantages + values

    def _build_sequence_windows(
        self,
        trajectories: List[Trajectory],
        seq_len: int,
        stride: int,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Build overlapping sequence windows from trajectories.

        Similar to build_sequence_windows in train_ppo.py but works with
        our trajectory format.
        """
        all_windows = {
            "states": [],
            "advantages": [],
            "returns": [],
            "old_log_probs": [],
            "values": [],
            "valid_mask": [],
        }

        # Get action heads from first trajectory
        action_keys = []
        for traj in trajectories:
            if len(traj.steps) > 0:
                for head in ["main_stick", "c_stick", "buttons", "shoulder"]:
                    if head in traj.steps[0].action_taken:
                        action_keys.append(head)
                break

        for head in action_keys:
            all_windows[f"actions_{head}"] = []

        for traj in trajectories:
            if traj.advantages is None or traj.returns is None:
                continue

            T = len(traj.steps)
            if T == 0:
                continue

            # Convert to tensors
            states = torch.stack([step.state for step in traj.steps])
            advantages = traj.advantages
            returns = traj.returns
            old_log_probs = torch.stack([step.log_prob for step in traj.steps]).squeeze(
                -1
            )
            values = torch.stack([step.value for step in traj.steps]).squeeze(-1)

            actions = {}
            for head in action_keys:
                actions[head] = torch.stack(
                    [step.action_taken[head] for step in traj.steps]
                )

            # Handle short trajectories
            if T < seq_len:
                pad_len = seq_len - T
                states = torch.cat([states[0:1].expand(pad_len, -1), states], dim=0)
                advantages = torch.cat([torch.zeros(pad_len), advantages], dim=0)
                returns = torch.cat([returns[0:1].expand(pad_len), returns], dim=0)
                old_log_probs = torch.cat(
                    [old_log_probs[0:1].expand(pad_len), old_log_probs], dim=0
                )
                values = torch.cat([values[0:1].expand(pad_len), values], dim=0)

                for head in action_keys:
                    if actions[head].dim() == 1:
                        actions[head] = torch.cat(
                            [actions[head][0:1].expand(pad_len), actions[head]], dim=0
                        )
                    else:
                        actions[head] = torch.cat(
                            [actions[head][0:1].expand(pad_len, -1), actions[head]],
                            dim=0,
                        )

                valid_mask = torch.cat(
                    [
                        torch.zeros(pad_len, dtype=torch.bool),
                        torch.ones(T, dtype=torch.bool),
                    ]
                )

                all_windows["states"].append(states.unsqueeze(0))
                all_windows["advantages"].append(advantages.unsqueeze(0))
                all_windows["returns"].append(returns.unsqueeze(0))
                all_windows["old_log_probs"].append(old_log_probs.unsqueeze(0))
                all_windows["values"].append(values.unsqueeze(0))
                all_windows["valid_mask"].append(valid_mask.unsqueeze(0))

                for head in action_keys:
                    all_windows[f"actions_{head}"].append(actions[head].unsqueeze(0))
            else:
                # Create sliding windows
                for start in range(0, T - seq_len + 1, stride):
                    end = start + seq_len
                    all_windows["states"].append(states[start:end].unsqueeze(0))
                    all_windows["advantages"].append(advantages[start:end].unsqueeze(0))
                    all_windows["returns"].append(returns[start:end].unsqueeze(0))
                    all_windows["old_log_probs"].append(
                        old_log_probs[start:end].unsqueeze(0)
                    )
                    all_windows["values"].append(values[start:end].unsqueeze(0))
                    all_windows["valid_mask"].append(
                        torch.ones(seq_len, dtype=torch.bool).unsqueeze(0)
                    )

                    for head in action_keys:
                        all_windows[f"actions_{head}"].append(
                            actions[head][start:end].unsqueeze(0)
                        )

                # Final window
                if (T - seq_len) % stride != 0:
                    start = T - seq_len
                    all_windows["states"].append(states[start:].unsqueeze(0))
                    all_windows["advantages"].append(advantages[start:].unsqueeze(0))
                    all_windows["returns"].append(returns[start:].unsqueeze(0))
                    all_windows["old_log_probs"].append(
                        old_log_probs[start:].unsqueeze(0)
                    )
                    all_windows["values"].append(values[start:].unsqueeze(0))
                    all_windows["valid_mask"].append(
                        torch.ones(seq_len, dtype=torch.bool).unsqueeze(0)
                    )

                    for head in action_keys:
                        all_windows[f"actions_{head}"].append(
                            actions[head][start:].unsqueeze(0)
                        )

        # Concatenate and move to device
        result = {}
        for key, tensors in all_windows.items():
            if len(tensors) > 0:
                result[key] = torch.cat(tensors, dim=0).to(device)

        return result

    def clear_records(self) -> None:
        """Clear recorded steps from coordinator after training."""
        self.coordinator.clear_step_records()
