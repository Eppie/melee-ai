# PPO Data Movement (Visual Guide)

This page shows how data moves through the PPO system—emulators, workers, queues, GPU inference, targets, returns/GAE, and losses. ASCII maps lead; short captions link to the exact code that moves each piece.

Legend: `[CPU]` vs `[GPU]`, `-->` data push, `<--` data pull, `(queue)` multiprocessing queue, `(torch)` tensor, `(np)` numpy, `(Td)` `TensorDict`.

## Distributed PPO (train_ppo.py --distributed)

```
              [CPU] SimulationWorker_i (ppo/simulation_worker.py:run_worker)
        ┌──────────────────────────────────────────────────────────────────────┐
        │ Dolphin emulator                                                    │
        │  └─ libmelee GameState (gamestate)                                  │
        │      └─ schema.extract_row → feature dict (schema.py:extract_row)   │
        │          └─ _frame_to_tensor → torch.FloatTensor[F] on CPU          │
        │              │ reward: compute_frame_rewards(prev,curr)             │
        │              │   (train/value_head.py:compute_frame_rewards)        │
        │              └─ state_queue.put((worker_id, features_np, reward,    │
        │                                   done))                            │
        └──────────────────────────────────────────────────────────────────────┘
                                  │  (mp.Queue, CPU→CPU copy of np array)
                                  ▼
[CPU main proc] TrajectorySlicer.collect_rollout (ppo/trajectory_slicer.py)
                                  │  batches worker tuples
                                  ▼
                      InferenceCoordinator.process_states
                      (ppo/inference_coordinator.py)
            ┌────────────────────────────────────────────────────────────┐
            │ update per-worker buffers (deque)                          │
            │   learner_buffer/opponent_buffer (Tensor[F])               │
            │ warmup check → stack seq_len frames → build_model_inputs   │
            │   (train.batch_utils.build_model_inputs)                   │
            └────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                        [GPU] Batched inference
             ┌─────────────────────────────────────────────────────────┐
             │ learner_model (GPT) forward (model/nano_gpt.py)         │
             │ opponent_model forward (if loaded)                      │
             │ outputs Td keys: main_stick, c_stick, buttons, shoulder │
             │          + value (critic)                               │
             │ sample actions                                          │
             │   _sample_actions_batch → compute_log_probs             │
             │   (ppo/ppo_loss.py:compute_log_probs)                   │
             └─────────────────────────────────────────────────────────┘
                                  │
                                  ▼
                   actions for each worker (torch on GPU)
                                  │  (convert to Python primitives)
                                  ▼
           action_queues[worker_id].put((p1_actions, p2_actions))
                                  │
                                  ▼
   [CPU] SimulationWorker_i._apply_actions → ControllerState → Dolphin
                                  │
                                  └─ loop continues per frame

After each processed frame: StepRecord saved (ppo/inference_coordinator.py:StepRecord)
  state (cpu Tensor[F]), action_logits, action_taken, log_prob, value, reward placeholder.

Rollout boundary:
  - TrajectorySlicer.collect_rollout stops at rollout_length frames.
  - bootstrap_values = coordinator.bootstrap_values() (critic V(s_T) on GPU).
  - StepRecords grouped per worker → Trajectory objects (ppo/trajectory.py).

```

### Converting Rollout → Training Windows

```
RolloutSlice (ppo/trajectory_slicer.py:RolloutSlice)
   ├─ worker_steps[worker_id] -> Trajectory (ppo/trajectory.py)
   └─ bootstrap_values[worker_id] (Tensor[1])

TrajectorySlicer._compute_gae_with_bootstrap
   advantages[t] = δ_t + γλ * advantages[t+1]
   returns[t]     = advantages[t] + values[t]
   (deltas use bootstrap for last step)              [CPU]

_build_sequence_windows
   - Stack per-step tensors: states, action_logits_*, actions_*, log_prob, value
   - Pad short seq_len with zeros + valid_mask
   - Slide windows stride=seq_len//4 (75% overlap)
   - Move to training device                              [GPU or CPU chosen in train_ppo.py]

train_ppo.py:train_on_windows
   mb_states -> build_model_inputs -> model forward (GPU)
   new_action_logits/head, new_values
   loss_mask zeroes warmup positions
   compute_total_ppo_loss (ppo/ppo_loss.py)
      policy: compute_ppo_loss → ratios, clip
      entropy: compute_entropy
      value: compute_value_loss (MSE or clipped)
   optimizer.step (AdamW) + GradScaler (torch.amp)        [GPU]

Metrics logged via WandbLogger (train/wandb_utils.py). Checkpoint saved with state_dict, optimizer, episode/rollout.
```

### CPU/GPU Boundaries (Distributed)

```
[CPU] Dolphin + feature extraction
    └─ state_queue (np) ─────────────┐
                                     ▼
[GPU] InferenceCoordinator + TrajectorySlicer + model.forward
                                     │
                                     └─ action_queue (py primitives) ─► [CPU] Controller apply

[GPU] Training step (train_on_windows) uses same learner_model parameters.
OpponentPool swaps opponent checkpoints on disk (ppo/opponent_pool.py) -> coordinator.load_opponent loads to GPU.
```

## Episode-Based PPO (train_ppo.py default)

```
[CPU] SelfPlayEnvironment (ppo/selfplay_env.py)
   Dolphin console + controllers
   collect_raw_inputs_from_gamestate (model_interface.py) → _frame_to_tensor
   buffers: learner_buffer/opponent_buffer (deque Tensor[F])
   warmup_frames gate before policy is used
        │
        ▼
[GPU/CPU] learner_model forward (build_model_inputs)        (device from train_ppo.py)
   _sample_action (stochastic for learner, deterministic opponent)
   log_prob via compute_log_probs
   value head -> step.value
        │
        ▼
[CPU] TrajectoryBuffer.add_step stores cpu tensors
        │
        ▼
episode end → _finalize_episode_trajectory
   returns = compute_value_targets(states, colmap, gamma, reward_features)
   rewards recovered from returns (difference)
   TrajectoryBuffer.finish_trajectory(returns=returns)
```

Training:
- train_ppo.py:train_on_trajectories loops trajectories.
- Trajectory.compute_gae (ppo/trajectory.py) without bootstrap (episode terminal V_{t+1}=0).
- build_sequence_windows (train_ppo.py) → same PPO loss path as distributed.

## PPO Tensor Shapes and Targets

- Features `states`: `[B, seq_len, F]`, `F=len(get_feature_names())` (schema.py). Includes `stage`, mirrored `p1_*`/`p2_*`, and `value_target` column (unused at runtime).
- Actions/logits per head:
  - `main_stick`: `[B, seq_len, |CONTROL_STICK_QUANTIZED|]`
  - `c_stick`: `[B, seq_len, |C_STICK_QUANTIZED|]`
  - `shoulder`: `[B, seq_len, |SHOULDER_QUANTIZED|]`
  - `buttons`: `[B, seq_len, 5]` Bernoulli heads (A, B, XY, Z, LR)
- Old log probs: `[B, seq_len]` from compute_log_probs at collection time.
- Values: `[B, seq_len]` critic predictions; returns: `[B, seq_len]` targets; advantages: `[B, seq_len]`.
- Loss mask: `[B, seq_len]` warmup zeros then ones (train_on_windows/train_on_trajectories).

## Where Rewards/GAE/Targets Live

- Reward per frame (distributed workers): `SimulationWorker._compute_reward` → `compute_frame_rewards` (`train/value_head.py`).
- Episode returns (non-distributed): `SelfPlayEnvironment._compute_episode_returns` → `compute_value_targets`.
- GAE:
  - `Trajectory.compute_gae` (episode mode) uses TD deltas with terminal next_value=0.
  - `TrajectorySlicer._compute_gae_with_bootstrap` (distributed) appends bootstrap_value for final state.
- Advantage normalization toggled by `config.ppo.normalize_advantages` (config loaded via `config/config.py`).

## Loss Stack (ppo/ppo_loss.py)

```
compute_total_ppo_loss
   ├─ flatten & mask sequence positions
   ├─ policy: compute_ppo_loss
   │     ratio = exp(new_log_prob - old_log_prob)
   │     clip to [1-ε, 1+ε]; entropy bonus β*H
   ├─ value: compute_value_loss (MSE, optional clip to old_value ± value_clip)
   └─ total = policy + value_coef * value + entropy_bonus
outputs metrics: ratio stats, entropy, value error, clipped_fraction, etc.
```

## Checkpoints, Pool, Devices

- Checkpoints saved in `train_ppo.py:save_checkpoint` with model/optimizer/config/episode.
- OpponentPool (`ppo/opponent_pool.py`) keeps on-disk `opponent_XXXX.pt` snapshots; `add_opponent` called after episodes/rollouts; `load_opponent_model` loads into GPU for coordinator or CPU for SelfPlayEnvironment.
- GPU hot path: `torch.compile` model for inference in distributed mode; `torch.set_float32_matmul_precision('high')`; AMP via `torch.amp.autocast` + `GradScaler`.
- CPU paths: Dolphin emulation, reward diffs, queue marshaling, padding/slicing before device transfer.

## Quick File Pointers

- Collection (episode): `ppo/selfplay_env.py:step`, `TrajectoryBuffer`.
- Collection (distributed): `ppo/simulation_worker.py:_process_frame`, `ppo/inference_coordinator.py:process_states`, `ppo/trajectory_slicer.py:collect_rollout`.
- Rewards/returns: `train/value_head.py` (`compute_frame_rewards`, `compute_value_targets`).
- Advantages: `ppo/trajectory.py:compute_gae`, `ppo/trajectory_slicer.py:_compute_gae_with_bootstrap`.
- Windows: `train_ppo.py:build_sequence_windows`, `ppo/trajectory_slicer.py:_build_sequence_windows`.
- Loss: `ppo/ppo_loss.py:compute_total_ppo_loss`.
- Training loops: `train_ppo.py:train_on_windows` (distributed), `train_ppo.py:train_on_trajectories` (episode).
- Orchestration: `train_ppo.py:run_distributed_training` / `run_episode_based_training`.
