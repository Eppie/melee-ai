# Distributed PPO Architecture for Melee AI

**A Visual Guide to Data Flow and Processing**

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Worker Process Architecture](#worker-process-architecture)
3. [Main Process & Inference Coordinator](#main-process--inference-coordinator)
4. [Trajectory Collection & GAE](#trajectory-collection--gae)
5. [PPO Training Loop](#ppo-training-loop)
6. [Hardware Mapping](#hardware-mapping)

---

## System Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MAIN PROCESS (CPU + GPU)                          │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     InferenceCoordinator                               │ │
│  │  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐    │ │
│  │  │ Learner Model│         │Opponent Model│         │ Step Records │    │ │
│  │  │   (GPU/FP16) │         │   (GPU/FP16) │         │   (CPU/RAM)  │    │ │
│  │  └──────────────┘         └──────────────┘         └──────────────┘    │ │
│  │         ▲                         ▲                         ▲          │ │
│  │         │ batched                 │ batched                 │          │ │
│  │         │ inference               │ inference               │ record   │ │
│  │         │                         │                         │          │ │
│  └─────────┼─────────────────────────┼─────────────────────────┼──────────┘ │
│            │                         │                         │            │
│     ┌──────┴─────────────────────────┴─────────┐               │            │
│     │         state_queue (MP Queue)           │               │            │
│     │  [worker_id, features_np, reward, done]  │               │            │
│     └──────▲───────────────────────────────────┘               │            │
│            │                                                   │            │
│            │ numpy arrays                                      │            │
│            │ (CPU serialization)                               │            │
│            │                                                   │            │
│     ┌──────┴───────────────────────────────────────────────────┴──────────┐ │
│     │              action_queues[worker_id] (MP Queues)                   │ │
│     │         {main_stick: int, c_stick: int, buttons: list, ...}         │ │
│     │                     (Python primitives)                             │ │
│     └──────┬───────────────┬────────────────┬──────────────┬──────────────┘ │
└────────────┼───────────────┼────────────────┼──────────────┼────────────────┘
             │               │                │              │
             ▼               ▼                ▼              ▼
    ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
    │  Worker 0   │ │  Worker 1   │ │  Worker 2   │ │  Worker 7   │
    │  (Process)  │ │  (Process)  │ │  (Process)  │ │  (Process)  │
    │             │ │             │ │             │ │             │
    │  Dolphin    │ │  Dolphin    │ │  Dolphin    │ │  Dolphin    │
    │  Emulator   │ │  Emulator   │ │  Emulator   │ │  Emulator   │
    │  + libmelee │ │  + libmelee │ │  + libmelee │ │  + libmelee │
    └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
         CPU             CPU             CPU             CPU
```

**Key Files:**
- `train_ppo.py:95-280` - Main distributed training loop
- `ppo/inference_coordinator.py` - Centralized GPU inference
- `ppo/simulation_worker.py` - Worker process implementation
- `ppo/trajectory_slicer.py` - Rollout collection orchestration

---

## Worker Process Architecture

### Worker Process (SimulationWorker)

Each worker runs in a separate Python process spawned via `multiprocessing`.

```
┌───────────────────────────────────────────────────────────────────────┐
│                        Worker Process (CPU)                            │
│                   ppo/simulation_worker.py:45-90                       │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐ │
│  │                     Dolphin Emulator Instance                     │ │
│  │  ┌────────────┐  ┌────────────┐  ┌─────────────────────────────┐ │ │
│  │  │   P1 (AI)  │  │   P2 (AI)  │  │  Game State (60 FPS)        │ │ │
│  │  │ Controller │  │ Controller │  │  - Player positions         │ │ │
│  │  │            │  │            │  │  - Velocities, percents     │ │ │
│  │  │  Inputs ◄──┼──┼────────────┼──┤  - Action states            │ │ │
│  │  │            │  │            │  │  - Stage, stocks, frame #   │ │ │
│  │  └────────────┘  └────────────┘  └─────────────┬───────────────┘ │ │
│  └────────────────────────────────────────────────┼─────────────────┘ │
│                                                    │                   │
│                                    GameState object│                   │
│                                                    │                   │
│  ┌─────────────────────────────────────────────────┼─────────────────┐ │
│  │              Feature Extraction Pipeline        ▼                 │ │
│  │  simulation_worker.py:294-308                                     │ │
│  │                                                                   │ │
│  │  1. collect_raw_inputs_from_gamestate(gamestate, p1, p2)         │ │
│  │     → model_interface.py:220-350                                 │ │
│  │     → RawInputs{original: Dict, transformed: Dict}               │ │
│  │                                                                   │ │
│  │  2. _frame_to_tensor(raw_inputs.transformed)                     │ │
│  │     → simulation_worker.py:337-342                               │ │
│  │     → torch.Tensor([908 features], dtype=float32) CPU            │ │
│  │                                                                   │ │
│  │  3. features.numpy() → np.ndarray                                │ │
│  │     → simulation_worker.py:358                                   │ │
│  │     → Convert to NumPy for fast serialization                    │ │
│  └───────────────────────────────────┬───────────────────────────────┘ │
│                                      │ features_np                     │
│  ┌───────────────────────────────────┼───────────────────────────────┐ │
│  │           Reward Computation      ▼                               │ │
│  │  simulation_worker.py:344-358                                     │ │
│  │                                                                   │ │
│  │  compute_frame_rewards(                                          │ │
│  │      X=[prev_features, curr_features],  # [1, 2, 908]            │ │
│  │      idx=reward_feature_idx             # Feature column indices │ │
│  │  ) → train/value_head.py:129-150                                 │ │
│  │                                                                   │ │
│  │  Computes zero-sum rewards:                                      │ │
│  │    - Damage dealt: diff(opponent_percent) * cfg.reward_damage    │ │
│  │    - Stock taken: -diff(opponent_stock) * cfg.reward_stock       │ │
│  │    - Hitlag advantage: (opp_hitlag - opp_def_hitlag) * cfg...   │ │
│  │    - Shield penalty: low_shield_strength * cfg.reward_shield     │ │
│  │                                                                   │ │
│  │  → reward: float (scalar)                                        │ │
│  └───────────────────────────────────┬───────────────────────────────┘ │
│                                      │                                 │
│  ┌───────────────────────────────────┼───────────────────────────────┐ │
│  │         Send to Main Process      ▼                               │ │
│  │  simulation_worker.py:359                                         │ │
│  │                                                                   │ │
│  │  state_queue.put((worker_id, features_np, reward, done))         │ │
│  │                   │        │          │        │                  │ │
│  │                   │        │          │        └─ bool: episode   │ │
│  │                   │        │          │           ended?          │ │
│  │                   │        │          └────────── float: reward   │ │
│  │                   │        └───────────────────── np.ndarray[908] │ │
│  │                   └────────────────────────────── int: worker ID  │ │
│  └───────────────────────────────────┬───────────────────────────────┘ │
│                                      │                                 │
│                                      ▼                                 │
│                              [Sent to main process]                    │
│                                                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │    Receive Actions from Main Process                            │  │
│  │    simulation_worker.py:362                                     │  │
│  │                                                                 │  │
│  │    (p1_actions, p2_actions) = action_queue.get()               │  │
│  │                                                                 │  │
│  │    p1_actions = {                                              │  │
│  │        'main_stick': int (0-63, palette index)                 │  │
│  │        'c_stick': int (0-8, palette index)                     │  │
│  │        'buttons': [bool, bool, bool, bool, bool]  # A,B,XY,Z,LR│  │
│  │        'shoulder': int (0-4, analog level)                     │  │
│  │    }                                                           │  │
│  └────────────────────────┬────────────────────────────────────────┘  │
│                           │                                           │
│  ┌────────────────────────┼─────────────────────────────────────────┐ │
│  │   Convert to Controller State                                   │ │
│  │   simulation_worker.py:397-430                                  │ │
│  │                        ▼                                        │ │
│  │   _actions_to_controller_state(p1_actions)                     │ │
│  │                                                                 │ │
│  │   main_idx = p1_actions['main_stick']  # int                   │ │
│  │   main_xy = CONTROL_STICK_QUANTIZED[main_idx]  # [-1,1] coords │ │
│  │   main_xy = (main_xy * 0.5 + 0.5)  # Scale to [0,1] for libmelee│ │
│  │                                                                 │ │
│  │   → ControllerState(                                           │ │
│  │       main_stick_x=0.73, main_stick_y=0.92,                    │ │
│  │       c_stick_x=0.5, c_stick_y=0.5,                            │ │
│  │       shoulder_analog=0.42,                                    │ │
│  │       button_a=True, button_b=False, ...                       │ │
│  │     )                                                          │ │
│  └────────────────────────┬────────────────────────────────────────┘ │
│                           │                                           │
│  ┌────────────────────────┼─────────────────────────────────────────┐ │
│  │   Apply to Dolphin Controllers                                  │ │
│  │   simulation_worker.py:364-396                                  │ │
│  │                        ▼                                        │ │
│  │   _apply_actions(self.controllers[p1_port], p1_actions)        │ │
│  │   _apply_actions(self.controllers[p2_port], p2_actions)        │ │
│  │                                                                 │ │
│  │   controller.press_button(Button.BUTTON_A)  # if button_a      │ │
│  │   controller.tilt_analog(Button.BUTTON_MAIN, x, y)             │ │
│  │   controller.press_shoulder(Button.BUTTON_L, analog)           │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│              [Loop continues at 60+ FPS per worker]                   │
└────────────────────────────────────────────────────────────────────────┘
```

**Key Data Structures:**
```python
# Features: np.ndarray, shape=[908], dtype=float32
# Contains all game state info for one frame:
# - Player positions (p1_x, p1_y, p2_x, p2_y)
# - Velocities (p1_vel_x, p1_vel_y, p2_vel_x, p2_vel_y)
# - Percents (p1_percent, p2_percent)
# - Stocks (p1_stock, p2_stock)
# - Action states (one-hot encoded)
# - Character IDs (one-hot encoded)
# - Stage ID (one-hot encoded)
# - Controller states (both players)
# - ... total 908 features

# Reward: float, scalar
# Zero-sum: P1's reward = -P2's reward
# Computed from state transitions

# Actions: Dict[str, Union[int, List[bool]]]
# Python primitives for fast serialization
```

---

## Main Process & Inference Coordinator

### Inference Coordinator Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                  Main Process - InferenceCoordinator                      │
│                  ppo/inference_coordinator.py:58-450                      │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    Per-Worker State Tracking                       │  │
│  │  inference_coordinator.py:34-42, 98-103                            │  │
│  │                                                                    │  │
│  │  worker_states[0]: WorkerState(                                   │  │
│  │      worker_id=0,                                                 │  │
│  │      learner_buffer=deque(maxlen=256),  # Sliding window buffer  │  │
│  │      opponent_buffer=deque(maxlen=256), # P2 perspective          │  │
│  │      frames_collected=256               # Warmup counter          │  │
│  │  )                                                                │  │
│  │  ... [similar for workers 1-7]                                    │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │         Batch Collection from Workers (Sequential)                 │  │
│  │         inference_coordinator.py:184-220                           │  │
│  │                                                                    │  │
│  │  for each (worker_id, features_np, reward, done) in worker_batch: │  │
│  │                                                                    │  │
│  │      # Convert numpy back to tensor                               │  │
│  │      features = torch.from_numpy(features_np)  # [908]            │  │
│  │                                                                    │  │
│  │      # Update sliding window buffers                              │  │
│  │      state.learner_buffer.append(features)      # Add to tail    │  │
│  │      opponent_features = swap_player_features(features)           │  │
│  │      state.opponent_buffer.append(opponent_features)              │  │
│  │                                                                    │  │
│  │      # Check if ready for inference (past warmup)                 │  │
│  │      if state.frames_collected >= warmup_frames (256):            │  │
│  │          # Stack buffer into tensor                               │  │
│  │          learner_frames = torch.stack(                            │  │
│  │              list(state.learner_buffer),  # All 256 frames        │  │
│  │              dim=0                                                │  │
│  │          )  # → [256, 908] tensor                                 │  │
│  │                                                                    │  │
│  │          ready_for_inference.append((worker_id, learner_frames))  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │         Batched GPU Inference (Learner Model)                      │  │
│  │         inference_coordinator.py:235-242                           │  │
│  │                                                                    │  │
│  │  # Stack all workers into batch                                   │  │
│  │  learner_batch = torch.stack(                                     │  │
│  │      [frames for (wid, frames) in ready_inputs],                  │  │
│  │      dim=0                                                        │  │
│  │  ).to(device='cuda')                                              │  │
│  │  # → [8, 256, 908] (batch_size, seq_len, features)                │  │
│  │                                                                    │  │
│  │  # Build model inputs (add one-hot encodings)                     │  │
│  │  learner_inputs = build_model_inputs(learner_batch, colmap)       │  │
│  │  # → TensorDict with:                                             │  │
│  │  #    'gamestate': [8, 256, ~700] - continuous features           │  │
│  │  #    'stage': [8, 256, 6] - one-hot stage ID                     │  │
│  │  #    'ego_char': [8, 256, 26] - one-hot P1 character             │  │
│  │  #    'opp_char': [8, 256, 26] - one-hot P2 character             │  │
│  │  #    'ego_action': [8, 256, 384] - one-hot P1 action state       │  │
│  │  #    'opp_action': [8, 256, 384] - one-hot P2 action state       │  │
│  │                                                                    │  │
│  │  # GPU INFERENCE (FP16 + torch.compile)                           │  │
│  │  with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):  │  │
│  │      learner_outputs = learner_model(learner_inputs)              │  │
│  │  # → TensorDict:                                                  │  │
│  │  #    'main_stick': [8, 256, 64] - logits for 64 stick positions  │  │
│  │  #    'c_stick': [8, 256, 9] - logits for 9 c-stick positions     │  │
│  │  #    'buttons': [8, 256, 5] - logits for 5 button states         │  │
│  │  #    'shoulder': [8, 256, 5] - logits for 5 shoulder levels      │  │
│  │  #    'value': [8, 256, 1] - value estimates V(s)                 │  │
│  │                                                                    │  │
│  │  # Inference time: ~5-10ms for batch of 8 workers                 │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │         Action Sampling from Logits                                │  │
│  │         inference_coordinator.py:302-366                           │  │
│  │                                                                    │  │
│  │  For each worker in batch:                                        │  │
│  │      # Extract logits at last timestep (current frame)            │  │
│  │      main_logits = outputs['main_stick'][worker_idx, -1]  # [64]  │  │
│  │      c_logits = outputs['c_stick'][worker_idx, -1]        # [9]   │  │
│  │      button_logits = outputs['buttons'][worker_idx, -1]   # [5]   │  │
│  │      value = outputs['value'][worker_idx, -1, 0]          # scalar│  │
│  │                                                                    │  │
│  │      # STOCHASTIC SAMPLING (exploration for learner)              │  │
│  │      main_probs = softmax(main_logits)                            │  │
│  │      main_action = multinomial(main_probs, 1)  # Sample 1 action  │  │
│  │                                                                    │  │
│  │      button_probs = sigmoid(button_logits)                        │  │
│  │      button_actions = bernoulli(button_probs)  # Independent      │  │
│  │                                                                    │  │
│  │      # Compute log probability for PPO                            │  │
│  │      log_prob = (                                                 │  │
│  │          log(main_probs[main_action]) +                           │  │
│  │          log(c_probs[c_action]) +                                 │  │
│  │          sum(log_bernoulli_probs(button_actions, button_probs)) + │  │
│  │          log(shoulder_probs[shoulder_action])                     │  │
│  │      )                                                            │  │
│  │                                                                    │  │
│  │      # Record step for trajectory                                 │  │
│  │      step_records[worker_id].append(StepRecord(                   │  │
│  │          state=features.cpu(),          # [908]                   │  │
│  │          action_logits=logits.cpu(),    # Dict of tensors         │  │
│  │          action_taken=actions.cpu(),    # Dict of sampled actions │  │
│  │          log_prob=log_prob.cpu(),       # Scalar                  │  │
│  │          value=value.cpu(),             # Scalar                  │  │
│  │          reward=reward                  # From worker             │  │
│  │      ))                                                           │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │         Convert Actions to Primitives & Send to Workers            │  │
│  │         trajectory_slicer.py:137-144                               │  │
│  │                                                                    │  │
│  │  For each worker:                                                 │  │
│  │      # Convert tensors to Python primitives                       │  │
│  │      p1_primitives = {                                            │  │
│  │          'main_stick': int(actions['main_stick'].item()),         │  │
│  │          'c_stick': int(actions['c_stick'].item()),               │  │
│  │          'buttons': actions['buttons'].tolist(),  # [bool, ...]   │  │
│  │          'shoulder': int(actions['shoulder'].item())              │  │
│  │      }                                                            │  │
│  │                                                                    │  │
│  │      action_queues[worker_id].put((p1_primitives, p2_primitives)) │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                                                                           │
│         [Actions sent back to workers, loop continues]                    │
└───────────────────────────────────────────────────────────────────────────┘
```

**Performance Metrics:**
- **Batch processing**: 8 workers processed simultaneously
- **Inference time**: ~5-10ms per batch (FP16 + torch.compile)
- **Throughput**: 8 frames / 0.01s = **800 FPS** (theoretical)
- **GPU utilization**: High (~80-90%)
- **CPU bottleneck**: Minimal with numpy serialization

---

## Trajectory Collection & GAE

### Rollout Collection Flow

```
┌───────────────────────────────────────────────────────────────────────────┐
│                        Trajectory Slicer (Main Process)                    │
│                        ppo/trajectory_slicer.py:85-195                     │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Phase 1: Collect Fixed-Length Rollout                              │ │
│  │  trajectory_slicer.py:107-157                                        │ │
│  │                                                                      │ │
│  │  Target: 5000 frames across all workers                             │ │
│  │                                                                      │ │
│  │  while frames_collected < rollout_length (5000):                    │ │
│  │      # Collect states from all 8 workers                            │ │
│  │      worker_states = []                                             │ │
│  │      for _ in range(num_workers=8):                                 │ │
│  │          (wid, features_np, reward, done) = state_queue.get()       │ │
│  │          features = torch.from_numpy(features_np)                   │ │
│  │          worker_states.append((wid, features, reward, done))        │ │
│  │                                                                      │ │
│  │      # Batched inference (8 workers)                                │ │
│  │      actions = coordinator.process_states(worker_states)            │ │
│  │      # → Returns 8 action dicts                                     │ │
│  │      # → Records StepRecord for each worker in coordinator          │ │
│  │                                                                      │ │
│  │      # Send actions back                                            │ │
│  │      for worker_id, (p1_act, p2_act) in actions.items():            │ │
│  │          action_queues[worker_id].put((p1_act, p2_act))             │ │
│  │                                                                      │ │
│  │      frames_collected += 8  # One frame per worker                 │ │
│  │                                                                      │ │
│  │  # Result: ~625 iterations × 8 workers = 5000 frames                │ │
│  │  # Time: ~625 × 10ms = 6.25 seconds @ 800 FPS                       │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Phase 2: Get Bootstrap Values (for incomplete episodes)            │ │
│  │  trajectory_slicer.py:160-161                                        │ │
│  │  inference_coordinator.py:411-450                                    │ │
│  │                                                                      │ │
│  │  # For each worker's current state, estimate V(s_T)                 │ │
│  │  bootstrap_values = coordinator.bootstrap_values()                  │ │
│  │                                                                      │ │
│  │  # Returns: {worker_id: value_tensor}                               │ │
│  │  # value_tensor: scalar, the model's estimate of V(s_final)         │ │
│  │  # Used for GAE when episode hasn't ended                           │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Phase 3: Convert StepRecords to Trajectory Steps                   │ │
│  │  trajectory_slicer.py:163-176                                        │ │
│  │                                                                      │ │
│  │  worker_steps = {}                                                  │ │
│  │  for worker_id, records in coordinator.step_records.items():        │ │
│  │      worker_steps[worker_id] = [                                    │ │
│  │          Step(                                                      │ │
│  │              state=record.state,           # [908] features         │ │
│  │              action_logits=record.logits,  # Dict of tensors        │ │
│  │              action_taken=record.actions,  # Dict of actions        │ │
│  │              log_prob=record.log_prob,     # Scalar                 │ │
│  │              value=record.value,           # Scalar V(s)            │ │
│  │              reward=record.reward,         # Scalar r               │ │
│  │              done=False                    # Continuous mode        │ │
│  │          )                                                          │ │
│  │          for record in records                                      │ │
│  │      ]                                                              │ │
│  │                                                                      │ │
│  │  # Result: RolloutSlice(                                            │ │
│  │  #     worker_steps={0: [Step, ...], 1: [Step, ...], ...},          │ │
│  │  #     bootstrap_values={0: tensor, 1: tensor, ...},                │ │
│  │  #     total_frames=5000                                            │ │
│  │  # )                                                                │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

### GAE (Generalized Advantage Estimation) Computation

```
┌───────────────────────────────────────────────────────────────────────────┐
│                   Advantage Computation (CPU)                              │
│                   ppo/trajectory.py:70-170                                 │
│                                                                            │
│  Input: RolloutSlice with worker steps                                    │
│         Each step has: (state, action, log_prob, value, reward)           │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Step 1: Prepare Trajectory Data (per worker)                       │ │
│  │  ppo/trajectory.py:88-115                                            │ │
│  │                                                                      │ │
│  │  For each worker's trajectory:                                       │ │
│  │      states = [step.state for step in steps]      # List[Tensor]    │ │
│  │      values = [step.value for step in steps]      # List[float]     │ │
│  │      rewards = [step.reward for step in steps]    # List[float]     │ │
│  │      log_probs = [step.log_prob for step in steps]  # List[float]   │ │
│  │                                                                      │ │
│  │      # Example for worker 0 with 625 steps:                          │ │
│  │      values = [0.234, 0.189, 0.245, ..., 0.412]   # 625 floats      │ │
│  │      rewards = [0.02, -0.01, 0.15, ..., 0.08]     # 625 floats      │ │
│  │      bootstrap_value = 0.389  # From coordinator for this worker    │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Step 2: Compute TD Residuals (δ_t)                                 │ │
│  │  ppo/trajectory.py:127-142                                           │ │
│  │                                                                      │ │
│  │  Temporal Difference (TD) residual at each timestep:                 │ │
│  │                                                                      │ │
│  │      δ_t = r_t + γ * V(s_{t+1}) - V(s_t)                            │ │
│  │                                                                      │ │
│  │  Where:                                                              │ │
│  │      r_t = rewards[t]           # Immediate reward                  │ │
│  │      V(s_t) = values[t]         # Value of current state            │ │
│  │      V(s_{t+1}) = values[t+1]   # Value of next state               │ │
│  │                   OR bootstrap_value (if t == last step)            │ │
│  │      γ = 0.995                  # Discount factor (config.rl.gamma) │ │
│  │                                                                      │ │
│  │  Example calculation:                                                │ │
│  │      t=0: δ_0 = 0.02 + 0.995*0.189 - 0.234 = -0.026                 │ │
│  │      t=1: δ_1 = -0.01 + 0.995*0.245 - 0.189 = 0.045                 │ │
│  │      ...                                                             │ │
│  │      t=624: δ_624 = 0.08 + 0.995*0.389 - 0.412 = 0.055              │ │
│  │                    (uses bootstrap_value for V(s_{625}))            │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Step 3: Compute GAE Advantages (Â_t)                               │ │
│  │  ppo/trajectory.py:143-160                                           │ │
│  │                                                                      │ │
│  │  Generalized Advantage Estimate (GAE):                               │ │
│  │                                                                      │ │
│  │      Â_t = Σ_{l=0}^{T-t} (γλ)^l * δ_{t+l}                           │ │
│  │                                                                      │ │
│  │  Where:                                                              │ │
│  │      λ = 0.95     # GAE lambda (config.ppo.gae_lambda)              │ │
│  │      γ = 0.995    # Discount factor                                 │ │
│  │                                                                      │ │
│  │  Computed backwards (more efficient):                                │ │
│  │                                                                      │ │
│  │      advantages = torch.zeros(T)                                    │ │
│  │      gae = 0.0                                                      │ │
│  │      for t in reversed(range(T)):                                   │ │
│  │          gae = δ_t + γ * λ * gae                                    │ │
│  │          advantages[t] = gae                                        │ │
│  │                                                                      │ │
│  │  Example (backwards from t=624):                                     │ │
│  │      t=624: gae = δ_624 = 0.055                                     │ │
│  │              Â_624 = 0.055                                          │ │
│  │      t=623: gae = δ_623 + 0.995*0.95*0.055 = δ_623 + 0.052          │ │
│  │              Â_623 = gae                                            │ │
│  │      ...                                                             │ │
│  │                                                                      │ │
│  │  Result: advantages tensor [625] with advantage estimates           │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Step 4: Compute Returns (targets for value function)               │ │
│  │  ppo/trajectory.py:161-170                                           │ │
│  │                                                                      │ │
│  │      returns = advantages + values                                  │ │
│  │                                                                      │ │
│  │  This gives us the target for the value function:                    │ │
│  │      V_target(s_t) = Â_t + V(s_t)                                   │ │
│  │                                                                      │ │
│  │  Example:                                                            │ │
│  │      t=0: return_0 = Â_0 + 0.234 = target for V(s_0)                │ │
│  │      t=1: return_1 = Â_1 + 0.189 = target for V(s_1)                │ │
│  │      ...                                                             │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Step 5: Normalize Advantages (optional)                            │ │
│  │  ppo/trajectory.py:171-175                                           │ │
│  │                                                                      │ │
│  │  if normalize_advantages:  # config.ppo.normalize_advantages=True   │ │
│  │      advantages = (advantages - advantages.mean()) / (               │ │
│  │          advantages.std() + 1e-8                                    │ │
│  │      )                                                              │ │
│  │                                                                      │ │
│  │  This stabilizes training by keeping advantages ~N(0, 1)             │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  Output: Trajectory object with:                                          │
│      - states: List[Tensor[908]]                                          │
│      - action_logits: List[Dict[str, Tensor]]                             │
│      - actions_taken: List[Dict[str, Tensor]]                             │
│      - old_log_probs: List[float]                                         │
│      - advantages: Tensor[T]      ← Used for policy gradient              │
│      - returns: Tensor[T]         ← Used for value loss                  │
│      - old_values: Tensor[T]      ← Old value predictions                │
└────────────────────────────────────────────────────────────────────────────┘
```

**Key Formulas:**

```
TD Residual:     δ_t = r_t + γ·V(s_{t+1}) - V(s_t)

GAE:             Â_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}

Returns:         R_t = Â_t + V(s_t)   (target for value function)

Normalized Adv:  Â'_t = (Â_t - μ) / σ
```

---

## PPO Training Loop

### Preparing Training Windows

```
┌───────────────────────────────────────────────────────────────────────────┐
│              Window Preparation for Sequential Model                       │
│              ppo/trajectory_slicer.py:192-250                              │
│                                                                            │
│  Input: RolloutSlice with ~5000 total steps across 8 workers              │
│         Each step: (state[908], actions, log_probs, value, advantage)     │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Step 1: Flatten All Worker Trajectories                            │ │
│  │                                                                      │ │
│  │  all_states = []                                                     │ │
│  │  all_advantages = []                                                 │ │
│  │  all_returns = []                                                    │ │
│  │  ...                                                                 │ │
│  │                                                                      │ │
│  │  for worker_id, steps in worker_steps.items():                       │ │
│  │      for step in steps:                                             │ │
│  │          all_states.append(step.state)                              │ │
│  │          all_advantages.append(step.advantage)                      │ │
│  │          all_returns.append(step.return)                            │ │
│  │          all_log_probs.append(step.log_prob)                        │ │
│  │          all_actions.append(step.action_taken)                      │ │
│  │          all_values.append(step.value)                              │ │
│  │                                                                      │ │
│  │  # Result: ~5000 individual frames mixed from all workers           │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Step 2: Create Sliding Windows (for sequential model)              │ │
│  │  trajectory_slicer.py:210-240                                        │ │
│  │                                                                      │ │
│  │  Model requires sequences of length seq_len (256)                    │ │
│  │                                                                      │ │
│  │  num_windows = len(all_states) - seq_len + 1                         │ │
│  │              = 5000 - 256 + 1 = 4745 windows                         │ │
│  │                                                                      │ │
│  │  But we subsample for efficiency:                                    │ │
│  │      stride = max(1, len(all_states) // (seq_len * 10))              │ │
│  │             = max(1, 5000 // 2560) = 1 or 2                          │ │
│  │                                                                      │ │
│  │  window_states = []                                                  │ │
│  │  window_advantages = []                                              │ │
│  │  ...                                                                 │ │
│  │                                                                      │ │
│  │  for i in range(0, num_windows, stride):                             │ │
│  │      # Extract window of 256 consecutive frames                      │ │
│  │      window_states.append(                                           │ │
│  │          torch.stack(all_states[i:i+256], dim=0)  # [256, 908]       │ │
│  │      )                                                               │ │
│  │      window_advantages.append(                                       │ │
│  │          torch.tensor(all_advantages[i:i+256])    # [256]            │ │
│  │      )                                                               │ │
│  │      # ... same for returns, actions, log_probs, values              │ │
│  │                                                                      │ │
│  │  # Stack all windows into tensors                                    │ │
│  │  states = torch.stack(window_states, dim=0)       # [N, 256, 908]    │ │
│  │  advantages = torch.stack(window_advantages, dim=0)  # [N, 256]      │ │
│  │  returns = torch.stack(window_returns, dim=0)     # [N, 256]         │ │
│  │  ...                                                                 │ │
│  │                                                                      │ │
│  │  where N ≈ 40-50 windows (depends on stride)                         │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  Output: Dictionary of tensors ready for training                         │
│      {                                                                     │
│          'states': [N, 256, 908],      # Input sequences                  │
│          'advantages': [N, 256],        # For policy loss                 │
│          'returns': [N, 256],           # For value loss                  │
│          'old_log_probs': [N, 256],     # For PPO clipping                │
│          'actions_taken': [N, 256, ...],# Actions to evaluate             │
│          'old_values': [N, 256]         # Old value predictions           │
│      }                                                                     │
└────────────────────────────────────────────────────────────────────────────┘
```

### PPO Training Epochs

```
┌───────────────────────────────────────────────────────────────────────────┐
│                         PPO Training Loop                                  │
│                         train_ppo.py:730-875                               │
│                                                                            │
│  Input: windows dict with N windows (N ≈ 40-50)                           │
│         config.ppo.ppo_epochs = 4    # Train on data 4 times              │
│         config.ppo.minibatch_size = 64                                     │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Outer Loop: PPO Epochs (reuse same data)                            │ │
│  │  train_ppo.py:785-873                                                │ │
│  │                                                                      │ │
│  │  for ppo_epoch in range(ppo_epochs=4):                               │ │
│  │      # Shuffle windows each epoch                                    │ │
│  │      indices = torch.randperm(N)                                     │ │
│  │                                                                      │ │
│  │      # Create minibatches                                            │ │
│  │      num_minibatches = ceil(N / minibatch_size)                      │ │
│  │                     = ceil(40 / 64) = 1 minibatch                    │ │
│  │      # (Or could be 2-3 minibatches if N is larger)                  │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Inner Loop: Minibatch Training                                      │ │
│  │  train_ppo.py:790-860                                                │ │
│  │                                                                      │ │
│  │  for mb_idx, start_idx in enumerate(range(0, N, minibatch_size)):    │ │
│  │      end_idx = min(start_idx + minibatch_size, N)                    │ │
│  │      mb_indices = indices[start_idx:end_idx]                         │ │
│  │                                                                      │ │
│  │      # Extract minibatch                                             │ │
│  │      mb_states = windows['states'][mb_indices]        # [B, 256, 908]│ │
│  │      mb_advantages = windows['advantages'][mb_indices]  # [B, 256]   │ │
│  │      mb_returns = windows['returns'][mb_indices]        # [B, 256]   │ │
│  │      mb_old_log_probs = windows['old_log_probs'][mb_indices]  # [B, 256]││
│  │      mb_actions_taken = windows['actions_taken'][mb_indices]         │ │
│  │      mb_old_values = windows['old_values'][mb_indices]  # [B, 256]   │ │
│  │                                                                      │ │
│  │      # Move to GPU                                                   │ │
│  │      mb_states = mb_states.to(device='cuda')                         │ │
│  │      mb_advantages = mb_advantages.to(device='cuda')                 │ │
│  │      mb_returns = mb_returns.to(device='cuda')                       │ │
│  │      ...                                                             │ │
│  │                                                                      │ │
│  │      ┌────────────────────────────────────────────────────────────┐  │ │
│  │      │  Forward Pass (GPU)                                        │  │ │
│  │      │  train_ppo.py:800-810                                      │  │ │
│  │      │                                                            │  │ │
│  │      │  # Build model inputs                                      │  │ │
│  │      │  inputs = build_model_inputs(mb_states, colmap)            │  │ │
│  │      │  # → TensorDict with one-hot encodings                     │  │ │
│  │      │                                                            │  │ │
│  │      │  # FP16 + torch.compile                                    │  │ │
│  │      │  with autocast('cuda', enabled=True):                      │  │ │
│  │      │      outputs = model(inputs)                               │  │ │
│  │      │  # → TensorDict:                                           │  │ │
│  │      │  #    'main_stick': [B, 256, 64]                           │  │ │
│  │      │  #    'c_stick': [B, 256, 9]                               │  │ │
│  │      │  #    'buttons': [B, 256, 5]                               │  │ │
│  │      │  #    'shoulder': [B, 256, 5]                              │  │ │
│  │      │  #    'value': [B, 256, 1]                                 │  │ │
│  │      └────────────────────────────────────────────────────────────┘  │ │
│  │                                                                      │ │
│  │      ┌────────────────────────────────────────────────────────────┐  │ │
│  │      │  Compute PPO Loss                                          │  │ │
│  │      │  ppo/ppo_loss.py:15-180                                    │  │ │
│  │      │                                                            │  │ │
│  │      │  loss, metrics = compute_total_ppo_loss(                   │  │ │
│  │      │      model_outputs=outputs,                                │  │ │
│  │      │      old_values=mb_old_values,                             │  │ │
│  │      │      actions_taken=mb_actions_taken,                       │  │ │
│  │      │      old_log_probs=mb_old_log_probs,                       │  │ │
│  │      │      advantages=mb_advantages,                             │  │ │
│  │      │      returns=mb_returns,                                   │  │ │
│  │      │      clip_ratio=0.2,                                       │  │ │
│  │      │      entropy_coef=0.01,                                    │  │ │
│  │      │      value_coef=0.5                                        │  │ │
│  │      │  )                                                         │  │ │
│  │      │                                                            │  │ │
│  │      │  ┌──────────────────────────────────────────────────────┐ │  │ │
│  │      │  │  Loss Components (see next section for details)      │ │  │ │
│  │      │  │                                                      │ │  │ │
│  │      │  │  1. Policy Loss (Clipped Surrogate)                  │ │  │ │
│  │      │  │     L_policy = -E[min(ratio·Â, clip(ratio)·Â)]       │ │  │ │
│  │      │  │     where ratio = π_new(a|s) / π_old(a|s)            │ │  │ │
│  │      │  │                                                      │ │  │ │
│  │      │  │  2. Value Loss (MSE with optional clipping)          │ │  │ │
│  │      │  │     L_value = E[(V_new(s) - R)²]                     │ │  │ │
│  │      │  │                                                      │ │  │ │
│  │      │  │  3. Entropy Bonus (for exploration)                  │ │  │ │
│  │      │  │     L_entropy = -E[H(π)]                             │ │  │ │
│  │      │  │                                                      │ │  │ │
│  │      │  │  Total: L = L_policy + c1·L_value + c2·L_entropy     │ │  │ │
│  │      │  └──────────────────────────────────────────────────────┘ │  │ │
│  │      └────────────────────────────────────────────────────────────┘  │ │
│  │                                                                      │ │
│  │      ┌────────────────────────────────────────────────────────────┐  │ │
│  │      │  Backward Pass & Optimizer Step                           │  │ │
│  │      │  train_ppo.py:835-858                                      │  │ │
│  │      │                                                            │  │ │
│  │      │  optimizer.zero_grad()                                     │  │ │
│  │      │  scaler.scale(loss).backward()  # FP16 gradient scaling    │  │ │
│  │      │  scaler.unscale_(optimizer)                                │  │ │
│  │      │                                                            │  │ │
│  │      │  # Gradient clipping (prevent explosions)                  │  │ │
│  │      │  grad_norm = clip_grad_norm_(                              │  │ │
│  │      │      model.parameters(),                                   │  │ │
│  │      │      max_norm=0.5  # config.ppo.max_grad_norm              │  │ │
│  │      │  )                                                         │  │ │
│  │      │                                                            │  │ │
│  │      │  scaler.step(optimizer)                                    │  │ │
│  │      │  scaler.update()                                           │  │ │
│  │      └────────────────────────────────────────────────────────────┘  │ │
│  │                                                                      │ │
│  │  # End of minibatch loop                                             │ │
│  │  # End of PPO epoch loop                                             │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  Result: Model updated via 4 epochs × 1-2 minibatches                     │
│         Gradient updates: 4-8 total per rollout                           │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## PPO Loss Computation (Detailed)

```
┌───────────────────────────────────────────────────────────────────────────┐
│                      PPO Loss Components (GPU)                             │
│                      ppo/ppo_loss.py:15-180                                │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Component 1: Policy Loss (Clipped Surrogate Objective)             │ │
│  │  ppo/ppo_loss.py:50-110                                              │ │
│  │                                                                      │ │
│  │  Goal: Maximize E[min(ratio·Â, clip(ratio)·Â)]                      │ │
│  │  (Equivalent to minimizing negative)                                 │ │
│  │                                                                      │ │
│  │  Step 1: Compute current policy log probabilities                    │ │
│  │  ────────────────────────────────────────────────────                │ │
│  │  From model outputs (logits):                                        │ │
│  │      main_logits: [B, 256, 64]                                       │ │
│  │      c_logits: [B, 256, 9]                                           │ │
│  │      button_logits: [B, 256, 5]                                      │ │
│  │      shoulder_logits: [B, 256, 5]                                    │ │
│  │                                                                      │ │
│  │  From actions_taken (what was actually done):                        │ │
│  │      main_actions: [B, 256]  (indices 0-63)                          │ │
│  │      c_actions: [B, 256]     (indices 0-8)                           │ │
│  │      button_actions: [B, 256, 5]  (boolean)                          │ │
│  │      shoulder_actions: [B, 256]   (indices 0-4)                      │ │
│  │                                                                      │ │
│  │  Compute log probability for each action component:                  │ │
│  │                                                                      │ │
│  │      # Main stick (categorical)                                      │ │
│  │      main_log_probs = F.log_softmax(main_logits, dim=-1)             │ │
│  │      main_lp = main_log_probs.gather(-1, main_actions.unsqueeze(-1)) │ │
│  │      # → [B, 256, 1] → squeeze → [B, 256]                            │ │
│  │                                                                      │ │
│  │      # C-stick (categorical)                                         │ │
│  │      c_log_probs = F.log_softmax(c_logits, dim=-1)                   │ │
│  │      c_lp = c_log_probs.gather(-1, c_actions.unsqueeze(-1)).squeeze()│ │
│  │                                                                      │ │
│  │      # Buttons (independent bernoulli for each)                      │ │
│  │      button_probs = torch.sigmoid(button_logits)                     │ │
│  │      button_lp = (                                                   │ │
│  │          button_actions * torch.log(button_probs + 1e-8) +           │ │
│  │          (1 - button_actions) * torch.log(1 - button_probs + 1e-8)   │ │
│  │      ).sum(dim=-1)  # Sum over 5 buttons → [B, 256]                  │ │
│  │                                                                      │ │
│  │      # Shoulder (categorical)                                        │ │
│  │      shoulder_lp = F.log_softmax(shoulder_logits, dim=-1).gather(...)│ │
│  │                                                                      │ │
│  │      # Total log probability (sum in log space = product in prob)    │ │
│  │      new_log_probs = main_lp + c_lp + button_lp + shoulder_lp        │ │
│  │      # → [B, 256]                                                    │ │
│  │                                                                      │ │
│  │  Step 2: Compute probability ratio                                   │ │
│  │  ────────────────────────────────                                   │ │
│  │      ratio = exp(new_log_probs - old_log_probs)                      │ │
│  │            = π_θ_new(a|s) / π_θ_old(a|s)                             │ │
│  │      # → [B, 256]                                                    │ │
│  │                                                                      │ │
│  │      # Interpretation:                                               │ │
│  │      #   ratio > 1: new policy assigns higher prob to action         │ │
│  │      #   ratio < 1: new policy assigns lower prob to action          │ │
│  │      #   ratio ≈ 1: policies are similar                             │ │
│  │                                                                      │ │
│  │  Step 3: Clipped surrogate objective                                 │ │
│  │  ────────────────────────────────                                   │ │
│  │      surr1 = ratio * advantages                                      │ │
│  │      surr2 = torch.clamp(ratio, 1-ε, 1+ε) * advantages               │ │
│  │              where ε = clip_ratio = 0.2                              │ │
│  │                                                                      │ │
│  │      # Take minimum (conservative update)                            │ │
│  │      policy_loss = -torch.min(surr1, surr2).mean()                   │ │
│  │                                                                      │ │
│  │      # Why clipping?                                                 │ │
│  │      # Prevents too-large policy updates that could harm performance │ │
│  │      # Only allows ratio to move within [0.8, 1.2]                   │ │
│  │                                                                      │ │
│  │  Example values:                                                     │ │
│  │      ratio=1.5, adv=0.3 → surr1=0.45, surr2=clip(1.5,0.8,1.2)*0.3=0.36│ │
│  │                        → min = 0.36 (clipped!)                       │ │
│  │      ratio=0.9, adv=0.3 → surr1=0.27, surr2=0.27 (no clip)           │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Component 2: Value Loss                                             │ │
│  │  ppo/ppo_loss.py:112-135                                             │ │
│  │                                                                      │ │
│  │  Goal: Train value function to predict returns accurately            │ │
│  │                                                                      │ │
│  │  new_values = model_outputs['value'][:, :, 0]  # [B, 256]            │ │
│  │                                                                      │ │
│  │  # Optional: Clip value predictions (similar to policy clipping)     │ │
│  │  if value_clip is not None:                                          │ │
│  │      v_clipped = old_values + torch.clamp(                           │ │
│  │          new_values - old_values,                                    │ │
│  │          -value_clip, value_clip                                     │ │
│  │      )                                                               │ │
│  │      v_loss1 = (new_values - returns) ** 2                           │ │
│  │      v_loss2 = (v_clipped - returns) ** 2                            │ │
│  │      value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()           │ │
│  │  else:                                                               │ │
│  │      value_loss = 0.5 * F.mse_loss(new_values, returns)              │ │
│  │                                                                      │ │
│  │  # MSE between predicted value and actual return                     │ │
│  │  # Factor of 0.5 is convention                                       │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Component 3: Entropy Bonus                                          │ │
│  │  ppo/ppo_loss.py:137-155                                             │ │
│  │                                                                      │ │
│  │  Goal: Encourage exploration by rewarding policy diversity           │ │
│  │                                                                      │ │
│  │  # Categorical distributions (main stick, c-stick, shoulder)         │ │
│  │  main_probs = F.softmax(main_logits, dim=-1)                         │ │
│  │  main_entropy = -(main_probs * torch.log(main_probs + 1e-8)).sum(-1) │ │
│  │  # → [B, 256]                                                        │ │
│  │  # High entropy = uniform distribution = more exploration            │ │
│  │  # Low entropy = peaked distribution = more exploitation             │ │
│  │                                                                      │ │
│  │  # Bernoulli distributions (buttons)                                 │ │
│  │  button_probs = torch.sigmoid(button_logits)                         │ │
│  │  button_entropy = -(                                                 │ │
│  │      button_probs * torch.log(button_probs + 1e-8) +                 │ │
│  │      (1 - button_probs) * torch.log(1 - button_probs + 1e-8)         │ │
│  │  ).sum(dim=-1)  # → [B, 256]                                         │ │
│  │                                                                      │ │
│  │  total_entropy = (main_entropy + c_entropy +                         │ │
│  │                   button_entropy + shoulder_entropy)                 │ │
│  │                                                                      │ │
│  │  entropy_bonus = total_entropy.mean()                                │ │
│  │  # We SUBTRACT this from loss (higher entropy = lower loss)          │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Combined Loss                                                       │ │
│  │  ppo/ppo_loss.py:157-170                                             │ │
│  │                                                                      │ │
│  │  total_loss = (                                                      │ │
│  │      policy_loss +                    # Clipped surrogate            │ │
│  │      value_coef * value_loss -        # Value MSE (c1=0.5)           │ │
│  │      entropy_coef * entropy_bonus     # Exploration (c2=0.01)        │ │
│  │  )                                                                   │ │
│  │                                                                      │ │
│  │  # Example values:                                                   │ │
│  │  #   policy_loss = 0.234                                             │ │
│  │  #   value_loss = 0.456                                              │ │
│  │  #   entropy = 2.103                                                 │ │
│  │  #   total = 0.234 + 0.5*0.456 - 0.01*2.103                          │ │
│  │  #         = 0.234 + 0.228 - 0.021 = 0.441                           │ │
│  │                                                                      │ │
│  │  metrics = {                                                         │ │
│  │      'train/policy_loss': policy_loss.item(),                        │ │
│  │      'train/value_loss': value_loss.item(),                          │ │
│  │      'train/entropy': entropy_bonus.item(),                          │ │
│  │      'train/total_loss': total_loss.item(),                          │ │
│  │      'train/approx_kl': ((ratio - 1) - ratio.log()).mean().item(),   │ │
│  │      'train/clipfrac': (torch.abs(ratio - 1) > 0.2).float().mean()   │ │
│  │  }                                                                   │ │
│  │                                                                      │ │
│  │  return total_loss, metrics                                          │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

**Key Metrics:**
- **approx_kl**: Approximation of KL divergence between old/new policies
- **clipfrac**: Fraction of samples where ratio was clipped (should be <0.3)

---

## Hardware Mapping

### CPU vs GPU Distribution

```
┌───────────────────────────────────────────────────────────────────────────┐
│                            Hardware Allocation                             │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                          GPU (CUDA)                                  │ │
│  │                                                                      │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Learner Model (26M parameters)                                │ │ │
│  │  │  - Transformer layers (8 blocks)                               │ │ │
│  │  │  - Output heads (5 heads with cross-attention)                 │ │ │
│  │  │  - FP16 precision via autocast                                 │ │ │
│  │  │  - torch.compile JIT optimization                              │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                      │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Opponent Model (copy of learner, frozen)                      │ │ │
│  │  │  - Same architecture                                           │ │ │
│  │  │  - FP16 precision                                              │ │ │
│  │  │  - Swapped from opponent pool periodically                     │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                      │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Forward Pass Data (during inference)                          │ │ │
│  │  │  - Input batch: [8, 256, 908] → ~18MB FP16                     │ │ │
│  │  │  - Activations: intermediate layer outputs                     │ │ │
│  │  │  - Output logits: [8, 256, ~83] → ~0.3MB FP16                  │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                      │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Training Data (during backward pass)                          │ │ │
│  │  │  - Minibatch states: [B, 256, 908]                             │ │ │
│  │  │  - Advantages, returns: [B, 256]                               │ │ │
│  │  │  - Gradients: same size as model parameters                    │ │ │
│  │  │  - Optimizer state: Adam momentum/variance buffers             │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                      │ │
│  │  GPU Memory Usage: ~2-4GB (model + activations + optimizer state)   │ │
│  │  GPU Utilization: ~80-90% during inference bursts                   │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                          CPU (Main Process)                          │ │
│  │                                                                      │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │  InferenceCoordinator State                                    │ │ │
│  │  │  - Worker buffers: 8 × deque(maxlen=256) × 908 floats          │ │ │
│  │  │  - Step records: ~5000 steps × (state + logits + actions)      │ │ │
│  │  │    ≈ 5000 × (908 + 83) floats ≈ 20MB                           │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                      │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Trajectory Processing                                         │ │ │
│  │  │  - GAE computation: vectorized operations on CPU               │ │ │
│  │  │  - Window slicing: creating training batches                   │ │ │
│  │  │  - NumPy/PyTorch CPU tensors                                   │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                      │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Multiprocessing Queues                                        │ │ │
│  │  │  - state_queue: receives numpy arrays from workers             │ │ │
│  │  │  - action_queues[0-7]: sends primitives to workers             │ │ │
│  │  │  - control_queues[0-7]: pause/resume signals                   │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                      │ │
│  │  CPU Usage: One core at ~100% (coordination + inference)            │ │
│  │  RAM Usage: ~500MB-1GB (buffers + trajectories)                     │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                    CPU (Worker Processes × 8)                        │ │
│  │                                                                      │ │
│  │  Each worker process:                                                │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Dolphin Emulator Instance                                     │ │ │
│  │  │  - Full game emulation at ~60 FPS native                       │ │ │
│  │  │  - Graphics, physics, AI (Melee engine)                        │ │ │
│  │  │  - Network socket to libmelee                                  │ │ │
│  │  │  CPU: 5-15% per instance (emulation is efficient)              │ │ │
│  │  │  RAM: ~200-500MB per instance                                  │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                      │ │
│  │  ┌────────────────────────────────────────────────────────────────┐ │ │
│  │  │  Python Worker Logic                                           │ │ │
│  │  │  - Feature extraction: dict → numpy array                      │ │ │
│  │  │  - Reward computation: CPU tensor operations                   │ │ │
│  │  │  - Action conversion: primitives → controller state            │ │ │
│  │  │  - Queue I/O: numpy serialization                              │ │ │
│  │  │  CPU: 1-3% per worker                                          │ │ │
│  │  │  RAM: ~100MB per worker                                        │ │ │
│  │  └────────────────────────────────────────────────────────────────┘ │ │
│  │                                                                      │ │
│  │  Total CPU: 8 workers × 20% ≈ 160% (2 cores worth)                  │ │
│  │  Total RAM: 8 × 600MB ≈ 5GB                                         │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │                      Overall System Profile                          │ │
│  │                                                                      │ │
│  │  CPU Cores: ~3-4 cores actively used                                 │ │
│  │    - 1 core: main process (coordination + light compute)             │ │
│  │    - 2-3 cores: 8 worker processes (emulation)                       │ │
│  │                                                                      │ │
│  │  RAM: ~6-7GB total                                                   │ │
│  │    - Main process: ~1GB (buffers + trajectories)                     │ │
│  │    - Workers: ~5GB (8 × Dolphin instances)                           │ │
│  │                                                                      │ │
│  │  GPU: High utilization during inference bursts                       │ │
│  │    - Memory: 2-4GB (model + optimizer + batch data)                  │ │
│  │    - Compute: 80-90% during forward/backward passes                  │ │
│  │    - Idle time: Minimal with good batching                           │ │
│  │                                                                      │ │
│  │  Network: Multiprocessing queues (localhost, very fast)              │ │
│  │    - Bandwidth: ~100-200 MB/s (numpy arrays)                         │ │
│  │    - Latency: <1ms (shared memory)                                   │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Data Flow Timeline

### Single Rollout Cycle (5000 frames)

```
Time    Main Process          GPU                Worker 0         Worker 1-7
────────────────────────────────────────────────────────────────────────────
0.0s    Waiting for          Idle               Extract          [similar]
        state_queue                             GameState
                                                   ↓
                                                Convert to
                                                features_np
                                                   ↓
0.001s                                          Compute
                                                reward
                                                   ↓
                                                queue.put()
                                                   ↓
0.002s  Receive 8 states     ──────────────→    Waiting
        Convert np→tensor                       for actions
           ↓
        Stack into batch
        [8, 256, 908]
           ↓
0.003s  Send to GPU ─────→

0.004s                       Forward pass
                             (FP16, compiled)
                             ~5-8ms
                                ↓
0.012s  ←──── Receive        Output logits
        outputs              [8, 256, 83]
           ↓
        Sample actions
        Record steps
           ↓
0.013s  Convert to
        primitives
           ↓
0.014s  queue.put() ────────────────────────→ Receive
                                                actions
                                                   ↓
0.015s                                          Apply to
                                                controllers
                                                   ↓
                                                Dolphin
                                                advances
                                                1 frame
                                                   ↓
                            [LOOP REPEATS]

────────────────────────────────────────────────────────────────────────────
After ~625 iterations (5000 frames collected):

Time    Main Process          GPU                Workers
────────────────────────────────────────────────────────────────
6.0s    pause_workers()                         Paused
           ↓
        Get bootstrap
        values ─────────→

6.1s                       Forward pass
                           (final states)
                              ↓
6.2s    ←──── Receive
        bootstrap vals
           ↓
        Compute GAE
        (CPU, vectorized)
        ~100ms
           ↓
6.3s    Slice windows
        (CPU)
        ~50ms
           ↓
6.4s    ──── Training ───→
        Start

        For 4 epochs:
          For each batch:
6.5s      Send batch ───→  Forward pass
                           ~10ms
                              ↓
6.51s   ←─ outputs         Backward pass
        Compute loss       ~15ms
           ↓                  ↓
6.52s   Send gradients ─→  Optimizer step
                           ~5ms
                              ↓
6.53s   ←─ done

        [Repeat for
         next batch]
           ↓
7.0s    Training done
           ↓
        resume_workers() ─────────────────→ Resume
                                             emulation

        [NEXT ROLLOUT]
```

**Throughput:**
- **Collection**: 5000 frames / 6.0s = **833 FPS**
- **Training**: ~0.5s for 4 epochs
- **Total cycle**: ~6.5s
- **Frames per second** (amortized): 5000 / 6.5 ≈ **770 FPS**

---

## Summary Statistics

### Per-Rollout Metrics

```
┌────────────────────────────────────────────────────────────┐
│                    Rollout Breakdown                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Phase 1: Collection                                       │
│    Duration: ~6.0 seconds                                  │
│    Frames: 5000 (625 batches × 8 workers)                  │
│    FPS: 833                                                │
│    GPU: 625 × 8ms = 5.0s (inference time)                  │
│    CPU: 625 × 2ms = 1.25s (queue overhead)                 │
│                                                            │
│  Phase 2: GAE Computation                                  │
│    Duration: ~0.15 seconds                                 │
│    Operations: Vectorized CPU tensor ops                   │
│    Memory: ~100MB temporary arrays                         │
│                                                            │
│  Phase 3: Training                                         │
│    Duration: ~0.5 seconds                                  │
│    Epochs: 4                                               │
│    Batches per epoch: 1-2                                  │
│    GPU utilization: 90%+ during training                   │
│                                                            │
│  Total: ~6.65 seconds per rollout                          │
│                                                            │
│  Throughput: 5000 frames / 6.65s = 752 FPS                 │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### Key Files Reference

```
train_ppo.py:95-280         - Main distributed training loop
ppo/inference_coordinator.py - GPU inference coordination
ppo/simulation_worker.py     - Worker process (Dolphin + features)
ppo/trajectory_slicer.py     - Rollout collection orchestration
ppo/trajectory.py           - GAE computation
ppo/ppo_loss.py             - Loss functions
model/nano_gpt.py           - Transformer model architecture
train/value_head.py         - Reward computation
controller_quantization.py   - Action space discretization
```

---

**End of Architecture Guide**

This document provides a comprehensive visual guide to the PPO training system.
For questions or clarifications, refer to the specific files and line numbers referenced.
