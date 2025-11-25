# PPO Data Flow Architecture

This document details the architecture and data flow of the Proximal Policy Optimization (PPO) implementation in Nano-Melee. It illustrates how data moves from the emulator through the distributed worker system to the GPU for inference and training.

## 1. High-Level System Architecture

The system is designed to decouple **simulation** (CPU-intensive, slow) from **inference/training** (GPU-intensive, fast).

```ascii
+-----------------------------------------------------------------------+
|                       Main Process (train_ppo.py)                     |
|                                                                       |
|  +---------------------+      +------------------------------------+  |
|  |  TrajectorySlicer   |<---->|       InferenceCoordinator         |  |
|  | (Data Collection &  |      |      (Batching & GPU Mgmt)         |  |
|  |  GAE Calculation)   |      +------------------+-----------------+  |
|  +----------+----------+                         |                    |
|             ^                                    v                    |
|             |                          +---------+---------+          |
|             |                          |    GPU Model      |          |
|             |                          | (Learner/Opponent)|          |
|             |                          +-------------------+          |
|             |                                                         |
+-------------|---------------------------------------------------------+
              |
      IPC (Multiprocessing Queues)
              |
+-------------v---------------------------------------------------------+
|                               Worker Pool                             |
|                                                                       |
| +----------------+  +----------------+      +----------------+        |
| | Simulation     |  | Simulation     |      | Simulation     |        |
| | Worker 0       |  | Worker 1       | ...  | Worker N       |        |
| | (CPU/Dolphin)  |  | (CPU/Dolphin)  |      | (CPU/Dolphin)  |        |
| +----------------+  +----------------+      +----------------+        |
+-----------------------------------------------------------------------+
```

---

## 2. The Simulation Loop (Data Collection)

This diagram shows the cycle of a single frame being processed.

**Key Files:**
*   `ppo/simulation_worker.py`: Worker logic.
*   `ppo/trajectory_slicer.py`: Central data router.
*   `ppo/inference_coordinator.py`: Inference server.

```ascii
   [EMULATOR (Dolphin)]              [WORKER PROCESS]                   [MAIN PROCESS]
           |                                |                                  |
    1. Game State -------------------> Extract Features                        |
       (Frame N)                            |                                  |
                                     Calc Reward                               |
                                            |                                  |
                                     2. Send Tuple ---------------------> [State Queue]
                                     (id, feats, rew, done)                    |
                                                                               |
                                                                      3. TrajectorySlicer
                                                                         Dequeues Batch
                                                                               |
                                                                               v
                                                                      4. InferenceCoordinator
                                                                         Batches Inputs
                                                                               |
                                                                               v
                                                                           [GPU Model]
                                                                         (Forward Pass)
                                                                               |
                                                                               v
                                                                        Returns:
                                                                        - Action Logits
                                                                        - Sampled Action
                                                                        - Value V(s)
                                                                        - Log Prob
                                                                               |
                                                                      5. TrajectorySlicer
                                                                         Receives Action
                                                                               |
    7. Execute Action <------------------ 6. Send Action <--------------- [Action Queue]
       (Frame N+1)                    (Converted to Primitives)
```

### Detailed Data Structures

*   **Features**: Tensor representing the game state (positions, velocities, etc.).
*   **StepRecord**: Created by `InferenceCoordinator`, stored in `TrajectorySlicer`.
    ```python
    @dataclass
    class StepRecord:
        state: torch.Tensor          # The input features
        action_logits: Dict          # Raw model output
        action_taken: Dict           # Sampled action indices
        log_prob: torch.Tensor       # Log probability of action
        value: torch.Tensor          # Critic's value estimate V(s)
        reward: float                # Reward received *after* previous action
    ```

---

## 3. Trajectory Processing & GAE

Once a "rollout" (e.g., 5000 frames) is complete, the collected steps are processed to calculate **Generalized Advantage Estimation (GAE)** and **Discounted Returns**.

**Key File:** `ppo/trajectory_slicer.py` -> `_compute_gae_with_bootstrap`

```ascii
Raw Trajectory (T steps)                 Calculations
+----------------------+
| Step 0: r0, V0       |  delta_0 = r0 + gamma * V1 - V0
+----------------------+  gae_0   = delta_0 + gamma * lambda * gae_1
| Step 1: r1, V1       |
+----------------------+  delta_1 = r1 + gamma * V2 - V1
| Step 2: r2, V2       |  gae_1   = delta_1 + gamma * lambda * gae_2
+----------------------+
| ...                  |
+----------------------+
| Step T: rT, VT       |  delta_T = rT + gamma * V_bootstrap - VT
+----------------------+  gae_T   = delta_T  (last step)

       ||
       \/

Processed Trajectory
+--------------------------------------------------+
| Step 0: State, Action, Old_LogProb, Adv_0, Ret_0 |  <-- Ret_0 = Adv_0 + V0
+--------------------------------------------------+
| Step 1: State, Action, Old_LogProb, Adv_1, Ret_1 |
+--------------------------------------------------+
| ...                                              |
+--------------------------------------------------+
```

**Formulas:**
*   **Delta (TD Error):** $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
*   **GAE (Advantage):** $A_t = \delta_t + (\gamma \lambda) A_{t+1}$
*   **Return (Target):** $R_t = A_t + V(s_t)$

---

## 4. Training Batch Preparation

The processed trajectories are sliced into overlapping windows (sequences) for the Transformer model.

**Key Method:** `trajectory_slicer.prepare_training_data`

```ascii
Trajectory (Length 1000)
[s0, s1, s2, s3, s4, s5, s6, s7, ...]

Sequence Length = 4
Stride = 2

      Window 1: [s0, s1, s2, s3]
            Window 2: [s2, s3, s4, s5]
                  Window 3: [s4, s5, s6, s7]

      ||
      \/

Batch Tensor (Shape: [BatchSize, SeqLen, FeatureDim])
+-----------------------------+
| Window 1 Data (State, Adv)  |
+-----------------------------+
| Window 2 Data (State, Adv)  |
+-----------------------------+
| Window 3 Data (State, Adv)  |
+-----------------------------+
            ...
```

---

## 5. The Training Step (PPO Loss)

This occurs in the Main Process on the GPU.

**Key File:** `ppo/ppo_loss.py`

```ascii
      [Batch of Windows]
(States, Actions, Old_LogProbs, Advantages, Returns)
              |
              |
    +---------+---------+
    |                   |
    v                   v
 [Model]           [Loss Function] <-----------------------+
 (Forward)                ^                                |
    |                     |                                |
    | New Logits          |                                |
    | New Values          |                                |
    v                     |                                |
+-------+                 |                                |
| PPO   | ----------------+                                |
| Loss  |                                                  |
| Calc  | <------------------------------------------------+
+-------+

Components:
1. Ratio = exp(New_LogProb - Old_LogProb)
2. Surr1 = Ratio * Advantage
3. Surr2 = clamp(Ratio, 1-eps, 1+eps) * Advantage
4. Policy Loss = -min(Surr1, Surr2)
5. Value Loss = MSE(New_Value, Return)
6. Entropy = -sum(probs * log_probs)

Total Loss = Policy_Loss + (0.5 * Value_Loss) - (0.01 * Entropy)
```

## 6. Summary of Data Movement

1.  **CPU (Worker)**: Raw Game State -> Features (Numpy)
2.  **IPC**: Features -> Main Process
3.  **GPU (Main)**: Features -> Model Inference -> Actions/Values
4.  **IPC**: Actions -> Worker
5.  **CPU (Main)**: Trajectory Accumulation -> GAE Calculation -> Windowing
6.  **GPU (Main)**: Batch of Windows -> Model Training -> Backprop

## 7. Hardware Utilization

*   **CPU:**
    *   **Heavy Load:** Running multiple instances of Dolphin (one per worker).
    *   **Medium Load:** Data marshaling, queue management, GAE calculation (vectorized in PyTorch CPU).
*   **GPU:**
    *   **Inference:** Frequent small batches during rollout collection.
    *   **Training:** Large batched forward/backward passes during the optimization phase.
*   **Memory (RAM):** Stores the active rollouts and trajectories.
*   **VRAM:** Stores the model weights and current training batch.
