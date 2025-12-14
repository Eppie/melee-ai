"""Coordinator: GPU process managing inference and PPO training."""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.distributions import Categorical

from column_map import ColumnMap
from config import Config
from model.nano_gpt import GPT
from train.batch_utils import build_model_inputs

from .arena_shard import shard_main
from .config import PPOConfig
from .ipc import Barrier, MessageType, ShardPipe, create_shard_pipe_pair
from .opponent import Matchmaker, OpponentPool
from .shared_memory import (
    ActionData,
    PinnedStagingBuffer,
    SharedMemorySlab,
    compute_rope_indices,
)


class Coordinator:
    """
    GPU process managing inference and PPO training.

    Responsibilities:
    - Load policy model, spawn 12 S8 processes
    - Maintain GPU ring buffer and pinned staging buffer
    - Wait for all S8s ready → gather features → async H2D copy
    - Batched forward pass (96 envs, 256 context) with RoPE indexing
    - Sample actions from logits, compute log probs and values
    - Scatter actions to S8 slabs, signal ready
    - Collect completed rollouts (every 1024 frames)
    - Run PPO training when enough rollouts accumulated
    - Manage opponent pool (80% historical, 20% self-play)
    """

    def __init__(self, config: Config, ppo_config: PPOConfig):
        self.config = config
        self.ppo_config = ppo_config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print(f"[CRD] Initializing Coordinator")
        print(f"[CRD] Device: {self.device}")
        print(f"[CRD] Total envs: {ppo_config.total_envs}")

        # Column map for feature indexing
        from schema import get_feature_names, get_target_names

        feature_names = get_feature_names()
        target_names = get_target_names()
        self.column_map = ColumnMap(feature_names, target_names)
        self.feature_dim = len(feature_names)
        if self.ppo_config.feature_dim != self.feature_dim:
            # Keep config/state aligned with schema-derived dimension.
            self.ppo_config.feature_dim = self.feature_dim
        print(f"[CRD] Feature dim: {self.feature_dim}")

        # Spawn S8 processes FIRST (before CUDA allocation)
        self.shards: List[mp.Process] = []
        self.pipes: List[ShardPipe] = []
        self.shard_slabs: List[SharedMemorySlab] = []
        self._spawn_shards()

        # NOW initialize CUDA and model
        print(f"[CRD] Loading policy model from {ppo_config.init_checkpoint}")
        self.policy = self._load_model(ppo_config.init_checkpoint)
        self.policy.to(self.device)
        self.policy.eval()

        # GPU ring buffer
        print(f"[CRD] Allocating GPU ring buffer")
        self.X_gpu = torch.zeros(
            (ppo_config.total_envs, ppo_config.context_length, self.feature_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )

        # Pinned staging buffer for H2D transfer
        self.staging = PinnedStagingBuffer(
            num_envs=ppo_config.total_envs,
            feature_dim=self.feature_dim,
        )

        # Barrier for synchronization
        self.barrier = Barrier(self.pipes, timeout=10.0)

        # Ring position tracking
        self.t_mod = 0
        self.step_id = 0

        # Initialize t_mod in all slabs (source of truth for workers)
        for slab in self.shard_slabs:
            slab.t_mod = self.t_mod

        # Training state
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=ppo_config.lr,
            weight_decay=ppo_config.weight_decay,
        )

        # Rollout collection
        self.rollouts_pending: List = []  # Rollouts being collected
        self.rollouts_ready: List = []  # Completed rollouts ready for training

        # Training metrics
        self.training_steps = 0  # Number of PPO training updates
        self.total_frames = 0  # Total environment frames processed

        # Opponent pool for self-play (80% historical, 20% current policy)
        print(f"[CRD] Initializing opponent pool")
        checkpoint_dir = ppo_config.checkpoint_dir  # Directory for PPO checkpoints
        self.opponent_pool = OpponentPool(
            checkpoint_dir=checkpoint_dir,
            historical_ratio=ppo_config.opponent_sample_prob,
            max_pool_size=ppo_config.opponent_pool_size,
            refresh_interval=ppo_config.pool_refresh_interval,
        )
        self.matchmaker = Matchmaker(
            opponent_pool=self.opponent_pool,
            num_envs=ppo_config.total_envs,
            device=self.device,
        )

        print(f"[CRD] Coordinator initialized successfully")

    def _load_model(self, checkpoint_path: Path) -> GPT:
        """Load GPT model from checkpoint."""
        checkpoint_path = Path(checkpoint_path).expanduser()

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Extract config
        if "config" in ckpt:
            model_config = ckpt["config"]
            if isinstance(model_config, dict):
                from config import Config as ConfigClass

                config = ConfigClass.model_validate(model_config)
            else:
                config = model_config
        else:
            # Use current config
            config = self.config

        # Create model
        model = GPT(config)

        # Load weights
        if "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Handle torch.compile prefix mismatch
        from utils import match_state_dict_keys

        state_dict = match_state_dict_keys(state_dict, model)

        model.load_state_dict(state_dict, strict=False)

        return model

    def _spawn_shards(self):
        """Spawn S8 shard processes via spawn() method."""
        print(f"[CRD] Spawning {self.ppo_config.num_shards} S8 shards")

        # Ensure spawn method is set
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # Already set

        for shard_id in range(self.ppo_config.num_shards):
            # Create pipe for communication
            coordinator_pipe, shard_pipe = create_shard_pipe_pair(shard_id)

            # Spawn process
            p = mp.get_context("spawn").Process(
                target=shard_main,
                args=(shard_id, shard_pipe, self.ppo_config, self.column_map),
                name=f"S8-{shard_id}",
            )
            p.start()

            self.shards.append(p)
            self.pipes.append(coordinator_pipe)

        # Wait for shards to create shared memory (retry to avoid races)
        import time

        for shard_id in range(self.ppo_config.num_shards):
            slab = None
            for attempt in range(50):  # 5s total at 0.1s per attempt
                try:
                    slab = SharedMemorySlab(
                        shard_id=shard_id,
                        envs_per_shard=self.ppo_config.envs_per_shard,
                        context_length=self.ppo_config.context_length,
                        feature_dim=self.feature_dim,
                        create=False,  # Attach to existing
                    )
                    break
                except FileNotFoundError:
                    if attempt == 0:
                        print(
                            f"[CRD] Waiting for shard {shard_id} shared memory to appear..."
                        )
                    time.sleep(0.1)

            if slab is None:
                raise FileNotFoundError(
                    f"Shared memory slab for shard {shard_id} not available after retrying"
                )

            self.shard_slabs.append(slab)

        print(f"[CRD] All {len(self.shards)} shards spawned successfully")

    def _gather_features_to_staging(self):
        """Gather features from all S8 slabs into staging buffer."""
        self.staging.copy_from_slabs(self.shard_slabs, self.t_mod)

    def _async_h2d_copy(self):
        """Async H2D copy: staging → GPU ring (only new column)."""
        # Copy only the new column at position t_mod
        self.X_gpu[:, self.t_mod, :] = self.staging.staging[:, 0, :].to(
            device=self.device,
            dtype=torch.bfloat16,
            non_blocking=True,
        )

    def _forward_policy(
        self,
        policy: GPT,
        features: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Run forward pass on features with given policy.

        Args:
            policy: Policy network to use
            features: Feature tensor [B, T, F]

        Returns:
            Tuple of (main_idx, c_idx, shoulder_idx, buttons, logps, values)
        """
        B, T, F = features.shape

        # Diagnostic logging (every 100 steps, all envs)
        if hasattr(self, "step_id") and self.step_id % 100 == 0 and B > 0:
            # Get last timestep for all envs
            last_frame_features = features[:, -1]  # [B, F]
            print(
                f"\n[CRD] ═══ Feature Diagnostics (step {self.step_id}, {B} envs, last frame) ═══"
            )
            print(f"  Feature shape: {features.shape} (B={B}, T={T}, F={F})")

            # Get feature names
            from schema import get_feature_names
            import numpy as np

            feature_names = get_feature_names()

            # Convert to float numpy for statistics
            features_np = last_frame_features.cpu().float().numpy()  # [B, F]

            # Print per-feature statistics (min/median/max across batch)
            print(f"\n  Per-feature statistics across {B} envs:")
            for i, name in enumerate(feature_names):
                feat_values = features_np[:, i]
                min_val = feat_values.min()
                median_val = np.median(feat_values)
                max_val = feat_values.max()

                # Add markers for categorical features
                if i == self.column_map.stage_idx:
                    print(
                        f"    [{i:2d}] {name:40s} min={min_val:7.3f} med={median_val:7.3f} max={max_val:7.3f}  ← STAGE"
                    )
                elif i == self.column_map.ego_char_idx:
                    print(
                        f"    [{i:2d}] {name:40s} min={min_val:7.3f} med={median_val:7.3f} max={max_val:7.3f}  ← EGO_CHAR"
                    )
                elif i == self.column_map.opp_char_idx:
                    print(
                        f"    [{i:2d}] {name:40s} min={min_val:7.3f} med={median_val:7.3f} max={max_val:7.3f}  ← OPP_CHAR"
                    )
                elif i == self.column_map.ego_action_idx:
                    print(
                        f"    [{i:2d}] {name:40s} min={min_val:7.3f} med={median_val:7.3f} max={max_val:7.3f}  ← EGO_ACTION"
                    )
                elif i == self.column_map.opp_action_idx:
                    print(
                        f"    [{i:2d}] {name:40s} min={min_val:7.3f} med={median_val:7.3f} max={max_val:7.3f}  ← OPP_ACTION"
                    )
                else:
                    print(
                        f"    [{i:2d}] {name:40s} min={min_val:7.3f} med={median_val:7.3f} max={max_val:7.3f}"
                    )
            print(f"[CRD] ═══ End Feature Diagnostics ═══\n")

        # Build model inputs
        batch_td = build_model_inputs(features.float(), self.column_map)

        # Forward pass
        outputs = policy(batch_td)

        # Extract logits for each head (only last frame)
        main_logits = outputs["main_stick"][:, -1, :]  # [B, 64]
        c_logits = outputs["c_stick"][:, -1, :]  # [B, 9]
        shoulder_logits = outputs["shoulder"][:, -1, :]  # [B, 5]
        button_logits = outputs["buttons"][:, -1, :]  # [B, 5]
        value = outputs["value"][:, -1, 0]  # [B]

        # Sample actions
        main_dist = Categorical(logits=main_logits)
        main_idx = main_dist.sample()
        main_logp = main_dist.log_prob(main_idx)

        c_dist = Categorical(logits=c_logits)
        c_idx = c_dist.sample()
        c_logp = c_dist.log_prob(c_idx)

        shoulder_dist = Categorical(logits=shoulder_logits)
        shoulder_idx = shoulder_dist.sample()
        shoulder_logp = shoulder_dist.log_prob(shoulder_idx)

        button_probs = torch.sigmoid(button_logits)
        button_samples = torch.bernoulli(button_probs).bool()
        button_logp = torch.log(
            torch.where(button_samples, button_probs, 1 - button_probs)
        ).sum(dim=1)

        # Total log probability
        total_logp = main_logp + c_logp + shoulder_logp + button_logp

        # Diagnostic logging (first env only, every 100 steps)
        if hasattr(self, "step_id") and self.step_id % 100 == 0:
            env0_idx = 0
            print(f"[CRD] Action Diagnostics (step {self.step_id}, env 0):")
            print(
                f"  Main stick: idx={main_idx[env0_idx].item()} (logit range: {main_logits[env0_idx].min():.2f} to {main_logits[env0_idx].max():.2f})"
            )
            print(
                f"  C-stick: idx={c_idx[env0_idx].item()} (logit range: {c_logits[env0_idx].min():.2f} to {c_logits[env0_idx].max():.2f})"
            )
            print(f"  Shoulder: idx={shoulder_idx[env0_idx].item()}")
            print(f"  Button probs: {button_probs[env0_idx].tolist()}")
            print(
                f"  Buttons sampled: {button_samples[env0_idx].tolist()} ({button_samples[env0_idx].sum().item()}/5)"
            )
            print(f"  Value estimate: {value[env0_idx].item():.3f}")

        return main_idx, c_idx, shoulder_idx, button_samples, total_logp, value

    def _batch_forward(self) -> tuple:
        """
        Run batched forward pass on all envs with RoPE adjustment.

        Computes both ego and opponent actions:
        - Ego: Always use current policy
        - Opponent: 80% historical (from pool), 20% self-play (current policy)

        Returns:
            Tuple of (ego_actions, opp_actions) where each is:
            (main_indices, c_indices, shoulder_indices, buttons, logps, values)
            All numpy arrays with shape [total_envs]
        """
        with torch.inference_mode():
            # Rotate ring buffer so newest frame is at position -1 (expected by model)
            # After writing to t_mod, the oldest frame is at (t_mod + 1) % T
            # We want oldest at position 0, newest at position -1
            rotated_X = torch.roll(self.X_gpu, shifts=-(self.t_mod + 1), dims=1)

            # 1. Compute ego actions for all envs (single batch)
            ego_actions = self._forward_policy(self.policy, rotated_X)

            # 2. Compute opponent actions
            # Group environments by opponent policy
            from collections import defaultdict

            opponent_groups = defaultdict(list)  # Maps opponent_model → [env_ids]

            for env_id in range(self.ppo_config.total_envs):
                opponent_model = self.matchmaker.get_opponent_model(env_id, self.policy)
                # Use id() for hashable key (None for self-play)
                key = id(opponent_model) if opponent_model is not None else None
                opponent_groups[key].append((env_id, opponent_model))

            # Initialize opponent action tensors (will be filled per-group)
            opp_main = torch.zeros(
                self.ppo_config.total_envs, dtype=torch.long, device=self.device
            )
            opp_c = torch.zeros(
                self.ppo_config.total_envs, dtype=torch.long, device=self.device
            )
            opp_shoulder = torch.zeros(
                self.ppo_config.total_envs, dtype=torch.long, device=self.device
            )
            opp_buttons = torch.zeros(
                self.ppo_config.total_envs, 5, dtype=torch.bool, device=self.device
            )
            opp_logps = torch.zeros(self.ppo_config.total_envs, device=self.device)
            opp_values = torch.zeros(self.ppo_config.total_envs, device=self.device)

            # 3. Compute actions for each opponent group
            for key, env_list in opponent_groups.items():
                env_ids = [e[0] for e in env_list]
                opponent_model = env_list[0][1]  # Same for all in group

                # Extract features for this group (use rotated buffer)
                group_features = rotated_X[
                    env_ids
                ]  # [len(env_ids), context_length, feature_dim]

                if opponent_model is None:
                    # Self-play: use ego actions
                    for i, env_id in enumerate(env_ids):
                        opp_main[env_id] = ego_actions[0][env_id]
                        opp_c[env_id] = ego_actions[1][env_id]
                        opp_shoulder[env_id] = ego_actions[2][env_id]
                        opp_buttons[env_id] = ego_actions[3][env_id]
                        opp_logps[env_id] = ego_actions[4][env_id]
                        opp_values[env_id] = ego_actions[5][env_id]
                else:
                    # Historical opponent: run forward pass
                    group_actions = self._forward_policy(opponent_model, group_features)

                    # Scatter results back to full arrays
                    for i, env_id in enumerate(env_ids):
                        opp_main[env_id] = group_actions[0][i]
                        opp_c[env_id] = group_actions[1][i]
                        opp_shoulder[env_id] = group_actions[2][i]
                        opp_buttons[env_id] = group_actions[3][i]
                        opp_logps[env_id] = group_actions[4][i]
                        opp_values[env_id] = group_actions[5][i]

        # Convert to numpy
        ego_np = tuple(x.cpu().numpy() for x in ego_actions)
        opp_np = (
            opp_main.cpu().numpy(),
            opp_c.cpu().numpy(),
            opp_shoulder.cpu().numpy(),
            opp_buttons.cpu().numpy(),
            opp_logps.cpu().numpy(),
            opp_values.cpu().numpy(),
        )

        return ego_np, opp_np

    def _scatter_actions_to_shards(
        self,
        ego_actions: tuple,
        opp_actions: tuple,
    ):
        """
        Scatter both ego and opponent actions to S8 action slots.

        Args:
            ego_actions: Tuple of (main, c, shoulder, buttons, logps, values)
            opp_actions: Tuple of (main, c, shoulder, buttons, logps, values)
        """
        ego_main, ego_c, ego_shoulder, ego_buttons, ego_logps, ego_values = ego_actions
        opp_main, opp_c, opp_shoulder, opp_buttons, opp_logps, opp_values = opp_actions

        for shard_idx, slab in enumerate(self.shard_slabs):
            shard_offset = shard_idx * self.ppo_config.envs_per_shard

            for env_id in range(self.ppo_config.envs_per_shard):
                global_env_id = shard_offset + env_id

                # Ego action
                ego_action = ActionData(
                    main_idx=int(ego_main[global_env_id]),
                    c_idx=int(ego_c[global_env_id]),
                    shoulder_idx=int(ego_shoulder[global_env_id]),
                    buttons=ego_buttons[global_env_id],
                    logp=float(ego_logps[global_env_id]),
                    value=float(ego_values[global_env_id]),
                )

                # Opponent action
                opp_action = ActionData(
                    main_idx=int(opp_main[global_env_id]),
                    c_idx=int(opp_c[global_env_id]),
                    shoulder_idx=int(opp_shoulder[global_env_id]),
                    buttons=opp_buttons[global_env_id],
                    logp=float(opp_logps[global_env_id]),
                    value=float(opp_values[global_env_id]),
                )

                # Write both actions (ENV will read both)
                slab.write_action(env_id, ego_action, is_ego=True)
                slab.write_action(env_id, opp_action, is_ego=False)

    def _signal_shards_actions_ready(self):
        """Signal all S8s that actions are ready."""
        self.barrier.signal_all_actions_ready(self.step_id)

    def run(self):
        """Main CRD loop."""
        print(
            f"[CRD] Waiting for all ENVs to finish menu navigation and enter matches..."
        )
        print(f"[CRD] This may take 30-60 seconds...")

        # Wait for first READY signal with extended timeout (menu navigation)
        try:
            _, _ = self.barrier.wait_all_ready(
                self.step_id, timeout=120.0
            )  # 2 minute timeout
            print(f"[CRD] All ENVs ready! Starting main inference loop")
        except TimeoutError:
            print(f"[CRD] ERROR: ENVs failed to enter matches within 2 minutes")
            print(f"[CRD] Check ENV logs for menu navigation issues")
            raise

        self.start_time = time.time()

        try:
            while True:
                # NOTE: Shards already sent READY before entering this loop
                # So we skip the wait on step 0
                if self.step_id > 0:
                    # 1. Wait for all S8s to signal READY
                    # This may also receive ROLLOUT_COMPLETE messages that we process immediately
                    _, queued_rollout_messages = self.barrier.wait_all_ready(
                        self.step_id
                    )

                    # Process any rollout messages that arrived during barrier wait
                    for msg in queued_rollout_messages:
                        rollouts = msg.payload.get("rollouts", [])
                        if rollouts:
                            self.rollouts_ready.extend(rollouts)
                            print(
                                f"[CRD] Received {len(rollouts)} rollouts from shard {msg.shard_id} "
                                f"(total ready: {len(self.rollouts_ready)})"
                            )
                            # Reassign opponents when rollouts complete
                            self.matchmaker.reassign_all()
                            self.matchmaker.clear_loaded_opponents()

                # 2. Gather features from all S8 slabs → staging buffer
                self._gather_features_to_staging()

                # 3. Async H2D copy: staging → GPU ring (only new column)
                self._async_h2d_copy()

                # 4. Run batched forward pass (ego + opponent)
                ego_actions, opp_actions = self._batch_forward()

                # 5. Scatter actions to S8 action slots
                self._scatter_actions_to_shards(ego_actions, opp_actions)

                # 6. Signal all S8s that actions are ready
                self._signal_shards_actions_ready()

                # 7. Update ring position
                self.t_mod = (self.t_mod + 1) % self.ppo_config.context_length
                self.step_id += 1

                # 8. Write t_mod to all slabs (source of truth for workers)
                for slab in self.shard_slabs:
                    slab.t_mod = self.t_mod
                self.total_frames += (
                    self.ppo_config.total_envs
                )  # Each step processes all envs

                # Periodic logging and health check
                if self.step_id % 100 == 0:
                    ego_main_idx = ego_actions[0]  # main stick indices
                    ego_c_idx = ego_actions[1]  # c stick indices
                    ego_buttons = ego_actions[3]  # button presses
                    ego_values = ego_actions[5]  # values are at index 5
                    fps = (
                        self.total_frames / (time.time() - self.start_time)
                        if hasattr(self, "start_time")
                        else 0
                    )

                    # Sample env 0 to check game state (use staging buffer for raw features)
                    env0_raw = self.staging.staging[0, 0, :].cpu().numpy()

                    # Use column_map to extract meaningful features
                    # Note: column_map has x_* for input features
                    try:
                        # Find indices for key features
                        from schema import get_feature_names

                        feature_names = get_feature_names()

                        # Create a feature dict for easier access
                        feat_dict = {
                            name: env0_raw[i] for i, name in enumerate(feature_names)
                        }

                        # Extract key game state info
                        # Note: Features are normalized, need to reverse scaling
                        ego_pos_x = (
                            feat_dict.get("p1_position_x", 0.0) * 20.0
                        )  # scaled by 1/20
                        ego_pos_y = feat_dict.get("p1_position_y", 0.0) * 20.0
                        ego_pct = (
                            feat_dict.get("p1_percent", 0.0) * 100.0
                        )  # scaled by 1/100
                        ego_stock = (
                            feat_dict.get("p1_stock", 0.0) * 4.0
                        )  # scaled by 1/4
                        opp_pos_x = feat_dict.get("p2_position_x", 0.0) * 20.0
                        opp_pos_y = feat_dict.get("p2_position_y", 0.0) * 20.0
                        opp_pct = feat_dict.get("p2_percent", 0.0) * 100.0
                        opp_stock = feat_dict.get("p2_stock", 0.0) * 4.0

                        gamestate_str = (
                            f"Ego: pos=({ego_pos_x:.1f},{ego_pos_y:.1f}) "
                            f"pct={ego_pct:.0f}% stock={ego_stock:.0f} | "
                            f"Opp: pos=({opp_pos_x:.1f},{opp_pos_y:.1f}) "
                            f"pct={opp_pct:.0f}% stock={opp_stock:.0f}"
                        )
                    except Exception as e:
                        gamestate_str = f"Error extracting features: {e}"

                    print(
                        f"[CRD] Step {self.step_id}, t_mod={self.t_mod}, "
                        f"frames={self.total_frames:,}, "
                        f"fps={fps:.1f}, "
                        f"ego_values mean={ego_values.mean():.3f}, "
                        f"main_stick unique={len(set(ego_main_idx))}, "
                        f"c_stick unique={len(set(ego_c_idx))}, "
                        f"buttons any={ego_buttons.any()}"
                    )
                    print(f"[CRD]   ENV0: {gamestate_str}")

                # Health check every 1000 steps
                if self.step_id % 1000 == 0:
                    self._health_check()

                # Check for completed rollouts from shards (non-blocking)
                self._collect_completed_rollouts()

                # Launch PPO training when enough rollouts accumulated
                if len(self.rollouts_ready) >= self.ppo_config.rollouts_per_batch:
                    self._run_ppo_training()

        except KeyboardInterrupt:
            print(f"[CRD] Interrupted by user")
        except Exception as e:
            print(f"[CRD] Error in main loop: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self._cleanup()

    def _health_check(self):
        """Check health of all shard processes.

        If shards die, remove them from the barrier to continue with fewer workers.
        Terminate training if too many shards are lost (>50%).
        """
        dead_shard_indices = []
        for i, shard in enumerate(self.shards):
            if not shard.is_alive():
                dead_shard_indices.append(i)

        if dead_shard_indices:
            print(f"[CRD] WARNING: Dead shards detected: {dead_shard_indices}")

            # Remove dead shards from barrier
            remaining_pipes = []
            for i, pipe in enumerate(self.barrier.pipes):
                if i not in dead_shard_indices:
                    remaining_pipes.append(pipe)
                else:
                    # Close pipe to dead shard
                    try:
                        pipe.close()
                    except Exception as e:
                        print(f"[CRD] Error closing pipe to dead shard {i}: {e}")

            # Update barrier with remaining pipes
            self.barrier.pipes = remaining_pipes
            self.barrier.num_shards = len(remaining_pipes)

            # Check if we have enough shards left to continue
            survival_rate = len(remaining_pipes) / len(self.shards)
            print(
                f"[CRD] Continuing with {len(remaining_pipes)}/{len(self.shards)} "
                f"shards ({survival_rate*100:.1f}% alive)"
            )

            if survival_rate < 0.5:
                raise RuntimeError(
                    f"Too many shards died ({len(dead_shard_indices)}/{len(self.shards)}). "
                    "Terminating training for safety."
                )
        # Don't log when everything is healthy (reduces verbosity)

    def _save_checkpoint(self):
        """Save checkpoint for opponent pool and resumption."""
        checkpoint_dir = self.ppo_config.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create checkpoint filename
        checkpoint_path = checkpoint_dir / f"checkpoint_{self.training_steps:06d}.pt"

        # Save checkpoint (using same format as imitation learning for compatibility)
        checkpoint = {
            "model": self.policy.state_dict(),  # Match imitation learning format
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.model_dump()
            if hasattr(self.config, "model_dump")
            else self.config,
            "ppo_config": self.ppo_config.model_dump()
            if hasattr(self.ppo_config, "model_dump")
            else self.ppo_config,
            # PPO-specific fields
            "training_steps": self.training_steps,
            "total_frames": self.total_frames,
            "step_id": self.step_id,
            # Compatibility fields for inference engine
            "epoch": 0,
            "resume_epoch": 0,
            "resume_iter": 0,
            "global_step": self.training_steps,
        }

        torch.save(checkpoint, checkpoint_path)
        print(
            f"[CRD] Saved checkpoint: {checkpoint_path.name} (step {self.training_steps})"
        )

        # Prune old checkpoints
        self._prune_old_checkpoints()

    def _prune_old_checkpoints(self):
        """Keep only the most recent N checkpoints."""
        checkpoint_dir = self.ppo_config.checkpoint_dir
        if not checkpoint_dir.exists():
            return

        # Find all checkpoints
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_*.pt"),
            key=lambda p: p.stat().st_mtime,
        )

        # Keep only max_pool_size + 5 (buffer for opponent pool)
        max_keep = self.ppo_config.opponent_pool_size + 5
        if len(checkpoints) > max_keep:
            to_delete = checkpoints[:-max_keep]
            for ckpt in to_delete:
                ckpt.unlink()
            # Only log count, not individual files (reduces verbosity)
            if len(to_delete) > 0:
                print(f"[CRD] Pruned {len(to_delete)} old checkpoints")

    def _collect_completed_rollouts(self):
        """
        Collect completed rollouts from S8 shards (non-blocking).

        Checks all shard pipes for ROLLOUT_COMPLETE messages and
        adds rollouts to the ready queue for PPO training.
        """
        # Non-blocking: check all shard pipes for rollout messages
        for pipe in self.barrier.pipes:
            if pipe.poll():  # Check if message available
                try:
                    msg = pipe.recv(timeout=0.001)  # Non-blocking read
                    if msg.msg_type == MessageType.ROLLOUT_COMPLETE:
                        rollouts = msg.payload.get("rollouts", [])
                        if rollouts:
                            self.rollouts_ready.extend(rollouts)
                            print(
                                f"[CRD] Received {len(rollouts)} rollouts from shard {msg.shard_id} "
                                f"(total ready: {len(self.rollouts_ready)})"
                            )

                            # Reassign opponents when rollouts complete
                            self.matchmaker.reassign_all()
                            self.matchmaker.clear_loaded_opponents()
                except TimeoutError:
                    pass  # No message available

    def _run_ppo_training(self):
        """Execute PPO training on accumulated rollouts."""
        print(f"[CRD] Starting PPO training with {len(self.rollouts_ready)} rollouts")
        print(
            f"[CRD] Training step {self.training_steps}, Total frames: {self.total_frames:,}"
        )
        print(f"[CRD] NOTE: Shards will pause during training (may take 30-60 seconds)")

        # Force garbage collection before training
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.policy.train()

        # Track metrics
        all_losses = []
        all_policy_losses = []
        all_value_losses = []
        all_entropy_losses = []
        all_ratios = []
        all_kl_divs = []

        try:
            # Create windowed batches from rollouts
            from .rollout import create_windowed_batches

            batches = create_windowed_batches(
                rollouts=self.rollouts_ready,
                context_length=self.ppo_config.context_length,
                batch_size=self.ppo_config.batch_size,
            )

            num_batches = len(batches)
            print(f"[CRD] Created {num_batches} windowed batches")
            print(
                f"[CRD] Training: {self.ppo_config.ppo_epochs} epochs × {num_batches} batches = {self.ppo_config.ppo_epochs * num_batches} total updates"
            )

            # Estimate memory usage
            batch_memory_mb = (
                self.ppo_config.batch_size
                * self.ppo_config.context_length
                * (self.feature_dim * 4 + 40 + 16)
                / 1024
                / 1024
            )
            total_memory_mb = batch_memory_mb * num_batches
            print(
                f"[CRD] Estimated batch memory: {total_memory_mb:.0f} MB ({batch_memory_mb:.1f} MB per batch)"
            )

            if total_memory_mb > 2000:
                print(
                    f"[CRD] WARNING: High memory usage! Consider reducing --rollouts-per-batch"
                )

            # PPO epochs
            for epoch in range(self.ppo_config.ppo_epochs):
                epoch_start = __import__("time").time()
                for batch_idx, batch in enumerate(batches):
                    # Move batch to device
                    features = batch["features"].to(self.device)
                    old_logp = batch["old_logp"].to(self.device)
                    advantages = batch["advantages"].to(self.device)
                    returns = batch["returns"].to(self.device)
                    actions = batch["actions"]  # Stays on CPU (structured array)

                    # Convert actions to dict format
                    # Actions are [B, T] but we only need the final timestep for PPO
                    # Use np.ascontiguousarray to fix stride alignment issues with structured arrays
                    action_dict = {
                        "main_idx": torch.from_numpy(
                            np.ascontiguousarray(actions["main_idx"][:, -1])
                        ).to(self.device),
                        "c_idx": torch.from_numpy(
                            np.ascontiguousarray(actions["c_idx"][:, -1])
                        ).to(self.device),
                        "shoulder_idx": torch.from_numpy(
                            np.ascontiguousarray(actions["shoulder_idx"][:, -1])
                        ).to(self.device),
                        "buttons": torch.from_numpy(
                            np.ascontiguousarray(actions["buttons"][:, -1, :])
                        ).to(self.device),
                    }

                    # Compute PPO loss
                    from .ppo_loss import compute_ppo_loss

                    loss_dict = compute_ppo_loss(
                        policy=self.policy,
                        batch_features=features,
                        batch_actions=action_dict,
                        old_logps=old_logp,
                        advantages=advantages,
                        returns=returns,
                        column_map=self.column_map,
                        clip_epsilon=self.ppo_config.clip_epsilon,
                        value_coef=self.ppo_config.value_coef,
                        entropy_coef=self.ppo_config.entropy_coef,
                    )

                    # Backward pass
                    self.optimizer.zero_grad()
                    loss_dict["total"].backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(),
                        self.ppo_config.grad_clip,
                    )

                    self.optimizer.step()

                    # Track metrics (extract values before deleting tensors)
                    loss_total = loss_dict["total"].item()
                    loss_policy = loss_dict["policy"].item()
                    loss_value = loss_dict["value"].item()
                    loss_entropy = loss_dict["entropy"].item()
                    ratio_mean = loss_dict["ratio_mean"].item()
                    approx_kl = loss_dict["approx_kl"].item()

                    all_losses.append(loss_total)
                    all_policy_losses.append(loss_policy)
                    all_value_losses.append(loss_value)
                    all_entropy_losses.append(loss_entropy)
                    all_ratios.append(ratio_mean)
                    all_kl_divs.append(approx_kl)

                    # Periodic logging (before deleting)
                    log_interval = max(1, num_batches // 10)
                    if batch_idx % log_interval == 0:
                        progress_pct = 100.0 * batch_idx / num_batches

                        # Action distribution diagnostics (first batch only)
                        if batch_idx == 0 and epoch == 0:
                            button_count = (
                                action_dict["buttons"].float().sum(dim=1).mean().item()
                            )
                            # Button breakdown
                            button_names = ["A", "B", "X", "Z", "L"]
                            button_rates = [
                                action_dict["buttons"][:, i].float().mean().item()
                                for i in range(5)
                            ]

                            print(
                                f"\n[CRD] ═══ Training Batch Diagnostics (step {self.training_steps}) ═══"
                            )
                            print(f"[CRD] Batch Action Stats:")
                            print(f"  Mean buttons pressed: {button_count:.2f}/5")
                            print(
                                f"  Button rates: {', '.join(f'{name}={rate:.1%}' for name, rate in zip(button_names, button_rates))}"
                            )
                            print(
                                f"  Main stick idx range: {action_dict['main_idx'].min().item()} to {action_dict['main_idx'].max().item()}"
                            )
                            print(
                                f"  C-stick idx range: {action_dict['c_idx'].min().item()} to {action_dict['c_idx'].max().item()}"
                            )
                            print(
                                f"  Shoulder idx range: {action_dict['shoulder_idx'].min().item()} to {action_dict['shoulder_idx'].max().item()}"
                            )
                            # old_logp is [B, T] for windowed batches
                            if old_logp.dim() == 2:
                                print(
                                    f"  Old logp: mean={old_logp[:, -1].mean().item():.3f}, std={old_logp[:, -1].std().item():.3f}"
                                )
                            else:
                                print(
                                    f"  Old logp: mean={old_logp.mean().item():.3f}, std={old_logp.std().item():.3f}"
                                )
                            print(
                                f"  Advantages: mean={advantages.mean().item():.3f}, std={advantages.std().item():.3f}"
                            )
                            print(
                                f"  Returns: mean={returns.mean().item():.3f}, std={returns.std().item():.3f}"
                            )

                            # Feature statistics for training batch
                            # features is [B, T, F], get last timestep
                            batch_last_frame = features[:, -1, :]  # [B, F]
                            from schema import get_feature_names
                            import numpy as np

                            feature_names = get_feature_names()
                            features_np = batch_last_frame.cpu().float().numpy()

                            print(
                                f"\n[CRD] Batch Feature Stats (last timestep, {features_np.shape[0]} examples):"
                            )
                            for i, name in enumerate(feature_names):
                                feat_values = features_np[:, i]
                                min_val = feat_values.min()
                                median_val = np.median(feat_values)
                                max_val = feat_values.max()

                                # Only print categorical features and a few key continuous ones
                                if i == self.column_map.stage_idx:
                                    print(
                                        f"    [{i:2d}] {name:40s} min={min_val:7.3f} med={median_val:7.3f} max={max_val:7.3f}  ← STAGE"
                                    )
                                elif i == self.column_map.ego_char_idx:
                                    print(
                                        f"    [{i:2d}] {name:40s} min={min_val:7.3f} med={median_val:7.3f} max={max_val:7.3f}  ← EGO_CHAR"
                                    )
                                elif i == self.column_map.opp_char_idx:
                                    print(
                                        f"    [{i:2d}] {name:40s} min={min_val:7.3f} med={median_val:7.3f} max={max_val:7.3f}  ← OPP_CHAR"
                                    )
                                elif i == self.column_map.ego_action_idx:
                                    print(
                                        f"    [{i:2d}] {name:40s} min={min_val:7.3f} med={median_val:7.3f} max={max_val:7.3f}  ← EGO_ACTION"
                                    )
                                elif i == self.column_map.opp_action_idx:
                                    print(
                                        f"    [{i:2d}] {name:40s} min={min_val:7.3f} med={median_val:7.3f} max={max_val:7.3f}  ← OPP_ACTION"
                                    )

                            # Check for NaN/Inf in features
                            has_nan = torch.isnan(features).any().item()
                            has_inf = torch.isinf(features).any().item()
                            if has_nan or has_inf:
                                print(
                                    f"  ⚠️ WARNING: features contain NaN={has_nan}, Inf={has_inf}"
                                )
                            print(f"[CRD] ═══ End Training Batch Diagnostics ═══\n")

                        print(
                            f"[CRD] Epoch {epoch+1}/{self.ppo_config.ppo_epochs} "
                            f"[{progress_pct:5.1f}%] "
                            f"Batch {batch_idx:3d}/{num_batches} | "
                            f"Loss: {loss_total:.4f} "
                            f"(P:{loss_policy:.4f} "
                            f"V:{loss_value:.4f} "
                            f"E:{loss_entropy:.4f}) | "
                            f"Ratio: {ratio_mean:.3f}"
                        )

                    # Explicitly delete tensors to free memory
                    del features, old_logp, advantages, returns, action_dict, loss_dict

                # End of epoch summary
                epoch_time = __import__("time").time() - epoch_start
                epoch_losses = all_losses[-num_batches:]  # Last epoch's losses
                print(
                    f"[CRD] Epoch {epoch+1}/{self.ppo_config.ppo_epochs} complete: "
                    f"avg_loss={np.mean(epoch_losses):.4f}, "
                    f"time={epoch_time:.1f}s"
                )

                # Force garbage collection after each epoch
                import gc

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Print summary metrics
            print(f"[CRD] ===== Training Summary =====")
            print(
                f"[CRD]   Total Loss:    {np.mean(all_losses):.4f} ± {np.std(all_losses):.4f}"
            )
            print(f"[CRD]   Policy Loss:   {np.mean(all_policy_losses):.4f}")
            print(f"[CRD]   Value Loss:    {np.mean(all_value_losses):.4f}")
            print(f"[CRD]   Entropy Loss:  {np.mean(all_entropy_losses):.4f}")
            print(
                f"[CRD]   Mean Ratio:    {np.mean(all_ratios):.3f} (clip at {self.ppo_config.clip_epsilon})"
            )
            print(f"[CRD]   Approx KL:     {np.mean(all_kl_divs):.4f}")
            print(f"[CRD] ==============================")

            # Increment training steps
            self.training_steps += 1

            # Save checkpoint periodically
            if self.training_steps % self.ppo_config.checkpoint_interval == 0:
                self._save_checkpoint()

            # Clear rollout buffer and batches
            self.rollouts_ready.clear()
            del batches

            # Final cleanup
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"[CRD] PPO training complete, resuming rollout collection")

        finally:
            self.policy.eval()

    def _cleanup(self):
        """Cleanup resources."""
        print(f"[CRD] Cleaning up")

        # Shutdown S8 shards
        for pipe in self.pipes:
            try:
                pipe.send(MessageType.SHUTDOWN)
            except Exception:
                pass

        # Wait for shards to exit
        for shard in self.shards:
            shard.join(timeout=2.0)
            if shard.is_alive():
                print(f"[CRD] Force terminating shard {shard.name}")
                shard.terminate()

        # Close shared memory slabs
        for slab in self.shard_slabs:
            try:
                slab.close()
            except Exception:
                pass

        # Close pipes
        for pipe in self.pipes:
            try:
                pipe.close()
            except Exception:
                pass

        print(f"[CRD] Cleanup complete")


def main():
    """Entry point for PPO training."""
    print("=" * 60)
    print("PPO Training - Coordinator Entry Point")
    print("=" * 60)

    # Force spawn method (CUDA-safe)
    mp.set_start_method("spawn", force=True)

    # Load config
    from config import Config

    config = Config()

    # Create PPO config
    ppo_config = PPOConfig(
        num_shards=1,  # Start with 1 shard for testing
        envs_per_shard=2,  # 2 envs for testing
        dolphin_path="/Applications/Slippi Dolphin.app",
        iso_path="~/Documents/SSBM.iso",
        init_checkpoint=Path("checkpoints/latest.pt"),
    )

    # Create coordinator
    coordinator = Coordinator(config, ppo_config)

    # Run main loop
    coordinator.run()


if __name__ == "__main__":
    main()
