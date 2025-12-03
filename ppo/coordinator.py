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
            (ppo_config.total_envs, ppo_config.context_length, 908),
            dtype=torch.bfloat16,
            device=self.device,
        )

        # Pinned staging buffer for H2D transfer
        self.staging = PinnedStagingBuffer(
            num_envs=ppo_config.total_envs,
            feature_dim=908,
        )

        # Barrier for synchronization
        self.barrier = Barrier(self.pipes, timeout=10.0)

        # Ring position tracking
        self.t_mod = 0
        self.step_id = 0

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
        model = GPT(config.model)

        # Load weights
        if "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        # Handle torch.compile prefix mismatch
        from utils import match_state_dict_keys

        state_dict = match_state_dict_keys(model.state_dict(), state_dict)

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

            # Map shared memory slab (read-only from CRD)
            slab = SharedMemorySlab(
                shard_id=shard_id,
                envs_per_shard=self.ppo_config.envs_per_shard,
                context_length=self.ppo_config.context_length,
                create=False,  # Attach to existing
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
            # 1. Compute ego actions for all envs (single batch)
            ego_actions = self._forward_policy(self.policy, self.X_gpu)

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

                # Extract features for this group
                group_features = self.X_gpu[
                    env_ids
                ]  # [len(env_ids), context_length, 908]

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
        print(f"[CRD] Starting main inference loop")
        self.start_time = time.time()

        try:
            while True:
                # 1. Wait for all S8s to signal READY
                self.barrier.wait_all_ready(self.step_id)

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
                self.total_frames += (
                    self.ppo_config.total_envs
                )  # Each step processes all envs

                # Periodic logging and health check
                if self.step_id % 100 == 0:
                    ego_values = ego_actions[5]  # values are at index 5
                    fps = (
                        self.total_frames / (time.time() - self.start_time)
                        if hasattr(self, "start_time")
                        else 0
                    )
                    print(
                        f"[CRD] Step {self.step_id}, t_mod={self.t_mod}, "
                        f"frames={self.total_frames:,}, "
                        f"fps={fps:.1f}, "
                        f"ego_values mean={ego_values.mean():.3f}"
                    )

                # Health check every 1000 steps
                if self.step_id % 1000 == 0:
                    self._health_check()

                # Check if rollouts completed (every rollout_length frames)
                if (
                    self.step_id % self.ppo_config.rollout_length == 0
                    and self.step_id > 0
                ):
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
        """Check health of all shard processes."""
        dead_shards = []
        for i, shard in enumerate(self.shards):
            if not shard.is_alive():
                dead_shards.append(i)

        if dead_shards:
            print(f"[CRD] WARNING: Dead shards detected: {dead_shards}")
            # TODO: Implement shard restart logic
            # For now, just log and continue
        else:
            print(f"[CRD] Health check: All {len(self.shards)} shards alive")

    def _save_checkpoint(self):
        """Save checkpoint for opponent pool and resumption."""
        checkpoint_dir = self.ppo_config.checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create checkpoint filename
        checkpoint_path = checkpoint_dir / f"checkpoint_{self.training_steps:06d}.pt"

        # Save checkpoint
        checkpoint = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.model_dump()
            if hasattr(self.config, "model_dump")
            else self.config,
            "ppo_config": self.ppo_config.model_dump()
            if hasattr(self.ppo_config, "model_dump")
            else self.ppo_config,
            "training_steps": self.training_steps,
            "total_frames": self.total_frames,
            "step_id": self.step_id,
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
                print(f"[CRD] Pruned old checkpoint: {ckpt.name}")

    def _collect_completed_rollouts(self):
        """
        Collect completed rollouts from S8 shards.

        This is called every rollout_length frames.
        In production, ENVs would signal completion via IPC.
        For now, we'll implement a simplified version.
        """
        print(f"[CRD] Collecting completed rollouts (step {self.step_id})")

        # Reassign opponents for next rollout
        self.matchmaker.reassign_all()

        # Clear loaded opponents to free memory
        self.matchmaker.clear_loaded_opponents()

        # TODO: Implement proper rollout collection from ENVs
        # For now, just log
        # In full implementation:
        # 1. ENVs signal rollout complete via IPC
        # 2. CRD reads rollout buffers from shared memory or separate channel
        # 3. Compute advantages/returns on GPU
        # 4. Add to rollouts_ready queue

    def _run_ppo_training(self):
        """Execute PPO training on accumulated rollouts."""
        print(f"[CRD] Starting PPO training with {len(self.rollouts_ready)} rollouts")
        print(
            f"[CRD] Training step {self.training_steps}, Total frames: {self.total_frames:,}"
        )

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

            print(f"[CRD] Created {len(batches)} windowed batches")

            # PPO epochs
            for epoch in range(self.ppo_config.ppo_epochs):
                for batch_idx, batch in enumerate(batches):
                    # Move batch to device
                    features = batch["features"].to(self.device)
                    old_logp = batch["old_logp"].to(self.device)
                    advantages = batch["advantages"].to(self.device)
                    returns = batch["returns"].to(self.device)
                    actions = batch["actions"]  # Stays on CPU (structured array)

                    # Convert actions to dict format
                    action_dict = {
                        "main_idx": torch.from_numpy(actions["main_idx"]).to(
                            self.device
                        ),
                        "c_idx": torch.from_numpy(actions["c_idx"]).to(self.device),
                        "shoulder_idx": torch.from_numpy(actions["shoulder_idx"]).to(
                            self.device
                        ),
                        "buttons": torch.from_numpy(actions["buttons"]).to(self.device),
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

                    # Track metrics
                    all_losses.append(loss_dict["total"].item())
                    all_policy_losses.append(loss_dict["policy"].item())
                    all_value_losses.append(loss_dict["value"].item())
                    all_entropy_losses.append(loss_dict["entropy"].item())
                    all_ratios.append(loss_dict["ratio_mean"].item())
                    all_kl_divs.append(loss_dict["approx_kl"].item())

                    # Periodic logging
                    if batch_idx % 10 == 0:
                        print(
                            f"[CRD] Epoch {epoch}/{self.ppo_config.ppo_epochs}, "
                            f"Batch {batch_idx}/{len(batches)}, "
                            f"Loss: {loss_dict['total'].item():.4f}, "
                            f"Policy: {loss_dict['policy'].item():.4f}, "
                            f"Value: {loss_dict['value'].item():.4f}, "
                            f"Ratio: {loss_dict['ratio_mean'].item():.3f}"
                        )

            # Print summary metrics
            print(f"[CRD] Training Summary:")
            print(f"  Total Loss:    {np.mean(all_losses):.4f}")
            print(f"  Policy Loss:   {np.mean(all_policy_losses):.4f}")
            print(f"  Value Loss:    {np.mean(all_value_losses):.4f}")
            print(f"  Entropy Loss:  {np.mean(all_entropy_losses):.4f}")
            print(f"  Mean Ratio:    {np.mean(all_ratios):.3f}")
            print(f"  Approx KL:     {np.mean(all_kl_divs):.4f}")

            # Increment training steps
            self.training_steps += 1

            # Save checkpoint periodically
            if self.training_steps % self.ppo_config.checkpoint_interval == 0:
                self._save_checkpoint()

            # Clear rollout buffer
            self.rollouts_ready.clear()

            print(f"[CRD] PPO training complete")

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
