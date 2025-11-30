from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


class PPOConfig(BaseModel):
    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    pool_size: int = Field(
        default=5,
        ge=1,
        description=(
            "Size of opponent pool for self-play. Maintains a FIFO queue of frozen past model checkpoints. "
            "Effect: Larger pool (5-10) = more diverse opponents, more stable training but more memory; "
            "smaller pool (2-3) = less diversity, faster opponent turnover. Reasonable range: [2, 10]. "
            "Interacts with: opponent_rotation_interval (how often opponents are swapped)."
        ),
    )
    clip_ratio: float = Field(
        default=0.2,
        gt=0,
        description=(
            "PPO clipping parameter (epsilon). Limits policy update magnitude to prevent destructive updates. "
            "Effect: Lower clip_ratio (0.1-0.15) = more conservative updates, more stable but slower learning; "
            "higher clip_ratio (0.2-0.3) = larger updates, faster but riskier. Reasonable range: [0.1, 0.3]. "
            "Standard PPO uses 0.2. Interacts with: lr (both control update size), ppo_epochs (more epochs may need lower clip)."
        ),
    )
    entropy_coef: float = Field(
        default=0.01,
        ge=0,
        description=(
            "Entropy bonus coefficient. Encourages exploration by penalizing overly deterministic policies. "
            "Effect: Higher entropy_coef (0.01-0.1) = more exploration, less exploitation; "
            "lower entropy_coef (0.0-0.005) = more exploitation, deterministic policy. Reasonable range: [0.0, 0.1]. "
            "Interacts with: clip_ratio (both affect policy updates), value_loss_coef (three-way balance)."
        ),
    )
    gae_lambda: float = Field(
        default=0.95,
        ge=0,
        le=1,
        description=(
            "GAE (Generalized Advantage Estimation) lambda parameter. Controls bias-variance tradeoff in advantage estimation. "
            "Effect: Higher lambda (0.95-0.99) = lower bias, higher variance (use more future timesteps); "
            "lower lambda (0.9-0.95) = higher bias, lower variance (trust value function more). "
            "Reasonable range: [0.9, 0.99]. Standard is 0.95. "
            "Interacts with: gamma in RLConfig (both affect advantage), normalize_advantages."
        ),
    )
    lr: float = Field(
        default=1e-5,
        gt=0,
        description=(
            "Learning rate for PPO optimizer. Typically much lower than imitation learning lr. "
            "Effect: Higher lr (1e-4 to 1e-3) = faster learning but may be unstable; "
            "lower lr (1e-6 to 1e-5) = more stable but slower. Reasonable range: [1e-6, 1e-4]. "
            "Interacts with: clip_ratio (both control update magnitude), ppo_epochs."
        ),
    )
    ppo_epochs: int = Field(
        default=4,
        ge=1,
        description=(
            "Number of optimization epochs per batch of rollout data. How many times to reuse each trajectory batch. "
            "Effect: More epochs (4-10) = better data efficiency but risk of overfitting to batch; "
            "fewer epochs (1-3) = less overfitting but less data efficient. Reasonable range: [3, 10]. "
            "Interacts with: clip_ratio (clipping prevents overfitting), minibatch_size."
        ),
    )
    minibatch_size: int = Field(
        default=64,
        ge=1,
        description=(
            "Minibatch size for PPO training. Rollout data is split into minibatches for training. "
            "Effect: Larger minibatches (128-256) = more stable gradients but less updates per epoch; "
            "smaller minibatches (32-64) = noisier gradients but more updates. Reasonable range: [32, 256]. "
            "Interacts with: rollout_length (determines number of minibatches), ppo_epochs."
        ),
    )
    max_grad_norm: float = Field(
        default=0.5,
        gt=0,
        description=(
            "Maximum gradient norm for clipping in PPO. Prevents exploding gradients. "
            "Effect: Lower values (0.3-0.5) = more aggressive clipping, more stable; "
            "higher values (0.5-1.0) = less clipping. Reasonable range: [0.3, 1.0]. "
            "Note: Lower than imitation learning grad_clip since PPO is more sensitive to instability."
        ),
    )
    max_episode_frames: int = Field(
        default=18000,
        ge=1,
        description=(
            "Maximum episode length in frames before forced termination. "
            "At 60 FPS: 18000 frames = 5 minutes. Prevents infinitely long episodes. "
            "Reasonable range: [10800, 36000] (3-10 minutes)."
        ),
    )
    num_workers: int = Field(
        default=8,
        ge=1,
        description=(
            "Number of parallel simulation workers for distributed PPO. "
            "Effect: More workers = faster data collection but more CPU/memory. "
            "Reasonable range: [4, 16]. Interacts with: distributed_mode."
        ),
    )
    normalize_advantages: bool = Field(
        default=True,
        description=(
            "Normalize advantages to mean=0, std=1 before policy update. "
            "Recommended: True for more stable training. False may be used if advantages already well-scaled."
        ),
    )
    value_clip: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Optional value function clipping parameter. If set, clips value function updates similar to policy clipping. "
            "Effect: Prevents large value function updates. Typical values: 0.2-0.5 (matching clip_ratio). "
            "None = no value clipping (original PPO). Reasonable range: [0.1, 0.5] or None. "
            "Interacts with: clip_ratio, value_loss_coef."
        ),
    )

    # Distributed PPO settings
    rollout_length: int = Field(
        default=5000,
        ge=100,
        description=(
            "Number of frames to collect per rollout before training. Determines batch size for PPO updates. "
            "Effect: Longer rollouts (5000-10000) = more data per update, better sample efficiency but slower iteration; "
            "shorter rollouts (2000-4000) = faster iteration, more frequent updates. "
            "Reasonable range: [2000, 10000]. At 60 FPS: 5000 frames ≈ 83 seconds of gameplay. "
            "Interacts with: minibatch_size (rollout_length / minibatch_size = number of minibatches), ppo_epochs."
        ),
    )

    opponent_rotation_interval: int = Field(
        default=10000,
        ge=1000,
        description=(
            "Frames between opponent model swaps in self-play. How often to rotate to next opponent in pool. "
            "Effect: Longer intervals (10000-20000) = more experience against same opponent, may overfit; "
            "shorter intervals (5000-10000) = more diversity, less overfitting. "
            "Reasonable range: [5000, 20000]. Interacts with: pool_size (larger pool = more time between repeats)."
        ),
    )

    warmup_frames: int = Field(
        default=256,
        ge=1,
        description=(
            "Frames to buffer before model predictions start. Fills initial context window. "
            "Should match block_size in GPTConfig for consistent shapes. "
            "Effect: Must be >= block_size for proper model input. Typical value: block_size (e.g., 256 or 512). "
            "Interacts with: model.block_size (must be ≤ block_size)."
        ),
    )

    distributed_mode: bool = Field(
        default=True,
        description=(
            "Use distributed architecture with centralized GPU inference. "
            "When True: simulation workers run on CPU, send states to central GPU for inference. "
            "When False: each worker runs full model locally. "
            "Effect: True = better GPU utilization, supports more workers, but adds communication overhead; "
            "False = simpler, better for single machine. Default: True (recommended for production)."
        ),
    )

    # Async training settings (NEW)
    async_training: bool = Field(
        default=True,
        description=(
            "Run PPO training in a separate process to enable parallel collection and training. "
            "When True: Training runs in background process while collection continues (eliminates pause-train-resume cycle). "
            "When False: Traditional synchronous training (workers pause during training). "
            "Effect: True = 20-30% throughput gain, more complex; False = simpler, traditional. "
            "Default: True (recommended for production). Requires distributed_mode=True."
        ),
    )

    gradient_accumulation_steps: int = Field(
        default=3,
        ge=1,
        description=(
            "Number of rollouts to accumulate before applying gradients. "
            "Effect: Higher values (3-4) = more stable gradients, smoother updates; "
            "lower values (1-2) = faster iteration, noisier gradients. "
            "Reasonable range: [1, 8]. Only used when async_training=True. "
            "Interacts with: rollout_length (total frames per update = rollout_length × gradient_accumulation_steps)."
        ),
    )

    # Safety mechanisms (NEW)
    max_mean_actor_kl: float = Field(
        default=0.01,
        ge=0,
        description=(
            "Maximum mean KL divergence between old and new policy before reverting update. "
            "Safety mechanism to prevent destructive policy changes. "
            "Effect: Lower values (0.005-0.01) = more conservative, prevent large changes; "
            "higher values (0.01-0.02) = allow larger updates. "
            "Set to 0.0 to disable checkpoint reversion. Reasonable range: [0.005, 0.02]. "
            "Interacts with: clip_ratio (both limit policy updates)."
        ),
    )

    max_clipped_fraction: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description=(
            "Maximum fraction of samples that can be clipped before early stopping. "
            "If more than this fraction of samples hit the clip ratio, training epoch stops early. "
            "Effect: Lower values (0.3-0.4) = stop earlier, prevent overfitting; "
            "higher values (0.5-0.7) = train longer per epoch. "
            "Reasonable range: [0.3, 0.7]. Typical value: 0.5."
        ),
    )

    # Teacher distillation (NEW) - Optional, disabled by default
    teacher_kl_weight: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Weight for teacher distillation loss to prevent catastrophic forgetting. "
            "Adds KL divergence penalty between policy and frozen teacher model. "
            "Effect: Higher values (0.001-0.01) = stay closer to teacher, less forgetting; "
            "0.0 = no teacher distillation. "
            "Reasonable range: [0.0, 0.01]. Set to 0.003 to enable. "
            "Requires: teacher_checkpoint_path to be set."
        ),
    )

    teacher_checkpoint_path: Optional[str] = Field(
        default=None,
        description=(
            "Path to teacher model checkpoint for distillation. "
            "Should be a pre-trained model checkpoint (.pt file). "
            "Only used if teacher_kl_weight > 0. "
            "Example: 'checkpoints/pretrained_model.pt'"
        ),
    )
