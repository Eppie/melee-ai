# global_config.py
from __future__ import annotations

import argparse
import ast
import copy
import json
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, get_args, get_origin, \
    get_type_hints, List

from zarr.codecs import BloscCodec, BloscShuffle

from controller_utils import CONTROL_STICK_QUANTIZED, C_STICK_QUANTIZED, SHOULDER_QUANTIZED
from schema import BUTTONS, get_feature_names, get_target_names


@dataclass
class _FreezeGuard:
    _frozen: bool = field(default=False, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False) and name != "_frozen":
            raise AttributeError(f"Config is frozen; cannot modify '{name}'.")
        object.__setattr__(self, name, value)


@dataclass
class ZarrConfig(_FreezeGuard):
    input_root: str = '/home/eppie/hal/replays'
    out_root: str = '/home/eppie/melee-ai/processed_data_1000'
    validation_root: str = '/home/eppie/melee-ai/validation_set'
    episode_count: int = 1000
    validation_count: int = 20
    shard_size: int = 100
    target_chunk_mb: float = 8.0
    compressor: BloscCodec = field(
        default_factory=lambda: BloscCodec(cname="zstd", clevel=7, shuffle=BloscShuffle.bitshuffle))
    seed: int = 42


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 100
    lr: float = 2e-4 # (DONE)
    weight_decay: float = 0.002 # (DONE)
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 10
    max_steps: Optional[int] = None
    num_workers: int = 16
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    stride = 1

    # losses
    grad_clip: float = 1.2 # (DONE)
    label_smoothing: float = 0.02 # (DONE)

    # Automatic Mixed Precision (AMP)
    use_amp: bool = True
    amp_dtype: str = "float16"  # "float16" or "bfloat16" (when MPS supports it)

    # random_windows sampler knobs
    num_samples: Optional[int] = None  # required if replacement=True

    # epoch sizing (per rank)
    windows_per_epoch: Optional[int] = None
    steps_per_epoch: Optional[int] = None  # if provided, overrides windows_per_epoch via steps * batch_size

    # checkpointing
    out_dir: str = "checkpoints"
    save_every_epochs: int = 1



@dataclass
class ProfileConfig:
    enable: bool = False
    out_dir: Optional[str] = None
    wait: int = 1
    warmup: int = 1
    active: int = 10
    repeat: int = 1
    record_shapes: bool = True
    with_stack: bool = False
    profile_memory: bool = False


@dataclass
class GPTConfig:
    """Model hyperparameters controlling GPTv7 architecture.

    Field details
    -------------
    block_size (int):
        - Model size: does not change parameter count but larger contexts increase KV-cache tensors and activation memory linearly.
        - Training speed: attention complexity grows with block_size^2; bigger windows slow training and increase FLOPs.
        - Restrictions / requirements: must be >= `Config.seq_len` and match the window length used by `window_dataset`; cannot exceed rotary embedding support without adjusting `rope_scaling`.
        - Interactions: keep in sync with `Config.seq_len`, dataloader window sizes, and any inference truncation logic.
        - Reasonable range: 128-1024 for current datasets; 256 is the tuned default.

    n_embd (int):
        - Model size: primary driver of parameter count (embedding tables, attention projections, MLP layers scale with n_embd^2).
        - Training speed: compute and memory roughly scale with n_embd^2; doubling often ~4x slower and more memory hungry.
        - Restrictions: must be divisible by `n_head`; ensure even numbers when `ffn_mult` produces integral hidden sizes.
        - Interactions: affects head dimension (`n_embd // n_head`), FFN width (`ffn_mult`), and MoE expert hidden sizes; adjust learning rate schedules when changing drastically.
        - Reasonable range: 192-1024 for mid-size models; 512 works well for current hardware.

    n_layer (int):
        - Model size: linearly increases parameter count and depth; each layer adds attention + MLP weights.
        - Training speed: slows training proportionally; also increases activation memory.
        - Restrictions: none, but very deep stacks (>48) require gradient checkpointing; ensure `num_stages` aligns with dataset semantics if tied to gameplay chronology.
        - Interactions: deeper models benefit from higher dropout or learning rate warmup.
        - Reasonable range: 4-24 for most experiments; 8 layers selected for balance.

    n_head (int):
        - Model size: attention projection weights scale with n_head; total params roughly `4 * n_embd * n_embd` independent, but multi-head normalisation changes layout.
        - Training speed: more heads slightly increase compute; head dimension (`n_embd // n_head`) must remain integer to keep kernels efficient.
        - Restrictions: must divide `n_embd`; for `attention_type in {"gqa", "mqa"}` additional constraints apply to `n_kv_head`.
        - Interactions: influences `attention_type`, `n_kv_head`, and rope rotations; extremely high head counts benefit from `qk_norm`.
        - Reasonable range: 4-32; 8 suits 512 embedding.

    dropout (float):
        - Model size: no effect on parameter count.
        - Training speed: incurs slight overhead due to random masking but mainly improves generalisation.
        - Restrictions: set to 0.0 during inference; must be between 0 and <1.
        - Interactions: pair with higher `n_layer`/`n_embd` to combat overfitting; raise when dataset is small.
        - Reasonable range: 0.0-0.2 for current workloads.

    bias (bool):
        - Model size: enabling adds bias terms to every Linear/LayerNorm, increasing parameters modestly (~<1%).
        - Training speed: negligible impact but biases can slow fused kernels slightly on GPUs.
        - Restrictions: disable when using implementations expecting bias=False (e.g., some FlashAttention kernels).
        - Interactions: `norm_affine` true still adds scale/shift even when biases off; for MoE, per-expert projections ignore this flag.
        - Reasonable choice: True for GPT-2 style parity, False for leaner models.

    input_size (int):
        - Model size: scales the input projection matrix (input_size × n_embd) and embedding lookups; direct control over first layer parameters.
        - Training speed: larger inputs marginally increase first projection FLOPs and data transfer.
        - Restrictions: must match the concatenated feature vector size produced by dataloader; ensure alignment with `ColumnMap`/dataset schema.
        - Interactions: changes require updating preprocessing to avoid dimension mismatches.
        - Reasonable range: dictated by feature engineering (currently 130 for controller state encoding).

    num_stages / num_characters / num_actions (ints):
        - Model size: control categorical embedding tables; parameter count grows with `embedding_dim × vocab_size`.
        - Training speed: negligible unless extremely large due to embedding lookups.
        - Restrictions: must match dataset label vocabularies; modifying requires regeneration of enums.
        - Interactions: associated embedding dims (`stage_embedding_dim`, `character_embedding_dim`, `action_embedding_dim`) should scale with log of vocab size.
        - Reasonable range: fixed by game rules (stages≈6, characters≈26, actions≈396).

    stage_embedding_dim / character_embedding_dim / action_embedding_dim (ints):
        - Model size: linear scaling of embedding tables and input fusion width.
        - Training speed: small impact on forward pass; impacts final concatenated input width.
        - Restrictions: keep modest to avoid bloating input embedding; must be <= `n_embd`.
        - Interactions: higher dims may require reducing `input_size` if embeddings are concatenated; coordinate with feature engineering.
        - Reasonable range: 4-32 depending on vocabulary diversity.

    gamma (float):
        - Model size: no effect.
        - Training speed: only affects loss discounting in RL-style objectives; negligible cost.
        - Restrictions: value in (0,1]; for purely supervised training keep near 1.0.
        - Interactions: works with sequence weighting logic in `train.py`; lower gamma emphasises recent frames.
        - Reasonable range: 0.95-0.999 for long-horizon credit assignment.

    norm_type (str):
        - Model size: choice (layernorm, rmsnorm, etc.) sets learnable parameters; RMSNorm omits mean subtraction, reducing parameters when paired with `norm_affine=False`.
        - Training speed: LayerNorm is slower than RMSNorm on GPU. Selecting `fused` implementations can help.
        - Restrictions: must be supported by implementation; `qk_norm` reuses this when `qk_norm_type` is None.
        - Interactions: `norm_eps`, `norm_affine`, `norm_placement` must be compatible; RMSNorm typically paired with `norm_affine=True`.
        - Reasonable options: "layernorm", "rmsnorm", "fused_rmsnorm" (if kernels available).

    norm_eps (float):
        - Model size: none.
        - Training speed: trivial cost; stabilises normalisation.
        - Restrictions: must be positive; too small risks numerical issues.
        - Interactions: tune alongside norm implementation; RMSNorm often prefers 1e-6 to 1e-5.
        - Reasonable range: 1e-6 to 1e-4.

    norm_affine (bool):
        - Model size: adds scale/bias parameters per norm when True.
        - Training speed: negligible runtime change.
        - Restrictions: set False to fully eliminate affine terms (for weight-tied inference); ensure downstream layers can absorb scaling.
        - Interactions: if `bias=False`, enabling `norm_affine` reintroduces per-dim scale/shift; consider lowering LR for stability when toggled.
        - Reasonable choice: True for flexibility, False for minimal parameter regimes.

    norm_placement (str):
        - Model size: none.
        - Training speed: `pre` yields Pre-LN transformer (stable, slightly faster); `post` can be slower; `both` doubles norm ops per block.
        - Restrictions: `both` requires attention/MLP implementations supporting dual norms.
        - Interactions: coordinate with residual dropout and `qk_norm`; `both` may need smaller learning rates.
        - Reasonable options: "pre" (default), "post" for GPT-2 style, "both" for experimental setups.

    qk_norm (bool) and qk_norm_type (Optional[str]):
        - Model size: adds per-head norm parameters when affine; negligible overall.
        - Training speed: introduces extra normalisation per attention head; slight overhead but can stabilise large head counts.
        - Restrictions: requires kernels supporting QK-normalisation; ensure `d_head` >= 16 for benefit.
        - Interactions: falls back to `norm_type` when `qk_norm_type` None; interacts with `attention_type` and `pe_type` (rope works best with qk_norm).
        - Reasonable usage: enable for models with `n_head >= 8` and long contexts; keep disabled for small models.

    attention_type (str):
        - Model size: `gqa`/`mqa` share key/value projections reducing parameters compared to `mha` when `n_head` large.
        - Training speed: grouped attention reduces memory traffic and can speed decoding; `mha` slightly heavier.
        - Restrictions: `gqa`/`mqa` require setting `n_kv_head`; ensure FlashAttention variant supports selected mode.
        - Interactions: influences `n_kv_head`, affects compatibility with rope scaling, and MoE gating (which may expect mha).
        - Reasonable options: "mha" default, "gqa" for head counts >=16, "mqa" for streaming inference.

    n_kv_head (Optional[int]):
        - Model size: sets number of key/value heads when using `gqa`; reduces parameters if < n_head.
        - Training speed: fewer KV heads reduce FLOPs and memory; must divide `n_head` cleanly.
        - Restrictions: only used for `gqa`; ignore for `mha`/`mqa` (where it should be None or 1 respectively).
        - Interactions: ensure `n_head % n_kv_head == 0`; interacts with caching and attention kernels.
        - Reasonable range: 1-`n_head`; typical `gqa` config uses `n_head // 4`.

    pe_type (str), rope_theta (float), rope_scaling (Optional[str]), rope_scaling_factor (float):
        - Model size: positional encoding choices do not alter parameter count (rope) or add minimal parameters (alibi).
        - Training speed: negligible differences; RoPE slightly more compute per token.
        - Restrictions: rope requires even head dimension; scaling options like "ntk" or "yarn" may need external libraries.
        - Interactions: `rope_scaling` modifies effective context window; pair with `block_size` adjustments; `rope_theta` influences frequency base.
        - Reasonable settings: `pe_type="rope"` with theta 10000, scaling None for <=2k context; use scaling_factor 0.5-2.0 when extending context.

    ffn_mult (float) and ffn_activation (str):
        - Model size: FFN hidden width = `int(ceil(ffn_mult * n_embd))`; scaling multiplier directly affects MLP parameter count (~2× hidden × n_embd).
        - Training speed: wider FFNs dominate compute; activations influence kernel choice (swiglu/gelu slightly slower than relu).
        - Restrictions: ensure resulting width divisible by tensor parallel shards if used; `swiglu`/`geglu` expect width multiple of 2.
        - Interactions: combine with `use_moe` (MoE may prefer smaller ffn_mult to offset extra experts); adjust learning rate when increasing mult.
        - Reasonable range: 2.0-4.0 for dense FFNs; `8/3` suits swiglu gating.

    head_flow (str):
        - Model size: no impact.
        - Training speed: `parallel` can reuse trunk outputs but increases memory; `sequential` matches current architecture.
        - Restrictions: `parallel` variant requires target head modules supporting concatenated inputs.
        - Interactions: influences `target_shapes_by_head` scheduling and output ordering.
        - Reasonable options: "sequential" default; "parallel" when targets predicted simultaneously.

    target_shapes_by_head (dict):
        - Model size: determines output projection sizes for each controller head; parameters scale with sum(product(shape) × hidden width).
        - Training speed: larger targets increase logits computation but modest compared to trunk.
        - Restrictions: keys must align with model forward outputs and loss functions; tuple lengths define per-token class counts.
        - Interactions: `head_flow` dictates ordering; adjusting shapes requires matching quantisation bins in `controller_quantization`.
        - Reasonable usage: keep close to dataset quantization cardinalities (e.g., FOX sticks 64-way, buttons 5 logits).

    use_moe (bool) and MoE-specific knobs:
        - Model size: enabling MoE replaces dense FFN with mixture of experts, inflating parameters roughly `moe_num_experts × n_embd × hidden` but only activating `moe_num_active` per token.
        - Training speed: sparse MoE introduces routing overhead; with adequate batching it can be faster per-token for large expert counts but slower if underutilised.
        - Restrictions: requires MoE kernels/support in `GPTv7`; ensure `moe_num_active <= moe_num_experts` and capacity factor >= 1.
        - Interactions: choose `ffn_mult` carefully (MoE hidden size derived from it), `moe_aux_loss_weight` balances load; `moe_shared_expert` adds dense expert used for fallback.
        - Reasonable settings: start with `use_moe=False`; when True, use 16-128 experts, `moe_num_active` 2-8, auxiliary loss weight 0.01-0.1.

      moe_num_experts (int):
        - Model size: increases total expert parameters linearly.
        - Training speed: more experts raise routing cost; ensure batch size large enough for load balancing.
        - Restrictions: must be >0; power-of-two counts simplify sharding.
        - Interactions: capacity factor may need raising when experts numerous.
        - Reasonable range: 16-256 depending on hardware.

      moe_num_active (int):
        - Model size: no change (affects runtime selection only).
        - Training speed: more active experts per token increases FLOPs proportionally; low values risk underfitting.
        - Restrictions: <= `moe_num_experts`; gating kernels often expect small integers (1,2,4,8).
        - Interactions: adjust `moe_expert_capacity_factor` to avoid token drops when active count high.
        - Reasonable range: 1-8.

      moe_aux_loss_weight (float):
        - Model size: none.
        - Training speed: adds auxiliary loss computation but trivial.
        - Restrictions: keep non-negative; too high prevents main loss from converging.
        - Interactions: tune with `moe_jitter_eps` and optimizer; interacts with gradient scale of gating network.
        - Reasonable range: 0.001-0.05 (0.01 default).

      moe_expert_capacity_factor (Optional[float]):
        - Model size: none.
        - Training speed: influences routing drops; higher capacity ensures tokens are routed but increases buffer sizes.
        - Restrictions: must be >=1.0 when set; None defers to implementation default.
        - Interactions: adjust with batch size and `moe_num_active`; low capacity plus high active count causes overflow/drops.
        - Reasonable range: 1.0-2.0; 1.25 balanced.

      moe_jitter_eps (float):
        - Model size: none.
        - Training speed: negligible; adds noise to gating logits for exploration.
        - Restrictions: keep small (<=0.1) to avoid instability.
        - Interactions: complements auxiliary loss weight; higher jitter may allow lower aux weight.
        - Reasonable range: 0.0-0.05; 0.01 default.

      moe_shared_expert (bool):
        - Model size: adds one dense expert shared across tokens, increasing parameters akin to baseline FFN.
        - Training speed: ensures fallback path; slight overhead even when use_moe True.
        - Restrictions: requires implementation support; ensure memory fits when combined with many experts.
        - Interactions: when True, you can reduce `moe_num_active` since shared expert provides baseline capacity.
        - Reasonable choice: False unless routing collapse observed.

      moe_normalize_expert_weights (bool):
        - Model size: none.
        - Training speed: applies softmax/normalisation to gating weights; minimal overhead.
        - Restrictions: disable only if custom gating normalisation is provided.
        - Interactions: pairs with `moe_jitter_eps`; keep enabled for stable routing.
        - Reasonable choice: True.

      moe_fine_grained (bool):
        - Model size: none directly but may change expert partitioning affecting memory layout.
        - Training speed: fine-grained routing can improve GPU utilisation but complicates batching.
        - Restrictions: only meaningful with implementations supporting per-token expert sharding.
        - Interactions: combine with high expert counts; may require tuning `moe_expert_capacity_factor`.
        - Reasonable choice: True for detailed routing, False for simpler all-to-all.
    """
    block_size: int = 512 # DONE
    n_embd: int = 512 # DONE
    n_layer: int = 4 # DONE
    n_head: int = 8 # DONE
    dropout: float = 0.03  # (DONE)
    bias: bool = False # DONE
    input_size: int = -1  # populated dynamically based on dataset schema
    num_stages: int = 6
    num_characters: int = 26
    num_actions: int = 396
    stage_embedding_dim: int = 4
    character_embedding_dim: int = 12
    action_embedding_dim: int = 32
    gamma: float = 0.999  # (DONE)
    norm_type: str = "layernorm"  # (DONE)
    norm_eps: float = 1e-7 # DONE
    norm_affine: bool = True  # (DONE)
    norm_placement: str = "post"  # options: pre, post, both (DONE)
    qk_norm: bool = False  # (DONE)
    qk_norm_type: Optional[str] = None  # defaults to norm_type when None
    attention_type: str = "gqa"  # TODO: maybe mqa?
    n_kv_head: Optional[int] = 4 # DONE
    pe_type: str = "rope"  # options: rope, alibi
    rope_theta: float = 10000.0
    rope_scaling: Optional[str] = None  # e.g., "ntk", "yarn"
    rope_scaling_factor: float = 1.0
    ffn_mult: float = 2 # DONE
    ffn_activation: str = "geglu" # DONE
    head_flow: str = "parallel"  # options: sequential, parallel
    target_shapes_by_head: dict[str, int] = field(default_factory=lambda: {
        "main_stick": len(CONTROL_STICK_QUANTIZED),
        "c_stick": len(C_STICK_QUANTIZED),
        "buttons": len(BUTTONS),
        "shoulder": len(SHOULDER_QUANTIZED),
    })

    # Value head for RL (outputs state value estimates)
    use_value_head: bool = True  # enable value head for PPO/A2C

    # Mixture-of-Experts (MoE) configuration
    use_moe: bool = False
    moe_num_experts: int = 128
    moe_num_active: int = 8
    moe_aux_loss_weight: float = 0.01
    moe_expert_capacity_factor: Optional[float] = 1.25
    moe_jitter_eps: float = 0.01
    moe_shared_expert: bool = False
    moe_normalize_expert_weights: bool = True
    moe_fine_grained: bool = True


@dataclass
class FeatureConfig(_FreezeGuard):
    """Feature preprocessing configuration."""

    transforms: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "transform": "stick_palette",
                "features": ["main_stick_x", "main_stick_y"],
                "palette": "fox_main",
            },
            {
                "transform": "stick_palette",
                "features": ["c_stick_x", "c_stick_y"],
                "palette": "c_stick",
            },
            {
                "transform": "scale",
                "features": ["facing"],
                "factor": 2.0,
            },
            {
                "transform": "offset",
                "features": ["facing"],
                "delta": -1.0,
            },
            {
                "transform": "scale",
                "features": ["percent"],
                "factor": 1 / 100.0,
            },
            {
                "transform": "scale",
                "features": ["shield_strength"],
                "factor": 1.0 / 60.0,
            },
            {
                "transform": "scale",
                "features": ["stock"],
                "factor": 1 / 4.0,
            },
            {
                "transform": "scale",
                "features": ["position_x", "position_y"],
                "factor": 1 / 20.0,
            },
            {
                "transform": "scale",
                "features": ["jumps_left"],
                "factor": 1 / 6.0,
            },
        ]
    )


@dataclass
class RLConfig(_FreezeGuard):
    """Reinforcement learning configuration."""
    
    # Algorithm selection
    algorithm: str = "ppo"  # ppo, a2c, reinforce
    
    # PPO hyperparameters
    ppo_epsilon: float = 0.2  # clip range for policy loss
    ppo_epochs: int = 4  # optimization epochs per rollout batch
    gae_lambda: float = 0.95  # GAE lambda for advantage estimation
    gamma: float = 0.99  # discount factor for rewards
    
    # Training parameters
    rollout_length: int = 512  # number of steps to collect per rollout
    batch_size: int = 64  # minibatch size for updates
    learning_rate: float = 3e-4  # learning rate for both policy and value
    value_loss_coef: float = 0.5  # coefficient for value loss in total loss
    entropy_coef: float = 0.01  # coefficient for entropy bonus
    max_grad_norm: float = 0.5  # max gradient norm for clipping
    
    # Mixed training (imitation + RL)
    use_mixed_training: bool = True  # combine expert demos with RL rollouts
    expert_ratio: float = 0.5  # fraction of batch from expert demos (0=pure RL, 1=pure imitation)
    expert_ratio_decay: float = 0.995  # multiply expert_ratio by this each epoch
    min_expert_ratio: float = 0.1  # minimum expert ratio (stops decay)
    
    # Reward weights (customize reward function)
    reward_win: float = 1.0
    reward_loss: float = -1.0
    reward_timeout: float = 0.0  # when game times out
    reward_damage_dealt: float = 0.01  # per % damage
    reward_damage_taken: float = -0.01  # per % damage
    reward_stock_lost: float = -0.3  # when losing a stock
    reward_stock_taken: float = 0.3  # when taking opponent's stock
    reward_stage_control: float = 0.001  # reward for center stage control
    reward_l_cancel: float = 0.02  # reward for successful L-cancel
    reward_combo_hit: float = 0.05  # reward for extending combo
    reward_hitlag_opponent: float = 0.02  # reward when opponent is in hitlag (attacking)
    reward_hitlag_self: float = -0.02  # penalty when we are in hitlag (being hit)
    reward_low_shield: float = -0.1  # penalty for low shield strength (magnified as shield -> 0)
    reward_per_frame: float = -0.001  # small constant penalty per frame to discourage stalling
    
    # Self-play configuration
    self_play_enabled: bool = True
    opponent_update_freq: int = 10  # update opponent checkpoint every N epochs
    evaluation_games: int = 20  # number of games for win rate evaluation
    use_opponent_pool: bool = False  # maintain pool of past checkpoints
    opponent_pool_size: int = 5  # size of opponent pool if enabled
    
    # Environment settings
    dolphin_path: str = "/Applications/Dolphin.app/Contents/MacOS/Dolphin"  # path to Dolphin executable
    iso_path: str = "/path/to/melee.iso"  # path to Melee ISO
    slippi_port: int = 51441  # port for Slippi communication
    
    # Replay buffer settings
    buffer_size: int = 100000  # maximum number of transitions to store
    prioritized_replay: bool = False  # use prioritized experience replay (future)
    alpha: float = 0.6  # prioritization exponent (if prioritized_replay=True)
    beta: float = 0.4  # importance sampling exponent (if prioritized_replay=True)
    
    # Logging and checkpointing
    log_interval: int = 10  # log metrics every N rollouts
    eval_interval: int = 50  # evaluate policy every N rollouts
    save_interval: int = 100  # save checkpoint every N rollouts
    use_wandb: bool = False  # log to Weights & Biases
    wandb_project: str = "melee-rl"  # W&B project name
    
    # Advanced RL techniques (future use)
    use_curiosity: bool = False  # curiosity-driven exploration
    use_her: bool = False  # hindsight experience replay
    use_auxiliary_tasks: bool = False  # auxiliary prediction tasks


@dataclass
class Config(_FreezeGuard):
    seq_len: int = 256

    zarr: ZarrConfig = field(default_factory=ZarrConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: GPTConfig = field(default_factory=GPTConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    rl: RLConfig = field(default_factory=RLConfig)

    def freeze(self) -> None:
        _freeze_dataclass(self)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of the config (recursively)."""
        return _to_jsonable(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        return _dataclass_from_dict(cls, data)

    @classmethod
    def from_json(cls, s: str) -> "Config":
        return cls.from_dict(json.loads(s))


def _player_prefixes(feature_names: Sequence[str]) -> List[str]:
    prefixes: set[str] = set()
    for name in feature_names:
        if len(name) < 3 or name[0] != "p" or name[1] not in "0123456789":
            continue
        head, _, tail = name.partition("_")
        if not tail:
            continue
        prefixes.add(head)
    if not prefixes:
        raise ValueError("No player-prefixed feature columns found (expected p1_/p2_ entries).")
    return sorted(prefixes)


def _controller_field_bases() -> List[str]:
    bases: set[str] = set()
    for name in get_target_names():
        prefix, _, base = name.partition("_")
        if not base or not prefix.startswith("p"):
            continue
        bases.add(base)
    if not bases:
        raise ValueError("Unable to infer controller field names from schema targets.")
    return sorted(bases)


def _compute_model_input_size(cfg: "Config") -> int:
    feature_names = get_feature_names()

    if "stage" not in feature_names:
        raise ValueError("Required feature 'stage' missing; cannot derive model input size.")

    prefixes = _player_prefixes(feature_names)

    categorical_bases = ("character", "action")
    categorical_names: List[str] = []
    for prefix in prefixes:
        for base in categorical_bases:
            name = f"{prefix}_{base}"
            if name not in feature_names:
                raise ValueError(
                    f"Required categorical feature '{name}' missing; ensure feature selection keeps it."
                )
            categorical_names.append(name)

    controller_bases = _controller_field_bases()
    controller_names: List[str] = []
    for prefix in prefixes:
        for base in controller_bases:
            name = f"{prefix}_{base}"
            if name not in feature_names:
                raise ValueError(
                    f"Controller feature '{name}' missing; update schema targets or feature selection."
                )
            controller_names.append(name)

    reserved = 1 + len(categorical_names) + len(controller_names)
    if reserved > len(feature_names):
        raise ValueError("Feature accounting failed; reserved columns exceed available features.")

    gamestate_count = len(feature_names) - reserved

    embedding_dims = (
            cfg.model.stage_embedding_dim
            + len(prefixes) * cfg.model.character_embedding_dim
            + len(prefixes) * cfg.model.action_embedding_dim
    )
    return embedding_dims + gamestate_count + len(controller_names)


def _apply_derived_fields(cfg: "Config") -> None:
    cfg.model.input_size = _compute_model_input_size(cfg)


_GLOBAL_CFG: Optional[Config] = None


def init_config(
        initial: Optional[Mapping[str, Any]] = None,
        cli_overrides: Optional[Mapping[str, str]] = None,
        *,
        freeze: bool = True,
) -> Config:
    global _GLOBAL_CFG
    cfg = Config.from_dict(initial or {})
    if cli_overrides:
        apply_overrides(cfg, cli_overrides)
    _apply_derived_fields(cfg)
    if freeze:
        cfg.freeze()
    _GLOBAL_CFG = cfg
    return cfg


def get_config() -> Config:
    if _GLOBAL_CFG is None:
        raise RuntimeError("Global config not initialized. Call init_config(...) early in your program.")
    return _GLOBAL_CFG


def reset_config_for_tests() -> None:
    global _GLOBAL_CFG
    _GLOBAL_CFG = None


def apply_overrides(cfg: Config, overrides: Mapping[str, str]) -> None:
    if getattr(cfg, "_frozen", False):
        raise AttributeError("Config is frozen; cannot apply overrides.")

    for dotted_key, raw in overrides.items():
        parts = dotted_key.split(".")
        parent, attr = _resolve_parent_and_attr(cfg, parts)
        if is_dataclass(parent):
            target_type = _dataclass_field_type(type(parent), attr)
            value = _coerce(raw, target_type)
            setattr(parent, attr, value)
        elif isinstance(parent, MutableMapping):
            parent[attr] = _coerce_best_effort(raw)
        else:
            raise TypeError(f"Cannot set '{dotted_key}'; parent is neither dataclass nor mapping.")

    _apply_derived_fields(cfg)


def parse_cli_overrides(argv: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Minimal CLI:
      --config_json PATH   (optional) load initial config values from JSON
      --set KEY=VALUE      (repeatable) e.g. --set learning_rate=5e-4 --set optimizer.weight_decay=0.02
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config_json", type=str, default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ns, _ = p.parse_known_args(argv)

    initial: Dict[str, Any] = {}
    if ns.config_json:
        with open(ns.config_json, "r", encoding="utf-8") as f:
            initial = json.load(f)

    overrides: Dict[str, str] = {}
    for item in ns.set:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected KEY=VALUE.")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()

    return initial, overrides


def save_config_json(path: Union[str, Path], cfg: Optional[Config] = None) -> Path:
    cfg = cfg or get_config()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg.to_json(), encoding="utf-8")
    return path


def load_config_json(path: Union[str, Path], *, freeze: bool = True) -> Config:
    s = Path(path).read_text(encoding="utf-8")
    cfg = Config.from_json(s)
    if freeze:
        cfg.freeze()
    return cfg


def _dataclass_from_dict(cls: type, data: Mapping[str, Any]) -> Any:
    """Recursively construct dataclass instance from dict, merging into defaults."""
    inst = cls()  # start from defaults
    type_hints = _dataclass_type_hints(cls)
    updates: Dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        incoming = data[f.name]
        ftype = type_hints.get(f.name, f.type)
        base = getattr(inst, f.name)

        dc_cls = _unwrap_dataclass_type(ftype)
        if dc_cls and isinstance(incoming, Mapping):
            # nested dataclass
            nested = _dataclass_from_dict(dc_cls, incoming)
            updates[f.name] = nested
        elif isinstance(base, dict) and isinstance(incoming, Mapping):
            merged = copy.deepcopy(base)
            merged.update(incoming)
            updates[f.name] = merged
        else:
            updates[f.name] = incoming

    if not updates:
        return inst

    dataclass_params = getattr(cls, "__dataclass_params__", None)
    if dataclass_params and dataclass_params.frozen:
        return replace(inst, **updates)

    for name, value in updates.items():
        setattr(inst, name, value)
    return inst


def _unwrap_dataclass_type(tp: Any) -> Optional[type]:
    """Return the dataclass type if tp is a dataclass or Optional[dataclass], else None."""
    if is_dataclass(tp):
        return tp  # type: ignore[return-value]
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is Union and len(args) == 2 and type(None) in args:
        t = args[0] if args[1] is type(None) else args[1]
        return t if is_dataclass(t) else None
    return None


def _dataclass_field_type(dc_type: type, name: str) -> Any:
    type_hints = _dataclass_type_hints(dc_type)
    if name in type_hints:
        return type_hints[name]
    raise KeyError(f"Unknown field '{name}' on {dc_type.__name__}")


@lru_cache(maxsize=None)
def _dataclass_type_hints(dc_type: type) -> Dict[str, Any]:
    """Return resolved type hints for a dataclass, resilient to postponed evaluation."""
    try:
        return get_type_hints(dc_type, include_extras=True)
    except Exception:
        # Fallback to the raw annotations if get_type_hints cannot resolve them.
        return {f.name: f.type for f in fields(dc_type)}


def _resolve_parent_and_attr(root: Any, parts: Sequence[str]) -> Tuple[Any, str]:
    """Walk parts[:-1] and return (parent, final_attr_name)."""
    if not parts:
        raise ValueError("Empty override key")
    cur = root
    for p in parts[:-1]:
        if is_dataclass(cur):
            if not hasattr(cur, p):
                raise KeyError(f"Unknown field '{p}' in path: {'.'.join(parts)}")
            cur = getattr(cur, p)
        elif isinstance(cur, Mapping):
            cur = cur[p]
        else:
            raise TypeError(f"Cannot traverse into '{p}' on {type(cur).__name__}")
    return cur, parts[-1]


def _coerce(raw: str, target_type: Any) -> Any:
    """Coerce string to annotated target_type. Handles Optional[T], bool/int/float/str; else literal_eval fallback."""
    origin = get_origin(target_type)
    args = get_args(target_type)
    if origin is Union and len(args) == 2 and type(None) in args:
        t = args[0] if args[1] is type(None) else args[1]
        return None if raw.lower() in {"none", "null"} else _coerce(raw, t)
    if target_type in (str, int, float):
        return target_type(raw)
    if target_type is bool:
        return _parse_bool(raw)
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def _coerce_best_effort(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except Exception:
        pass
    for caster in (_parse_bool, int, float):
        try:
            return caster(raw)  # type: ignore[misc]
        except Exception:
            continue
    return raw


def _parse_bool(s: str) -> bool:
    s = s.lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean from '{s}'")


def _freeze_dataclass(dc: _FreezeGuard) -> None:
    """Recursively freeze a dataclass (convert containers to immutable, set _frozen=True everywhere)."""
    assert is_dataclass(dc)
    for f in fields(dc):
        if f.name.startswith("_"):
            continue
        val = getattr(dc, f.name)
        frozen_val = _deep_freeze_value(val)
        object.__setattr__(dc, f.name, frozen_val)
    object.__setattr__(dc, "_frozen", True)


def _deep_freeze_value(obj: Any) -> Any:
    if is_dataclass(obj) and isinstance(obj, _FreezeGuard):
        _freeze_dataclass(obj)
        return obj
    if isinstance(obj, dict):
        return MappingProxyType({k: _deep_freeze_value(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_deep_freeze_value(v) for v in obj)
    if isinstance(obj, set):
        return frozenset(_deep_freeze_value(v) for v in obj)
    return obj


def _to_jsonable(obj: Any) -> Any:
    # Primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Enums -> their .value (e.g., BloscCname.zstd -> "zstd")
    if isinstance(obj, Enum):
        return obj.value

    # Dataclasses -> dict (skip private fields)
    if is_dataclass(obj):
        return {
            f.name: _to_jsonable(getattr(obj, f.name))
            for f in fields(obj)
            if not f.name.startswith("_")
        }

    # Mappings -> dict with stringified keys
    if isinstance(obj, Mapping):
        return {str(_to_jsonable(k)): _to_jsonable(v) for k, v in obj.items()}

    # Sequences & sets -> lists
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in obj]

    # Paths
    if isinstance(obj, Path):
        return str(obj)

    # Objects that know how to serialize themselves (Zarr v3 codecs, etc.)
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return _to_jsonable(obj.to_dict())
        except Exception:
            pass  # fall through to other options

    # Numcodecs codecs (and others) often expose get_config()
    if hasattr(obj, "get_config") and callable(getattr(obj, "get_config")):
        try:
            return _to_jsonable(obj.get_config())
        except Exception:
            pass

    # Numpy: scalars -> Python scalars; arrays -> lists
    try:
        import numpy as np  # optional dependency
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # PyTorch: represent dtypes/devices/sizes as strings/lists; tensors as lists
    try:
        import torch  # optional dependency
        if isinstance(obj, torch.dtype) or isinstance(obj, torch.device):
            return str(obj)
        if isinstance(obj, torch.Size):
            return list(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass

    # Last resort: string representation
    return repr(obj)
