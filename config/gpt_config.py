from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field, ValidationInfo, model_validator
from pydantic_settings import SettingsConfigDict

from column_map import ColumnMap
from constants import BUTTON_TARGET_NAMES
from controller_utils import (
    CONTROL_STICK_QUANTIZED,
    C_STICK_QUANTIZED,
    SHOULDER_QUANTIZED,
)
from schema import get_feature_names, get_target_names


@lru_cache(maxsize=1)
def _schema_feature_dims() -> Tuple[int, int]:
    """Return canonical (gamestate_dim, controller_dim) from the schema."""
    colmap = ColumnMap(get_feature_names(), get_target_names())
    return len(colmap.gamestate_idxs), len(colmap.controller_idxs)


class GPTConfig(BaseModel):
    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    block_size: int = Field(
        default=512,
        ge=1,
        description=(
            "Maximum sequence length (context window) in frames. Determines how much game history "
            "the model can see. Effect: Larger values (512-1024) allow longer-term planning but require "
            "more memory and compute; smaller values (128-256) are faster but limit strategic depth. "
            "Reasonable range: [128, 1024]. At 60 FPS: 512 frames ≈ 8.5 seconds. "
            "Interacts with: n_layer (deeper models can better use longer context), training memory, stride."
        ),
    )
    n_embd: int = Field(
        default=768,
        ge=1,
        description=(
            "Model embedding dimension. Controls model capacity. "
            "Effect: Higher values (768-1024) = more capacity, better performance but slower and more memory; "
            "lower values (256-512) = faster, less memory, may underfit. Reasonable range: [256, 1024]. "
            "Interacts with: n_head (n_embd must be divisible by n_head), n_layer, input_size."
        ),
    )
    n_layer: int = Field(
        default=4,
        ge=1,
        description=(
            "Number of transformer layers. Controls model depth and abstraction capability. "
            "Effect: More layers (12-24) = better abstraction, longer-term dependencies but slower; "
            "fewer layers (4-8) = faster, may miss complex patterns. Reasonable range: [4, 16]. "
            "Interacts with: n_embd (wider + deeper = more capacity), grad_clip (deeper may need lower clip)."
        ),
    )
    n_head: int = Field(
        default=12,
        ge=1,
        description=(
            "Number of attention heads. Controls attention parallelism. "
            "Effect: More heads (12-16) = more diverse attention patterns; fewer heads (4-8) = simpler attention. "
            "Reasonable range: [4, 16]. IMPORTANT: n_embd must be divisible by n_head. "
            "Interacts with: n_embd (determines head dimension = n_embd/n_head), n_kv_head (for MQA)."
        ),
    )
    dropout: float = Field(
        default=0.05,
        ge=0,
        le=1,
        description=(
            "Dropout probability for regularization. Applied after attention residual, after MLP, and in output heads. "
            "Prevents overfitting by randomly zeroing activations. "
            "Effect: Higher values (0.1-0.2) = stronger regularization, may hurt capacity; "
            "lower values (0.0-0.05) = less regularization. Reasonable range: [0.0, 0.2]. "
            "Interacts with: weight_decay (both regularize), model size (larger models may need more dropout)."
        ),
    )
    input_size: int = Field(
        default=-1,
        description=(
            "Total input feature dimension (auto-computed if -1). "
            "Calculated as: num_stages + num_characters*2 + num_actions*2 + gamestate_dim + controller_dim. "
            "Usually leave at -1 for automatic computation."
        ),
    )
    num_stages: int = Field(
        default=6,
        ge=1,
        description="Number of Melee stages (one-hot encoded). Default 6 for common competitive stages.",
    )
    num_characters: int = Field(
        default=26,
        ge=1,
        description="Number of Melee characters (one-hot encoded). Default 26 for full roster.",
    )
    num_actions: int = Field(
        default=396,
        ge=1,
        description="Number of possible action states (one-hot encoded). Default 396 covers all Melee action states.",
    )
    gamma: float = Field(
        default=0.999,
        ge=0,
        le=1,
        description=(
            "Discount factor for value head bootstrapping. Used when computing value targets. "
            "Effect: Higher gamma (0.999-0.9995) = value head considers longer horizons; "
            "lower gamma (0.99-0.995) = more myopic value estimates. Reasonable range: [0.99, 0.9995]. "
            "Interacts with: value head training, reward scaling."
        ),
    )
    n_kv_head: Optional[int] = Field(
        default=12,
        ge=1,
        description=(
            "Number of key-value heads for Multi-Query Attention (MQA). "
            "If None, uses standard multi-head attention (n_kv_head = n_head). "
            "Effect: Lower n_kv_head (1-4) = less memory/compute, slightly lower quality; "
            "n_kv_head = n_head = standard attention. Reasonable values: 1 (MQA), n_head/2, n_head (standard). "
            "Interacts with: n_head (must be divisible by n_kv_head)."
        ),
    )
    target_shapes_by_head: Dict[str, int] = Field(
        default_factory=lambda: {
            "main_stick": len(CONTROL_STICK_QUANTIZED),
            "c_stick": len(C_STICK_QUANTIZED),
            "buttons": len(BUTTON_TARGET_NAMES),
            "shoulder": len(SHOULDER_QUANTIZED),
        },
        description=(
            "Output dimension for each prediction head. Auto-populated from quantization tables. "
            "main_stick: 64 positions, c_stick: 9 positions, buttons: 5 binary, shoulder: 5 levels."
        ),
    )
    separate_button_heads: bool = Field(
        default=True,
        description=(
            "If True, use separate heads for each button (A, B, X/Y, Z, L/R) instead of a single "
            "unified button head. Each separate head independently predicts one button. "
            "Effect: May improve button prediction by allowing specialized representations per button. "
            "Disabled by default for backward compatibility."
        ),
    )
    exclude_p1_controller: bool = Field(
        default=True,
        description=(
            "If True, exclude P1 (ego) controller features from model input. "
            "The model will only see P2 (opponent) controller state, not its own previous inputs. "
            "Effect: Forces the model to rely solely on game state without autoregressive controller conditioning. "
            "May reduce training/inference distribution shift but removes useful temporal information. "
            "Disabled by default for backward compatibility."
        ),
    )
    pass_button_hidden_to_stick: bool = Field(
        default=True,
        description=(
            "If True and separate_button_heads is True, pass the hidden activations from each "
            "button head (5 x 128 = 640 dim) to the main_stick_head in addition to the button logits. "
            "This provides richer context about the button heads' reasoning, helping the stick head "
            "coordinate better with button predictions (e.g., for wavedash coordination). "
            "Effect: Increases main_stick_head input size by 640 dimensions. "
            "Only effective when separate_button_heads=True."
        ),
    )

    # TODO: This wasn't working before, so we might have implemented the same logic elsewhere, find it and remove it
    @model_validator(mode="before")
    def compute_input_size(cls, data: Any, info: ValidationInfo):
        """Dynamically compute input_size if context provides dimensions."""
        if not isinstance(data, dict):
            return data

        # Allow explicit overrides to take precedence.
        if "input_size" in data and data["input_size"] not in (-1, None):
            return data

        context = (info.context or {}) if info is not None else {}
        gamestate_dim = context.get("gamestate_dim")
        controller_dim = context.get("controller_dim")

        if gamestate_dim is None or controller_dim is None:
            default_gamestate, default_controller = _schema_feature_dims()
            gamestate_dim = gamestate_dim or default_gamestate
            controller_dim = controller_dim or default_controller

        num_stages = data.get("num_stages", cls.model_fields["num_stages"].default)
        num_characters = data.get(
            "num_characters", cls.model_fields["num_characters"].default
        )
        num_actions = data.get("num_actions", cls.model_fields["num_actions"].default)

        # If excluding P1 controller, subtract 10 features (main_stick: 2, c_stick: 2,
        # buttons: 5, shoulder: 1)
        exclude_p1_controller = data.get(
            "exclude_p1_controller", cls.model_fields["exclude_p1_controller"].default
        )
        if exclude_p1_controller:
            controller_dim = controller_dim - 10

        data = dict(data)
        data["input_size"] = (
            num_stages
            + num_characters * 2
            + num_actions * 2
            + gamestate_dim
            + controller_dim
        )
        return data


__all__ = ["GPTConfig", "_schema_feature_dims"]
