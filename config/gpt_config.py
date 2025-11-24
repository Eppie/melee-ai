from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Literal, Optional, Tuple

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

    block_size: int = Field(default=512, ge=1)
    n_embd: int = Field(default=512, ge=1)
    n_layer: int = Field(default=10, ge=1)
    n_head: int = Field(default=8, ge=1)
    dropout: float = Field(default=0.03, ge=0, le=1)
    input_size: int = Field(default=-1)
    num_stages: int = Field(default=6, ge=1)
    num_characters: int = Field(default=26, ge=1)
    num_actions: int = Field(default=396, ge=1)
    gamma: float = Field(default=0.999, ge=0, le=1)
    n_kv_head: Optional[int] = Field(default=8, ge=1)
    head_flow: Literal["sequential", "parallel"] = "parallel"
    use_head_cross_attention: bool = Field(default=False)
    head_cross_attention_heads: int = Field(default=4, ge=1)
    target_shapes_by_head: Dict[str, int] = Field(
        default_factory=lambda: {
            "main_stick": len(CONTROL_STICK_QUANTIZED),
            "c_stick": len(C_STICK_QUANTIZED),
            "buttons": len(BUTTON_TARGET_NAMES),
            "shoulder": len(SHOULDER_QUANTIZED),
        }
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
