from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


class ValueNetworkConfig(BaseModel):
    """Configuration for the separate value network with LSTM + FFW ResBlock architecture."""

    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    enabled: bool = Field(
        default=True,
        description=(
            "Whether to use the separate value network. If False, value estimation is disabled."
        ),
    )

    hidden_dim: int = Field(
        default=256,
        ge=64,
        description=(
            "Hidden dimension for encoder, LSTM, and FFW blocks. "
            "Smaller than policy transformer to reduce overhead."
        ),
    )

    ffw_expansion: int = Field(
        default=2,
        ge=1,
        description=(
            "FFW expansion factor. hidden_dim * expansion = intermediate dimension. "
            "Default 2 gives 512 -> 1024 -> 512."
        ),
    )

    lstm_num_layers: int = Field(
        default=1,
        ge=1,
        description="Number of LSTM layers in the residual LSTM block.",
    )

    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Dropout probability for LSTM and FFW blocks.",
    )

    grad_clip: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Gradient clipping max norm for value network parameters. "
            "Separate from policy model gradient clipping to prevent value network "
            "from corrupting training if it diverges."
        ),
    )
