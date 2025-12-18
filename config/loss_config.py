from __future__ import annotations

from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


class LossConfig(BaseModel):
    """Configuration related to loss computation and weighting."""

    model_config = SettingsConfigDict(validate_assignment=True, extra="forbid")

    # Class balancing controls
    enable_class_balancing: bool = Field(
        default=False,
        description=(
            "Enable class balancing for cross-entropy losses (stick/shoulder outputs). "
            "When enabled, rare classes get higher weights. Helps with imbalanced datasets "
            "where some stick positions/shoulder values are much more common than others."
        ),
    )
    ce_weight_min: float = Field(
        default=0.5,
        gt=0,
        description=(
            "Minimum class weight for cross-entropy balancing. Prevents over-penalizing very common classes. "
            "Effect: Lower values (0.01-0.1) allow more aggressive down-weighting of common classes; "
            "higher values (0.2-0.5) keep more balanced weighting. Reasonable range: [0.01, 0.5]. "
            "Interacts with: ce_weight_max (defines weight range), enable_class_balancing."
        ),
    )
    ce_weight_max: float = Field(
        default=3.0,
        gt=0,
        description=(
            "Maximum class weight for cross-entropy balancing. Prevents over-emphasizing very rare classes. "
            "Effect: Higher values (10.0-50.0) boost rare classes more; lower values (2.0-5.0) limit boosting. "
            "Reasonable range: [2.0, 50.0]. Interacts with: ce_weight_min, enable_class_balancing."
        ),
    )
    enable_pos_weighting: bool = Field(
        default=True,
        description=(
            "Enable positive class weighting for binary cross-entropy (button outputs). "
            "When enabled, button presses (positive class) get weighted by neg/pos ratio. "
            "Helps when buttons are pressed rarely (e.g., Z button)."
        ),
    )
    pos_weight_max: float = Field(
        default=4.0,
        gt=0,
        description=(
            "Maximum positive class weight for BCE. Caps how much button presses are up-weighted. "
            "Effect: Higher values (5.0-10.0) boost rare button presses more; lower values (1.5-3.0) limit boosting. "
            "Reasonable range: [1.0, 10.0]. Interacts with: enable_pos_weighting, button-specific weights."
        ),
    )
    use_weighted_component_means: bool = Field(
        default=True,
        description=(
            "Use weighted mean (by sample weights) when averaging loss components. "
            "When True, samples with higher weights contribute more to final loss. "
            "Recommended: True for change-based weighting to work properly."
        ),
    )

    # Change-based loss weights (multiplied by sample_weights from change detection)
    main_change: float = Field(
        default=10.0,
        gt=0,
        description=(
            "Weight multiplier for main stick when it changes. Encourages model to predict stick movements. "
            "Effect: Higher values (10.0-20.0) = stronger focus on movement vs holding position; "
            "lower values (2.0-5.0) = more balanced. Reasonable range: [2.0, 20.0]. "
            "Interacts with: hold_base (weight for unchanged frames), imbalance_scale (global scaling)."
        ),
    )
    c_change: float = Field(
        default=10.0,
        gt=0,
        description=(
            "Weight multiplier for C-stick when it changes. C-stick moves are rare and important (smash attacks). "
            "Effect: Higher values (15.0-30.0) heavily prioritize C-stick usage; lower values (5.0-10.0) reduce emphasis. "
            "Reasonable range: [5.0, 30.0]. Usually higher than main_change since C-stick is rarer."
        ),
    )
    shoulder_change: float = Field(
        default=2.0,
        gt=0,
        description=(
            "Weight multiplier for shoulder buttons when they change. "
            "Effect: Higher values (10.0-20.0) emphasize shield/lightshield/airdodge initiation; "
            "lower values (2.0-5.0) reduce emphasis. Reasonable range: [2.0, 20.0]."
        ),
    )
    buttons_change_default: float = Field(
        default=5.0,
        gt=0,
        description=(
            "Default weight multiplier for button state changes (when no button-specific weight applies). "
            "Fallback for any buttons without explicit weights. Reasonable range: [5.0, 20.0]."
        ),
    )

    # Button-specific weights (applied when button state changes)
    button_z: float = Field(
        default=10.0,
        gt=0,
        description=(
            "Weight for Z button (grab). Z is rare but critical. "
            "Effect: Higher values (20.0-40.0) heavily prioritize grab usage; lower values (10.0-15.0) reduce emphasis. "
            "Reasonable range: [10.0, 40.0]. Often highest button weight due to rarity and importance."
        ),
    )
    button_b: float = Field(
        default=8.0,
        gt=0,
        description=(
            "Weight for B button (special moves). Important for recovery, projectiles, etc. "
            "Reasonable range: [8.0, 20.0]."
        ),
    )
    button_a: float = Field(
        default=8.0,
        gt=0,
        description=(
            "Weight for A button (standard attacks). Common but important. "
            "Reasonable range: [8.0, 20.0]."
        ),
    )
    button_xy: float = Field(
        default=5.0,
        gt=0,
        description=(
            "Weight for X/Y buttons (jump). Very common, slightly lower weight. "
            "Reasonable range: [5.0, 15.0]."
        ),
    )
    button_lr: float = Field(
        default=2.0,
        gt=0,
        description=(
            "Weight for L/R digital press (shield/airdodge when combined with shoulder analog). "
            "Reasonable range: [5.0, 15.0]."
        ),
    )

    # Base weights
    hold_base: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Base weight for frames where controls don't change (holding position). "
            "Effect: Higher values (2.0-5.0) increase importance of maintaining state; "
            "lower values (0.5-1.0) de-emphasize holding. Reasonable range: [0.5, 5.0]. "
            "Usually kept at 1.0, with change weights being the main tuning knob. "
            "Interacts with: all *_change weights (defines change vs hold tradeoff)."
        ),
    )
    value_change: float = Field(
        default=1.0,
        gt=0,
        description=(
            "Weight for value head loss. Value head predicts future rewards. "
            "Usually kept at 1.0; see rl_config.value_loss_coef for main value loss scaling. "
            "Reasonable range: [0.5, 2.0]."
        ),
    )
