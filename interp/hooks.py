"""
Universal activation hooking system for the Nano-Melee GPT model.

Provides standardized hook points for extracting activations at various
locations in the model architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from model.nano_gpt import GPT


class HookPointType(Enum):
    """Types of hook points in the model architecture."""

    # Input processing
    POST_EMBED = "post_embed"  # After _embed_inputs concatenation
    POST_PROJECTION = "post_projection"  # After projection_down

    # Transformer blocks
    BLOCK_OUTPUT = "block_output"  # After full block (attn + mlp + residuals)
    BLOCK_POST_ATTN = "block_post_attn"  # After attention + residual
    BLOCK_POST_MLP = "block_post_mlp"  # After MLP (same as block_output)

    # MLP internals
    MLP_PRE_ACT = "mlp_pre_act"  # After fully_connected, before ReLU²
    MLP_POST_ACT = "mlp_post_act"  # After ReLU², before output_projection

    # Final processing
    FINAL_NORM = "final_norm"  # After final RMSNorm, before heads

    # Output heads
    HEAD_BUTTONS = "head_buttons"
    HEAD_MAIN_STICK = "head_main_stick"
    HEAD_C_STICK = "head_c_stick"
    HEAD_SHOULDER = "head_shoulder"
    HEAD_VALUE = "head_value"


@dataclass(frozen=True)
class HookPoint:
    """Specification of a single hook location in the model."""

    hook_type: HookPointType
    layer_idx: Optional[int] = None  # For block/MLP hooks

    def __str__(self) -> str:
        if self.layer_idx is not None:
            return f"{self.hook_type.value}_L{self.layer_idx}"
        return self.hook_type.value

    @property
    def is_block_hook(self) -> bool:
        """Whether this hook requires a layer index."""
        return self.hook_type in {
            HookPointType.BLOCK_OUTPUT,
            HookPointType.BLOCK_POST_ATTN,
            HookPointType.BLOCK_POST_MLP,
            HookPointType.MLP_PRE_ACT,
            HookPointType.MLP_POST_ACT,
        }


# Common hook point configurations
def all_block_outputs(n_layers: int) -> List[HookPoint]:
    """Get hook points for all block outputs."""
    return [HookPoint(HookPointType.BLOCK_OUTPUT, i) for i in range(n_layers)]


def residual_stream_hooks(n_layers: int) -> List[HookPoint]:
    """Get hook points for the full residual stream (input + all blocks + final)."""
    hooks = [HookPoint(HookPointType.POST_PROJECTION)]
    hooks.extend(all_block_outputs(n_layers))
    hooks.append(HookPoint(HookPointType.FINAL_NORM))
    return hooks


def mlp_activation_hooks(n_layers: int) -> List[HookPoint]:
    """Get hook points for MLP activations (post-ReLU²) at each layer."""
    return [HookPoint(HookPointType.MLP_POST_ACT, i) for i in range(n_layers)]


class HookManager:
    """
    Manages forward hooks on a GPT model for activation extraction.

    Usage:
        manager = HookManager(model)
        manager.install_hooks([
            HookPoint(HookPointType.BLOCK_OUTPUT, layer_idx=4),
            HookPoint(HookPointType.FINAL_NORM),
        ])

        with torch.no_grad():
            outputs = model(inputs)

        activations = manager.get_activations()
        manager.clear()  # Clear stored activations
        manager.remove_hooks()  # Remove all hooks
    """

    def __init__(self, model: "GPT"):
        self.model = model
        self._hooks: Dict[HookPoint, torch.utils.hooks.RemovableHandle] = {}
        self._activations: Dict[HookPoint, List[Tensor]] = {}
        self._installed_points: List[HookPoint] = []

    def install_hooks(self, hook_points: List[HookPoint]) -> None:
        """Install forward hooks at specified locations."""
        self.remove_hooks()  # Clear any existing hooks

        for point in hook_points:
            module = self._get_module(point)
            if module is None:
                raise ValueError(f"Cannot find module for hook point: {point}")

            hook_fn = self._make_hook_fn(point)
            handle = module.register_forward_hook(hook_fn)
            self._hooks[point] = handle
            self._activations[point] = []
            self._installed_points.append(point)

    def _get_module(self, point: HookPoint) -> Optional[nn.Module]:
        """Get the module to hook for a given hook point."""
        model = self.model

        if point.hook_type == HookPointType.POST_EMBED:
            # Hook the projection_down layer to capture post-embed
            # (We'll hook the input to projection_down via a pre-hook later if needed)
            return None  # Special handling needed

        if point.hook_type == HookPointType.POST_PROJECTION:
            return model.projection_down

        if point.hook_type == HookPointType.FINAL_NORM:
            # The model applies norm() inline, not as a module
            # We'll hook the last block and apply norm manually
            return model.blocks[-1]

        if point.hook_type == HookPointType.BLOCK_OUTPUT:
            if point.layer_idx is None:
                raise ValueError("BLOCK_OUTPUT requires layer_idx")
            return model.blocks[point.layer_idx]

        if point.hook_type == HookPointType.MLP_POST_ACT:
            if point.layer_idx is None:
                raise ValueError("MLP_POST_ACT requires layer_idx")
            return model.blocks[point.layer_idx].mlp.fully_connected

        # Output heads
        if point.hook_type == HookPointType.HEAD_BUTTONS:
            return model.button_head
        if point.hook_type == HookPointType.HEAD_MAIN_STICK:
            return model.main_stick_head
        if point.hook_type == HookPointType.HEAD_C_STICK:
            return model.c_stick_head
        if point.hook_type == HookPointType.HEAD_SHOULDER:
            return model.shoulder_head
        if point.hook_type == HookPointType.HEAD_VALUE:
            return model.value_head

        return None

    def _make_hook_fn(
        self, point: HookPoint
    ) -> Callable[[nn.Module, Tuple[Tensor, ...], Tensor], None]:
        """Create a hook function for a specific hook point."""

        def hook_fn(
            module: nn.Module, inputs: Tuple[Tensor, ...], output: Tensor
        ) -> None:
            # For MLP_POST_ACT, we need to capture after ReLU², not just after linear
            if point.hook_type == HookPointType.MLP_POST_ACT:
                # The hook is on fully_connected, so output is pre-activation
                # Apply ReLU² manually to get post-activation
                import torch.nn.functional as F

                post_act = F.relu(output).square()
                self._activations[point].append(post_act.detach())
            else:
                self._activations[point].append(output.detach())

        return hook_fn

    def get_activations(self, point: Optional[HookPoint] = None) -> Dict[HookPoint, Tensor]:
        """
        Get stored activations, concatenated along batch dimension.

        Args:
            point: Specific hook point to get, or None for all

        Returns:
            Dict mapping hook points to concatenated activation tensors
        """
        if point is not None:
            if point not in self._activations:
                raise ValueError(f"No activations stored for {point}")
            acts = self._activations[point]
            if not acts:
                raise ValueError(f"No activations captured for {point}")
            return {point: torch.cat(acts, dim=0)}

        result = {}
        for p, acts in self._activations.items():
            if acts:
                result[p] = torch.cat(acts, dim=0)
        return result

    def get_single(self, point: HookPoint) -> Tensor:
        """Get activations for a single hook point."""
        result = self.get_activations(point)
        return result[point]

    def clear(self) -> None:
        """Clear stored activations without removing hooks."""
        for acts in self._activations.values():
            acts.clear()

    def remove_hooks(self) -> None:
        """Remove all installed hooks and clear activations."""
        for handle in self._hooks.values():
            handle.remove()
        self._hooks.clear()
        self._activations.clear()
        self._installed_points.clear()

    @property
    def installed_points(self) -> List[HookPoint]:
        """List of currently installed hook points."""
        return list(self._installed_points)

    def __enter__(self) -> "HookManager":
        return self

    def __exit__(self, *args) -> None:
        self.remove_hooks()


__all__ = [
    "HookPointType",
    "HookPoint",
    "HookManager",
    "all_block_outputs",
    "residual_stream_hooks",
    "mlp_activation_hooks",
]
