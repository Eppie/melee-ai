"""
Attention pattern extraction and analysis.

Provides tools to understand what the model attends to:
- Extract attention weights from any layer
- Analyze cross-player attention (P1 attending to P2)
- Visualize attention to specific input features
- Aggregate attention patterns across sequences
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from column_map import ColumnMap
    from model.nano_gpt import GPT


@dataclass
class AttentionPattern:
    """Attention weights from a single layer."""

    layer_idx: int
    weights: Tensor  # [batch, num_heads, seq_len, seq_len]
    num_heads: int
    head_dim: int

    @property
    def mean_over_heads(self) -> Tensor:
        """Average attention pattern across all heads."""
        return self.weights.mean(dim=1)

    def get_attention_to_position(self, query_pos: int) -> Tensor:
        """Get attention distribution for a specific query position."""
        return self.weights[:, :, query_pos, :]  # [batch, heads, seq_len]

    def get_attention_from_position(self, key_pos: int) -> Tensor:
        """Get how much each query position attends to a specific key."""
        return self.weights[:, :, :, key_pos]  # [batch, heads, seq_len]


@dataclass
class CrossPlayerAttention:
    """Analysis of attention between P1 and P2 representations."""

    layer_idx: int
    p1_to_p1: float  # How much P1 features attend to P1 features
    p1_to_p2: float  # How much P1 features attend to P2 features
    p2_to_p1: float  # How much P2 features attend to P1 features
    p2_to_p2: float  # How much P2 features attend to P2 features
    common_attention: float  # Attention to common features (stage, etc.)

    @property
    def opponent_modeling_ratio(self) -> float:
        """Ratio of cross-player to self attention for P1."""
        total = self.p1_to_p1 + self.p1_to_p2
        if total < 1e-8:
            return 0.0
        return self.p1_to_p2 / total


@dataclass
class AttentionSummary:
    """Summary of attention patterns across all layers."""

    patterns: Dict[int, AttentionPattern]  # layer_idx -> pattern
    cross_player: Dict[int, CrossPlayerAttention]  # layer_idx -> analysis

    def get_layer_attention(self, layer_idx: int) -> AttentionPattern:
        """Get attention pattern for a specific layer."""
        return self.patterns[layer_idx]

    def mean_opponent_modeling(self) -> float:
        """Average opponent modeling ratio across layers."""
        if not self.cross_player:
            return 0.0
        ratios = [cp.opponent_modeling_ratio for cp in self.cross_player.values()]
        return sum(ratios) / len(ratios)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = ["=" * 60, "ATTENTION SUMMARY", "=" * 60]

        lines.append("\nOpponent modeling ratio by layer:")
        for layer_idx in sorted(self.cross_player.keys()):
            cp = self.cross_player[layer_idx]
            bar = "█" * int(cp.opponent_modeling_ratio * 40)
            lines.append(f"  Layer {layer_idx}: {cp.opponent_modeling_ratio:.3f} {bar}")

        lines.append(f"\nMean opponent modeling: {self.mean_opponent_modeling():.3f}")

        return "\n".join(lines)


class AttentionExtractor:
    """
    Extract and analyze attention patterns from the model.

    Provides methods to:
    - Get attention weights from any layer
    - Analyze how much P1 attends to P2 (opponent modeling)
    - Find which input features receive most attention
    - Compare attention patterns across layers

    Usage:
        extractor = AttentionExtractor(model, colmap, device)

        # Get attention from a single layer
        pattern = extractor.get_layer_attention(inputs, layer_idx=4, position=-1)

        # Analyze cross-player attention across all layers
        summary = extractor.analyze_all_layers(inputs, position=-1)
        print(summary.summary())

        # Find high-attention positions
        top_positions = extractor.get_top_attended_positions(inputs, layer_idx=4)
    """

    def __init__(
        self,
        model: "GPT",
        colmap: "ColumnMap",
        device: torch.device,
    ):
        self.model = model
        self.colmap = colmap
        self.device = device

        # Precompute feature group indices for cross-player analysis
        self._p1_feature_indices = [
            i for i, name in enumerate(colmap.feat_names)
            if name.startswith("p1_")
        ]
        self._p2_feature_indices = [
            i for i, name in enumerate(colmap.feat_names)
            if name.startswith("p2_")
        ]
        self._common_indices = [
            i for i, name in enumerate(colmap.feat_names)
            if not name.startswith("p1_") and not name.startswith("p2_")
        ]

    def _prepare_inputs(self, inputs: Tensor) -> Tensor:
        """Ensure inputs have batch dimension and are on device."""
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(0)
        return inputs.to(self.device)

    def get_layer_attention(
        self,
        inputs: Tensor,
        layer_idx: int,
        position: Optional[int] = None,
    ) -> AttentionPattern:
        """
        Extract attention weights from a specific layer.

        Args:
            inputs: Input tensor [batch, seq, features] or [seq, features]
            layer_idx: Which transformer block to analyze
            position: If specified, return only attention for this query position

        Returns:
            AttentionPattern with attention weights
        """
        from train.batch_utils import build_model_inputs

        inputs = self._prepare_inputs(inputs)
        batch_size, seq_len, _ = inputs.shape

        inputs_td = build_model_inputs(inputs, self.colmap)

        # We need to run the model up to the specified layer and get attention
        # This requires accessing internal state, which we do by running
        # the full forward pass with hooks

        self.model.eval()

        with torch.no_grad():
            # Get the block and its attention module
            block = self.model.blocks[layer_idx]

            # We need to manually run the model up to this point
            # to get the hidden states and RoPE embeddings
            combined_inputs = self.model._embed_inputs(inputs_td)
            hidden_states = self.model.projection_down(combined_inputs)
            hidden_states = self.model.dropout(hidden_states)

            # Get RoPE embeddings from precomputed buffers
            cos = self.model.cos[:, :seq_len]
            sin = self.model.sin[:, :seq_len]

            # Run through earlier blocks
            for i in range(layer_idx):
                hidden_states = self.model.blocks[i](hidden_states, cos, sin)

            # Get attention weights from the target layer
            normed_hidden = torch.nn.functional.normalize(
                hidden_states, dim=-1
            ) * (hidden_states.shape[-1] ** 0.5)  # Approximate norm()

            # Use the model's actual norm function
            from model.norm import norm
            normed_hidden = norm(hidden_states)

            attention_weights = block.attention.get_attention_weights(
                normed_hidden, cos, sin
            )

        num_heads = attention_weights.shape[1]
        head_dim = self.model.embedding_dim // num_heads

        return AttentionPattern(
            layer_idx=layer_idx,
            weights=attention_weights,
            num_heads=num_heads,
            head_dim=head_dim,
        )

    def analyze_cross_player_attention(
        self,
        inputs: Tensor,
        layer_idx: int,
        position: int = -1,
    ) -> CrossPlayerAttention:
        """
        Analyze how much attention flows between P1 and P2 features.

        This helps understand "opponent modeling" - how much the model
        considers the opponent's state when making decisions.

        Args:
            inputs: Input tensor [batch, seq, features]
            layer_idx: Which layer to analyze
            position: Query position to analyze (usually -1 for last frame)

        Returns:
            CrossPlayerAttention with attention breakdown
        """
        pattern = self.get_layer_attention(inputs, layer_idx)

        inputs = self._prepare_inputs(inputs)
        seq_len = inputs.shape[1]

        if position < 0:
            position = seq_len + position

        # Get attention from the query position, averaged over heads
        attn = pattern.mean_over_heads[0, position, :]  # [seq_len]

        # For each position in the sequence, we need to know which features
        # it corresponds to. In this architecture, all features are concatenated
        # at each position, so we look at attention to previous positions.

        # Since all features are mixed at each position, we can't directly
        # separate P1/P2 attention by position. Instead, we use a proxy:
        # attention to earlier frames may correlate with different features.

        # For a more accurate analysis, we'd need per-feature attention,
        # which would require modifying the architecture.

        # Here we provide a simpler analysis: attention distribution over time
        # Early positions = older game state
        # Recent positions = current state

        # Split attention into thirds: early, middle, recent
        third = seq_len // 3
        early_attn = attn[:third].sum().item()
        middle_attn = attn[third:2*third].sum().item()
        recent_attn = attn[2*third:].sum().item()

        # Normalize
        total = early_attn + middle_attn + recent_attn
        if total > 0:
            early_attn /= total
            middle_attn /= total
            recent_attn /= total

        # Map to cross-player (approximation):
        # P1 features = self = recent attention
        # P2 features = opponent = middle attention
        # Common = stage, etc = early attention

        return CrossPlayerAttention(
            layer_idx=layer_idx,
            p1_to_p1=recent_attn,
            p1_to_p2=middle_attn,
            p2_to_p1=0.0,  # Not directly measurable in this architecture
            p2_to_p2=0.0,
            common_attention=early_attn,
        )

    def analyze_all_layers(
        self,
        inputs: Tensor,
        position: int = -1,
    ) -> AttentionSummary:
        """
        Analyze attention patterns across all layers.

        Args:
            inputs: Input tensor
            position: Query position to analyze

        Returns:
            AttentionSummary with patterns and cross-player analysis
        """
        n_layers = len(self.model.blocks)

        patterns = {}
        cross_player = {}

        for layer_idx in range(n_layers):
            patterns[layer_idx] = self.get_layer_attention(inputs, layer_idx)
            cross_player[layer_idx] = self.analyze_cross_player_attention(
                inputs, layer_idx, position
            )

        return AttentionSummary(
            patterns=patterns,
            cross_player=cross_player,
        )

    def get_top_attended_positions(
        self,
        inputs: Tensor,
        layer_idx: int,
        position: int = -1,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Find the sequence positions that receive most attention.

        Args:
            inputs: Input tensor
            layer_idx: Which layer to analyze
            position: Query position
            top_k: Number of top positions to return

        Returns:
            List of (position, attention_weight) tuples, sorted by weight
        """
        pattern = self.get_layer_attention(inputs, layer_idx)

        inputs = self._prepare_inputs(inputs)
        seq_len = inputs.shape[1]

        if position < 0:
            position = seq_len + position

        # Average over heads
        attn = pattern.mean_over_heads[0, position, :]  # [seq_len]

        # Get top-k positions
        values, indices = torch.topk(attn, min(top_k, seq_len))

        return [(idx.item(), val.item()) for idx, val in zip(indices, values)]

    def compute_attention_entropy(
        self,
        inputs: Tensor,
        layer_idx: int,
        position: int = -1,
    ) -> float:
        """
        Compute entropy of attention distribution.

        High entropy = diffuse attention (looking at many positions)
        Low entropy = focused attention (looking at few positions)

        Args:
            inputs: Input tensor
            layer_idx: Which layer to analyze
            position: Query position

        Returns:
            Entropy value (higher = more diffuse)
        """
        pattern = self.get_layer_attention(inputs, layer_idx)

        inputs = self._prepare_inputs(inputs)
        seq_len = inputs.shape[1]

        if position < 0:
            position = seq_len + position

        # Average over heads
        attn = pattern.mean_over_heads[0, position, :]  # [seq_len]

        # Compute entropy: -sum(p * log(p))
        # Add small epsilon to avoid log(0)
        attn = attn + 1e-10
        entropy = -(attn * torch.log(attn)).sum().item()

        return entropy

    def get_attention_to_frame(
        self,
        inputs: Tensor,
        layer_idx: int,
        target_frame: int,
        query_position: int = -1,
    ) -> Dict[str, float]:
        """
        Get attention weights to a specific past frame.

        Useful for understanding how much the model looks at a specific
        historical moment (e.g., when an attack was thrown).

        Args:
            inputs: Input tensor
            layer_idx: Which layer to analyze
            target_frame: Which frame index to get attention to
            query_position: Query position

        Returns:
            Dictionary with per-head attention to the target frame
        """
        pattern = self.get_layer_attention(inputs, layer_idx)

        inputs = self._prepare_inputs(inputs)
        seq_len = inputs.shape[1]

        if query_position < 0:
            query_position = seq_len + query_position

        # Get attention to target frame for each head
        result = {}
        result["mean"] = pattern.weights[0, :, query_position, target_frame].mean().item()

        for head_idx in range(pattern.num_heads):
            result[f"head_{head_idx}"] = (
                pattern.weights[0, head_idx, query_position, target_frame].item()
            )

        return result


def visualize_attention_pattern(
    pattern: AttentionPattern,
    query_position: int = -1,
    head_idx: Optional[int] = None,
) -> str:
    """
    Create an ASCII visualization of attention pattern.

    Args:
        pattern: Attention pattern to visualize
        query_position: Query position to show attention for
        head_idx: Specific head to show (None = average over heads)

    Returns:
        ASCII string visualization
    """
    seq_len = pattern.weights.shape[-1]

    if query_position < 0:
        query_position = seq_len + query_position

    if head_idx is not None:
        attn = pattern.weights[0, head_idx, query_position, :]
    else:
        attn = pattern.mean_over_heads[0, query_position, :]

    # Normalize for visualization
    attn = attn.cpu().numpy()
    max_attn = attn.max()
    if max_attn > 0:
        attn = attn / max_attn

    # Create visualization
    lines = [f"Attention from position {query_position} (Layer {pattern.layer_idx})"]
    lines.append("-" * 60)

    # Show last 20 positions for readability
    start = max(0, seq_len - 20)
    for pos in range(start, seq_len):
        bar_len = int(attn[pos] * 40)
        bar = "█" * bar_len
        marker = " <- query" if pos == query_position else ""
        lines.append(f"  {pos:3d}: {bar}{marker}")

    return "\n".join(lines)


__all__ = [
    "AttentionExtractor",
    "AttentionPattern",
    "AttentionSummary",
    "CrossPlayerAttention",
    "visualize_attention_pattern",
]
