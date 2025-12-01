"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Multi-Query Attention (MQA) support for more efficient inference
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from model.attention import CausalSelfAttention
from model.head_cross_attention import HeadCrossAttention
from model.norm import norm
from model.output_head import SimpleHead
from utils import _resolve_device


class MLP(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.fully_connected = nn.Linear(embedding_dim, 4 * embedding_dim, bias=False)
        self.output_projection = nn.Linear(4 * embedding_dim, embedding_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fully_connected(hidden_states)
        hidden_states = F.relu(hidden_states).square()
        hidden_states = self.output_projection(hidden_states)
        return hidden_states


class Block(nn.Module):
    def __init__(self, embedding_dim, num_heads, num_key_value_heads, dropout):
        super().__init__()
        self.attention = CausalSelfAttention(
            embedding_dim, num_heads, num_key_value_heads, dropout
        )
        self.mlp = MLP(embedding_dim)
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(
        self, hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(norm(hidden_states), cos, sin)
        hidden_states = hidden_states + self.mlp_dropout(self.mlp(norm(hidden_states)))
        return hidden_states


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = config.model
        self.block_size = model_config.block_size
        self.embedding_dim: int = model_config.n_embd
        self.input_size: int = model_config.input_size

        self.projection_down = nn.Linear(self.input_size, self.embedding_dim, bias=True)
        self.dropout = nn.Dropout(model_config.dropout)

        self.blocks = nn.ModuleList(
            [
                Block(
                    self.embedding_dim,
                    model_config.n_head,
                    model_config.n_kv_head,
                    model_config.dropout,
                )
                for _ in range(model_config.n_layer)
            ]
        )

        self.target_shapes_by_head: dict[str, int] = model_config.target_shapes_by_head
        self.shoulder_output_size = self.target_shapes_by_head["shoulder"]
        self.c_stick_output_size = self.target_shapes_by_head["c_stick"]
        self.main_stick_output_size = self.target_shapes_by_head["main_stick"]
        self.button_output_size = self.target_shapes_by_head["buttons"]

        # TODO: Move this to config
        head_hidden_dim = 128

        # Controller output heads - input sizes depend on head_flow mode
        self.head_flow = model_config.head_flow

        if self.head_flow == "sequential":
            # Sequential mode: each head receives concatenated outputs from previous heads
            # Order: buttons → main_stick → c_stick → shoulder
            button_input_size = self.embedding_dim
            main_stick_input_size = self.embedding_dim + self.button_output_size
            c_stick_input_size = (
                self.embedding_dim
                + self.button_output_size
                + self.main_stick_output_size
            )
            shoulder_input_size = (
                self.embedding_dim
                + self.button_output_size
                + self.main_stick_output_size
                + self.c_stick_output_size
            )
        else:
            # Parallel and mix modes: all heads receive same base features
            button_input_size = self.embedding_dim
            main_stick_input_size = self.embedding_dim
            c_stick_input_size = self.embedding_dim
            shoulder_input_size = self.embedding_dim

        self.button_head = SimpleHead(
            button_input_size, self.button_output_size, hidden=head_hidden_dim
        )
        self.main_stick_head = SimpleHead(
            main_stick_input_size, self.main_stick_output_size, hidden=head_hidden_dim
        )
        self.c_stick_head = SimpleHead(
            c_stick_input_size, self.c_stick_output_size, hidden=head_hidden_dim
        )
        self.shoulder_head = SimpleHead(
            shoulder_input_size, self.shoulder_output_size, hidden=head_hidden_dim
        )
        self.value_head = SimpleHead(self.embedding_dim, 1, hidden=head_hidden_dim * 2)

        # Cross-attention for heads (only used in "mix" mode)
        if self.head_flow == "mix":
            self.head_cross_attention = HeadCrossAttention(
                hidden_dim=head_hidden_dim,
                num_head_types=4,  # buttons, main_stick, c_stick, shoulder
                num_attn_heads=4,
            )

        # TODO: Do we need this multiplier?
        self.rotary_sequence_length = self.block_size * 2
        head_dim = model_config.n_embd // model_config.n_head
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_sequence_length, head_dim
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)

        for block in self.blocks:
            torch.nn.init.zeros_(block.mlp.output_projection.weight)
            torch.nn.init.zeros_(block.attention.output_projection.weight)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, sequence_length, head_dim, base=256.0):
        # Create on CPU to avoid CUDA initialization in forked processes
        # Tensors will be moved to the correct device when model.to(device) is called
        device = torch.device("cpu")
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inverse_frequency = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        timesteps = torch.arange(sequence_length, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        frequencies = torch.outer(timesteps, inverse_frequency)
        cos, sin = frequencies.cos(), frequencies.sin()
        cos, sin = cos.to(torch.float32), sin.to(torch.float32)
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims for later broadcasting
        return cos, sin

    # TODO: Is there a way to pre-compute and cache the one-hot results?
    # TODO: Why do we need the `.long()` calls?
    def _embed_inputs(self, inputs: TensorDict) -> torch.Tensor:
        """Includes categorical embeddings, one-hot encodings, and numerical features."""
        return torch.cat(
            [
                F.one_hot(
                    inputs["stage"].squeeze(-1).long(),
                    num_classes=self.config.model.num_stages,
                ).float(),
                F.one_hot(
                    inputs["ego_character"].squeeze(-1).long(),
                    num_classes=self.config.model.num_characters,
                ).float(),
                F.one_hot(
                    inputs["opponent_character"].squeeze(-1).long(),
                    num_classes=self.config.model.num_characters,
                ).float(),
                F.one_hot(
                    inputs["ego_action"].squeeze(-1).long(),
                    num_classes=self.config.model.num_actions,
                ).float(),
                F.one_hot(
                    inputs["opponent_action"].squeeze(-1).long(),
                    num_classes=self.config.model.num_actions,
                ).float(),
                inputs["gamestate"],
                inputs["controller"],
            ],
            dim=-1,
        )

    def forward(self, inputs: TensorDict) -> TensorDict:
        batch_size, sequence_length, _ = inputs["gamestate"].shape
        assert (
            sequence_length <= self.block_size
        ), f"Cannot forward sequence of length {sequence_length}, block size is only {self.block_size}"

        combined_inputs = self._embed_inputs(inputs)
        hidden_states = self.projection_down(combined_inputs)
        hidden_states = self.dropout(hidden_states)
        cos = self.cos[:, :sequence_length]
        sin = self.sin[:, :sequence_length]

        for block in self.blocks:
            hidden_states = block(hidden_states, cos, sin)

        hidden_states = norm(hidden_states)

        base_hidden_states = hidden_states

        if self.head_flow == "parallel":
            # Parallel heads: each head operates independently on base features
            button_logits = self.button_head(base_hidden_states)
            main_stick = self.main_stick_head(base_hidden_states)
            c_stick = self.c_stick_head(base_hidden_states)
            shoulder = self.shoulder_head(base_hidden_states)

        elif self.head_flow == "mix":
            # Mix mode: cross-attention between head intermediate features
            # Get intermediate features from each head
            button_features = self.button_head.forward_intermediate(base_hidden_states)
            main_stick_features = self.main_stick_head.forward_intermediate(
                base_hidden_states
            )
            c_stick_features = self.c_stick_head.forward_intermediate(
                base_hidden_states
            )
            shoulder_features = self.shoulder_head.forward_intermediate(
                base_hidden_states
            )

            # Apply cross-attention across heads
            head_features_list = [
                button_features,
                main_stick_features,
                c_stick_features,
                shoulder_features,
            ]
            attended_features = self.head_cross_attention(head_features_list)

            # Project to final outputs from attended features
            button_logits = self.button_head.forward_from_intermediate(
                attended_features[0]
            )
            main_stick = self.main_stick_head.forward_from_intermediate(
                attended_features[1]
            )
            c_stick = self.c_stick_head.forward_from_intermediate(attended_features[2])
            shoulder = self.shoulder_head.forward_from_intermediate(
                attended_features[3]
            )

        elif self.head_flow == "sequential":
            # Sequential heads: each head receives concatenated outputs from previous heads
            # Order: buttons → main_stick → c_stick → shoulder
            button_logits = self.button_head(base_hidden_states)

            main_stick = self.main_stick_head(
                torch.cat((base_hidden_states, button_logits.detach()), dim=-1)
            )

            c_stick = self.c_stick_head(
                torch.cat(
                    (base_hidden_states, button_logits.detach(), main_stick.detach()),
                    dim=-1,
                )
            )

            shoulder = self.shoulder_head(
                torch.cat(
                    (
                        base_hidden_states,
                        button_logits.detach(),
                        main_stick.detach(),
                        c_stick.detach(),
                    ),
                    dim=-1,
                )
            )

        else:
            raise ValueError(f"Unknown head_flow mode: {self.head_flow}")

        outputs = TensorDict(
            {
                "buttons": button_logits,
                "main_stick": main_stick,
                "c_stick": c_stick,
                "shoulder": shoulder,
            },
            batch_size=(batch_size, sequence_length),
        )

        value = self.value_head(hidden_states)
        outputs.set("value", value)

        return outputs
