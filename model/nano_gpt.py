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

from constants import CONTROLLER_KEY_GROUPS
from model.attention import CausalSelfAttention
from model.norm import norm
from model.output_head import SimpleHead
from utils import _resolve_device

# Button names for separate heads
BUTTON_NAMES = CONTROLLER_KEY_GROUPS["buttons"]  # ("button_a", "button_b", "button_xy", "button_z", "button_lr")


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

        # Controller output heads - sequential mode where each head receives
        # concatenated outputs from previous heads
        # Order: buttons → main_stick → c_stick → shoulder
        button_input_size = self.embedding_dim
        main_stick_input_size = self.embedding_dim + self.button_output_size
        c_stick_input_size = (
            self.embedding_dim + self.button_output_size + self.main_stick_output_size
        )
        shoulder_input_size = (
            self.embedding_dim
            + self.button_output_size
            + self.main_stick_output_size
            + self.c_stick_output_size
        )

        # Button heads: either unified (single head predicting all 5 buttons)
        # or separate (5 autoregressive heads, each predicting one button)
        # Order: A → B → X/Y → Z → L/R (each sees previous button predictions)
        self.separate_button_heads = model_config.separate_button_heads
        if self.separate_button_heads:
            # Create separate heads for each button with autoregressive input sizes
            # Each head receives hidden_states + all previous button logits
            self.button_heads = nn.ModuleDict(
                {
                    name: SimpleHead(
                        button_input_size + i,  # +i for previous button logits
                        1,
                        hidden=head_hidden_dim,
                    )
                    for i, name in enumerate(BUTTON_NAMES)
                }
            )
            self.button_head = None  # Not used in separate mode
        else:
            # Unified button head (default) - outputs all 5 button logits
            self.button_head = SimpleHead(
                button_input_size, self.button_output_size, hidden=head_hidden_dim
            )
            self.button_heads = None  # Not used in unified mode
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
        device = _resolve_device()
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

        # Sequential heads: each head receives concatenated outputs from previous heads
        # Order: buttons → main_stick → c_stick → shoulder
        if self.separate_button_heads:
            # Autoregressive button heads: each head sees previous button predictions
            # Order: A → B → X/Y → Z → L/R
            button_logit_list = []
            button_input = hidden_states
            for name in BUTTON_NAMES:
                logit = self.button_heads[name](button_input)  # [B, L, 1]
                button_logit_list.append(logit)
                # Next head receives hidden_states + all previous logits (detached)
                button_input = torch.cat(
                    [hidden_states] + [l.detach() for l in button_logit_list], dim=-1
                )
            button_logits = torch.cat(button_logit_list, dim=-1)  # [B, L, 5]
        else:
            # Unified button head (default)
            button_logits = self.button_head(hidden_states)

        main_stick = self.main_stick_head(
            torch.cat((hidden_states, button_logits.detach()), dim=-1)
        )

        c_stick = self.c_stick_head(
            torch.cat(
                (hidden_states, button_logits.detach(), main_stick.detach()),
                dim=-1,
            )
        )

        shoulder = self.shoulder_head(
            torch.cat(
                (
                    hidden_states,
                    button_logits.detach(),
                    main_stick.detach(),
                    c_stick.detach(),
                ),
                dim=-1,
            )
        )

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
