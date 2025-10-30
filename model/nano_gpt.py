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
from model.norm import norm
from model.output_head import SimpleHead, ButtonHead
from model.value_head import ValueHead
from utils import _resolve_device

class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head, n_kv_head, dropout):
        super().__init__()
        self.attn = CausalSelfAttention(n_embd, n_head, n_kv_head, dropout)
        self.mlp = MLP(n_embd)

    def forward(
            self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        x = x + self.attn(norm(x), cos, sin)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        cfg = config.model
        self.block_size = cfg.block_size
        self.n_embd: int = cfg.n_embd
        self.input_size: int = cfg.input_size

        # TODO: Might want to enable bias here
        self.proj_down = nn.Linear(self.input_size, self.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList(
            [
                Block(self.n_embd, cfg.n_head, cfg.n_kv_head, cfg.dropout)
                for _ in range(cfg.n_layer)
            ]
        )

        self.target_shapes_by_head = cfg.target_shapes_by_head
        self.shoulder_output_size = self.target_shapes_by_head["shoulder"]
        self.c_stick_output_size = self.target_shapes_by_head["c_stick"]
        self.main_stick_output_size = self.target_shapes_by_head["main_stick"]
        self.button_output_size = self.target_shapes_by_head["buttons"]

        # TODO: Move this to config
        head_hidden_dim = 128

        # TODO: Is there a way to make the sizes of these heads nicer / more even?
        self.button_head = ButtonHead(
            self.n_embd, self.button_output_size, hidden=head_hidden_dim
        )

        main_stick_input_size = self.n_embd + self.button_output_size
        self.main_stick_head = SimpleHead(
            main_stick_input_size, self.main_stick_output_size, hidden=head_hidden_dim
        )

        c_stick_input_size = (
                self.n_embd + self.button_output_size + self.main_stick_output_size
        )
        self.c_stick_head = SimpleHead(
            c_stick_input_size, self.c_stick_output_size, hidden=head_hidden_dim
        )

        shoulder_input_size = (
                self.n_embd
                + self.button_output_size
                + self.main_stick_output_size
                + self.c_stick_output_size
        )
        self.shoulder_head = SimpleHead(
            shoulder_input_size, self.shoulder_output_size, hidden=head_hidden_dim
        )
        self.value_head = ValueHead(self.n_embd, hidden=head_hidden_dim)

        self.rotary_seq_len = self.block_size * 2
        head_dim = cfg.n_embd // cfg.n_head
        rope_base = getattr(cfg, "rope_theta", 10000.0)
        cos, sin = self._precompute_rotary_embeddings(
            self.rotary_seq_len, head_dim, rope_base
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)

        for block in self.blocks:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)

    # TODO: Check if this is getting applied correctly
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:  # TODO: confirm unused then remove
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: Lower base since we have shorter sequences?
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000):
        device = _resolve_device()
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()  # keep them in bfloat16
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims for later broadcasting
        return cos, sin

    # TODO: Is there a way to pre-compute and cache the one-hot results?
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
        B, L, _ = inputs["gamestate"].shape
        assert (
                L <= self.block_size
        ), f"Cannot forward sequence of length {L}, block size is only {self.block_size}"

        combined_inputs = self._embed_inputs(inputs)
        x = self.proj_down(combined_inputs)
        x = self.drop(x)
        cos = self.cos[:, :L]
        sin = self.sin[:, :L]

        for block in self.blocks:
            x = block(x, cos, sin)

        x = norm(x)

        base = x
        button_logits, button_probs = self.button_head(base)

        main_stick = self.main_stick_head(
            torch.cat((base, button_logits.detach()), dim=-1)
        )

        c_stick = self.c_stick_head(
            torch.cat((base, button_logits.detach(), main_stick.detach()), dim=-1)
        )

        shoulder = self.shoulder_head(
            torch.cat(
                (base, button_logits.detach(), main_stick.detach(), c_stick.detach()),
                dim=-1,
            )
        )

        outputs = TensorDict(
            {
                "buttons": button_logits,
                "buttons_probs": button_probs,
                "main_stick": main_stick,
                "c_stick": c_stick,
                "shoulder": shoulder,
            },
            batch_size=(B, L),
        )

        value = self.value_head(x)
        outputs.set("value", value)

        return outputs
