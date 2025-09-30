"""Adapted from Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT."""
import math
from typing import Tuple

import attr
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.nn import functional as F

from config import get_config
from preprocess import C_STICK_XY_CLUSTER_CENTERS_V0_1, FOX_STICK_64


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.scale


def apply_rope(q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    d = q.size(-1)
    device = q.device
    pos = torch.arange(q.size(-2), device=device)
    theta = 10000 ** (-2 * torch.arange(d // 2, device=device) / d)
    freqs = torch.einsum("l,d->l d", pos, theta)
    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]

    def rope(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)

    return rope(q), rope(k)


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: float = 8 / 3) -> None:
        super().__init__()
        inner = int(mult * d)
        self.w1 = nn.Linear(d, inner, bias=False)
        self.v1 = nn.Linear(d, inner, bias=False)
        self.w2 = nn.Linear(inner, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.v1(x))


# DataConfig(data_dir='/Users/eppie/hal_original/mds', streams='', stream_stats='', seq_len=256,
# replay_filter=ReplayFilter(replay_uuid=None, stage=None, ego_character=None, opponent_character=None),
# debug_repeat_batch=False, debug_save_batch=False, input_preprocessing_fn='baseline_controller_fine_main_analog_shoulder',
# target_preprocessing_fn='fine_main_analog_shoulder', pred_postprocessing_fn='fine_main_analog_shoulder',
# num_stages=6, num_characters=27, num_actions=396, stage_embedding_dim=4, character_embedding_dim=12, action_embedding_dim=32, gamma=0.999)


@attr.s(auto_attribs=True, frozen=True)
class GPTConfig:
    block_size: int
    n_embd: int
    n_layer: int
    n_head: int
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

    # the below are new
    input_size: int = 130  # TODO: how is this calculated?
    num_stages: int = 7
    num_characters: int = 26
    num_actions: int = 396
    stage_embedding_dim: int = 4
    character_embedding_dim: int = 12
    action_embedding_dim: int = 32
    gamma: float = 0.999
    target_shapes_by_head: dict = {
        "main_stick": (len(FOX_STICK_64),),
        "c_stick": (len(C_STICK_XY_CLUSTER_CENTERS_V0_1),),
        "buttons": (5,),
        "shoulder": (3,),
    }


class MultiLabelButtonHead(nn.Module):
    """Predict independent button probabilities via shared features."""

    def __init__(self, input_size: int, output_size: int, *, bias: bool) -> None:
        super().__init__()
        hidden = max(1, max(input_size // 2, output_size * 2))
        mult = hidden / input_size if input_size > 0 else 1.0
        self.net = nn.Sequential(
            RMSNorm(input_size),
            SwiGLU(input_size, mult=mult),
            nn.Linear(input_size, output_size, bias=bias),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        probs = torch.sigmoid(logits)
        return logits, probs


class CausalSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = get_config()
        assert config.model.n_embd % config.model.n_head == 0
        config = config
        self.n_head = config.model.n_head
        self.n_embd = config.model.n_embd
        self.dropout = config.model.dropout

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.model.bias)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.model.bias)
        # regularization
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x: torch.Tensor):
        B, L, D = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = q.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = v.view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        q, k = apply_rope(q, k)

        # causal self-attention; Self-attend: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :L, :L] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = get_config()
        self.swiglu = SwiGLU(config.model.n_embd, mult=8 / 3)
        self.dropout = nn.Dropout(config.model.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.swiglu(x))


class Block(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        config = get_config()
        self.ln_1 = RMSNorm(config.model.n_embd)
        self.attn = CausalSelfAttention()
        self.ln_2 = RMSNorm(config.model.n_embd)
        self.mlp = MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BaseGPT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = get_config()
        self.block_size = config.model.block_size

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs: TensorDict):
        raise NotImplementedError


class GPTv7(BaseGPT):
    def __init__(self) -> None:
        super().__init__()
        config = get_config()
        self.input_size = config.model.input_size  # G
        self.n_embd = config.model.n_embd  # D

        # Categorical input embeddings
        self.stage_emb = nn.Embedding(config.model.num_stages, config.model.stage_embedding_dim)
        self.character_emb = nn.Embedding(config.model.num_characters, config.model.character_embedding_dim)
        self.action_emb = nn.Embedding(config.model.num_actions, config.model.action_embedding_dim)

        self.transformer = nn.ModuleDict(
            dict(
                proj_down=nn.Linear(self.input_size, config.model.n_embd),  # G -> D
                drop=nn.Dropout(config.model.dropout),
                h=nn.ModuleList([Block() for _ in range(config.model.n_layer)]),
                ln_f=RMSNorm(self.n_embd),
            )
        )

        # Output heads
        self.target_shapes_by_head = config.model.target_shapes_by_head
        shoulder_output_size = self.target_shapes_by_head["shoulder"][0]

        c_stick_input_size = self.n_embd + shoulder_output_size
        c_stick_output_size = self.target_shapes_by_head["c_stick"][0]

        main_stick_input_size = self.n_embd + shoulder_output_size + c_stick_output_size
        main_stick_output_size = self.target_shapes_by_head["main_stick"][0]

        button_input_size = self.n_embd + shoulder_output_size + c_stick_output_size + main_stick_output_size
        button_output_size = self.target_shapes_by_head["buttons"][0]

        # Put shoulder and c-stick first because they are less complex and they modify/override other inputs
        shoulder_hidden = max(1, self.n_embd // 2)
        self.shoulder_head = nn.Sequential(
            RMSNorm(self.n_embd),
            SwiGLU(self.n_embd, mult=shoulder_hidden / self.n_embd),
            nn.Linear(self.n_embd, shoulder_output_size, bias=config.model.bias),
        )

        c_stick_hidden = max(1, c_stick_input_size // 2)
        self.c_stick_head = nn.Sequential(
            RMSNorm(c_stick_input_size),
            SwiGLU(c_stick_input_size, mult=c_stick_hidden / c_stick_input_size),
            nn.Linear(c_stick_input_size, c_stick_output_size, bias=config.model.bias),
        )

        main_stick_hidden = max(1, main_stick_input_size // 2)
        self.main_stick_head = nn.Sequential(
            RMSNorm(main_stick_input_size),
            SwiGLU(main_stick_input_size, mult=main_stick_hidden / main_stick_input_size),
            nn.Linear(main_stick_input_size, main_stick_output_size, bias=config.model.bias),
        )

        self.button_head = MultiLabelButtonHead(
            button_input_size,
            button_output_size,
            bias=config.model.bias,
        )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.model.n_layer))

    def _embed_inputs(self, inputs: TensorDict) -> torch.Tensor:
        return torch.cat(
            [
                self.stage_emb(inputs["stage"]).squeeze(-2),
                self.character_emb(inputs["ego_character"]).squeeze(-2),
                self.character_emb(inputs["opponent_character"]).squeeze(-2),
                self.action_emb(inputs["ego_action"]).squeeze(-2),
                self.action_emb(inputs["opponent_action"]).squeeze(-2),
                inputs["gamestate"],
                inputs["controller"],
            ],
            dim=-1,
        )

    def forward(self, inputs: TensorDict) -> TensorDict:
        # loguru.logger.info(f"{inputs}")
        # B = batch size
        # L = sequence length
        B, L, _ = inputs["gamestate"].shape
        assert L <= self.block_size, f"Cannot forward sequence of length {L}, block size is only {self.block_size}"

        # Concatenate embeddings and numerical inputs -> project down
        combined_inputs_BLG = self._embed_inputs(inputs)
        proj_inputs_BLD = self.transformer.proj_down(combined_inputs_BLG)

        x_BLD = self.transformer.drop(proj_inputs_BLD)
        for block in self.transformer.h:
            x_BLD = block(x_BLD)
        x_BLD = self.transformer.ln_f(x_BLD)

        # Detach to avoid multiplying gradient flow through earlier heads
        shoulder: torch.Tensor = self.shoulder_head(x_BLD)
        c_stick: torch.Tensor = self.c_stick_head(torch.cat((x_BLD, shoulder.detach()), dim=-1))
        main_stick: torch.Tensor = self.main_stick_head(
            torch.cat((x_BLD, shoulder.detach(), c_stick.detach()), dim=-1)
        )
        button_logits, button_probs = self.button_head(
            torch.cat((x_BLD, shoulder.detach(), c_stick.detach(), main_stick.detach()), dim=-1)
        )

        return TensorDict(
            {
                "buttons": button_logits,
                "buttons_probs": button_probs,
                "main_stick": main_stick,
                "c_stick": c_stick,
                "shoulder": shoulder,
            },
            batch_size=(B, L),
        )
