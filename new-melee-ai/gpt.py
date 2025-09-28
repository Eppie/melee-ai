"""Adapted from Karpathy's nanoGPT: https://github.com/karpathy/nanoGPT."""
import math
from typing import Tuple

import attr
import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.nn import functional as F

from preprocess import C_STICK_XY_CLUSTER_CENTERS_V0_1, FOX_STICK_64


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
    num_stages: int = 6
    num_characters: int = 27
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
        hidden = max(input_size // 2, output_size * 2)
        self.net = nn.Sequential(
            nn.LayerNorm(input_size, bias=bias),
            nn.Linear(input_size, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_size),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        probs = torch.sigmoid(logits)
        return logits, probs


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)
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
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BaseGPT(nn.Module):
    def __init__(self, gpt_config: GPTConfig) -> None:
        super().__init__()
        self.gpt_config = gpt_config
        self.block_size = self.gpt_config.block_size

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wpe.weight.numel()
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


def sinusoidal_positional_encoding_1d(seq_len: int, d_model: int, device: torch.device | None = None) -> torch.Tensor:
    """
    :param d_model: dimension of the model
    :param seq_len: length of positions
    :return: seq_len*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with " "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(torch.tensor(10000.0).log() / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe


class GPTv7(BaseGPT):
    def __init__(self, gpt_config: GPTConfig) -> None:
        super().__init__(gpt_config)
        self.input_size = self.gpt_config.input_size  # G
        self.n_embd = gpt_config.n_embd  # D

        # Categorical input embeddings
        self.stage_emb = nn.Embedding(self.gpt_config.num_stages, self.gpt_config.stage_embedding_dim)
        self.character_emb = nn.Embedding(self.gpt_config.num_characters, self.gpt_config.character_embedding_dim)
        self.action_emb = nn.Embedding(self.gpt_config.num_actions, self.gpt_config.action_embedding_dim)

        self.wpe = nn.Parameter(sinusoidal_positional_encoding_1d(self.block_size, gpt_config.n_embd))
        self.transformer = nn.ModuleDict(
            dict(
                proj_down=nn.Linear(self.input_size, gpt_config.n_embd),  # G -> D
                drop=nn.Dropout(gpt_config.dropout),
                h=nn.ModuleList([Block(gpt_config) for _ in range(gpt_config.n_layer)]),
                ln_f=nn.LayerNorm(self.n_embd, bias=gpt_config.bias),
            )
        )

        # Output heads
        self.target_shapes_by_head = self.gpt_config.target_shapes_by_head
        shoulder_output_size = self.target_shapes_by_head["shoulder"][0]

        c_stick_input_size = self.n_embd + shoulder_output_size
        c_stick_output_size = self.target_shapes_by_head["c_stick"][0]

        main_stick_input_size = self.n_embd + shoulder_output_size + c_stick_output_size
        main_stick_output_size = self.target_shapes_by_head["main_stick"][0]

        button_input_size = self.n_embd + shoulder_output_size + c_stick_output_size + main_stick_output_size
        button_output_size = self.target_shapes_by_head["buttons"][0]

        # Put shoulder and c-stick first because they are less complex and they modify/override other inputs
        self.shoulder_head = nn.Sequential(
            nn.LayerNorm(self.n_embd, bias=gpt_config.bias),
            nn.Linear(self.n_embd, self.n_embd // 2),
            nn.GELU(),
            nn.Linear(self.n_embd // 2, shoulder_output_size),
        )

        self.c_stick_head = nn.Sequential(
            nn.LayerNorm(c_stick_input_size, bias=gpt_config.bias),
            nn.Linear(c_stick_input_size, c_stick_input_size // 2),
            nn.GELU(),
            nn.Linear(c_stick_input_size // 2, c_stick_output_size),
        )

        self.main_stick_head = nn.Sequential(
            nn.LayerNorm(main_stick_input_size, bias=gpt_config.bias),
            nn.Linear(main_stick_input_size, main_stick_input_size // 2),
            nn.GELU(),
            nn.Linear(main_stick_input_size // 2, main_stick_output_size),
        )

        self.button_head = MultiLabelButtonHead(
            button_input_size,
            button_output_size,
            bias=gpt_config.bias,
        )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * gpt_config.n_layer))

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

        wpe_BLD = self.wpe[:L, :]
        proj_inputs_BLD = proj_inputs_BLD + wpe_BLD

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
