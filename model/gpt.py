import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from tensordict import TensorDict

from config import get_config
from model.attention import CausalSelfAttention
from model.feed_forward import ActivationFFN, MoEFFN
from model.norm import _create_norm
from model.output_head import MLPHead, MultiLabelButtonHead, MultiLabelButtonHeadTiny, TinyMLPHead
from model.value_head import TinyValueHead


class MoEMLP(nn.Module):
    """Transformer MLP block with optional MoE FFN."""

    def __init__(self) -> None:
        super().__init__()
        cfg = get_config().model
        if cfg.use_moe:
            self.ffn = MoEFFN(
                cfg.n_embd,
                num_experts=cfg.moe_num_experts,
                num_active=cfg.moe_num_active,
                mult=cfg.ffn_mult,
                activation=cfg.ffn_activation,
                bias=cfg.bias,
            )
            self.is_moe = True
        else:
            self.ffn = ActivationFFN(
                cfg.n_embd,
                mult=cfg.ffn_mult,
                activation=cfg.ffn_activation,
                bias=cfg.bias,
            )
            self.is_moe = False
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.is_moe:
            out, aux_loss = self.ffn(x)
            return self.dropout(out), aux_loss
        return self.dropout(self.ffn(x)), None


class Block(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        config = get_config()
        self.attn = CausalSelfAttention()
        self.mlp = MoEMLP()
        placement = config.model.norm_placement.lower()
        if placement not in {"pre", "post", "both"}:
            raise ValueError(f"Unsupported norm placement '{config.model.norm_placement}'.")

        use_pre = placement in {"pre", "both"}
        use_post = placement in {"post", "both"}

        self.pre_attn_norm = _create_norm(config.model.n_embd) if use_pre else None
        self.post_attn_norm = _create_norm(config.model.n_embd) if use_post else None
        self.pre_mlp_norm = _create_norm(config.model.n_embd) if use_pre else None
        self.post_mlp_norm = _create_norm(config.model.n_embd) if use_post else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x
        attn_input = self.pre_attn_norm(x) if self.pre_attn_norm is not None else x
        attn_out = self.attn(attn_input)
        if self.post_attn_norm is not None:
            attn_out = self.post_attn_norm(attn_out)
        x = residual + attn_out

        residual = x
        mlp_input = self.pre_mlp_norm(x) if self.pre_mlp_norm is not None else x
        mlp_out, aux_loss = self.mlp(mlp_input)
        if self.post_mlp_norm is not None:
            mlp_out = self.post_mlp_norm(mlp_out)
        x = residual + mlp_out
        return x, aux_loss


class GPTv7(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = get_config()
        self.block_size = config.model.block_size
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
                ln_f=_create_norm(self.n_embd),
            )
        )

        # Output heads
        self.target_shapes_by_head = config.model.target_shapes_by_head
        shoulder_output_size = self.target_shapes_by_head["shoulder"]

        c_stick_input_size = self.n_embd + shoulder_output_size
        c_stick_output_size = self.target_shapes_by_head["c_stick"]

        main_stick_input_size = self.n_embd + shoulder_output_size + c_stick_output_size
        main_stick_output_size = self.target_shapes_by_head["main_stick"]

        button_input_size = self.n_embd + shoulder_output_size + c_stick_output_size + main_stick_output_size
        button_output_size = self.target_shapes_by_head["buttons"]

        # Put shoulder and c-stick first because they are less complex and they modify/override other inputs

        def head_block(input_dim: int, output_dim: int) -> nn.Sequential:
            return nn.Sequential(
                _create_norm(input_dim),
                ActivationFFN(
                    input_dim,
                    mult=config.model.ffn_mult,
                    activation=config.model.ffn_activation,
                    bias=config.model.bias,
                ),
                nn.Linear(input_dim, output_dim, bias=config.model.bias),
            )

        #
        base_dim: int = self.n_embd
        # self.shoulder_head = LinearHead(base_dim, shoulder_output_size, bias=config.model.bias)
        # self.c_stick_head = LinearHead(c_stick_input_size, c_stick_output_size, bias=config.model.bias)
        # self.main_stick_head = LinearHead(main_stick_input_size, main_stick_output_size, bias=config.model.bias)
        # self.button_head = MultiLabelButtonHeadLinear(
        #     button_input_size, button_output_size, bias=config.model.bias
        # )
        # self.shoulder_head = head_block(self.n_embd, shoulder_output_size)
        # self.c_stick_head = head_block(c_stick_input_size, c_stick_output_size)
        # self.main_stick_head = head_block(main_stick_input_size, main_stick_output_size)
        # self.button_head = MultiLabelButtonHead(
        #     button_input_size,
        #     button_output_size,
        #     bias=config.model.bias,
        # )

        # --- NEW: Define a small, shared hidden dimension for all heads ---
        head_hidden_dim = 128  # Drastically smaller than 1024+ before!
        head_activation = "gelu"  # GELU is a standard, efficient choice

        # --- Instantiate the Lightweight Heads ---
        self.shoulder_head = TinyMLPHead(
            input_size=base_dim,
            output_size=shoulder_output_size,
            hidden=head_hidden_dim,
            activation=head_activation,
            bias=config.model.bias
        )

        self.c_stick_head = TinyMLPHead(
            input_size=c_stick_input_size,
            output_size=c_stick_output_size,
            hidden=head_hidden_dim,
            activation=head_activation,
            bias=config.model.bias
        )

        self.main_stick_head = TinyMLPHead(
            input_size=main_stick_input_size,
            output_size=main_stick_output_size,
            hidden=head_hidden_dim,
            activation=head_activation,
            bias=config.model.bias
        )

        # Wrap the button head to get both logits and probs
        self.button_head = MultiLabelButtonHeadTiny(
            TinyMLPHead(
                input_size=button_input_size,
                output_size=button_output_size,
                hidden=head_hidden_dim,
                activation=head_activation,
                bias=config.model.bias
            )
        )
        
        # Value head for RL (critic network)
        self.value_head: Optional[TinyValueHead] = None
        if config.model.use_value_head:
            self.value_head = TinyValueHead(
                input_dim=self.n_embd,
                hidden=head_hidden_dim,
                activation=head_activation,
                bias=config.model.bias,
            )
        
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.model.n_layer))

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        # B = batch size
        # L = sequence length
        B, L, _ = inputs["gamestate"].shape
        assert L <= self.block_size, f"Cannot forward sequence of length {L}, block size is only {self.block_size}"

        # Concatenate embeddings and numerical inputs -> project down
        combined_inputs_BLG = self._embed_inputs(inputs)
        proj_inputs_BLD = self.transformer.proj_down(combined_inputs_BLG)

        x_BLD = self.transformer.drop(proj_inputs_BLD)
        total_aux_loss: Optional[torch.Tensor] = None
        num_moe_layers = 0
        for block in self.transformer.h:
            x_BLD, aux_loss = block(x_BLD)
            if aux_loss is not None:
                total_aux_loss = aux_loss if total_aux_loss is None else total_aux_loss + aux_loss
                num_moe_layers += 1
        x_BLD = self.transformer.ln_f(x_BLD)

        head_flow = get_config().model.head_flow.lower()

        if head_flow == "parallel":
            base = x_BLD
            shoulder = self.shoulder_head(base)
            c_stick = self.c_stick_head(torch.cat((base, shoulder), dim=-1))
            main_stick = self.main_stick_head(torch.cat((base, shoulder, c_stick), dim=-1))
            button_logits, button_probs = self.button_head(
                torch.cat((base, shoulder, c_stick, main_stick), dim=-1)
            )
        elif head_flow == "sequential":
            shoulder = self.shoulder_head(x_BLD)
            c_input = torch.cat((x_BLD, shoulder.detach()), dim=-1)
            c_stick = self.c_stick_head(c_input)
            main_input = torch.cat((x_BLD, shoulder.detach(), c_stick.detach()), dim=-1)
            main_stick = self.main_stick_head(main_input)
            button_input = torch.cat(
                (x_BLD, shoulder.detach(), c_stick.detach(), main_stick.detach()),
                dim=-1,
            )
            button_logits, button_probs = self.button_head(button_input)
        else:
            raise ValueError(f"Unsupported head_flow '{get_config().model.head_flow}'.")

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

        cfg = get_config().model
        
        # Add value prediction if value head is enabled
        if cfg.use_value_head and self.value_head is not None:
            value = self.value_head(x_BLD)  # [B, L, 1]
            outputs.set("value", value)
        
        if cfg.use_moe and total_aux_loss is not None and num_moe_layers > 0:
            avg_aux_loss = total_aux_loss / num_moe_layers
            outputs.set("moe_aux_loss", avg_aux_loss.reshape(1, 1).expand(B, L))

        return outputs
