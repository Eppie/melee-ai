from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Callable, Optional

import torch
from torch import nn, Tensor

from config import TARGET_COLUMNS, REGRESSION_TARGETS, CLASSIFICATION_TARGETS, BUTTON_SLICE, STICK_SLICE


def rnn_init(m: nn.Module) -> None:
    # (1) Linear layers --------------------------------------------------
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:                      # ← guard for bias=None
            nn.init.zeros_(m.bias)

    # (2) Recurrent layers ----------------------------------------------
    for name, p in m.named_parameters(recurse=False):
        if "weight_ih_l" in name:                   # input → hidden
            nn.init.xavier_uniform_(p)
        elif "weight_hh_l" in name:                 # hidden → hidden
            nn.init.orthogonal_(p)
        elif "bias_ih_l" in name or "bias_hh_l" in name:
            if p is not None:                       # RNNs always have bias, but safe
                nn.init.zeros_(p)
            # LSTM forget-gate trick
            if isinstance(m, nn.LSTM) and p.ndim == 1:
                H = p.shape[0] // 4
                p.data[H:2 * H].fill_(1.0)

@dataclass(slots=True)
class RecurrentConfig:
    """Hyper-parameters for encoder + auto-regressive decoder."""
    input_dim: int
    output_dim: int                                # must be 9
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False
    kind: Literal["gru", "lstm"] = "gru"
    act: type[nn.Module] = nn.GELU
    embed_dim: int = 32                            # size of feedback embeddings
    weight_init: Optional[Callable[[nn.Module], None]] = rnn_init


class RecurrentNet(nn.Module):
    """
    Encoder (LayerNorm + GRU/LSTM) → last hidden state → 2-step decoder loop:

        step 0: predict sticks  (sigmoid [0,1]) → embed → GRUCell
        step 1: predict buttons (logits) → embed → GRUCell

    Returns tensor shaped [B, 9] in TARGET_COLUMNS order
    (buttons logits 0-4 | stick values 5-8).
    """

    def __init__(self, cfg: RecurrentConfig, *, feature_dim: int | None = None):
        super().__init__()
        if cfg.output_dim != len(TARGET_COLUMNS):
            raise ValueError("cfg.output_dim must be 9 (buttons 5 + sticks 4)")

        feat = feature_dim if feature_dim is not None else cfg.input_dim
        self.pre_norm = nn.LayerNorm(feat)

        rnn_cls = nn.GRU if cfg.kind == "gru" else nn.LSTM
        self.encoder = rnn_cls(
            input_size=feat,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
            batch_first=True,
        )
        enc_dim = cfg.hidden_size * (2 if cfg.bidirectional else 1)

        # Decoder: tiny GRUCell + embeddings
        self.enc2dec = nn.Linear(enc_dim, cfg.hidden_size)
        self.dec_cell = nn.GRUCell(cfg.embed_dim, cfg.hidden_size)

        self.stick_head = nn.Linear(cfg.hidden_size, len(REGRESSION_TARGETS))    # 4
        self.button_head = nn.Linear(cfg.hidden_size, len(CLASSIFICATION_TARGETS))  # 5

        self.stick_embed = nn.Linear(len(REGRESSION_TARGETS), cfg.embed_dim, bias=False)
        self.button_embed = nn.Linear(len(CLASSIFICATION_TARGETS), cfg.embed_dim, bias=False)

        if cfg.weight_init is not None:
            self.apply(cfg.weight_init)

    def forward(self, x: Tensor, y: Tensor | None = None) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor  [B, T, F]   input window
        y : Tensor | None  [B, 9] ground-truth targets for teacher forcing

        Returns
        -------
        Tensor  [B, 9]   logits|values in TARGET_COLUMNS order
        """
        # 1) Encode sequence
        x = self.pre_norm(x)
        _, h = self.encoder(x)
        h_last = h[0][-1] if isinstance(self.encoder, nn.LSTM) else h[-1]  # [B,H]

        # 2) Initialise decoder hidden state
        h_dec = torch.tanh(self.enc2dec(h_last))

        # Teacher forcing?
        use_teacher = y is not None and self.training
        if use_teacher:
            y_btn, y_stick = y[:, BUTTON_SLICE], y[:, STICK_SLICE]

        # ------- step 0 : sticks -------
        stick_out = torch.sigmoid(self.stick_head(h_dec))        # [0,1]
        stick_in = y_stick if use_teacher else stick_out.detach()
        h_dec = self.dec_cell(self.stick_embed(stick_in), h_dec)

        # ------- step 1 : buttons -------
        btn_logits = self.button_head(h_dec)                    # raw logits
        btn_in = y_btn if use_teacher else (btn_logits.detach() > 0).float()
        _ = self.dec_cell(self.button_embed(btn_in), h_dec)     # state not used

        # 3) Concatenate in canonical order
        return torch.cat([btn_logits, stick_out], dim=-1)       # [B,9]