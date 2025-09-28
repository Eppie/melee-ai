from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict, TypedDict, Any, cast
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from config import SEQUENCE_LENGTH
from libmelee_parse import get_slp


def _get_cf_distance(fr: Mapping[str, object], key: str) -> float:
    p = fr.get(key, {})  # player dict
    cf = (p.get("computed_features") or {})  # type: ignore[union-attr]
    try:
        return float(cf.get("distance", 0.0))
    except Exception:
        return 0.0


def proximity_reward_series(
        frames: Sequence[Mapping[str, object]],
        self_key: str,
) -> np.ndarray:
    """
    r_t = distance_{t-1} - distance_{t}
    Positive when you moved closer this frame, negative when you drifted away.
    r_0 = 0.
    """
    T = len(frames)
    if T == 0:
        return np.zeros((0,), dtype=np.float32)
    d = np.asarray([_get_cf_distance(fr, self_key) for fr in frames], dtype=np.float32)  # (T,)
    r = np.zeros_like(d)
    if T > 1:
        r[1:] = d[:-1] - d[1:]
    return r


@dataclass(frozen=True)
class ActionSpace:
    n_cont: int  # e.g., main_x, main_y, c_x, c_y, L, R => 6
    n_bin: int  # e.g., A,B,X,Y,Z,Start,Dpad(4),L_dig,R_dig => up to ~12


@dataclass(frozen=True)
class Episode:
    """One full game episode as contiguous tensors (on CPU)."""
    states: Tensor  # (T, D_state), float32
    actions_cont: Tensor  # (T, n_cont), float32 in [-1, 1] or [0,1]
    actions_bin: Tensor  # (T, n_bin), float32 in {0,1}
    rewards: Tensor  # (T,), float32. Use zeros if starting with pure BC.

    def to(self, device: torch.device | str) -> "Episode":
        return Episode(
            self.states.to(device),
            self.actions_cont.to(device),
            self.actions_bin.to(device),
            self.rewards.to(device),
        )


class DTDataset(Dataset):
    """
    Yields padded, fixed-length segments for DT training.

    Input at time t: (rtg_t, state_t, action_{t-1})  -> predict action_t
    """

    def __init__(
            self,
            episodes: List[Episode],
            action_space: ActionSpace,
            seq_len: int,
            discount: float = 1.0,  # DT uses undiscounted; keep here if you want both
            rtg_scale: float = 1.0,  # divide RTG by this for numerical stability
            pad_value: float = 0.0,
            weight_mode: str = "none", weight_beta: float = 3.0,
            weight_clip: float = 10.0, norm_weights: bool = True
    ) -> None:
        super().__init__()
        self.eps = episodes
        self.aspace = action_space
        self.L = seq_len
        self.gamma = discount
        self.rtg_scale = max(rtg_scale, 1e-8)
        self.pad_value = pad_value
        self.weight_mode = weight_mode
        self.weight_beta = float(weight_beta)
        self.weight_clip = float(weight_clip)
        self.norm_weights = bool(norm_weights)

        # Precompute start indices for uniform sampling
        self.index: List[Tuple[int, int]] = []  # (ep_id, start_t)
        for i, ep in enumerate(self.eps):
            T = ep.states.shape[0]
            # we allow segments that end at episode end; start from 0..T-1
            for t0 in range(T):
                self.index.append((i, t0))

        # Precompute RTGs (both undiscounted + discounted) for each episode
        self.rtg_cache: List[Tensor] = []
        for ep in tqdm(self.eps):
            r = ep.rewards
            T = r.shape[0]
            rtg = torch.empty(T, dtype=r.dtype)
            acc = 0.0
            if abs(self.gamma - 1.0) < 1e-8:
                # Suffix sum (undiscounted)
                s = 0.0
                for t in reversed(range(T)):
                    s += float(r[t])
                    rtg[t] = s
            else:
                # Discounted suffix sum
                for t in reversed(range(T)):
                    acc = float(r[t]) + self.gamma * acc
                    rtg[t] = acc
            self.rtg_cache.append(rtg / self.rtg_scale)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        ep_id, t0 = self.index[idx]
        ep = self.eps[ep_id]
        T = ep.states.shape[0]
        t1 = min(t0 + self.L, T)  # exclusive

        # Slice actual segment
        s = ep.states[t0:t1]  # (L', D_state)
        a_cont = ep.actions_cont[t0:t1]  # (L', n_cont)
        a_bin = ep.actions_bin[t0:t1]  # (L', n_bin)
        rtg = self.rtg_cache[ep_id][t0:t1]  # (L',)

        Lp = s.shape[0]
        pad = self.L - Lp

        # Build prev actions (shift right by 1, pad zero at left)
        a_cont_prev = torch.empty_like(a_cont)
        a_bin_prev = torch.empty_like(a_bin)
        a_cont_prev[0].zero_()
        a_bin_prev[0].zero_()

        if Lp > 1:
            a_cont_prev[1:] = a_cont[:-1]
            a_bin_prev[1:] = a_bin[:-1]

        # Pad to fixed length (right-pad)
        def rpad(x: Tensor, value: float = 0.0) -> Tensor:
            if pad <= 0:
                return x
            pad_shape = list(x.shape)
            pad_shape[0] = pad
            return torch.cat([x, x.new_full(pad_shape, value)], dim=0)

        rewards_slice = ep.rewards[t0:t1]  # (L',)
        rtg_slice = self.rtg_cache[ep_id][t0:t1]  # (L',)

        # --- NEW: token weights ---
        if self.weight_mode == "rpos":
            w = torch.clamp(rewards_slice, min=0.0)
        elif self.weight_mode == "rtg_pos":
            w = torch.clamp(rtg_slice, min=0.0)
        elif self.weight_mode == "exp_rtg":
            # emphasize future progress; ensure nonneg and clipped
            w = torch.exp(self.weight_beta * rtg_slice) - 1.0
            w = torch.clamp(w, min=0.0)
        else:  # "none"
            w = torch.ones_like(rtg_slice)

        if self.norm_weights:
            m = float(w.mean().item()) if w.numel() > 0 else 1.0
            if m > 0:
                w = w / m
        if self.weight_clip > 0:
            w = torch.clamp(w, max=self.weight_clip)

        # helper to right-pad a (L',) vector → (L,1)
        def rpad1d(x: Tensor, value: float = 0.0) -> Tensor:
            x = x.unsqueeze(-1)  # (L',1)
            return rpad(x, value)

        sample = {
            "states": rpad(s, self.pad_value),
            "actions_cont_in": rpad(a_cont_prev, 0.0),
            "actions_bin_in": rpad(a_bin_prev, 0.0),
            "actions_cont_tgt": rpad(a_cont, 0.0),
            "actions_bin_tgt": rpad(a_bin, 0.0),
            "rtg": rpad(rtg_slice.unsqueeze(-1), 0.0),
            "rewards": rpad1d(rewards_slice, 0.0),  # (L,1)  # optional for metrics
            "token_weight": rpad1d(w, 1.0),  # (L,1)  # NEW
            "attn_mask": torch.cat([torch.ones(Lp, dtype=torch.bool),
                                    torch.zeros(pad, dtype=torch.bool)], 0),
        }
        return sample


class DecisionTransformer(nn.Module):
    """
    DT-style transformer:
      token_t = Emb(rtg_t) + Emb(state_t) + Emb(action_{t-1}) + pos_emb[t]
      predict action_t (continuous + binary) from hidden_t

    We keep heads split for cont/bin to use MSE and BCE losses naturally.
    """

    def __init__(
            self,
            d_state: int,
            aspace: ActionSpace,
            d_model: int = 256,
            n_layers: int = 6,
            n_heads: int = 8,
            d_ff: int = 1024,
            max_len: int = 16384,
            dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.aspace = aspace
        self.d_model = d_model

        self.embed_state = nn.Linear(d_state, d_model)
        self.embed_rtg = nn.Linear(1, d_model)
        self.embed_acont = nn.Linear(aspace.n_cont, d_model) if aspace.n_cont > 0 else None
        self.embed_abin = nn.Linear(aspace.n_bin, d_model) if aspace.n_bin > 0 else None

        self.pos_emb = nn.Embedding(max_len, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Heads
        self.head_cont = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, aspace.n_cont) if aspace.n_cont > 0 else nn.Identity(),
        )
        self.head_bin_logits = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, aspace.n_bin) if aspace.n_bin > 0 else nn.Identity(),
        )

        self.head_r = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        # Output squash for continuous actions (optional)
        self.tanh_out = True  # set False if you don't want [-1,1] squashing

    def forward(
            self,
            rtg: Tensor,  # (B, L, 1)
            states: Tensor,  # (B, L, D_state)
            actions_cont_in: Tensor,  # (B, L, n_cont)
            actions_bin_in: Tensor,  # (B, L, n_bin)
            attn_mask: Optional[Tensor] = None,  # (B, L) True for real tokens
    ) -> Dict[str, Tensor]:
        B, L, _ = states.shape
        device = states.device

        x = self.embed_state(states) + self.embed_rtg(rtg)

        if self.aspace.n_cont > 0:
            x = x + self.embed_acont(actions_cont_in)  # type: ignore[arg-type]
        if self.aspace.n_bin > 0:
            x = x + self.embed_abin(actions_bin_in)  # type: ignore[arg-type]

        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(pos_ids)

        # Build masks
        # - Causal mask (L, L) with True where attention is NOT allowed
        causal_mask = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
        # - Key padding mask (B, L) with True for PAD tokens
        key_padding: Optional[Tensor] = None
        if attn_mask is not None:
            key_padding = ~attn_mask  # invert: True=pad, False=keep

        h = self.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding)

        a_cont = self.head_cont(h) if self.aspace.n_cont > 0 else torch.empty(B, L, 0, device=device)
        if self.tanh_out and self.aspace.n_cont > 0:
            a_cont = torch.tanh(a_cont)
            # a_cont = torch.clamp(a_cont, 0.0, 1.0)

        a_bin_logits = (
            self.head_bin_logits(h) if self.aspace.n_bin > 0 else torch.empty(B, L, 0, device=device)
        )

        return {"a_cont": a_cont, "a_bin_logits": a_bin_logits, "r_pred": self.head_r(h)}


# =========================
# Loss + Train step
# =========================

@dataclass(frozen=True)
class LossWeights:
    cont: float = 1.0
    bin: float = 1.0


class ActionLoss(nn.Module):
    def __init__(self, weights: LossWeights) -> None:
        super().__init__()
        self.w = weights
        self.mse = nn.MSELoss(reduction="none")
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(
            self,
            pred_cont: Tensor, tgt_cont: Tensor,
            pred_bin_logits: Tensor, tgt_bin: Tensor,
            attn_mask: Tensor,
            token_weight: Optional[Tensor] = None,  # NEW: (B,L,1)
    ) -> Tuple[Tensor, Dict[str, float]]:
        mask = attn_mask.unsqueeze(-1).float()  # (B,L,1)
        w = (token_weight if token_weight is not None else 1.0)
        w = w * mask  # zero-out pads

        loss_cont = torch.tensor(0.0, device=pred_cont.device)
        loss_bin = torch.tensor(0.0, device=pred_bin_logits.device)

        if pred_cont.numel() > 0:
            per = (pred_cont - tgt_cont).pow(2) * w  # (B,L,C)
            denom = w.sum() * pred_cont.shape[-1] + 1e-8
            loss_cont = per.sum() / denom

        if pred_bin_logits.numel() > 0:
            per = nn.functional.binary_cross_entropy_with_logits(
                pred_bin_logits, tgt_bin, reduction="none"
            ) * w  # (B,L,Bn)
            denom = w.sum() * pred_bin_logits.shape[-1] + 1e-8
            loss_bin = per.sum() / denom

        total = self.w.cont * loss_cont + self.w.bin * loss_bin
        stats = {
            "loss": float(total.detach()),
            "loss_cont": float(loss_cont.detach()),
            "loss_bin": float(loss_bin.detach()),
            "w_mean": float((w.sum() / (mask.sum() + 1e-8)).detach()),
        }
        return total, stats


# =========================
# Training loop (prototype)
# =========================

@dataclass
class TrainConfig:
    d_state: int
    action_space: ActionSpace
    seq_len: int = 60
    batch_size: int = 64
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    lr: float = 3e-4
    weight_decay: float = 0.01
    max_len: int = 16384
    dropout: float = 0.1
    discount: float = 1.0  # keep 1.0 for DT-style RTG
    rtg_scale: float = 1.0
    grad_clip: float = 1.0
    epochs: int = 10
    device: str = "mps" if torch.mps.is_available() else "cpu"
    num_workers: int = 4
    weight_mode: str = "none"  # "none" | "rpos" | "rtg_pos" | "exp_rtg"
    weight_beta: float = 3.0  # only for "exp_rtg": w = exp(beta * rtg_norm)
    weight_clip: float = 10.0  # clamp weights to [0, weight_clip]
    norm_weights: bool = True  # normalize per-batch to mean 1.0


def train_dt(
        episodes: List[Episode],
        cfg: TrainConfig,
        val_episodes: Optional[List[Episode]] = None,
) -> DecisionTransformer:
    device = torch.device(cfg.device)

    ds = DTDataset(
        episodes=episodes,
        action_space=cfg.action_space,
        seq_len=cfg.seq_len,
        discount=cfg.discount,
        rtg_scale=cfg.rtg_scale,
        weight_mode=cfg.weight_mode,
        weight_beta=cfg.weight_beta,
        weight_clip=cfg.weight_clip,
        norm_weights=cfg.norm_weights,
    )
    print(ds)
    # sampler = WeightedRandomSampler(build_segment_weights(ds), num_samples=len(ds), replacement=True)
    # print(sampler)
    dl = DataLoader(ds, batch_size=cfg.batch_size, drop_last=True, num_workers=cfg.num_workers)
    # dl = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=True, num_workers=cfg.num_workers)
    print(dl)

    model = DecisionTransformer(
        d_state=cfg.d_state,
        aspace=cfg.action_space,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        max_len=cfg.max_len,
        dropout=cfg.dropout,
    ).to(device)

    print(model)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    print(opt)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg.epochs))
    print(sched)
    loss_fn = ActionLoss(LossWeights(cont=1.0, bin=1.0))
    print(loss_fn)
    global_step = 0

    for epoch in range(1, cfg.epochs + 1):
        ds.weight_beta = np.interp(epoch, [0, cfg.epochs - 1], [1.0, 8.0])
        print(f"{epoch} / {cfg.epochs}, ds.weight_beta = {ds.weight_beta}")
        model.train()
        running: Dict[str, float] = {
            "loss": 0.0, "loss_cont": 0.0, "loss_bin": 0.0,
            "r_mse": 0.0, "total_loss": 0.0
        }
        n_batches = 0

        for batch in dl:
            rtg = batch["rtg"].to(device)  # (B,L,1)
            states = batch["states"].to(device)  # (B,L,Ds)
            acont_in = batch["actions_cont_in"].to(device)  # (B,L,C)
            abin_in = batch["actions_bin_in"].to(device)  # (B,L,Bn)
            acont_tgt = batch["actions_cont_tgt"].to(device)  # (B,L,C)
            abin_tgt = batch["actions_bin_tgt"].to(device)  # (B,L,Bn)
            attn_mask = batch["attn_mask"].to(device)  # (B,L) bool
            tok_w = batch.get("token_weight", None)
            tok_w = tok_w.to(device) if tok_w is not None else None
            out = model(rtg, states, acont_in, abin_in, attn_mask=attn_mask)
            # 1) normal imitation loss
            action_loss, stats = loss_fn(
                out["a_cont"], acont_tgt, out["a_bin_logits"], abin_tgt, attn_mask, token_weight=tok_w
            )

            r_t = batch["rewards"].to(device)  # (B,L,1)
            mask = attn_mask.unsqueeze(-1).float()  # (B,L,1)
            r_mse = ((out["r_pred"] - r_t) ** 2) * mask
            # optional: weight by token_weight if you’re using RWI
            if "token_weight" in batch:
                r_mse = r_mse * batch["token_weight"].to(device)
            r_mse = r_mse.sum() / (mask.sum() + 1e-8)
            aux_lambda = 0.2  # tune 0.2–0.5

            total_loss = action_loss + aux_lambda * r_mse

            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if tok_w is not None:
                w_mean = float((tok_w * mask).sum().item() / (mask.sum().item() + 1e-8))
            else:
                w_mean = 1.0

            stats["r_mse"] = float(r_mse.detach())
            stats["total_loss"] = float(total_loss.detach())
            stats["w_mean"] = w_mean
            for k in ("loss", "loss_cont", "loss_bin", "r_mse", "total_loss", "w_mean"):
                running[k] = running.get(k, 0.0) + stats[k]

            n_batches += 1
            global_step += 1

            print(
                f"step {global_step} | "
                f"total {stats['total_loss']:.4f} | act {stats['loss']:.4f} "
                f"(cont {stats['loss_cont']:.4f} bin {stats['loss_bin']:.4f}) | "
                f"r_mse {stats['r_mse']:.4f} | w_mean {stats['w_mean']:.3f}"
            )
            with torch.no_grad():
                m = batch["attn_mask"].to(device).unsqueeze(-1).float()  # (B,L,1)
                pred_c = out["a_cont"].detach()  # (B,L,C)
                # masked mean per-dimension
                denom = m.sum(dim=(0, 1)).clamp_min(1e-8)  # (1,)
                mu = (pred_c * m).sum(dim=(0, 1)) / denom  # (C,)
                var = (((pred_c - mu) ** 2) * m).sum(dim=(0, 1)) / denom  # (C,)
                std = var.sqrt()  # (C,)
            # e.g., print first four dims
            print(f"pred_std main: x={std[0]:.3f}, y={std[1]:.3f}, c_x={std[2]:.3f}, c_y={std[3]:.3f}")

        for k in running:
            running[k] /= max(1, n_batches)
        sched.step()

        print(
            f"epoch {epoch:02d} | "
            f"total {running['total_loss']:.4f} | act {running['loss']:.4f} "
            f"(cont {running['loss_cont']:.4f} bin {running['loss_bin']:.4f}) | "
            f"r_mse {running['r_mse']:.4f} | w_mean {running['w_mean']:.3f} | "
            f"lr {sched.get_last_lr()[0]:.2e}"
        )

        save_dt(model, cfg, path="melee_dt_demo.pt")

    return model


@torch.no_grad()
def evaluate_dt(model: DecisionTransformer, episodes: List[Episode], cfg: TrainConfig) -> float:
    device = torch.device(cfg.device)
    ds = DTDataset(
        episodes=episodes,
        action_space=cfg.action_space,
        seq_len=cfg.seq_len,
        discount=cfg.discount,
        rtg_scale=cfg.rtg_scale,
    )
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)
    loss_fn = ActionLoss(LossWeights())

    model.eval()
    total, denom = 0.0, 0
    for batch in dl:
        rtg = batch["rtg"].to(device)
        states = batch["states"].to(device)
        acont_in = batch["actions_cont_in"].to(device)
        abin_in = batch["actions_bin_in"].to(device)
        acont_tgt = batch["actions_cont_tgt"].to(device)
        abin_tgt = batch["actions_bin_tgt"].to(device)
        attn_mask = batch["attn_mask"].to(device)

        out = model(rtg, states, acont_in, abin_in, attn_mask=attn_mask)
        loss, _ = loss_fn(out["a_cont"], acont_tgt, out["a_bin_logits"], abin_tgt, attn_mask)
        total += float(loss) * states.shape[0]
        denom += states.shape[0]
    return total / max(1, denom)


# =========================
# Inference helper (greedy)
# =========================

@torch.no_grad()
def act_greedy(
        model: DecisionTransformer,
        rtg_step: float,
        state_seq: Tensor,  # (L_ctx, D_state), most recent last
        prev_action_cont: Tensor,  # (L_ctx, n_cont)
        prev_action_bin: Tensor,  # (L_ctx, n_bin)
        rtg_scale: float = 1.0,
        device: str = "mps",
) -> Tuple[Tensor, Tensor]:
    """
    Produce next action given a context window (L_ctx <= model.max_len).
    You can update rtg_step each step if you have a reward signal; otherwise keep constant.
    """
    model.eval()
    device_t = torch.device(device)
    state_seq = state_seq.unsqueeze(0).to(device_t)
    acont_in = prev_action_cont.unsqueeze(0).to(device_t)
    abin_in = prev_action_bin.unsqueeze(0).to(device_t)
    rtg = torch.full((1, state_seq.shape[1], 1), rtg_step / max(rtg_scale, 1e-8), device=device_t)
    attn = torch.ones(1, state_seq.shape[1], dtype=torch.bool, device=device_t)
    out = model(rtg, state_seq, acont_in, abin_in, attn_mask=attn)
    acont = out["a_cont"][:, -1]  # (1, n_cont)
    abin = (out["a_bin_logits"][:, -1] > 0.0).float()
    return acont.squeeze(0).cpu(), abin.squeeze(0).cpu()


# Buttons present in your libmelee_parse schema
BTN_ORDER: Tuple[str, ...] = ("A", "B", "X", "Y", "Z", "L", "R", "D_UP")


@dataclass(frozen=True)
class FeatureDims:
    d_state: int
    n_cont: int
    n_bin: int


def _safe_bool(x: object) -> float:
    return 1.0 if bool(x) else 0.0


def _safe_float(x: object) -> float:
    try:
        return float(x)  # handles numpy scalars too
    except Exception:
        return 0.0


def _facing_to_pm1(b: object) -> float:
    # your schema stores facing as bool (True = right). Map to {-1, +1}.
    return 1.0 if bool(b) else -1.0


def _get_nested(d: Mapping[str, object], *keys: str) -> object:
    cur: object = d
    for k in keys:
        cur = (cur if isinstance(cur, Mapping) else {})  # type: ignore[assignment]
        cur = cur.get(k, 0.0)  # type: ignore[union-attr]
    return cur


def _state_vector(frame: Mapping[str, object], self_key: str, opp_key: str) -> np.ndarray:
    """Build a compact numeric state vector from your per-frame nested dict."""
    # stage_id = _safe_float(frame.get("stage", 0))

    me: dict = frame[self_key]  # type: ignore[index]
    opp: dict = frame[opp_key]  # type: ignore[index]

    me_cf = (me.get("computed_features") or {})  # type: ignore[union-attr]
    opp_cf = (opp.get("computed_features") or {})  # type: ignore[union-attr]

    # --- self subset (16 dims) ---
    s_self = [
        _safe_float(me.get("percent", 0)),
        # _facing_to_pm1(me.get("facing", False)),
        _safe_float(me.get("x", 0)),
        _safe_float(me.get("y", 0)),
        _safe_float(me.get("action", 0)),
        _safe_bool(me.get("invulnerable", False)),
        # _safe_float(me.get("character", 0)),
        _safe_float(me.get("jumps_left", 0)),
        _safe_float(me.get("shield_strength", 0)),
        _safe_bool(me.get("on_ground", False)),
        _safe_float(me.get("action_frame", 0)),
        _safe_float(me.get("hitstun_frames_left", 0)),
        _safe_bool(me.get("off_stage", False)),
        _safe_float(me_cf.get("distance", 0.0)),
        _safe_bool(me_cf.get("facing_opponent", False)),
        _safe_float(me_cf.get("distance_to_blastzones", 0.0)),
        _safe_float(me.get("speed_air_x_self", 0.0)),
        _safe_float(me.get("speed_y_self", 0.0)),
        _safe_float(me.get("speed_x_attack", 0.0)),
        _safe_float(me.get("speed_y_attack", 0.0)),
        _safe_float(me.get("speed_ground_x_self", 0.0)),
    ]

    # --- opponent subset (8 dims) ---
    s_opp = [
        _safe_float(opp.get("x", 0)),
        _safe_float(opp.get("y", 0)),
        _safe_float(opp.get("percent", 0)),
        _safe_float(opp.get("action", 0)),
        _safe_bool(opp.get("on_ground", False)),
        _safe_bool(opp.get("off_stage", False)),
        _safe_float(opp_cf.get("distance", 0.0)),
        _safe_float(opp_cf.get("distance_to_blastzones", 0.0)),
        _safe_float(opp.get("speed_air_x_self", 0.0)),
        _safe_float(opp.get("speed_y_self", 0.0)),
        _safe_float(opp.get("speed_x_attack", 0.0)),
        _safe_float(opp.get("speed_y_attack", 0.0)),
        _safe_float(opp.get("speed_ground_x_self", 0.0)),
    ]

    # --- global (1 dim) ---
    # s_global = [stage_id]

    vec = np.asarray(s_self + s_opp, dtype=np.float32)
    # vec = np.asarray(s_self + s_opp + s_global, dtype=np.float32)
    return vec


def _to_pm1(x: float) -> float:
    return 2.0 * x - 1.0


def _action_vectors(frame: Mapping[str, object], self_key: str) -> Tuple[np.ndarray, np.ndarray]:
    me = frame[self_key]  # type: ignore[index]
    ctrl = (me.get("controller") or {})  # type: ignore[union-attr]
    main = (ctrl.get("main_stick") or {})  # type: ignore[union-attr]
    cstk = (ctrl.get("c_stick") or {})  # type: ignore[union-attr]

    a_cont = np.asarray([
        _to_pm1(_safe_float(main.get("x", 0.0))),
        _to_pm1(_safe_float(main.get("y", 0.0))),
        _to_pm1(_safe_float(cstk.get("x", 0.0))),
        _to_pm1(_safe_float(cstk.get("y", 0.0))),
        _to_pm1(_safe_float(ctrl.get("l_shoulder", 0.0))),  # keep if you want shoulder in [-1,1]
    ], dtype=np.float32)

    btns = (ctrl.get("buttons") or {})  # type: ignore[union-attr]
    a_bin = np.asarray([_safe_bool(btns.get(name, False)) for name in BTN_ORDER], dtype=np.float32)
    return a_cont, a_bin


def _finite_min_max(a: Tensor) -> Tuple[Optional[float], Optional[float], int]:
    """Return (min, max, count) over finite values of |a| on CPU."""
    v = _abs_cpu(a)
    mask = torch.isfinite(v)
    if not torch.any(mask):
        return None, None, 0
    v = v[mask]
    return float(v.min()), float(v.max()), int(v.numel())


def estimate_rtg_scale(
        episodes: Sequence["Episode"],
        *,
        q: float = 0.95,
        horizon_frames: float = 60.0,
        concat_cap: int = 5_000_000,  # if total steps <= this, use exact path
        hist_bins: int = 4096,  # for streaming approx path
) -> float:
    """
    Returns max(1e-3, p95(|reward|) * horizon_frames).

    - Exact path if total steps is small enough.
    - Streaming histogram approximation if huge.
    - Always runs on CPU to dodge GPU kernel limits and fragmentation.
    """
    total_steps = _total_reward_steps(episodes)
    if total_steps == 0:
        return max(1e-3, 1.0 * horizon_frames)

    if total_steps <= concat_cap:
        # Exact path (CPU): build once, use nan-safe quantile.
        rews = torch.cat([_abs_cpu(ep.rewards) for ep in episodes], dim=0)
        mask = torch.isfinite(rews)
        p = float(torch.quantile(rews[mask], q)) if torch.any(mask) else 1.0
    else:
        # Big-data path: streaming histogram (no giant concat).
        p = float(_streaming_abs_quantile(episodes, q=q, bins=hist_bins))

    return max(1e-3, p * horizon_frames)


def _total_reward_steps(episodes: Sequence["Episode"]) -> int:
    total = 0
    for ep in episodes:
        total += int(ep.rewards.numel())
    return total


def _abs_cpu(x: Tensor) -> Tensor:
    # Detach & move to CPU only once; ensure 1D view for hist/quantile.
    return x.detach().abs().to("cpu").view(-1)


def _streaming_abs_quantile(
        episodes: Sequence["Episode"], *, q: float, bins: int = 4096
) -> Tensor:
    assert 0.0 <= q <= 1.0

    # Pass 1: finite-only min/max and total count
    vmin_f = math.inf
    vmax_f = -math.inf
    total = 0
    for ep in episodes:
        mn, mx, n = _finite_min_max(ep.rewards)
        if n == 0:
            continue
        total += n
        if mn is not None and mn < vmin_f:
            vmin_f = mn
        if mx is not None and mx > vmax_f:
            vmax_f = mx

    if total == 0 or not math.isfinite(vmin_f) or not math.isfinite(vmax_f):
        return torch.tensor(1.0)

    if vmin_f == vmax_f:
        return torch.tensor(vmin_f)

    # Pass 2: histogram over finite values only
    hist = torch.zeros(bins, dtype=torch.int64)
    scale = (bins - 1) / (vmax_f - vmin_f)
    for ep in episodes:
        v = _abs_cpu(ep.rewards)
        m = torch.isfinite(v)
        if not torch.any(m):
            continue
        v = v[m]
        idx = torch.clamp(((v - vmin_f) * scale).to(torch.long), 0, bins - 1)
        hist.index_add_(0, idx, torch.ones_like(idx, dtype=torch.int64))

    kth = int(math.ceil(q * total))
    cdf = hist.cumsum(0)

    # torch.searchsorted may not exist on very old versions — fall back to bucketize
    try:
        bin_idx = int(torch.searchsorted(cdf, torch.tensor(kth), right=False))
    except AttributeError:
        bin_idx = int(torch.bucketize(torch.tensor(kth), cdf, right=False))

    quant = vmin_f + (vmax_f - vmin_f) * (bin_idx / max(1, bins - 1))
    return torch.tensor(quant)


def structarray_to_episodes(arr: pa.StructArray) -> List[Episode]:
    frames: List[Mapping[str, object]] = arr.to_pylist()
    if not frames:
        return []

    def build_for(self_key: str, opp_key: str) -> Episode:
        states = np.stack([_state_vector(fr, self_key, opp_key) for fr in frames], axis=0)
        cont_list, bin_list = zip(*[_action_vectors(fr, self_key) for fr in frames])
        actions_cont = np.stack(cont_list, axis=0)
        actions_bin = np.stack(bin_list, axis=0)

        # --- NEW: proximity reward ---
        rewards = proximity_reward_series(frames, self_key)  # (T,)

        return Episode(
            states=torch.from_numpy(states),
            actions_cont=torch.from_numpy(actions_cont),
            actions_bin=torch.from_numpy(actions_bin),
            rewards=torch.from_numpy(rewards),
        )

    ep_p0 = build_for("p0", "p1")
    ep_p1 = build_for("p1", "p0")
    return [ep_p0, ep_p1]


class ColumnInfo(TypedDict):
    name: str  # flattened "p0_controller_main_stick_x"
    arrow_type: str  # e.g., "bool", "int32", "float64"
    numpy_dtype: str  # dtype before final cast when saving, e.g., "int8", "float32"


def _load_shard(npy_path: Path) -> Tuple[np.ndarray, List[ColumnInfo], str]:
    """Load a matrix + column metadata from a single shard."""
    sidecar = npy_path.with_suffix(".columns.json")
    if not sidecar.exists():
        raise FileNotFoundError(f"Missing sidecar {sidecar.name} for {npy_path.name}")

    mat = np.load(npy_path)
    with sidecar.open("r") as f:
        meta = json.load(f)

    cols: List[ColumnInfo] = cast(List[ColumnInfo], meta["columns"])
    sep: str = cast(str, meta.get("sep", "_"))
    if mat.shape[1] != len(cols):
        raise ValueError(
            f"Column count mismatch for {npy_path.name}: matrix has {mat.shape[1]}, "
            f"columns.json has {len(cols)}"
        )
    return mat, cols, sep


def _matrix_rows_to_frames(
        mat: np.ndarray, cols: List[ColumnInfo], sep: str
) -> List[Dict[str, Any]]:
    """Rebuild nested frame dicts row-by-row from flat matrix + metadata."""
    frames: List[Dict[str, Any]] = []
    # Pre-split paths once
    paths: List[List[str]] = [c["name"].split(sep) for c in cols]

    for r in range(mat.shape[0]):
        row = mat[r]
        fr: Dict[str, Any] = {}
        for j, path in enumerate(paths):
            info = cols[j]
            atype = info.get("arrow_type", "")
            ndt = info.get("numpy_dtype", "")
            raw = row[j]

            # Cast back to something close to original logical type.
            if atype == "bool":
                val: Any = bool(raw != 0)
            elif ndt.startswith("int") or ndt.startswith("uint"):
                # Values were cast to float for the matrix; recover ints
                val = int(raw)
            else:
                val = float(raw)

            # Insert into nested dict
            d = fr
            for k in path[:-1]:
                nxt = d.get(k)
                if not isinstance(nxt, dict):
                    nxt = {}
                    d[k] = nxt
                d = nxt
            d[path[-1]] = val
        frames.append(fr)
    return frames


def _frames_to_episodes(frames: List[Mapping[str, object]]) -> List["Episode"]:
    """Same logic as your old structarray_to_episodes, but starts from frames."""
    if not frames:
        return []

    def build_for(self_key: str, opp_key: str) -> "Episode":
        states = np.stack([_state_vector(fr, self_key, opp_key) for fr in frames], axis=0)
        cont_list, bin_list = zip(*[_action_vectors(fr, self_key) for fr in frames])
        actions_cont = np.stack(cont_list, axis=0)
        actions_bin = np.stack(bin_list, axis=0)
        rewards = proximity_reward_series(frames, self_key)  # (T,)
        return Episode(
            states=torch.from_numpy(states),
            actions_cont=torch.from_numpy(actions_cont),
            actions_bin=torch.from_numpy(actions_bin),
            rewards=torch.from_numpy(rewards),
        )

    return [build_for("p0", "p1"), build_for("p1", "p0")]


def _process_shard_for_episodes(npy_path: Path) -> List["Episode"]:
    """
    Work function executed in a separate process.
    NOTE: Everything it touches must be importable at module import time and picklable.
    """
    mat, cols, sep = _load_shard(npy_path)
    frames = _matrix_rows_to_frames(mat, cols, sep)
    return _frames_to_episodes(frames)


def _process_shard_safe(npy_path: Path) -> Tuple[str, Optional[List["Episode"]], Optional[str]]:
    """
    Wrapper that catches exceptions so the parent can log [skip] and continue.
    Returns (filename, episodes_or_None, error_or_None).
    """
    try:
        eps = _process_shard_for_episodes(npy_path)
        return (npy_path.name, eps, None)
    except Exception as e:  # noqa: BLE001 - we want to log and continue
        return (npy_path.name, None, str(e))


def load_replays(*, max_workers: Optional[int] = None) -> List["Episode"]:
    eps: List["Episode"] = []

    # Sort numerically if shards are named "0.npy", "1.npy", ...
    def sort_key(p: Path) -> Tuple[int, str]:
        return (int(p.stem), p.name) if p.stem.isdigit() else (-1, p.name)

    npy_files = sorted(SHARDS_DIR.glob("*.npy"), key=sort_key)
    if not npy_files:
        print(f"[warn] No shards found in {SHARDS_DIR}")
        return eps

    # Single file? Don’t spin up extra processes.
    if len(npy_files) == 1:
        name, shard_eps, err = _process_shard_safe(npy_files[0])
        if err is not None:
            print(f"[skip] {name}: {err}")
        else:
            eps.extend(shard_eps or [])
        return eps

    # Parallel path — uses processes (good for NumPy / CPU-heavy steps).
    # executor.map preserves input order so output remains shard-sorted.
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for name, shard_eps, err in ex.map(_process_shard_safe, npy_files):
            if err is not None:
                print(f"[skip] {name}: {err}")
                continue
            # shard_eps is not None if err is None
            eps.extend(shard_eps or [])

    return eps


# Bump this if Episode’s pickled shape/meaning changes.
SCHEMA_VERSION: int = 1

# Where to keep cached outputs (per-shard-dir).
CACHE_SUBDIR_NAME = ".replay_cache"

_MEMOIZED: Optional[List["Episode"]] = None  # in-process memo


@dataclass(frozen=True)
class _ShardEntry:
    path: str
    size: int
    mtime_ns: int


def _numeric_sort_key(p: Path) -> Tuple[int, str]:
    return (int(p.stem), p.name) if p.stem.isdigit() else (-1, p.name)


def _snapshot_shards_dir(shards_dir: Path) -> Dict[str, Any]:
    """Build a deterministic manifest describing shard inputs."""
    npy_files = sorted(shards_dir.glob("*.npy"), key=_numeric_sort_key)
    entries: List[_ShardEntry] = []
    for p in npy_files:
        st = p.stat()
        entries.append(_ShardEntry(path=str(p.resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns))
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "shards_dir": str(shards_dir.resolve()),
        "shards": [asdict(e) for e in entries],
    }
    return manifest


def _manifest_digest(manifest: Dict[str, Any]) -> str:
    blob = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _cache_dir_for(shards_dir: Path) -> Path:
    return shards_dir / CACHE_SUBDIR_NAME


def _cache_path_for(shards_dir: Path, digest: str) -> Path:
    cache_dir = _cache_dir_for(shards_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"episodes-{digest}.pkl.gz"


def _load_from_cache(path: Path) -> Optional[List["Episode"]]:
    try:
        with gzip.open(path, "rb") as f:
            obj = pickle.load(f)
        # A light runtime type assertion for safety:
        eps = obj if isinstance(obj, list) else None
        return eps  # type: ignore[return-value]
    except Exception:
        return None


def _save_to_cache(path: Path, eps: List["Episode"]) -> None:
    print(f"[info] Saving {len(eps)} episodes to {path}")
    # Write atomically to avoid torn writes
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wb") as f:
        pickle.dump(eps, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_replays_cached(
        *,
        force_recompute: bool = False,
        cache_ok: bool = True,
) -> List["Episode"]:
    """
    Load episodes with in-process memoization and on-disk cache.

    - If the shard set hasn't changed (by filenames/sizes/mtimes) and schema_version matches,
      the cached episodes are returned instantly.
    - Set `force_recompute=True` to ignore any cache.
    - Set env REPLAYS_NO_CACHE=1 to disable persistent cache ad-hoc.
    """
    global _MEMOIZED

    # 1) In-process memo (fast path within a single run)
    if not force_recompute and _MEMOIZED is not None:
        return _MEMOIZED

    # 2) Persistent cache
    persistent_ok = cache_ok and (os.environ.get("REPLAYS_NO_CACHE") != "1")
    manifest = _snapshot_shards_dir(SHARDS_DIR)
    digest = _manifest_digest(manifest)
    cache_path = _cache_path_for(SHARDS_DIR, digest)

    if not force_recompute and persistent_ok and cache_path.exists():
        cached = _load_from_cache(cache_path)
        if cached is not None:
            _MEMOIZED = cached
            return cached

    # 3) Miss → compute then save
    eps = load_replays()
    _MEMOIZED = eps

    if persistent_ok:
        try:
            _save_to_cache(cache_path, eps)
            # Optionally prune older caches to keep only a few:
        except Exception as e:
            print(f"[warn] failed to write cache {cache_path.name}: {e}")

    return eps


def infer_feature_dims(ep: Episode) -> FeatureDims:
    return FeatureDims(
        d_state=int(ep.states.shape[1]),
        n_cont=int(ep.actions_cont.shape[1]),
        n_bin=int(ep.actions_bin.shape[1]),
    )


class NormStats(TypedDict, total=False):
    state_mean: Tensor  # (D_state,)
    state_std: Tensor  # (D_state,)
    acont_mean: Tensor  # (n_cont,)
    acont_std: Tensor  # (n_cont,)


def save_dt(
        model: DecisionTransformer,
        cfg: TrainConfig,
        path: str,
        norm: Optional[NormStats] = None,
        extra: Optional[Dict] = None,
) -> None:
    """
    Saves a bundle with model state, config, action space, optional normalization stats,
    and any extra metadata you care about (e.g., git commit).
    """
    bundle = {
        "format": "melee-dt.v1",
        "model_state": model.state_dict(),
        "cfg": asdict(cfg),
        "action_space": {"n_cont": model.aspace.n_cont, "n_bin": model.aspace.n_bin},
        "norm": {k: v.cpu() for k, v in (norm or {}).items()},
        "extra": extra or {},
    }
    torch.save(bundle, path)


def load_dt(path: str, device: str = "mps") -> Tuple[DecisionTransformer, TrainConfig, NormStats]:
    """
    Reconstructs a DecisionTransformer from a saved bundle.
    Returns (model, cfg, norm).
    """
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("format") != "melee-dt.v1":
        raise ValueError(f"Unexpected bundle format: {ckpt.get('format')}")

    cfg_dict = ckpt["cfg"]
    cfg = TrainConfig(**cfg_dict)
    aspace_dict = ckpt["action_space"]
    aspace = ActionSpace(n_cont=aspace_dict["n_cont"], n_bin=aspace_dict["n_bin"])

    model = DecisionTransformer(
        d_state=cfg.d_state,
        aspace=aspace,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        max_len=cfg.max_len,
        dropout=cfg.dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    norm: NormStats = ckpt.get("norm", {})
    # Move tensors to device if present
    for k, v in list(norm.items()):
        norm[k] = v.to(device)  # type: ignore[assignment]
    return model, cfg, norm


def prepare_context(ep: Episode, L_ctx: int) -> tuple[Tensor, Tensor, Tensor]:
    """
    Build the context tensors the model expects:
      - state_seq: (L_ctx, D_state)
      - prev_action_cont: (L_ctx, n_cont)
      - prev_action_bin:  (L_ctx, n_bin)
    We feed the **previous** actions; the target for the last slot is the next action.
    """
    T = ep.states.shape[0]
    t1 = T  # use the last L_ctx frames
    t0 = max(0, t1 - L_ctx)
    states = ep.states[t0:t1]  # (L', Ds)
    acont = ep.actions_cont[t0:t1]  # (L', C)
    abin = ep.actions_bin[t0:t1]  # (L', B)

    # Shift actions right by one; pad zeros at the first slot
    acont_prev = torch.zeros_like(acont)
    abin_prev = torch.zeros_like(abin)
    if states.size(0) > 1:
        acont_prev[1:] = acont[:-1]
        abin_prev[1:] = abin[:-1]

    return states, acont_prev, abin_prev


def main_inference() -> None:
    device = "mps" if torch.mps.is_available() else "cpu"
    model, cfg, norm = load_dt("/Users/eppie/melee-ai/melee_dt_demo.pt", device=device)

    # Load one replay and build episodes
    arr = get_slp(
        '/Users/eppie/Downloads/replays_sorted/FOX_vs_FOX/1 - Cody Schwab (Fox), Azel (Fox) - Battlefield_1752517533904.slp')
    eps = structarray_to_episodes(arr)
    ep = eps[0]  # choose p0 perspective
    ep = ep.to(device)

    # (Optional) apply normalization if you saved stats
    # if "state_mean" in norm and "state_std" in norm:
    #     ep = Episode(
    #         states=(ep.states - norm["state_mean"]) / (norm["state_std"] + 1e-6),
    #         actions_cont=ep.actions_cont, actions_bin=ep.actions_bin, rewards=ep.rewards
    #     )

    L_ctx = min(cfg.seq_len, ep.states.shape[0])
    state_seq, acont_prev, abin_prev = prepare_context(ep, L_ctx)

    # For pure imitation you can keep RTG as a constant (e.g., 0.0).
    target_per_step = 0.2  # “aim to reduce distance by ~0.2 per frame”
    rtg_step = target_per_step * cfg.seq_len  # per-token constant RTG target
    rtg_step /= max(cfg.rtg_scale, 1e-8)

    acont_next, abin_next = act_greedy(
        model, rtg_step=rtg_step, state_seq=state_seq,
        prev_action_cont=acont_prev, prev_action_bin=abin_prev,
        rtg_scale=cfg.rtg_scale, device=device
    )
    print("Predicted continuous:", acont_next)  # tensor of shape (n_cont,)
    print("Predicted buttons:", (abin_next > 0.5))  # bools for each binary action


def build_segment_weights(ds: DTDataset) -> List[float]:
    # Priority = mean positive RTG over the segment (suffix sum already cached)
    weights = []
    for ep_id, t0 in tqdm(ds.index):
        ep = ds.eps[ep_id]
        T = ep.states.shape[0]
        t1 = min(t0 + ds.L, T)
        # use positive RTG (future progress) or positive rewards (immediate)
        rtg = ds.rtg_cache[ep_id][t0:t1]
        w = float(torch.clamp(rtg, min=0).mean().item())
        weights.append(max(w, 1e-4))  # avoid zeros
    return weights


SHARDS_DIR = Path("/Users/eppie/melee-ai/shards")


def load_episodes_from_shards(shards_dir: Path = SHARDS_DIR) -> List[Episode]:
    """
    Load all episodes from .npy shard files in `shards_dir`.

    Each .npy file may contain:
      - a 1-D object array of Episode objects,
      - a 0-D object array wrapping a list[Episode] or a single Episode,
      - (edge case) anything else -> skipped with a warning.

    Returns:
        List[Episode]: concatenated episodes from all shards.
    """
    episodes: List[Episode] = []
    shard_paths = sorted(shards_dir.glob("*.npy"))

    if not shard_paths:
        print(f"[warn] No .npy shards found in {shards_dir}")

    for p in shard_paths:
        try:
            arr = np.load(p, allow_pickle=True)
        except Exception as e:
            print(f"[skip] {p.name}: failed to load – {e}")
            continue

        try:
            if isinstance(arr, np.ndarray) and arr.dtype == object:
                if arr.ndim == 0:
                    # 0-D object array: unwrap
                    obj: Any = arr.item()
                    if isinstance(obj, (list, tuple)):
                        episodes.extend(cast(List[Episode], list(obj)))
                    else:
                        episodes.append(cast(Episode, obj))
                elif arr.ndim == 1:
                    # 1-D array of Episodes
                    items: List[Episode] = [cast(Episode, x) for x in arr.tolist()]
                    episodes.extend(items)
                else:
                    print(f"[skip] {p.name}: unexpected ndarray shape {arr.shape}")
            else:
                # Unexpected content type
                print(f"[skip] {p.name}: unexpected content type {type(arr).__name__}")
        except Exception as e:
            print(f"[skip] {p.name}: content parse error – {e}")

    return episodes


def main() -> None:
    # paths = [Path(
    #     '/Users/eppie/Downloads/replays_sorted/FOX_vs_FOX/1 - Cody Schwab (Fox), Azel (Fox) - Battlefield_1752517533904.slp')]
    #
    # print(f"Loading {len(paths)} replays …")
    # episodes = load_replays(paths)
    # if not episodes:
    #     raise SystemExit("No episodes parsed.")
    print(f"Loading episodes from shards in {SHARDS_DIR} …")
    episodes = load_replays_cached()
    if not episodes:
        raise SystemExit(f"No episodes parsed from shards in {SHARDS_DIR}.")
    # Infer dims from the first episode
    dims = infer_feature_dims(episodes[0])
    aspace = ActionSpace(n_cont=dims.n_cont, n_bin=dims.n_bin)

    # Split train/val
    split = max(2, int(0.8 * len(episodes)))
    train_eps, val_eps = episodes[:split], episodes[split:]
    print(f"Training {len(train_eps)} episodes")
    print(f"Validation {len(val_eps)} episodes")
    rtg_scale = estimate_rtg_scale(episodes)
    print(f"RTG Scale: {rtg_scale}")
    cfg = TrainConfig(
        d_state=dims.d_state,
        action_space=aspace,
        seq_len=SEQUENCE_LENGTH,  # 1s context @ 60 FPS
        batch_size=128,
        d_model=512,
        n_layers=2,
        n_heads=4,
        d_ff=1024,
        lr=1e-3,
        epochs=5,  # short demo run
        device="mps" if torch.mps.is_available() else "cpu",
        num_workers=16,
        rtg_scale=rtg_scale,
        weight_mode="exp_rtg",
        weight_beta=3.0,
        weight_clip=0,
        norm_weights=False,
        # weight_clip=20.0,
        # norm_weights=True,
    )

    print(f"State dim={dims.d_state} | n_cont={dims.n_cont} | n_bin={dims.n_bin}")
    model = train_dt(train_eps, cfg, val_episodes=val_eps)
    save_dt(model, cfg, path="melee_dt_demo.pt")
    print("Saved to melee_dt_demo.pt")


if __name__ == "__main__":
    main()
    # main_inference()
