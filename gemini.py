from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Tuple, List, Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

# ================================================================
# CHOOSE YOUR TASK (1-10)
# ================================================================
# This single constant reconfigures the entire script.
TASK_ID = 2


# ================================================================


# ================================================================
# Configuration Block
# ================================================================
@dataclass(frozen=True)
class Config:
    # Task & Data
    task_id: int
    seq_len: int
    tgt_len: int
    in_vocab: int
    out_vocab: int
    bos_id: int = field(init=False)
    eos_id: int = field(init=False)

    # Infrastructure
    device: torch.device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    train_size: int = 8_000
    val_size: int = 1_000

    # Model
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    ffn_dim: int = 4 * 128
    dropout: float = 0.1

    # Training
    batch_size: int = 128
    epochs: int = 10  # Reduced for faster demo runs; can be increased
    lr: float = 3e-4
    print_every: int = 1

    def __post_init__(self):
        # BOS/EOS are derived from the output vocabulary size
        object.__setattr__(self, "bos_id", self.out_vocab)
        object.__setattr__(self, "eos_id", self.out_vocab + 1)
        # The final vocabulary must include BOS and EOS
        object.__setattr__(self, "out_vocab", self.out_vocab + 2)


# ================================================================
# Task-Specific Parameter Setup
# ================================================================
if TASK_ID == 1:  # "Go to Center"
    _in_vocab, _out_vocab = 21, 3  # Input: 21 positions (-10 to 10), Output: 3 actions
    CFG = Config(
        task_id=TASK_ID, seq_len=1, tgt_len=2, in_vocab=_in_vocab, out_vocab=_out_vocab
    )
elif TASK_ID == 2:  # "Maintain Optimal Distance"
    _in_vocab, _out_vocab = 21 * 21, 3  # Input: my_pos * opp_pos, Output: 3 actions
    CFG = Config(
        task_id=TASK_ID, seq_len=1, tgt_len=2, in_vocab=_in_vocab, out_vocab=_out_vocab
    )
elif TASK_ID == 3:  # "Execute a Hard-Coded Combo"
    _in_vocab, _out_vocab = 2, 4  # Input: hitstun (T/F), Output: 4 actions
    CFG = Config(
        task_id=TASK_ID, seq_len=1, tgt_len=5, in_vocab=_in_vocab, out_vocab=_out_vocab
    )
elif TASK_ID == 4:  # "Reactive Defense"
    _in_vocab, _out_vocab = 3, 4  # Input: 3 opp actions, Output: 4 my actions
    CFG = Config(
        task_id=TASK_ID, seq_len=3, tgt_len=4, in_vocab=_in_vocab, out_vocab=_out_vocab
    )
elif TASK_ID == 5:  # "Anti-Air Defense"
    _in_vocab, _out_vocab = (
        21 * 21,
        3,
    )  # Input: my_y * opp_y, Output: 3 actions (move, stay, up-tilt)
    CFG = Config(
        task_id=TASK_ID, seq_len=5, tgt_len=4, in_vocab=_in_vocab, out_vocab=_out_vocab
    )
elif TASK_ID == 6:  # "Melee Triangle (RPS)"
    _in_vocab, _out_vocab = 4, 4  # Input: 4 opp actions, Output: 4 my actions
    CFG = Config(
        task_id=TASK_ID, seq_len=1, tgt_len=2, in_vocab=_in_vocab, out_vocab=_out_vocab
    )
elif TASK_ID == 7:  # "Wavedash Spacing"
    # Input: target distance token, Output: sequence of quantized stick/button presses
    _in_vocab, _out_vocab = 40, 10
    CFG = Config(
        task_id=TASK_ID, seq_len=1, tgt_len=5, in_vocab=_in_vocab, out_vocab=_out_vocab
    )
elif TASK_ID == 8:  # "SHFFL" (Multi-modal action)
    # Input: simple state, Output: flattened multi-modal action sequence
    _in_vocab = 2
    _out_vocab = 4 * 3  # 4 joystick-Y states * 3 button states (None, A, L)
    CFG = Config(
        task_id=TASK_ID, seq_len=1, tgt_len=6, in_vocab=_in_vocab, out_vocab=_out_vocab
    )
elif TASK_ID == 9:  # "Kill Confirm"
    # Input: hitstun * opponent_percent_bucket
    _in_vocab, _out_vocab = (
        2 * 10,
        5,
    )  # 2 hitstun states, 10 percent buckets; 5 possible moves
    CFG = Config(
        task_id=TASK_ID, seq_len=1, tgt_len=4, in_vocab=_in_vocab, out_vocab=_out_vocab
    )
elif TASK_ID == 10:  # "Full Imitation (simplified)"
    # Input: sequence of (my_pos, opp_pos), Output: sequence of actions
    _in_vocab, _out_vocab = 11 * 11, 4
    CFG = Config(
        task_id=TASK_ID, seq_len=4, tgt_len=3, in_vocab=_in_vocab, out_vocab=_out_vocab
    )
else:
    raise ValueError(f"Invalid TASK_ID: {TASK_ID}")


# ================================================================
# Reproducibility & Model Code (Largely Unchanged)
# ================================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


set_seed(42)


def generate_causal_mask(tgt_len: int, device: torch.device) -> Tensor:
    return torch.triu(
        torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool), diagonal=1
    )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1), :]


class Seq2SeqTransformer(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.src_emb = nn.Embedding(cfg.in_vocab, cfg.d_model)
        self.tgt_emb = nn.Embedding(cfg.out_vocab, cfg.d_model)
        self.pos = PositionalEncoding(cfg.d_model)

        # This part was correct - use keyword arguments
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.num_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.num_layers)
        self.proj = nn.Linear(cfg.d_model, cfg.out_vocab)

    def encode(self, src: Tensor) -> Tensor:
        return self.encoder(self.pos(self.src_emb(src)))

    # We now accept the mask as an argument again
    def decode(self, tgt_in: Tensor, memory: Tensor, tgt_mask: Tensor) -> Tensor:
        return self.decoder(self.pos(self.tgt_emb(tgt_in)), memory, tgt_mask=tgt_mask)

    def forward(self, src: Tensor, tgt_in: Tensor) -> Tensor:
        # Generate the causal mask and pass it to the decoder
        tgt_len = tgt_in.size(1)
        tgt_mask = generate_causal_mask(tgt_len, src.device)

        memory = self.encode(src)
        return self.proj(self.decode(tgt_in, memory, tgt_mask))


# ================================================================
# Universal Dataset for All Melee Tasks
# ================================================================
class MeleeTaskDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, length: int, cfg: Config):
        self.length = length
        self.cfg = cfg
        self._g = torch.Generator().manual_seed(42)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        cfg = self.cfg

        # --- Quantization/Tokenization Helpers ---
        def quantize_pos(p, min_v, max_v):
            return max(0, min(max_v - min_v, int(p - min_v)))

        # --- Task-Specific Data Generation ---
        if cfg.task_id == 1:  # "Go to Center"
            pos = random.randint(-10, 10)
            src_token = quantize_pos(pos, -10, 10)
            if pos < -2:
                action = 0  # move_right
            elif pos > 2:
                action = 1  # move_left
            else:
                action = 2  # stay
            src = torch.tensor([src_token], dtype=torch.long)
            y = torch.tensor([action, cfg.eos_id], dtype=torch.long)

        elif cfg.task_id == 2:  # "Maintain Optimal Distance"
            my_pos, opp_pos = random.randint(-10, 10), random.randint(-10, 10)
            my_tok = quantize_pos(my_pos, -10, 10)
            opp_tok = quantize_pos(opp_pos, -10, 10)
            src_token = my_tok * 21 + opp_tok  # Flatten into single token
            dist = abs(my_pos - opp_pos)
            if dist > 7:
                action = 0 if my_pos < opp_pos else 1  # move towards
            elif dist < 3:
                action = 1 if my_pos < opp_pos else 0  # move away
            else:
                action = 2  # stay
            src = torch.tensor([src_token], dtype=torch.long)
            y = torch.tensor([action, cfg.eos_id], dtype=torch.long)

        elif cfg.task_id == 3:  # "Execute a Hard-Coded Combo"
            in_hitstun = random.choice([True, False])
            src_token = 1 if in_hitstun else 0
            if in_hitstun:
                actions = [0, 1, 2, 3]  # d-tilt, jump, f-air, EOS (already added)
            else:
                actions = [4] * 4  # wait, wait, wait, EOS
            src = torch.tensor([src_token], dtype=torch.long)
            y = torch.tensor(actions[:-1] + [cfg.eos_id], dtype=torch.long)

        elif cfg.task_id == 4:  # "Reactive Defense"
            opp_actions = torch.randint(
                0, 3, (cfg.seq_len,), generator=self._g
            ).tolist()
            if opp_actions[-1] == 1:  # is running at you
                my_actions = [0, cfg.eos_id, -1, -1]  # dash_back
            elif opp_actions[-1] == 2:  # is charging smash
                my_actions = [1, 2, cfg.eos_id, -1]  # shield, roll
            else:
                my_actions = [3, cfg.eos_id, -1, -1]  # wait
            src = torch.tensor(opp_actions, dtype=torch.long)
            y = torch.tensor([a for a in my_actions if a != -1], dtype=torch.long)

        elif cfg.task_id == 6:  # "Melee Triangle (RPS)"
            opp_action = random.randint(1, 3)  # 1:attack, 2:shield, 3:grab
            if opp_action == 1:
                my_action = 2  # shield
            elif opp_action == 2:
                my_action = 3  # grab
            else:
                my_action = 1  # attack
            src = torch.tensor([opp_action], dtype=torch.long)
            y = torch.tensor([my_action, cfg.eos_id], dtype=torch.long)

        # NOTE: Tasks 5, 7, 8, 9, 10 are left as exercises but follow the same pattern.
        # This implementation covers a representative subset.
        else:  # Fallback for unimplemented tasks
            src = torch.randint(
                0, cfg.in_vocab, (cfg.seq_len,), generator=self._g, dtype=torch.long
            )
            y = torch.randint(
                0,
                cfg.out_vocab - 2,
                (cfg.tgt_len - 1,),
                generator=self._g,
                dtype=torch.long,
            )
            y = torch.cat([y, torch.tensor([cfg.eos_id], dtype=torch.long)])

        # Ensure target `y` has the correct fixed length for batching, padding if necessary
        y_padded = torch.full((cfg.tgt_len,), -1, dtype=torch.long)  # Pad with -1
        y_padded[: len(y)] = y
        return src, y_padded


# ================================================================
# Universal Training & Evaluation
# ================================================================
def build_tgt_in(y: Tensor, cfg: Config) -> Tensor:
    bos_col = torch.full((y.size(0), 1), cfg.bos_id, dtype=torch.long, device=y.device)
    # y might be shorter than TGT_LEN-1 due to early EOS, so we slice from y
    tgt_in = torch.cat([bos_col, y[:, :-1]], dim=1)
    return tgt_in


def train_one_epoch(model, loader, opt, loss_fn, cfg):
    model.train()
    total_loss, total_tok, total_correct = 0, 0, 0
    for src, y in loader:
        src, y = src.to(cfg.device), y.to(cfg.device)
        tgt_in = build_tgt_in(y, cfg)
        logits = model(src, tgt_in)

        # Ignore padding (-1) in loss calculation
        loss = loss_fn(logits.view(-1, cfg.out_vocab), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            mask = y != -1
            pred = logits.argmax(dim=-1)
            total_correct += (pred[mask] == y[mask]).sum().item()
            total_tok += mask.sum().item()
            total_loss += loss.item() * src.size(0)

    return total_loss / len(loader), total_correct / max(total_tok, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, cfg):
    model.eval()
    total_loss, total_tok, total_correct = 0, 0, 0
    for src, y in loader:
        src, y = src.to(cfg.device), y.to(cfg.device)
        tgt_in = build_tgt_in(y, cfg)
        logits = model(src, tgt_in)

        mask = y != -1
        loss = loss_fn(logits.view(-1, cfg.out_vocab), y.view(-1))
        pred = logits.argmax(dim=-1)
        total_correct += (pred[mask] == y[mask]).sum().item()
        total_tok += mask.sum().item()
        total_loss += loss.item() * src.size(0)

    return total_loss / len(loader), total_correct / max(total_tok, 1)


@torch.no_grad()
def greedy_decode(model: Seq2SeqTransformer, src_single: Tensor) -> Tensor:
    cfg = model.cfg
    device = src_single.device
    memory = model.encode(src_single)
    y = torch.tensor([[cfg.bos_id]], dtype=torch.long, device=device)
    for _ in range(cfg.tgt_len):
        # Generate the mask for the current sequence length
        tgt_len = y.size(1)
        tgt_mask = generate_causal_mask(tgt_len, device)

        dec = model.decode(y, memory, tgt_mask)
        next_logits = model.proj(dec[:, -1:, :])  # Get logits for the last token
        next_token = next_logits.argmax(dim=-1)

        if int(next_token.item()) == cfg.eos_id:
            break
        y = torch.cat([y, next_token], dim=1)
    return y[:, 1:]


def get_demo_data(task_id: int) -> Tuple[List[List[int]], Callable, Callable]:
    if task_id == 1:
        examples = [[-8], [20], [10]]

        def d_src(s):
            return f"pos={s[0] - 10}"

        def d_tgt(t):
            return {0: "→", 1: "←", 2: "stay"}.get(t, "eos")

    elif task_id == 2:
        examples = [[5 * 21 + 15], [10 * 21 + 11], [8 * 21 + 8]]

        def d_src(s):
            return f"me={s[0] // 21 - 10}, opp={s[0] % 21 - 10}"

        def d_tgt(t):
            return {0: "→", 1: "←", 2: "stay"}.get(t, "eos")

    elif task_id == 3:
        examples = [[1], [0]]

        def d_src(s):
            return f"in_hitstun={bool(s[0])}"

        def d_tgt(t):
            return {0: "d-tilt", 1: "jump", 2: "f-air", 3: "eos", 4: "wait"}.get(
                t, "eos"
            )

    elif task_id == 4:
        examples = [[0, 0, 1], [0, 2, 2], [0, 0, 0]]
        opp_map = {0: "stand", 1: "run", 2: "smash"}

        def d_src(s):
            return f"opp_hist={[opp_map[x] for x in s]}"

        def d_tgt(t):
            return {0: "d-back", 1: "shield", 2: "roll", 3: "wait"}.get(t, "eos")

    elif task_id == 6:
        examples = [[1], [2], [3]]

        def d_src(s):
            return f"opp_does={ {1: 'attack', 2: 'shield', 3: 'grab'}[s[0]]}"

        def d_tgt(t):
            return {1: "attack", 2: "shield", 3: "grab"}.get(t, "eos")

    else:  # Fallback
        examples = [[i for i in range(CFG.seq_len)]]

        def d_src(s):
            return str(s)

        def d_tgt(t):
            return str(t)

    return examples, d_src, d_tgt


# ================================================================
# Main Orchestrator
# ================================================================
def main() -> None:
    print(f"--- Task {CFG.task_id} ---")
    print(
        f"Config: L={CFG.seq_len}, T_max={CFG.tgt_len}, Vx={CFG.in_vocab}, Vy={CFG.out_vocab} (incl. BOS/EOS)"
    )

    train_set = MeleeTaskDataset(CFG.train_size, CFG)
    train_loader = DataLoader(train_set, batch_size=CFG.batch_size, shuffle=True)
    set_seed(123)

    val_set = MeleeTaskDataset(CFG.val_size, CFG)
    val_loader = DataLoader(val_set, batch_size=CFG.batch_size, shuffle=False)

    model = Seq2SeqTransformer(CFG).to(CFG.device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding in loss

    print(f"\nTraining on {CFG.device} for {CFG.epochs} epochs...")
    for epoch in range(1, CFG.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, loss_fn, CFG)
        va_loss, va_acc = evaluate(model, val_loader, loss_fn, CFG)
        if epoch % CFG.print_every == 0 or epoch == 1 or epoch == CFG.epochs:
            print(
                f"Epoch {epoch:02d} | Train Loss {tr_loss:.4f} Acc {tr_acc * 100:5.2f}% | Val Loss {va_loss:.4f} Acc {va_acc * 100:5.2f}%"
            )

    print("\n--- Greedy Decode Demo ---")

    def get_task1_gold_action(pos_token: int) -> int:
        """Given a source token for Task 1, return the correct action token."""
        pos = pos_token - 10  # De-quantize the token back to a position
        if pos < -2:
            return 0  # move_right
        if pos > 2:
            return 1  # move_left
        return 2  # stay

    demo_examples, detokenize_src, detokenize_tgt = get_demo_data(CFG.task_id)

    for seq in demo_examples:
        src_token = seq[0]
        src = torch.tensor([seq], dtype=torch.long, device=CFG.device)
        out_tokens = greedy_decode(model, src).cpu().squeeze(0).tolist()

        # Directly generate the correct gold standard
        gold_action_token = get_task1_gold_action(src_token)

        print(f"IN:   {detokenize_src(seq)}")
        print(f"GOLD: {[detokenize_tgt(gold_action_token)]}")
        print(f"OUT:  {[detokenize_tgt(t) for t in out_tokens]}")
        print("---")


if __name__ == "__main__":
    main()
