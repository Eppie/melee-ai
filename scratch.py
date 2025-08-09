from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

# ===========================
# ==== CONFIG CONSTANTS =====
# ===========================
# TODO: Tasks 3, 5, 9 don't learn properly
# TODO: Task 6 throws an exception
# TODO: Task 7 has 0 loss but incorrect
# TODO: Task 8 doesn't learn properly and also throws an exception

TASK_ID: int = 10                  # 1..10
SEED: int = 42
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

L: int = 12
BATCH_SIZE: int = 128
EPOCHS: int = 40
LR: float = 2e-3

D_MODEL: int = 128
NHEAD: int = 4
N_LAYERS: int = 2
FFN_DIM: int = 4 * D_MODEL
DROPOUT: float = 0.1

# Per-task params (defaults; some used conditionally)
DELAY: int = 3                 # tasks 1, 6
MOD_BASE: int = 3              # task 2
COOLDOWN_E: int = 4            # task 3
JUMP_SQUAT_FRAMES: int = 3     # task 4
CANCEL_K: int = 3              # task 5
DI_M_FRAMES: int = 3           # task 6
STAGE_RANGE: int = 6           # task 7 1-D line [-6, +6]
OFFSTAGE_THRESH: int = 5       # task 7 |x|>5 is offstage
FEINT_PROB: float = 0.35       # task 9
HISTORY_N: int = 8             # task 9

PLAN_HORIZON: int = 16         # task 10 (internal, output still length L)

# Task-3 loss config (does not affect other tasks)
TASK3_USE_WEIGHTED_CE: bool = True
TASK3_MIN_POS_WEIGHT: float = 1.0  # lower bound on class-1 weight
# Task-3 loss variant and focal parameters
TASK3_LOSS_MODE: str = "hazard"  # one of {"balanced_ce", "weighted_ce", "focal", "hazard"}
TASK3_FOCAL_GAMMA: float = 2.0
TASK3_FOCAL_ALPHA_POS: float = 0.75  # alpha for positive class in focal loss

# Demo prints
DEMO_SAMPLES: int = 3

# Task-specific vocab declarations (will be overridden by task registry)
IN_VOCAB: int = 16
DEC_VOCAB: int = 8
USE_DEC_CE: bool = True
BOS_ID: int = 6
EOS_ID: int = 7

# Extra heads per task will be provided by registry:
#   EXTRA_CE: Dict[str, int]   (per-timestep class id)
#   EXTRA_BCE: Dict[str, int]  (per-timestep multi-label)
#   LOSS_WEIGHTS: Dict[str, float] including 'dec_ce' and head names

# =================================
# ==== UTILS / COMMON COMPONENTS ===
# =================================
g = torch.Generator().manual_seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    try:
        torch.mps.manual_seed(SEED)  # type: ignore[attr-defined]
    except Exception:
        pass

def posenc(d_model: int, max_len: int) -> Tensor:
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, max_len, D)

def causal_mask(T: int, device: torch.device) -> Tensor:
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), 1)

# =================================
# ==== MODEL ======================
# =================================
class Model(nn.Module):
    def __init__(self, in_vocab: int, dec_vocab: int, extra_ce: Dict[str, int], extra_bce: Dict[str, int]):
        super().__init__()
        self.src_emb = nn.Embedding(in_vocab, D_MODEL)
        self.tgt_emb = nn.Embedding(max(dec_vocab, 1), D_MODEL)  # allow dec_vocab=1
        self.pe = nn.Parameter(posenc(D_MODEL, 2048), requires_grad=False)

        enc = nn.TransformerEncoderLayer(D_MODEL, NHEAD, FFN_DIM, DROPOUT, batch_first=True, activation="gelu")
        dec = nn.TransformerDecoderLayer(D_MODEL, NHEAD, FFN_DIM, DROPOUT, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=N_LAYERS)
        self.decoder = nn.TransformerDecoder(dec, num_layers=N_LAYERS)

        self.dec_head = nn.Linear(D_MODEL, dec_vocab) if dec_vocab > 1 else None
        self.ce_heads = nn.ModuleDict({k: nn.Linear(D_MODEL, v) for k, v in extra_ce.items()})
        self.bce_heads = nn.ModuleDict({k: nn.Linear(D_MODEL, v) for k, v in extra_bce.items()})

    def forward(self, src: Tensor, tgt_in: Tensor) -> Dict[str, Tensor]:
        # src: (B, L) ints; tgt_in: (B, T=L or L+1) ints in [0..dec_vocab-1]
        B, L_ = src.shape
        T = tgt_in.shape[1]
        src_h = self.encoder(self.src_emb(src) + self.pe[:, :L_, :])
        tgt_mask = causal_mask(T, src.device)
        dec_h = self.decoder(self.tgt_emb(tgt_in) + self.pe[:, :T, :], src_h, tgt_mask=tgt_mask)

        out: Dict[str, Tensor] = {}
        if self.dec_head is not None:
            out["dec_ce"] = self.dec_head(dec_h)            # (B, T, dec_vocab)
        for k, head in self.ce_heads.items():
            out[k] = head(dec_h)                            # (B, T, C_k)
        for k, head in self.bce_heads.items():
            out[k] = head(dec_h)                            # (B, T, K_k) logits
        return out

# =================================
# ==== TASK REGISTRY ==============
# =================================
@dataclass(frozen=True)
class TaskSpec:
    in_vocab: int
    dec_vocab: int
    use_dec_ce: bool
    extra_ce: Dict[str, int]
    extra_bce: Dict[str, int]
    loss_weights: Dict[str, float]

class ToyTasks:
    @staticmethod
    def _bos_cat(y: Tensor, dec_vocab: int) -> Tuple[Tensor, Tensor]:
        if dec_vocab <= 1:
            tgt_in = torch.zeros((y.size(0), y.size(1)), dtype=torch.long)  # dummy zeros
            return tgt_in, y
        bos = torch.full((y.size(0), 1), BOS_ID, dtype=torch.long)
        return torch.cat([bos, y[:, :-1]], 1), y

    @staticmethod
    def task1_shifted_copy(L: int, delay: int) -> Tuple[TaskSpec, Dataset]:
        Vx = 5; Vy = 6
        class DS(Dataset):
            def __len__(self): return 8000
            def __getitem__(self, idx):
                x = torch.randint(0, Vx, (L,), generator=g)
                y = torch.empty(L, dtype=torch.long)
                y[:delay] = 0
                y[delay:] = x[:-delay]
                y[-1] = 4 if delay <= L-1 else 0
                y = torch.where(torch.arange(L)==L-1, torch.tensor(Vy-1), y)  # EOS=5
                return x, y, {}, {}, {}
        spec = TaskSpec(Vx, Vy, True, {}, {}, {"dec_ce":1.0})
        return spec, DS()

    @staticmethod
    def task2_running_mod(L: int, base: int) -> Tuple[TaskSpec, Dataset]:
        Vx = 5; Vy = base + 1  # +EOS
        class DS(Dataset):
            def __len__(self): return 8000
            def __getitem__(self, idx):
                x = torch.randint(0, Vx, (L,), generator=g)
                s = 0; y_list: List[int] = []
                for t in range(L-1):
                    s = (s + int(x[t])) % base
                    y_list.append(s)
                s = (s + int(x[L-1])) % base
                y_list.append(s)
                y = torch.tensor(y_list, dtype=torch.long)
                y[-1] = Vy-1
                return x, y, {}, {}, {}
        spec = TaskSpec(Vx, Vy, True, {}, {}, {"dec_ce":1.0})
        return spec, DS()

    @staticmethod
    def task3_cooldown(L: int, E: int) -> Tuple[TaskSpec, Dataset]:
        Vx = 2; Vy = 1  # no dec ce; extra ce 'press' with 2 classes
        class DS(Dataset):
            def __len__(self): return 8000
            def __getitem__(self, idx):
                req = torch.bernoulli(torch.full((L,), 0.3, dtype=torch.float32), generator=g).long()
                press = torch.zeros(L, dtype=torch.long)
                cd = 0
                for t in range(L):
                    if cd==0 and req[t]==1:
                        press[t]=1; cd=E
                    else:
                        press[t]=0
                    cd=max(0, cd-1)
                x = req
                return x, torch.zeros(L, dtype=torch.long), {"press": press}, {}, {}
        spec = TaskSpec(Vx, Vy, False, {"press":2}, {}, {"press":1.0})
        return spec, DS()

    @staticmethod
    def task4_jump_squat(L: int, k: int) -> Tuple[TaskSpec, Dataset]:
        Vx = 2; Vy = 1
        class DS(Dataset):
            def __len__(self): return 8000
            def __getitem__(self, idx):
                airborne_goal = torch.zeros(L, dtype=torch.long)
                for t in range(k, L, 5):
                    airborne_goal[t]=1
                jump = torch.zeros(L, dtype=torch.long)
                for t in range(L):
                    if airborne_goal[t]==1:
                        for j in range(t-k, t+1):
                            if 0 <= j < L: jump[j]=1
                return airborne_goal, torch.zeros(L, dtype=torch.long), {"jump": jump, "airborne": airborne_goal}, {}, {}
        spec = TaskSpec(Vx, Vy, False, {"jump":2, "airborne":2}, {}, {"jump":0.7, "airborne":0.3})
        return spec, DS()

    @staticmethod
    def task5_cancel_window(L: int, K: int) -> Tuple[TaskSpec, Dataset]:
        Vx = 3; Vy = 1
        N,H,M = 0,1,2
        class DS(Dataset):
            def __len__(self): return 8000
            def __getitem__(self, idx):
                ev = torch.full((L,), N, dtype=torch.long)
                for t in range(2, L, 6):
                    ev[t]=H
                for t in range(5, L, 11):
                    ev[t]=M
                cancel = torch.zeros(L, dtype=torch.long)
                t=0
                while t<L:
                    if ev[t]==H:
                        fire = min(L-1, t+1)
                        cancel[fire]=1
                        t = t+K
                    else:
                        t += 1
                return ev, torch.zeros(L, dtype=torch.long), {"cancel": cancel}, {}, {}
        spec = TaskSpec(Vx, Vy, False, {"cancel":2}, {}, {"cancel":1.0})
        return spec, DS()

    @staticmethod
    def task6_di_response(L: int, delay: int, M: int) -> Tuple[TaskSpec, Dataset]:
        dirs = ["C","L","R","U","D"]
        Vx = len(dirs); Vy = len(dirs)  # use dec_ce to predict stick bins
        idx = {s:i for i,s in enumerate(dirs)}
        opp = {idx["L"]:idx["R"], idx["R"]:idx["L"], idx["U"]:idx["D"], idx["D"]:idx["U"], idx["C"]:idx["C"]}
        class DS(Dataset):
            def __len__(self): return 8000
            def __getitem__(self, _):
                impact = torch.full((L,), idx["C"], dtype=torch.long)
                for t in range(2, L, 7):
                    impact[t] = random.choice([idx["L"], idx["R"], idx["U"], idx["D"]])
                y = torch.full((L,), idx["C"], dtype=torch.long)
                for t in range(L):
                    if impact[t]!=idx["C"]:
                        t0 = t + delay
                        for j in range(t0, min(L, t0+M)):
                            y[j] = opp[impact[t]]
                return impact, y, {}, {}, {}
        spec = TaskSpec(Vx, Vy, True, {}, {}, {"dec_ce":1.0})
        return spec, DS()

    @staticmethod
    def task7_stage_positioning(L: int, rng: int, off: int) -> Tuple[TaskSpec, Dataset]:
        pos_vals = list(range(-rng, rng+1))  # map to tokens by offset
        Vx = len(pos_vals); Vy = 5  # action bins {-2,-1,0,+1,+2}
        offset = rng
        class DS(Dataset):
            def __len__(self): return 8000
            def __getitem__(self, _):
                x0 = random.randint(-rng, rng)
                pos = [x0]
                for _ in range(L-1):
                    pos.append(max(-rng, min(rng, pos[-1] + random.choice([-1,0,1]))))
                actions = torch.zeros(L, dtype=torch.long)
                for t in range(L):
                    x = pos[t]
                    if abs(x) > off: actions[t] = 2 + (-1 if x>0 else 1)  # push strongly toward inside
                    else:
                        actions[t] = 2 + (-1 if x>0 else (1 if x<0 else 0))
                    actions[t] = actions[t] + 2  # map {-2,-1,0,+1,+2} -> {0..4}
                src = torch.tensor([p+offset for p in pos], dtype=torch.long)
                y = actions
                return src, y, {}, {}, {}
        spec = TaskSpec(Vx, Vy, True, {}, {}, {"dec_ce":1.0})
        return spec, DS()

    @staticmethod
    def task8_multi_head_coherence(L: int) -> Tuple[TaskSpec, Dataset]:
        states = {"GROUND":0, "AIR":1}
        hitst = {0,1}
        facing = {"L":0,"R":1}
        Vx = 4  # coarse state token (GROUND/AIR x FACING)
        stick_bins = 5
        class DS(Dataset):
            def __len__(self): return 8000
            def __getitem__(self, _):
                ground = random.choice([0,1]); face = random.choice([0,1]); hit = random.choice([0,1])
                src = []
                y_stick = torch.zeros(L, dtype=torch.long)
                y_btn = torch.zeros(L, 3, dtype=torch.float32)  # [JUMP, SHIELD, ATTACK]
                for t in range(L):
                    ground = random.choice([0,1]); face = random.choice([0,1]); hit = random.choice([0,1])
                    src.append(ground*2 + face)
                    if hit==1:
                        y_btn[t] = torch.tensor([0,0,0])
                        y_stick[t] = 0
                    else:
                        # simple rule: if ATTACK=1, stick forward; AIR forbids SHIELD
                        if random.random() < 0.5:
                            y_btn[t] = torch.tensor([0, int(ground==1 and 0), 1])
                            y_stick[t] = 2 + (1 if face==1 else -1)  # map forward to 3 or 1
                        else:
                            y_btn[t] = torch.tensor([1, 0 if ground==0 else 1, 0])
                            y_stick[t] = 0
                src = torch.tensor(src, dtype=torch.long)
                return src, torch.zeros(L, dtype=torch.long), {"stick": y_stick}, {"buttons": y_btn}, {}
        spec = TaskSpec(Vx, 1, False, {"stick": stick_bins}, {"buttons":3}, {"stick":0.6, "buttons":0.4})
        return spec, DS()

    @staticmethod
    def task9_opponent_feints(L: int, p_feint: float, hist_n: int) -> Tuple[TaskSpec, Dataset]:
        Vx = 4  # tokens: W_A, W_B, RES_A, RES_B
        Vy = 2  # counters C_A, C_B (predict before resolve)
        WA, WB, RA, RB = 0,1,2,3
        class DS(Dataset):
            def __len__(self): return 8000
            def __getitem__(self, _):
                x = torch.full((L,), WA, dtype=torch.long)
                y = torch.zeros(L, dtype=torch.long)
                prob_A_feints = 0.3; prob_B_feints = p_feint
                recent_A, recent_B = 0,0
                for t in range(L):
                    if random.random()<0.5:
                        wind = WA; feint = random.random() < prob_A_feints
                        res = RB if feint else RA
                        x[t] = wind
                        # pick counter using last hist_n: if B feints often, counter opposite
                        pB = prob_B_feints
                        y[t] = 0 if res==RB else 1  # ideally counter move; we supervise the oracle
                    else:
                        wind = WB; feint = random.random() < prob_B_feints
                        res = RA if feint else RB
                        x[t] = wind
                        y[t] = 0 if res==RB else 1
                return x, y, {}, {}, {}
        spec = TaskSpec(Vx, Vy, True, {}, {}, {"dec_ce":1.0})
        return spec, DS()

    @staticmethod
    def task10_micro_plan(L: int, horizon: int) -> Tuple[TaskSpec, Dataset]:
        actions = 6  # idle, dashL, dashR, jump, attack, cancel
        Vx = 8      # coarse world token
        class DS(Dataset):
            def __len__(self): return 8000
            def __getitem__(self, _):
                world = torch.randint(0, Vx, (L,), generator=g)
                y = torch.zeros(L, dtype=torch.long)
                # simple plan: first get to edge (use dash), then attack, then cancel, then return center
                phase = 0
                for t in range(L):
                    if phase==0:
                        y[t] = random.choice([1,2])
                        if t>3: phase=1
                    elif phase==1:
                        y[t] = 4
                        phase=2
                    elif phase==2:
                        y[t] = 5
                        phase=3
                    else:
                        y[t] = random.choice([1,2,0])
                return world, y, {}, {}, {}
        spec = TaskSpec(Vx, actions, True, {}, {}, {"dec_ce":1.0})
        return spec, DS()

# map TASK_ID-> factory
TASK_FACTORIES = {
    1: lambda: ToyTasks.task1_shifted_copy(L, DELAY),
    2: lambda: ToyTasks.task2_running_mod(L, MOD_BASE),
    3: lambda: ToyTasks.task3_cooldown(L, COOLDOWN_E),
    4: lambda: ToyTasks.task4_jump_squat(L, JUMP_SQUAT_FRAMES),
    5: lambda: ToyTasks.task5_cancel_window(L, CANCEL_K),
    6: lambda: ToyTasks.task6_di_response(L, DELAY, DI_M_FRAMES),
    7: lambda: ToyTasks.task7_stage_positioning(L, STAGE_RANGE, OFFSTAGE_THRESH),
    8: lambda: ToyTasks.task8_multi_head_coherence(L),
    9: lambda: ToyTasks.task9_opponent_feints(L, FEINT_PROB, HISTORY_N),
    10: lambda: ToyTasks.task10_micro_plan(L, PLAN_HORIZON),
}

# =================================
# ==== DATA WRANGLING =============
# =================================
class UnifiedDataset(Dataset):
    def __init__(self, base: Dataset, spec: TaskSpec):
        self.base = base
        self.spec = spec

    def __len__(self): return len(self.base)

    def __getitem__(self, idx: int):
        x, dec_y, ce_targets, bce_targets, masks = self.base[idx]
        tgt_in, tgt_out = ToyTasks._bos_cat(dec_y.unsqueeze(0), self.spec.dec_vocab)
        return (
            x.long(),
            tgt_in.squeeze(0).long(),
            tgt_out.long(),
            {k: v.long() for k, v in ce_targets.items()},
            {k: v.float() for k, v in bce_targets.items()},
            {k: v for k, v in (masks or {}).items()},
        )

# =================================
# ==== HAZARD LOSS FOR TASK 3 =====
# =================================
def task3_hazard_loss(logits: Tensor, y: Tensor, cooldown_E: int) -> Tensor:
    # logits: (B, T, 2), y: (B, T) in {0,1}
    # Converts class-1 probability into a per-frame hazard and uses a renewal-process NLL
    # with refractory period `cooldown_E`. This penalizes early/late fires and multiple fires.
    probs = logits.softmax(dim=-1)[..., 1].clamp(1e-6, 1.0 - 1e-6)  # (B, T)
    B, T = y.shape
    total = probs.new_zeros(())
    for b in range(B):
        cd = 0
        for t in range(T):
            p = probs[b, t]
            yi = int(y[b, t].item())
            if cd > 0:
                total = total + (-torch.log(1.0 - p))  # survival during cooldown
                cd -= 1
                continue
            if yi == 1:
                total = total + (-torch.log(p))        # event at t
                cd = cooldown_E
            else:
                total = total + (-torch.log(1.0 - p))  # survival before next event
    return total / (B * T)

# =================================
# ==== TRAIN / EVAL ===============
# =================================
def train_eval_loop() -> None:
    spec, base_ds = TASK_FACTORIES[TASK_ID]()
    ds = UnifiedDataset(base_ds, spec)
    val_len = max(1000, len(ds)//8)
    train_len = len(ds) - val_len
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_len, val_len], generator=g)

    model = Model(spec.in_vocab, spec.dec_vocab, spec.extra_ce, spec.extra_bce).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    def step(loader, train: bool):
        if train: model.train()
        else: model.eval()
        tot_loss = 0.0; n_tok = 0
        accs: Dict[str, float] = {}
        with torch.set_grad_enabled(train):
            for x, tgt_in, tgt_out, ce_tgts, bce_tgts, masks in loader:
                x = x.to(DEVICE); tgt_in = tgt_in.to(DEVICE); tgt_out = tgt_out.to(DEVICE)
                ce_tgts = {k:v.to(DEVICE) for k,v in ce_tgts.items()}
                bce_tgts = {k:v.to(DEVICE) for k,v in bce_tgts.items()}

                out = model(x, tgt_in)
                loss = 0.0
                T = tgt_in.size(1)

                if spec.use_dec_ce and "dec_ce" in out:
                    B, T_, V = out["dec_ce"].shape
                    loss += spec.loss_weights.get("dec_ce",1.0) * ce_loss(out["dec_ce"].reshape(B*T_, V), tgt_out.reshape(B*T_))
                    pred = out["dec_ce"].argmax(-1)
                    accs["dec_ce"] = accs.get("dec_ce",0.0) + (pred.eq(tgt_out).float().sum().item())
                    n_tok += B*T_

                for k, logits in out.items():
                    if k=="dec_ce": continue
                    if k in spec.extra_ce:
                        B, T_, C = logits.shape
                        y = ce_tgts[k]
                        # Task-3 selectable loss for 'press' head
                        if TASK_ID == 3 and k == "press" and TASK3_USE_WEIGHTED_CE:
                            y_flat = y.reshape(B*T_)
                            logits_flat = logits.reshape(B*T_, C)
                            with torch.no_grad():
                                counts = torch.bincount(y_flat, minlength=C).to(logits.device)
                                neg = counts[0].clamp(min=1).float()
                                pos = counts[1].clamp(min=1).float()
                                total = (neg + pos)
                            if TASK3_LOSS_MODE == "hazard":
                                loss_k = task3_hazard_loss(logits, y, COOLDOWN_E)
                            elif TASK3_LOSS_MODE == "balanced_ce":
                                w0 = (total / (2.0 * neg)).item()
                                w1 = (total / (2.0 * pos)).item()
                                class_weight = torch.tensor([w0, w1], device=logits.device, dtype=torch.float32)
                                loss_k = F.cross_entropy(logits_flat, y_flat, weight=class_weight)
                            elif TASK3_LOSS_MODE == "focal":
                                probs = F.softmax(logits_flat, dim=-1)
                                p_t = probs.gather(1, y_flat.unsqueeze(1)).squeeze(1)
                                alpha = torch.where(
                                    y_flat == 1,
                                    torch.full_like(p_t, TASK3_FOCAL_ALPHA_POS),
                                    torch.full_like(p_t, 1.0 - TASK3_FOCAL_ALPHA_POS),
                                )
                                focal_factor = (1.0 - p_t).clamp(min=1e-6).pow(TASK3_FOCAL_GAMMA)
                                loss_k = -(alpha * focal_factor * p_t.clamp(min=1e-6).log()).mean()
                            else:  # "weighted_ce"
                                w1 = (neg / pos).clamp(min=TASK3_MIN_POS_WEIGHT)
                                class_weight = torch.tensor([1.0, float(w1.item())], device=logits.device, dtype=torch.float32)
                                loss_k = F.cross_entropy(logits_flat, y_flat, weight=class_weight)
                        else:
                            loss_k = ce_loss(logits.reshape(B*T_, C), y.reshape(B*T_))
                        loss += spec.loss_weights.get(k,1.0) * loss_k
                        pred = logits.argmax(-1)
                        accs[k] = accs.get(k,0.0) + (pred.eq(y).float().sum().item())
                        n_tok += B*T_
                    elif k in spec.extra_bce:
                        y = bce_tgts[k]
                        loss += spec.loss_weights.get(k,1.0) * bce_loss(logits, y)
                        # simple per-bit accuracy:
                        accs[k] = accs.get(k,0.0) + ((logits.sigmoid()>0.5).eq(y>0.5).float().mean(dim=(1,2)).sum().item())
                        n_tok += logits.size(0)  # per-sequence score

                if train:
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                tot_loss += float(loss.detach())

        # normalize metrics
        metrics = {k: (v / (len(loader) if k in spec.extra_bce else n_tok)) for k,v in accs.items()}
        return tot_loss/len(loader), metrics

    print(f"Task {TASK_ID} | in_vocab={spec.in_vocab} dec_vocab={spec.dec_vocab} headsCE={spec.extra_ce} headsBCE={spec.extra_bce}")
    for e in range(1, EPOCHS+1):
        tr_loss, tr_m = step(train_loader, True)
        va_loss, va_m = step(val_loader, False)
        if e==1 or e%5==0:
            def fmt(m): return " ".join([f"{k}:{(v*100):.1f}%" for k,v in m.items()])
            print(f"Epoch {e:03d} | train {tr_loss:.4f} [{fmt(tr_m)}] | val {va_loss:.4f} [{fmt(va_m)}]")

    # demos
    model.eval()
    print("\n=== Demo ===")
    demo_loader = DataLoader(val_ds, batch_size=1, shuffle=True)
    for i, batch in enumerate(demo_loader):
        if i>=DEMO_SAMPLES: break
        x, tgt_in, tgt_out, ce_tgts, bce_tgts, _ = batch
        x = x.to(DEVICE); tgt_in = tgt_in.to(DEVICE); tgt_out = tgt_out.to(DEVICE)
        out = model(x, tgt_in)
        print(f"src: {x.squeeze(0).cpu().tolist()}")
        if spec.use_dec_ce and "dec_ce" in out:
            pred = out["dec_ce"].argmax(-1).squeeze(0).cpu().tolist()
            print(f"y_true(dec): {tgt_out.squeeze(0).cpu().tolist()}")
            print(f"y_pred(dec): {pred}")
        for k in spec.extra_ce:
            y = ce_tgts[k].squeeze(0).cpu().tolist()
            p = out[k].argmax(-1).squeeze(0).cpu().tolist()
            print(f"{k}_true: {y}")
            print(f"{k}_pred: {p}")
        for k in spec.extra_bce:
            y = (bce_tgts[k].squeeze(0).cpu().numpy()>0.5).astype(int).tolist()
            p = (out[k].sigmoid().squeeze(0).cpu().numpy()>0.5).astype(int).tolist()
            print(f"{k}_true: {y}")
            print(f"{k}_pred: {p}")
        print("---")

# =================================
# ==== RUN ========================
# =================================
if __name__ == "__main__":
    train_eval_loop()