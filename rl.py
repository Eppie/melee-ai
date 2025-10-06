from __future__ import annotations

import argparse
import json
import os
import signal
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Bernoulli, Categorical
from torch.nn.utils import clip_grad_norm_

from config import get_config, init_config
from gpt import GPTv7
from libmelee.reward import compute_reward
from libmelee.melee.console import Console
from libmelee.melee.controller import Controller
from libmelee.melee.enums import Character, ControllerType, Menu, Stage
from libmelee.melee.gamestate import GameState
from libmelee.melee.menuhelper import MenuHelper
from model_interface import (
    ControllerState,
    InferenceColumnMap,
    apply_model_outputs_to_game,
    collect_raw_inputs_from_gamestate,
    model_to_dolphin01,
    _DEFAULT_FEATURE_NAMES,
    _DEFAULT_TARGET_NAMES,
)
from preprocess import FOX_STICK_64, C_STICK_XY_CLUSTER_CENTERS_V0_1
from train import build_inputs_for_gptv7
from utils import AmpFP16, _resolve_device


@dataclass
class RLConfig:
    """Hyper-parameters driving PPO-style reinforcement learning."""

    total_updates: int = 500
    rollout_length: int = 2048
    minibatch_size: int = 256
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    value_clip: Optional[float] = 0.2
    max_grad_norm: float = 1.0
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    target_kl: Optional[float] = 0.015
    device: str = "auto"
    rollout_device: Optional[str] = None
    seq_len: Optional[int] = None
    warmup_frames: int = 128
    log_interval: int = 10
    save_every: Optional[int] = 50
    checkpoint_dir: str = "rl_checkpoints"


@dataclass
class MatchConfig:
    character: Character = Character.FOX
    stage: Stage = Stage.BATTLEFIELD
    bot_port: int = 1
    opponent_port: int = 2
    cpu_level: int = 9
    autostart: bool = True
    swag: bool = False
    opponent_character: Optional[Character] = None


class ObservationEncoder:
    """Maintains a rolling history of frames and exposes TensorDict inputs."""

    def __init__(
        self,
        feature_names: Sequence[str],
        target_names: Sequence[str],
        seq_len: int,
        device: torch.device,
    ) -> None:
        self.colmap = InferenceColumnMap(feature_names, target_names)
        self.seq_len = seq_len
        self.device = device
        self._buffer: Deque[torch.Tensor] = deque(maxlen=seq_len)
        self._frame_dim = len(feature_names)

    def reset(self) -> None:
        self._buffer.clear()

    def push(self, raw_inputs: Dict[str, float]) -> Optional[TensorDict]:
        frame = torch.zeros(self._frame_dim, dtype=torch.float32)
        for idx, name in enumerate(self.colmap.feat_names):
            frame[idx] = float(raw_inputs.get(name, 0.0))
        self._buffer.append(frame)
        if len(self._buffer) < self.seq_len:
            return None
        stacked = torch.stack(list(self._buffer), dim=0).unsqueeze(0).to(self.device)
        inputs = build_inputs_for_gptv7(stacked, self.colmap)
        return inputs


class ConsoleRolloutEnv:
    """Thin wrapper around libmelee Console to expose RL-friendly transitions."""

    def __init__(
        self,
        console: Console,
        controllers: Dict[int, Controller],
        feature_names: Sequence[str],
        target_names: Sequence[str],
        *,
        seq_len: int,
        device: torch.device,
        match: Optional[MatchConfig] = None,
        shoulder_centers: Optional[Sequence[float]] = None,
        warmup_frames: int = 128,
    ) -> None:
        self.console = console
        self.controllers = controllers
        self.match = match or MatchConfig()
        self.menu_helper = MenuHelper()
        self.encoder = ObservationEncoder(feature_names, target_names, seq_len, device)
        self.shoulder_centers = list(shoulder_centers) if shoulder_centers else None
        self.warmup_frames = warmup_frames
        self.device = device
        self._previous_state: Optional[GameState] = None

    def connect(self, *, iso_path: Optional[str] = None) -> None:
        if iso_path is not None:
            self.console.run(iso_path=iso_path)
        if not self.console.connect():
            raise RuntimeError("Failed to connect to the console.")
        for controller in self.controllers.values():
            if not controller.connect():
                raise RuntimeError(f"Failed to connect controller on port {controller.port}.")

    def _step_console(self) -> GameState:
        while True:
            gamestate = self.console.step()
            if gamestate is None:
                continue
            return gamestate

    def _handle_menus(self, gamestate: GameState) -> None:
        self.menu_helper.menu_helper_simple(
            gamestate,
            self.controllers[self.match.bot_port],
            self.match.character,
            self.match.stage,
            costume=0,
            autostart=self.match.autostart,
            swag=self.match.swag,
        )

    def reset(self) -> TensorDict:
        self.encoder.reset()
        self._previous_state = None
        neutral = ControllerState.neutral()
        for ctrl in self.controllers.values():
            apply_model_outputs_to_game(ctrl, neutral)

        filled = 0
        while True:
            gamestate = self._step_console()
            if gamestate.menu_state not in (Menu.IN_GAME, Menu.SUDDEN_DEATH):
                self._handle_menus(gamestate)
                continue

            raw_inputs = collect_raw_inputs_from_gamestate(
                gamestate,
                self.match.bot_port,
                self.match.opponent_port,
            )
            inputs = self.encoder.push(raw_inputs)
            self._previous_state = gamestate
            filled += 1
            if inputs is not None and filled >= max(1, self.warmup_frames):
                return inputs
            apply_model_outputs_to_game(self.controllers[self.match.bot_port], neutral)

    def step(
        self,
        action: ControllerState,
        *,
        opponent_action: Optional[ControllerState] = None,
    ) -> Tuple[Optional[TensorDict], float, bool, Dict[str, float]]:
        apply_model_outputs_to_game(self.controllers[self.match.bot_port], action)
        if opponent_action is not None and self.match.opponent_port in self.controllers:
            apply_model_outputs_to_game(self.controllers[self.match.opponent_port], opponent_action)
        gamestate = self._step_console()

        if gamestate.menu_state not in (Menu.IN_GAME, Menu.SUDDEN_DEATH):
            self._handle_menus(gamestate)
            self._previous_state = None
            return None, 0.0, True, {}

        breakdown = compute_reward(
            self._previous_state,
            gamestate,
            self.match.bot_port,
            self.match.opponent_port,
        )
        reward = breakdown.total
        info = breakdown.as_dict() if self._previous_state is not None else {}

        raw_inputs = collect_raw_inputs_from_gamestate(
            gamestate,
            self.match.bot_port,
            self.match.opponent_port,
        )
        inputs = self.encoder.push(raw_inputs)
        self._previous_state = gamestate
        if inputs is None:
            inputs = self.reset()
        return inputs, reward, False, info


def _stack_tensordicts(tds: List[TensorDict]) -> TensorDict:
    if not tds:
        raise ValueError("Cannot stack an empty list of TensorDicts.")
    keys = list(tds[0].keys())
    data: Dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [td.get(key) for td in tds]
        squeezed = [t.squeeze(0) if t.shape[0] == 1 else t for t in tensors]
        data[key] = torch.stack(squeezed, dim=0)
    sample = tds[0]
    sample_batch = sample.batch_size
    if len(sample_batch) < 1:
        raise ValueError("Expected sample TensorDict to have batch dimensions.")
    batch_size = (len(tds),) + sample_batch[1:]
    return TensorDict(data, batch_size=batch_size)


def _stack_actions(tds: List[TensorDict]) -> TensorDict:
    if not tds:
        raise ValueError("Cannot stack an empty list of action TensorDicts.")
    keys = list(tds[0].keys())
    data: Dict[str, torch.Tensor] = {}
    for key in keys:
        tensors = [td.get(key) for td in tds]
        squeezed = [t.squeeze(0) if t.dim() > 0 and t.shape[0] == 1 else t for t in tensors]
        data[key] = torch.stack(squeezed, dim=0)
    batch = (len(tds),)
    return TensorDict(data, batch_size=batch)


class RolloutBuffer:
    """Simple storage for on-policy rollouts."""

    def __init__(self) -> None:
        self.observations: List[TensorDict] = []
        self.actions: List[TensorDict] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.entropies: List[torch.Tensor] = []

        self._advantages: Optional[torch.Tensor] = None
        self._returns: Optional[torch.Tensor] = None

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.entropies.clear()
        self._advantages = None
        self._returns = None

    def __len__(self) -> int:
        return len(self.rewards)

    def add(
        self,
        observation: TensorDict,
        action: TensorDict,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        done: bool,
        entropy: torch.Tensor,
    ) -> None:
        self.observations.append(observation.to("cpu"))
        self.actions.append(action.to("cpu"))
        self.log_probs.append(log_prob.detach().to("cpu"))
        self.values.append(value.detach().to("cpu"))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.entropies.append(entropy.detach().to("cpu"))

    def compute_advantages(
        self,
        *,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        device: torch.device,
    ) -> None:
        if len(self) == 0:
            raise RuntimeError("Cannot compute advantages with an empty buffer.")
        rewards = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(self.dones, dtype=torch.float32, device=device)
        values = torch.stack(self.values).to(device).squeeze(-1)
        next_value = last_value.to(device).squeeze(-1)

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(1, device=device)
        for t in reversed(range(len(self))):
            mask = 1.0 - dones[t]
            next_val = next_value if t == len(self) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_val * mask - values[t]
            gae = delta + gamma * gae_lambda * mask * gae
            advantages[t] = gae
        returns = advantages + values

        self._advantages = advantages.detach().to("cpu")
        self._returns = returns.detach().to("cpu")

    @property
    def advantages(self) -> torch.Tensor:
        if self._advantages is None:
            raise RuntimeError("Advantages have not been computed yet.")
        return self._advantages

    @property
    def returns(self) -> torch.Tensor:
        if self._returns is None:
            raise RuntimeError("Returns have not been computed yet.")
        return self._returns

    def iter_minibatches(
        self,
        batch_size: int,
        *,
        shuffle: bool,
        device: torch.device,
    ) -> Iterator[Tuple[TensorDict, TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        num_steps = len(self)
        if batch_size <= 0 or batch_size > num_steps:
            batch_size = num_steps
        indices = torch.randperm(num_steps) if shuffle else torch.arange(num_steps)
        advantages = self.advantages
        returns = self.returns
        log_probs = torch.stack(self.log_probs).squeeze(-1)
        values = torch.stack(self.values).squeeze(-1)

        for start in range(0, num_steps, batch_size):
            idx = indices[start : start + batch_size]
            obs_batch = _stack_tensordicts([self.observations[i] for i in idx.tolist()]).to(device)
            act_batch = _stack_actions([self.actions[i] for i in idx.tolist()]).to(device)
            yield (
                obs_batch,
                act_batch,
                log_probs[idx].to(device),
                values[idx].to(device),
                advantages[idx].to(device),
                returns[idx].to(device),
            )


def _policy_last_step(policy: TensorDict) -> Dict[str, torch.Tensor]:
    data: Dict[str, torch.Tensor] = {}
    for key in policy.keys():
        tensor = policy.get(key)
        if tensor.dim() >= 2:
            data[key] = tensor[:, -1, ...]
        else:
            data[key] = tensor
    return data


def sample_actions(policy: TensorDict) -> Tuple[TensorDict, torch.Tensor, torch.Tensor]:
    last = _policy_last_step(policy)
    device = next(iter(last.values())).device
    batch = next(iter(last.values())).shape[0]
    log_prob = torch.zeros(batch, device=device)
    entropy = torch.zeros(batch, device=device)
    actions: Dict[str, torch.Tensor] = {}

    if "buttons_logits" in last:
        btn_logits = last["buttons_logits"]
        btn_dist = Bernoulli(logits=btn_logits)
        btn_sample = btn_dist.sample()
        log_prob = log_prob + btn_dist.log_prob(btn_sample).sum(dim=-1)
        entropy = entropy + btn_dist.entropy().sum(dim=-1)
        actions["buttons"] = btn_sample

    if "main_stick_logits" in last:
        main_logits = last["main_stick_logits"]
        main_dist = Categorical(logits=main_logits)
        main_sample = main_dist.sample()
        log_prob = log_prob + main_dist.log_prob(main_sample)
        entropy = entropy + main_dist.entropy()
        actions["main_stick"] = main_sample

    if "c_stick_logits" in last:
        c_logits = last["c_stick_logits"]
        c_dist = Categorical(logits=c_logits)
        c_sample = c_dist.sample()
        log_prob = log_prob + c_dist.log_prob(c_sample)
        entropy = entropy + c_dist.entropy()
        actions["c_stick"] = c_sample

    if "shoulder_logits" in last:
        s_logits = last["shoulder_logits"]
        s_dist = Categorical(logits=s_logits)
        s_sample = s_dist.sample()
        log_prob = log_prob + s_dist.log_prob(s_sample)
        entropy = entropy + s_dist.entropy()
        actions["shoulder"] = s_sample

    actions_td = TensorDict(actions, batch_size=torch.Size([batch]))
    return actions_td, log_prob, entropy


def evaluate_actions(policy: TensorDict, actions: TensorDict) -> Tuple[torch.Tensor, torch.Tensor]:
    last = _policy_last_step(policy)
    device = next(iter(last.values())).device
    batch = next(iter(last.values())).shape[0]
    log_prob = torch.zeros(batch, device=device)
    entropy = torch.zeros(batch, device=device)

    if "buttons_logits" in last and "buttons" in actions.keys():
        logits = last["buttons_logits"]
        dist = Bernoulli(logits=logits)
        act = actions.get("buttons").to(device)
        log_prob = log_prob + dist.log_prob(act).sum(dim=-1)
        entropy = entropy + dist.entropy().sum(dim=-1)

    if "main_stick_logits" in last and "main_stick" in actions.keys():
        logits = last["main_stick_logits"]
        dist = Categorical(logits=logits)
        act = actions.get("main_stick").to(device)
        log_prob = log_prob + dist.log_prob(act)
        entropy = entropy + dist.entropy()

    if "c_stick_logits" in last and "c_stick" in actions.keys():
        logits = last["c_stick_logits"]
        dist = Categorical(logits=logits)
        act = actions.get("c_stick").to(device)
        log_prob = log_prob + dist.log_prob(act)
        entropy = entropy + dist.entropy()

    if "shoulder_logits" in last and "shoulder" in actions.keys():
        logits = last["shoulder_logits"]
        dist = Categorical(logits=logits)
        act = actions.get("shoulder").to(device)
        log_prob = log_prob + dist.log_prob(act)
        entropy = entropy + dist.entropy()

    return log_prob, entropy


def actions_to_controller_state(
    actions: TensorDict,
    *,
    shoulder_centers: Optional[Sequence[float]] = None,
) -> ControllerState:
    def _buttons() -> Tuple[bool, bool, bool, bool, bool]:
        if "buttons" not in actions.keys():
            return (False, False, False, False, False)
        btn = actions.get("buttons").float()
        if btn.dim() > 1:
            btn = btn[0]
        btn = btn.cpu().numpy().tolist()
        while len(btn) < 5:
            btn.append(0.0)
        return tuple(bool(round(v)) for v in btn[:5])

    def _palette_sample(key: str, palette: Sequence[Sequence[float]], neutral: Tuple[float, float]) -> Tuple[float, float]:
        if key not in actions.keys():
            return neutral
        idx = actions.get(key)
        if idx.dim() > 0:
            idx = idx[0]
        index = int(idx.item())
        coords01 = model_to_dolphin01(
            [index],
            palette11=palette,
        )
        arr = coords01.reshape(-1)
        if arr.size != 2:
            raise RuntimeError("Palette conversion produced unexpected shape.")
        return float(arr[0]), float(arr[1])

    btn_a, btn_b, btn_xy, btn_z, btn_lr = _buttons()
    main_x, main_y = _palette_sample("main_stick", FOX_STICK_64, (0.5, 0.5))
    c_x, c_y = _palette_sample(
        "c_stick",
        C_STICK_XY_CLUSTER_CENTERS_V0_1.tolist(),
        (0.5, 0.5),
    )

    shoulder_value = 0.0
    if shoulder_centers and "shoulder" in actions.keys():
        idx = actions.get("shoulder")
        if idx.dim() > 0:
            idx = idx[0]
        shoulder_index = int(idx.item())
        shoulder_index = max(0, min(shoulder_index, len(shoulder_centers) - 1))
        shoulder_value = float(shoulder_centers[shoulder_index])

    return ControllerState(
        main_stick_x=main_x,
        main_stick_y=main_y,
        c_stick_x=c_x,
        c_stick_y=c_y,
        shoulder_analog=shoulder_value,
        button_a=btn_a,
        button_b=btn_b,
        button_xy=btn_xy,
        button_lr=btn_lr,
        button_z=btn_z,
    )


def _normalize(t: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (t - t.mean()) / (t.std() + eps)


def ppo_update(
    model: GPTv7,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    cfg: RLConfig,
    *,
    device: torch.device,
) -> Dict[str, float]:
    metrics = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "num_minibatches": 0,
    }

    last_kl = 0.0
    for epoch in range(cfg.ppo_epochs):
        minibatches = buffer.iter_minibatches(
            cfg.minibatch_size,
            shuffle=True,
            device=device,
        )
        for obs, actions, old_logp, old_values, advantages, returns in minibatches:
            advantages = _normalize(advantages)

            outputs: TensorDict = model(obs, return_rl_outputs=True)
            policy_td = outputs.get("policy")
            value_td = outputs.get("value")
            value_pred = value_td[:, -1]

            logp, entropy = evaluate_actions(policy_td, actions)
            ratio = torch.exp(logp - old_logp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            if cfg.value_clip is not None:
                value_clipped = old_values + (value_pred - old_values).clamp(
                    -cfg.value_clip, cfg.value_clip
                )
                value_loss_unclipped = (value_pred - returns) ** 2
                value_loss_clipped = (value_clipped - returns) ** 2
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = 0.5 * F.mse_loss(value_pred, returns)

            entropy_loss = entropy.mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            with AmpFP16(device) as amp:
                amp.backward(loss)
                clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                amp.step(optimizer)

            last_kl = torch.mean(old_logp - logp).abs().item()
            metrics["policy_loss"] += float(policy_loss.item())
            metrics["value_loss"] += float(value_loss.item())
            metrics["entropy"] += float(entropy_loss.item())
            metrics["approx_kl"] += last_kl
            metrics["num_minibatches"] += 1

        if cfg.target_kl is not None and last_kl > cfg.target_kl:
            break

    if metrics["num_minibatches"] > 0:
        inv = 1.0 / metrics["num_minibatches"]
        metrics = {k: (v * inv if k != "num_minibatches" else v) for k, v in metrics.items()}
    return metrics


def _save_checkpoint(
    model: GPTv7,
    optimizer: torch.optim.Optimizer,
    *,
    update: int,
    config: RLConfig,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"update_{update:05d}.pt"
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rl_config": asdict(config),
        "update": update,
    }
    torch.save(payload, path)
    return path


def train_rl(
    model: GPTv7,
    env: ConsoleRolloutEnv,
    cfg: RLConfig,
    *,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> None:
    device = _resolve_device(cfg.device)
    rollout_device = _resolve_device(cfg.rollout_device or cfg.device)
    model = model.to(device)
    model.train()

    opt = optimizer or torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    buffer = RolloutBuffer()
    observation = env.reset().to(rollout_device)
    global_step = 0
    start_time = time.time()

    for update in range(1, cfg.total_updates + 1):
        buffer.clear()
        for _ in range(cfg.rollout_length):
            obs_model = observation.to(device)
            with torch.no_grad():
                outputs = model(obs_model, return_rl_outputs=True)
                value = outputs.get("value")[:, -1]
            policy = outputs.get("policy")

            actions, log_prob, entropy = sample_actions(policy)
            ctrl_state = actions_to_controller_state(
                actions,
                shoulder_centers=env.shoulder_centers,
            )
            opp_ctrl_state: Optional[ControllerState] = None
            if env.match.opponent_port in env.controllers:
                opponent_actions, _, _ = sample_actions(policy)
                opp_ctrl_state = actions_to_controller_state(
                    opponent_actions,
                    shoulder_centers=env.shoulder_centers,
                )
            next_obs, reward, done, info = env.step(
                ctrl_state,
                opponent_action=opp_ctrl_state,
            )

            buffer.add(
                observation.clone().to("cpu"),
                actions.to("cpu"),
                log_prob.detach().to("cpu"),
                value.detach().to("cpu"),
                reward,
                done,
                entropy.detach().to("cpu"),
            )

            observation = next_obs if next_obs is not None else env.reset()
            global_step += 1
            if done:
                observation = env.reset()

        with torch.no_grad():
            bootstrap_td = observation.to(device)
            bootstrap_value = model(bootstrap_td, return_rl_outputs=True).get("value")[:, -1]

        buffer.compute_advantages(
            last_value=bootstrap_value,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            device=device,
        )

        metrics = ppo_update(model, opt, buffer, cfg, device=device)

        if update % cfg.log_interval == 0 or update == 1:
            elapsed = time.time() - start_time
            print(
                f"[update {update:04d}] loss_pi={metrics['policy_loss']:.4f} "
                f"loss_v={metrics['value_loss']:.4f} entropy={metrics['entropy']:.4f} "
                f"kl={metrics['approx_kl']:.5f} steps={global_step} elapsed={elapsed:.1f}s"
            )

        if cfg.save_every and update % cfg.save_every == 0:
            out_path = _save_checkpoint(
                model,
                opt,
                update=update,
                config=cfg,
                out_dir=Path(cfg.checkpoint_dir),
            )
            print(f"Saved checkpoint to {out_path}")


def _load_schema(meta_path: Optional[Path]) -> Tuple[List[str], List[str], Optional[int]]:
    if meta_path is None:
        return list(_DEFAULT_FEATURE_NAMES), list(_DEFAULT_TARGET_NAMES), None
    path = Path(meta_path)
    if not path.exists():
        print(f"Warning: meta file '{path}' not found; falling back to built-in schema.")
        return list(_DEFAULT_FEATURE_NAMES), list(_DEFAULT_TARGET_NAMES), None
    with path.open("r") as f:
        data = json.load(f)
    schema = data.get("schema", {})
    features = schema.get("features", list(_DEFAULT_FEATURE_NAMES))
    targets = schema.get("targets", list(_DEFAULT_TARGET_NAMES))
    seq_len = data.get("seq_len")
    return list(features), list(targets), int(seq_len) if seq_len is not None else None


def _parse_character(value: str) -> Character:
    try:
        return Character[value.upper()]
    except KeyError as exc:
        valid = ", ".join(sorted([c.name.lower() for c in Character]))
        raise argparse.ArgumentTypeError(f"Unknown character '{value}'. Valid options: {valid}") from exc


def _parse_stage(value: str) -> Stage:
    try:
        return Stage[value.upper()]
    except KeyError as exc:
        valid = ", ".join(sorted([s.name.lower() for s in Stage]))
        raise argparse.ArgumentTypeError(f"Unknown stage '{value}'. Valid options: {valid}") from exc


def _build_rl_config(args: argparse.Namespace, *, default: RLConfig) -> RLConfig:
    cfg = RLConfig(
        total_updates=args.total_updates,
        rollout_length=args.rollout_length,
        minibatch_size=args.minibatch_size,
        ppo_epochs=args.ppo_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_coef=args.clip_coef,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        value_clip=args.value_clip,
        max_grad_norm=args.max_grad_norm,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        target_kl=args.target_kl,
        device=args.device,
        rollout_device=args.rollout_device,
        seq_len=args.seq_len,
        warmup_frames=args.warmup_frames,
        log_interval=args.log_interval,
        save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir,
    )
    return cfg


def _setup_signal_handlers(console: Console, controllers: Dict[int, Controller]) -> None:
    def _cleanup_and_exit(signum, frame):  # type: ignore[unused-argument]
        print(f"Received signal {signum}; stopping console and controllers.")
        for ctrl in controllers.values():
            try:
                ctrl.disconnect()
            except Exception:
                pass
        try:
            console.stop()
        except Exception:
            pass
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _cleanup_and_exit)
    signal.signal(signal.SIGTERM, _cleanup_and_exit)


def _load_initial_checkpoint(model: GPTv7, path: Optional[Path]) -> None:
    if path is None:
        return
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Checkpoint at '{path}' is malformed; expected dict with model state.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing parameters when loading checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected parameters ignored from checkpoint: {unexpected}")


def parse_args() -> argparse.Namespace:
    defaults = RLConfig()
    parser = argparse.ArgumentParser(description="Self-play PPO training for the GPT controller.")
    parser.add_argument("--dolphin-path", type=str, default=None, help="Path to the Dolphin executable directory.")
    parser.add_argument("--iso", type=str, default=None, help="Path to the Melee ISO.")
    parser.add_argument("--address", type=str, default="127.0.0.1", help="Address for the Slippi console connection.")
    parser.add_argument("--meta-json", type=Path, default=None, help="Optional meta.json describing feature schema.")
    parser.add_argument("--device", type=str, default=defaults.device, help="Training device (cuda/mps/cpu/auto).")
    parser.add_argument("--rollout-device", type=str, default=None, help="Device for environment rollouts (defaults to training device).")
    parser.add_argument("--total-updates", type=int, default=defaults.total_updates)
    parser.add_argument("--rollout-length", type=int, default=defaults.rollout_length)
    parser.add_argument("--minibatch-size", type=int, default=defaults.minibatch_size)
    parser.add_argument("--ppo-epochs", type=int, default=defaults.ppo_epochs)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--gae-lambda", type=float, default=defaults.gae_lambda)
    parser.add_argument("--clip-coef", type=float, default=defaults.clip_coef)
    parser.add_argument("--entropy-coef", type=float, default=defaults.entropy_coef)
    parser.add_argument("--value-coef", type=float, default=defaults.value_coef)
    parser.add_argument("--value-clip", type=float, default=defaults.value_clip)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--target-kl", type=float, default=defaults.target_kl)
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length for the transformer context window.")
    parser.add_argument("--warmup-frames", type=int, default=defaults.warmup_frames)
    parser.add_argument("--log-interval", type=int, default=defaults.log_interval)
    parser.add_argument("--save-every", type=int, default=defaults.save_every)
    parser.add_argument("--checkpoint-dir", type=str, default=defaults.checkpoint_dir)
    parser.add_argument("--init-checkpoint", type=Path, default=None, help="Optional supervised checkpoint to bootstrap from.")
    parser.add_argument("--character", type=_parse_character, default=Character.FOX, help="Bot character (e.g., fox, falco).")
    parser.add_argument("--opponent-character", type=_parse_character, default=None, help="Opponent character (defaults to same as bot).")
    parser.add_argument("--stage", type=_parse_stage, default=Stage.BATTLEFIELD, help="Stage selection (e.g., battlefield).")
    parser.add_argument("--bot-port", type=int, default=1)
    parser.add_argument("--opponent-port", type=int, default=2)
    parser.add_argument("--no-autostart", action="store_true", help="Disable automatic match start in menus.")
    parser.add_argument("--swag", action="store_true", help="Enable extra menu flair via MenuHelper.")
    parser.add_argument("--cpu-level", type=int, default=0, help="CPU level for menu helper (unused in self-play but kept for parity).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    overrides = {"model.enable_rl_heads": "True"}
    init_config(cli_overrides=overrides)
    global_cfg = get_config()

    rl_cfg = _build_rl_config(args, default=RLConfig())

    feature_names, target_names, meta_seq_len = _load_schema(args.meta_json)
    seq_len = rl_cfg.seq_len or meta_seq_len or global_cfg.seq_len
    rl_cfg.seq_len = seq_len

    rollout_device = _resolve_device(rl_cfg.rollout_device or rl_cfg.device)

    console = Console(
        path=args.dolphin_path,
        slippi_address=args.address,
        save_replays=False,
        copy_home_directory=False,
        tmp_home_directory=False,
        blocking_input=True,
    )
    controllers: Dict[int, Controller] = {
        args.bot_port: Controller(console=console, port=args.bot_port, type=ControllerType.STANDARD),
        args.opponent_port: Controller(console=console, port=args.opponent_port, type=ControllerType.STANDARD),
    }

    _setup_signal_handlers(console, controllers)

    match_cfg = MatchConfig(
        character=args.character,
        stage=args.stage,
        bot_port=args.bot_port,
        opponent_port=args.opponent_port,
        autostart=not args.no_autostart,
        swag=args.swag,
        opponent_character=args.opponent_character or args.character,
    )

    env = ConsoleRolloutEnv(
        console,
        controllers,
        feature_names,
        target_names,
        seq_len=seq_len,
        device=rollout_device,
        match=match_cfg,
        shoulder_centers=global_cfg.train.shoulder_centers,
        warmup_frames=rl_cfg.warmup_frames,
    )

    print("Connecting to console and controllers...")
    env.connect(iso_path=args.iso)

    model = GPTv7()
    _load_initial_checkpoint(model, args.init_checkpoint)

    try:
        train_rl(model, env, rl_cfg)
    finally:
        for ctrl in controllers.values():
            try:
                ctrl.disconnect()
            except Exception:
                pass
        try:
            console.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
