# global_config.py
from __future__ import annotations

import argparse
import ast
import copy
import json
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, get_args, get_origin, Final

from zarr.codecs import BloscCodec, BloscShuffle

from preprocess import FOX_STICK_64, C_STICK_XY_CLUSTER_CENTERS_V0_1


@dataclass
class _FreezeGuard:
    _frozen: bool = field(default=False, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False) and name != "_frozen":
            raise AttributeError(f"Config is frozen; cannot modify '{name}'.")
        object.__setattr__(self, name, value)


@dataclass
class ZarrConfig(_FreezeGuard):
    input_root: str = '/Users/eppie/Downloads/ALL_REPLAYS/FOX_vs_FOX'
    out_root: str = '/Users/eppie/PycharmProjects/new-melee-ai/processed_data'
    shard_size: int = 100
    target_chunk_mb: float = 8.0
    compressor: BloscCodec = field(
        default_factory=lambda: BloscCodec(cname="zstd", clevel=7, shuffle=BloscShuffle.bitshuffle))
    seed: int = 42


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 0
    max_steps: Optional[int] = None  # cap total steps (useful for quick tests)
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # losses
    grad_clip: float = 1.0
    label_smoothing: float = 0.0

    # quantization / shoulder
    shoulder_centers: Optional[Sequence[float]] = field(default_factory=lambda: [0.0, 0.7, 1.0])

    # episode_linear sampler knobs
    episodes_per_epoch: Optional[int] = None  # per rank
    with_replacement_episodes: bool = False

    # random_windows sampler knobs
    replacement: bool = False
    num_samples: Optional[int] = None  # required if replacement=True

    # epoch sizing (per rank)
    windows_per_epoch: Optional[int] = None
    steps_per_epoch: Optional[int] = None  # if provided, overrides windows_per_epoch via steps * batch_size

    # checkpointing
    out_dir: str = "checkpoints"
    save_every_epochs: int = 1

    # column pruning (optional)
    feature_keep: Optional[Sequence[str]] = None
    target_keep: Optional[Sequence[str]] = None

@dataclass
class GPTConfig:
    block_size: int = 256
    n_embd: int = 512
    n_layer: int = 8
    n_head: int = 8
    dropout: float = 0.0
    bias: bool = True  # TODO: remove. True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    input_size: int = 130  # TODO: how is this calculated?
    num_stages: int = 7
    num_characters: int = 26
    num_actions: int = 396
    stage_embedding_dim: int = 4
    character_embedding_dim: int = 12
    action_embedding_dim: int = 32
    gamma: float = 0.999
    target_shapes_by_head: dict = field(default_factory=lambda: {
        "main_stick": (len(FOX_STICK_64),),
        "c_stick": (len(C_STICK_XY_CLUSTER_CENTERS_V0_1),),
        "buttons": (5,),
        "shoulder": (3,),
    })


@dataclass
class Config(_FreezeGuard):
    seq_len: int = 256

    zarr: ZarrConfig = field(default_factory=ZarrConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: GPTConfig = field(default_factory=GPTConfig)

    def freeze(self) -> None:
        _freeze_dataclass(self)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of the config (recursively)."""
        return _to_jsonable(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Config":
        return _dataclass_from_dict(cls, data)

    @classmethod
    def from_json(cls, s: str) -> "Config":
        return cls.from_dict(json.loads(s))


_GLOBAL_CFG: Optional[Config] = None
_GLOBAL_CFG_NAME: Final[str] = "GLOBAL_CONFIG"


def init_config(
        initial: Optional[Mapping[str, Any]] = None,
        cli_overrides: Optional[Mapping[str, str]] = None,
        *,
        freeze: bool = True,
) -> Config:
    global _GLOBAL_CFG
    cfg = Config.from_dict(initial or {})
    if cli_overrides:
        apply_overrides(cfg, cli_overrides)
    if freeze:
        cfg.freeze()
    _GLOBAL_CFG = cfg
    return cfg


def get_config() -> Config:
    if _GLOBAL_CFG is None:
        raise RuntimeError("Global config not initialized. Call init_config(...) early in your program.")
    return _GLOBAL_CFG


def reset_config_for_tests() -> None:
    global _GLOBAL_CFG
    _GLOBAL_CFG = None


def apply_overrides(cfg: Config, overrides: Mapping[str, str]) -> None:
    if getattr(cfg, "_frozen", False):
        raise AttributeError("Config is frozen; cannot apply overrides.")

    for dotted_key, raw in overrides.items():
        parts = dotted_key.split(".")
        parent, attr = _resolve_parent_and_attr(cfg, parts)
        if is_dataclass(parent):
            target_type = _dataclass_field_type(type(parent), attr)
            value = _coerce(raw, target_type)
            setattr(parent, attr, value)
        elif isinstance(parent, MutableMapping):
            parent[attr] = _coerce_best_effort(raw)
        else:
            raise TypeError(f"Cannot set '{dotted_key}'; parent is neither dataclass nor mapping.")


def parse_cli_overrides(argv: Sequence[str]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Minimal CLI:
      --config_json PATH   (optional) load initial config values from JSON
      --set KEY=VALUE      (repeatable) e.g. --set learning_rate=5e-4 --set optimizer.weight_decay=0.02
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config_json", type=str, default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ns, _ = p.parse_known_args(argv)

    initial: Dict[str, Any] = {}
    if ns.config_json:
        with open(ns.config_json, "r", encoding="utf-8") as f:
            initial = json.load(f)

    overrides: Dict[str, str] = {}
    for item in ns.set:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected KEY=VALUE.")
        k, v = item.split("=", 1)
        overrides[k.strip()] = v.strip()

    return initial, overrides


def save_config_json(path: Union[str, Path], cfg: Optional[Config] = None) -> Path:
    cfg = cfg or get_config()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cfg.to_json(), encoding="utf-8")
    return path


def load_config_json(path: Union[str, Path], *, freeze: bool = True) -> Config:
    s = Path(path).read_text(encoding="utf-8")
    cfg = Config.from_json(s)
    if freeze:
        cfg.freeze()
    return cfg


def _dataclass_from_dict(cls: type, data: Mapping[str, Any]) -> Any:
    """Recursively construct dataclass instance from dict, merging into defaults."""
    inst = cls()  # start from defaults
    for f in fields(cls):
        if f.name not in data:
            continue
        incoming = data[f.name]
        ftype = f.type
        base = getattr(inst, f.name)

        dc_cls = _unwrap_dataclass_type(ftype)
        if dc_cls and isinstance(incoming, Mapping):
            # nested dataclass
            nested = _dataclass_from_dict(dc_cls, incoming)
            setattr(inst, f.name, nested)
        elif isinstance(base, dict) and isinstance(incoming, Mapping):
            merged = copy.deepcopy(base)
            merged.update(incoming)
            setattr(inst, f.name, merged)
        else:
            setattr(inst, f.name, incoming)
    return inst


def _unwrap_dataclass_type(tp: Any) -> Optional[type]:
    """Return the dataclass type if tp is a dataclass or Optional[dataclass], else None."""
    if is_dataclass(tp):
        return tp  # type: ignore[return-value]
    origin = get_origin(tp)
    args = get_args(tp)
    if origin is Union and len(args) == 2 and type(None) in args:
        t = args[0] if args[1] is type(None) else args[1]
        return t if is_dataclass(t) else None
    return None


def _dataclass_field_type(dc_type: type, name: str) -> Any:
    for f in fields(dc_type):
        if f.name == name:
            return f.type
    raise KeyError(f"Unknown field '{name}' on {dc_type.__name__}")


def _resolve_parent_and_attr(root: Any, parts: Sequence[str]) -> Tuple[Any, str]:
    """Walk parts[:-1] and return (parent, final_attr_name)."""
    if not parts:
        raise ValueError("Empty override key")
    cur = root
    for p in parts[:-1]:
        if is_dataclass(cur):
            if not hasattr(cur, p):
                raise KeyError(f"Unknown field '{p}' in path: {'.'.join(parts)}")
            cur = getattr(cur, p)
        elif isinstance(cur, Mapping):
            cur = cur[p]
        else:
            raise TypeError(f"Cannot traverse into '{p}' on {type(cur).__name__}")
    return cur, parts[-1]


def _coerce(raw: str, target_type: Any) -> Any:
    """Coerce string to annotated target_type. Handles Optional[T], bool/int/float/str; else literal_eval fallback."""
    origin = get_origin(target_type)
    args = get_args(target_type)
    if origin is Union and len(args) == 2 and type(None) in args:
        t = args[0] if args[1] is type(None) else args[1]
        return None if raw.lower() in {"none", "null"} else _coerce(raw, t)
    if target_type in (str, int, float):
        return target_type(raw)
    if target_type is bool:
        return _parse_bool(raw)
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def _coerce_best_effort(raw: str) -> Any:
    try:
        return ast.literal_eval(raw)
    except Exception:
        pass
    for caster in (_parse_bool, int, float):
        try:
            return caster(raw)  # type: ignore[misc]
        except Exception:
            continue
    return raw


def _parse_bool(s: str) -> bool:
    s = s.lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean from '{s}'")


def _freeze_dataclass(dc: _FreezeGuard) -> None:
    """Recursively freeze a dataclass (convert containers to immutable, set _frozen=True everywhere)."""
    assert is_dataclass(dc)
    for f in fields(dc):
        if f.name.startswith("_"):
            continue
        val = getattr(dc, f.name)
        frozen_val = _deep_freeze_value(val)
        object.__setattr__(dc, f.name, frozen_val)
    object.__setattr__(dc, "_frozen", True)


def _deep_freeze_value(obj: Any) -> Any:
    if is_dataclass(obj) and isinstance(obj, _FreezeGuard):
        _freeze_dataclass(obj)
        return obj
    if isinstance(obj, dict):
        return MappingProxyType({k: _deep_freeze_value(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return tuple(_deep_freeze_value(v) for v in obj)
    if isinstance(obj, set):
        return frozenset(_deep_freeze_value(v) for v in obj)
    return obj


def _to_jsonable(obj: Any) -> Any:
    # Primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Enums -> their .value (e.g., BloscCname.zstd -> "zstd")
    if isinstance(obj, Enum):
        return obj.value

    # Dataclasses -> dict (skip private fields)
    if is_dataclass(obj):
        return {
            f.name: _to_jsonable(getattr(obj, f.name))
            for f in fields(obj)
            if not f.name.startswith("_")
        }

    # Mappings -> dict with stringified keys
    if isinstance(obj, Mapping):
        return {str(_to_jsonable(k)): _to_jsonable(v) for k, v in obj.items()}

    # Sequences & sets -> lists
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in obj]

    # Paths
    if isinstance(obj, Path):
        return str(obj)

    # Objects that know how to serialize themselves (Zarr v3 codecs, etc.)
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            return _to_jsonable(obj.to_dict())
        except Exception:
            pass  # fall through to other options

    # Numcodecs codecs (and others) often expose get_config()
    if hasattr(obj, "get_config") and callable(getattr(obj, "get_config")):
        try:
            return _to_jsonable(obj.get_config())
        except Exception:
            pass

    # Numpy: scalars -> Python scalars; arrays -> lists
    try:
        import numpy as np  # optional dependency
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    # PyTorch: represent dtypes/devices/sizes as strings/lists; tensors as lists
    try:
        import torch  # optional dependency
        if isinstance(obj, torch.dtype) or isinstance(obj, torch.device):
            return str(obj)
        if isinstance(obj, torch.Size):
            return list(obj)
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except Exception:
        pass

    # Last resort: string representation
    return repr(obj)
