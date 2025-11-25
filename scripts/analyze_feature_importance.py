#!/usr/bin/env python
"""
One-off script to inspect per-input feature importance and ablation sensitivity.

The workflow mirrors the manual investigation performed in the CLI:

* Loads the latest model checkpoint (path hard-coded below).
* Samples a handful of windows from the processed dataset.
* Runs the standard feature transforms + `_embed_inputs`.
* Computes a magnitude-aware importance score for every input column:
    avg(|embedded value|) × sum(|projection_down weights|) for that column.
* Zeroes each non-one-hot feature (gamestate + controller) in turn and measures
  how much each output head changes relative to the baseline.

Outputs land in `analysis/feature_importance_results.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import zarr

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from column_map import ColumnMap  # noqa: E402
from config.config import Config  # noqa: E402
from feature_transforms import apply_feature_transforms  # noqa: E402
from libmelee.melee.enums import Action, Character  # noqa: E402
from model.nano_gpt import GPT  # noqa: E402
from train.batch_utils import build_model_inputs  # noqa: E402
from utils import strip_compiled_prefix  # noqa: E402


CHECKPOINT_PATH = Path("checkpoints/model_ep014_050001.pt")
DATA_ROOT = Path("processed_data_1000")
NUM_WINDOWS = 8
OUTPUT_PATH = Path("analysis/feature_importance_results.json")

# Disable MPS auto-promotion so everything stays on CPU when sampling tensors.
if getattr(torch.backends, "mps", None):
    torch.backends.mps.is_available = lambda: False  # type: ignore[assignment]


def _load_windows(
    *, seq_len: int, feature_names: List[str], limit: int
) -> torch.Tensor:
    """Load `limit` windows (each length `seq_len`) and apply transforms."""

    windows = []
    for shard_dir in sorted(DATA_ROOT.glob("shard_*.zarr")):
        for ep_dir in sorted(shard_dir.glob("ep_*")):
            grp = zarr.open_group(str(ep_dir), mode="r")
            X = np.asarray(grp["X"])
            if X.shape[0] < seq_len:
                continue
            window = np.ascontiguousarray(X[:seq_len, :])
            window = apply_feature_transforms(window, feature_names)
            windows.append(torch.from_numpy(window.astype(np.float32)))
            if len(windows) >= limit:
                return torch.stack(windows, dim=0)
    raise RuntimeError(
        f"Unable to load {limit} windows (only found {len(windows)} sequences)."
    )


def _build_label_catalog(
    *,
    config: Config,
    feature_names: List[str],
    colmap: ColumnMap,
) -> List[Dict[str, Optional[object]]]:
    """Create metadata for every combined input column."""

    catalog: List[Dict[str, Optional[object]]] = []
    num_stages = config.model.num_stages
    num_characters = config.model.num_characters
    num_actions = config.model.num_actions

    # Stage one-hot.
    for stage_idx in range(num_stages):
        catalog.append(
            {
                "label": f"stage_{stage_idx}",
                "group": "stage_one_hot",
                "dataset_col": None,
                "local_index": None,
            }
        )

    # Character one-hot (ego/opponent).
    char_names = []
    for value in range(num_characters):
        try:
            char_names.append(Character(value).name)
        except ValueError:
            char_names.append(f"Character_{value}")
    for prefix in ("ego_char", "opp_char"):
        for name in char_names:
            catalog.append(
                {
                    "label": f"{prefix}_{name}",
                    "group": f"{prefix}_one_hot",
                    "dataset_col": None,
                    "local_index": None,
                }
            )

    # Action one-hot (ego/opponent).
    action_names = []
    for value in range(num_actions):
        member = Action._value2member_map_.get(value)
        action_names.append(member.name if member else f"Action_{value}")
    for prefix in ("ego_action", "opp_action"):
        for name in action_names:
            catalog.append(
                {
                    "label": f"{prefix}_{name}",
                    "group": f"{prefix}_one_hot",
                    "dataset_col": None,
                    "local_index": None,
                }
            )

    # Gamestate block.
    for local_idx, feat_idx in enumerate(colmap.gamestate_idxs):
        catalog.append(
            {
                "label": feature_names[feat_idx],
                "group": "gamestate",
                "dataset_col": feat_idx,
                "local_index": local_idx,
            }
        )

    # Controller block.
    for local_idx, feat_idx in enumerate(colmap.controller_idxs):
        catalog.append(
            {
                "label": feature_names[feat_idx],
                "group": "controller",
                "dataset_col": feat_idx,
                "local_index": local_idx,
            }
        )

    return catalog


def _summarize_head_deltas(baseline, ablated) -> Dict[str, Dict[str, float]]:
    """Return absolute + relative differences per output head."""

    def _stats(key: str) -> Dict[str, float]:
        diff = (baseline[key] - ablated[key]).abs().mean().item()
        ref = baseline[key].abs().mean().item()
        rel = diff / (ref + 1e-8)
        return {"mean_abs_diff": diff, "relative_percent": rel * 100}

    return {
        "buttons": _stats("buttons"),
        "main_stick": _stats("main_stick"),
        "c_stick": _stats("c_stick"),
        "shoulder": _stats("shoulder"),
        "value": _stats("value"),
    }


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Dataset root not found: {DATA_ROOT}")

    state = torch.load(CHECKPOINT_PATH, map_location="cpu")
    model_state = strip_compiled_prefix(state["model"])

    config = Config()
    config.model.input_size = model_state["projection_down.weight"].shape[1]
    model = GPT(config)
    model.load_state_dict(model_state)
    model.eval()

    meta = json.loads((DATA_ROOT / "meta.json").read_text())
    seq_len = int(meta["build_config"]["seq_len"])
    feature_names = list(meta["schema"]["features"])
    target_names = list(meta["schema"]["targets"])
    colmap = ColumnMap(feature_names, target_names)

    batch = _load_windows(
        seq_len=seq_len,
        feature_names=feature_names,
        limit=NUM_WINDOWS,
    )
    inputs = build_model_inputs(batch, colmap)

    with torch.no_grad():
        embedded = model._embed_inputs(inputs)

    flat_inputs = embedded.reshape(-1, embedded.shape[-1])
    avg_abs_values = flat_inputs.abs().mean(dim=0)
    weight_abs_sum = model.projection_down.weight.abs().sum(dim=0)
    importance = (avg_abs_values * weight_abs_sum).tolist()

    catalog = _build_label_catalog(
        config=config, feature_names=feature_names, colmap=colmap
    )
    if len(catalog) != len(importance):
        raise RuntimeError(
            f"Label catalog mismatch: {len(catalog)} entries vs {len(importance)} columns."
        )

    with torch.no_grad():
        baseline_outputs = model(inputs)

    non_one_hot_start = len(catalog) - (
        len(colmap.gamestate_idxs) + len(colmap.controller_idxs)
    )
    feature_results: List[Dict[str, object]] = []

    for idx, meta_row in enumerate(catalog):
        row = {
            "input_index": idx,
            "label": meta_row["label"],
            "group": meta_row["group"],
            "dataset_col": meta_row["dataset_col"],
            "importance": importance[idx],
            "ablation": None,
        }

        if idx >= non_one_hot_start:
            inputs_ablated = inputs.clone()
            group = meta_row["group"]
            local_index = meta_row["local_index"]
            if local_index is None:
                raise RuntimeError(f"Missing local_index for feature {meta_row}")
            if group == "gamestate":
                inputs_ablated["gamestate"][..., local_index] = 0.0
            elif group == "controller":
                inputs_ablated["controller"][..., local_index] = 0.0
            else:
                raise RuntimeError(f"Unexpected group for ablation: {group}")

            with torch.no_grad():
                ablated_outputs = model(inputs_ablated)
            row["ablation"] = _summarize_head_deltas(baseline_outputs, ablated_outputs)

        feature_results.append(row)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(CHECKPOINT_PATH),
        "dataset_root": str(DATA_ROOT),
        "num_windows": NUM_WINDOWS,
        "notes": (
            "importance = avg(|embedded value|) * sum(|projection_down weights|); "
            "ablations zero individual non-one-hot features before the first transformer block."
        ),
        "features": feature_results,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Wrote feature analysis to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
