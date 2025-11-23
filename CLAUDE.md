# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nano-Melee trains a GPT-style transformer to play Super Smash Bros. Melee. The model predicts controller outputs from game state via:
1. **Imitation Learning** - Train on human replays from Slippi `.slp` files
2. **Reinforcement Learning (PPO)** - Self-play with opponent pool (largely untested)

## Commands

**Always activate the shared venv before running any Python command:**
```bash
source ~/.venvs/slippi312/bin/activate
```

```bash
# Tests
pytest                                                    # full suite
pytest test/test_controller_quantization.py -k shoulder   # focused

# Data preprocessing (Slippi .slp → Zarr)
python zarr_storage.py

# Imitation learning
python train.py
python train.py --set train.lr=1e-4 --set model.n_layer=8  # CLI overrides

# Validation
python validation.py                          # uses latest checkpoint
python validation.py --checkpoint path/to/model.pt

# PPO self-play (edit paths in script first)
./run_ppo.sh
python train_ppo.py --dolphin-path /path/to/dolphin-emu --iso /path/to/melee.iso

# Hyperparameter sweeps
python sweep.py
```

## Architecture

### Data Pipeline
1. **Slippi `.slp` files** → parsed via `py-slippi` and `libmelee/`
2. **`zarr_storage.py`** → Zarr shards with `X` (features) and `Y` (targets) arrays
3. **`window_dataset.py`** → `WindowDataset` + `ZarrCorpusIndex` for sliding window access
4. **`feature_transforms.py`** → Input transforms (stick palette snapping, scaling)

### Model (`model/nano_gpt.py`)
GPT architecture with:
- Rotary embeddings (no learned positional embeddings)
- QK norm, ReLU² MLP, no bias in linear layers
- Multi-Query Attention (configurable `n_kv_head`)
- One-hot encoding for categoricals (stage, character, action)
- 5 output heads: `main_stick`, `c_stick`, `buttons`, `shoulder`, `value`

### Controller Quantization (`controller_utils.py`)
- **Main stick**: 64 discrete positions (wavedash angles, DI, Firefox angles, etc.)
- **C-stick**: 9 positions (cardinals + diagonals + neutral)
- **Shoulder**: 5 levels `[0.0, 0.31, 0.42, 0.55, 1.0]`
- **Buttons**: 5 binary outputs (A, B, X/Y, Z, L/R)

### Config System (`config.py`)
Pydantic models: `ZarrConfig`, `TrainConfig`, `GPTConfig`, `LossConfig`, `RLConfig`, `PPOConfig`, `FeatureConfig`
- CLI overrides: `--set section.key=value` (e.g., `--set train.lr=1e-4`)
- Auto-detects platform paths (Darwin/Linux) for data directories

### Training (`train/`)
- `loop.py` - Main training loop with epoch/batch iteration
- `step.py` - Forward/backward pass helpers
- `batch_utils.py` - `build_model_inputs()`, `quantize_controller_targets()`
- `checkpoint.py` - Save/load/prune checkpoints
- `metrics.py` - Accuracy, confusion matrices, PRF metrics
- `value_head.py` - Reward computation and value targets for RL

### PPO Self-Play (`ppo/`)
- `opponent_pool.py` - FIFO pool of frozen past models
- `trajectory.py` - Experience buffer + GAE advantage estimation
- `ppo_loss.py` - Clipped surrogate + value + entropy loss
- `selfplay_env.py` - libmelee environment wrapper

## Key Files

| File | Purpose |
|------|---------|
| `schema.py` | Feature/target specs (`PLAYER_SPEC`, `COMMON_SPEC`) |
| `column_map.py` | Maps feature names to tensor column indices |
| `loss.py` | Cross-entropy with class balancing, BCE for buttons |
| `model_interface.py` | High-level inference API for live play |
| `validation.py` | Model evaluation with detailed metrics |
| `sweep.py` | Hyperparameter grid search with FLOP estimation |

## Coding Conventions

- 4-space indent, full type hints, `snake_case` functions, `CamelCase` classes
- Pydantic models: JSON-serializable, use `field_validator`/`model_validator`
- State tensor shapes/dtypes explicitly to avoid silent torch casting
- Shared constants go in `constants.py`
- Absolute imports (`pytest.ini` sets `pythonpath = .`)

## Testing

- Tests in `test/test_*.py`, use `test/test.slp` for replay fixtures
- Run `pytest -q` before pushing
- Add regression tests when changing quantization tables or loss math
