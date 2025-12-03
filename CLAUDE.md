# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nano-Melee trains a GPT-style transformer to play Super Smash Bros. Melee. The model predicts controller outputs from game state via:
1. **Imitation Learning** - Train on human replays from Slippi `.slp` files

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

# Statistics generation
python -m stats                               # compute all statistics
python -m stats --zarr-dir processed_data_100/ --max-episodes 100

# Hyperparameter sweeps
python sweep.py

# Analysis scripts
python scripts/analyze_feature_importance.py
python scripts/interpret.py
python scripts/benchmark_dataloader.py
```

## Architecture

### Data Pipeline
1. **Slippi `.slp` files** → parsed via `py-slippi` and `libmelee/`
2. **`zarr_storage.py`** → Zarr shards with `X` (features) and `Y` (targets) arrays
3. **`window_dataset.py`** → `WindowDataset` + `ZarrCorpusIndex` for sliding window access
4. **`feature_transforms.py`** → Input transforms (stick palette snapping, scaling)

### Model (`model/`)
GPT architecture split across modular files:
- `nano_gpt.py` - Main `GPT` class, `Block`, `MLP`
- `attention.py` - Causal self-attention with MQA support
- `output_head.py` - Simple output head implementation
- `head_cross_attention.py` - Cross-attention between output heads
- `positional_encoding.py` - Rotary positional embeddings
- `norm.py` - RMSNorm implementation
- `compile_utils.py` - torch.compile utilities

Architecture features:
- Rotary embeddings (no learned positional embeddings)
- QK norm, ReLU² MLP, no bias in linear layers
- Multi-Query Attention (configurable `n_kv_head`)
- One-hot encoding for categoricals (stage, character, action)
- 5 output heads: `main_stick`, `c_stick`, `buttons`, `shoulder`, `value`
- Optional cross-attention between heads for information sharing

### Controller Quantization (`controller_quantization.py`)
- **Main stick**: 64 discrete positions (wavedash angles, DI, Firefox angles, etc.)
- **C-stick**: 9 positions (cardinals + diagonals + neutral)
- **Shoulder**: 5 levels `[0.0, 0.31, 0.42, 0.55, 1.0]`
- **Buttons**: 5 binary outputs (A, B, X/Y, Z, L/R)

### Config System (`config/`)
Modular Pydantic configs split by domain:
- `config.py` - Main `Config` class aggregating all sub-configs
- `zarr_config.py` - `ZarrConfig` for data paths and processing
- `train_config.py` - `TrainConfig` for training hyperparameters
- `gpt_config.py` - `GPTConfig` for model architecture
- `loss_config.py` - `LossConfig` for loss function weights
- `feature_config.py` - `FeatureConfig` for feature engineering
- `rl_config.py` - `RLConfig` for reinforcement learning
- `imitation_config.py` - `ImitationConfig` for imitation learning

Features:
- CLI overrides: `--set section.key=value` (e.g., `--set train.lr=1e-4`)
- Auto-detects platform paths (Darwin/Linux) for data directories
- JSON serializable for checkpoint storage

### Training (`train/`)
Modular training utilities with clean separation of concerns:
- `loop.py` - Main training loop with epoch/batch iteration
- `step.py` - Forward/backward pass helpers (`perform_forward_pass`, `perform_backward_pass`)
- `batch_utils.py` - Batch processing (`build_model_inputs`, quantization)
- `checkpoint.py` - Save/load/prune checkpoints with versioning
- `metrics.py` - Accuracy, confusion matrices, PRF metrics, `MetricsAccumulator`
- `value_head.py` - Reward computation and value targets for RL
- `setup.py` - Initialize training components, parse CLI args, build optimizer
- `logging.py` - Logging bundle preparation and emission
- `validation.py` - Validation runs on checkpoints
- `display.py` - Terminal output formatting for metrics/losses
- `gradients.py` - Gradient diagnostics and clipping
- `lr_schedule.py` - Learning rate schedules (cosine)
- `wandb_utils.py` - Weights & Biases integration
- `components.py` - Shared training components

### Statistics (`stats/`)
Comprehensive data analysis module with modular collectors:
- `run.py` - Main orchestration, parallel processing, `COLLECTOR_REGISTRY`
- `__main__.py` - CLI entry point (`python -m stats`)
- `config.py` - Statistics configuration
- `index.py` - Zarr index loading
- `collectors/` - Pluggable stat collectors:
  - `column_stats.py` - Per-column statistics (mean, std, percentiles, run lengths)
  - `episode_stats.py` - Episode-level statistics (length, stage, outcome)
  - `action_states.py` - Action state analysis (categories, transitions)
  - `controller_inputs.py` - Controller input analysis (buttons, sticks)
  - `cross_feature.py` - Cross-feature correlations and joint distributions
  - `derived_metrics.py` - Derived metrics (distance, combos, advantage)
  - `temporal.py` - Temporal patterns (transitions, change rates)
  - `data_quality.py` - Data quality validation and scoring
- `output/` - Output formatting:
  - `json_writer.py` - JSON output formatting
  - `terminal.py` - Terminal progress display
- `utils/` - Helper utilities:
  - `formatting.py` - Number/data formatting
  - `melee_constants.py` - Melee-specific constants
  - `parallel.py` - Parallel processing helpers

### Scripts (`scripts/`)
Analysis and utilities:
- `analyze_feature_importance.py` - Feature importance analysis
- `interpret.py` - Model interpretation tools
- `interpret_multilayer.py` - Multi-layer model interpretation
- `benchmark_dataloader.py` - Dataloader performance testing
- `wrapper.py` - Utility wrappers

## Key Files

| File | Purpose |
|------|---------|
| `schema.py` | Feature/target specs (`PLAYER_SPEC`, `COMMON_SPEC`) |
| `column_map.py` | Maps feature names to tensor column indices |
| `loss.py` | Cross-entropy with class balancing, BCE for buttons |
| `model_interface.py` | High-level inference API for live play |
| `validation.py` | Model evaluation with detailed metrics |
| `sweep.py` | Hyperparameter grid search with FLOP estimation |
| `constants.py` | Shared constants used across the codebase |
| `controller_quantization.py` | Controller input quantization tables |
| `controller_utils.py` | Controller utilities and helper functions |
| `feature_transforms.py` | Input feature transforms (stick snapping, scaling) |
| `utils.py` | General utility functions |
| `data_types.py` | Type definitions and custom types |

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
