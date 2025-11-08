# Repository Guidelines

## Project Structure & Module Organization
Training entry points sit at the root: `train.py` covers supervised experiments while `train_ppo.py` plus `run_ppo.sh` drive self-play. Reinforcement logic lives in `ppo/` (opponent pool, trajectories, losses, environment), preprocessing helpers live in `transforms/`, `feature_transforms.py`, and `controller_*`, and emulator bindings sit in `libmelee/`. Artifacts land in `processed_data_*`, `validation_set/`, `checkpoints/`, and `model/`, while regression tests live in `test/`.

## Build, Test, and Development Commands
- Always run `source ~/.venvs/slippi312/bin/activate` before any Python command to ensure the shared environment is active.
- `uv venv && source .venv/bin/activate && uv pip install -r requirements.txt` — standard environment bootstrap.
- `pytest` or `pytest test/test_controller_quantization.py -k shoulder` — execute the suite or a focused regression.

## Coding Style & Naming Conventions
Stick to 4-space indentation, full type hints, and `snake_case` functions with `CamelCase` classes as shown in `config.PPOConfig` and `controller_quantization`. Keep modules narrowly scoped, place shared values in `constants.py`, and document non-obvious math with short comments. Pydantic models (`config.py`, `schema.py`) must stay JSON-serializable and enforce invariants via `field_validator` / `model_validator`. When adding tensors, state shapes/dtypes to avoid silent torch casting.

## Testing Guidelines
Pytest already adds the repo root to `PYTHONPATH`, so prefer absolute imports. Place new suites in `test_<feature>.py` files with descriptive `test_*` names and fixtures that reuse `test/test.slp` for controller traces. Run `pytest -q` before pushing, and add regression cases whenever you change quantization tables, window datasets, or PPO loss math.