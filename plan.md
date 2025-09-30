# Refactoring Plan for Melee AI

This plan incorporates findings from code audit (radon complexity scan, file inspection) and lays out concrete, incremental tasks to align the project with core software engineering principles: unified configuration, isolation & SRP, lower complexity, DRY, observability, and long-term maintainability.

---
## 0. Current State Summary

* **Entry points** – Scripts such as `process_replays.py`, `preprocess.py`, `train.py`, and `to_parquet.py` embed configuration (file paths, hyperparameters, constants) directly in code. `TrainConfig` exists but is not shared outside `train.py`.
* **Data pipeline** – Replay preprocessing (`preprocess.py` + `process_replays.py`) feeds structured rows (see `schema.Row`) into parquet/zarr writers. There is duplicated mapping logic between preprocessing and schema definitions.
* **Dataset loading** – `window_dataset.py` wraps Zarr shards with samplers. Complexity hotspots (e.g., `RandomWindowSampler.__iter__`, `EpisodeThenLinearSampler._episode_order_for_rank`) have cyclomatic complexity ≥9 (per `radon`).
* **Modeling** – `train.py` contains a 39-branch `train_loop` function orchestrating device setup, logging, gradient steps, evaluation metrics, and checkpointing.
* **Observability** – Logging is inconsistent (e.g., `process_replays.py` uses `loguru`, other modules rely on `print`). Error handling often swallows context.
* **Tests** – Minimal automated tests; no shared fixtures or integration harness.

---
## 1. Establish a Single Source of Truth for Configuration

### 1.1 Inventory and normalize configuration needs
- Catalogue config knobs already surfaced in code:
  * Data roots (`dataset_FOX_vs_FOX/`, replay directories) in `process_replays.py`, `window_dataset.py`, `train.py`.
  * Training hyperparameters stored in `TrainConfig` (batch size, lr, warmup, quantization palettes).
  * Preprocessing constants (`FOX_STICK_64`, `C_STICK_XY_CLUSTER_CENTERS_V0_1`, `MAX_FRAMES`) in `preprocess.py`.
  * Logging and output paths (checkpoint dir, parquet destination).
- Convert this inventory into a structured schema grouped by domain: `data`, `preprocessing`, `model`, `training`, `logging`, `runtime`.

### 1.2 Implement configuration package
- Add `melee_ai/config/__init__.py` that exposes:
  * `Settings` dataclass models (e.g., via `pydantic` or `dataclasses`) backed by `default.yaml`.
  * Load precedence: defaults → environment overrides (prefixed) → CLI overrides.
  * Validation hooks (e.g., ensure dataset root exists, `steps_per_epoch` not set alongside `windows_per_epoch`).
- Provide helpers to derive frequently computed values (e.g., `total_steps = epochs * steps_per_epoch`).

### 1.3 Adopt config in entry points
- Wrap `process_replays.py` and `train.py` as Click/Typer CLIs that accept `--config` path and apply overrides.
- Replace hard-coded paths (e.g., `process_one_replay("...test.slp")`) with `settings.data.replay_glob`.
- Expose configuration values needed in other modules through dependency injection instead of module-level imports.

### 1.4 Sync shared constants
- Relocate palette arrays (`FOX_STICK_64`, `C_STICK_XY_CLUSTER_CENTERS_V0_1`, `STICK_XY_CLUSTER_CENTERS_V2`) into `config/constants.py` to avoid cross-module duplication.
- Update `controller_quantization.py`, `train.py`, and `preprocess.py` to consume the centralized constants.

---
## 2. Clarify Architecture & Boundaries (Isolation, SRP)

### 2.1 Define module responsibilities
- Create a top-level `docs/architecture.md` diagramming data flow: replay ingestion → preprocessing → storage → dataset windows → training loop.
- Introduce packages:
  * `melee_ai/preprocessing/` – frame/action processing utilities and Replay extractors.
  * `melee_ai/data/` – schema definitions, parquet/zarr writers, dataset samplers.
  * `melee_ai/training/` – model interfaces, trainer orchestration, metrics.
  * `melee_ai/cli/` – CLI entry points delegating into package modules.
- Move existing scripts into these packages, leaving thin wrappers in the repo root for backward compatibility during transition.

### 2.2 Extract reusable services
- Split `process_replays.extract` into:
  * `PlayerFeatureExtractor` handling player-specific features.
  * `ReplayExtractor` orchestrating Console iteration, error handling, and logging.
- In `train.py`, introduce `Trainer` class with methods (`setup()`, `train_epoch()`, `evaluate()`, `save_checkpoint()`) to separate concerns currently combined in `train_loop`.
- Dedicate a `MetricAggregator` module to handle confusion matrices, PRF calculations, and formatting to reduce clutter in `Trainer`.

### 2.3 Boundary enforcement
- Introduce interfaces/protocols for dependencies:
  * `DatasetProvider` protocol consumed by training to fetch dataloaders (`WindowDataset` becomes one implementation).
  * `ModelAdapter` for `GPTv7` to abstract architecture-specific logic.
- Document each interface contract and add unit tests mocking implementations to verify training logic without real data.

---
## 3. Reduce Cyclomatic Complexity & Improve Readability

### 3.1 Targeted refactors (guided by radon results)
- `train.py::train_loop` (CC 39):
  * Break into `for epoch in range(...):` driver calling smaller private methods (`_train_step`, `_log_batch`, `_evaluate_epoch`).
  * Replace nested conditionals controlling schedulers/checkpointing with strategy objects configured from `Settings`.
- `window_dataset.py`:
  * Simplify `RandomWindowSampler.__iter__` and `EpisodeThenLinearSampler._episode_order_for_rank` by extracting helper functions (e.g., `_generate_episode_permutation`, `_yield_window_indices`).
  * Unit test new helpers with deterministic RNG seeds.

### 3.2 Control flow patterns
- Replace repeated manual `try/except`/`return None` patterns with context managers yielding structured results (`Result[T]` dataclass with `.ok` flag) to make error paths explicit.
- Use guard clauses to bail early instead of deep indentation (e.g., in `process_one_replay`).

### 3.3 Type coverage
- Gradually introduce `mypy` (strict optional checks) leveraging existing type hints. Integrate `mypy.ini` aligned with new module layout.

---
## 4. Eliminate Duplication (DRY) & Consolidate Utilities

### 4.1 Schema & preprocessing alignment
- Generate schema from a single specification:
  * Authoritative YAML/JSON schema enumerating features/targets, derived to produce `Row` dataclass and to drive preprocessing mapping.
  * Replace `_prefixed` logic with a generator that iterates over schema definitions to avoid manual updates.
- Provide a `FeatureRegistry` mapping action enums → category/dense indices, reused by both preprocessing and training.

### 4.2 Shared math/helpers
- Move repeated normalization logic (button OR-ing, stick clamping) into `preprocessing/features.py` with unit tests verifying expected transformations.
- Deduplicate palette quantization math between `controller_quantization.py` and training metrics (currently similar vector math is repeated).

### 4.3 CLI helpers
- Centralize CLI argument parsing (e.g., `add_common_cli_options(parser)`) to prevent divergence among scripts.

---
## 5. Observability, Debuggability & Error Handling

### 5.1 Logging standardization
- Adopt `structlog` or the stdlib `logging` configured via `Settings.logging` (level, format, file vs console, optional JSON).
- Provide module-level loggers using `logging.getLogger(__name__)`; remove direct `loguru` dependency once parity achieved.
- Add contextual fields (replay filename, epoch, shard id) using `LoggerAdapter`.

### 5.2 Instrumentation & metrics
- Wrap long-running tasks (zarr loading, training epochs) with timing decorators that emit metrics via logging or optional Prometheus exporters.
- Add debug dumps:
  * Persist first N processed rows and dataloader batches for repro.
  * Provide CLI flag `--debug-sample` to run pipeline on a subset for faster iteration.

### 5.3 Error transparency
- Define domain-specific exception hierarchy (`MeleeAIError`, `ReplayLoadError`, `DatasetConsistencyError`).
- Ensure CLI entry points convert exceptions into structured exit codes while preserving tracebacks behind a `--verbose` flag.

---
## 6. Testing & Quality Gates

### 6.1 Test harness expansion
- Set up `tests/` structure mirroring packages (e.g., `tests/preprocessing/test_features.py`).
- Add fixtures for sample replays/zarr shards (use `dataset_FOX_vs_FOX` slices).
- Write integration tests covering:
  * End-to-end preprocessing on a mini replay to Row objects.
  * Zarr index iteration consistency with metadata.
  * Trainer stepping on a synthetic dataset to ensure gradients/logging execute without GPU.

### 6.2 Tooling integration
- Add `tox` or `nox` sessions for lint (`ruff`), type-check (`mypy`), tests (`pytest`), and complexity guard (`python -m radon cc ... --fail`).
- Configure CI (GitHub Actions) to run these sessions per push.

### 6.3 Continuous documentation
- Auto-generate API docs via `mkdocs` or `sphinx` to keep module boundaries explicit; host architecture diagrams and configuration reference.

---
## 7. Success Criteria

* All entry points load a validated `Settings` object; no hard-coded paths remain.
* Modules align with defined package boundaries and expose SRP-compliant classes/functions.
* Cyclomatic complexity reduced below agreed thresholds (e.g., `train_loop` < 10, sampler methods < 7).
* Shared schema registry ensures preprocessing/training column order stays in sync.
* Logging output includes contextual metadata enabling replay of failures without stepping through code.
* Automated CI enforces lint, type, test, and complexity gates, providing quick feedback for new features.