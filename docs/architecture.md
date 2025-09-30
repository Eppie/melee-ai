# Melee AI Architecture

## Overview

This document describes the high-level architecture of the Melee AI system, which processes Super Smash Bros. Melee replay data to train AI models for controller inputs.

## System Components

### 1. Configuration (`melee_ai/config/`)
**Purpose**: Centralized configuration management with single source of truth.

**Components**:
- `Settings`: Hierarchical configuration dataclass with validation
- `constants.py`: Centralized constants and quantization palettes
- `cli.py`: CLI argument parsing and configuration loading

**Key Features**:
- Environment variable overrides (`MELEE_AI_*`)
- CLI argument support
- Validation hooks
- Computed properties

### 2. Preprocessing (`melee_ai/preprocessing/`)
**Purpose**: Transform raw replay data into structured training features.

**Data Flow**:
```
Raw SLP Replay → Console → Frame Data → Feature Extraction → Structured Rows
```

**Key Components**:
- `ReplayExtractor`: Orchestrates console iteration and error handling
- `PlayerFeatureExtractor`: Extracts player-specific features (position, buttons, sticks)
- `ActionMapper`: Maps game actions to dense indices
- `Quantizer`: Converts continuous inputs to discrete training targets

**Responsibilities**:
- Parse SLP replay files using libmelee
- Extract game state at each frame
- Normalize and quantize controller inputs
- Handle preprocessing errors gracefully

### 3. Data (`melee_ai/data/`)
**Purpose**: Manage data storage, loading, and dataset creation.

**Data Flow**:
```
Structured Rows → Schema Validation → Storage Format → Dataset Creation → DataLoader
```

**Key Components**:
- `Schema`: Data structure definitions and validation
- `StorageManager`: Handle parquet/zarr storage formats
- `DatasetProvider`: Protocol for dataset implementations
- `WindowDataset`: Sliding window dataset for sequence modeling
- `Samplers`: Episode-linear and random window sampling strategies

**Responsibilities**:
- Define data schemas with validation
- Store processed data in efficient formats
- Create datasets with proper sampling strategies
- Provide data loading interfaces for training

### 4. Training (`melee_ai/training/`)
**Purpose**: Train models using prepared datasets.

**Data Flow**:
```
Dataset → DataLoader → Model → Loss Calculation → Gradient Updates → Checkpointing
```

**Key Components**:
- `Trainer`: Main training orchestration with lifecycle management
- `ModelAdapter`: Protocol for model implementations
- `MetricAggregator`: Compute and track training metrics
- `CheckpointManager`: Handle model saving/loading
- `LRScheduler`: Learning rate scheduling

**Responsibilities**:
- Manage training lifecycle (setup, epochs, evaluation)
- Compute and track metrics (accuracy, loss, baselines)
- Handle checkpointing and model persistence
- Provide training progress monitoring

### 5. Models (`melee_ai/models/`)
**Purpose**: Define neural network architectures.

**Key Components**:
- `GPTModel`: Transformer-based architecture for sequence modeling
- `ModelHeads`: Specialized output heads for different controller components
- `ModelRegistry`: Factory for model instantiation

**Responsibilities**:
- Define model architectures
- Handle model initialization and configuration
- Provide model interfaces for training/inference

### 6. CLI (`melee_ai/cli/`)
**Purpose**: Command-line interfaces for the system.

**Key Components**:
- `train.py`: Training command interface
- `preprocess.py`: Data preprocessing interface
- `evaluate.py`: Model evaluation interface

**Responsibilities**:
- Parse command-line arguments
- Load and validate configuration
- Delegate to appropriate service modules

## Data Flow Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Entry     │───▶│   Configuration │───▶│   Preprocessing │
│   Points        │    │   Management    │    │   Services      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌─────────────────┐             │
│   Training      │◀───│   Data Storage  │◀────────────┘
│   Services      │    │   & Loading     │
└─────────────────┘    └─────────────────┘
         │
┌─────────────────┐
│   Model         │
│   Architectures │
└─────────────────┘
```

## Design Principles

### Single Responsibility Principle (SRP)
Each module has a single, well-defined purpose:
- Configuration: Centralized settings management
- Preprocessing: Raw data transformation
- Data: Storage and dataset creation
- Training: Model training orchestration
- Models: Architecture definitions

### Dependency Injection
Components depend on abstractions, not concretions:
- `Trainer` depends on `DatasetProvider` protocol
- `ModelAdapter` abstracts model implementations
- Configuration injected rather than imported

### Error Handling
- Domain-specific exceptions with clear hierarchies
- Graceful degradation for non-critical failures
- Comprehensive logging with contextual information

### Configuration Synchronization
- Single source of truth in `Settings`
- Environment variable overrides
- CLI argument support
- Validation at all levels

## Interface Contracts

### DatasetProvider Protocol
```python
class DatasetProvider(Protocol):
    def get_dataset(self, settings: Settings) -> Dataset:
        """Create dataset from configuration."""
        ...

    def get_dataloader(self, settings: Settings) -> DataLoader:
        """Create dataloader from configuration."""
        ...
```

### ModelAdapter Protocol
```python
class ModelAdapter(Protocol):
    def train_step(self, batch: Batch) -> LossInfo:
        """Perform single training step."""
        ...

    def evaluate(self, batch: Batch) -> Metrics:
        """Evaluate model on batch."""
        ...
```

## Development Workflow

1. **Configuration**: Define settings in `Settings` dataclass
2. **Preprocessing**: Extract features using `ReplayExtractor`
3. **Storage**: Store processed data using `StorageManager`
4. **Dataset**: Create datasets using `DatasetProvider` implementations
5. **Training**: Train models using `Trainer` orchestration
6. **Evaluation**: Evaluate using metrics from `MetricAggregator`

## Future Extensions

- **Distributed Training**: Add support for multi-GPU/multi-node training
- **Model Registry**: Support for multiple model architectures
- **Experiment Tracking**: Integration with MLflow/TensorBoard
- **Data Versioning**: Track dataset versions and changes
- **Model Deployment**: Inference serving capabilities
