"""Builder pattern for training session initialization.

Provides flexible, composable initialization for different training modes:
- Imitation learning (train.py)
- Reinforcement learning (ppo_train.py)
- Validation (validation.py)

Key features:
- Order-safe: Enforces dependency order automatically
- Flexible: Support custom loaders for each component
- Non-breaking: Preserves all existing behavior via callbacks
- Fluent interface: Method chaining for readable configuration

Example (Imitation Learning):
    session = (TrainingSessionBuilder()
        .with_config()
        .with_fresh_model()
        .with_dataloader()
        .with_optimizer()
        .with_amp()
        .build())

Example (PPO):
    session = (TrainingSessionBuilder()
        .with_config(overrides=cli_overrides)
        .with_model_from_checkpoint(checkpoint_path)
        .with_optimizer(factory=lambda m, c: torch.optim.Adam(m.parameters(), lr=3e-4))
        .with_amp(override_use_amp=False)
        .build())
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.amp import GradScaler

from config import get_config, init_config
from model.nano_gpt import GPT
from train.components import AMPContext
from train.setup import (
    build_optimizer,
    configure_amp,
    configure_performance_settings,
    parse_cli_overrides,
)
from utils import _resolve_device


@dataclass
class TrainingSession:
    """Immutable training session state.

    Contains all components needed for training/validation:
    - config: Configuration object
    - model: GPT model
    - device: torch.device
    - amp: AMP context
    - optimizer: Optional optimizer
    - scaler: Optional GradScaler
    - loader: Optional DataLoader
    - dataset: Optional dataset
    - sampler: Optional sampler
    - column_map: Optional ColumnMap
    """

    config: Any
    model: GPT
    device: torch.device
    amp: AMPContext
    optimizer: Optional[torch.optim.Optimizer] = None
    scaler: Optional[GradScaler] = None
    loader: Optional[Any] = None
    dataset: Optional[Any] = None
    sampler: Optional[Any] = None
    column_map: Optional[Any] = None


# Type aliases for callbacks
ModelLoader = Callable[[Any], GPT]  # config -> model
OptimizerFactory = Callable[[GPT, Any], torch.optim.Optimizer]  # model, config -> optimizer
DataLoaderFactory = Callable[[Any], Tuple[Any, Any, Any]]  # config -> (loader, dataset, sampler)


class TrainingSessionBuilder:
    """Fluent builder for training session initialization.

    Enforces dependency order:
    1. Config (required first)
    2. Model (required)
    3. Device resolution (automatic)
    4. AMP configuration (automatic)
    5. DataLoader (optional)
    6. Optimizer (optional)
    7. GradScaler (automatic if AMP + optimizer)
    """

    def __init__(self):
        """Initialize empty builder."""
        self._config: Optional[Any] = None
        self._model: Optional[GPT] = None
        self._device: Optional[torch.device] = None
        self._amp: Optional[AMPContext] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._scaler: Optional[GradScaler] = None
        self._loader: Optional[Any] = None
        self._dataset: Optional[Any] = None
        self._sampler: Optional[Any] = None
        self._column_map: Optional[Any] = None

        # Callbacks for customization
        self._model_loader: Optional[ModelLoader] = None
        self._optimizer_factory: Optional[OptimizerFactory] = None
        self._dataloader_factory: Optional[DataLoaderFactory] = None
        self._amp_override: Optional[bool] = None

    def with_config(self, overrides: Optional[Dict[str, str]] = None) -> TrainingSessionBuilder:
        """Initialize config with optional CLI overrides.

        Must be called first.

        Args:
            overrides: Optional CLI overrides dict

        Returns:
            Self for method chaining
        """
        init_config(overrides=overrides or {})
        self._config = get_config()
        return self

    def with_fresh_model(self) -> TrainingSessionBuilder:
        """Create fresh model from config.

        Requires: config

        Returns:
            Self for method chaining
        """
        self._ensure_config()
        self._model_loader = lambda config: GPT(config)
        return self

    def with_model_from_checkpoint(
        self,
        checkpoint_path: Path,
        loader: Optional[Callable[[Path], GPT]] = None,
    ) -> TrainingSessionBuilder:
        """Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            loader: Custom loader function (default: GPTInferenceEngine)

        Requires: config

        Returns:
            Self for method chaining
        """
        self._ensure_config()

        if loader is None:
            # Default: Use GPTInferenceEngine
            def default_loader(config):
                from model_interface import GPTInferenceEngine

                engine = GPTInferenceEngine(checkpoint_path=checkpoint_path)
                return engine.model

            self._model_loader = default_loader
        else:
            self._model_loader = lambda config: loader(checkpoint_path)

        return self

    def with_dataloader(
        self,
        factory: Optional[DataLoaderFactory] = None,
    ) -> TrainingSessionBuilder:
        """Setup DataLoader.

        Args:
            factory: Custom factory (default: make_dataloader from window_dataset)

        Requires: config

        Returns:
            Self for method chaining
        """
        self._ensure_config()

        if factory is None:
            from window_dataset import make_dataloader

            self._dataloader_factory = lambda config: make_dataloader(config)
        else:
            self._dataloader_factory = factory

        return self

    def with_optimizer(
        self,
        factory: Optional[OptimizerFactory] = None,
    ) -> TrainingSessionBuilder:
        """Setup optimizer.

        Args:
            factory: Custom factory (default: build_optimizer from train.setup)

        Requires: model (set during build)

        Returns:
            Self for method chaining
        """
        self._optimizer_factory = factory or build_optimizer
        return self

    def with_amp(
        self,
        override_use_amp: Optional[bool] = None,
    ) -> TrainingSessionBuilder:
        """Configure AMP (Automatic Mixed Precision).

        Args:
            override_use_amp: Override config.train.use_amp (for PPO)

        Requires: config, device (set during build)

        Returns:
            Self for method chaining
        """
        self._amp_override = override_use_amp
        return self

    def build(self) -> TrainingSession:
        """Build training session.

        Enforces dependency order:
        1. Config (required)
        2. Model (required)
        3. Device resolution
        4. AMP configuration
        5. DataLoader (optional)
        6. Optimizer (optional)
        7. GradScaler (if AMP enabled)

        Returns:
            TrainingSession with all components initialized

        Raises:
            ValueError: If config or model not set
        """
        # 1. Validate config
        self._ensure_config()

        # 2. Load/create model
        if self._model_loader is None:
            raise ValueError(
                "Must call with_fresh_model() or with_model_from_checkpoint() before build()"
            )
        self._model = self._model_loader(self._config)

        # 3. Resolve device and move model
        self._device = _resolve_device(None)
        self._model = self._model.to(self._device)

        # 4. Configure global performance settings
        configure_performance_settings(self._config, self._device)

        # 5. Configure AMP
        if self._amp_override is not None:
            # Temporarily override config (PPO pattern)
            original_use_amp = self._config.train.use_amp
            self._config.train.use_amp = self._amp_override
            self._amp = configure_amp(self._config, self._device)
            self._config.train.use_amp = original_use_amp
        else:
            self._amp = configure_amp(self._config, self._device)

        # 6. Setup DataLoader (optional)
        if self._dataloader_factory is not None:
            self._loader, self._dataset, self._sampler = self._dataloader_factory(
                self._config
            )

            # Create ColumnMap from dataset
            from column_map import ColumnMap

            self._column_map = ColumnMap.from_dataset(self._dataset)

        # 7. Setup optimizer (optional)
        if self._optimizer_factory is not None:
            self._optimizer = self._optimizer_factory(self._model, self._config)

        # 8. Setup GradScaler (if AMP enabled and optimizer present)
        if self._optimizer is not None:
            use_grad_scaler = self._amp.enabled and self._amp.dtype == torch.float16
            scaler_device = self._amp.device_type if use_grad_scaler else "cpu"
            self._scaler = GradScaler(device=scaler_device, enabled=use_grad_scaler)

        return TrainingSession(
            config=self._config,
            model=self._model,
            device=self._device,
            amp=self._amp,
            optimizer=self._optimizer,
            scaler=self._scaler,
            loader=self._loader,
            dataset=self._dataset,
            sampler=self._sampler,
            column_map=self._column_map,
        )

    def _ensure_config(self):
        """Ensure config is set.

        Raises:
            ValueError: If config not set
        """
        if self._config is None:
            raise ValueError("Must call with_config() first")


# Convenience functions for common patterns


def build_imitation_session(
    cli_overrides: Dict[str, str],
    debug: bool = False,
) -> TrainingSession:
    """Build session for imitation learning (train.py pattern).

    Args:
        cli_overrides: CLI override dictionary
        debug: Debug mode flag (currently unused, for compatibility)

    Returns:
        TrainingSession configured for imitation learning
    """
    builder = TrainingSessionBuilder()
    return (
        builder.with_config(overrides=cli_overrides)
        .with_dataloader()
        .with_fresh_model()
        .with_optimizer()
        .with_amp()
        .build()
    )


def build_ppo_session(
    cli_overrides: Dict[str, str],
    checkpoint_path: Path,
    ppo_config: Any,
) -> TrainingSession:
    """Build session for PPO training (ppo_train.py pattern).

    Args:
        cli_overrides: CLI override dictionary
        checkpoint_path: Path to checkpoint to load
        ppo_config: PPO configuration object

    Returns:
        TrainingSession configured for PPO
    """
    builder = TrainingSessionBuilder()

    # Custom optimizer factory for PPO
    def ppo_optimizer_factory(model, config):
        return torch.optim.Adam(
            model.parameters(),
            lr=ppo_config.learning_rate,
            eps=1e-5,
        )

    return (
        builder.with_config(overrides=cli_overrides)
        .with_model_from_checkpoint(checkpoint_path)
        .with_optimizer(factory=ppo_optimizer_factory)
        .with_amp(override_use_amp=ppo_config.use_amp)
        .build()
    )


def build_validation_session(
    cli_overrides: Dict[str, str],
    checkpoint_path: Path,
    with_dataloader: bool = True,
) -> TrainingSession:
    """Build session for validation (validation.py pattern).

    Args:
        cli_overrides: CLI override dictionary
        checkpoint_path: Path to checkpoint to load
        with_dataloader: Whether to include dataloader

    Returns:
        TrainingSession configured for validation (no optimizer)
    """
    builder = TrainingSessionBuilder()
    builder = builder.with_config(overrides=cli_overrides)
    builder = builder.with_model_from_checkpoint(checkpoint_path)

    if with_dataloader:
        builder = builder.with_dataloader()

    # No optimizer for validation
    return builder.build()
