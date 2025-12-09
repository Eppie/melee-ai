"""Configuration dataclasses."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Any
import json


@dataclass
class DataConfig:
    """Data processing configuration."""
    slp_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    n_workers: int = 8
    context_length: int = 16
    horizons: List[int] = field(default_factory=lambda: [5, 10, 20, 30, 40, 50, 60])
    val_fraction: float = 0.1
    random_seed: int = 42


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    mlp_ratio: int = 3
    dropout: float = 0.0
    character_embed_dim: int = 16
    action_state_embed_dim: int = 24
    stage_embed_dim: int = 8
    horizon_embed_dim: int = 32
    output_hidden_dim: int = 256
    n_resblocks: int = 1
    n_mixture_components: int = 6
    min_sigma: float = 1.0
    max_sigma: float = 40.0
    sigma_epsilon: float = 0.5


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_epochs: int = 100
    warmup_steps: int = 1000
    scheduler_type: str = "cosine"
    min_lr: float = 1e-5
    warmup_frac: float = 0.05  # fraction of total steps for warmup (cap applied)
    max_grad_norm: float = 1.0
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    save_every: int = 1000
    keep_top_k: int = 5
    log_every: int = 100
    device: str = "auto"
    use_compile: bool = True
    compile_mode: str = "max-autotune"
    checkpoint_dir: Optional[Path] = None
    max_episodes: Optional[int] = None
    sigma_penalty: float = 1e-3


@dataclass
class InferenceConfig:
    """Inference configuration."""
    checkpoint_path: Optional[Path] = None
    device: str = "cuda"
    batch_size: int = 1
    use_compile: bool = False


@dataclass
class VisualizationConfig:
    """Visualization configuration."""
    fps: int = 60
    dpi: int = 100
    resolution: Tuple[int, int] = (1200, 1000)
    heatmap_resolution: int = 64
    heatmap_alpha: float = 0.8
    show_ground_truth: bool = True
    show_prediction: bool = True
    display_horizons: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50, 60])


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for Path objects."""
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


@dataclass
class Config:
    """Top-level configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'Config':
        """Load from dictionary."""
        # Helper to recursively convert dicts to dataclasses
        def _from_dict(cls, data):
            if isinstance(data, dict):
                return cls(**{k: _from_dict(cls.__annotations__[k], v) if k in cls.__annotations__ and not isinstance(v, cls.__annotations__[k]) else v for k, v in data.items()})
            return data

        # Handle nested configs specifically
        data_config = DataConfig(**d.get('data', {}))
        model_config = ModelConfig(**d.get('model', {}))
        training_config = TrainingConfig(**d.get('training', {}))
        inference_config = InferenceConfig(**d.get('inference', {}))
        visualization_config = VisualizationConfig(**d.get('visualization', {}))

        # Convert path strings back to Path objects
        if data_config.slp_dir:
            data_config.slp_dir = Path(data_config.slp_dir)
        if data_config.output_dir:
            data_config.output_dir = Path(data_config.output_dir)
        if training_config.checkpoint_dir:
            training_config.checkpoint_dir = Path(training_config.checkpoint_dir)
        if inference_config.checkpoint_path:
            inference_config.checkpoint_path = Path(inference_config.checkpoint_path)

        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            inference=inference_config,
            visualization=visualization_config
        )

    def save(self, path: Path) -> None:
        """Save to JSON."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, cls=JSONEncoder, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'Config':
        """Load from JSON."""
        with open(path, 'r') as f:
            d = json.load(f)
        return cls.from_dict(d)
