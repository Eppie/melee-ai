"""Inference engine for real-time prediction."""

import torch
import numpy as np
from pathlib import Path
from collections import deque
from typing import List, Dict, Optional, Tuple

from ..model import FuturePositionPredictor
from ..config import Config, InferenceConfig
from ..constants import CONTEXT_LENGTH, HORIZONS


class FuturePredictor:
    """Inference engine for future position prediction.

    Maintains rolling context window and provides fast prediction.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        config: Optional[InferenceConfig] = None,
        device: str = 'cuda',
    ):
        """Initialize predictor.

        Args:
            checkpoint_path: Path to model checkpoint
            config: Optional inference config
            device: Device to run on
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Load model and config
        self.model, self.model_config = self.load_checkpoint(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        if config:
            self.inference_config = config
        else:
            self.inference_config = InferenceConfig(device=device)

        if self.inference_config.use_compile:
            self.model = torch.compile(self.model)

        self.context_length = CONTEXT_LENGTH
        self.horizons = HORIZONS
        self.buffer = deque(maxlen=CONTEXT_LENGTH)

    @torch.inference_mode()
    def predict(
        self,
        current_frame: np.ndarray,
        horizons: Optional[List[int]] = None,
        return_samples: bool = False,
        n_samples: int = 100,
    ) -> Optional[Dict]:
        """Predict future positions.

        Args:
            current_frame: [feature_dim] current frame features
            horizons: Optional list of horizons (default: all)
            return_samples: Whether to sample from mixture
            n_samples: Number of samples if sampling

        Returns:
            Dict with predictions:
                'p1': List of mixture_params (per horizon)
                'p2': List of mixture_params (per horizon)
            Returns None if buffer is not full.
        """
        self.buffer.append(current_frame)

        if len(self.buffer) < self.context_length:
            return None

        # Stack buffer to context tensor [1, T, feature_dim]
        context_np = np.array(self.buffer)
        context_tensor = torch.from_numpy(context_np).unsqueeze(0) # Add batch dim

        # Move to device
        if torch.cuda.is_bf16_supported() and self.device == 'cuda':
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        
        context_tensor = context_tensor.to(self.device, dtype=dtype)

        # Forward pass
        p1_trajectories, p2_trajectories = self.model(context_tensor)

        # Filter horizons if requested (though model predicts all)
        # p1_trajectories is a list of dicts, index corresponds to HORIZONS
        
        return {
            'p1': p1_trajectories,
            'p2': p2_trajectories,
        }

    def reset(self) -> None:
        """Reset context buffer."""
        self.buffer.clear()

    def warmup(self, initial_frames: np.ndarray) -> None:
        """Warmup buffer with initial frames.

        Args:
            initial_frames: [n_frames, feature_dim] initial context

        Notes:
            - Should be called before first prediction
            - Fills buffer with initial frames
        """
        for frame in initial_frames:
            self.buffer.append(frame)

    @staticmethod
    def load_checkpoint(checkpoint_path: Path) -> Tuple[FuturePositionPredictor, Config]:
        """Load checkpoint and return model + config.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            (model, config)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load config
        config_dict = checkpoint['config']
        config = Config.from_dict(config_dict)
        
        # Initialize model
        model = FuturePositionPredictor(
            d_model=config.model.d_model,
            n_layers=config.model.n_layers,
            n_heads=config.model.n_heads,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout,
            context_length=config.data.context_length,
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, config
