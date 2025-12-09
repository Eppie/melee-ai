"""Feature extraction utilities.

Helper functions for building feature vectors from parsed game state.
"""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


def normalize_position(x: float, y: float, stage_half_width: float, stage_half_height: float) -> tuple:
    """Normalize position coordinates to [-1, 1] range.

    Args:
        x: X position in game units
        y: Y position in game units
        stage_half_width: Half-width of stage for normalization
        stage_half_height: Half-height of stage for normalization

    Returns:
        (normalized_x, normalized_y)
    """
    normalized_x = x / stage_half_width
    normalized_y = y / stage_half_height
    return normalized_x, normalized_y


def denormalize_position(x_norm: float, y_norm: float, stage_half_width: float, stage_half_height: float) -> tuple:
    """Convert normalized coordinates back to game units.

    Args:
        x_norm: Normalized X in [-1, 1]
        y_norm: Normalized Y in [-1, 1]
        stage_half_width: Half-width of stage
        stage_half_height: Half-height of stage

    Returns:
        (x, y) in game units
    """
    x = x_norm * stage_half_width
    y = y_norm * stage_half_height
    return x, y


def build_feature_vector(
    p1_features: np.ndarray,
    p2_features: np.ndarray,
    global_features: np.ndarray,
    relational_features: np.ndarray,
) -> np.ndarray:
    """Concatenate all feature components into single vector.

    Args:
        p1_features: Player 1 features
        p2_features: Player 2 features
        global_features: Stage, frame count, etc.
        relational_features: Distance between players, relative position

    Returns:
        Concatenated feature vector
    """
    return np.concatenate([
        p1_features,
        p2_features,
        global_features,
        relational_features,
    ])


def compute_relational_features(
    p1_x: float,
    p1_y: float,
    p2_x: float,
    p2_y: float,
) -> np.ndarray:
    """Compute features describing relationship between players.

    Args:
        p1_x, p1_y: Player 1 position
        p2_x, p2_y: Player 2 position

    Returns:
        [3] array: [distance, relative_x, relative_y]
            where relative = p2 - p1
    """
    relative_x = p2_x - p1_x
    relative_y = p2_y - p1_y
    distance = np.sqrt(relative_x**2 + relative_y**2)
    return np.array([distance, relative_x, relative_y])


class FeatureNormalizer:
    """Handles feature normalization statistics.

    Computes and applies mean/std normalization to continuous features.
    Saves normalization stats for inference time.
    """

    def __init__(self):
        """Initialize normalizer."""
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, features: np.ndarray) -> None:
        """Compute normalization statistics from data.

        Args:
            features: [n_samples, feature_dim] array
        """
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0) + 1e-8 # Add epsilon to prevent division by zero

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Apply normalization to features.

        Args:
            features: [n_samples, feature_dim] array

        Returns:
            Normalized features
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer has not been fitted yet. Call .fit() first.")
        return (features - self.mean) / self.std

    def inverse_transform(self, features: np.ndarray) -> np.ndarray:
        """Reverse normalization.

        Args:
            features: [n_samples, feature_dim] normalized array

        Returns:
            Original scale features
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer has not been fitted yet. Call .fit() first.")
        return features * self.std + self.mean

    def save(self, path: Path) -> None:
        """Save normalization stats to file.

        Args:
            path: Path to save the .npz file
        """
        if self.mean is None or self.std is None:
            raise ValueError("Normalizer has not been fitted yet. Call .fit() first.")
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: Path) -> 'FeatureNormalizer':
        """Load normalization stats from file.

        Args:
            path: Path to the .npz file

        Returns:
            A FeatureNormalizer instance with loaded statistics
        """
        data = np.load(path)
        normalizer = cls()
        normalizer.mean = data['mean']
        normalizer.std = data['std']
        return normalizer
