"""
Preprocessing module for Melee AI.

This module handles the transformation of raw replay data into structured
training features, including frame extraction, feature normalization,
and quantization.
"""

from .extractor import ReplayExtractor, PlayerFeatureExtractor
from .features import ActionMapper, Quantizer, FeatureNormalizer

__all__ = [
    "ReplayExtractor",
    "PlayerFeatureExtractor",
    "ActionMapper",
    "Quantizer",
    "FeatureNormalizer",
]
