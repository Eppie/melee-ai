"""Sparse Autoencoder implementations for interpretability."""

from interp.sae.topk import TopKSparseAutoencoder, SAEOutput
from interp.sae.trainer import SAETrainer, TrainingHistory, train_sae
from interp.sae.steering import (
    SAESteering,
    SteeringEffect,
    SweetSpotResult,
    compute_persona_vector,
    decompose_vector_into_sae_features,
)

__all__ = [
    "TopKSparseAutoencoder",
    "SAEOutput",
    "SAETrainer",
    "TrainingHistory",
    "train_sae",
    # Steering
    "SAESteering",
    "SteeringEffect",
    "SweetSpotResult",
    "compute_persona_vector",
    "decompose_vector_into_sae_features",
]
