"""
Frame labeling toolkit for probe training.

This module provides tools for labeling game frames with probe categories:
- LabelStorage: Persistent storage for labels
- FrameSampler: Smart frame sampling strategies
- TerminalLabeler: Interactive terminal UI for labeling

Usage:
    from interp.labeling import LabelStorage, FrameSampler, TerminalLabeler
    from interp.probes import MELEE_PROBES

    # Sample diverse frames
    sampler = FrameSampler(model, dataset, colmap, device)
    frames = sampler.sample(100, strategy="diversity")

    # Create storage and labeler
    storage = LabelStorage("labels/game_phase.json", probe_name="game_phase")
    config = MELEE_PROBES["game_phase"]
    labeler = TerminalLabeler(config, storage, frames.frames)

    # Run labeling session
    stats = labeler.run()

    # Export for training
    activations, labels = storage.to_tensors()
"""

from interp.labeling.storage import (
    LabeledFrame,
    LabelingSession,
    LabelStorage,
    combine_storages,
)
from interp.labeling.sampler import (
    SampledFrame,
    SamplingResult,
    FrameSampler,
)
from interp.labeling.interface import (
    TerminalLabeler,
    LabelingStats,
    quick_label,
    get_character_name,
    get_stage_name,
    get_action_name,
)

__all__ = [
    # Storage
    "LabeledFrame",
    "LabelingSession",
    "LabelStorage",
    "combine_storages",
    # Sampler
    "SampledFrame",
    "SamplingResult",
    "FrameSampler",
    # Interface
    "TerminalLabeler",
    "LabelingStats",
    "quick_label",
    "get_character_name",
    "get_stage_name",
    "get_action_name",
]
