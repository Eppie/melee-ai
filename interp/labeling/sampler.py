"""
Smart frame sampling for labeling.

Provides strategies for selecting diverse, informative frames to label:
- Random sampling (baseline)
- Diversity sampling (K-means on activations)
- Uncertainty sampling (high model loss)
- Stratified sampling (by action, character, etc.)

Usage:
    from interp.labeling import FrameSampler

    sampler = FrameSampler(model, dataset, colmap, device)

    # Sample diverse frames
    frames = sampler.sample(
        n_samples=100,
        strategy="diversity",
        hook_point=HookPoint(HookPointType.BLOCK_OUTPUT, 4),
    )

    # Sample uncertain frames (where model is confused)
    uncertain_frames = sampler.sample(
        n_samples=50,
        strategy="uncertainty",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

if TYPE_CHECKING:
    from column_map import ColumnMap
    from interp.hooks import HookPoint
    from model.nano_gpt import GPT


@dataclass
class SampledFrame:
    """A frame selected for labeling."""

    episode_idx: int
    frame_idx: int
    frame_id: str  # Unique identifier

    # Raw features
    features: Tensor  # [num_features]

    # Cached activations (optional)
    activations: Optional[Tensor] = None  # [embedding_dim]

    # Sampling metadata
    sampling_score: float = 0.0  # Score from sampling strategy
    cluster_id: Optional[int] = None  # For diversity sampling

    # Game state info (for display)
    stage: Optional[str] = None
    p1_character: Optional[str] = None
    p2_character: Optional[str] = None
    p1_action: Optional[str] = None
    p2_action: Optional[str] = None
    p1_percent: float = 0.0
    p2_percent: float = 0.0
    p1_stocks: int = 4
    p2_stocks: int = 4
    p1_position: Tuple[float, float] = (0.0, 0.0)
    p2_position: Tuple[float, float] = (0.0, 0.0)

    # Model predictions (optional)
    model_prediction: Optional[Dict[str, int]] = None
    model_confidence: Optional[Dict[str, float]] = None


@dataclass
class SamplingResult:
    """Result of a sampling operation."""

    frames: List[SampledFrame]
    strategy: str
    n_requested: int
    n_sampled: int
    activations_cached: bool
    metadata: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def __getitem__(self, idx: int) -> SampledFrame:
        return self.frames[idx]


class FrameSampler:
    """
    Smart frame sampler for labeling.

    Implements multiple sampling strategies to select informative
    frames for human labeling.
    """

    def __init__(
        self,
        model: "GPT",
        dataset,  # WindowDataset or similar
        colmap: "ColumnMap",
        device: torch.device,
    ):
        self.model = model
        self.dataset = dataset
        self.colmap = colmap
        self.device = device

        # Caches
        self._activation_cache: Dict[str, Tensor] = {}
        self._feature_cache: Dict[str, Tensor] = {}

    def sample(
        self,
        n_samples: int,
        strategy: str = "diversity",
        hook_point: Optional["HookPoint"] = None,
        batch_size: int = 64,
        exclude_frame_ids: Optional[List[str]] = None,
        **strategy_kwargs,
    ) -> SamplingResult:
        """
        Sample frames for labeling.

        Args:
            n_samples: Number of frames to sample
            strategy: Sampling strategy:
                - "random": Uniform random sampling
                - "diversity": K-means clustering on activations
                - "uncertainty": High model loss/entropy
                - "stratified": Stratified by action/character
            hook_point: Where to extract activations (for diversity)
            batch_size: Batch size for processing
            exclude_frame_ids: Frame IDs to exclude (already labeled)
            **strategy_kwargs: Strategy-specific parameters

        Returns:
            SamplingResult with selected frames
        """
        exclude_set = set(exclude_frame_ids or [])

        if strategy == "random":
            return self._sample_random(n_samples, exclude_set, batch_size)
        elif strategy == "diversity":
            return self._sample_diversity(
                n_samples, hook_point, exclude_set, batch_size, **strategy_kwargs
            )
        elif strategy == "uncertainty":
            return self._sample_uncertainty(
                n_samples, exclude_set, batch_size, **strategy_kwargs
            )
        elif strategy == "stratified":
            return self._sample_stratified(
                n_samples, exclude_set, batch_size, **strategy_kwargs
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _sample_random(
        self,
        n_samples: int,
        exclude_set: set,
        batch_size: int,
    ) -> SamplingResult:
        """Uniform random sampling."""
        # Get all valid indices
        all_indices = []
        for ep_idx in range(len(self.dataset.corpus_index.episodes)):
            ep_len = self.dataset.corpus_index.episodes[ep_idx].length
            for frame_idx in range(ep_len):
                frame_id = f"ep{ep_idx}_frame{frame_idx}"
                if frame_id not in exclude_set:
                    all_indices.append((ep_idx, frame_idx))

        # Random sample
        n_available = len(all_indices)
        n_to_sample = min(n_samples, n_available)

        rng = np.random.default_rng()
        selected_indices = rng.choice(
            n_available, size=n_to_sample, replace=False
        )

        frames = []
        for idx in selected_indices:
            ep_idx, frame_idx = all_indices[idx]
            frame = self._create_sampled_frame(ep_idx, frame_idx)
            frames.append(frame)

        return SamplingResult(
            frames=frames,
            strategy="random",
            n_requested=n_samples,
            n_sampled=len(frames),
            activations_cached=False,
        )

    def _sample_diversity(
        self,
        n_samples: int,
        hook_point: Optional["HookPoint"],
        exclude_set: set,
        batch_size: int,
        n_candidates: int = 1000,
        **kwargs,
    ) -> SamplingResult:
        """
        Diversity sampling using K-means clustering on activations.

        Samples frames that are spread out in activation space.
        """
        from sklearn.cluster import MiniBatchKMeans

        # First, get candidate frames with activations
        candidates = self._collect_activations(
            n_candidates, hook_point, exclude_set, batch_size
        )

        if len(candidates) < n_samples:
            # Not enough candidates, return all
            return SamplingResult(
                frames=candidates,
                strategy="diversity",
                n_requested=n_samples,
                n_sampled=len(candidates),
                activations_cached=True,
                metadata={"n_candidates": len(candidates)},
            )

        # Stack activations for clustering
        activations = torch.stack([f.activations for f in candidates])
        act_np = activations.numpy()

        # K-means clustering
        n_clusters = min(n_samples, len(candidates))
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=3,
        )
        cluster_labels = kmeans.fit_predict(act_np)

        # Select one frame per cluster (closest to centroid)
        selected_frames = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            # Find closest to centroid
            centroid = kmeans.cluster_centers_[cluster_id]
            cluster_acts = act_np[cluster_mask]
            distances = np.linalg.norm(cluster_acts - centroid, axis=1)
            best_idx = cluster_indices[np.argmin(distances)]

            frame = candidates[best_idx]
            frame.cluster_id = cluster_id
            frame.sampling_score = float(-distances.min())  # Negative distance
            selected_frames.append(frame)

        return SamplingResult(
            frames=selected_frames,
            strategy="diversity",
            n_requested=n_samples,
            n_sampled=len(selected_frames),
            activations_cached=True,
            metadata={
                "n_candidates": len(candidates),
                "n_clusters": n_clusters,
            },
        )

    def _sample_uncertainty(
        self,
        n_samples: int,
        exclude_set: set,
        batch_size: int,
        n_candidates: int = 1000,
        head_name: str = "main_stick",
        **kwargs,
    ) -> SamplingResult:
        """
        Uncertainty sampling - select frames where model is least confident.

        Uses entropy of model predictions as uncertainty measure.
        """
        from train.batch_utils import build_model_inputs

        # Collect candidate frames with predictions
        candidates = []
        uncertainties = []

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if len(candidates) >= n_candidates:
                    break

                X = batch["X"].to(self.device)
                episode_indices = batch.get("episode_idx", [0] * X.shape[0])
                frame_indices = batch.get("frame_idx", list(range(X.shape[0])))

                # Get model predictions
                inputs_td = build_model_inputs(X, self.colmap)
                outputs = self.model(inputs_td)

                # Compute entropy for each sample
                logits = outputs[head_name][:, -1, :]  # Last position
                probs = torch.softmax(logits, dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

                for i in range(X.shape[0]):
                    ep_idx = int(episode_indices[i]) if torch.is_tensor(episode_indices[i]) else episode_indices[i]
                    fr_idx = int(frame_indices[i]) if torch.is_tensor(frame_indices[i]) else frame_indices[i]
                    frame_id = f"ep{ep_idx}_frame{fr_idx}"

                    if frame_id in exclude_set:
                        continue

                    frame = SampledFrame(
                        episode_idx=ep_idx,
                        frame_idx=fr_idx,
                        frame_id=frame_id,
                        features=X[i, -1, :].cpu(),
                        sampling_score=float(entropy[i]),
                    )
                    self._populate_frame_info(frame)
                    candidates.append(frame)
                    uncertainties.append(float(entropy[i]))

                    if len(candidates) >= n_candidates:
                        break

        # Select top-k most uncertain
        if len(candidates) <= n_samples:
            selected = candidates
        else:
            # Sort by uncertainty (descending)
            sorted_indices = np.argsort(uncertainties)[::-1]
            selected = [candidates[i] for i in sorted_indices[:n_samples]]

        return SamplingResult(
            frames=selected,
            strategy="uncertainty",
            n_requested=n_samples,
            n_sampled=len(selected),
            activations_cached=False,
            metadata={
                "n_candidates": len(candidates),
                "head_name": head_name,
            },
        )

    def _sample_stratified(
        self,
        n_samples: int,
        exclude_set: set,
        batch_size: int,
        stratify_by: str = "action",
        **kwargs,
    ) -> SamplingResult:
        """
        Stratified sampling by game state attribute.

        Ensures diverse representation of actions, characters, etc.
        """
        # Collect frames grouped by stratum
        strata: Dict[str, List[SampledFrame]] = {}
        max_per_stratum = max(10, n_samples // 10)  # At least 10 per stratum

        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        for batch in dataloader:
            X = batch["X"]
            episode_indices = batch.get("episode_idx", [0] * X.shape[0])
            frame_indices = batch.get("frame_idx", list(range(X.shape[0])))

            for i in range(X.shape[0]):
                ep_idx = int(episode_indices[i]) if torch.is_tensor(episode_indices[i]) else episode_indices[i]
                fr_idx = int(frame_indices[i]) if torch.is_tensor(frame_indices[i]) else frame_indices[i]
                frame_id = f"ep{ep_idx}_frame{fr_idx}"

                if frame_id in exclude_set:
                    continue

                # Get stratum key
                if stratify_by == "action":
                    action_col = self.colmap.feat_names.index("p1_action")
                    stratum_key = f"action_{int(X[i, -1, action_col])}"
                elif stratify_by == "character":
                    char_col = self.colmap.feat_names.index("p1_character")
                    stratum_key = f"char_{int(X[i, -1, char_col])}"
                else:
                    stratum_key = "all"

                if stratum_key not in strata:
                    strata[stratum_key] = []

                if len(strata[stratum_key]) < max_per_stratum:
                    frame = SampledFrame(
                        episode_idx=ep_idx,
                        frame_idx=fr_idx,
                        frame_id=frame_id,
                        features=X[i, -1, :],
                    )
                    self._populate_frame_info(frame)
                    strata[stratum_key].append(frame)

            # Check if we have enough
            total = sum(len(v) for v in strata.values())
            if total >= n_samples * 2:  # 2x for selection buffer
                break

        # Sample proportionally from each stratum
        samples_per_stratum = max(1, n_samples // len(strata)) if strata else 0
        selected = []

        rng = np.random.default_rng()
        for stratum_key, frames in strata.items():
            n_from_stratum = min(samples_per_stratum, len(frames))
            if n_from_stratum > 0:
                indices = rng.choice(len(frames), size=n_from_stratum, replace=False)
                for idx in indices:
                    frames[idx].sampling_score = 1.0  # All equal
                    selected.append(frames[idx])

        # If we need more, sample randomly from remaining
        while len(selected) < n_samples:
            all_remaining = []
            for frames in strata.values():
                for f in frames:
                    if f not in selected:
                        all_remaining.append(f)
            if not all_remaining:
                break
            selected.append(rng.choice(all_remaining))

        return SamplingResult(
            frames=selected[:n_samples],
            strategy="stratified",
            n_requested=n_samples,
            n_sampled=min(len(selected), n_samples),
            activations_cached=False,
            metadata={
                "stratify_by": stratify_by,
                "n_strata": len(strata),
            },
        )

    def _collect_activations(
        self,
        n_samples: int,
        hook_point: Optional["HookPoint"],
        exclude_set: set,
        batch_size: int,
    ) -> List[SampledFrame]:
        """Collect frames with their activations."""
        from interp.hooks import HookManager, HookPoint, HookPointType
        from train.batch_utils import build_model_inputs

        # Default hook point: last layer output
        if hook_point is None:
            n_layers = len(self.model.blocks)
            hook_point = HookPoint(HookPointType.BLOCK_OUTPUT, n_layers - 1)

        hook_manager = HookManager(self.model)
        hook_manager.install_hooks([hook_point])

        frames = []
        dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        self.model.eval()
        try:
            with torch.no_grad():
                for batch in dataloader:
                    if len(frames) >= n_samples:
                        break

                    X = batch["X"].to(self.device)
                    episode_indices = batch.get("episode_idx", [0] * X.shape[0])
                    frame_indices = batch.get("frame_idx", list(range(X.shape[0])))

                    # Forward pass to collect activations
                    inputs_td = build_model_inputs(X, self.colmap)
                    _ = self.model(inputs_td)

                    # Get cached activations
                    activations = hook_manager.get_single(hook_point)
                    activations = activations[:, -1, :]  # Last position

                    for i in range(X.shape[0]):
                        ep_idx = int(episode_indices[i]) if torch.is_tensor(episode_indices[i]) else episode_indices[i]
                        fr_idx = int(frame_indices[i]) if torch.is_tensor(frame_indices[i]) else frame_indices[i]
                        frame_id = f"ep{ep_idx}_frame{fr_idx}"

                        if frame_id in exclude_set:
                            continue

                        frame = SampledFrame(
                            episode_idx=ep_idx,
                            frame_idx=fr_idx,
                            frame_id=frame_id,
                            features=X[i, -1, :].cpu(),
                            activations=activations[i].cpu(),
                        )
                        self._populate_frame_info(frame)
                        frames.append(frame)

                        if len(frames) >= n_samples:
                            break

                    hook_manager.clear()
        finally:
            hook_manager.remove_hooks()

        return frames

    def _create_sampled_frame(
        self,
        episode_idx: int,
        frame_idx: int,
    ) -> SampledFrame:
        """Create a SampledFrame from dataset indices."""
        # Get the actual data
        # This is a simplified version - actual implementation depends on dataset structure
        frame_id = f"ep{episode_idx}_frame{frame_idx}"

        # Try to get from dataset
        try:
            item = self.dataset[episode_idx * 1000 + frame_idx]  # Approximate
            features = item["X"][-1] if item["X"].dim() > 1 else item["X"]
        except (IndexError, KeyError):
            features = torch.zeros(len(self.colmap.feat_names))

        frame = SampledFrame(
            episode_idx=episode_idx,
            frame_idx=frame_idx,
            frame_id=frame_id,
            features=features,
        )
        self._populate_frame_info(frame)
        return frame

    def _populate_frame_info(self, frame: SampledFrame) -> None:
        """Populate game state info from features."""
        try:
            features = frame.features

            # Stage
            stage_col = self.colmap.feat_names.index("stage")
            frame.stage = f"stage_{int(features[stage_col])}"

            # Characters
            p1_char_col = self.colmap.feat_names.index("p1_character")
            p2_char_col = self.colmap.feat_names.index("p2_character")
            frame.p1_character = f"char_{int(features[p1_char_col])}"
            frame.p2_character = f"char_{int(features[p2_char_col])}"

            # Actions
            p1_action_col = self.colmap.feat_names.index("p1_action")
            p2_action_col = self.colmap.feat_names.index("p2_action")
            frame.p1_action = f"action_{int(features[p1_action_col])}"
            frame.p2_action = f"action_{int(features[p2_action_col])}"

            # Percent - find the percent columns
            for i, name in enumerate(self.colmap.feat_names):
                if name == "p1_percent":
                    frame.p1_percent = float(features[i])
                elif name == "p2_percent":
                    frame.p2_percent = float(features[i])
                elif name == "p1_stock":
                    frame.p1_stocks = int(features[i])
                elif name == "p2_stock":
                    frame.p2_stocks = int(features[i])

            # Positions
            for i, name in enumerate(self.colmap.feat_names):
                if name == "p1_position_x":
                    p1_x = float(features[i])
                elif name == "p1_position_y":
                    p1_y = float(features[i])
                    frame.p1_position = (p1_x, p1_y)
                elif name == "p2_position_x":
                    p2_x = float(features[i])
                elif name == "p2_position_y":
                    p2_y = float(features[i])
                    frame.p2_position = (p2_x, p2_y)

        except (ValueError, IndexError):
            pass  # Some fields may not be available


__all__ = [
    "SampledFrame",
    "SamplingResult",
    "FrameSampler",
]
