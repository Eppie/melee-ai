"""
Label storage for frame annotations.

Provides persistent storage for labeled frames with support for:
- Incremental saves (save as you go)
- Multiple probe types per frame
- Metadata tracking (who labeled, when, etc.)
- Export to training-ready format

Usage:
    from interp.labeling import LabelStorage, LabeledFrame

    storage = LabelStorage("labels/game_phase.json")

    # Add labeled frames
    storage.add_label(
        frame_id="ep42_frame1234",
        probe_name="game_phase",
        label=0,  # "neutral"
        activations=cached_activations,
        metadata={"episode": 42, "frame": 1234}
    )

    # Save periodically
    storage.save()

    # Export for training
    activations, labels = storage.to_tensors()
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor


@dataclass
class LabeledFrame:
    """A single labeled frame."""

    frame_id: str  # Unique identifier (e.g., "ep42_frame1234")
    probe_name: str  # Which probe this label is for
    label: int  # Class index
    label_name: str  # Human-readable class name

    # Optional activation data (stored separately for efficiency)
    has_activations: bool = False

    # Metadata
    episode_idx: Optional[int] = None
    frame_idx: Optional[int] = None
    timestamp: str = ""
    labeler: str = "unknown"
    confidence: Optional[float] = None  # Labeler's confidence (1-5)
    notes: str = ""

    # Game state context (for display during labeling)
    p1_character: Optional[str] = None
    p2_character: Optional[str] = None
    p1_action: Optional[str] = None
    p2_action: Optional[str] = None
    p1_percent: Optional[float] = None
    p2_percent: Optional[float] = None
    p1_stocks: Optional[int] = None
    p2_stocks: Optional[int] = None
    stage: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (without activations)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabeledFrame":
        """Deserialize from dictionary."""
        return cls(**data)


@dataclass
class LabelingSession:
    """Metadata about a labeling session."""

    session_id: str
    probe_name: str
    started_at: str
    labeler: str
    num_frames_labeled: int = 0
    notes: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> "LabelingSession":
        return cls(**data)


class LabelStorage:
    """
    Persistent storage for labeled frames.

    Stores labels in JSON format with optional activation tensors
    saved separately as .pt files.
    """

    def __init__(
        self,
        path: Union[str, Path],
        probe_name: Optional[str] = None,
        labeler: str = "unknown",
        auto_save: bool = True,
        auto_save_interval: int = 10,
    ):
        """
        Initialize label storage.

        Args:
            path: Path to JSON label file
            probe_name: Default probe name for new labels
            labeler: Name of person labeling
            auto_save: Automatically save after adding labels
            auto_save_interval: Save every N labels
        """
        self.path = Path(path)
        self.probe_name = probe_name
        self.labeler = labeler
        self.auto_save = auto_save
        self.auto_save_interval = auto_save_interval

        self._labels: Dict[str, LabeledFrame] = {}
        self._activations: Dict[str, Tensor] = {}
        self._sessions: List[LabelingSession] = []
        self._unsaved_count = 0

        # Load existing if present
        if self.path.exists():
            self.load()

    @property
    def num_labels(self) -> int:
        """Number of labeled frames."""
        return len(self._labels)

    @property
    def labels(self) -> List[LabeledFrame]:
        """All labeled frames."""
        return list(self._labels.values())

    def add_label(
        self,
        frame_id: str,
        label: int,
        label_name: str,
        probe_name: Optional[str] = None,
        activations: Optional[Tensor] = None,
        confidence: Optional[float] = None,
        notes: str = "",
        **metadata,
    ) -> LabeledFrame:
        """
        Add a labeled frame.

        Args:
            frame_id: Unique frame identifier
            label: Class index
            label_name: Human-readable class name
            probe_name: Probe this label is for
            activations: Optional activation tensor
            confidence: Labeler confidence (1-5)
            notes: Optional notes
            **metadata: Additional frame metadata

        Returns:
            The created LabeledFrame
        """
        probe_name = probe_name or self.probe_name
        if probe_name is None:
            raise ValueError("probe_name must be specified")

        labeled_frame = LabeledFrame(
            frame_id=frame_id,
            probe_name=probe_name,
            label=label,
            label_name=label_name,
            has_activations=activations is not None,
            labeler=self.labeler,
            confidence=confidence,
            notes=notes,
            **metadata,
        )

        self._labels[frame_id] = labeled_frame

        if activations is not None:
            self._activations[frame_id] = activations.detach().cpu()

        self._unsaved_count += 1

        # Auto-save if enabled
        if self.auto_save and self._unsaved_count >= self.auto_save_interval:
            self.save()
            self._unsaved_count = 0

        return labeled_frame

    def remove_label(self, frame_id: str) -> bool:
        """Remove a labeled frame."""
        if frame_id in self._labels:
            del self._labels[frame_id]
            if frame_id in self._activations:
                del self._activations[frame_id]
            return True
        return False

    def get_label(self, frame_id: str) -> Optional[LabeledFrame]:
        """Get a labeled frame by ID."""
        return self._labels.get(frame_id)

    def has_label(self, frame_id: str) -> bool:
        """Check if frame is labeled."""
        return frame_id in self._labels

    def get_activations(self, frame_id: str) -> Optional[Tensor]:
        """Get activations for a frame."""
        return self._activations.get(frame_id)

    def start_session(self, notes: str = "") -> LabelingSession:
        """Start a new labeling session."""
        session = LabelingSession(
            session_id=f"session_{len(self._sessions)}",
            probe_name=self.probe_name or "unknown",
            started_at=datetime.utcnow().isoformat() + "Z",
            labeler=self.labeler,
            notes=notes,
        )
        self._sessions.append(session)
        return session

    def end_session(self, num_labeled: int) -> None:
        """End current labeling session."""
        if self._sessions:
            self._sessions[-1].num_frames_labeled = num_labeled

    def save(self) -> None:
        """Save labels to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Save labels JSON
        data = {
            "probe_name": self.probe_name,
            "num_labels": self.num_labels,
            "labels": [lf.to_dict() for lf in self._labels.values()],
            "sessions": [s.to_dict() for s in self._sessions],
        }

        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

        # Save activations separately
        if self._activations:
            act_path = self.path.with_suffix(".activations.pt")
            torch.save(self._activations, act_path)

    def load(self) -> None:
        """Load labels from disk."""
        if not self.path.exists():
            return

        with open(self.path) as f:
            data = json.load(f)

        self.probe_name = data.get("probe_name", self.probe_name)
        self._labels = {
            lf["frame_id"]: LabeledFrame.from_dict(lf)
            for lf in data.get("labels", [])
        }
        self._sessions = [
            LabelingSession.from_dict(s)
            for s in data.get("sessions", [])
        ]

        # Load activations if present
        act_path = self.path.with_suffix(".activations.pt")
        if act_path.exists():
            self._activations = torch.load(act_path, weights_only=True)

    def to_tensors(
        self,
        require_activations: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Export labels as tensors for training.

        Args:
            require_activations: Only include frames with activations

        Returns:
            Tuple of (activations [N, dim], labels [N])
        """
        activations_list = []
        labels_list = []

        for frame_id, labeled_frame in self._labels.items():
            if require_activations and frame_id not in self._activations:
                continue

            if frame_id in self._activations:
                activations_list.append(self._activations[frame_id])
            labels_list.append(labeled_frame.label)

        if not activations_list:
            raise ValueError("No frames with activations found")

        activations = torch.stack(activations_list)
        labels = torch.tensor(labels_list, dtype=torch.long)

        return activations, labels

    def to_numpy(
        self,
        require_activations: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Export as numpy arrays."""
        activations, labels = self.to_tensors(require_activations)
        return activations.numpy(), labels.numpy()

    def get_label_distribution(self) -> Dict[str, int]:
        """Get count of each label class."""
        distribution: Dict[str, int] = {}
        for lf in self._labels.values():
            distribution[lf.label_name] = distribution.get(lf.label_name, 0) + 1
        return distribution

    def summary(self) -> str:
        """Generate summary of stored labels."""
        lines = [
            "=" * 50,
            "LABEL STORAGE SUMMARY",
            "=" * 50,
            f"Path: {self.path}",
            f"Probe: {self.probe_name}",
            f"Total labels: {self.num_labels}",
            f"With activations: {len(self._activations)}",
        ]

        dist = self.get_label_distribution()
        if dist:
            lines.append("\nLabel distribution:")
            for label_name, count in sorted(dist.items()):
                pct = count / self.num_labels * 100
                lines.append(f"  {label_name}: {count} ({pct:.1f}%)")

        if self._sessions:
            lines.append(f"\nSessions: {len(self._sessions)}")
            for session in self._sessions[-3:]:  # Show last 3
                lines.append(f"  {session.session_id}: {session.num_frames_labeled} frames")

        return "\n".join(lines)

    def merge(self, other: "LabelStorage") -> int:
        """
        Merge labels from another storage.

        Args:
            other: Another LabelStorage to merge from

        Returns:
            Number of labels added
        """
        added = 0
        for frame_id, labeled_frame in other._labels.items():
            if frame_id not in self._labels:
                self._labels[frame_id] = labeled_frame
                if frame_id in other._activations:
                    self._activations[frame_id] = other._activations[frame_id]
                added += 1
        return added


def combine_storages(
    storages: List[LabelStorage],
    output_path: Union[str, Path],
) -> LabelStorage:
    """
    Combine multiple label storages into one.

    Args:
        storages: List of LabelStorage objects
        output_path: Path for combined storage

    Returns:
        Combined LabelStorage
    """
    if not storages:
        raise ValueError("No storages to combine")

    combined = LabelStorage(
        output_path,
        probe_name=storages[0].probe_name,
        auto_save=False,
    )

    for storage in storages:
        combined.merge(storage)

    combined.save()
    return combined


__all__ = [
    "LabeledFrame",
    "LabelingSession",
    "LabelStorage",
    "combine_storages",
]
