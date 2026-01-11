"""
Linear probes for detecting game concepts in model activations.

Probes are simple linear classifiers trained on frozen model activations
to detect high-level game concepts like:
- Game phase (neutral, advantage, disadvantage, edgeguard)
- Combo state (in combo, can escape, just hit, recovering)
- Threat level, positioning, etc.

This helps understand what information the model has learned to represent.

Usage:
    from interp.probes import LinearProbe, ProbeTrainer, MELEE_PROBES

    # Get probe config
    config = MELEE_PROBES["game_phase"]

    # Create and train probe
    probe = LinearProbe(input_dim=512, num_classes=config.num_classes)
    trainer = ProbeTrainer(probe, config)
    history = trainer.train(activations, labels, epochs=100)

    # Evaluate
    results = trainer.evaluate(test_activations, test_labels)
    print(f"Accuracy: {results.accuracy:.2%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ProbeConfig:
    """Configuration for a probe classifier."""

    name: str
    num_classes: int
    class_names: List[str]
    description: str = ""

    def __post_init__(self):
        assert len(self.class_names) == self.num_classes, (
            f"class_names length ({len(self.class_names)}) must match "
            f"num_classes ({self.num_classes})"
        )

    def class_name(self, idx: int) -> str:
        """Get class name by index."""
        return self.class_names[idx]

    def class_index(self, name: str) -> int:
        """Get class index by name."""
        return self.class_names.index(name)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ProbeConfig":
        """Deserialize from dictionary."""
        return cls(**data)


# Predefined probes for Melee game concepts
MELEE_PROBES: Dict[str, ProbeConfig] = {
    "game_phase": ProbeConfig(
        name="game_phase",
        num_classes=4,
        class_names=["neutral", "advantage", "disadvantage", "edgeguard"],
        description="Overall game phase - who has the upper hand?",
    ),
    "combo_state": ProbeConfig(
        name="combo_state",
        num_classes=4,
        class_names=["none", "starting", "continuing", "ending"],
        description="Combo state for P1",
    ),
    "recovery_state": ProbeConfig(
        name="recovery_state",
        num_classes=3,
        class_names=["on_stage", "offstage_safe", "offstage_danger"],
        description="Recovery situation for P1",
    ),
    "pressure_state": ProbeConfig(
        name="pressure_state",
        num_classes=3,
        class_names=["neutral", "pressuring", "being_pressured"],
        description="Shield pressure / offensive pressure state",
    ),
    "threat_level": ProbeConfig(
        name="threat_level",
        num_classes=3,
        class_names=["safe", "cautious", "danger"],
        description="How dangerous is the current situation for P1?",
    ),
    "positioning": ProbeConfig(
        name="positioning",
        num_classes=4,
        class_names=["center", "corner", "ledge", "platform"],
        description="Stage positioning for P1",
    ),
}


class LinearProbe(nn.Module):
    """
    Simple linear probe (logistic regression) on frozen activations.

    This is intentionally simple - if a linear classifier can detect
    a concept, then that concept is linearly represented in the activations.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        bias: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.classifier = nn.Linear(input_dim, num_classes, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Activations [batch, input_dim] or [batch, seq, input_dim]

        Returns:
            Logits [batch, num_classes] or [batch, seq, num_classes]
        """
        return self.classifier(x)

    def predict(self, x: Tensor) -> Tensor:
        """Get predicted class indices."""
        logits = self.forward(x)
        return logits.argmax(dim=-1)

    def predict_proba(self, x: Tensor) -> Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class MLPProbe(nn.Module):
    """
    MLP probe with one hidden layer.

    Use this if linear probes fail - it may indicate the concept
    is represented non-linearly.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)

    def predict(self, x: Tensor) -> Tensor:
        logits = self.forward(x)
        return logits.argmax(dim=-1)

    def predict_proba(self, x: Tensor) -> Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class PooledProbe(nn.Module):
    """
    Probe that pools SAE features across the context window before classification.

    This is crucial for behavioral concepts that depend on information from
    earlier in the sequence. For example, "tech away" depends on:
    - When the opponent entered knockdown (10-30 frames ago)
    - The trajectory leading to the tech situation
    - DI inputs during hitstun

    Pooling modes:
    - "max": Max-pool across time (captures if feature ever fired)
    - "mean": Average across time
    - "topk_mean": Average of top-k activations per feature
    - "last": Only use last timestep (no pooling, for comparison)
    - "attention": Learned attention weights across timesteps

    Usage:
        # From SAE feature activations [batch, seq_len, n_features]
        probe = PooledProbe(n_features=2048, num_classes=4, pool="max")
        logits = probe(sae_activations)  # [batch, num_classes]

        # Training on labeled tech situations
        from interp.probes import ProbeTrainer, ProbeConfig

        tech_config = ProbeConfig(
            name="tech_option",
            num_classes=4,
            class_names=["tech_in_place", "tech_away", "tech_toward", "missed_tech"],
        )
        trainer = ProbeTrainer(probe, tech_config, device)
        history = trainer.train(pooled_features, labels)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        pool: str = "max",
        topk: int = 8,
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize pooled probe.

        Args:
            input_dim: Number of SAE features (hidden_dim of SAE)
            num_classes: Number of classes to predict
            pool: Pooling strategy - "max", "mean", "topk_mean", "last", "attention"
            topk: k value for topk_mean pooling
            hidden_dim: Optional MLP hidden layer (None = linear probe)
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pool = pool
        self.topk = topk

        # Attention weights if using attention pooling
        if pool == "attention":
            self.attention = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
        else:
            self.attention = None

        # Classifier head
        if hidden_dim is not None:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(input_dim, num_classes)

    def _pool(self, x: Tensor) -> Tensor:
        """
        Pool features across sequence dimension.

        Args:
            x: [batch, seq_len, n_features]

        Returns:
            [batch, n_features]
        """
        if self.pool == "max":
            return x.max(dim=1).values
        elif self.pool == "mean":
            return x.mean(dim=1)
        elif self.pool == "topk_mean":
            # Average of top-k activations per feature
            k = min(self.topk, x.shape[1])
            topk_vals = x.topk(k, dim=1).values  # [batch, k, n_features]
            return topk_vals.mean(dim=1)
        elif self.pool == "last":
            return x[:, -1, :]
        elif self.pool == "attention":
            # Learned attention weights
            attn_scores = self.attention(x).squeeze(-1)  # [batch, seq_len]
            attn_weights = F.softmax(attn_scores, dim=1)  # [batch, seq_len]
            return torch.einsum("bs,bsf->bf", attn_weights, x)
        else:
            raise ValueError(f"Unknown pooling mode: {self.pool}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: SAE activations [batch, seq_len, n_features]
               or pre-pooled [batch, n_features]

        Returns:
            Logits [batch, num_classes]
        """
        # Handle both 2D (pre-pooled) and 3D (needs pooling) input
        if x.dim() == 3:
            x = self._pool(x)

        return self.classifier(x)

    def predict(self, x: Tensor) -> Tensor:
        """Get predicted class indices."""
        logits = self.forward(x)
        return logits.argmax(dim=-1)

    def predict_proba(self, x: Tensor) -> Tensor:
        """Get class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def get_feature_importances(self) -> Tensor:
        """
        Get importance of each SAE feature for classification.

        For linear classifier, this is the L2 norm of weights per feature.

        Returns:
            Feature importances [n_features]
        """
        if isinstance(self.classifier, nn.Linear):
            # Shape: [num_classes, input_dim]
            weights = self.classifier.weight
            return weights.norm(dim=0)
        else:
            # For MLP, get first layer weights
            first_layer = self.classifier[0]
            if isinstance(first_layer, nn.Linear):
                return first_layer.weight.norm(dim=0)
            raise ValueError("Cannot extract importances from non-linear classifier")


@dataclass
class ProbeResults:
    """Results from probe evaluation."""

    accuracy: float
    per_class_accuracy: Dict[str, float]
    confusion_matrix: Tensor
    num_samples: int
    predictions: Optional[Tensor] = None
    probabilities: Optional[Tensor] = None

    def summary(self, config: ProbeConfig) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 50,
            f"PROBE RESULTS: {config.name}",
            "=" * 50,
            f"\nOverall Accuracy: {self.accuracy:.2%}",
            f"Samples: {self.num_samples}",
            "\nPer-class accuracy:",
        ]

        for class_name, acc in self.per_class_accuracy.items():
            lines.append(f"  {class_name}: {acc:.2%}")

        lines.append("\nConfusion Matrix:")
        # Header
        header = "         " + " ".join(f"{c[:8]:>8}" for c in config.class_names)
        lines.append(header)

        # Rows
        for i, row_name in enumerate(config.class_names):
            row_vals = " ".join(f"{int(v):>8}" for v in self.confusion_matrix[i])
            lines.append(f"{row_name[:8]:>8} {row_vals}")

        return "\n".join(lines)


@dataclass
class TrainingHistory:
    """Training history for a probe."""

    train_losses: List[float] = field(default_factory=list)
    train_accuracies: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    val_accuracies: List[float] = field(default_factory=list)
    best_val_accuracy: float = 0.0
    best_epoch: int = 0


class ProbeTrainer:
    """
    Trainer for linear/MLP probes.

    Handles training, evaluation, and checkpointing of probes.
    """

    def __init__(
        self,
        probe: Union[LinearProbe, MLPProbe],
        config: ProbeConfig,
        device: Optional[torch.device] = None,
    ):
        self.probe = probe
        self.config = config
        self.device = device or torch.device("cpu")
        self.probe.to(self.device)

    def train(
        self,
        train_activations: Tensor,
        train_labels: Tensor,
        val_activations: Optional[Tensor] = None,
        val_labels: Optional[Tensor] = None,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        early_stopping_patience: int = 10,
        class_weights: Optional[Tensor] = None,
        verbose: bool = True,
    ) -> TrainingHistory:
        """
        Train the probe on labeled activations.

        Args:
            train_activations: Training activations [N, dim]
            train_labels: Training labels [N]
            val_activations: Validation activations (optional)
            val_labels: Validation labels (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            weight_decay: L2 regularization
            early_stopping_patience: Stop if val acc doesn't improve
            class_weights: Optional class weights for imbalanced data
            verbose: Print progress

        Returns:
            TrainingHistory with loss/accuracy curves
        """
        # Create data loaders
        train_dataset = TensorDataset(
            train_activations.to(self.device),
            train_labels.to(self.device),
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            self.probe.parameters(), lr=lr, weight_decay=weight_decay
        )

        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Training history
        history = TrainingHistory()
        patience_counter = 0

        for epoch in range(epochs):
            # Training
            self.probe.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                logits = self.probe(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_x.size(0)
                train_correct += (logits.argmax(dim=-1) == batch_y).sum().item()
                train_total += batch_x.size(0)

            train_loss /= train_total
            train_acc = train_correct / train_total
            history.train_losses.append(train_loss)
            history.train_accuracies.append(train_acc)

            # Validation
            if val_activations is not None:
                self.probe.eval()
                with torch.no_grad():
                    val_x = val_activations.to(self.device)
                    val_y = val_labels.to(self.device)
                    val_logits = self.probe(val_x)
                    val_loss = criterion(val_logits, val_y).item()
                    val_acc = (val_logits.argmax(dim=-1) == val_y).float().mean().item()

                history.val_losses.append(val_loss)
                history.val_accuracies.append(val_acc)

                # Early stopping
                if val_acc > history.best_val_accuracy:
                    history.best_val_accuracy = val_acc
                    history.best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break

                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{epochs}: "
                        f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, "
                        f"val_loss={val_loss:.4f}, val_acc={val_acc:.2%}"
                    )
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{epochs}: "
                        f"train_loss={train_loss:.4f}, train_acc={train_acc:.2%}"
                    )

        return history

    def evaluate(
        self,
        activations: Tensor,
        labels: Tensor,
        return_predictions: bool = False,
    ) -> ProbeResults:
        """
        Evaluate probe on test data.

        Args:
            activations: Test activations [N, dim]
            labels: Test labels [N]
            return_predictions: Include predictions in results

        Returns:
            ProbeResults with accuracy and confusion matrix
        """
        self.probe.eval()

        with torch.no_grad():
            x = activations.to(self.device)
            y = labels.to(self.device)

            logits = self.probe(x)
            predictions = logits.argmax(dim=-1)
            probabilities = F.softmax(logits, dim=-1)

            # Overall accuracy
            accuracy = (predictions == y).float().mean().item()

            # Per-class accuracy
            per_class_accuracy = {}
            for i, class_name in enumerate(self.config.class_names):
                mask = y == i
                if mask.sum() > 0:
                    class_acc = (predictions[mask] == i).float().mean().item()
                else:
                    class_acc = 0.0
                per_class_accuracy[class_name] = class_acc

            # Confusion matrix
            confusion = torch.zeros(
                self.config.num_classes, self.config.num_classes,
                dtype=torch.long, device=self.device
            )
            for true_class in range(self.config.num_classes):
                for pred_class in range(self.config.num_classes):
                    confusion[true_class, pred_class] = (
                        (y == true_class) & (predictions == pred_class)
                    ).sum()

        return ProbeResults(
            accuracy=accuracy,
            per_class_accuracy=per_class_accuracy,
            confusion_matrix=confusion.cpu(),
            num_samples=len(labels),
            predictions=predictions.cpu() if return_predictions else None,
            probabilities=probabilities.cpu() if return_predictions else None,
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save probe weights and config."""
        path = Path(path)
        torch.save({
            "probe_state_dict": self.probe.state_dict(),
            "config": self.config.to_dict(),
            "input_dim": self.probe.input_dim,
            "num_classes": self.probe.num_classes,
            "probe_type": type(self.probe).__name__,
        }, path)

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        device: Optional[torch.device] = None,
    ) -> "ProbeTrainer":
        """Load probe from checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=device or "cpu")

        config = ProbeConfig.from_dict(checkpoint["config"])

        # Create probe of correct type
        probe_type = checkpoint.get("probe_type", "LinearProbe")
        if probe_type == "MLPProbe":
            probe = MLPProbe(
                input_dim=checkpoint["input_dim"],
                num_classes=checkpoint["num_classes"],
            )
        else:
            probe = LinearProbe(
                input_dim=checkpoint["input_dim"],
                num_classes=checkpoint["num_classes"],
            )

        probe.load_state_dict(checkpoint["probe_state_dict"])

        return cls(probe, config, device)


def compute_class_weights(labels: Tensor) -> Tensor:
    """
    Compute inverse frequency class weights for imbalanced data.

    Args:
        labels: Label tensor [N]

    Returns:
        Class weights tensor [num_classes]
    """
    unique, counts = torch.unique(labels, return_counts=True)
    weights = 1.0 / counts.float()
    weights = weights / weights.sum() * len(unique)  # Normalize

    # Handle missing classes
    num_classes = int(unique.max().item()) + 1
    full_weights = torch.ones(num_classes)
    for cls, weight in zip(unique, weights):
        full_weights[int(cls)] = weight

    return full_weights


def train_probe_on_cached_activations(
    activations: Tensor,
    labels: Tensor,
    config: ProbeConfig,
    val_split: float = 0.2,
    probe_type: str = "linear",
    device: Optional[torch.device] = None,
    **train_kwargs,
) -> Tuple[ProbeTrainer, TrainingHistory, ProbeResults]:
    """
    Convenience function to train and evaluate a probe.

    Args:
        activations: Activation tensor [N, dim]
        labels: Label tensor [N]
        config: Probe configuration
        val_split: Fraction for validation
        probe_type: "linear" or "mlp"
        device: Torch device
        **train_kwargs: Passed to ProbeTrainer.train()

    Returns:
        Tuple of (trainer, history, validation_results)
    """
    # Split data
    n_samples = len(activations)
    n_val = int(n_samples * val_split)
    indices = torch.randperm(n_samples)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_x, train_y = activations[train_idx], labels[train_idx]
    val_x, val_y = activations[val_idx], labels[val_idx]

    # Create probe
    input_dim = activations.shape[-1]
    if probe_type == "mlp":
        probe = MLPProbe(input_dim, config.num_classes)
    else:
        probe = LinearProbe(input_dim, config.num_classes)

    # Train
    trainer = ProbeTrainer(probe, config, device)

    # Compute class weights if imbalanced
    class_weights = compute_class_weights(train_y)

    history = trainer.train(
        train_x, train_y,
        val_x, val_y,
        class_weights=class_weights,
        **train_kwargs,
    )

    # Final evaluation
    results = trainer.evaluate(val_x, val_y, return_predictions=True)

    return trainer, history, results


__all__ = [
    "ProbeConfig",
    "MELEE_PROBES",
    "LinearProbe",
    "MLPProbe",
    "PooledProbe",
    "ProbeResults",
    "TrainingHistory",
    "ProbeTrainer",
    "compute_class_weights",
    "train_probe_on_cached_activations",
]
