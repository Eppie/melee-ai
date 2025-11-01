"""
Imitation learning strategies for sample weighting
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod


class ImitationStrategy(ABC):
    """Base class for imitation learning strategies"""

    @abstractmethod
    def compute_weights(self, values: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute sample weights for imitation learning

        Args:
            values: (B, L) or (B, L, 1) predicted values
            **kwargs: strategy-specific arguments

        Returns:
            weights: (B, L) sample weights
        """
        pass


class UniformStrategy(ImitationStrategy):
    """Baseline: uniform weights for all samples"""

    def compute_weights(self, values: torch.Tensor, **kwargs) -> torch.Tensor:
        if values.dim() == 3:
            values = values.squeeze(-1)
        values = values.float()
        return torch.ones_like(values)


class ValueWeightedStrategy(ImitationStrategy):
    """Weight samples by sigmoid-transformed value relative to median"""

    def __init__(self, k=1.0, temperature=1.0, use_exp=False):
        """
        Args:
            k: scaling factor for value difference
            temperature: softmax temperature for normalization
            use_exp: if True, use exp weighting instead of sigmoid
        """
        self.k = k
        self.temperature = temperature
        self.use_exp = use_exp

    def compute_weights(self, values: torch.Tensor, **kwargs) -> torch.Tensor:
        if values.dim() == 3:
            values = values.squeeze(-1)  # (B, L)

        # Ensure float dtype
        values = values.float()

        # Compute median per batch
        B, L = values.shape
        values_flat = values.reshape(B * L)
        median = values_flat.median()

        # Compute weights
        v_centered = self.k * (values - median)

        if self.use_exp:
            weights = torch.exp(v_centered / self.temperature)
        else:
            weights = torch.sigmoid(v_centered / self.temperature)

        # Normalize per batch
        weights = weights / (weights.mean(dim=1, keepdim=True) + 1e-8)

        return weights


class ValueAdvantageStrategy(ImitationStrategy):
    """Weight samples by advantage: V_{t+n} - V_t"""

    def __init__(self, n_steps=5, alpha=1.0, use_gae=False, gamma=0.99, lambda_=0.95):
        """
        Args:
            n_steps: look-ahead window
            alpha: exponent for advantage (higher = more emphasis on large advantages)
            use_gae: if True, use GAE instead of simple n-step advantage
            gamma: discount factor for GAE
            lambda_: GAE lambda parameter
        """
        self.n_steps = n_steps
        self.alpha = alpha
        self.use_gae = use_gae
        self.gamma = gamma
        self.lambda_ = lambda_

    def compute_weights(self, values: torch.Tensor, **kwargs) -> torch.Tensor:
        if values.dim() == 3:
            values = values.squeeze(-1)  # (B, L)

        # Ensure float dtype
        values = values.float()

        B, L = values.shape

        if self.use_gae:
            # Compute GAE advantages
            advantages = self._compute_gae(values)
        else:
            # Simple n-step advantage
            # Pad values for lookahead
            values_padded = F.pad(values, (0, self.n_steps), value=0.0)
            future_values = values_padded[:, self.n_steps :]
            advantages = future_values - values

        # Apply ReLU and power
        weights = torch.clamp(advantages, min=0.0) ** self.alpha

        # Normalize per batch
        weights = weights / (weights.mean(dim=1, keepdim=True) + 1e-8)

        return weights

    def _compute_gae(self, values: torch.Tensor) -> torch.Tensor:
        """Compute Generalized Advantage Estimation"""
        B, L = values.shape
        advantages = torch.zeros_like(values)

        # Assume rewards are value differences (TD residuals)
        rewards = torch.zeros_like(values)
        rewards[:, :-1] = values[:, 1:] - values[:, :-1]

        gae = 0
        for t in reversed(range(L)):
            if t == L - 1:
                next_value = 0
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + self.gamma * next_value - values[:, t]
            gae = delta + self.gamma * self.lambda_ * gae
            advantages[:, t] = gae

        return advantages


class ValueFilterStrategy(ImitationStrategy):
    """Filter out samples below a value percentile"""

    def __init__(self, percentile=50.0, soft=False, temperature=1.0):
        """
        Args:
            percentile: percentile threshold (0-100)
            soft: if True, use soft filtering with sigmoid
            temperature: temperature for soft filtering
        """
        self.percentile = percentile
        self.soft = soft
        self.temperature = temperature

    def compute_weights(self, values: torch.Tensor, **kwargs) -> torch.Tensor:
        if values.dim() == 3:
            values = values.squeeze(-1)

        # Ensure float dtype
        values = values.float()

        # Compute threshold per batch
        B, L = values.shape
        values_flat = values.reshape(B * L)
        threshold = torch.quantile(values_flat, self.percentile / 100.0)

        if self.soft:
            # Soft filtering with sigmoid
            weights = torch.sigmoid((values - threshold) / self.temperature)
        else:
            # Hard filtering
            weights = (values >= threshold).float()

        # Normalize per batch (avoid division by zero)
        weights = weights / (weights.mean(dim=1, keepdim=True) + 1e-8)

        return weights


class HybridStrategy(ImitationStrategy):
    """Combine multiple strategies"""

    def __init__(self, strategies: list, weights: list = None):
        """
        Args:
            strategies: list of ImitationStrategy instances
            weights: list of weights for each strategy (default: uniform)
        """
        self.strategies = strategies
        if weights is None:
            weights = [1.0 / len(strategies)] * len(strategies)
        self.weights = weights

    def compute_weights(self, values: torch.Tensor, **kwargs) -> torch.Tensor:
        # Compute weights from each strategy
        all_weights = []
        for strategy in self.strategies:
            w = strategy.compute_weights(values, **kwargs)
            all_weights.append(w)

        # Combine with weighted sum
        combined = torch.zeros_like(all_weights[0])
        for w, weight_factor in zip(all_weights, self.weights):
            combined += weight_factor * w

        # Normalize
        combined = combined / (combined.mean(dim=(-1), keepdim=True) + 1e-8)

        return combined
