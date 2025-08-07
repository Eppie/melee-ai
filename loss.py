from __future__ import annotations

from typing import Literal
from typing import Sequence, Dict, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F


class SimpleHybridLoss(nn.Module):
    """
    BCE-with-logits for buttons  +  Smooth-L1 for sticks.

    Args
    ----
    bce_indices   : indices of the 5 button targets   (length 5)
    reg_indices   : indices of the 4 stick  targets   (length 4)
    pos_weight    : optional per-button positive weight tensor, shape [5]
    bce_weight    : scalar weight for the BCE term
    reg_weight    : scalar weight for the regression term
    """

    def __init__(
            self,
            bce_indices: Sequence[int],
            reg_indices: Sequence[int],
            *,
            pos_weight: Optional[Tensor] = None,
            bce_weight: float = 1.0,
            reg_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.bce_idx = list(bce_indices)
        self.reg_idx = list(reg_indices)
        self.register_buffer(
            "pos_weight",
            None if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32),
        )
        self.bce_weight = float(bce_weight)
        self.reg_weight = float(reg_weight)

    # ------------------------------------------------------------------ #
    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        """
        preds, targets : shape [B, 9] in TARGET_COLUMNS order
        returns scalar total loss
        """
        # Buttons
        bce = F.binary_cross_entropy_with_logits(
            preds[:, self.bce_idx], targets[:, self.bce_idx],
            pos_weight=self.pos_weight, reduction="mean"
        )

        # Sticks
        reg = F.smooth_l1_loss(
            preds[:, self.reg_idx], targets[:, self.reg_idx],
            beta=1.0, reduction="mean"
        )

        return self.bce_weight * bce + self.reg_weight * reg

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def components(self, preds: Tensor, targets: Tensor) -> Dict[str, float]:
        """Return {'bce':…, 'reg':…, 'total':…} for logging."""
        bce = F.binary_cross_entropy_with_logits(
            preds[:, self.bce_idx], targets[:, self.bce_idx],
            pos_weight=self.pos_weight, reduction="mean"
        ).item()

        reg = F.smooth_l1_loss(
            preds[:, self.reg_idx], targets[:, self.reg_idx],
            beta=1.0, reduction="mean"
        ).item()

        return {"bce": bce, "reg": reg, "total": self.bce_weight * bce + self.reg_weight * reg}


def _focal_bce_with_logits(
        logits: Tensor,
        targets: Tensor,
        *,
        pos_weight: Optional[Tensor] = None,  # shape [C] or None
        alpha: Optional[Tensor] = None,  # weight for positives per label in [0,1]; shape [C] or scalar
        gamma: float = 0.0,  # 0 -> plain BCE, >0 -> focal
) -> Tensor:
    """
    Numerically-stable BCE with logits + optional focal modulation and alpha balancing.
    Returns mean over batch and labels.
    """
    # Base BCE (per-element), with correct positive-class weighting
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )

    if gamma > 0.0:
        # p_t = p for y=1, 1-p for y=0
        p = torch.sigmoid(logits)
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        mod = (1.0 - p_t).clamp_min(1e-4).pow(gamma)
        bce = bce * mod

    if alpha is not None:
        # α for positives; (1-α) for negatives. Broadcast to [B, C]
        alpha = torch.as_tensor(alpha, device=logits.device, dtype=logits.dtype)
        alpha_pos = alpha
        alpha_neg = 1.0 - alpha
        alpha_factor = targets * alpha_pos + (1.0 - targets) * alpha_neg
        bce = bce * alpha_factor

    return bce.mean()


class BalancedHybridLoss(nn.Module):
    """
    Hybrid classification+regression loss with optional gating derived from regression targets.

    Classification:
      • BCE-with-logits (optionally focal), with per-label pos_weight/alpha.

    Regression:
      • Smooth L1 (Huber) per-dimension, optional per-target scaling.
      • Optional gating driven by how far each regression dimension is from a rest value:
          - gate_source='target' (recommended): use targets to determine activity.
          - gate_source='pred':    use predictions to determine activity.
          - gate_type='hard':      1{ |v - rest_value| >= deadzone } per dim.
          - gate_type='soft':      weight grows smoothly with distance beyond deadzone.
        Regression loss is a weighted mean over regression dimensions:
            sum(reg_loss * weight) / max(sum(weight), 1).

    Args:
        bce_indices: column indices of binary targets.
        reg_indices: column indices of regression targets.
        pos_weight:  per-label positive weight for BCE, shape [#bce] (optional).
        alpha:       per-label alpha in [0,1] for focal-BCE (optional).
        focal_gamma: focal loss gamma >= 0 (0 disables focal).
        reg_beta:    Smooth L1 (Huber) transition point.
        reg_scale:   per-regression weight (e.g., inverse std), shape [#reg] (optional).
        bce_group_weight, reg_group_weight: scalar weights for the two groups.

        # Gating from regression channels:
        enable_gating:  Enable/disable regression gating.
        gate_source:    'target' or 'pred' (where to read values for gating).
        gate_type:      'hard' or 'soft' (how to convert distance to weights).
        rest_value:     Resting value for regression dimensions (default 0.5).
        deadzone:       Threshold around rest for 'inactive' (e.g., 0.05).
        soft_floor:     Minimum weight for soft gating (e.g., 0.1 keeps tiny gradients).
        clamp_for_gating: Clamp source values to [0,1] before measuring distance.
    """

    def __init__(
            self,
            bce_indices: Sequence[int],
            reg_indices: Sequence[int],
            *,
            pos_weight: Optional[Tensor] = None,
            alpha: Optional[Tensor] = None,
            focal_gamma: float = 0.0,
            reg_beta: float = 0.02,
            reg_scale: Optional[Tensor] = None,
            bce_group_weight: float = 1.0,
            reg_group_weight: float = 1.0,
            # gating params:
            enable_gating: bool = True,
            gate_source: Literal["target", "pred"] = "target",
            gate_type: Literal["hard", "soft"] = "hard",
            rest_value: float = 0.5,
            deadzone: float = 0.05,
            soft_floor: float = 0.0,
            clamp_for_gating: bool = True,
    ) -> None:
        super().__init__()
        self.register_buffer("pos_weight",
                             None if pos_weight is None else torch.as_tensor(pos_weight, dtype=torch.float32))
        self.register_buffer("alpha",
                             None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))
        self.register_buffer("reg_scale",
                             None if reg_scale is None else torch.as_tensor(reg_scale, dtype=torch.float32))

        self.bce_idx = list(bce_indices)
        self.reg_idx = list(reg_indices)
        self.focal_gamma = float(focal_gamma)
        self.reg_beta = float(reg_beta)
        self.bce_group_weight = float(bce_group_weight)
        self.reg_group_weight = float(reg_group_weight)

        # gating controls (from regression targets/preds)
        self.enable_gating = bool(enable_gating)
        self.gate_source = gate_source
        self.gate_type = gate_type
        self.rest_value = float(rest_value)
        self.deadzone = float(deadzone)
        self.soft_floor = float(soft_floor)
        self.clamp_for_gating = bool(clamp_for_gating)

        if not (0.0 <= self.rest_value <= 1.0):
            raise ValueError("rest_value should be in [0, 1].")
        if not (0.0 <= self.deadzone <= 0.5):
            raise ValueError("deadzone should be in [0, 0.5].")
        if self.gate_type == "soft" and self.soft_floor < 0.0:
            raise ValueError("soft_floor must be >= 0.")

    def forward(self, preds: Tensor, targets: Tensor) -> Tensor:
        loss_bce = preds.new_tensor(0.0)
        loss_reg = preds.new_tensor(0.0)

        # ----- Classification (BCE/focal) -----
        if self.bce_idx:
            logits = preds[:, self.bce_idx]  # [B, C_bce]
            tgt_bce = targets[:, self.bce_idx]
            loss_bce = _focal_bce_with_logits(
                logits, tgt_bce, pos_weight=self.pos_weight, alpha=self.alpha, gamma=self.focal_gamma
            )  # scalar

        # ----- Regression (Smooth L1 / Huber), gated from regression channels -----
        if self.reg_idx:
            p = preds[:, self.reg_idx]  # [B, C_reg]
            t = targets[:, self.reg_idx]  # [B, C_reg]

            reg = F.smooth_l1_loss(p, t, beta=self.reg_beta, reduction="none")  # [B, C_reg]
            if self.reg_scale is not None:
                reg = reg * self.reg_scale  # broadcast [C_reg] -> [B, C_reg]

            if self.enable_gating:
                # choose source for gating
                src = t if self.gate_source == "target" else p
                if self.clamp_for_gating:
                    src = src.clamp(0.0, 1.0)

                # distance from rest per element
                delta = (src - self.rest_value).abs()  # [B, C_reg]

                if self.gate_type == "hard":
                    w = (delta >= self.deadzone).to(reg.dtype)  # [B, C_reg] in {0,1}
                else:
                    # Soft weight grows with distance beyond deadzone, normalized to [0,1]
                    max_dev = max(self.rest_value, 1.0 - self.rest_value)  # e.g., 0.5 if rest=0.5
                    denom = max(1e-6, (max_dev - self.deadzone))
                    w = ((delta - self.deadzone) / denom).clamp(0.0, 1.0).to(reg.dtype)
                    if self.soft_floor > 0.0:
                        w = torch.clamp(w, min=self.soft_floor)

                numer = (reg * w).sum()
                denom_w = w.sum().clamp_min(1.0)
                loss_reg = numer / denom_w
            else:
                loss_reg = reg.mean()

        return self.bce_group_weight * loss_bce + self.reg_group_weight * loss_reg

    @torch.no_grad()
    def components(self, preds: Tensor, targets: Tensor) -> dict[str, float]:
        """Optional helper for logging the two parts separately.
        Mirrors gating behavior from `forward` so 'reg' matches the trained objective.
        """
        out: dict[str, float] = {}

        # ----- Classification (BCE/focal) -----
        if self.bce_idx:
            logits = preds[:, self.bce_idx]
            tgt_bce = targets[:, self.bce_idx]
            out["bce"] = _focal_bce_with_logits(
                logits, tgt_bce, pos_weight=self.pos_weight, alpha=self.alpha, gamma=self.focal_gamma
            ).item()

        # ----- Regression (Smooth L1 / Huber), with optional gating from regression channels -----
        if self.reg_idx:
            p = preds[:, self.reg_idx]  # [B, C_reg]
            t = targets[:, self.reg_idx]  # [B, C_reg]
            reg = F.smooth_l1_loss(p, t, beta=self.reg_beta, reduction="none")  # [B, C_reg]
            if self.reg_scale is not None:
                reg = reg * self.reg_scale  # broadcast [C_reg] -> [B, C_reg]

            if self.enable_gating:
                # Source for gating: targets (recommended) or predictions
                src = t if self.gate_source == "target" else p
                if self.clamp_for_gating:
                    src = src.clamp(0.0, 1.0)

                # Distance from rest per element
                delta = (src - self.rest_value).abs()  # [B, C_reg]

                if self.gate_type == "hard":
                    w = (delta >= self.deadzone).to(reg.dtype)  # {0,1}
                else:
                    max_dev = max(self.rest_value, 1.0 - self.rest_value)  # typically 0.5 when rest=0.5
                    denom = max(1e-6, (max_dev - self.deadzone))
                    w = ((delta - self.deadzone) / denom).clamp(0.0, 1.0).to(reg.dtype)
                    if self.soft_floor > 0.0:
                        w = torch.clamp(w, min=self.soft_floor)

                numer = (reg * w).sum()
                denom_w = w.sum().clamp_min(1.0)
                reg_mean = (numer / denom_w).item()
            else:
                reg_mean = reg.mean().item()

            out["reg"] = reg_mean

        out["total"] = self.bce_group_weight * out.get("bce", 0.0) + self.reg_group_weight * out.get("reg", 0.0)
        return out
