"""
src/models/loss.py
------------------
Custom loss functions for multi-horizon attack forecasting under severe class imbalance:
  1. MultiHorizonWeightedBCEWithLogitsLoss: Horizon-specific or global positive class weighting.
  2. BinaryFocalLossWithLogits: Downweights easy well-classified negative examples to focus on hard attacks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def compute_pos_weight(
    targets: np.ndarray | torch.Tensor,
    horizon_specific: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute pos_weight = N_negative / N_positive strictly from training targets.

    Parameters
    ----------
    targets : np.ndarray | torch.Tensor
        Binary target matrix of shape (N, K) or (N,).
    horizon_specific : bool
        If True, returns pos_weight per horizon of shape (K,).
        If False, returns a scalar pos_weight of shape (1,).
    eps : float
        Epsilon to prevent division by zero.

    Returns
    -------
    torch.Tensor
        Calculated pos_weight tensor on CPU.
    """
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets).float()
    else:
        targets = targets.float()

    if targets.ndim == 1:
        n_pos = torch.sum(targets == 1.0).item()
        n_neg = torch.sum(targets == 0.0).item()
        w = float(n_neg) / float(max(n_pos, eps))
        return torch.tensor([w], dtype=torch.float32)

    # shape (N, K)
    N, K = targets.shape
    if horizon_specific:
        pos_weights = []
        for k in range(K):
            col = targets[:, k]
            n_pos = torch.sum(col == 1.0).item()
            n_neg = torch.sum(col == 0.0).item()
            w = float(n_neg) / float(max(n_pos, eps))
            pos_weights.append(w)
        return torch.tensor(pos_weights, dtype=torch.float32)
    else:
        n_pos = torch.sum(targets == 1.0).item()
        n_neg = torch.sum(targets == 0.0).item()
        w = float(n_neg) / float(max(n_pos, eps))
        return torch.tensor([w], dtype=torch.float32)


class MultiHorizonWeightedBCEWithLogitsLoss(nn.Module):
    """Multi-horizon BCE with logits loss supporting per-horizon positive weighting."""

    def __init__(
        self,
        pos_weight: torch.Tensor | np.ndarray | float | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.reduction = reduction
        if pos_weight is not None:
            if isinstance(pos_weight, (int, float)):
                pos_weight = torch.tensor([float(pos_weight)], dtype=torch.float32)
            elif isinstance(pos_weight, np.ndarray):
                pos_weight = torch.from_numpy(pos_weight).float()
            elif isinstance(pos_weight, torch.Tensor):
                pos_weight = pos_weight.float()
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate weighted BCE loss.

        Parameters
        ----------
        logits : torch.Tensor
            Model raw logits of shape (B, K).
        targets : torch.Tensor
            Binary targets of shape (B, K), values in {0, 1}.

        Returns
        -------
        torch.Tensor
            Scalar loss (if reduction == 'mean' or 'sum') or (B, K) tensor.
        """
        targets = targets.float()
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction=self.reduction
        )


class BinaryFocalLossWithLogits(nn.Module):
    """Numerically stable Binary Focal Loss with logits for class-imbalanced multi-horizon targets.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Implementation uses BCEWithLogits to prevent underflow/overflow:
      ce_loss = BCEWithLogits(logits, targets, reduction='none')
      p_t = exp(-ce_loss)
      focal_weight = alpha_t * (1 - p_t)^gamma
      loss = focal_weight * ce_loss
    """

    def __init__(
        self,
        alpha: float | torch.Tensor = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if isinstance(alpha, (int, float)):
            self.register_buffer("alpha", torch.tensor(float(alpha), dtype=torch.float32))
        elif isinstance(alpha, (np.ndarray, list)):
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float32))
        elif isinstance(alpha, torch.Tensor):
            self.register_buffer("alpha", alpha.float())
        else:
            self.register_buffer("alpha", torch.tensor(0.75, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Binary Focal Loss.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits of shape (B, K).
        targets : torch.Tensor
            Target labels of shape (B, K) with values in {0, 1}.

        Returns
        -------
        torch.Tensor
            Reduced or unreduced focal loss.
        """
        targets = targets.float()
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)

        # Alpha weighting: alpha for positive class (1), (1 - alpha) for negative class (0)
        alpha = self.alpha.to(logits.device)
        if alpha.ndim == 0 or (alpha.ndim == 1 and alpha.shape[0] == 1):
            alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        else:
            # Broadcast over horizons if alpha is shape (K,)
            alpha_t = alpha.unsqueeze(0) * targets + (1.0 - alpha.unsqueeze(0)) * (1.0 - targets)

        focal_weight = alpha_t * ((1.0 - p_t) ** self.gamma)
        loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
