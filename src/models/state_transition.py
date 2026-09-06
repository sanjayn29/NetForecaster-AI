"""
NetForecaster AI — Latent State Transition Network (Phase 4)
Residual MLP modeling step-by-step latent network state dynamics:
S_{t+1} = S_t + TransitionMLP(S_t)
"""

from typing import Dict, Any, Optional
import torch
import torch.nn as nn


class ResidualStateTransition(nn.Module):
    """
    Residual Latent State Transition Network:
    Maps S_t in R^latent_dim to Delta S_t in R^latent_dim,
    producing S_{t+1} = S_t + Delta S_t.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout_p = dropout

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute one-step latent state transition.

        Args:
            state: Latent state tensor of shape (B, latent_dim)

        Returns:
            Next latent state tensor of shape (B, latent_dim)
        """
        delta = self.net(state)
        return state + delta

    def get_config(self) -> Dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout_p,
        }
