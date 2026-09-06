"""
src/models/world_model.py
-------------------------
Latent Network State World Model for NetForecaster AI (Phase 4).

Architecture:
  Observed Traffic X_t in R^(B, 20, 68)
        |
  Temporal Causal Encoder (TCN Backbone)
        |
  Initial Latent State S_t in R^(B, 64)
        |
  Recursive Latent State Rollout:
    S_1 = S_0 + Transition(S_0)
    S_2 = S_1 + Transition(S_1)
    S_3 = S_2 + Transition(S_2)
    S_4 = S_3 + Transition(S_3)
    S_5 = S_4 + Transition(S_4)
        |
  Decoders at each step k in {1..5}:
    - Attack Probability Head: S_k -> logit_k in R^(B, 1)
    - State Reconstruction Head: S_k -> x_hat_k in R^(B, 68)

Guarantees:
  - Strict zero-future-leakage: actual future features are NEVER injected during rollout.
  - Rollout is purely autoregressive in latent space.
  - Multi-task training support (Weighted BCE + Feature Reconstruction + State Regularization).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tcn_model import TemporalConvNet
from src.models.state_transition import ResidualStateTransition


class LatentNetworkWorldModel(nn.Module):
    """
    Latent Network State World Model with Recursive K-Step Forecasting.
    """

    def __init__(
        self,
        num_features: int = 68,
        latent_dim: int = 64,
        encoder_channels: Optional[List[int]] = None,
        encoder_dilations: Optional[List[int]] = None,
        kernel_size: int = 3,
        transition_hidden_dim: int = 128,
        forecast_horizon_k: int = 5,
        dropout: float = 0.1,
        enable_reconstruction: bool = True,
    ) -> None:
        super().__init__()
        if encoder_channels is None:
            encoder_channels = [64, 64, 64, 64]
        if encoder_dilations is None:
            encoder_dilations = [1, 2, 4, 8]

        self.num_features = num_features
        self.latent_dim = latent_dim
        self.encoder_channels = encoder_channels
        self.encoder_dilations = encoder_dilations
        self.kernel_size = kernel_size
        self.transition_hidden_dim = transition_hidden_dim
        self.forecast_horizon_k = forecast_horizon_k
        self.dropout = dropout
        self.enable_reconstruction = enable_reconstruction

        # 1. Temporal Encoder (Causal TCN Backbone)
        self.encoder = TemporalConvNet(
            num_inputs=num_features,
            num_channels=encoder_channels,
            kernel_size=kernel_size,
            dilations=encoder_dilations,
            dropout=dropout,
        )

        # 2. Initial Latent State Projector
        encoder_out_dim = encoder_channels[-1]
        self.latent_projector = nn.Sequential(
            nn.Linear(encoder_out_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )

        # 3. Residual Latent State Transition Network: S_{t+1} = S_t + f(S_t)
        self.transition = ResidualStateTransition(
            latent_dim=latent_dim,
            hidden_dim=transition_hidden_dim,
            dropout=dropout,
        )

        # 4. Attack Probability Decoder Head (Binary Logits)
        self.attack_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # 5. State Feature Reconstruction Head (Reconstructs next 68-d flow vector)
        self.recon_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_features),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode historical flow window X_t into initial latent state S_t.

        Args:
            x: Input tensor of shape (B, W, num_features) or (B, num_features, W)

        Returns:
            Initial latent state S_0 of shape (B, latent_dim)
        """
        if x.dim() == 3 and x.shape[1] != self.num_features and x.shape[2] == self.num_features:
            # (B, W, D) -> (B, D, W)
            x = x.transpose(1, 2)

        # Causal TCN encoding: (B, D_out, W)
        h = self.encoder(x)
        # Final causal timestep representation: (B, D_out)
        h_final = h[:, :, -1]
        # Latent state: (B, latent_dim)
        s_0 = self.latent_projector(h_final)
        return s_0

    def rollout_from_state(
        self,
        s_0: torch.Tensor,
        horizon: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform recursive autoregressive rollout in latent space starting from S_0.

        Args:
            s_0: Initial latent state (B, latent_dim)
            horizon: Number of recursive steps (defaults to forecast_horizon_k)

        Returns:
            Dictionary containing:
              - 'logits': (B, horizon) attack logits
              - 'states': (B, horizon, latent_dim) rolled-out latent states
              - 'reconstructions': (B, horizon, num_features) reconstructed flow features
              - 'initial_state': (B, latent_dim)
        """
        k_steps = horizon or self.forecast_horizon_k
        current_state = s_0

        logits_list: List[torch.Tensor] = []
        states_list: List[torch.Tensor] = []
        recons_list: List[torch.Tensor] = []

        for _ in range(k_steps):
            # Step transition: S_{k} = S_{k-1} + Transition(S_{k-1})
            next_state = self.transition(current_state)
            states_list.append(next_state)

            # Decode attack logit: (B, 1)
            logit_k = self.attack_head(next_state)
            logits_list.append(logit_k)

            # Decode reconstructed feature vector: (B, num_features)
            recon_k = self.recon_head(next_state)
            recons_list.append(recon_k)

            # Advance state for next recursion
            current_state = next_state

        logits = torch.cat(logits_list, dim=-1)  # (B, k_steps)
        states = torch.stack(states_list, dim=1)  # (B, k_steps, latent_dim)
        reconstructions = torch.stack(recons_list, dim=1)  # (B, k_steps, num_features)

        return {
            "logits": logits,
            "states": states,
            "reconstructions": reconstructions,
            "initial_state": s_0,
        }

    def forward(
        self,
        x: torch.Tensor,
        horizon: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        End-to-End Forward Pass:
          X_t -> S_0 -> Recursive Rollout -> (logits, states, reconstructions)
        """
        s_0 = self.encode(x)
        return self.rollout_from_state(s_0, horizon=horizon)

    @torch.no_grad()
    def predict_proba(
        self,
        x: Union[np.ndarray, torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Inference method returning calibrated attack probabilities across all K horizons.

        Args:
            x: Input sequences (B, 20, 68)
            device: Optional torch compute device

        Returns:
            Probabilities array of shape (B, K)
        """
        self.eval()
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x.float()

        if device is not None:
            x_tensor = x_tensor.to(device)

        out = self.forward(x_tensor)
        probs = torch.sigmoid(out["logits"])
        return probs.cpu().numpy()

    def get_config(self) -> Dict[str, Any]:
        """Return model architecture configuration dictionary."""
        return {
            "num_features": self.num_features,
            "latent_dim": self.latent_dim,
            "encoder_channels": self.encoder_channels,
            "encoder_dilations": self.encoder_dilations,
            "kernel_size": self.kernel_size,
            "transition_hidden_dim": self.transition_hidden_dim,
            "forecast_horizon_k": self.forecast_horizon_k,
            "dropout": self.dropout,
            "enable_reconstruction": self.enable_reconstruction,
            "receptive_field": self.encoder.receptive_field,
        }

    def save_pretrained(self, save_dir: Union[str, Path]) -> Path:
        """Save model weights and config."""
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)

        config = self.get_config()
        with open(path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        torch.save(self.state_dict(), path / "best_model.pt")
        return path

    @classmethod
    def from_pretrained(cls, save_dir: Union[str, Path], device: Optional[torch.device] = None) -> LatentNetworkWorldModel:
        """Load pretrained model from directory."""
        path = Path(save_dir)
        with open(path / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        config_keys = [
            "num_features",
            "latent_dim",
            "encoder_channels",
            "encoder_dilations",
            "kernel_size",
            "transition_hidden_dim",
            "forecast_horizon_k",
            "dropout",
            "enable_reconstruction",
        ]
        init_kwargs = {k: config[k] for k in config_keys if k in config}
        model = cls(**init_kwargs)

        weights_path = path / "best_model.pt"
        map_location = device if device is not None else torch.device("cpu")
        state_dict = torch.load(weights_path, map_location=map_location, weights_only=True)
        model.load_state_dict(state_dict)

        if device is not None:
            model.to(device)
        model.eval()
        return model
