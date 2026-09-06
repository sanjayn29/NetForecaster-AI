"""
src/models/transformer_model.py
-------------------------------
Transformer Sequence Encoder for Multi-Horizon Network Attack Forecasting.

Architecture:
  Input: (B, 20, 68)
  -> Linear feature projection: 68 -> d_model (default 128)
  -> Positional Encoding: Sinusoidal or learnable temporal encodings
  -> Transformer Encoder: 2 layers, 4 heads, dim_feedforward=256, dropout=0.1
  -> Temporal representation H in R^(B, 20, d_model)
  -> Final causal step representation H[:, -1, :] in R^(B, d_model)
  -> Prediction head -> 5 logits (T+1 ... T+5)

Guarantees & Properties:
  - Full self-attention across the 20 observed past timesteps in the sliding window.
  - Zero access to future forecast horizons (T+1 ... T+5) or external timesteps.
  - Attention extraction method for diagnostic interpretability.
  - Reusable positional encoding module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal temporal positional encoding module."""

    def __init__(self, d_model: int = 128, max_len: int = 100, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (B, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Tensor of shape (B, seq_len, d_model).
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x)


class TransformerEncoderWithAttention(nn.Module):
    """Transformer Encoder layer stack that supports extracting self-attention weights."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass through encoder layers."""
        out = x
        for layer in self.layers:
            out = layer(out)
        return self.norm(out)

    def extract_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Extract self-attention map from the first / last attention layer for diagnostics.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (B, seq_len, d_model).

        Returns
        -------
        torch.Tensor
            Attention map of shape (B, nhead, seq_len, seq_len) or (B, seq_len, seq_len).
        """
        # Multi-head attention computation on last layer's input
        out = x
        for layer in self.layers[:-1]:
            out = layer(out)

        last_layer = self.layers[-1]
        norm_x = last_layer.norm1(out) if hasattr(last_layer, "norm1") else out
        attn_out, attn_weights = last_layer.self_attn(
            norm_x, norm_x, norm_x, need_weights=True, average_attn_weights=False
        )
        return attn_weights


class TransformerForecaster(nn.Module):
    """Multi-Horizon Attack Forecaster using Transformer Sequence Encoder."""

    def __init__(
        self,
        num_features: int = 68,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        forecast_horizon_k: int = 5,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.K = forecast_horizon_k
        self.num_classes = num_classes
        self.dropout = dropout

        # Feature projection
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=100, dropout=dropout)

        # Transformer Encoder
        self.encoder = TransformerEncoderWithAttention(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Multi-horizon forecasting prediction heads
        if num_classes == 2:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1),
                )
                for _ in range(self.K)
            ])
        else:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, num_classes),
                )
                for _ in range(self.K)
            ])

    def get_temporal_representation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract full encoded sequence H and final timestep state vector.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence (B, W, D) = (B, 20, 68).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            H: (B, 20, d_model), last_step: (B, d_model)
        """
        proj = self.input_projection(x)  # (B, 20, d_model)
        pos_emb = self.pos_encoder(proj)  # (B, 20, d_model)
        h = self.encoder(pos_emb)  # (B, 20, d_model)
        last_step = h[:, -1, :]  # (B, d_model)
        return h, last_step

    def get_attention_weights(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        """Extract multi-head attention matrix for diagnostic visualization.

        Parameters
        ----------
        x : torch.Tensor | np.ndarray
            Input sequence (B, 20, 68) or (20, 68).

        Returns
        -------
        np.ndarray
            Attention weights array (B, nhead, 20, 20).
        """
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            proj = self.input_projection(x)
            pos_emb = self.pos_encoder(proj)
            attn = self.encoder.extract_attention(pos_emb)
            return attn.cpu().numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 20, 68).

        Returns
        -------
        torch.Tensor
            If binary: (B, 5) logits.
            If multiclass: (B, 5, C) logits.
        """
        _, last_step = self.get_temporal_representation(x)
        horizon_logits = [head(last_step) for head in self.heads]

        if self.num_classes == 2:
            return torch.cat(horizon_logits, dim=1)
        else:
            return torch.stack(horizon_logits, dim=1)

    def predict_forecasting(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        """Discrete class predictions for all K horizons."""
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 2:
                probs = torch.sigmoid(logits).cpu().numpy()
                return (probs >= 0.5).astype(int)
            else:
                return torch.argmax(logits, dim=-1).cpu().numpy()

    def predict_proba_forecasting(self, x: torch.Tensor | np.ndarray) -> list[np.ndarray]:
        """Probability matrices for all K horizons."""
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 2:
                probs = torch.sigmoid(logits).cpu().numpy()  # (B, K)
                return [np.column_stack([1.0 - probs[:, k], probs[:, k]]) for k in range(self.K)]
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (B, K, C)
                return [probs[:, k, :] for k in range(self.K)]

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_type": "TransformerForecaster",
            "num_features": self.num_features,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "dim_feedforward": self.dim_feedforward,
            "forecast_horizon_k": self.K,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
        }
        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        torch.save(self.state_dict(), save_dir / "best_model.pt")

    @classmethod
    def load(cls, save_dir: str | Path, map_location: str = "cpu") -> TransformerForecaster:
        save_dir = Path(save_dir)
        config_path = save_dir / "config.json"
        if not config_path.exists():
            config_path = save_dir / "metadata.json"
        with open(config_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model = cls(
            num_features=meta["num_features"],
            d_model=meta["d_model"],
            nhead=meta["nhead"],
            num_layers=meta["num_layers"],
            dim_feedforward=meta["dim_feedforward"],
            forecast_horizon_k=meta["forecast_horizon_k"],
            num_classes=meta["num_classes"],
            dropout=meta.get("dropout", 0.1),
        )
        weights_path = save_dir / "best_model.pt"
        if not weights_path.exists():
            weights_path = save_dir / "model.pt"
        model.load_state_dict(torch.load(weights_path, map_location=map_location))
        model.eval()
        return model
