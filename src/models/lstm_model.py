"""
src/models/lstm_model.py
------------------------
PyTorch LSTM baseline for multi-horizon attack forecasting.

Input: (batch_size, window_size=20, num_features=68)
Output:
  - Binary: (batch_size, forecast_horizon=5) logits / probabilities
  - Multiclass: (batch_size, forecast_horizon=5, num_classes=15) logits / probabilities
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
import torch
import torch.nn as nn
import numpy as np


class LSTMForecaster(nn.Module):
    """Multi-horizon LSTM sequence forecasting neural network."""

    def __init__(
        self,
        num_features: int = 68,
        hidden_dim: int = 128,
        num_layers: int = 2,
        forecast_horizon_k: int = 5,
        num_classes: int = 2,  # 2 for binary, 15 for multiclass
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.K = forecast_horizon_k
        self.num_classes = num_classes
        self.dropout = dropout

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Horizon projection heads: one head per forecast step or shared projected rollout
        # We use horizon-specific linear projections from the final recurrent state
        if num_classes == 2:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1),
                )
                for _ in range(self.K)
            ])
        else:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, num_classes),
                )
                for _ in range(self.K)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (B, W, D) = (batch_size, 20, 68).

        Returns
        -------
        torch.Tensor
            - If binary: (B, K)
            - If multiclass: (B, K, C)
        """
        out, (h_n, c_n) = self.lstm(x)
        # Use final step hidden representation
        last_hidden = out[:, -1, :]  # (B, hidden_dim)

        horizon_outputs = []
        for head in self.heads:
            horizon_outputs.append(head(last_hidden))

        if self.num_classes == 2:
            # (B, K)
            return torch.cat(horizon_outputs, dim=1)
        else:
            # (B, K, C)
            return torch.stack(horizon_outputs, dim=1)

    def predict_forecasting(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        """Get discrete class predictions for all K horizons."""
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 2:
                probs = torch.sigmoid(logits).cpu().numpy()
                return (probs >= 0.5).astype(int)
            else:
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                return preds

    def predict_proba_forecasting(self, x: torch.Tensor | np.ndarray) -> list[np.ndarray]:
        """Get probability matrices for all K horizons."""
        self.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        with torch.no_grad():
            logits = self.forward(x)
            if self.num_classes == 2:
                probs = torch.sigmoid(logits).cpu().numpy()  # (B, K)
                return [np.column_stack([1 - probs[:, k], probs[:, k]]) for k in range(self.K)]
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (B, K, C)
                return [probs[:, k, :] for k in range(self.K)]

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_type": "LSTMForecaster",
            "num_features": self.num_features,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "forecast_horizon_k": self.K,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
        }
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        torch.save(self.state_dict(), save_dir / "model.pt")

    @classmethod
    def load(cls, save_dir: str | Path, map_location: str = "cpu") -> LSTMForecaster:
        save_dir = Path(save_dir)
        with open(save_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        model = cls(
            num_features=meta["num_features"],
            hidden_dim=meta["hidden_dim"],
            num_layers=meta["num_layers"],
            forecast_horizon_k=meta["forecast_horizon_k"],
            num_classes=meta["num_classes"],
            dropout=meta["dropout"],
        )
        model.load_state_dict(torch.load(save_dir / "model.pt", map_location=map_location))
        model.eval()
        return model
