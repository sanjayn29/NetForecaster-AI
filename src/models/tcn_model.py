"""
src/models/tcn_model.py
-----------------------
Causal Temporal Convolutional Network (TCN) for multi-horizon attack forecasting.

Architecture:
  Input: (B, 20, 68) -> Transpose to (B, 68, 20)
  -> Input 1x1 projection
  -> Residual Causal Blocks with dilations [1, 2, 4, 8]
  -> Temporal representation H in R^(B, 20, hidden_dim)
  -> Final causal step representation H[:, -1, :] in R^(B, hidden_dim)
  -> Prediction head -> 5 logits (T+1 ... T+5)

Guarantees:
  - Strict temporal causality: output at step t never receives information from steps > t.
  - Receptive field calculation: covers full 20-step observed sequence.
  - Binary and multiclass forecasting support.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Chomp1d(nn.Module):
    """Trims trailing right-side padding to enforce strict causality."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Residual Causal Convolutional Block with dilation, normalization, and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.norm1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.norm2, self.relu2, self.dropout2,
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Stack of dilated residual causal blocks forming the core TCN encoder."""

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        num_levels = len(num_channels)
        if dilations is None:
            dilations = [2 ** i for i in range(num_levels)]

        for i in range(num_levels):
            dilation_size = dilations[i]
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.num_channels = num_channels

    @property
    def receptive_field(self) -> int:
        """Calculate the theoretical receptive field size in timesteps."""
        # Each block contains 2 convolutions
        return 1 + sum(2 * (self.kernel_size - 1) * d for d in self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: (B, C, W), Output: (B, C_out, W)."""
        return self.network(x)


class TCNForecaster(nn.Module):
    """End-to-End Causal TCN Multi-Horizon Attack Forecaster."""

    def __init__(
        self,
        num_features: int = 68,
        num_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        dilations: Optional[List[int]] = None,
        forecast_horizon_k: int = 5,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_channels is None:
            num_channels = [64, 64, 64, 64]
        if dilations is None:
            dilations = [1, 2, 4, 8]

        self.num_features = num_features
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.K = forecast_horizon_k
        self.num_classes = num_classes
        self.dropout = dropout

        self.tcn = TemporalConvNet(
            num_inputs=num_features,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )

        final_dim = num_channels[-1]

        # Multi-horizon forecasting prediction heads
        if num_classes == 2:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(final_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1),
                )
                for _ in range(self.K)
            ])
        else:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(final_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, num_classes),
                )
                for _ in range(self.K)
            ])

    @property
    def receptive_field(self) -> int:
        return self.tcn.receptive_field

    def get_temporal_representation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract full temporal representation H and final causal vector.

        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (B, W, D) = (B, 20, 68).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            H: (B, W, C_out), last_step: (B, C_out)
        """
        # Transpose (B, W, D) -> (B, D, W)
        x_trans = x.transpose(1, 2)
        h = self.tcn(x_trans)  # (B, C_out, W)
        h_seq = h.transpose(1, 2)  # (B, W, C_out)
        last_step = h_seq[:, -1, :]  # (B, C_out)
        return h_seq, last_step

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
            # (B, K)
            return torch.cat(horizon_logits, dim=1)
        else:
            # (B, K, C)
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
                preds = torch.argmax(logits, dim=-1).cpu().numpy()
                return preds

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
            "model_type": "TCNForecaster",
            "num_features": self.num_features,
            "num_channels": self.num_channels,
            "kernel_size": self.kernel_size,
            "dilations": self.dilations,
            "forecast_horizon_k": self.K,
            "num_classes": self.num_classes,
            "dropout": self.dropout,
            "receptive_field": self.receptive_field,
        }
        with open(save_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        torch.save(self.state_dict(), save_dir / "best_model.pt")

    @classmethod
    def load(cls, save_dir: str | Path, map_location: str = "cpu") -> TCNForecaster:
        save_dir = Path(save_dir)
        config_path = save_dir / "config.json"
        if not config_path.exists():
            config_path = save_dir / "metadata.json"
        with open(config_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        model = cls(
            num_features=meta["num_features"],
            num_channels=meta["num_channels"],
            kernel_size=meta["kernel_size"],
            dilations=meta["dilations"],
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
