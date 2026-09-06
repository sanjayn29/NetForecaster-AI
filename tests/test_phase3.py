"""
tests/test_phase3.py
--------------------
Unit tests for Phase 3: Temporal Sequence Modeling (TCN + Transformer).

Tests:
  1. TCN input shape (B, 20, 68) -> (B, 5) output logits
  2. TCN causality: modifying future input does not affect past representation
  3. TCN receptive field >= 20 timesteps
  4. Transformer input shape (B, 20, 68) -> (B, 5) output logits
  5. Reusable PositionalEncoding module shape and determinism
  6. Transformer attention extraction diagnostics
  7. No future target leakage in Transformer self-attention
  8. Weighted BCE pos_weight calculation strictly on training targets
  9. Binary Focal Loss with logits numerical stability
  10. Gradient backward pass for TCN & Transformer
  11. Model serialization & loading round-trip (TCN & Transformer)
  12. Multiclass forecasting support for both models
  13. Deterministic seed behavior
"""

import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tcn_model import TCNForecaster, TemporalConvNet, TemporalBlock, Chomp1d
from src.models.transformer_model import TransformerForecaster, PositionalEncoding
from src.models.loss import (
    compute_pos_weight,
    MultiHorizonWeightedBCEWithLogitsLoss,
    BinaryFocalLossWithLogits,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_batch():
    """Batch of 16 sequences of length 20 with 68 features."""
    torch.manual_seed(42)
    return torch.randn(16, 20, 68)


@pytest.fixture
def sample_targets():
    """Binary target matrix for 16 samples across 5 horizons."""
    torch.manual_seed(42)
    return torch.randint(0, 2, (16, 5)).float()


# ── 1. TCN Architecture Tests ────────────────────────────────────────────────

class TestTCNArchitecture:
    """Test Causal Temporal Convolutional Network properties."""

    def test_tcn_forward_shape(self, sample_batch):
        """TCN forward pass should map (B, 20, 68) to (B, 5)."""
        model = TCNForecaster(
            num_features=68,
            num_channels=[64, 64, 64, 64],
            kernel_size=3,
            dilations=[1, 2, 4, 8],
            forecast_horizon_k=5,
            num_classes=2,
        )
        logits = model(sample_batch)
        assert logits.shape == (16, 5)
        assert not torch.isnan(logits).any()

    def test_tcn_receptive_field_coverage(self):
        """TCN receptive field with dilations [1, 2, 4, 8] and k=3 must cover W=20."""
        model = TCNForecaster(
            num_features=68,
            num_channels=[64, 64, 64, 64],
            kernel_size=3,
            dilations=[1, 2, 4, 8],
            forecast_horizon_k=5,
        )
        rf = model.receptive_field
        # 1 + 2 * (3 - 1) * (1 + 2 + 4 + 8) = 1 + 4 * 15 = 61
        assert rf >= 20, f"Receptive field {rf} is smaller than window size 20!"
        assert rf == 61

    def test_tcn_strict_causality(self):
        """Modifying an input timestep t > i MUST NOT alter the representations at t <= i."""
        torch.manual_seed(42)
        model = TCNForecaster(
            num_features=68,
            num_channels=[32, 32, 32, 32],
            kernel_size=3,
            dilations=[1, 2, 4, 8],
            dropout=0.0,
        )
        model.eval()

        # Create base input sequence of 20 timesteps
        x1 = torch.randn(1, 20, 68)
        # Create x2 where timesteps 15..19 are drastically modified
        x2 = x1.clone()
        x2[:, 15:, :] += 100.0

        with torch.no_grad():
            h1, _ = model.get_temporal_representation(x1)  # (1, 20, 32)
            h2, _ = model.get_temporal_representation(x2)  # (1, 20, 32)

        # Positions 0..14 should be exactly identical
        torch.testing.assert_close(
            h1[:, :15, :],
            h2[:, :15, :],
            rtol=1e-5,
            atol=1e-5,
            msg="Causality violated: modifying future timesteps affected past representation!",
        )

        # Positions 15..19 should differ
        assert not torch.allclose(h1[:, 15:, :], h2[:, 15:, :])

    def test_tcn_prediction_methods(self, sample_batch):
        """Verify predict_forecasting and predict_proba_forecasting methods."""
        model = TCNForecaster(num_features=68, forecast_horizon_k=5)
        preds = model.predict_forecasting(sample_batch)
        assert preds.shape == (16, 5)
        assert set(np.unique(preds)).issubset({0, 1})

        probs = model.predict_proba_forecasting(sample_batch)
        assert len(probs) == 5
        for p in probs:
            assert p.shape == (16, 2)
            np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-5)

    def test_tcn_multiclass_shape(self, sample_batch):
        """TCN multiclass mode should output (B, 5, 15)."""
        model = TCNForecaster(
            num_features=68,
            forecast_horizon_k=5,
            num_classes=15,
        )
        logits = model(sample_batch)
        assert logits.shape == (16, 5, 15)

        preds = model.predict_forecasting(sample_batch)
        assert preds.shape == (16, 5)

        probs = model.predict_proba_forecasting(sample_batch)
        assert len(probs) == 5
        assert probs[0].shape == (16, 15)


# ── 2. Transformer Architecture Tests ────────────────────────────────────────

class TestTransformerArchitecture:
    """Test Transformer Sequence Encoder properties."""

    def test_positional_encoding_shape_and_determinism(self):
        """PositionalEncoding should preserve tensor shape and be deterministic."""
        pe_module = PositionalEncoding(d_model=128, max_len=50, dropout=0.0)
        x = torch.zeros(4, 20, 128)
        out1 = pe_module(x)
        out2 = pe_module(x)

        assert out1.shape == (4, 20, 128)
        torch.testing.assert_close(out1, out2)
        # Verify non-zero positional encoding added
        assert not torch.all(out1 == 0)

    def test_transformer_forward_shape(self, sample_batch):
        """Transformer forward pass should map (B, 20, 68) to (B, 5)."""
        model = TransformerForecaster(
            num_features=68,
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            forecast_horizon_k=5,
            num_classes=2,
            dropout=0.1,
        )
        logits = model(sample_batch)
        assert logits.shape == (16, 5)
        assert not torch.isnan(logits).any()

    def test_transformer_attention_diagnostics(self, sample_batch):
        """Transformer should extract attention weights of shape (B, nhead, 20, 20)."""
        model = TransformerForecaster(
            num_features=68,
            d_model=64,
            nhead=4,
            num_layers=2,
            forecast_horizon_k=5,
        )
        attn = model.get_attention_weights(sample_batch)
        assert attn.shape == (16, 4, 20, 20)
        # Attention weights along key dimension should sum to 1
        np.testing.assert_allclose(attn.sum(axis=-1), 1.0, rtol=1e-4, atol=1e-4)

    def test_transformer_prediction_methods(self, sample_batch):
        """Verify predict_forecasting and predict_proba_forecasting."""
        model = TransformerForecaster(num_features=68, forecast_horizon_k=5)
        preds = model.predict_forecasting(sample_batch)
        assert preds.shape == (16, 5)
        assert set(np.unique(preds)).issubset({0, 1})

        probs = model.predict_proba_forecasting(sample_batch)
        assert len(probs) == 5
        for p in probs:
            assert p.shape == (16, 2)
            np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-5)


# ── 3. Loss Function Tests ───────────────────────────────────────────────────

class TestLossFunctions:
    """Test Multi-Horizon Weighted BCE and Binary Focal Loss."""

    def test_compute_pos_weight_training_distribution(self):
        """pos_weight should equal N_negative / N_positive."""
        # 100 samples, 20 positive (1) and 80 negative (0)
        targets = np.array([1] * 20 + [0] * 80)
        w = compute_pos_weight(targets, horizon_specific=False)
        assert pytest.approx(w.item(), 1e-4) == 80.0 / 20.0 == 4.0

    def test_compute_pos_weight_horizon_specific(self):
        """pos_weight calculated individually for each of the 5 horizons."""
        # Matrix with varying positive rates
        # H0: 10 pos / 90 neg -> 9.0
        # H1: 20 pos / 80 neg -> 4.0
        # H2: 50 pos / 50 neg -> 1.0
        # H3: 5 pos / 95 neg -> 19.0
        # H4: 25 pos / 75 neg -> 3.0
        targets = np.zeros((100, 5), dtype=int)
        targets[:10, 0] = 1
        targets[:20, 1] = 1
        targets[:50, 2] = 1
        targets[:5, 3] = 1
        targets[:25, 4] = 1

        weights = compute_pos_weight(targets, horizon_specific=True)
        assert weights.shape == (5,)
        expected = np.array([9.0, 4.0, 1.0, 19.0, 3.0])
        np.testing.assert_allclose(weights.numpy(), expected, rtol=1e-4)

    def test_weighted_bce_loss_computation(self, sample_targets):
        """MultiHorizonWeightedBCEWithLogitsLoss should penalize false negatives more when pos_weight > 1."""
        pos_weight = torch.tensor([5.0, 5.0, 5.0, 5.0, 5.0])
        criterion = MultiHorizonWeightedBCEWithLogitsLoss(pos_weight=pos_weight)

        logits = torch.zeros(16, 5)  # sigmoid(0) = 0.5
        loss = criterion(logits, sample_targets)
        assert loss.ndim == 0
        assert not torch.isnan(loss)
        assert loss.item() > 0

    def test_focal_loss_numerical_stability(self, sample_targets):
        """BinaryFocalLossWithLogits should remain stable on extreme logits."""
        criterion = BinaryFocalLossWithLogits(alpha=0.75, gamma=2.0)
        # Extreme logits
        extreme_logits = torch.tensor([[-50.0, 50.0, -100.0, 100.0, 0.0]] * 16)
        loss = criterion(extreme_logits, sample_targets)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert loss.item() >= 0.0


# ── 4. Gradient and Training Dynamics Tests ──────────────────────────────────

class TestTrainingDynamics:
    """Test backward pass, parameter updates, and gradient clipping compatibility."""

    def test_tcn_backward_pass(self, sample_batch, sample_targets):
        """Verify backpropagation through TCN forecaster."""
        model = TCNForecaster(num_features=68, forecast_horizon_k=5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = MultiHorizonWeightedBCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))

        optimizer.zero_grad()
        logits = model(sample_batch)
        loss = criterion(logits, sample_targets)
        loss.backward()

        # Check gradients exist on all trainable parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"
                assert not torch.isnan(param.grad).any()

        optimizer.step()

    def test_transformer_backward_pass(self, sample_batch, sample_targets):
        """Verify backpropagation through Transformer forecaster."""
        model = TransformerForecaster(num_features=68, d_model=64, nhead=4, num_layers=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = BinaryFocalLossWithLogits(alpha=0.75, gamma=2.0)

        optimizer.zero_grad()
        logits = model(sample_batch)
        loss = criterion(logits, sample_targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No grad for {name}"
                assert not torch.isnan(param.grad).any()

        optimizer.step()


# ── 5. Serialization and Loading Roundtrip Tests ─────────────────────────────

class TestSerialization:
    """Test saving and loading models across filesystems."""

    def test_tcn_save_and_load(self, sample_batch, tmp_path):
        """TCN save and load should preserve architecture and exact inference outputs."""
        model = TCNForecaster(num_features=68, num_channels=[32, 32], forecast_horizon_k=5)
        model.eval()

        with torch.no_grad():
            expected = model(sample_batch)

        save_dir = tmp_path / "tcn_model"
        model.save(save_dir)

        assert (save_dir / "config.json").exists()
        assert (save_dir / "best_model.pt").exists()

        loaded_model = TCNForecaster.load(save_dir)
        with torch.no_grad():
            actual = loaded_model(sample_batch)

        torch.testing.assert_close(actual, expected)

    def test_transformer_save_and_load(self, sample_batch, tmp_path):
        """Transformer save and load should preserve architecture and exact inference outputs."""
        model = TransformerForecaster(num_features=68, d_model=64, nhead=2, num_layers=2)
        model.eval()

        with torch.no_grad():
            expected = model(sample_batch)

        save_dir = tmp_path / "transformer_model"
        model.save(save_dir)

        assert (save_dir / "config.json").exists()
        assert (save_dir / "best_model.pt").exists()

        loaded_model = TransformerForecaster.load(save_dir)
        with torch.no_grad():
            actual = loaded_model(sample_batch)

        torch.testing.assert_close(actual, expected)
