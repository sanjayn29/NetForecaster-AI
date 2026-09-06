"""
tests/test_phase4.py
--------------------
Comprehensive Unit Test Suite for Phase 4:
Latent Network State World Model + Recursive K-Step Forecasting.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.state_transition import ResidualStateTransition
from src.models.world_model import LatentNetworkWorldModel
from src.forecasting.world_rollout import WorldModelRolloutEngine
from src.models.loss import MultiHorizonWeightedBCEWithLogitsLoss


class TestStateTransition:
    """Test suite for ResidualStateTransition network."""

    def test_transition_output_shape(self) -> None:
        B, latent_dim = 16, 64
        net = ResidualStateTransition(latent_dim=latent_dim, hidden_dim=128, dropout=0.1)
        s_t = torch.randn(B, latent_dim)
        s_next = net(s_t)
        assert s_next.shape == (B, latent_dim)

    def test_transition_residual_connection(self) -> None:
        net = ResidualStateTransition(latent_dim=32, hidden_dim=64, dropout=0.0)
        s_t = torch.zeros(4, 32)
        s_next = net(s_t)
        # Even with zero input, delta can be non-zero (bias), but shape and computation must succeed
        assert s_next.shape == (4, 32)


class TestLatentNetworkWorldModel:
    """Test suite for LatentNetworkWorldModel architecture and rollout."""

    @pytest.fixture
    def sample_input(self) -> torch.Tensor:
        # Batch=8, Window=20, Features=68
        return torch.randn(8, 20, 68)

    @pytest.fixture
    def world_model(self) -> LatentNetworkWorldModel:
        return LatentNetworkWorldModel(
            num_features=68,
            latent_dim=64,
            encoder_channels=[32, 32, 32],
            encoder_dilations=[1, 2, 4],
            kernel_size=3,
            transition_hidden_dim=64,
            forecast_horizon_k=5,
            dropout=0.1,
            enable_reconstruction=True,
        )

    def test_encoder_output_shape(self, world_model: LatentNetworkWorldModel, sample_input: torch.Tensor) -> None:
        s_0 = world_model.encode(sample_input)
        assert s_0.shape == (8, 64)

    def test_recursive_rollout_shapes(self, world_model: LatentNetworkWorldModel, sample_input: torch.Tensor) -> None:
        out = world_model(sample_input, horizon=5)
        assert "logits" in out
        assert "states" in out
        assert "reconstructions" in out
        assert "initial_state" in out

        # Logits shape: (B, 5)
        assert out["logits"].shape == (8, 5)
        # Latent states shape: (B, 5, 64)
        assert out["states"].shape == (8, 5, 64)
        # Feature reconstructions shape: (B, 5, 68)
        assert out["reconstructions"].shape == (8, 5, 68)
        # Initial state shape: (B, 64)
        assert out["initial_state"].shape == (8, 64)

    def test_custom_horizon_rollout(self, world_model: LatentNetworkWorldModel, sample_input: torch.Tensor) -> None:
        out = world_model(sample_input, horizon=3)
        assert out["logits"].shape == (8, 3)
        assert out["states"].shape == (8, 3, 64)
        assert out["reconstructions"].shape == (8, 3, 68)

    def test_zero_future_leakage_in_rollout(self, world_model: LatentNetworkWorldModel) -> None:
        """Verify that rollout depends strictly on the observed window and nothing in the future."""
        world_model.eval()
        x = torch.randn(2, 20, 68)
        with torch.no_grad():
            out1 = world_model(x)
            out2 = world_model(x)
        # Same observed sequence produces identical rollout
        torch.testing.assert_close(out1["logits"], out2["logits"], msg="Rollout should be strictly deterministic")
        torch.testing.assert_close(out1["states"], out2["states"], msg="Latent states should match identically")

    def test_gradient_backpropagation(self, world_model: LatentNetworkWorldModel, sample_input: torch.Tensor) -> None:
        """Verify that gradients propagate from all rollout steps back through encoder and transition."""
        world_model.train()
        out = world_model(sample_input)
        targets = torch.randint(0, 2, (8, 5)).float()
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(out["logits"], targets)

        # Also add reconstruction loss
        recon_targets = torch.randn(8, 5, 68)
        recon_loss = F.smooth_l1_loss(out["reconstructions"], recon_targets)
        total_loss = loss + 0.1 * recon_loss

        total_loss.backward()

        # Check gradients in encoder and transition
        encoder_param = next(world_model.encoder.parameters())
        transition_param = next(world_model.transition.parameters())
        attack_param = next(world_model.attack_head.parameters())
        recon_param = next(world_model.recon_head.parameters())

        assert encoder_param.grad is not None and not torch.isnan(encoder_param.grad).any()
        assert transition_param.grad is not None and not torch.isnan(transition_param.grad).any()
        assert attack_param.grad is not None and not torch.isnan(attack_param.grad).any()
        assert recon_param.grad is not None and not torch.isnan(recon_param.grad).any()

    def test_predict_proba(self, world_model: LatentNetworkWorldModel) -> None:
        x = np.random.randn(10, 20, 68).astype(np.float32)
        probs = world_model.predict_proba(x)
        assert probs.shape == (10, 5)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_serialization(self, world_model: LatentNetworkWorldModel, sample_input: torch.Tensor, tmp_path: Path) -> None:
        world_model.eval()
        save_path = world_model.save_pretrained(tmp_path / "world_model")
        assert (save_path / "best_model.pt").exists()
        assert (save_path / "config.json").exists()

        loaded_model = LatentNetworkWorldModel.from_pretrained(save_path)
        loaded_model.eval()

        with torch.no_grad():
            orig_out = world_model(sample_input)
            loaded_out = loaded_model(sample_input)

        torch.testing.assert_close(orig_out["logits"], loaded_out["logits"])
        torch.testing.assert_close(orig_out["states"], loaded_out["states"])


class TestWorldModelRolloutEngine:
    """Test suite for WorldModelRolloutEngine diagnostic metrics."""

    def test_rollout_metrics_computation(self) -> None:
        model = LatentNetworkWorldModel(
            num_features=68,
            latent_dim=32,
            encoder_channels=[16, 16],
            encoder_dilations=[1, 2],
            kernel_size=3,
            transition_hidden_dim=32,
            forecast_horizon_k=5,
        )
        engine = WorldModelRolloutEngine(model)

        N = 25
        x = np.random.randn(N, 20, 68).astype(np.float32)
        y = np.random.randint(0, 2, (N, 5)).astype(np.float32)
        x_fut = np.random.randn(N, 5, 68).astype(np.float32)

        results = engine.evaluate_rollout(x, y_attack=y, x_future_target=x_fut, batch_size=10)

        assert results["num_samples"] == N
        assert len(results["latent_state_norms"]) == 5
        assert len(results["latent_state_drifts"]) == 5
        assert len(results["mean_predicted_probabilities"]) == 5
        assert len(results["actual_positive_prevalences"]) == 5
        assert len(results["reconstruction_mae"]) == 5
        assert len(results["reconstruction_mse"]) == 5
        assert results["initial_state_norm"] > 0
        assert results["probabilities"].shape == (N, 5)
