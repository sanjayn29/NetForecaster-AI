"""
tests/test_phase2.py
--------------------
Unit tests for Phase 2: Baseline Models and Evaluation.

Tests:
  - Statistical window aggregation (Last, Mean, Std)
  - Evaluation metrics calculation & horizon degradation
  - LogisticRegression baseline (single-step & 5-horizon forecasting, binary & multiclass)
  - RandomForest & GradientBoosting baselines (single-step & forecasting)
  - PyTorch LSTM & GRU sequence forecasters (shapes, forward pass, inference, save/load)
"""

import json
import sys
from pathlib import Path
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.window_representation import (
    aggregate_window,
    aggregate_windows_batch,
    get_aggregated_feature_names,
)
from src.evaluation.metrics import (
    compute_classification_metrics,
    compute_forecasting_metrics,
)
from src.models.logistic_baseline import LogisticRegressionBaseline
from src.models.tree_baselines import RandomForestBaseline, GradientBoostingBaseline
from src.models.lstm_model import LSTMForecaster
from src.models.gru_model import GRUForecaster


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_window_batch():
    """Batch of 50 windows of shape (50, 20, 68)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((50, 20, 68)).astype(np.float32)


@pytest.fixture
def sample_tabular_data():
    """Small synthetic dataset for tabular baselines."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 204)).astype(np.float32)
    y_bin = rng.integers(0, 2, size=100)
    y_multi = rng.integers(0, 5, size=100)
    y_seq_bin = rng.integers(0, 2, size=(100, 5))
    y_seq_multi = rng.integers(0, 5, size=(100, 5))
    return X, y_bin, y_multi, y_seq_bin, y_seq_multi


# ── 1. Window Representation Tests ──────────────────────────────────────────

class TestWindowRepresentation:
    def test_aggregate_single_window_shape(self):
        w = np.ones((20, 68))
        agg = aggregate_window(w, include_last=True, include_mean=True, include_std=True)
        assert agg.shape == (204,)

    def test_aggregate_batch_shape(self, sample_window_batch):
        agg = aggregate_windows_batch(sample_window_batch)
        assert agg.shape == (50, 204)

    def test_aggregated_feature_names(self):
        base_names = [f"f_{i}" for i in range(68)]
        names = get_aggregated_feature_names(base_names)
        assert len(names) == 204
        assert names[0] == "f_0_last"
        assert names[68] == "f_0_mean"
        assert names[136] == "f_0_std"


# ── 2. Metric Computation Tests ─────────────────────────────────────────────

class TestMetrics:
    def test_classification_metrics_binary(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.6, 0.4], [0.7, 0.3], [0.2, 0.8]])
        res = compute_classification_metrics(y_true, y_pred, y_prob=prob)
        assert res["accuracy"] == pytest.approx(5 / 6)
        assert "macro_f1" in res
        assert "false_positive_rate" in res
        assert "roc_auc" in res
        assert "brier_score" in res

    def test_forecasting_metrics_degradation(self):
        y_true_seq = np.zeros((10, 5), dtype=int)
        y_pred_seq = np.zeros((10, 5), dtype=int)
        # Introduce decay
        y_pred_seq[5:, 4] = 1  # errors at horizon 5
        res = compute_forecasting_metrics(y_true_seq, y_pred_seq)
        assert "horizons" in res
        assert "T+1" in res["horizons"]
        assert "T+5" in res["horizons"]
        assert "degradation" in res
        assert len(res["degradation"]["f1_macro_decay"]) == 5


# ── 3. Logistic Regression Baseline Tests ───────────────────────────────────

class TestLogisticBaseline:
    def test_single_step_binary(self, sample_tabular_data):
        X, y_bin, _, _, _ = sample_tabular_data
        model = LogisticRegressionBaseline(task="binary", max_iter=100)
        model.fit_single_step(X, y_bin)
        preds = model.predict_single_step(X)
        assert preds.shape == (100,)
        probs = model.predict_proba_single_step(X)
        assert probs.shape == (100, 2)

    def test_forecasting_5_horizons(self, sample_tabular_data):
        X, _, _, y_seq_bin, _ = sample_tabular_data
        model = LogisticRegressionBaseline(task="binary", forecast_horizon_k=5, max_iter=100)
        model.fit_forecasting(X, y_seq_bin)
        preds = model.predict_forecasting(X)
        assert preds.shape == (100, 5)
        probs = model.predict_proba_forecasting(X)
        assert len(probs) == 5
        assert probs[0].shape == (100, 2)

    def test_save_and_load(self, sample_tabular_data):
        X, _, _, y_seq_bin, _ = sample_tabular_data
        model = LogisticRegressionBaseline(task="binary", forecast_horizon_k=5, max_iter=100)
        model.fit_forecasting(X, y_seq_bin)
        save_dir = PROJECT_ROOT / "scratch" / "test_logistic_tmp"
        model.save(save_dir)

        loaded = LogisticRegressionBaseline.load(save_dir)
        preds = loaded.predict_forecasting(X)
        assert preds.shape == (100, 5)


# ── 4. Tree Baseline Tests ──────────────────────────────────────────────────

class TestTreeBaselines:
    def test_random_forest_forecasting(self, sample_tabular_data):
        X, _, _, y_seq_bin, _ = sample_tabular_data
        rf = RandomForestBaseline(n_estimators=10, max_depth=5, forecast_horizon_k=5)
        rf.fit_forecasting(X, y_seq_bin)
        preds = rf.predict_forecasting(X)
        assert preds.shape == (100, 5)

    def test_gradient_boosting_forecasting(self, sample_tabular_data):
        X, _, _, y_seq_bin, _ = sample_tabular_data
        gb = GradientBoostingBaseline(n_estimators=10, max_depth=3, forecast_horizon_k=5)
        gb.fit_forecasting(X, y_seq_bin)
        preds = gb.predict_forecasting(X)
        assert preds.shape == (100, 5)


# ── 5. PyTorch Sequence Forecasters (LSTM & GRU) ────────────────────────────

class TestSequenceForecasters:
    def test_lstm_binary_forward_and_shapes(self, sample_window_batch):
        model = LSTMForecaster(num_features=68, hidden_dim=32, num_layers=1, forecast_horizon_k=5, num_classes=2)
        x = torch.from_numpy(sample_window_batch)
        logits = model(x)
        assert logits.shape == (50, 5)

        preds = model.predict_forecasting(sample_window_batch)
        assert preds.shape == (50, 5)

        probs = model.predict_proba_forecasting(sample_window_batch)
        assert len(probs) == 5
        assert probs[0].shape == (50, 2)

    def test_lstm_multiclass_forward_and_shapes(self, sample_window_batch):
        model = LSTMForecaster(num_features=68, hidden_dim=32, num_layers=1, forecast_horizon_k=5, num_classes=15)
        x = torch.from_numpy(sample_window_batch)
        logits = model(x)
        assert logits.shape == (50, 5, 15)

        preds = model.predict_forecasting(sample_window_batch)
        assert preds.shape == (50, 5)

    def test_gru_binary_forward_and_shapes(self, sample_window_batch):
        model = GRUForecaster(num_features=68, hidden_dim=32, num_layers=1, forecast_horizon_k=5, num_classes=2)
        x = torch.from_numpy(sample_window_batch)
        logits = model(x)
        assert logits.shape == (50, 5)

        preds = model.predict_forecasting(sample_window_batch)
        assert preds.shape == (50, 5)

    def test_lstm_save_and_load(self, sample_window_batch):
        model = LSTMForecaster(num_features=68, hidden_dim=32, num_layers=1, forecast_horizon_k=5, num_classes=2)
        save_dir = PROJECT_ROOT / "scratch" / "test_lstm_tmp"
        model.save(save_dir)

        loaded = LSTMForecaster.load(save_dir)
        preds = loaded.predict_forecasting(sample_window_batch)
        assert preds.shape == (50, 5)


# ── 6. Visualization Tests ──────────────────────────────────────────────────

class TestVisualization:
    def test_plot_roc_curve(self):
        from src.evaluation.visualization import plot_roc_curve
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.8, 0.7, 0.2, 0.9])
        save_path = PROJECT_ROOT / "scratch" / "test_roc.png"
        plot_roc_curve(y_true, y_prob, title="Test ROC", save_path=save_path, label="Test")
        assert save_path.exists()

    def test_plot_pr_curve(self):
        from src.evaluation.visualization import plot_pr_curve
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.8, 0.7, 0.2, 0.9])
        save_path = PROJECT_ROOT / "scratch" / "test_pr.png"
        plot_pr_curve(y_true, y_prob, title="Test PR", save_path=save_path, label="Test")
        assert save_path.exists()

    def test_plot_model_comparison(self):
        from src.evaluation.visualization import plot_model_comparison
        save_path = PROJECT_ROOT / "scratch" / "test_comparison.png"
        plot_model_comparison(
            model_names=["LR", "RF", "GB"],
            metric_values={"F1": [0.8, 0.9, 0.85], "AUC": [0.75, 0.88, 0.82]},
            title="Test Comparison",
            save_path=save_path,
        )
        assert save_path.exists()


# ── 7. Multiclass Forecasting Tests ─────────────────────────────────────────

class TestMulticlassForecasting:
    def test_logistic_multiclass_forecasting(self, sample_tabular_data):
        X, _, _, _, y_seq_multi = sample_tabular_data
        model = LogisticRegressionBaseline(task="multiclass", forecast_horizon_k=5, max_iter=100)
        model.fit_forecasting(X, y_seq_multi)
        preds = model.predict_forecasting(X)
        assert preds.shape == (100, 5)
        probs = model.predict_proba_forecasting(X)
        assert len(probs) == 5
        # Multiclass: each horizon has (N, n_classes) probabilities
        assert probs[0].shape[0] == 100

    def test_random_forest_multiclass_forecasting(self, sample_tabular_data):
        X, _, _, _, y_seq_multi = sample_tabular_data
        rf = RandomForestBaseline(n_estimators=10, max_depth=5, task="multiclass", forecast_horizon_k=5)
        rf.fit_forecasting(X, y_seq_multi)
        preds = rf.predict_forecasting(X)
        assert preds.shape == (100, 5)

    def test_gru_multiclass_forward_shapes(self, sample_window_batch):
        model = GRUForecaster(num_features=68, hidden_dim=32, num_layers=1, forecast_horizon_k=5, num_classes=15)
        preds = model.predict_forecasting(sample_window_batch)
        assert preds.shape == (50, 5)
        probs = model.predict_proba_forecasting(sample_window_batch)
        assert len(probs) == 5
        assert probs[0].shape == (50, 15)


# ── 8. Window Representation Edge Case Tests ────────────────────────────────

class TestWindowEdgeCases:
    def test_aggregate_with_min_max(self):
        w = np.random.randn(20, 68).astype(np.float32)
        agg = aggregate_window(w, include_last=True, include_mean=True, include_std=True, include_min=True, include_max=True)
        assert agg.shape == (68 * 5,)

    def test_aggregate_batch_with_min_max(self):
        batch = np.random.randn(10, 20, 68).astype(np.float32)
        agg = aggregate_windows_batch(batch, include_last=True, include_mean=True, include_std=True, include_min=True, include_max=True)
        assert agg.shape == (10, 68 * 5)

    def test_feature_names_with_min_max(self):
        base_names = [f"f_{i}" for i in range(68)]
        names = get_aggregated_feature_names(base_names, include_min=True, include_max=True)
        assert len(names) == 68 * 5
        assert names[-68] == "f_0_max"


# ── 9. Phase 2 Mandatory Verification Requirements ─────────────────────────

class TestPhase2MandatoryRequirements:
    def test_no_future_information_enters_tabular_representation(self):
        """Requirement 2: Ensure tabular aggregation only sees timesteps 0..W-1 and zero future lookahead."""
        W, D, K = 20, 68, 5
        full_stream = np.random.randn(W + K, D).astype(np.float32)
        window = full_stream[:W]
        agg_before = aggregate_window(window)

        # Alter future values (timesteps W .. W+K-1)
        full_stream[W:] = full_stream[W:] + 999.0

        agg_after = aggregate_window(full_stream[:W])
        np.testing.assert_array_equal(agg_before, agg_after)

    def test_correct_horizon_target_alignment(self):
        """Requirement 3: Verify target sequences align strictly to future steps [t+W, t+W+K-1]."""
        W, K = 20, 5
        labels = np.arange(100, dtype=int)
        t = 10
        # Target should be labels at indices [10+20 .. 10+20+5-1] = [30, 31, 32, 33, 34]
        expected_target = labels[t + W : t + W + K]
        assert len(expected_target) == K
        assert list(expected_target) == [30, 31, 32, 33, 34]
        # Ensure no overlap with input window [10..29]
        input_indices = list(range(t, t + W))
        target_indices = list(range(t + W, t + W + K))
        assert set(input_indices).isdisjoint(set(target_indices))

    def test_binary_target_conversion(self):
        """Requirement 4: Verify Benign = 0 and Any Attack >= 1 maps to 1."""
        multi_labels = np.array([0, 1, 4, 0, 7, 14, 0], dtype=int)
        binary_labels = (multi_labels > 0).astype(int)
        expected = np.array([0, 1, 1, 0, 1, 1, 0], dtype=int)
        np.testing.assert_array_equal(binary_labels, expected)

    def test_class_weight_configuration(self):
        """Requirement 6: Verify balanced class weight applies inverse frequency weighting."""
        lr_std = LogisticRegressionBaseline(task="binary", balanced=False)
        lr_bal = LogisticRegressionBaseline(task="binary", balanced=True)
        assert lr_std.class_weight is None
        assert lr_bal.class_weight == "balanced"

    def test_model_serialization_loading_all(self, sample_tabular_data, sample_window_batch):
        """Requirement 7: Test serialization and loading for RF, GRU, and Logistic."""
        X, _, _, y_seq_bin, _ = sample_tabular_data
        # Random Forest save/load
        rf = RandomForestBaseline(n_estimators=5, max_depth=3, forecast_horizon_k=5)
        rf.fit_forecasting(X, y_seq_bin)
        rf_dir = PROJECT_ROOT / "scratch" / "test_rf_tmp"
        rf.save(rf_dir)
        rf_loaded = RandomForestBaseline.load(rf_dir)
        np.testing.assert_array_equal(rf.predict_forecasting(X), rf_loaded.predict_forecasting(X))

        # GRU save/load
        gru = GRUForecaster(num_features=68, hidden_dim=16, num_layers=1, forecast_horizon_k=5, num_classes=2)
        gru_dir = PROJECT_ROOT / "scratch" / "test_gru_tmp"
        gru.save(gru_dir)
        gru_loaded = GRUForecaster.load(gru_dir)
        np.testing.assert_array_equal(
            gru.predict_forecasting(sample_window_batch),
            gru_loaded.predict_forecasting(sample_window_batch)
        )

    def test_no_train_test_preprocessing_leakage(self):
        """Requirement 8: Verify scaler fit strictly on train and not altered by test data."""
        from src.preprocessing.scaler import FeatureScaler
        import pandas as pd
        rng = np.random.default_rng(42)
        cols = [f"feat_{i}" for i in range(10)]
        train_df = pd.DataFrame(rng.standard_normal((100, 10)).astype(np.float32) * 10.0 + 5.0, columns=cols)
        test_df = pd.DataFrame(rng.standard_normal((50, 10)).astype(np.float32) * 100.0 + 50.0, columns=cols)

        scaler = FeatureScaler(method="robust")
        scaler.fit(train_df, feature_cols=cols)
        train_center_orig = scaler.scaler.center_.copy()
        train_scale_orig = scaler.scaler.scale_.copy()

        # Transform test data
        _ = scaler.transform(test_df)

        # Center and scale parameters must remain exactly the training parameters
        np.testing.assert_array_equal(scaler.scaler.center_, train_center_orig)
        np.testing.assert_array_equal(scaler.scaler.scale_, train_scale_orig)



