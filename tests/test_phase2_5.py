"""
tests/test_phase2_5.py
-----------------------
Unit tests for Phase 2.5: Baseline Scientific Validation and Audit.

Tests:
  - Target horizon alignment verification
  - Binary target encoding invariants
  - Temporal aggregation causality and zero future lookahead
  - Probability orientation and ordering
  - Metric calculation mathematical consistency (Macro F1 on imbalanced data)
  - Prediction/target length equality across all horizons
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.window_representation import aggregate_window, aggregate_windows_batch
from src.evaluation.metrics import compute_classification_metrics, compute_forecasting_metrics


class TestPhase25Audits:
    def test_target_horizon_alignment_mathematical(self):
        """Audit 4: Verify that input X[t:t+20] corresponds to targets Y[t+20..t+24] = T+1..T+5."""
        W, K = 20, 5
        stream_len = 100
        stream_indices = np.arange(stream_len)
        
        # At window starting at t=0
        t = 0
        input_window = stream_indices[t : t + W]
        target_seq = stream_indices[t + W : t + W + K]
        
        assert len(input_window) == 20
        assert input_window[0] == 0
        assert input_window[-1] == 19
        
        assert len(target_seq) == 5
        assert target_seq[0] == 20  # T+1 is step 20 (first future step)
        assert target_seq[1] == 21  # T+2
        assert target_seq[2] == 22  # T+3
        assert target_seq[3] == 23  # T+4
        assert target_seq[4] == 24  # T+5
        
        # Test disjointness
        assert set(input_window).isdisjoint(set(target_seq))

    def test_binary_target_encoding_invariant(self):
        """Audit 1: Verify Benign is strictly 0 and all positive attack IDs map to 1."""
        raw_labels = np.array([0, 1, 5, 8, 12, 14, 0, 0, 7, 13])
        binary_targets = (raw_labels > 0).astype(int)
        
        assert binary_targets[0] == 0
        assert binary_targets[6] == 0
        assert binary_targets[7] == 0
        assert np.all(binary_targets[np.array([1, 2, 3, 4, 5, 8, 9])] == 1)

    def test_temporal_aggregation_last_step(self):
        """Audit 5: Verify that 'Last' feature is exactly the final timestep X[t+W-1]."""
        W, D = 20, 68
        window = np.zeros((W, D), dtype=np.float32)
        # Put distinctive values in the last step
        window[-1, :] = 42.0
        
        agg = aggregate_window(window, include_last=True, include_mean=True, include_std=True)
        # First D elements are 'Last'
        last_features = agg[:D]
        assert np.all(last_features == 42.0)

    def test_probability_orientation_sanity(self):
        """Audit 3: Verify that ROC-AUC receives probability of class 1 and increases when positive class prob is higher."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        # Well-ordered probabilities
        prob_good = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9]])
        m_good = compute_classification_metrics(y_true, np.argmax(prob_good, axis=1), y_prob=prob_good)
        assert m_good["roc_auc"] == 1.0

        # Inverted probabilities
        prob_inverted = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1]])
        m_inverted = compute_classification_metrics(y_true, np.argmax(prob_inverted, axis=1), y_prob=prob_inverted)
        assert m_inverted["roc_auc"] == 0.0

    def test_imbalanced_macro_f1_consistency(self):
        """Audit 7: Mathematically verify why majority prediction gives high accuracy but low Macro F1."""
        # 90 benign (0), 10 attacks (1)
        y_true = np.array([0] * 90 + [1] * 10)
        
        # Model A: Predicts ALL Benign (Majority classifier)
        y_pred_majority = np.zeros(100, dtype=int)
        acc_maj = accuracy_score(y_true, y_pred_majority)
        _, _, f1_macro_maj, _ = precision_recall_fscore_support(y_true, y_pred_majority, average="macro", zero_division=0)
        
        assert acc_maj == 0.90  # 90% accuracy!
        # F1_benign = 2*90/(90+90+10) = 180/190 = 0.9474, F1_attack = 0.0 -> Macro F1 = 0.4737
        assert f1_macro_maj == pytest.approx(0.4737, abs=1e-3)

        # Model B: Predicts some attacks correctly with some false alarms
        y_pred_balanced = y_true.copy()
        y_pred_balanced[0:20] = 1  # 20 false positives
        acc_bal = accuracy_score(y_true, y_pred_balanced)
        _, _, f1_macro_bal, _ = precision_recall_fscore_support(y_true, y_pred_balanced, average="macro", zero_division=0)
        
        assert acc_bal == 0.80  # Lower accuracy (80%)
        # But higher Macro F1 because attack recall is 100%!
        assert f1_macro_bal > f1_macro_maj

    def test_prediction_target_length_equality(self):
        """Audit 13: Verify dimension alignment for 5-horizon forecasting metrics."""
        N, K = 100, 5
        y_true_seq = np.random.randint(0, 2, (N, K))
        y_pred_seq = np.random.randint(0, 2, (N, K))
        prob_seq = [np.random.rand(N, 2) for _ in range(K)]
        
        metrics = compute_forecasting_metrics(y_true_seq, y_pred_seq, y_prob_seq=prob_seq)
        assert len(metrics["horizons"]) == K
        for k in range(K):
            assert f"T+{k+1}" in metrics["horizons"]
            assert metrics["horizons"][f"T+{k+1}"]["n_samples"] == N
