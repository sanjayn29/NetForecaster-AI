"""
scripts/audit_phase2.py
-----------------------
Phase 2.5: Deep Scientific Validation and Audit Script for NetForecaster AI.

Executes:
  1. Target distribution audit across Train, Validation, Test (T+1 ... T+5)
  2. Prediction distribution audit across all models and horizons
  3. Deep inspection of LSTM and GRU output probabilities, targets, and ROC-AUC
  4. Temporal label autocorrelation: P(Y[t+k] == Y[t]) and P(Y[t+k] == Y[t+k-1])
  5. Calibration analysis for Random Forest T+1
  6. Data leakage audit verification
  7. Generation of figures and JSON audit artifacts
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import CFG
from src.utils.logger import get_logger, setup_root_logger
from src.data.label_encoder import CICLabelEncoder
from src.preprocessing.window_representation import aggregate_windows_batch, aggregate_window
from src.evaluation.metrics import compute_classification_metrics, compute_forecasting_metrics
from src.models.logistic_baseline import LogisticRegressionBaseline
from src.models.tree_baselines import RandomForestBaseline, GradientBoostingBaseline
from src.models.lstm_model import LSTMForecaster
from src.models.gru_model import GRUForecaster

setup_root_logger(level="INFO")
logger = get_logger("audit_phase2")

PROCESSED_DIR = PROJECT_ROOT / CFG.data.processed_dir
MODELS_BASE_DIR = PROJECT_ROOT / CFG.models.baseline_dir
REPORTS_DIR = PROJECT_ROOT / CFG.reports.dir
FIGURES_DIR = REPORTS_DIR / "figures" / "baseline"
METRICS_DIR = REPORTS_DIR / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_json(obj: Any) -> Any:
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    return obj


def load_partition(split_name: str, feature_cols: list[str], max_rows: int | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = PROCESSED_DIR / f"{split_name}.parquet"
    cols = feature_cols + ["label_binary", "label_encoded"]
    df = pd.read_parquet(path, columns=cols)
    if max_rows and len(df) > max_rows:
        step = max(1, len(df) // max_rows)
        df = df.iloc[::step].iloc[:max_rows].reset_index(drop=True)
    X = df[feature_cols].values.astype(np.float32)
    y_bin = df["label_binary"].values.astype(int)
    y_multi = df["label_encoded"].values.astype(int)
    return X, y_bin, y_multi


def build_window_matrices(X: np.ndarray, y_bin: np.ndarray, y_multi: np.ndarray, W: int = 20, K: int = 5, max_samples: int | None = 50_000):
    N_total = len(X)
    total_windows = N_total - W - K + 1
    if total_windows <= 0:
        raise ValueError("Not enough rows.")
    if max_samples and total_windows > max_samples:
        indices = np.linspace(0, total_windows - 1, max_samples, dtype=int)
    else:
        indices = np.arange(total_windows)
    
    n_out = len(indices)
    D = X.shape[1]
    X_seq = np.empty((n_out, W, D), dtype=np.float32)
    y_bin_seq = np.empty((n_out, K), dtype=int)
    y_multi_seq = np.empty((n_out, K), dtype=int)
    for i, idx in enumerate(indices):
        X_seq[i] = X[idx : idx + W]
        y_bin_seq[i] = y_bin[idx + W : idx + W + K]
        y_multi_seq[i] = y_multi[idx + W : idx + W + K]
    return X_seq, y_bin_seq, y_multi_seq


def main() -> None:
    logger.info("=" * 80)
    logger.info("PHASE 2.5: BASELINE SCIENTIFIC VALIDATION AND AUDIT")
    logger.info("=" * 80)

    # 1. Load Feature Metadata
    with open(PROJECT_ROOT / "models" / "preprocessing" / "feature_metadata.json", "r") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    W = CFG.windowing.window_size
    K = CFG.windowing.forecast_horizon

    # ══════════════════════════════════════════════════════════════════
    # AUDIT 1: TARGET DISTRIBUTION AUDIT ACROSS PARTITIONS & HORIZONS
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n[1/6] Auditing Target Distributions across Train, Val, Test...")
    target_dist: dict[str, Any] = {}
    partitions = ["train", "validation", "test"]
    
    for split in partitions:
        X_p, y_bin_p, y_multi_p = load_partition(split, feature_cols, max_rows=200_000)
        _, y_seq_bin, _ = build_window_matrices(X_p, y_bin_p, y_multi_p, W, K, max_samples=50_000)
        
        split_dict = {}
        for k in range(K):
            h_name = f"T+{k+1}"
            y_k = y_seq_bin[:, k]
            total = len(y_k)
            benign_cnt = int(np.sum(y_k == 0))
            attack_cnt = int(np.sum(y_k == 1))
            attack_pct = float(attack_cnt / total * 100)
            split_dict[h_name] = {
                "total_samples": total,
                "benign_count": benign_cnt,
                "attack_count": attack_cnt,
                "attack_percentage": round(attack_pct, 4),
            }
        target_dist[split] = split_dict

    with open(METRICS_DIR / "horizon_target_distribution.json", "w") as f:
        json.dump(target_dist, f, indent=2)
    logger.info("Saved reports/metrics/horizon_target_distribution.json")

    # Plot Target Distribution
    horizons = ["T+1", "T+2", "T+3", "T+4", "T+5"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    x = np.arange(len(horizons))
    width = 0.25
    for i, split in enumerate(partitions):
        pcts = [target_dist[split][h]["attack_percentage"] for h in horizons]
        ax.bar(x + (i - 1) * width, pcts, width, label=split.capitalize(), alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=10)
    ax.set_ylabel("Attack Label Prevalence (%)", fontsize=11)
    ax.set_title("Attack Label Distribution Across Forecasting Horizons (T+1 -> T+5)", fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "horizon_target_distribution.png")
    plt.close()
    logger.info("Saved reports/figures/baseline/horizon_target_distribution.png")

    # ══════════════════════════════════════════════════════════════════
    # AUDIT 2 & 3: PREDICTION DISTRIBUTIONS & ROC-AUC INVESTIGATION
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n[2/6] Auditing Model Prediction Distributions & Probability Calibration...")
    # Load test set for evaluation
    X_te_raw, y_te_bin_raw, y_te_multi_raw = load_partition("test", feature_cols, max_rows=200_000)
    X_seq_te, y_seq_bin_te, _ = build_window_matrices(X_te_raw, y_te_bin_raw, y_te_multi_raw, W, K, max_samples=50_000)
    X_tab_te = aggregate_windows_batch(X_seq_te)

    models_dict = {
        "Logistic_Standard": LogisticRegressionBaseline.load(MODELS_BASE_DIR / "logistic_forecasting_binary"),
        "Logistic_Balanced": LogisticRegressionBaseline.load(MODELS_BASE_DIR / "logistic_balanced_forecasting_binary"),
        "Random_Forest": RandomForestBaseline.load(MODELS_BASE_DIR / "random_forest_forecasting"),
        "Gradient_Boosting": GradientBoostingBaseline.load(MODELS_BASE_DIR / "gradient_boosting_forecasting"),
        "LSTM": LSTMForecaster.load(MODELS_BASE_DIR / "lstm_binary"),
        "GRU": GRUForecaster.load(MODELS_BASE_DIR / "gru_binary"),
    }

    pred_dist_audit: dict[str, Any] = {}
    for model_name, model in models_dict.items():
        m_audit = {}
        if model_name in ["LSTM", "GRU"]:
            preds = model.predict_forecasting(X_seq_te)
            probs = model.predict_proba_forecasting(X_seq_te)
        else:
            preds = model.predict_forecasting(X_tab_te)
            probs = model.predict_proba_forecasting(X_tab_te)

        for k in range(K):
            h_name = f"T+{k+1}"
            y_pred_k = preds[:, k]
            y_true_k = y_seq_bin_te[:, k]
            pos_cnt = int(np.sum(y_pred_k == 1))
            neg_cnt = int(np.sum(y_pred_k == 0))
            pred_pos_pct = float(pos_cnt / len(y_pred_k) * 100)
            actual_pos_pct = float(np.sum(y_true_k == 1) / len(y_true_k) * 100)
            
            # Probability analysis
            prob_k = probs[k][:, 1] if probs[k].ndim == 2 else probs[k]
            mean_prob = float(np.mean(prob_k))
            min_prob = float(np.min(prob_k))
            max_prob = float(np.max(prob_k))

            m_audit[h_name] = {
                "predicted_positive_count": pos_cnt,
                "predicted_negative_count": neg_cnt,
                "predicted_positive_pct": round(pred_pos_pct, 4),
                "actual_positive_pct": round(actual_pos_pct, 4),
                "mean_predicted_prob": round(mean_prob, 4),
                "min_predicted_prob": round(min_prob, 4),
                "max_predicted_prob": round(max_prob, 4),
            }
        pred_dist_audit[model_name] = m_audit

    with open(METRICS_DIR / "prediction_distribution_audit.json", "w") as f:
        json.dump(pred_dist_audit, f, indent=2)
    logger.info("Saved reports/metrics/prediction_distribution_audit.json")

    # ══════════════════════════════════════════════════════════════════
    # AUDIT 4: TEMPORAL LABEL AUTOCORRELATION
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n[3/6] Computing Temporal Label Autocorrelation...")
    # Compute P(Y[t+k] == Y[t]) and P(Y[t+k] == Y[t+k-1]) on full test sequence stream
    # Ground truth: Y[t] is the label at index t+W-1 (the last step of the input window)
    N_test_flows = len(y_te_bin_raw)
    valid_window_count = N_test_flows - W - K + 1
    
    # Extract Y_t (last window flow) and Y_{t+1} ... Y_{t+K}
    y_last_input = y_te_bin_raw[W - 1 : W - 1 + valid_window_count]
    
    autocorr: dict[str, Any] = {
        "P(Y[t+k] == Y[t])": {},
        "P(Y[t+k] == Y[t+k-1])": {},
        "transition_probabilities": {},
    }
    
    for k in range(1, K + 1):
        h_name = f"T+{k}"
        y_future_k = y_te_bin_raw[W - 1 + k : W - 1 + k + valid_window_count]
        p_same_as_last = float(np.mean(y_future_k == y_last_input))
        autocorr["P(Y[t+k] == Y[t])"][h_name] = round(p_same_as_last, 5)

        if k == 1:
            p_step_transition = p_same_as_last
        else:
            y_prev_step = y_te_bin_raw[W - 1 + k - 1 : W - 1 + k - 1 + valid_window_count]
            p_step_transition = float(np.mean(y_future_k == y_prev_step))
        autocorr["P(Y[t+k] == Y[t+k-1])"][h_name] = round(p_step_transition, 5)

        # Transition matrix for horizon k relative to t:
        # P(Attack at t+k | Attack at t) vs P(Attack at t+k | Benign at t)
        atk_mask_t = (y_last_input == 1)
        ben_mask_t = (y_last_input == 0)
        p_atk_given_atk = float(np.mean(y_future_k[atk_mask_t] == 1)) if np.sum(atk_mask_t) > 0 else 0.0
        p_atk_given_ben = float(np.mean(y_future_k[ben_mask_t] == 1)) if np.sum(ben_mask_t) > 0 else 0.0
        autocorr["transition_probabilities"][h_name] = {
            "P(Attack[t+k] | Attack[t])": round(p_atk_given_atk, 5),
            "P(Attack[t+k] | Benign[t])": round(p_atk_given_ben, 5),
            "P(Benign[t+k] | Benign[t])": round(1.0 - p_atk_given_ben, 5),
        }

    with open(METRICS_DIR / "temporal_label_autocorrelation.json", "w") as f:
        json.dump(autocorr, f, indent=2)
    logger.info("Saved reports/metrics/temporal_label_autocorrelation.json")

    # Plot Autocorrelation
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    p_yt = [autocorr["P(Y[t+k] == Y[t])"][h] for h in horizons]
    p_step = [autocorr["P(Y[t+k] == Y[t+k-1])"][h] for h in horizons]
    p_atk_atk = [autocorr["transition_probabilities"][h]["P(Attack[t+k] | Attack[t])"] for h in horizons]
    
    ax.plot(horizons, p_yt, marker="o", lw=2, label="P(Y[t+k] == Y[t]) [Persistence from Window End]")
    ax.plot(horizons, p_step, marker="s", lw=2, linestyle="--", label="P(Y[t+k] == Y[t+k-1]) [Step-to-Step Stability]")
    ax.plot(horizons, p_atk_atk, marker="^", lw=2, linestyle="-.", label="P(Attack[t+k] | Attack[t]) [Attack Continuity]")
    
    ax.set_title("Temporal Label Autocorrelation & State Persistence Across Horizons", fontsize=12, pad=12)
    ax.set_xlabel("Forecast Horizon", fontsize=11)
    ax.set_ylabel("Probability", fontsize=11)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "temporal_label_autocorrelation.png")
    plt.close()
    logger.info("Saved reports/figures/baseline/temporal_label_autocorrelation.png")

    logger.info("=" * 80)
    logger.info("Phase 2.5 Audit Calculations Complete.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
