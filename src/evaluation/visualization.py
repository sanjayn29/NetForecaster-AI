"""
src/evaluation/visualization.py
-------------------------------
Figure generation utilities for NetForecaster AI Phase 2:
  - Confusion matrix heatmap
  - ROC and Precision-Recall curves
  - Horizon-wise performance degradation plots (T+1 -> T+5)
  - Logistic regression coefficient feature importance bar plots
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI safety
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str | Path] = None,
    normalize: bool = True,
) -> None:
    """Plot and save a confusion matrix."""
    if normalize:
        cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-12)
    else:
        cm_norm = cm

    plt.figure(figsize=(10, 8), dpi=150)
    plt.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title, fontsize=14, pad=15)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right", fontsize=9)
    plt.yticks(tick_marks, class_names, fontsize=9)

    plt.ylabel("True Label", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_horizon_degradation(
    decay_dict: dict[str, list[float]],
    horizons: Sequence[str] = ("T+1", "T+2", "T+3", "T+4", "T+5"),
    title: str = "Forecasting Performance Degradation Across Horizons",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot metric degradation across forecast horizons T+1 to T+K."""
    plt.figure(figsize=(8, 5), dpi=150)

    for metric_name, values in decay_dict.items():
        plt.plot(horizons[: len(values)], values, marker="o", linewidth=2, label=metric_name)

    plt.title(title, fontsize=13, pad=12)
    plt.xlabel("Forecast Horizon", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_logistic_feature_importance(
    feature_names: list[str],
    coefficients: np.ndarray,
    top_n: int = 25,
    title: str = "Top Predictive Features (Logistic Regression)",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot top positive and negative coefficients from Logistic Regression."""
    coefs = np.asarray(coefficients).ravel()
    if len(feature_names) != len(coefs):
        feature_names = [f"feat_{i}" for i in range(len(coefs))]

    # Sort by absolute magnitude
    abs_indices = np.argsort(np.abs(coefs))[::-1][:top_n]
    top_features = [feature_names[i] for i in abs_indices][::-1]
    top_coefs = [coefs[i] for i in abs_indices][::-1]

    colors = ["crimson" if c > 0 else "royalblue" for c in top_coefs]

    plt.figure(figsize=(10, 8), dpi=150)
    plt.barh(range(len(top_features)), top_coefs, color=colors, alpha=0.85)
    plt.yticks(range(len(top_features)), top_features, fontsize=9)
    plt.axvline(0, color="black", linestyle="--", linewidth=0.8)
    plt.title(title, fontsize=13, pad=12)
    plt.xlabel("Coefficient Weight (Positive = Attack, Negative = Benign)", fontsize=10)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[str | Path] = None,
    label: str = "Model",
) -> None:
    """Plot binary ROC curve with AUC."""
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6), dpi=150)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"{label} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", alpha=0.6)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11)
    plt.ylabel("True Positive Rate", fontsize=11)
    plt.title(title, fontsize=13, pad=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[str | Path] = None,
    label: str = "Model",
) -> None:
    """Plot binary Precision-Recall curve with AP."""
    from sklearn.metrics import precision_recall_curve, average_precision_score

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(7, 6), dpi=150)
    plt.plot(recall, precision, color="steelblue", lw=2, label=f"{label} (AP = {ap:.4f})")
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.title(title, fontsize=13, pad=12)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()


def plot_model_comparison(
    model_names: list[str],
    metric_values: dict[str, list[float]],
    title: str = "Model Comparison",
    save_path: Optional[str | Path] = None,
) -> None:
    """Bar chart comparing multiple models on multiple metrics."""
    n_models = len(model_names)
    n_metrics = len(metric_values)
    x = np.arange(n_models)
    width = 0.8 / max(n_metrics, 1)
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.5), 6), dpi=150)
    for i, (metric_name, values) in enumerate(metric_values.items()):
        ax.bar(x + i * width, values[:n_models], width, label=metric_name, color=colors[i], alpha=0.85)

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xticks(x + width * (n_metrics - 1) / 2)
    ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    plt.close()

