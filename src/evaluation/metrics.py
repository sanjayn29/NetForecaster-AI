"""
src/evaluation/metrics.py
-------------------------
Comprehensive evaluation metric calculation for single-step classification
and multi-horizon temporal attack forecasting.

Calculates:
  - Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1
  - Per-class Precision, Recall, F1, Support
  - False Positive Rate (FPR) = FP / (FP + TN)
  - ROC-AUC (Binary & Multiclass OvR)
  - PR-AUC (Average Precision Score)
  - Brier Calibration Score
  - Confusion Matrix
  - Horizon-wise Degradation Analysis (T+1 -> T+5)
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[list[str]] = None,
    benign_class_idx: int = 0,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics for a single prediction set.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth integer class labels (N,).
    y_pred : np.ndarray
        Predicted integer class labels (N,).
    y_prob : np.ndarray, optional
        Predicted class probabilities (N, C) or (N,) for binary.
    class_names : list[str], optional
        List of class names.
    benign_class_idx : int, default=0
        Index of the Benign/Normal class for binary FPR calculation.

    Returns
    -------
    dict[str, Any]
        Dictionary of computed metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    unique_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    n_classes = len(unique_labels)
    is_binary = n_classes <= 2 and max(unique_labels) <= 1

    # Overall accuracy
    acc = float(accuracy_score(y_true, y_pred))

    # Macro and Weighted metrics
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Per-class metrics
    p_per, r_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_labels, zero_division=0
    )

    per_class_dict: dict[str, dict[str, float]] = {}
    for i, lbl in enumerate(unique_labels):
        name = class_names[lbl] if class_names and lbl < len(class_names) else f"Class_{lbl}"
        per_class_dict[name] = {
            "precision": float(p_per[i]),
            "recall": float(r_per[i]),
            "f1": float(f1_per[i]),
            "support": int(sup_per[i]),
        }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    # False Positive Rate (FPR) for Benign (class 0) vs Any Attack
    # FPR = FP / (FP + TN) where negative is Benign
    benign_mask = y_true == benign_class_idx
    total_benign = int(np.sum(benign_mask))
    if total_benign > 0:
        false_positives = int(np.sum((y_true == benign_class_idx) & (y_pred != benign_class_idx)))
        fpr = float(false_positives / total_benign)
    else:
        fpr = 0.0

    metrics: dict[str, Any] = {
        "accuracy": acc,
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "weighted_precision": float(p_weighted),
        "weighted_recall": float(r_weighted),
        "false_positive_rate": fpr,
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(y_true),
        "is_binary": is_binary,
    }

    # Probability-based metrics (ROC-AUC, PR-AUC, Brier score)
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        try:
            if is_binary:
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 and y_prob.shape[1] == 2 else y_prob.ravel()
                metrics["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
                metrics["pr_auc"] = float(average_precision_score(y_true, prob_pos))
                metrics["brier_score"] = float(brier_score_loss(y_true, prob_pos))
            else:
                if y_prob.ndim == 2 and y_prob.shape[1] >= len(unique_labels):
                    # Multi-class OvR ROC-AUC
                    # Only calculate on classes present in y_true
                    classes_in_true = np.unique(y_true)
                    if len(classes_in_true) > 1:
                        metrics["roc_auc_ovr"] = float(
                            roc_auc_score(
                                y_true,
                                y_prob[:, classes_in_true],
                                multi_class="ovr",
                                labels=classes_in_true,
                            )
                        )
                    metrics["brier_score"] = float(
                        np.mean(np.sum((np.eye(y_prob.shape[1])[y_true] - y_prob) ** 2, axis=1))
                    )
        except Exception:
            pass

    return metrics


def compute_forecasting_metrics(
    y_true_seq: np.ndarray,
    y_pred_seq: np.ndarray,
    y_prob_seq: Optional[list[np.ndarray]] = None,
    class_names: Optional[list[str]] = None,
    benign_class_idx: int = 0,
) -> dict[str, Any]:
    """Compute horizon-by-horizon forecasting metrics for T+1 ... T+K.

    Parameters
    ----------
    y_true_seq : np.ndarray
        Ground truth sequence targets (N, K).
    y_pred_seq : np.ndarray
        Predicted sequence targets (N, K).
    y_prob_seq : list[np.ndarray], optional
        List of K probability matrices [prob_T1, prob_T2, ..., prob_TK].
    class_names : list[str], optional
        List of class names.
    benign_class_idx : int, default=0
        Index of benign class.

    Returns
    -------
    dict[str, Any]
        Horizon metrics dictionary with T+1 ... T+K and degradation summary.
    """
    K = y_true_seq.shape[1]
    horizon_results: dict[str, dict[str, Any]] = {}
    f1_decay: list[float] = []
    acc_decay: list[float] = []
    roc_decay: list[float] = []

    for k in range(K):
        h_name = f"T+{k + 1}"
        prob_k = y_prob_seq[k] if y_prob_seq is not None and k < len(y_prob_seq) else None
        h_metrics = compute_classification_metrics(
            y_true=y_true_seq[:, k],
            y_pred=y_pred_seq[:, k],
            y_prob=prob_k,
            class_names=class_names,
            benign_class_idx=benign_class_idx,
        )
        horizon_results[h_name] = h_metrics
        f1_decay.append(h_metrics["macro_f1"])
        acc_decay.append(h_metrics["accuracy"])
        if "roc_auc" in h_metrics:
            roc_decay.append(h_metrics["roc_auc"])
        elif "roc_auc_ovr" in h_metrics:
            roc_decay.append(h_metrics["roc_auc_ovr"])

    degradation = {
        "f1_macro_decay": f1_decay,
        "accuracy_decay": acc_decay,
        "f1_degradation_total": float(f1_decay[0] - f1_decay[-1]) if f1_decay else 0.0,
        "acc_degradation_total": float(acc_decay[0] - acc_decay[-1]) if acc_decay else 0.0,
    }
    if roc_decay:
        degradation["roc_auc_decay"] = roc_decay
        degradation["roc_degradation_total"] = float(roc_decay[0] - roc_decay[-1])

    return {
        "horizons": horizon_results,
        "degradation": degradation,
        "forecast_horizon_k": K,
    }
