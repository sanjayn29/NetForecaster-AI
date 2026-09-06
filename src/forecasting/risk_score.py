"""
src/forecasting/risk_score.py
-----------------------------
Enterprise Threat Risk Scoring and Model Confidence Engine (Phase 5).

Converts multi-horizon attack forecasting probabilities (T+1 ... T+5) into:
  1. Horizon-weighted Risk Score (0 - 100)
  2. Standardized Categorical Risk Level (LOW, MEDIUM, HIGH, CRITICAL)
  3. Model Decision Margin Confidence Metric (0.0 - 1.0)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


DEFAULT_HORIZON_WEIGHTS = [0.30, 0.25, 0.20, 0.15, 0.10]
DEFAULT_RISK_THRESHOLDS = {
    "LOW": (0.0, 25.0),
    "MEDIUM": (25.0, 50.0),
    "HIGH": (50.0, 75.0),
    "CRITICAL": (75.0, 100.0),
}


def calculate_risk_score(
    probabilities: Union[np.ndarray, List[float]],
    weights: Optional[List[float]] = None,
) -> float:
    """
    Compute enterprise threat risk score from multi-horizon attack probabilities.

    Formula:
        Risk = 100 * sum(w_k * P(Attack at T+k)) / sum(w)

    Args:
        probabilities: Array or list of future attack probabilities [P_{T+1}, ..., P_{T+K}]
        weights: Optional list of horizon importance weights (defaults to [0.30, 0.25, 0.20, 0.15, 0.10])

    Returns:
        Calibrated enterprise risk score in range [0.0, 100.0]
    """
    probs = np.asarray(probabilities, dtype=float).ravel()
    if len(probs) == 0:
        return 0.0

    if weights is None:
        if len(probs) <= len(DEFAULT_HORIZON_WEIGHTS):
            w = np.array(DEFAULT_HORIZON_WEIGHTS[: len(probs)], dtype=float)
        else:
            w = np.ones(len(probs), dtype=float)
    else:
        w = np.array(weights[: len(probs)], dtype=float)

    # Normalize weights
    w = w / np.sum(w)
    score = float(np.sum(w * probs) * 100.0)
    return float(np.clip(score, 0.0, 100.0))


def get_risk_level(
    risk_score: float,
    thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> str:
    """
    Categorize numerical risk score into an operational SOC severity level.

    Categories:
        0 - 24.99   : LOW
        25 - 49.99  : MEDIUM
        50 - 74.99  : HIGH
        75 - 100.00 : CRITICAL
    """
    score = float(np.clip(risk_score, 0.0, 100.0))
    if score < 25.0:
        return "LOW"
    elif score < 50.0:
        return "MEDIUM"
    elif score < 75.0:
        return "HIGH"
    else:
        return "CRITICAL"


def calculate_forecast_confidence(
    probabilities: Union[np.ndarray, List[float]],
) -> float:
    """
    Calculate model decision margin confidence across all forecasting horizons.

    Definition:
        Confidence = mean(2 * |P_{T+k} - 0.5|) in [0.0, 1.0]

    Note:
        This represents model certainty relative to the 0.5 decision boundary.
        It is NOT a calibrated probability of correctness.
    """
    probs = np.asarray(probabilities, dtype=float).ravel()
    if len(probs) == 0:
        return 0.0

    # Distance from 0.5 boundary scaled to [0, 1]
    certainties = 2.0 * np.abs(probs - 0.5)
    conf = float(np.mean(certainties))
    return float(np.clip(conf, 0.0, 1.0))


class RiskScorer:
    """
    Configurable Risk Scorer for Security Operations Center pipelines.
    """

    def __init__(
        self,
        weights: Optional[List[float]] = None,
    ) -> None:
        self.weights = weights or DEFAULT_HORIZON_WEIGHTS

    def evaluate(
        self,
        probabilities: Union[np.ndarray, List[float]],
    ) -> Dict[str, Any]:
        """
        Evaluate full risk profile from multi-horizon probabilities.
        """
        score = calculate_risk_score(probabilities, self.weights)
        level = get_risk_level(score)
        confidence = calculate_forecast_confidence(probabilities)

        return {
            "risk_score": round(score, 2),
            "risk_level": level,
            "confidence": round(confidence, 4),
            "horizon_weights": self.weights,
        }
