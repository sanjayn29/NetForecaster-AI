"""
src/models/logistic_baseline.py
-------------------------------
Logistic Regression baselines for NetForecaster AI:
  1. Single-step Flow Classifier (Standard & Balanced)
  2. Multi-Horizon Forecasting Baseline (5 independent horizon models: T+1 ... T+5)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticRegressionBaseline:
    """Logistic Regression model supporting single-step and K-horizon attack forecasting.

    Parameters
    ----------
    task : str, default="binary"
        "binary" (Benign vs Attack) or "multiclass" (15 classes).
    balanced : bool, default=False
        Whether to use class_weight="balanced".
    max_iter : int, default=1000
        Maximum solver iterations.
    C : float, default=1.0
        Inverse regularization strength.
    forecast_horizon_k : int, default=5
        Number of future steps to forecast.
    """

    def __init__(
        self,
        task: str = "binary",
        balanced: bool = False,
        max_iter: int = 1000,
        C: float = 1.0,
        forecast_horizon_k: int = 5,
        random_state: int = 42,
    ) -> None:
        self.task = task
        self.balanced = balanced
        self.max_iter = max_iter
        self.C = C
        self.K = forecast_horizon_k
        self.random_state = random_state

        self.class_weight = "balanced" if balanced else None

        # For single-step classification
        self.single_model: Optional[LogisticRegression] = None

        # For K-horizon forecasting: K independent models
        self.horizon_models: list[LogisticRegression] = []
        self.is_fitted: bool = False
        self.feature_names_: list[str] = []
        self.classes_: np.ndarray = np.array([])

    def _create_estimator(self) -> LogisticRegression:
        return LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=self.random_state,
            solver="lbfgs",
        )

    def fit_single_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> LogisticRegressionBaseline:
        """Fit single-step current flow classification."""
        self.single_model = self._create_estimator()
        self.single_model.fit(X, y)
        self.classes_ = self.single_model.classes_
        if feature_names is not None:
            self.feature_names_ = feature_names
        self.is_fitted = True
        return self

    def predict_single_step(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for single-step current flow."""
        if self.single_model is None:
            raise RuntimeError("Model has not been fitted on single-step task.")
        return self.single_model.predict(X)

    def predict_proba_single_step(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for single-step current flow."""
        if self.single_model is None:
            raise RuntimeError("Model has not been fitted on single-step task.")
        return self.single_model.predict_proba(X)

    def fit_forecasting(
        self,
        X: np.ndarray,
        y_seq: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> LogisticRegressionBaseline:
        """Fit K independent horizon-specific Logistic Regression models.

        Parameters
        ----------
        X : np.ndarray
            Input window representation of shape (N, D_agg).
        y_seq : np.ndarray
            Target sequences of shape (N, K).
        feature_names : list[str], optional
            Names of the input features.
        """
        N, K = y_seq.shape
        self.K = K
        self.horizon_models = []

        for k in range(K):
            y_k = y_seq[:, k]
            clf = self._create_estimator()
            clf.fit(X, y_k)
            self.horizon_models.append(clf)

        self.classes_ = self.horizon_models[0].classes_
        if feature_names is not None:
            self.feature_names_ = feature_names
        self.is_fitted = True
        return self

    def predict_forecasting(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for all K forecast horizons.

        Returns
        -------
        np.ndarray
            Predicted sequence matrix of shape (N, K).
        """
        if not self.horizon_models:
            raise RuntimeError("Forecasting models have not been fitted.")

        preds = [clf.predict(X) for clf in self.horizon_models]
        return np.column_stack(preds)

    def predict_proba_forecasting(self, X: np.ndarray) -> list[np.ndarray]:
        """Predict probability distributions for all K forecast horizons.

        Returns
        -------
        list[np.ndarray]
            List of K probability matrices.
        """
        if not self.horizon_models:
            raise RuntimeError("Forecasting models have not been fitted.")

        return [clf.predict_proba(X) for clf in self.horizon_models]

    def get_feature_importance(self, horizon_idx: int = 0) -> dict[str, float]:
        """Extract coefficient feature importance for a given horizon model."""
        if self.single_model is not None and not self.horizon_models:
            model = self.single_model
        elif self.horizon_models and horizon_idx < len(self.horizon_models):
            model = self.horizon_models[horizon_idx]
        else:
            raise RuntimeError("No model fitted.")

        coef = model.coef_
        if coef.shape[0] == 1:
            weights = coef[0]
        else:
            weights = np.mean(np.abs(coef), axis=0)

        names = self.feature_names_ if self.feature_names_ else [f"feat_{i}" for i in range(len(weights))]
        return {name: float(w) for name, w in zip(names, weights)}

    def save(self, save_dir: str | Path) -> None:
        """Save model artifacts, metadata, and coefficients."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "model_type": "LogisticRegression",
            "task": self.task,
            "balanced": self.balanced,
            "C": self.C,
            "max_iter": self.max_iter,
            "forecast_horizon_k": self.K,
            "classes": self.classes_.tolist() if hasattr(self.classes_, "tolist") else list(self.classes_),
            "n_features": len(self.feature_names_),
            "feature_names": self.feature_names_,
        }
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if self.single_model is not None:
            joblib.dump(self.single_model, save_dir / "single_model.joblib")

        if self.horizon_models:
            for k, clf in enumerate(self.horizon_models):
                joblib.dump(clf, save_dir / f"horizon_{k + 1}_model.joblib")

    @classmethod
    def load(cls, save_dir: str | Path) -> LogisticRegressionBaseline:
        """Load model artifacts and metadata from disk."""
        save_dir = Path(save_dir)
        with open(save_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(
            task=meta["task"],
            balanced=meta["balanced"],
            max_iter=meta["max_iter"],
            C=meta["C"],
            forecast_horizon_k=meta["forecast_horizon_k"],
        )
        instance.feature_names_ = meta.get("feature_names", [])
        instance.classes_ = np.array(meta.get("classes", []))

        if (save_dir / "single_model.joblib").exists():
            instance.single_model = joblib.load(save_dir / "single_model.joblib")

        horizon_models = []
        for k in range(meta["forecast_horizon_k"]):
            path = save_dir / f"horizon_{k + 1}_model.joblib"
            if path.exists():
                horizon_models.append(joblib.load(path))
        instance.horizon_models = horizon_models
        instance.is_fitted = True
        return instance
