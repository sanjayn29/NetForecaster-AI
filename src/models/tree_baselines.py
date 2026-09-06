"""
src/models/tree_baselines.py
----------------------------
Non-linear tree-based baselines for NetForecaster AI:
  1. Random Forest Baseline (RandomForestBaseline)
  2. HistGradientBoosting Baseline (HistGradientBoostingBaseline)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


class RandomForestBaseline:
    """Random Forest baseline for single-step and K-horizon attack forecasting.

    Parameters
    ----------
    n_estimators : int, default=100
    max_depth : int, default=15
    task : str, default="binary"
    forecast_horizon_k : int, default=5
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 15,
        task: str = "binary",
        forecast_horizon_k: int = 5,
        class_weight: Optional[str] = "balanced_subsample",
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.task = task
        self.K = forecast_horizon_k
        self.class_weight = class_weight
        self.random_state = random_state

        self.single_model: Optional[RandomForestClassifier] = None
        self.horizon_models: list[RandomForestClassifier] = []
        self.feature_names_: list[str] = []
        self.classes_: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def _create_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=1,
        )

    def fit_single_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> RandomForestBaseline:
        self.single_model = self._create_estimator()
        self.single_model.fit(X, y)
        self.classes_ = self.single_model.classes_
        if feature_names is not None:
            self.feature_names_ = feature_names
        self.is_fitted = True
        return self

    def predict_single_step(self, X: np.ndarray) -> np.ndarray:
        if self.single_model is None:
            raise RuntimeError("Model not fitted.")
        return self.single_model.predict(X)

    def predict_proba_single_step(self, X: np.ndarray) -> np.ndarray:
        if self.single_model is None:
            raise RuntimeError("Model not fitted.")
        return self.single_model.predict_proba(X)

    def fit_forecasting(
        self,
        X: np.ndarray,
        y_seq: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> RandomForestBaseline:
        N, K = y_seq.shape
        self.K = K
        self.horizon_models = []
        for k in range(K):
            clf = self._create_estimator()
            clf.fit(X, y_seq[:, k])
            self.horizon_models.append(clf)

        self.classes_ = self.horizon_models[0].classes_
        if feature_names is not None:
            self.feature_names_ = feature_names
        self.is_fitted = True
        return self

    def predict_forecasting(self, X: np.ndarray) -> np.ndarray:
        if not self.horizon_models:
            raise RuntimeError("Forecasting models not fitted.")
        preds = [clf.predict(X) for clf in self.horizon_models]
        return np.column_stack(preds)

    def predict_proba_forecasting(self, X: np.ndarray) -> list[np.ndarray]:
        if not self.horizon_models:
            raise RuntimeError("Forecasting models not fitted.")
        return [clf.predict_proba(X) for clf in self.horizon_models]

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_type": "RandomForest",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "task": self.task,
            "forecast_horizon_k": self.K,
            "classes": self.classes_.tolist() if hasattr(self.classes_, "tolist") else list(self.classes_),
            "feature_names": self.feature_names_,
        }
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if self.single_model is not None:
            joblib.dump(self.single_model, save_dir / "single_model.joblib")
        for k, clf in enumerate(self.horizon_models):
            joblib.dump(clf, save_dir / f"horizon_{k + 1}_model.joblib")

    @classmethod
    def load(cls, save_dir: str | Path) -> RandomForestBaseline:
        save_dir = Path(save_dir)
        with open(save_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        inst = cls(
            n_estimators=meta["n_estimators"],
            max_depth=meta["max_depth"],
            task=meta["task"],
            forecast_horizon_k=meta["forecast_horizon_k"],
        )
        inst.feature_names_ = meta.get("feature_names", [])
        inst.classes_ = np.array(meta.get("classes", []))
        if (save_dir / "single_model.joblib").exists():
            inst.single_model = joblib.load(save_dir / "single_model.joblib")
        horizon_models = []
        for k in range(meta["forecast_horizon_k"]):
            path = save_dir / f"horizon_{k + 1}_model.joblib"
            if path.exists():
                horizon_models.append(joblib.load(path))
        inst.horizon_models = horizon_models
        inst.is_fitted = True
        return inst


class GradientBoostingBaseline:
    """Gradient Boosting baseline using Scikit-Learn GradientBoostingClassifier."""

    def __init__(
        self,
        n_estimators: int = 50,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        task: str = "binary",
        forecast_horizon_k: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.task = task
        self.K = forecast_horizon_k
        self.random_state = random_state

        self.single_model: Optional[Any] = None
        self.horizon_models: list[Any] = []
        self.feature_names_: list[str] = []
        self.classes_: np.ndarray = np.array([])
        self.is_fitted: bool = False

    def _create_estimator(self) -> Any:
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )

    def fit_single_step(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> GradientBoostingBaseline:
        self.single_model = self._create_estimator()
        self.single_model.fit(X, y)
        self.classes_ = self.single_model.classes_
        if feature_names is not None:
            self.feature_names_ = feature_names
        self.is_fitted = True
        return self

    def predict_single_step(self, X: np.ndarray) -> np.ndarray:
        if self.single_model is None:
            raise RuntimeError("Model not fitted.")
        return self.single_model.predict(X)

    def predict_proba_single_step(self, X: np.ndarray) -> np.ndarray:
        if self.single_model is None:
            raise RuntimeError("Model not fitted.")
        return self.single_model.predict_proba(X)

    def fit_forecasting(
        self,
        X: np.ndarray,
        y_seq: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> GradientBoostingBaseline:
        N, K = y_seq.shape
        self.K = K
        self.horizon_models = []
        for k in range(K):
            clf = self._create_estimator()
            clf.fit(X, y_seq[:, k])
            self.horizon_models.append(clf)

        self.classes_ = self.horizon_models[0].classes_
        if feature_names is not None:
            self.feature_names_ = feature_names
        self.is_fitted = True
        return self

    def predict_forecasting(self, X: np.ndarray) -> np.ndarray:
        if not self.horizon_models:
            raise RuntimeError("Forecasting models not fitted.")
        preds = [clf.predict(X) for clf in self.horizon_models]
        return np.column_stack(preds)

    def predict_proba_forecasting(self, X: np.ndarray) -> list[np.ndarray]:
        if not self.horizon_models:
            raise RuntimeError("Forecasting models not fitted.")
        return [clf.predict_proba(X) for clf in self.horizon_models]

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_type": "GradientBoosting",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "task": self.task,
            "forecast_horizon_k": self.K,
            "classes": self.classes_.tolist() if hasattr(self.classes_, "tolist") else list(self.classes_),
            "feature_names": self.feature_names_,
        }
        with open(save_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        if self.single_model is not None:
            joblib.dump(self.single_model, save_dir / "single_model.joblib")
        for k, clf in enumerate(self.horizon_models):
            joblib.dump(clf, save_dir / f"horizon_{k + 1}_model.joblib")

    @classmethod
    def load(cls, save_dir: str | Path) -> GradientBoostingBaseline:
        save_dir = Path(save_dir)
        with open(save_dir / "metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        inst = cls(
            n_estimators=meta.get("n_estimators", 50),
            max_depth=meta.get("max_depth", 4),
            learning_rate=meta.get("learning_rate", 0.1),
            task=meta["task"],
            forecast_horizon_k=meta["forecast_horizon_k"],
        )
        inst.feature_names_ = meta.get("feature_names", [])
        inst.classes_ = np.array(meta.get("classes", []))
        if (save_dir / "single_model.joblib").exists():
            inst.single_model = joblib.load(save_dir / "single_model.joblib")
        horizon_models = []
        for k in range(meta["forecast_horizon_k"]):
            path = save_dir / f"horizon_{k + 1}_model.joblib"
            if path.exists():
                horizon_models.append(joblib.load(path))
        inst.horizon_models = horizon_models
        inst.is_fitted = True
        return inst
