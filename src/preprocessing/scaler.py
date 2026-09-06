"""
src/preprocessing/scaler.py
-----------------------------
Feature scaling for NetForecaster AI.

Design decisions
----------------
- Uses RobustScaler by default (config: features.scaler = "robust").
  Rationale: network traffic features have extreme outliers (e.g., DDoS flows
  with millions of packets/second). RobustScaler uses median and IQR,
  making it more robust to these outliers than StandardScaler.

- The scaler is fitted ONLY on the training portion of the data.
  It is then applied to validation and test sets using the training
  fit. This is critical to prevent preprocessing leakage.

- Scalers are saved to disk for reproducibility and inference.

- Only numerical feature columns are scaled. Label, Timestamp, and
  provenance columns are excluded.

- Categorical columns (Protocol, Dst Port when used as ordinal) are
  encoded separately and not included in scaling.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.utils.logger import get_logger
from src.utils.config import CFG

logger = get_logger(__name__)

# Columns that should NEVER be scaled
_NON_FEATURE_COLS = {
    CFG.columns.timestamp,
    CFG.columns.label,
    "label_encoded",
    "label_binary",
    "source_file",
    "source_day",
}

_SCALER_REGISTRY = {
    "robust": RobustScaler,
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
}


class FeatureScaler:
    """Wrapper around scikit-learn scalers for NetForecaster feature columns.

    Attributes
    ----------
    method : str
        Scaler method: 'robust', 'standard', or 'minmax'.
    feature_cols : list[str]
        The column names the scaler was fitted on.
    scaler : sklearn scaler instance
        The underlying fitted scaler.
    """

    def __init__(self, method: Optional[str] = None) -> None:
        if method is None:
            method = CFG.features.scaler
        if method not in _SCALER_REGISTRY:
            raise ValueError(
                f"Unknown scaler method '{method}'. "
                f"Choose from: {list(_SCALER_REGISTRY.keys())}"
            )
        self.method = method
        self.scaler = _SCALER_REGISTRY[method]()
        self.feature_cols: list[str] = []
        self._fitted = False

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Identify numeric columns eligible for scaling.

        Excludes: timestamp, label, provenance columns, and any columns
        whose name contains 'label' (case-insensitive).

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        list[str]
        """
        exclude = _NON_FEATURE_COLS | {
            col for col in df.columns if "label" in col.lower()
        }
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude]

    def fit(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> "FeatureScaler":
        """Fit scaler on training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training split DataFrame.
        feature_cols : list[str], optional
            Columns to scale. If None, auto-detected from df.

        Returns
        -------
        self
        """
        if feature_cols is None:
            feature_cols = self.get_feature_columns(df)
        self.feature_cols = feature_cols
        self.scaler.fit(df[feature_cols].values.astype(np.float64))
        self._fitted = True
        logger.info(
            "[FeatureScaler] Fitted %s on %d columns, %d rows.",
            self.method,
            len(feature_cols),
            len(df),
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scaler to a DataFrame.

        Only scales columns in ``self.feature_cols``. Other columns are
        returned unchanged.

        Parameters
        ----------
        df : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with scaled feature columns (all others unchanged).
        """
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call .fit() first.")

        df = df.copy()
        present = [c for c in self.feature_cols if c in df.columns]
        missing_at_transform = set(self.feature_cols) - set(present)
        if missing_at_transform:
            logger.warning(
                "[FeatureScaler] Columns fitted on but absent at transform time: %s",
                missing_at_transform,
            )

        scaled = self.scaler.transform(df[present].values.astype(np.float64))
        df[present] = scaled
        return df

    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fit and transform in one step (use ONLY on training data)."""
        self.fit(df, feature_cols)
        return self.transform(df)

    def save(self, path: str | Path) -> None:
        """Save scaler to disk using pickle.

        Parameters
        ----------
        path : str or Path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": self.method,
            "feature_cols": self.feature_cols,
            "scaler": self.scaler,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        logger.info("[FeatureScaler] Saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureScaler":
        """Load scaler from disk.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        FeatureScaler
        """
        with open(path, "rb") as f:
            payload = pickle.load(f)
        instance = cls(method=payload["method"])
        instance.feature_cols = payload["feature_cols"]
        instance.scaler = payload["scaler"]
        instance._fitted = True
        logger.info(
            "[FeatureScaler] Loaded %s scaler from %s (%d columns)",
            instance.method,
            path,
            len(instance.feature_cols),
        )
        return instance
