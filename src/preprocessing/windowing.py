"""
src/preprocessing/windowing.py
--------------------------------
Temporal window builder for NetForecaster AI.

This module creates sliding temporal windows over sorted flow sequences.

Window structure
----------------
Given WINDOW_SIZE = W and FORECAST_HORIZON = K:

  Input window (past):     [t-W+1, t-W+2, ..., t]    → shape (W, n_features)
  Target (future):         [t+1, t+2, ..., t+K]       → shape (K,) labels
  Target binary:           [t+1, ..., t+K]             → shape (K,) attack indicators
  Target stage:            [t+1, ..., t+K]             → shape (K,) stage indices

CRITICAL: No information from the target window enters the input window.
The boundary between input and target is strict.

Inter-day gap handling
----------------------
The dataset has inter-day gaps (e.g., between 2018-02-16 and 2018-02-20).
A window that spans a day boundary may capture a temporal gap that does
not represent real network continuity. Two strategies are supported:

1. ``cross_day=True`` (default):  Allow windows to cross day boundaries.
   This is acceptable because the model can learn day-level features.

2. ``cross_day=False``:  Only create windows within a single source_day.
   This is stricter and avoids the gap problem but reduces sample count.

Memory strategy
---------------
For 16M rows with W=20 and stride=1, the number of windows is ~16M.
Each window stores W=20 rows × F features (e.g., 67 features).
Storing all windows in RAM simultaneously (~16M × 20 × 67 × 4 bytes) ≈ 80 GB.

Therefore:
- Windows are NOT stored as a dense 3D numpy array.
- Instead, we store only the (start_idx, end_idx) index pairs.
- The actual feature matrix is created on-demand in the Dataset class
  (see sequence_builder.py / dataset.py).

This index-based approach uses O(n) memory instead of O(n × W × F).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.config import CFG

logger = get_logger(__name__)

_TS_COL = CFG.columns.timestamp


@dataclass
class WindowIndex:
    """Lightweight index of a single temporal window.

    Attributes
    ----------
    input_start : int
        Index of the first row in the input window (inclusive).
    input_end : int
        Index of the last row in the input window (inclusive).
        input_end - input_start + 1 == window_size
    target_start : int
        Index of the first row in the target window.
    target_end : int
        Index of the last row in the target window (inclusive).
    """

    input_start: int
    input_end: int
    target_start: int
    target_end: int


def build_window_indices(
    n_rows: int,
    window_size: Optional[int] = None,
    forecast_horizon: Optional[int] = None,
    stride: Optional[int] = None,
    day_labels: Optional[np.ndarray] = None,
    cross_day: bool = True,
) -> List[WindowIndex]:
    """Build sliding window index pairs over a sorted sequence.

    Parameters
    ----------
    n_rows : int
        Total number of rows in the sequence.
    window_size : int, optional
        Number of past steps. Defaults to config value.
    forecast_horizon : int, optional
        Number of future steps to predict. Defaults to config value.
    stride : int, optional
        Step size between windows. Defaults to config value.
    day_labels : np.ndarray, optional
        Array of day identifiers (e.g., "Wednesday-14-02-2018") for each row.
        Required when ``cross_day=False``.
    cross_day : bool
        If False, skip windows that cross source_day boundaries.

    Returns
    -------
    List[WindowIndex]
    """
    if window_size is None:
        window_size = CFG.windowing.window_size
    if forecast_horizon is None:
        forecast_horizon = CFG.windowing.forecast_horizon
    if stride is None:
        stride = CFG.windowing.stride

    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if forecast_horizon < 1:
        raise ValueError(f"forecast_horizon must be >= 1, got {forecast_horizon}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    total_needed = window_size + forecast_horizon
    if n_rows < total_needed:
        logger.warning(
            "[Windowing] n_rows=%d < window_size+forecast_horizon=%d. No windows created.",
            n_rows,
            total_needed,
        )
        return []

    indices: List[WindowIndex] = []
    n_skipped = 0

    for i in range(0, n_rows - total_needed + 1, stride):
        input_start = i
        input_end = i + window_size - 1
        target_start = i + window_size
        target_end = i + window_size + forecast_horizon - 1

        # Optional: reject windows that cross day boundaries
        if not cross_day and day_labels is not None:
            input_days = day_labels[input_start : input_end + 1]
            target_days = day_labels[target_start : target_end + 1]
            all_days = np.concatenate([input_days, target_days])
            if len(np.unique(all_days)) > 1:
                n_skipped += 1
                continue

        indices.append(WindowIndex(input_start, input_end, target_start, target_end))

    logger.info(
        "[Windowing] Created %d window indices (W=%d, K=%d, stride=%d, skipped=%d cross-day).",
        len(indices),
        window_size,
        forecast_horizon,
        stride,
        n_skipped,
    )
    return indices


def windows_from_indices(
    df: pd.DataFrame,
    indices: List[WindowIndex],
    feature_cols: List[str],
    label_col: str = "label_encoded",
    binary_col: str = "label_binary",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialise window arrays from index list.

    WARNING: This is memory-intensive for large datasets.
    For training, use the index-based Dataset class instead.
    Use this function only for small validation/test sets or debugging.

    Parameters
    ----------
    df : pd.DataFrame
        Sorted feature DataFrame with label columns.
    indices : List[WindowIndex]
    feature_cols : List[str]
        Ordered list of feature column names.
    label_col : str
        Multi-class encoded label column.
    binary_col : str
        Binary attack label column.

    Returns
    -------
    X : np.ndarray, shape (n_windows, window_size, n_features)
    y_multi : np.ndarray, shape (n_windows, forecast_horizon)
    y_binary : np.ndarray, shape (n_windows, forecast_horizon)
    """
    feature_arr = df[feature_cols].values.astype(np.float32)
    label_arr = df[label_col].values.astype(np.int64) if label_col in df.columns else None
    binary_arr = df[binary_col].values.astype(np.int64) if binary_col in df.columns else None

    n = len(indices)
    W = indices[0].input_end - indices[0].input_start + 1
    K = indices[0].target_end - indices[0].target_start + 1
    F = len(feature_cols)

    X = np.zeros((n, W, F), dtype=np.float32)
    y_multi = np.full((n, K), -1, dtype=np.int64)
    y_binary = np.zeros((n, K), dtype=np.int64)

    for j, win in enumerate(indices):
        X[j] = feature_arr[win.input_start : win.input_end + 1]
        if label_arr is not None:
            y_multi[j] = label_arr[win.target_start : win.target_end + 1]
        if binary_arr is not None:
            y_binary[j] = binary_arr[win.target_start : win.target_end + 1]

    logger.info(
        "[Windowing] Materialised: X=%s, y_multi=%s, y_binary=%s",
        X.shape,
        y_multi.shape,
        y_binary.shape,
    )
    return X, y_multi, y_binary


def save_window_indices(indices: List[WindowIndex], path: str) -> None:
    """Save window indices to a .npz file for fast reloading.

    Parameters
    ----------
    indices : List[WindowIndex]
    path : str
        Output file path (e.g., ``data/processed/train_windows.npz``).
    """
    arr = np.array(
        [(w.input_start, w.input_end, w.target_start, w.target_end) for w in indices],
        dtype=np.int64,
    )
    np.savez_compressed(path, window_indices=arr)
    logger.info("[Windowing] Saved %d window indices to %s", len(indices), path)


def load_window_indices(path: str) -> List[WindowIndex]:
    """Load window indices from a .npz file.

    Parameters
    ----------
    path : str

    Returns
    -------
    List[WindowIndex]
    """
    data = np.load(path)
    arr = data["window_indices"]
    indices = [
        WindowIndex(int(row[0]), int(row[1]), int(row[2]), int(row[3]))
        for row in arr
    ]
    logger.info("[Windowing] Loaded %d window indices from %s", len(indices), path)
    return indices
