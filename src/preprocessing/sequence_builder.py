"""
src/preprocessing/sequence_builder.py
---------------------------------------
PyTorch Dataset wrapper for temporal sequence windows.

This module provides:
1. ``FlowSequenceDataset`` — a memory-efficient PyTorch Dataset that
   loads windows on-demand from a pre-sorted Parquet file and index list.
2. ``build_sequence_datasets`` — convenience function to create train/
   validation/test Dataset objects from a SplitResult.

Memory efficiency
-----------------
Instead of materialising a 3D array (n_windows × W × F) in RAM, this
Dataset:
- Stores the feature DataFrame as a numpy array once.
- Stores only a list of (start, end) index pairs.
- Returns individual windows on __getitem__ access.

This enables training on the full 16M-row dataset on a normal machine.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.preprocessing.windowing import (
    WindowIndex,
    build_window_indices,
    load_window_indices,
    save_window_indices,
)
from src.utils.logger import get_logger
from src.utils.config import CFG

logger = get_logger(__name__)

_LABEL_COL = CFG.columns.label


class FlowSequenceDataset:
    """Memory-efficient temporal window dataset for network flow sequences.

    Can be used directly as a PyTorch Dataset if PyTorch is available.
    Falls back to a simple indexable object if PyTorch is not installed.

    Parameters
    ----------
    feature_array : np.ndarray, shape (n_rows, n_features)
        All feature values, sorted chronologically.
    label_multi : np.ndarray, shape (n_rows,)
        Multi-class encoded labels.
    label_binary : np.ndarray, shape (n_rows,)
        Binary (attack/benign) labels.
    window_indices : List[WindowIndex]
        Pre-computed window index list.
    window_size : int
    forecast_horizon : int
    """

    def __init__(
        self,
        feature_array: np.ndarray,
        label_multi: np.ndarray,
        label_binary: np.ndarray,
        window_indices: List[WindowIndex],
        window_size: int,
        forecast_horizon: int,
    ) -> None:
        self.feature_array = feature_array
        self.label_multi = label_multi
        self.label_binary = label_binary
        self.window_indices = window_indices
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon

    def __len__(self) -> int:
        return len(self.window_indices)

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (X, y_multi, y_binary) for window at idx.

        Returns
        -------
        X : np.ndarray, shape (window_size, n_features)  — float32
        y_multi : np.ndarray, shape (forecast_horizon,)  — int64
        y_binary : np.ndarray, shape (forecast_horizon,) — int64
        """
        win = self.window_indices[idx]
        X = self.feature_array[win.input_start : win.input_end + 1].astype(np.float32)
        y_m = self.label_multi[win.target_start : win.target_end + 1].astype(np.int64)
        y_b = self.label_binary[win.target_start : win.target_end + 1].astype(np.int64)
        return X, y_m, y_b

    @property
    def input_shape(self) -> Tuple[int, int]:
        return (self.window_size, self.feature_array.shape[1])

    @property
    def n_classes(self) -> int:
        return int(self.label_multi.max()) + 1


# Optional: Register as a PyTorch Dataset if available
try:
    from torch.utils.data import Dataset as TorchDataset

    class TorchFlowSequenceDataset(FlowSequenceDataset, TorchDataset):
        """PyTorch-compatible version of FlowSequenceDataset."""

        def __getitem__(self, idx: int):
            import torch
            X, y_m, y_b = super().__getitem__(idx)
            return (
                torch.from_numpy(X),
                torch.from_numpy(y_m),
                torch.from_numpy(y_b),
            )

except ImportError:
    TorchFlowSequenceDataset = None  # type: ignore


def build_sequence_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    window_size: Optional[int] = None,
    forecast_horizon: Optional[int] = None,
    stride: Optional[int] = None,
    save_dir: Optional[str | Path] = None,
    cross_day: bool = True,
) -> Tuple[FlowSequenceDataset, FlowSequenceDataset, FlowSequenceDataset]:
    """Build train, validation, and test sequence datasets.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        The three chronological splits (already scaled and encoded).
    feature_cols : List[str]
        Ordered feature column names.
    window_size : int, optional
    forecast_horizon : int, optional
    stride : int, optional
    save_dir : str or Path, optional
        If provided, saves window index .npz files here.
    cross_day : bool
        Whether to allow windows that cross day boundaries.

    Returns
    -------
    train_ds, val_ds, test_ds : FlowSequenceDataset
    """
    if window_size is None:
        window_size = CFG.windowing.window_size
    if forecast_horizon is None:
        forecast_horizon = CFG.windowing.forecast_horizon
    if stride is None:
        stride = CFG.windowing.stride

    datasets: list[FlowSequenceDataset] = []
    split_names = ["train", "validation", "test"]
    split_dfs = [train_df, val_df, test_df]

    for name, sdf in zip(split_names, split_dfs):
        logger.info("[SequenceBuilder] Building '%s' dataset — %d rows.", name, len(sdf))

        # Extract arrays
        feat_arr = sdf[feature_cols].values.astype(np.float32)
        label_multi = sdf["label_encoded"].values.astype(np.int64) if "label_encoded" in sdf.columns else np.zeros(len(sdf), dtype=np.int64)
        label_binary = sdf["label_binary"].values.astype(np.int64) if "label_binary" in sdf.columns else np.zeros(len(sdf), dtype=np.int64)

        day_labels: Optional[np.ndarray] = None
        if "source_day" in sdf.columns and not cross_day:
            day_labels = sdf["source_day"].values

        indices = build_window_indices(
            n_rows=len(sdf),
            window_size=window_size,
            forecast_horizon=forecast_horizon,
            stride=stride,
            day_labels=day_labels,
            cross_day=cross_day,
        )

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_window_indices(indices, str(save_dir / f"{name}_window_indices.npz"))

        # Use PyTorch-compatible dataset if available
        ds_class = TorchFlowSequenceDataset if TorchFlowSequenceDataset is not None else FlowSequenceDataset
        ds = ds_class(
            feature_array=feat_arr,
            label_multi=label_multi,
            label_binary=label_binary,
            window_indices=indices,
            window_size=window_size,
            forecast_horizon=forecast_horizon,
        )
        datasets.append(ds)

        logger.info(
            "[SequenceBuilder] '%s' -> %d windows | input shape: %s",
            name,
            len(ds),
            ds.input_shape,
        )

    return tuple(datasets)  # type: ignore
