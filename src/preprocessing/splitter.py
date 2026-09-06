"""
src/preprocessing/splitter.py
------------------------------
Chronological train/validation/test splitter for NetForecaster AI.

CRITICAL DESIGN PRINCIPLE:
This is a temporal forecasting project. A random split would leak future
information into the training set, invalidating all evaluation results.
All splits are strictly chronological.

Split strategy
--------------
Given the dataset spans 2018-02-14 to 2018-03-02 (10 days with data):

  TRAIN       = earliest 70% of rows (chronologically)
  VALIDATION  = next 15% of rows
  TEST        = latest 15% of rows

The boundary is computed on sorted row index (by timestamp), NOT by
calendar date, to respect the configured ratios exactly.

Day-boundary alignment (recommended):
  When possible, the split boundaries are aligned to day boundaries
  to avoid splitting a single day's flows across train and validation.
  This is optional and controlled by ``align_to_day_boundary``.

Preprocessing leakage prevention:
  The scaler, label encoder, and any other fitted transformers
  must be fitted ONLY on the training portion.
  Validation and test sets must be transformed using the training fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.config import CFG

logger = get_logger(__name__)

_TS_COL = CFG.columns.timestamp


@dataclass
class SplitResult:
    """Holds the three temporal splits and their metadata."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame

    # Boundary timestamps
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    # Row counts
    n_train: int
    n_val: int
    n_test: int
    n_total: int

    def summary(self) -> str:
        lines = [
            "Chronological Split Summary",
            "=" * 50,
            f"  TRAIN : {self.n_train:>10,} rows  [{self.train_start} -> {self.train_end}]",
            f"  VAL   : {self.n_val:>10,} rows  [{self.val_start} -> {self.val_end}]",
            f"  TEST  : {self.n_test:>10,} rows  [{self.test_start} -> {self.test_end}]",
            f"  TOTAL : {self.n_total:>10,} rows",
        ]
        return "\n".join(lines)


def chronological_split(
    df: pd.DataFrame,
    train_ratio: Optional[float] = None,
    val_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    align_to_day_boundary: bool = True,
) -> SplitResult:
    """Split DataFrame chronologically into train, validation, and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, timestamp-sorted DataFrame.
    train_ratio : float, optional
        Fraction for training. Defaults to config value.
    val_ratio : float, optional
        Fraction for validation. Defaults to config value.
    test_ratio : float, optional
        Fraction for test. Defaults to config value.
    align_to_day_boundary : bool
        If True, snap the split boundary to the nearest day boundary
        to avoid mid-day splits.

    Returns
    -------
    SplitResult
    """
    if train_ratio is None:
        train_ratio = CFG.split.train_ratio
    if val_ratio is None:
        val_ratio = CFG.split.validation_ratio
    if test_ratio is None:
        test_ratio = CFG.split.test_ratio

    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.4f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    if _TS_COL not in df.columns:
        raise ValueError(f"Timestamp column '{_TS_COL}' not found in DataFrame.")

    # Ensure sorted
    if not df[_TS_COL].is_monotonic_increasing:
        logger.warning("[Splitter] DataFrame is not sorted by timestamp. Sorting now.")
        df = df.sort_values(_TS_COL, ascending=True).reset_index(drop=True)

    n = len(df)
    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))

    if align_to_day_boundary and pd.api.types.is_datetime64_any_dtype(df[_TS_COL]):
        # Find the date of the row at the natural cutoff
        train_boundary_date = df[_TS_COL].iloc[train_end_idx].date()
        val_boundary_date = df[_TS_COL].iloc[val_end_idx].date()

        # Find the last row on that date (keep full day in train)
        train_end_idx = int(
            df[df[_TS_COL].dt.date <= train_boundary_date].index[-1] + 1
        )
        val_end_idx = int(
            df[df[_TS_COL].dt.date <= val_boundary_date].index[-1] + 1
        )

        logger.info(
            "[Splitter] Day-aligned split boundaries: train ends at date %s, val ends at %s",
            train_boundary_date,
            val_boundary_date,
        )

    train_df = df.iloc[:train_end_idx].copy()
    val_df = df.iloc[train_end_idx:val_end_idx].copy()
    test_df = df.iloc[val_end_idx:].copy()

    def _ts_range(split_df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
        ts = split_df[_TS_COL].dropna()
        if len(ts) == 0:
            return pd.NaT, pd.NaT
        return ts.min(), ts.max()

    train_start, train_end = _ts_range(train_df)
    val_start, val_end = _ts_range(val_df)
    test_start, test_end = _ts_range(test_df)

    result = SplitResult(
        train=train_df,
        validation=val_df,
        test=test_df,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        test_start=test_start,
        test_end=test_end,
        n_train=len(train_df),
        n_val=len(val_df),
        n_test=len(test_df),
        n_total=n,
    )

    logger.info("[Splitter] %s", result.summary().replace("\n", " | "))

    # Sanity check: no temporal overlap
    if val_start <= train_end:
        logger.warning(
            "[Splitter] WARNING: Validation start (%s) overlaps with train end (%s)!",
            val_start,
            train_end,
        )
    if test_start <= val_end:
        logger.warning(
            "[Splitter] WARNING: Test start (%s) overlaps with validation end (%s)!",
            test_start,
            val_end,
        )

    return result
