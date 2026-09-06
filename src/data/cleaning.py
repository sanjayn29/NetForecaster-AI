"""
src/data/cleaning.py
--------------------
Data cleaning for NetForecaster AI.

Each cleaning step is documented with the rationale from Phase 0 findings.
No raw data files are modified.

Cleaning steps
--------------
1.  Remove header-leak rows (Label == "Label")
    Reason: 59 rows in 3 files are duplicated header lines from concatenation.

2.  Coerce numeric column dtypes
    Reason: Files with header-leak rows have all columns as object dtype.
    After removing header-leak rows, safe to coerce.

3.  Remove 1970-epoch timestamp anomalies
    Reason: 14 rows have CICFlowMeter parsing errors resulting in 1970 dates.
    Strategy: remove rows (only 14 total out of 16M).

4.  Parse and validate timestamps
    Reason: Timestamp column is raw string; must be datetime for temporal ops.

5.  Handle infinite values in Flow Byts/s and Flow Pkts/s
    Reason: Zero-duration flows produce x/0 = Inf in CICFlowMeter.
    Strategy: replace Inf with NaN, then impute (see step 6).
    Rationale: capping at 99th percentile would be misleading for rate features.

6.  Handle NaN values in Flow Byts/s and Flow Pkts/s
    Reason: NaN here means flow_duration = 0 microseconds.
    Strategy: set to 0.0 (a flow of zero duration has zero byte-rate).
    Rationale: These are instantaneous/measurement-limit flows, not missing data.

7.  Remove duplicate rows
    Reason: Phase 0 found 0.03%–21.52% duplicates per file.
    Strategy: keep first occurrence within each source_day group.
    Rationale: for temporal forecasting, keeping the first occurrence preserves
    chronological order while removing exact repeats. For brute-force attacks
    with many identical flows, this is a conscious trade-off — we log the count.
    NOTE: duplicate detection excludes provenance columns (source_file, source_day).

8.  Sort chronologically
    Reason: Temporal forecasting requires strict chronological ordering.

9.  Drop constant-zero columns
    Reason: Phase 0 identified 10 columns that are identically zero.
    These carry no discriminative information and inflate feature dimensions.

10. Drop Tuesday-only identifier columns (if present)
    Reason: Flow ID, Src IP, Dst IP are not available in 9/10 files.
    These should not be used as model features.
    NOTE: These columns are preserved in a separate graph-ready output
    if they exist (for Phase 6 graph construction).
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.config import CFG

logger = get_logger(__name__)

# ── Column sets from config (Phase 0 findings) ────────────────────────────
_CONSTANT_ZERO_COLS: list[str] = list(CFG.columns.constant_zero)
_RATE_INF_COLS: list[str] = list(CFG.columns.rate_cols_with_inf)
_TUESDAY_ONLY_COLS: list[str] = list(CFG.columns.tuesday_only)
_TS_COL: str = CFG.columns.timestamp
_LABEL_COL: str = CFG.columns.label
_MIN_VALID_YEAR: int = CFG.preprocessing.min_valid_year
_RATE_NAN_FILL: float = CFG.preprocessing.rate_nan_fill


def _log_step(step: str, msg: str) -> None:
    logger.info("[Cleaning] %-35s %s", step, msg)


def remove_header_leak_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows where Label column equals 'Label' (header row contamination).

    Phase 0 finding: 59 rows across 3 files.

    Returns
    -------
    (cleaned_df, n_removed)
    """
    if _LABEL_COL not in df.columns:
        return df, 0
    mask = df[_LABEL_COL] == "Label"
    n_removed = int(mask.sum())
    df = df[~mask].copy()
    _log_step("header_leak_rows", f"Removed {n_removed} rows.")
    return df, n_removed


def coerce_numeric_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object-typed columns (except Label, Timestamp, provenance) to numeric.

    Phase 0 finding: 3 files have all columns as object due to header-leak rows.
    After removing header rows, numeric coercion is safe.

    Columns that fail to convert are left as-is and logged.
    """
    skip_cols = {_LABEL_COL, _TS_COL, "source_file", "source_day"}
    n_coerced = 0
    for col in df.columns:
        if col in skip_cols:
            continue
        if df[col].dtype == object:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() >= 0.5 * len(df):
                df[col] = coerced
                n_coerced += 1
            else:
                logger.warning(
                    "[Cleaning] Column '%s' could not be reliably coerced to numeric. Left as object.",
                    col,
                )
    _log_step("coerce_dtypes", f"Coerced {n_coerced} object column(s) to numeric.")
    return df


def remove_epoch_anomalies(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove rows with 1970-epoch timestamp anomalies.

    Phase 0 finding: 14 rows in 2 files have CICFlowMeter parse errors
    resulting in Unix-epoch (1970) fallback timestamps.

    Strategy: remove these rows entirely (14 out of 16M is negligible).

    Returns
    -------
    (cleaned_df, n_removed)
    """
    if _TS_COL not in df.columns:
        return df, 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts_parsed = pd.to_datetime(df[_TS_COL], dayfirst=True, errors="coerce")

    n_before = len(df)
    # Keep rows where year >= MIN_VALID_YEAR OR timestamp could not be parsed
    # (NaT will be caught in later validation)
    valid_mask = ts_parsed.isna() | (ts_parsed.dt.year >= _MIN_VALID_YEAR)
    df = df[valid_mask].copy()
    n_removed = n_before - len(df)
    _log_step("epoch_anomalies", f"Removed {n_removed} row(s) with pre-{_MIN_VALID_YEAR} timestamps.")
    return df, n_removed


def parse_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Timestamp column from string to datetime64.

    Stores parsed datetime back in the Timestamp column.
    Rows with unparseable timestamps become NaT and are logged.
    """
    if _TS_COL not in df.columns:
        logger.warning("[Cleaning] Timestamp column '%s' not found — skipping.", _TS_COL)
        return df

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df[_TS_COL] = pd.to_datetime(df[_TS_COL], dayfirst=True, errors="coerce")

    n_nat = int(df[_TS_COL].isna().sum())
    if n_nat > 0:
        logger.warning(
            "[Cleaning] %d row(s) have unparseable Timestamp (NaT) after coercion.",
            n_nat,
        )
    _log_step("parse_timestamp", f"Parsed. NaT count: {n_nat}.")
    return df


def handle_infinite_values(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Replace infinite values in rate columns with NaN.

    Phase 0 finding: Inf values in 'Flow Byts/s' and 'Flow Pkts/s' caused
    by zero-duration flows (duration = 0 → rate = bytes/0 = Inf).

    Strategy: replace Inf/-Inf with NaN. NaN will be filled in the
    next step (handle_nan_values).

    Returns
    -------
    (cleaned_df, total_inf_replaced)
    """
    total_replaced = 0
    for col in _RATE_INF_COLS:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        n_inf = int(np.isinf(df[col].values).sum())
        if n_inf > 0:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            total_replaced += n_inf
            logger.debug("[Cleaning] Replaced %d Inf in '%s'.", n_inf, col)
    _log_step("infinite_values", f"Replaced {total_replaced} Inf -> NaN in rate columns.")
    return df, total_replaced


def handle_nan_values(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill NaN values in rate columns with 0.

    Phase 0 finding: NaN in 'Flow Byts/s' and 'Flow Pkts/s' occurs when
    Flow Duration = 0 (instantaneous or measurement-limit flows).

    Rationale: A flow with zero duration has a measured byte/packet rate
    of 0 (no time elapsed for transfer). Setting to 0.0 is semantically
    correct and avoids information leakage from imputation with statistics.

    NOTE: We do NOT impute with column mean/median because that would
    mix Benign and Attack distributions, introducing subtle leakage.

    Returns
    -------
    (cleaned_df, total_filled)
    """
    total_filled = 0
    for col in _RATE_INF_COLS:
        if col not in df.columns:
            continue
        n_nan = int(df[col].isna().sum())
        if n_nan > 0:
            df[col] = df[col].fillna(_RATE_NAN_FILL)
            total_filled += n_nan
            logger.debug("[Cleaning] Filled %d NaN in '%s' with %s.", n_nan, col, _RATE_NAN_FILL)

    # Check for any remaining NaN in other numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    remaining_nan_cols: dict[str, int] = {}
    for col in numeric_cols:
        n = int(df[col].isna().sum())
        if n > 0:
            remaining_nan_cols[col] = n

    if remaining_nan_cols:
        logger.warning(
            "[Cleaning] Remaining NaN in non-rate numeric columns: %s. "
            "Filling with column median (conservative).",
            remaining_nan_cols,
        )
        for col in remaining_nan_cols:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            total_filled += remaining_nan_cols[col]

    _log_step("nan_values", f"Filled {total_filled} NaN value(s).")
    return df, total_filled


def remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows, keeping the first chronological occurrence.

    Phase 0 finding: 0.03%–21.52% duplicates per file.
    Notable: Wednesday-14 (21.52%) and Friday-16 (14.07%).

    Strategy: Within each source_day group, after sorting by timestamp,
    drop exact duplicates on all feature columns (excluding provenance).
    Keep first occurrence (earliest timestamp) to preserve temporal order.

    Rationale for keeping first:
    - We are building a temporal forecasting system.
    - The first occurrence of an attack flow is informative.
    - Subsequent exact duplicates within the same day add no new information.
    - For brute-force attacks, repeated identical-feature flows may be a
      legitimate CICFlowMeter artifact rather than truly distinct events.

    Returns
    -------
    (cleaned_df, n_removed)
    """
    prov_cols = {"source_file", "source_day"}

    if subset is None:
        # Use all columns except provenance for duplicate detection
        subset = [c for c in df.columns if c not in prov_cols]

    n_before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    n_removed = n_before - len(df)
    _log_step(
        "duplicate_rows",
        f"Removed {n_removed:,} duplicates ({100.0 * n_removed / max(n_before, 1):.2f}%).",
    )
    return df.reset_index(drop=True), n_removed


def drop_constant_zero_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns identified as constant-zero in Phase 0.

    Phase 0 finding: 10 columns are identically 0 across all files.
    These provide no discriminative information for classification or
    temporal forecasting and inflate feature dimensionality.

    Dropped columns are logged but NOT permanently deleted from the
    raw files — only from the processed output.

    Returns
    -------
    (cleaned_df, dropped_col_names)
    """
    to_drop = [col for col in _CONSTANT_ZERO_COLS if col in df.columns]
    df = df.drop(columns=to_drop)
    _log_step("constant_zero_cols", f"Dropped {len(to_drop)} column(s): {to_drop}")
    return df, to_drop


def drop_tuesday_only_columns(
    df: pd.DataFrame,
    save_graph_cols: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Drop Flow ID, Src IP, Dst IP, Src Port (Tuesday-only identifier columns).

    These columns are only present in the Tuesday file (1 of 10 files) and
    cannot be used as features in a unified model. They are high-leakage
    identifiers in the lab environment.

    If ``save_graph_cols=True``, return these columns separately so they
    can be used for graph construction (Phase 6).

    Returns
    -------
    (feature_df, graph_cols_df | None)
    """
    present = [col for col in _TUESDAY_ONLY_COLS if col in df.columns]

    graph_df: pd.DataFrame | None = None
    if present and save_graph_cols:
        graph_df = df[present + [_TS_COL, _LABEL_COL]].copy()
        logger.info(
            "[Cleaning] Preserved %d Tuesday-only column(s) for graph construction.",
            len(present),
        )

    df = df.drop(columns=present)
    _log_step(
        "tuesday_only_cols",
        f"Dropped {len(present)} identifier column(s): {present}",
    )
    return df, graph_df


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Sort DataFrame by Timestamp in ascending order.

    Critical for temporal forecasting: ensures all downstream windowing
    preserves strict chronological order.
    """
    if _TS_COL not in df.columns:
        logger.warning("[Cleaning] Cannot sort: Timestamp column missing.")
        return df
    df = df.sort_values(_TS_COL, ascending=True, na_position="last")
    df = df.reset_index(drop=True)
    _log_step("chronological_sort", "Sorted by Timestamp ascending.")
    return df


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the complete cleaning pipeline on a DataFrame.

    Applies all cleaning steps in the correct order and returns a
    cleaning report with row counts and decisions made.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (post-load) DataFrame.

    Returns
    -------
    (cleaned_df, cleaning_report)
    """
    report: dict[str, Any] = {"rows_before": len(df)}
    logger.info("=" * 60)
    logger.info("[Cleaning] Starting pipeline — rows: %d", len(df))
    logger.info("=" * 60)

    # Step 1: Remove header-leak rows
    df, report["header_leak_removed"] = remove_header_leak_rows(df)

    # Step 2: Coerce dtypes (safe after removing header rows)
    df = coerce_numeric_dtypes(df)

    # Step 3: Remove epoch anomalies
    df, report["epoch_anomalies_removed"] = remove_epoch_anomalies(df)

    # Step 4: Parse timestamps
    df = parse_timestamp(df)

    # Step 5: Handle infinite values
    df, report["inf_replaced"] = handle_infinite_values(df)

    # Step 6: Handle NaN values
    df, report["nan_filled"] = handle_nan_values(df)

    # Step 7: Sort chronologically (before dedup so first = earliest)
    df = sort_chronologically(df)

    # Step 8: Remove duplicates
    df, report["duplicates_removed"] = remove_duplicates(df)

    # Step 9: Drop constant-zero columns
    df, report["constant_cols_dropped"] = drop_constant_zero_columns(df)

    # Step 10: Drop Tuesday-only identifier columns
    df, graph_df = drop_tuesday_only_columns(df, save_graph_cols=True)
    report["tuesday_cols_dropped"] = list(CFG.columns.tuesday_only)
    report["graph_cols_preserved"] = graph_df is not None

    report["rows_after"] = len(df)
    report["cols_after"] = df.shape[1]
    report["rows_removed_total"] = report["rows_before"] - report["rows_after"]

    logger.info("=" * 60)
    logger.info(
        "[Cleaning] Pipeline complete: %d -> %d rows (removed %d). Cols: %d.",
        report["rows_before"],
        report["rows_after"],
        report["rows_removed_total"],
        report["cols_after"],
    )
    logger.info("=" * 60)

    return df, report
