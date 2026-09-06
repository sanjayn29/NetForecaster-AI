"""
src/data/validation.py
-----------------------
Data validation for NetForecaster AI.

Validates a loaded DataFrame against expected properties derived from
the Phase 0 inspection of the CIC-IDS2018 dataset.

Checks performed
----------------
1.  Expected columns present
2.  No missing expected columns
3.  No unexpected columns (beyond provenance columns)
4.  Data types sanity (numeric columns should be numeric after cleaning)
5.  Missing value counts per column
6.  Infinite value counts per column
7.  Duplicate row count
8.  Constant columns (zero variance)
9.  Label column: presence, dtype, unique values
10. Timestamp column: presence, parseability, anomalies (1970 epoch)
11. Numerical value range sanity
12. Header-leak rows still present (Label == "Label")

Returns a structured ValidationReport dictionary and logs all findings.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.config import CFG

logger = get_logger(__name__)

# ── Expected values from Phase 0 ──────────────────────────────────────────
_KNOWN_LABELS = {
    "Benign",
    "Bot",
    "FTP-BruteForce",
    "SSH-Bruteforce",
    "DoS attacks-Hulk",
    "DoS attacks-SlowHTTPTest",
    "DoS attacks-GoldenEye",
    "DoS attacks-Slowloris",
    "DDOS attack-HOIC",
    "DDoS attacks-LOIC-HTTP",
    "DDOS attack-LOIC-UDP",
    "Brute Force -Web",
    "Brute Force -XSS",
    "SQL Injection",
    "Infilteration",
}

_CONSTANT_ZERO_COLS = set(CFG.columns.constant_zero)
_RATE_INF_COLS = set(CFG.columns.rate_cols_with_inf)
_TS_COL = CFG.columns.timestamp
_LABEL_COL = CFG.columns.label
_MIN_VALID_YEAR = CFG.preprocessing.min_valid_year


def validate_dataframe(
    df: pd.DataFrame,
    expected_cols: list[str] | None = None,
    stage: str = "raw",
) -> dict[str, Any]:
    """Run all validation checks on a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    expected_cols : list[str], optional
        Column names expected to be present.  If None, uses Phase-0 columns.
    stage : str
        Descriptive label for the validation stage (e.g., 'raw', 'cleaned').

    Returns
    -------
    dict
        ValidationReport with all check results.
    """
    report: dict[str, Any] = {
        "stage": stage,
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "checks": {},
        "warnings": [],
        "errors": [],
    }

    # ── Helper ─────────────────────────────────────────────────────────────
    def _warn(msg: str) -> None:
        report["warnings"].append(msg)
        logger.warning("[Validation/%s] %s", stage, msg)

    def _error(msg: str) -> None:
        report["errors"].append(msg)
        logger.error("[Validation/%s] %s", stage, msg)

    def _ok(check: str, detail: Any = None) -> None:
        report["checks"][check] = {"status": "OK", "detail": detail}
        logger.debug("[Validation/%s] %-40s OK  %s", stage, check, detail or "")

    def _fail(check: str, detail: Any = None) -> None:
        report["checks"][check] = {"status": "FAIL", "detail": detail}

    # ── 1. Basic shape ─────────────────────────────────────────────────────
    logger.info("[Validation/%s] Starting — shape: %d × %d", stage, *df.shape)
    if len(df) == 0:
        _error("DataFrame is empty.")
        return report

    # ── 2. Schema check ────────────────────────────────────────────────────
    if expected_cols is not None:
        prov_cols = {"source_file", "source_day"}
        df_feature_cols = set(df.columns) - prov_cols
        expected_set = set(expected_cols) - prov_cols
        missing_cols = expected_set - df_feature_cols
        extra_cols = df_feature_cols - expected_set

        if missing_cols:
            _fail("schema_missing_cols", sorted(missing_cols))
            _error(f"Missing columns: {sorted(missing_cols)}")
        else:
            _ok("schema_missing_cols", "none")

        if extra_cols:
            _warn(f"Unexpected extra columns present: {sorted(extra_cols)}")
            report["checks"]["schema_extra_cols"] = {
                "status": "WARN",
                "detail": sorted(extra_cols),
            }
        else:
            _ok("schema_extra_cols", "none")

    # ── 3. Label column ────────────────────────────────────────────────────
    if _LABEL_COL not in df.columns:
        _fail("label_present")
        _error(f"Label column '{_LABEL_COL}' not found.")
    else:
        _ok("label_present")

        # Header-leak rows
        header_leak = int((df[_LABEL_COL] == "Label").sum())
        if header_leak > 0:
            _fail("label_header_leak", header_leak)
            _warn(f"{header_leak} row(s) have Label == 'Label' (header-leak artifact).")
        else:
            _ok("label_header_leak", 0)

        # Unknown labels
        unique_labels = set(df[_LABEL_COL].dropna().unique())
        unknown_labels = unique_labels - _KNOWN_LABELS - {"Label"}
        if unknown_labels:
            _warn(f"Unknown label values found: {unknown_labels}")
            report["checks"]["label_unknown"] = {
                "status": "WARN",
                "detail": sorted(unknown_labels),
            }
        else:
            _ok("label_unknown", "none")

        report["label_distribution"] = df[_LABEL_COL].value_counts(dropna=False).to_dict()
        report["unique_labels"] = sorted(str(l) for l in unique_labels)

    # ── 4. Timestamp column ────────────────────────────────────────────────
    if _TS_COL not in df.columns:
        _fail("timestamp_present")
        _error(f"Timestamp column '{_TS_COL}' not found.")
    else:
        _ok("timestamp_present")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ts_parsed = pd.to_datetime(df[_TS_COL], errors="coerce")

        n_nat = int(ts_parsed.isna().sum())
        n_epoch_anomaly = int((ts_parsed.dt.year < _MIN_VALID_YEAR).sum() if n_nat < len(df) else 0)

        if n_nat > 0:
            _warn(f"Timestamp: {n_nat} rows could not be parsed (NaT).")
            report["checks"]["timestamp_parse"] = {
                "status": "WARN",
                "detail": f"{n_nat} NaT values",
            }
        else:
            _ok("timestamp_parse", "all parseable")

        if n_epoch_anomaly > 0:
            _warn(f"Timestamp: {n_epoch_anomaly} rows have pre-{_MIN_VALID_YEAR} epoch anomaly.")
            report["checks"]["timestamp_epoch_anomaly"] = {
                "status": "WARN",
                "detail": f"{n_epoch_anomaly} rows before {_MIN_VALID_YEAR}",
            }
        else:
            _ok("timestamp_epoch_anomaly", 0)

        valid_ts = ts_parsed.dropna()
        if len(valid_ts) > 0:
            report["timestamp_range"] = {
                "min": str(valid_ts.min()),
                "max": str(valid_ts.max()),
                "span_days": float((valid_ts.max() - valid_ts.min()).total_seconds() / 86400),
            }

    # ── 5. Missing values ──────────────────────────────────────────────────
    missing_per_col = df.isnull().sum()
    missing_cols_detail = {
        col: int(cnt)
        for col, cnt in missing_per_col[missing_per_col > 0].items()
    }
    total_missing = int(missing_per_col.sum())
    report["missing_values"] = {
        "total": total_missing,
        "by_column": missing_cols_detail,
    }
    if total_missing > 0:
        _warn(f"Total missing values: {total_missing} in columns: {list(missing_cols_detail.keys())}")
        report["checks"]["missing_values"] = {
            "status": "WARN",
            "detail": missing_cols_detail,
        }
    else:
        _ok("missing_values", 0)

    # ── 6. Infinite values ─────────────────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_per_col: dict[str, int] = {}
    for col in numeric_cols:
        n_inf = int(np.isinf(df[col].values).sum())
        if n_inf > 0:
            inf_per_col[col] = n_inf
    total_inf = sum(inf_per_col.values())
    report["infinite_values"] = {"total": total_inf, "by_column": inf_per_col}
    if total_inf > 0:
        _warn(f"Infinite values: {total_inf} in columns: {list(inf_per_col.keys())}")
        report["checks"]["infinite_values"] = {
            "status": "WARN",
            "detail": inf_per_col,
        }
    else:
        _ok("infinite_values", 0)

    # ── 7. Duplicate rows ─────────────────────────────────────────────────
    dup_count = int(df.duplicated().sum())
    report["duplicate_rows"] = dup_count
    if dup_count > 0:
        _warn(f"Duplicate rows: {dup_count:,} ({100.0 * dup_count / len(df):.2f}%)")
        report["checks"]["duplicates"] = {
            "status": "WARN",
            "detail": dup_count,
        }
    else:
        _ok("duplicates", 0)

    # ── 8. Constant columns ────────────────────────────────────────────────
    const_found: list[str] = []
    for col in numeric_cols:
        if df[col].nunique(dropna=True) <= 1:
            const_found.append(col)
    report["constant_cols"] = const_found
    if const_found:
        _warn(f"Constant (zero-variance) columns: {const_found}")
        report["checks"]["constant_cols"] = {
            "status": "WARN",
            "detail": const_found,
        }
    else:
        _ok("constant_cols", "none")

    # ── 9. Dtype sanity ────────────────────────────────────────────────────
    object_numeric_cols = [
        col for col in df.columns
        if df[col].dtype == object and col not in {_LABEL_COL, _TS_COL, "source_file", "source_day"}
    ]
    if object_numeric_cols:
        _warn(
            f"Columns expected to be numeric but are 'object' dtype: {object_numeric_cols}. "
            "May be caused by header-leak rows not yet removed."
        )
        report["checks"]["dtype_contamination"] = {
            "status": "WARN",
            "detail": object_numeric_cols,
        }
    else:
        _ok("dtype_contamination", "none")

    # ── Summary ────────────────────────────────────────────────────────────
    n_errors = len(report["errors"])
    n_warnings = len(report["warnings"])
    report["valid"] = n_errors == 0
    logger.info(
        "[Validation/%s] Complete — %d error(s), %d warning(s). Valid: %s",
        stage,
        n_errors,
        n_warnings,
        report["valid"],
    )

    return report
