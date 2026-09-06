"""
scripts/prepare_data.py
------------------------
NetForecaster AI — Phase 1 Data Preprocessing Pipeline

Executes the complete preprocessing pipeline:
  Raw CSV files
       ↓
  Schema validation
       ↓
  Cleaning (header-leak, epoch anomalies, inf/NaN, duplicates, constants)
       ↓
  Timestamp parsing & chronological sort
       ↓
  Label encoding (multi-class + binary)
       ↓
  Chronological train/validation/test split
       ↓
  Feature scaling (fitted on train only)
       ↓
  Save processed Parquet files
       ↓
  Temporal window index generation
       ↓
  Save preprocessing artifacts
       ↓
  Generate preprocessing report

Usage
-----
  python scripts/prepare_data.py

All outputs are controlled by config.yaml.
Raw files are NEVER modified.

Large-dataset strategy
----------------------
This script processes the 16M-row dataset in chunks to avoid RAM exhaustion.
Each file is read and cleaned independently, then results are concatenated
and saved as compressed Parquet.

On an 8 GB RAM machine:
  - Peak RAM usage: ~4–6 GB during concatenation step
  - Processed Parquet files: ~1–2 GB (columnar + snappy compression)
  - Processing time: ~5–15 minutes depending on disk speed
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ── Ensure project root is on Python path ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import CFG
from src.utils.logger import get_logger, setup_root_logger
from src.utils.seed import set_seed
from src.data.csv_loader import discover_csv_files, get_common_columns, load_file_chunked
from src.data.validation import validate_dataframe
from src.data.cleaning import clean_dataframe
from src.data.label_encoder import CICLabelEncoder
from src.preprocessing.splitter import chronological_split
from src.preprocessing.scaler import FeatureScaler
from src.preprocessing.windowing import build_window_indices, save_window_indices
from src.preprocessing.sequence_builder import build_sequence_datasets

# ── Setup ──────────────────────────────────────────────────────────────────
LOG_FILE = PROJECT_ROOT / "logs" / "preprocessing.log"
(PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
setup_root_logger(level="INFO", log_file=LOG_FILE)
logger = get_logger("prepare_data")

set_seed(CFG.project.seed)

# ── Paths ──────────────────────────────────────────────────────────────────
RAW_DIR = PROJECT_ROOT / CFG.data.raw_dir
PROCESSED_DIR = PROJECT_ROOT / CFG.data.processed_dir
MODELS_PREP_DIR = PROJECT_ROOT / CFG.models.preprocessing_dir
REPORTS_DIR = PROJECT_ROOT / CFG.reports.dir

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_PREP_DIR.mkdir(parents=True, exist_ok=True)
(PROJECT_ROOT / "reports" / "metrics").mkdir(parents=True, exist_ok=True)

_TS_COL = CFG.columns.timestamp
_LABEL_COL = CFG.columns.label


def main() -> None:
    t_start = time.time()
    pipeline_report: dict = {}

    logger.info("=" * 70)
    logger.info("NetForecaster AI — Phase 1: Data Preprocessing Pipeline")
    logger.info("=" * 70)

    # ══════════════════════════════════════════════════════════════════
    # STEP 1: Discover files and establish common schema
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 1] Discovering CSV files and common schema...")
    csv_files = discover_csv_files(RAW_DIR)
    common_cols = get_common_columns(csv_files)
    pipeline_report["n_input_files"] = len(csv_files)
    pipeline_report["input_files"] = [f.name for f in csv_files]
    pipeline_report["original_col_count"] = len(common_cols)

    # ══════════════════════════════════════════════════════════════════
    # STEP 2: Load + clean each file in chunks, then concatenate
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 2] Loading and cleaning all files...")
    all_cleaning_reports: list[dict] = []
    file_row_counts: dict[str, int] = {}
    all_parts: list[pd.DataFrame] = []

    chunk_size = CFG.preprocessing.chunk_size

    for fpath in csv_files:
        file_t = time.time()
        logger.info("  Processing: %s", fpath.name)

        # Collect chunks for this file
        file_chunks: list[pd.DataFrame] = []
        for chunk in load_file_chunked(fpath, common_cols, chunk_size):
            file_chunks.append(chunk)

        if not file_chunks:
            logger.warning("  No chunks loaded from %s — skipping.", fpath.name)
            continue

        # Concatenate file chunks
        file_df = pd.concat(file_chunks, ignore_index=True)
        rows_before_clean = len(file_df)

        # Validate raw file
        _ = validate_dataframe(file_df, common_cols, stage=f"raw/{fpath.stem[:30]}")

        # Clean the file
        file_df, clean_report = clean_dataframe(file_df)
        clean_report["file"] = fpath.name
        clean_report["rows_before"] = rows_before_clean
        all_cleaning_reports.append(clean_report)
        file_row_counts[fpath.name] = len(file_df)

        all_parts.append(file_df)
        logger.info(
            "  -> %s: %d rows retained  (%.1f s)",
            fpath.name,
            len(file_df),
            time.time() - file_t,
        )

    # Concatenate all files
    logger.info("\n[Step 2] Concatenating all files...")
    df = pd.concat(all_parts, ignore_index=True)
    logger.info("  Combined: %d rows × %d columns", *df.shape)
    pipeline_report["rows_per_file"] = file_row_counts
    pipeline_report["total_rows_before_clean"] = sum(
        r["rows_before"] for r in all_cleaning_reports
    )
    pipeline_report["total_rows_after_clean"] = len(df)

    # ══════════════════════════════════════════════════════════════════
    # STEP 3: Sort chronologically (across all files combined)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 3] Sorting combined dataset chronologically...")
    df = df.sort_values(_TS_COL, ascending=True, na_position="last")
    df = df.reset_index(drop=True)
    ts_valid = df[_TS_COL].dropna()
    pipeline_report["timestamp_range"] = {
        "min": str(ts_valid.min()),
        "max": str(ts_valid.max()),
        "span_days": float((ts_valid.max() - ts_valid.min()).total_seconds() / 86400),
    }
    logger.info(
        "  Time range: %s -> %s",
        pipeline_report["timestamp_range"]["min"],
        pipeline_report["timestamp_range"]["max"],
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 4: Validate combined cleaned dataset
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 4] Validating combined cleaned dataset...")
    val_report = validate_dataframe(df, stage="cleaned_combined")
    pipeline_report["post_clean_validation"] = {
        "valid": val_report["valid"],
        "n_warnings": len(val_report["warnings"]),
        "n_errors": len(val_report["errors"]),
        "missing_total": val_report["missing_values"]["total"],
        "inf_total": val_report["infinite_values"]["total"],
        "duplicates": val_report["duplicate_rows"],
    }

    # ══════════════════════════════════════════════════════════════════
    # STEP 5: Label encoding
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 5] Encoding labels...")
    label_encoder = CICLabelEncoder()
    label_encoder.fit(df[_LABEL_COL])

    df["label_encoded"] = label_encoder.transform(df[_LABEL_COL])
    df["label_binary"] = label_encoder.to_binary(df[_LABEL_COL])

    label_dist = df[_LABEL_COL].value_counts(dropna=False).to_dict()
    pipeline_report["label_distribution"] = {str(k): int(v) for k, v in label_dist.items()}
    pipeline_report["n_unique_labels"] = label_encoder.n_classes
    pipeline_report["unique_labels"] = label_encoder.classes_

    # Save label encoder
    label_encoder.save(MODELS_PREP_DIR / "label_encoder.json")
    logger.info("  %s", label_encoder.summary())

    # ══════════════════════════════════════════════════════════════════
    # STEP 6: Identify feature columns
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 6] Identifying feature columns...")
    exclude_from_features = {
        _TS_COL, _LABEL_COL, "label_encoded", "label_binary",
        "source_file", "source_day",
    }
    feature_cols = [c for c in df.columns if c not in exclude_from_features]
    pipeline_report["original_feature_count"] = pipeline_report["original_col_count"]
    pipeline_report["final_feature_count"] = len(feature_cols)
    pipeline_report["feature_cols"] = feature_cols
    logger.info("  Feature columns: %d", len(feature_cols))
    logger.info("  First 10 features: %s", str(feature_cols[:10]))

    # Save feature list
    feature_meta = {
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "label_col": _LABEL_COL,
        "timestamp_col": _TS_COL,
        "window_size": CFG.windowing.window_size,
        "forecast_horizon": CFG.windowing.forecast_horizon,
    }
    with open(MODELS_PREP_DIR / "feature_metadata.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    # ══════════════════════════════════════════════════════════════════
    # STEP 7: Chronological train/validation/test split
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 7] Chronological train/validation/test split...")
    split_result = chronological_split(df, align_to_day_boundary=True)
    logger.info(split_result.summary())

    pipeline_report["split"] = {
        "n_train": split_result.n_train,
        "n_val": split_result.n_val,
        "n_test": split_result.n_test,
        "train_period": {
            "start": str(split_result.train_start),
            "end": str(split_result.train_end),
        },
        "val_period": {
            "start": str(split_result.val_start),
            "end": str(split_result.val_end),
        },
        "test_period": {
            "start": str(split_result.test_start),
            "end": str(split_result.test_end),
        },
    }

    # ══════════════════════════════════════════════════════════════════
    # STEP 8: Feature scaling (fit on train only)
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 8] Fitting scaler on training data...")
    scaler = FeatureScaler(method=CFG.features.scaler)
    scaler.fit(split_result.train, feature_cols=feature_cols)

    logger.info("  Transforming train split...")
    train_scaled = scaler.transform(split_result.train)
    logger.info("  Transforming validation split...")
    val_scaled = scaler.transform(split_result.validation)
    logger.info("  Transforming test split...")
    test_scaled = scaler.transform(split_result.test)

    scaler.save(MODELS_PREP_DIR / "feature_scaler.pkl")

    # ══════════════════════════════════════════════════════════════════
    # STEP 9: Save processed Parquet files
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 9] Saving processed Parquet files...")
    # Save with all columns (features + labels + metadata)
    cols_to_save = feature_cols + [
        _TS_COL, _LABEL_COL, "label_encoded", "label_binary",
        "source_file", "source_day",
    ]
    cols_to_save = [c for c in cols_to_save if c in train_scaled.columns]

    def _save_parquet_chunked(df: pd.DataFrame, path: Path, row_chunk: int = 500_000) -> None:
        """Write a large DataFrame to Parquet in row-group chunks.

        Avoids the PyArrow ArrowMemoryError that occurs when trying to
        allocate a single 1 GB buffer for the full 12M-row train set.
        Writes each chunk as a separate row group in the same file.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        n = len(df)
        logger.info("  Writing %d rows to %s in chunks of %d...", n, path.name, row_chunk)
        writer = None
        for start in range(0, n, row_chunk):
            chunk = df.iloc[start : start + row_chunk]
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(path), table.schema, compression="snappy")
            writer.write_table(table)
            logger.info("    ... wrote rows %d-%d", start, min(start + row_chunk, n) - 1)
        if writer is not None:
            writer.close()
        logger.info("  Saved: %s", path)

    _save_parquet_chunked(train_scaled[cols_to_save], PROCESSED_DIR / "train.parquet")
    _save_parquet_chunked(val_scaled[cols_to_save], PROCESSED_DIR / "validation.parquet")
    _save_parquet_chunked(test_scaled[cols_to_save], PROCESSED_DIR / "test.parquet")
    logger.info(
        "  Saved: train.parquet (%d rows), validation.parquet (%d rows), test.parquet (%d rows)",
        len(train_scaled), len(val_scaled), len(test_scaled),
    )
    # NOTE: full_cleaned.parquet intentionally omitted — it would require
    # concatenating all three splits back into RAM (~15M rows). The three
    # individual split files are sufficient for all downstream phases.

    # ══════════════════════════════════════════════════════════════════
    # STEP 10: Temporal window index generation
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 10] Generating temporal window indices...")
    WINDOW_SIZE = CFG.windowing.window_size
    FORECAST_HORIZON = CFG.windowing.forecast_horizon
    STRIDE = CFG.windowing.stride

    windows_dir = PROCESSED_DIR / "windows"
    windows_dir.mkdir(parents=True, exist_ok=True)

    n_windows_per_split: dict[str, int] = {}
    for name, sdf in [("train", train_scaled), ("validation", val_scaled), ("test", test_scaled)]:
        day_labels = sdf["source_day"].values if "source_day" in sdf.columns else None
        indices = build_window_indices(
            n_rows=len(sdf),
            window_size=WINDOW_SIZE,
            forecast_horizon=FORECAST_HORIZON,
            stride=STRIDE,
            day_labels=day_labels,
            cross_day=True,  # Allow cross-day for maximum coverage
        )
        save_window_indices(indices, str(windows_dir / f"{name}_window_indices.npz"))
        n_windows_per_split[name] = len(indices)

    pipeline_report["windowing"] = {
        "window_size": WINDOW_SIZE,
        "forecast_horizon": FORECAST_HORIZON,
        "stride": STRIDE,
        "n_train_windows": n_windows_per_split.get("train", 0),
        "n_val_windows": n_windows_per_split.get("validation", 0),
        "n_test_windows": n_windows_per_split.get("test", 0),
        "input_shape": [WINDOW_SIZE, len(feature_cols)],
        "target_shape": [FORECAST_HORIZON],
    }
    logger.info(
        "  Windows — train: %d | val: %d | test: %d",
        n_windows_per_split.get("train", 0),
        n_windows_per_split.get("validation", 0),
        n_windows_per_split.get("test", 0),
    )

    # ══════════════════════════════════════════════════════════════════
    # STEP 11: Save full pipeline report as JSON
    # ══════════════════════════════════════════════════════════════════
    pipeline_report["total_time_seconds"] = round(time.time() - t_start, 2)
    pipeline_report["cleaning_steps"] = all_cleaning_reports

    report_json_path = REPORTS_DIR / "metrics" / "preprocessing_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_report, f, indent=2, default=str)
    logger.info("\n[Step 11] Preprocessing report saved to %s", report_json_path)

    # ══════════════════════════════════════════════════════════════════
    # STEP 12: Generate human-readable Markdown report
    # ══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 12] Writing markdown preprocessing report...")
    _write_markdown_report(pipeline_report, REPORTS_DIR / "preprocessing_report.md")

    logger.info("\n" + "=" * 70)
    logger.info("Phase 1 preprocessing complete in %.1f seconds.", time.time() - t_start)
    logger.info("=" * 70)

    # ── Final summary printout ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE — PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"  Input files          : {pipeline_report['n_input_files']}")
    print(f"  Rows before cleaning : {pipeline_report['total_rows_before_clean']:,}")
    print(f"  Rows after cleaning  : {pipeline_report['total_rows_after_clean']:,}")
    print(f"  Feature columns      : {pipeline_report['final_feature_count']}")
    print(f"  Unique labels        : {pipeline_report['n_unique_labels']}")
    print(f"  Train rows           : {pipeline_report['split']['n_train']:,}")
    print(f"  Val rows             : {pipeline_report['split']['n_val']:,}")
    print(f"  Test rows            : {pipeline_report['split']['n_test']:,}")
    print(f"  Train period         : {pipeline_report['split']['train_period']['start']} -> {pipeline_report['split']['train_period']['end']}")
    print(f"  Val period           : {pipeline_report['split']['val_period']['start']} -> {pipeline_report['split']['val_period']['end']}")
    print(f"  Test period          : {pipeline_report['split']['test_period']['start']} -> {pipeline_report['split']['test_period']['end']}")
    print(f"  Window size          : {WINDOW_SIZE}")
    print(f"  Forecast horizon     : {FORECAST_HORIZON}")
    print(f"  Train windows        : {n_windows_per_split.get('train', 0):,}")
    print(f"  Val windows          : {n_windows_per_split.get('validation', 0):,}")
    print(f"  Test windows         : {n_windows_per_split.get('test', 0):,}")
    print(f"  Input tensor shape   : ({WINDOW_SIZE}, {len(feature_cols)})")
    print(f"  Target tensor shape  : ({FORECAST_HORIZON},)")
    print(f"  Total time           : {pipeline_report['total_time_seconds']:.1f} s")
    print("=" * 70)
    print(f"\nOutputs saved to:")
    print(f"  {PROCESSED_DIR}")
    print(f"  {MODELS_PREP_DIR}")
    print(f"  {REPORTS_DIR / 'preprocessing_report.md'}")


def _write_markdown_report(report: dict, output_path: Path) -> None:
    """Write the preprocessing report as Markdown."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clean_reports = report.get("cleaning_steps", [])
    total_header_leak = sum(r.get("header_leak_removed", 0) for r in clean_reports)
    total_epoch = sum(r.get("epoch_anomalies_removed", 0) for r in clean_reports)
    total_inf = sum(r.get("inf_replaced", 0) for r in clean_reports)
    total_nan = sum(r.get("nan_filled", 0) for r in clean_reports)
    total_dups = sum(r.get("duplicates_removed", 0) for r in clean_reports)
    const_cols_dropped = clean_reports[0].get("constant_cols_dropped", []) if clean_reports else []

    label_dist = report.get("label_distribution", {})

    lines = [
        "# Preprocessing Report — Phase 1",
        "",
        f"**Project:** NetForecaster AI  ",
        f"**Dataset:** CIC-IDS2018  ",
        f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
        "",
        "---",
        "",
        "## 1. Input",
        "",
        f"| Item | Value |",
        f"|------|-------|",
        f"| Input files | {report['n_input_files']} |",
        f"| Total rows (raw) | {report['total_rows_before_clean']:,} |",
        f"| Original columns | {report['original_feature_count']} |",
        "",
        "### Rows per file",
        "",
        "| File | Rows (post-cleaning) |",
        "|------|--------------------:|",
    ]
    for fname, cnt in report.get("rows_per_file", {}).items():
        lines.append(f"| `{fname}` | {cnt:,} |")

    lines += [
        "",
        "---",
        "",
        "## 2. Cleaning Summary",
        "",
        "| Step | Count |",
        "|------|------:|",
        f"| Header-leak rows removed | {total_header_leak:,} |",
        f"| Epoch-anomaly rows removed | {total_epoch:,} |",
        f"| Inf values replaced → NaN | {total_inf:,} |",
        f"| NaN values filled | {total_nan:,} |",
        f"| Duplicate rows removed | {total_dups:,} |",
        f"| Constant-zero cols dropped | {len(const_cols_dropped)} |",
        "",
        f"**Constant-zero columns dropped:** `{'`, `'.join(const_cols_dropped)}`",
        "",
        f"**Rows before cleaning:** {report['total_rows_before_clean']:,}  ",
        f"**Rows after cleaning:** {report['total_rows_after_clean']:,}  ",
        f"**Rows removed total:** {report['total_rows_before_clean'] - report['total_rows_after_clean']:,}",
        "",
        "---",
        "",
        "## 3. Feature Columns",
        "",
        f"| Item | Value |",
        f"|------|-------|",
        f"| Original columns | {report['original_feature_count']} |",
        f"| Constant-zero removed | {len(const_cols_dropped)} |",
        f"| Tuesday-only ID cols removed | 4 |",
        f"| Non-feature cols (label, ts, source) | 5 |",
        f"| **Final feature count** | **{report['final_feature_count']}** |",
        "",
        "---",
        "",
        "## 4. Label Distribution (After Cleaning)",
        "",
        "| Label | Count | % |",
        "|-------|------:|--:|",
    ]
    total_rows = report["total_rows_after_clean"]
    for lbl, cnt in sorted(label_dist.items(), key=lambda x: -x[1]):
        pct = 100.0 * int(cnt) / total_rows if total_rows > 0 else 0
        lines.append(f"| {lbl} | {int(cnt):,} | {pct:.2f}% |")

    ts_range = report.get("timestamp_range", {})
    split = report.get("split", {})
    windowing = report.get("windowing", {})

    lines += [
        "",
        "---",
        "",
        "## 5. Timestamp Coverage",
        "",
        f"| Item | Value |",
        f"|------|-------|",
        f"| Min timestamp | {ts_range.get('min', 'N/A')} |",
        f"| Max timestamp | {ts_range.get('max', 'N/A')} |",
        f"| Span | {ts_range.get('span_days', 0):.2f} days |",
        "",
        "---",
        "",
        "## 6. Chronological Split",
        "",
        "| Split | Rows | Start | End |",
        "|-------|-----:|-------|-----|",
        f"| Train | {split.get('n_train', 0):,} | {split.get('train_period', {}).get('start', 'N/A')} | {split.get('train_period', {}).get('end', 'N/A')} |",
        f"| Validation | {split.get('n_val', 0):,} | {split.get('val_period', {}).get('start', 'N/A')} | {split.get('val_period', {}).get('end', 'N/A')} |",
        f"| Test | {split.get('n_test', 0):,} | {split.get('test_period', {}).get('start', 'N/A')} | {split.get('test_period', {}).get('end', 'N/A')} |",
        "",
        "---",
        "",
        "## 7. Temporal Windowing",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Window size (W) | {windowing.get('window_size', 'N/A')} |",
        f"| Forecast horizon (K) | {windowing.get('forecast_horizon', 'N/A')} |",
        f"| Stride | {windowing.get('stride', 'N/A')} |",
        f"| Train windows | {windowing.get('n_train_windows', 0):,} |",
        f"| Validation windows | {windowing.get('n_val_windows', 0):,} |",
        f"| Test windows | {windowing.get('n_test_windows', 0):,} |",
        f"| Input tensor shape | `({windowing.get('window_size')}, {report['final_feature_count']})` |",
        f"| Target tensor shape | `({windowing.get('forecast_horizon')},)` |",
        "",
        "---",
        "",
        "## 8. Preprocessing Artifacts",
        "",
        "| Artifact | Path |",
        "|----------|------|",
        "| Train Parquet | `data/processed/train.parquet` |",
        "| Validation Parquet | `data/processed/validation.parquet` |",
        "| Test Parquet | `data/processed/test.parquet` |",
        "| Label encoder | `models/preprocessing/label_encoder.json` |",
        "| Feature scaler | `models/preprocessing/feature_scaler.pkl` |",
        "| Feature metadata | `models/preprocessing/feature_metadata.json` |",
        "| Train window indices | `data/processed/windows/train_window_indices.npz` |",
        "| Val window indices | `data/processed/windows/validation_window_indices.npz` |",
        "| Test window indices | `data/processed/windows/test_window_indices.npz` |",
        "",
        f"---",
        "",
        f"*Total preprocessing time: {report.get('total_time_seconds', 0):.1f} seconds*",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("[Report] Markdown report written to %s", output_path)


if __name__ == "__main__":
    main()
