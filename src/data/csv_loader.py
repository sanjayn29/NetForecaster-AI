"""
src/data/csv_loader.py
----------------------
CICFlowMeter CSV loader for NetForecaster AI.

Design goals
------------
- Discover all CSV files in data/raw/
- Safely load large files using chunked reading
- Preserve source filename / capture day metadata
- Validate schema compatibility across files
- Handle the Tuesday file's extended 84-column schema
- Return a combined, schema-aligned DataFrame (or iterator of chunks)

Key findings from Phase 0 that drive this implementation
---------------------------------------------------------
- 10 CSV files, total ~16.2M rows
- 9 files: 80-column common schema
- 1 file (Thuesday-20-02-2018): 84 columns — 4 extra IP/ID columns
- Header-leak rows: some files have rows where Label == "Label"
- Dtypes may be fully `object` in files with header-leak contamination
- Chunk size of 200,000 rows is used by default to stay memory-safe
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd
import numpy as np

from src.utils.logger import get_logger
from src.utils.config import CFG

logger = get_logger(__name__)

# ── Constants derived from Phase 0 findings ───────────────────────────────
# Columns present only in the Tuesday file — must be tracked but excluded
# from the common feature schema.
_TUESDAY_EXTRA_COLS: set[str] = {"Flow ID", "Src IP", "Src Port", "Dst IP"}

# Known header-leak artifact: rows where Label column contains "Label"
_HEADER_LEAK_VALUE = "Label"

# Timestamp column name
_TS_COL = "Timestamp"

# Label column name
_LABEL_COL = "Label"

# Minimum expected number of columns in a valid CICFlowMeter file
_MIN_COLS = 79


def _extract_day_label(filename: str) -> str:
    """Extract a human-readable day identifier from the CSV filename.

    Example
    -------
    ``"Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"``
    → ``"Friday-02-03-2018"``
    """
    stem = Path(filename).stem
    return stem.split("_")[0]


def discover_csv_files(raw_dir: str | Path | None = None) -> List[Path]:
    """Return a sorted list of CSV file paths found in ``raw_dir``.

    Parameters
    ----------
    raw_dir : str or Path, optional
        Directory to search.  Defaults to ``CFG.data.raw_dir``.

    Returns
    -------
    List[Path]
        Sorted list of CSV file paths.

    Raises
    ------
    FileNotFoundError
        If ``raw_dir`` does not exist.
    ValueError
        If no CSV files are found.
    """
    if raw_dir is None:
        raw_dir = Path(CFG.data.raw_dir)
    raw_dir = Path(raw_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in: {raw_dir}")

    logger.info("Discovered %d CSV file(s) in %s", len(csv_files), raw_dir)
    for p in csv_files:
        size_mb = p.stat().st_size / (1024 ** 2)
        logger.info("  %-60s  %7.1f MB", p.name, size_mb)

    return csv_files


def get_common_columns(csv_files: List[Path]) -> List[str]:
    """Determine the 80-column common schema by reading only headers.

    Reads only the first row of each file to extract column names.
    The Tuesday file's extra columns are excluded from the common schema.

    Parameters
    ----------
    csv_files : List[Path]
        List of CSV file paths.

    Returns
    -------
    List[str]
        Ordered list of common column names (stripped of whitespace).
    """
    col_sets: list[list[str]] = []
    for fpath in csv_files:
        try:
            header_df = pd.read_csv(
                fpath,
                nrows=0,
                encoding="utf-8",
                encoding_errors="replace",
            )
            cols = [c.strip() for c in header_df.columns]
            col_sets.append(cols)
        except Exception as exc:
            logger.warning("Could not read header from %s: %s", fpath.name, exc)

    if not col_sets:
        raise RuntimeError("Failed to read headers from any CSV file.")

    # Identify the reference (most common) column set — 80-column schema
    # Strategy: use the set that appears most frequently
    from collections import Counter

    frozen_sets = [tuple(c) for c in col_sets]
    most_common_schema = Counter(frozen_sets).most_common(1)[0][0]
    common_cols = list(most_common_schema)

    # Validate that the extra Tuesday columns are NOT in the common schema
    for col in _TUESDAY_EXTRA_COLS:
        if col in common_cols:
            common_cols.remove(col)
            logger.debug("Removed Tuesday-only column '%s' from common schema.", col)

    logger.info(
        "Common schema established: %d columns (excluding Tuesday-only identifiers).",
        len(common_cols),
    )
    return common_cols


def load_file_chunked(
    fpath: Path,
    common_cols: List[str],
    chunk_size: int = 200_000,
) -> Iterator[pd.DataFrame]:
    """Yield cleaned DataFrame chunks from a single CSV file.

    Processing per chunk
    --------------------
    1. Strip column name whitespace.
    2. Align to common schema (drop Tuesday-only extra columns).
    3. Remove header-leak rows (Label == "Label").
    4. Add provenance metadata: ``source_file`` and ``source_day`` columns.

    Parameters
    ----------
    fpath : Path
        Path to the CSV file.
    common_cols : List[str]
        The 80-column common schema column list.
    chunk_size : int
        Number of rows per chunk.

    Yields
    ------
    pd.DataFrame
        A chunk of the file with aligned schema and provenance columns.
    """
    source_day = _extract_day_label(fpath.name)
    source_file = fpath.name

    reader = pd.read_csv(
        fpath,
        low_memory=False,
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines="warn",
        chunksize=chunk_size,
    )

    chunk_idx = 0
    for chunk in reader:
        # 1. Strip whitespace from column names
        chunk.columns = [c.strip() for c in chunk.columns]

        # 2. Align schema — keep only common columns that are present
        present_common = [c for c in common_cols if c in chunk.columns]
        missing = [c for c in common_cols if c not in chunk.columns]
        if missing:
            logger.warning(
                "%s chunk %d: missing common columns %s — filling with NaN.",
                source_file,
                chunk_idx,
                missing,
            )
            for col in missing:
                chunk[col] = np.nan
        chunk = chunk[common_cols]  # enforce exact column order

        # 3. Remove header-leak rows
        if _LABEL_COL in chunk.columns:
            n_before = len(chunk)
            chunk = chunk[chunk[_LABEL_COL] != _HEADER_LEAK_VALUE]
            removed = n_before - len(chunk)
            if removed > 0:
                logger.debug(
                    "%s chunk %d: removed %d header-leak row(s).",
                    source_file,
                    chunk_idx,
                    removed,
                )

        # 4. Add provenance columns
        chunk["source_file"] = source_file
        chunk["source_day"] = source_day

        if len(chunk) > 0:
            yield chunk

        chunk_idx += 1


def load_all_files(
    raw_dir: str | Path | None = None,
    chunk_size: int | None = None,
    return_chunks: bool = False,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Load and combine all CSV files from raw_dir.

    Parameters
    ----------
    raw_dir : str or Path, optional
        Raw data directory.  Defaults to config value.
    chunk_size : int, optional
        Rows per chunk when reading.  Defaults to config value.
    return_chunks : bool
        If True, yield chunks as an iterator instead of concatenating.
        Use for very large workloads where RAM is limited.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all files (when ``return_chunks=False``).
    Iterator[pd.DataFrame]
        Chunk iterator (when ``return_chunks=True``).
    """
    if chunk_size is None:
        chunk_size = CFG.preprocessing.chunk_size

    csv_files = discover_csv_files(raw_dir)
    common_cols = get_common_columns(csv_files)

    def _all_chunks() -> Iterator[pd.DataFrame]:
        for fpath in csv_files:
            logger.info("Loading: %s", fpath.name)
            yield from load_file_chunked(fpath, common_cols, chunk_size)

    if return_chunks:
        return _all_chunks()

    # Concatenate into a single DataFrame
    parts: list[pd.DataFrame] = []
    total_rows = 0
    for fpath in csv_files:
        file_chunks: list[pd.DataFrame] = []
        logger.info("Loading: %s", fpath.name)
        for chunk in load_file_chunked(fpath, common_cols, chunk_size):
            file_chunks.append(chunk)
            total_rows += len(chunk)
        if file_chunks:
            parts.append(pd.concat(file_chunks, ignore_index=True))
            logger.info(
                "  -> %d rows loaded from %s",
                sum(len(c) for c in file_chunks),
                fpath.name,
            )

    combined = pd.concat(parts, ignore_index=True)
    logger.info(
        "Combined dataset: %d rows × %d columns (before type coercion).",
        *combined.shape,
    )
    return combined


def validate_schema(
    df: pd.DataFrame,
    common_cols: List[str],
    source_name: str = "DataFrame",
) -> dict:
    """Check that a DataFrame conforms to the expected common schema.

    Parameters
    ----------
    df : pd.DataFrame
    common_cols : List[str]
        Expected column names.
    source_name : str
        Label for log messages.

    Returns
    -------
    dict
        Schema validation report.
    """
    df_cols = set(df.columns) - {"source_file", "source_day"}
    expected = set(common_cols)
    missing = expected - df_cols
    unexpected = df_cols - expected

    report = {
        "source": source_name,
        "expected_cols": len(expected),
        "found_cols": len(df_cols),
        "missing_cols": sorted(missing),
        "unexpected_cols": sorted(unexpected),
        "schema_valid": len(missing) == 0,
    }

    if missing:
        logger.warning("%s: Missing columns: %s", source_name, sorted(missing))
    if unexpected:
        logger.warning("%s: Unexpected columns: %s", source_name, sorted(unexpected))
    if report["schema_valid"]:
        logger.info("%s: Schema valid ✓", source_name)

    return report
