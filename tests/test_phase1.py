"""
tests/test_phase1.py
---------------------
Unit tests for Phase 1: Data Ingestion, Validation, Cleaning, Preprocessing.

Tests use small synthetic samples that mirror the real CIC-IDS2018 schema.
No real CSV files are read in these tests.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Add project root to path ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.label_encoder import CICLabelEncoder, CANONICAL_LABELS, BINARY_MAP
from src.data.cleaning import (
    remove_header_leak_rows,
    coerce_numeric_dtypes,
    remove_epoch_anomalies,
    parse_timestamp,
    handle_infinite_values,
    handle_nan_values,
    remove_duplicates,
    drop_constant_zero_columns,
    sort_chronologically,
)
from src.data.validation import validate_dataframe
from src.preprocessing.splitter import chronological_split
from src.preprocessing.scaler import FeatureScaler
from src.preprocessing.windowing import build_window_indices, save_window_indices, load_window_indices, WindowIndex
from src.utils.config import CFG

# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_sample_df(n: int = 100, seed: int = 0) -> pd.DataFrame:
    """Create a small DataFrame with the CIC-IDS2018 schema structure."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2018-02-14 01:00:00", periods=n, freq="1s")
    labels = rng.choice(CANONICAL_LABELS, size=n)
    df = pd.DataFrame(
        {
            "Timestamp": ts,
            "Dst Port": rng.integers(0, 65535, size=n),
            "Protocol": rng.choice([6, 17, 0], size=n),
            "Flow Duration": rng.integers(0, 1_000_000, size=n),
            "Tot Fwd Pkts": rng.integers(1, 1000, size=n),
            "Tot Bwd Pkts": rng.integers(0, 500, size=n),
            "TotLen Fwd Pkts": rng.integers(0, 100_000, size=n),
            "TotLen Bwd Pkts": rng.integers(0, 50_000, size=n),
            "Flow Byts/s": rng.uniform(0, 1e6, size=n),
            "Flow Pkts/s": rng.uniform(0, 1000, size=n),
            "Fwd IAT Mean": rng.uniform(0, 1e6, size=n),
            "Label": labels,
            "source_file": "test_file.csv",
            "source_day": "Wednesday-14-02-2018",
        }
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Label Encoder Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestLabelEncoder:
    def test_canonical_labels_count(self):
        assert len(CANONICAL_LABELS) == 15, "Expected 15 canonical labels (Benign + 14 attack types)"

    def test_benign_is_class_zero(self):
        enc = CICLabelEncoder()
        assert enc.label_to_int["Benign"] == 0

    def test_transform_known_labels(self):
        enc = CICLabelEncoder()
        labels = ["Benign", "Bot", "FTP-BruteForce"]
        encoded = enc.transform(labels)
        assert encoded[0] == 0  # Benign
        assert all(encoded >= 0)

    def test_transform_unknown_label(self):
        enc = CICLabelEncoder()
        encoded = enc.transform(["UnknownAttack"])
        assert encoded[0] == -1

    def test_inverse_transform_roundtrip(self):
        enc = CICLabelEncoder()
        labels = ["Benign", "Bot", "SSH-Bruteforce"]
        encoded = enc.transform(labels)
        decoded = enc.inverse_transform(encoded)
        assert decoded == labels

    def test_binary_encoding_benign(self):
        enc = CICLabelEncoder()
        binary = enc.to_binary(["Benign"])
        assert binary[0] == 0

    def test_binary_encoding_attack(self):
        enc = CICLabelEncoder()
        for lbl in CANONICAL_LABELS:
            if lbl != "Benign":
                binary = enc.to_binary([lbl])
                assert binary[0] == 1, f"{lbl} should be binary=1"

    def test_save_load_roundtrip(self, tmp_path):
        enc = CICLabelEncoder()
        save_path = tmp_path / "encoder_test.json"
        enc.save(save_path)
        enc2 = CICLabelEncoder.load(save_path)
        assert enc2.n_classes == enc.n_classes
        assert enc2.classes_ == enc.classes_

    def test_fit_with_unseen_label(self):
        enc = CICLabelEncoder()
        enc.fit(["Benign", "Bot", "NewAttackType"])
        assert "NewAttackType" in enc.classes_
        assert enc.n_classes == 16  # 15 canonical + 1 new


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Cleaning Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCleaning:
    def test_remove_header_leak(self):
        df = _make_sample_df(50)
        # Inject 3 header-leak rows
        df.loc[0, "Label"] = "Label"
        df.loc[5, "Label"] = "Label"
        df.loc[10, "Label"] = "Label"
        cleaned, n_removed = remove_header_leak_rows(df)
        assert n_removed == 3
        assert len(cleaned) == 47
        assert "Label" not in cleaned["Label"].values

    def test_remove_epoch_anomaly(self):
        df = _make_sample_df(20)
        # Inject 2 epoch-anomaly rows
        df.loc[3, "Timestamp"] = pd.Timestamp("1970-01-05")
        df.loc[7, "Timestamp"] = pd.Timestamp("1970-01-12")
        cleaned, n_removed = remove_epoch_anomalies(df)
        assert n_removed == 2
        assert len(cleaned) == 18

    def test_handle_infinite_values(self):
        df = _make_sample_df(30)
        df.loc[0, "Flow Byts/s"] = np.inf
        df.loc[1, "Flow Pkts/s"] = -np.inf
        df.loc[2, "Flow Byts/s"] = np.inf
        cleaned, n_replaced = handle_infinite_values(df)
        assert n_replaced == 3
        assert not np.isinf(cleaned["Flow Byts/s"].values).any()
        assert not np.isinf(cleaned["Flow Pkts/s"].values).any()
        # Should be NaN after replacement
        assert cleaned.loc[0, "Flow Byts/s"] != np.inf

    def test_handle_nan_values(self):
        df = _make_sample_df(30)
        df.loc[0, "Flow Byts/s"] = np.nan
        df.loc[1, "Flow Pkts/s"] = np.nan
        cleaned, n_filled = handle_nan_values(df)
        assert n_filled >= 2
        assert not cleaned["Flow Byts/s"].isna().any()
        assert not cleaned["Flow Pkts/s"].isna().any()

    def test_remove_duplicates(self):
        df = _make_sample_df(20)
        # Add 5 exact duplicates
        extra = df.iloc[:5].copy()
        df = pd.concat([df, extra], ignore_index=True)
        assert len(df) == 25
        cleaned, n_removed = remove_duplicates(df)
        assert n_removed == 5
        assert len(cleaned) == 20

    def test_sort_chronologically(self):
        df = _make_sample_df(50)
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        sorted_df = sort_chronologically(df)
        ts = sorted_df["Timestamp"].dropna()
        assert (ts.diff().dropna() >= pd.Timedelta(0)).all(), "Not monotonically increasing"

    def test_parse_timestamp(self):
        df = _make_sample_df(10)
        # Convert to string first to simulate raw CSV state
        df["Timestamp"] = df["Timestamp"].astype(str)
        df = parse_timestamp(df)
        assert pd.api.types.is_datetime64_any_dtype(df["Timestamp"])


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Validation Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestValidation:
    def test_valid_df_passes(self):
        df = _make_sample_df(100)
        report = validate_dataframe(df, stage="test_valid")
        # Empty df has no errors
        assert isinstance(report, dict)
        assert "n_rows" in report
        assert report["n_rows"] == 100

    def test_detects_header_leak(self):
        df = _make_sample_df(50)
        df.loc[0, "Label"] = "Label"
        report = validate_dataframe(df, stage="test_header_leak")
        assert report["checks"]["label_header_leak"]["status"] == "FAIL"

    def test_detects_missing_values(self):
        df = _make_sample_df(50)
        df.loc[3, "Flow Byts/s"] = np.nan
        report = validate_dataframe(df, stage="test_missing")
        assert report["missing_values"]["total"] > 0

    def test_detects_infinite_values(self):
        df = _make_sample_df(50)
        df.loc[0, "Flow Byts/s"] = np.inf
        report = validate_dataframe(df, stage="test_inf")
        assert report["infinite_values"]["total"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Splitter Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSplitter:
    def test_basic_split_sizes(self):
        df = _make_sample_df(1000)
        result = chronological_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, align_to_day_boundary=False)
        total = result.n_train + result.n_val + result.n_test
        assert total == 1000

    def test_no_temporal_overlap(self):
        df = _make_sample_df(1000)
        result = chronological_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, align_to_day_boundary=False)
        assert result.train_end <= result.val_start or result.n_val == 0
        assert result.val_end <= result.test_start or result.n_test == 0

    def test_chronological_ordering_preserved(self):
        df = _make_sample_df(500)
        result = chronological_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, align_to_day_boundary=False)
        assert result.train["Timestamp"].is_monotonic_increasing
        assert result.validation["Timestamp"].is_monotonic_increasing
        assert result.test["Timestamp"].is_monotonic_increasing

    def test_invalid_ratios_raise(self):
        df = _make_sample_df(100)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            chronological_split(df, train_ratio=0.5, val_ratio=0.3, test_ratio=0.3)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Scaler Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestScaler:
    def _get_num_df(self, n=200):
        df = _make_sample_df(n)
        df["label_encoded"] = 0
        df["label_binary"] = 0
        return df

    def test_fit_transform_returns_same_shape(self):
        df = self._get_num_df()
        scaler = FeatureScaler(method="robust")
        result = scaler.fit_transform(df)
        assert result.shape == df.shape

    def test_transform_on_val_uses_train_stats(self):
        """Ensure validation data is transformed using training statistics."""
        df_train = self._get_num_df(200)
        df_val = self._get_num_df(50)
        scaler = FeatureScaler(method="robust")
        scaler.fit(df_train)
        # Should not raise
        result_val = scaler.transform(df_val)
        assert result_val.shape == df_val.shape

    def test_not_fitted_raises(self):
        df = self._get_num_df()
        scaler = FeatureScaler(method="standard")
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.transform(df)

    def test_save_load_roundtrip(self, tmp_path):
        df = self._get_num_df(100)
        scaler = FeatureScaler(method="robust")
        scaler.fit(df)
        save_path = tmp_path / "scaler_test.pkl"
        scaler.save(save_path)
        scaler2 = FeatureScaler.load(save_path)
        assert scaler2.method == "robust"
        assert scaler2.feature_cols == scaler.feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Windowing Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestWindowing:
    def test_window_count(self):
        """n_windows = n_rows - (W + K) + 1 when stride=1"""
        n, W, K = 100, 10, 5
        indices = build_window_indices(n, window_size=W, forecast_horizon=K, stride=1)
        expected = n - (W + K) + 1
        assert len(indices) == expected

    def test_no_windows_if_too_few_rows(self):
        indices = build_window_indices(10, window_size=15, forecast_horizon=5, stride=1)
        assert len(indices) == 0

    def test_window_input_target_no_overlap(self):
        indices = build_window_indices(50, window_size=5, forecast_horizon=3, stride=1)
        for win in indices:
            assert win.input_end < win.target_start, "Input and target windows overlap!"

    def test_window_sizes_correct(self):
        W, K = 8, 4
        indices = build_window_indices(50, window_size=W, forecast_horizon=K, stride=1)
        for win in indices:
            assert (win.input_end - win.input_start + 1) == W
            assert (win.target_end - win.target_start + 1) == K

    def test_stride(self):
        """With stride=2, should have roughly half the windows."""
        n, W, K = 100, 10, 5
        idx_s1 = build_window_indices(n, W, K, stride=1)
        idx_s2 = build_window_indices(n, W, K, stride=2)
        assert len(idx_s2) < len(idx_s1)

    def test_save_load_roundtrip(self, tmp_path):
        indices = build_window_indices(100, window_size=5, forecast_horizon=3, stride=1)
        save_path = str(tmp_path / "test_windows.npz")
        save_window_indices(indices, save_path)
        loaded = load_window_indices(save_path)
        assert len(loaded) == len(indices)
        for orig, load in zip(indices, loaded):
            assert orig.input_start == load.input_start
            assert orig.target_end == load.target_end

    def test_no_future_leakage(self):
        """Every target window must come strictly after its input window."""
        indices = build_window_indices(200, window_size=20, forecast_horizon=5, stride=1)
        for win in indices:
            assert win.target_start > win.input_end


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CSV Loader Tests (using temp files)
# ═══════════════════════════════════════════════════════════════════════════════


class TestCSVLoader:
    def test_extract_day_label(self):
        from src.data.csv_loader import _extract_day_label
        fname = "Friday-02-03-2018_TrafficForML_CICFlowMeter.csv"
        assert _extract_day_label(fname) == "Friday-02-03-2018"

    def test_discover_csv_files_missing_dir(self):
        from src.data.csv_loader import discover_csv_files
        with pytest.raises(FileNotFoundError):
            discover_csv_files("/nonexistent/path/xyz")

    def test_validate_schema_all_present(self):
        from src.data.csv_loader import validate_schema
        df = _make_sample_df(10)
        # Exclude provenance columns from the "expected" schema list,
        # since validate_schema treats source_file/source_day as provenance
        prov = {"source_file", "source_day"}
        cols = [c for c in df.columns if c not in prov]
        report = validate_schema(df, common_cols=cols)
        assert report["schema_valid"]

    def test_validate_schema_missing_col(self):
        from src.data.csv_loader import validate_schema
        df = _make_sample_df(10)
        cols = list(df.columns) + ["NonExistentColumn"]
        report = validate_schema(df, common_cols=cols)
        assert not report["schema_valid"]
        assert "NonExistentColumn" in report["missing_cols"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
