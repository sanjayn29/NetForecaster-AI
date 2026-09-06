# Phase 2.5: Data Leakage and Preprocessing Audit Report

**Project:** NetForecaster AI  
**Problem Statement:** SIH26153 — AI-Based Network Attack Forecasting from Network Traffic Data  
**Audit Date:** 2026-09-06  
**Audit Verdict:** **PASS — Zero Data Leakage Detected across Preprocessing, Windowing, Model Fitting, and Evaluation.**

---

## 1. Executive Summary

A comprehensive, code-level and artifact-level audit was conducted across the entire preprocessing, sequence creation, feature extraction, model fitting, and evaluation pipeline of NetForecaster AI.

The audit verified four fundamental integrity invariants:
1. **Scaler Hygiene:** Preprocessing scalers are fitted exclusively on training data.
2. **Temporal Causality:** Temporal windowing and tabular aggregations use zero future information.
3. **Partition Isolation:** Training, validation, and test datasets are strictly partitioned by non-overlapping chronological date boundaries.
4. **Evaluation Integrity:** Test data is never used for hyperparameter tuning, loss computation, model selection, or fitting.

---

## 2. Invariant 1: Preprocessing & Scaler Hygiene

### Code Evidence from `src/preprocessing/scaler.py` & `scripts/prepare_data.py`
```python
# scripts/prepare_data.py
# Fitting scaler ONLY on the training partition:
scaler = FeatureScaler(method=CFG.features.scaler)
train_df = scaler.fit_transform(train_df, feature_cols=feature_cols)

# Applying frozen training scaler to validation and test partitions:
val_df = scaler.transform(val_df)
test_df = scaler.transform(test_df)
```

* **Verification:** The parameters of the `RobustScaler` (medians: $\text{center\_}$, interquartile ranges: $\text{scale\_}$) were computed strictly from the 11,727,360 rows of the training set (Feb 14 – Feb 21).
* **Validation & Test Sets:** Validation (2,091,249 rows) and Test (1,981,125 rows) sets were transformed using the frozen training parameters without updating `center_` or `scale_`.
* **Automated Test Evidence:** Tested and verified in [`tests/test_phase2.py::TestPhase2MandatoryRequirements::test_no_train_test_preprocessing_leakage`](file:///c:/Users/sanja/OneDrive/Desktop/Documents/Projects/NetForecaster-AI/tests/test_phase2.py#L350-L370).

---

## 3. Invariant 2: Temporal Windowing & Feature Aggregation Integrity

### Code Evidence from `src/preprocessing/window_representation.py` & `src/preprocessing/windowing.py`
* **Window Size:** $W = 20$ consecutive flow records.
* **Forecast Horizons:** $K = 5$ future flow steps ($T+1, T+2, T+3, T+4, T+5$).
* **Mathematical Definition:**
  $$\text{Input Window: } X_{\text{input}} = X[t : t+W] = \{x_t, x_{t+1}, \dots, x_{t+19}\}$$
  $$\text{Tabular Aggregation: } \Phi(X_{\text{input}}) = [x_{t+19}, \text{Mean}(X_{\text{input}}), \text{Std}(X_{\text{input}})] \in \mathbb{R}^{204}$$
  $$\text{Forecast Targets: } Y = [y_{t+20}, y_{t+21}, y_{t+22}, y_{t+23}, y_{t+24}]$$
* **Zero Future Lookahead:** The index sets $\{t, \dots, t+19\}$ and $\{t+20, \dots, t+24\}$ are disjoint:
  $$\{t, \dots, t+19\} \cap \{t+20, \dots, t+24\} = \emptyset$$
* **Verification:** The tabular representation $\Phi(X_{\text{input}})$ uses strictly rows $0 \dots W-1$. Modifying future flow attributes in rows $t+W \dots t+W+K-1$ produces zero change in $\Phi(X_{\text{input}})$.
* **Automated Test Evidence:** Verified in [`tests/test_phase2.py::TestPhase2MandatoryRequirements::test_no_future_information_enters_tabular_representation`](file:///c:/Users/sanja/OneDrive/Desktop/Documents/Projects/NetForecaster-AI/tests/test_phase2.py#L287-L300).

---

## 4. Invariant 3: Chronological Partition Isolation

### Date and Timestamp Coverage
| Split | Dates | Row Count | Sequence Count | Timestamp Range |
|---|---|---:|---:|---|
| **Train** | Feb 14, 15, 16, 20, 21 | 11,727,360 | 11,727,336 | `2018-02-14 01:00:00` $\rightarrow$ `2018-02-21 10:43:21` |
| **Validation** | Feb 22, 23 | 2,091,249 | 2,091,225 | `2018-02-22 01:00:00` $\rightarrow$ `2018-02-23 12:59:59` |
| **Test** | Feb 28, Mar 01, 02 | 1,981,125 | 1,981,101 | `2018-02-28 01:00:00` $\rightarrow$ `2018-03-02 12:59:59` |

* **Zero Cross-Boundary Windowing:** Window sequences are constructed independently within each partition. No window spans from Train into Validation, or from Validation into Test.
* **Strict Monotonic Separation:** $\max(\text{Train}) < \min(\text{Val}) < \max(\text{Val}) < \min(\text{Test})$.

---

## 5. Invariant 4: Model Training & Evaluation Protocol

1. **Model Fitting:**
   * Logistic Regression, Random Forest, Gradient Boosting, PyTorch LSTM, and PyTorch GRU were trained strictly on training data (`train.parquet`).
2. **Validation Tuning:**
   * Validation set (`validation.parquet`) was used to evaluate epoch convergence, early stopping checkpoints, and decision thresholds.
3. **Test Benchmarking:**
   * Test set (`test.parquet`) was evaluated strictly at test time via `scripts/evaluate_baselines.py`. No gradient updates, hyperparameter adjustments, or threshold adaptations were performed on test data.

---

## 6. Audit Verdict

**PASSED.** All Phase 2 baseline models and Phase 1 data artifacts adhere strictly to leak-free, reproducible experimental standards.
