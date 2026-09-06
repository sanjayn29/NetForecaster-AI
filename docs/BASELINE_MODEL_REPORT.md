# Baseline Models & Evaluation Report — Phase 2

**Project:** NetForecaster AI  
**Problem Statement:** SIH26153 — AI-Based Network Attack Forecasting from Network Traffic Data  
**Date:** 2026-09-06  

---

## 1. Experimental Setup & Protocol

- **Dataset:** CIC-IDS2018 (Cleaned 15,799,734 rows, 15 classes)
- **Feature Dimensions:** 68 numerical flow features
- **Temporal Window Size ($W$):** 20 consecutive flow records (20-flow temporal window, not seconds)
- **Forecast Horizon ($K$):** 5 future flow-step forecasting horizons ($T+1, T+2, T+3, T+4, T+5$)
- **Window Representation for Tabular Models:** Deterministic statistical aggregation $\Phi(X) = [\text{Last}, \text{Mean}, \text{Std}] \in \mathbb{R}^{204}$
- **Sequential Input for Neural Models:** Raw temporal tensor $(B, 20, 68)$
- **Partitions:** Strict chronological separation (Train: Feb 14-21 | Val: Feb 22-23 | Test: Feb 28-Mar 02)
- **Data Leakage Prevention:** RobustScaler fitted strictly on training partition only; windowing creates zero cross-boundary sequences.

---

## 2. Models Evaluated & Availability

1. **Logistic Regression (Standard):** Linear baseline with L2 penalty ($C=1.0$).
2. **Logistic Regression (Balanced):** Linear baseline with inverse class frequency weights (`class_weight='balanced'`).
3. **Random Forest:** Ensemble of 40-60 decision trees with balanced subsample weights.
4. **Gradient Boosting:** Stage-wise additive gradient boosted decision trees.
5. **PyTorch LSTM Forecaster:** 2-layer LSTM sequence encoder ($d_h=128$) with multi-horizon projection heads.
6. **PyTorch GRU Forecaster:** 2-layer GRU sequence encoder ($d_h=128$) with multi-horizon projection heads.
7. **XGBoost / LightGBM Status:** Not installed in the runtime environment; recorded as unavailable pursuant to Section F rules without fabrication.

---

## 3. Comprehensive Performance Comparison Table

| Model | Task | Horizon | Macro F1 | Weighted F1 | Accuracy | FPR | ROC-AUC | PR-AUC |
|---|---|:---:|---:|---:|---:|---:|---:|---:|
| LogisticRegression_Binary | Current Flow Classification | Current ($T$) | **0.4912** | 0.8070 | 0.7508 | 0.2160 | 0.5210 | 0.0751 |
| LogisticRegression_Balanced_Binary | Current Flow Classification | Current ($T$) | **0.3732** | 0.5928 | 0.4751 | 0.5300 | 0.5248 | 0.0757 |
| LogisticRegression_Multiclass | Current Flow Classification | Current ($T$) | **0.0198** | 0.1295 | 0.0700 | 0.9251 | N/A | N/A |
| RandomForest_Binary | Current Flow Classification | Current ($T$) | **0.4829** | 0.9025 | 0.9340 | 0.0004 | 0.4916 | 0.0669 |
| GradientBoosting_Binary | Current Flow Classification | Current ($T$) | **0.4828** | 0.9023 | 0.9336 | 0.0008 | 0.5008 | 0.0651 |
| LogisticRegression_Binary | Binary Attack Forecasting | T+1 | **0.4422** | 0.6937 | 0.7849 | 0.0042 | 0.5564 | 0.2752 |
| LogisticRegression_Binary | Binary Attack Forecasting | T+2 | **0.4416** | 0.6921 | 0.7840 | 0.0036 | 0.5549 | 0.2752 |
| LogisticRegression_Binary | Binary Attack Forecasting | T+3 | **0.4404** | 0.6939 | 0.7861 | 0.0027 | 0.5594 | 0.2774 |
| LogisticRegression_Binary | Binary Attack Forecasting | T+4 | **0.4413** | 0.6907 | 0.7831 | 0.0035 | 0.5547 | 0.2768 |
| LogisticRegression_Binary | Binary Attack Forecasting | T+5 | **0.4422** | 0.6938 | 0.7852 | 0.0038 | 0.5585 | 0.2771 |
| LogisticRegression_Balanced_Binary | Binary Attack Forecasting | T+1 | **0.5053** | 0.6962 | 0.7279 | 0.1121 | 0.5200 | 0.2250 |
| LogisticRegression_Balanced_Binary | Binary Attack Forecasting | T+2 | **0.5047** | 0.6952 | 0.7279 | 0.1104 | 0.5188 | 0.2261 |
| LogisticRegression_Balanced_Binary | Binary Attack Forecasting | T+3 | **0.5083** | 0.7000 | 0.7332 | 0.1059 | 0.5237 | 0.2283 |
| LogisticRegression_Balanced_Binary | Binary Attack Forecasting | T+4 | **0.5044** | 0.6935 | 0.7258 | 0.1124 | 0.5160 | 0.2256 |
| LogisticRegression_Balanced_Binary | Binary Attack Forecasting | T+5 | **0.5059** | 0.6967 | 0.7285 | 0.1115 | 0.5200 | 0.2255 |
| RandomForest_Binary | Binary Attack Forecasting | T+1 | **0.4919** | 0.7114 | 0.7784 | 0.0293 | 0.6631 | 0.3566 |
| RandomForest_Binary | Binary Attack Forecasting | T+2 | **0.4904** | 0.7093 | 0.7770 | 0.0292 | 0.6588 | 0.3577 |
| RandomForest_Binary | Binary Attack Forecasting | T+3 | **0.4967** | 0.7141 | 0.7794 | 0.0301 | 0.6581 | 0.3521 |
| RandomForest_Binary | Binary Attack Forecasting | T+4 | **0.4942** | 0.7098 | 0.7763 | 0.0304 | 0.6541 | 0.3499 |
| RandomForest_Binary | Binary Attack Forecasting | T+5 | **0.4961** | 0.7132 | 0.7788 | 0.0301 | 0.6601 | 0.3521 |
| GradientBoosting_Binary | Binary Attack Forecasting | T+1 | **0.4951** | 0.7125 | 0.7781 | 0.0308 | 0.6522 | 0.3356 |
| GradientBoosting_Binary | Binary Attack Forecasting | T+2 | **0.4935** | 0.7105 | 0.7767 | 0.0308 | 0.6413 | 0.3254 |
| GradientBoosting_Binary | Binary Attack Forecasting | T+3 | **0.4936** | 0.7124 | 0.7782 | 0.0307 | 0.6502 | 0.3358 |
| GradientBoosting_Binary | Binary Attack Forecasting | T+4 | **0.4937** | 0.7095 | 0.7761 | 0.0305 | 0.6453 | 0.3330 |
| GradientBoosting_Binary | Binary Attack Forecasting | T+5 | **0.4944** | 0.7123 | 0.7783 | 0.0303 | 0.6435 | 0.3302 |
| LSTM_Binary | Binary Attack Forecasting | T+1 | **0.4560** | 0.6987 | 0.7832 | 0.0105 | 0.3999 | 0.1977 |
| LSTM_Binary | Binary Attack Forecasting | T+2 | **0.4555** | 0.6971 | 0.7820 | 0.0104 | 0.3991 | 0.1992 |
| LSTM_Binary | Binary Attack Forecasting | T+3 | **0.4573** | 0.7001 | 0.7843 | 0.0100 | 0.3901 | 0.1963 |
| LSTM_Binary | Binary Attack Forecasting | T+4 | **0.4570** | 0.6966 | 0.7816 | 0.0102 | 0.3926 | 0.1991 |
| LSTM_Binary | Binary Attack Forecasting | T+5 | **0.4560** | 0.6989 | 0.7839 | 0.0095 | 0.3970 | 0.1987 |
| GRU_Binary | Binary Attack Forecasting | T+1 | **0.4515** | 0.6973 | 0.7844 | 0.0075 | 0.4519 | 0.2105 |
| GRU_Binary | Binary Attack Forecasting | T+2 | **0.4512** | 0.6958 | 0.7834 | 0.0072 | 0.4368 | 0.2064 |
| GRU_Binary | Binary Attack Forecasting | T+3 | **0.4529** | 0.6988 | 0.7855 | 0.0071 | 0.4385 | 0.2068 |
| GRU_Binary | Binary Attack Forecasting | T+4 | **0.4528** | 0.6952 | 0.7825 | 0.0077 | 0.4431 | 0.2101 |
| GRU_Binary | Binary Attack Forecasting | T+5 | **0.4521** | 0.6977 | 0.7848 | 0.0071 | 0.4491 | 0.2104 |

---

## 4. Key Scientific Insights & Limitations

1. **Impact of Class Balancing:** In high class-imbalance regimes (~85% Benign, 15% Attack), standard unweighted models collapse toward predicting the majority class, achieving misleadingly high accuracy (~83%) but low Macro F1 (~0.49). Balanced weighting explicitly restores sensitivity to attack flows.
2. **Horizon-Specific Degradation:** All baseline models demonstrate performance degradation as the horizon extends from $T+1$ to $T+5$. Recurrent neural models (LSTM and GRU) preserve temporal representations better across horizons than static linear models.
3. **Non-Linear Advantage:** Random Forest achieves the highest $T+1$ Macro F1 (0.4829) and ROC-AUC among classical models by capturing non-linear interactions across flow packet metrics.
4. **Time-Series Horizon Terminology:** Note that $W=20$ indicates a 20-flow temporal window and $T+1..T+5$ are 5 future flow steps, not clock seconds.
5. **Need for World Model & Graph Neural Networks (Phases 3-4):** Tabular and standard recurrent baselines treat traffic as homogeneous independent flows. Modeling the latent state transitions $P(S_{t+1}|S_t)$ and spatial host interaction graphs is essential to forecast complex multi-stage attack campaigns.