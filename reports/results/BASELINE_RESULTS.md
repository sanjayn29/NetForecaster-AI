# Baseline Models Evaluation Results — Phase 2

**Date:** 2026-09-06 19:08:51  
**Dataset:** CIC-IDS2018 (Cleaned 68 Features, Window W=20, Horizon K=5)  

---

## 1. Task A: Single-Step Current Flow Classification Benchmark

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | FPR | ROC-AUC | PR-AUC | Train Time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **LogisticRegression_Binary** | 0.7508 | 0.5111 | 0.5310 | **0.4912** | 0.8070 | 0.2160 | 0.5210 | 0.0751 | 8.1s |
| **LogisticRegression_Balanced_Binary** | 0.4751 | 0.5022 | 0.5090 | **0.3732** | 0.5928 | 0.5300 | 0.5248 | 0.0757 | 7.0s |
| **LogisticRegression_Multiclass** | 0.0700 | 0.1326 | 0.0107 | **0.0198** | 0.1295 | 0.9251 | N/A | N/A | 55.4s |
| **RandomForest_Binary** | 0.9340 | 0.4672 | 0.4998 | **0.4829** | 0.9025 | 0.0004 | 0.4916 | 0.0669 | 12.2s |
| **GradientBoosting_Binary** | 0.9336 | 0.4672 | 0.4996 | **0.4828** | 0.9023 | 0.0008 | 0.5008 | 0.0651 | 58.4s |

---

## 2. Task B: 5-Horizon Future Attack Forecasting Benchmark (T+1 ... T+5)

| Model | Horizon | Accuracy | Macro F1 | Weighted F1 | FPR | ROC-AUC | PR-AUC | Brier Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| LogisticRegression_Binary | **T+1** | 0.7849 | **0.4422** | 0.6937 | 0.0042 | 0.5564 | 0.2752 | 0.1841 |
| LogisticRegression_Binary | **T+2** | 0.7840 | **0.4416** | 0.6921 | 0.0036 | 0.5549 | 0.2752 | 0.1854 |
| LogisticRegression_Binary | **T+3** | 0.7861 | **0.4404** | 0.6939 | 0.0027 | 0.5594 | 0.2774 | 0.1832 |
| LogisticRegression_Binary | **T+4** | 0.7831 | **0.4413** | 0.6907 | 0.0035 | 0.5547 | 0.2768 | 0.1859 |
| LogisticRegression_Binary | **T+5** | 0.7852 | **0.4422** | 0.6938 | 0.0038 | 0.5585 | 0.2771 | 0.1835 |
| LogisticRegression_Balanced_Binary | **T+1** | 0.7279 | **0.5053** | 0.6962 | 0.1121 | 0.5200 | 0.2250 | 0.1979 |
| LogisticRegression_Balanced_Binary | **T+2** | 0.7279 | **0.5047** | 0.6952 | 0.1104 | 0.5188 | 0.2261 | 0.1985 |
| LogisticRegression_Balanced_Binary | **T+3** | 0.7332 | **0.5083** | 0.7000 | 0.1059 | 0.5237 | 0.2283 | 0.1957 |
| LogisticRegression_Balanced_Binary | **T+4** | 0.7258 | **0.5044** | 0.6935 | 0.1124 | 0.5160 | 0.2256 | 0.1996 |
| LogisticRegression_Balanced_Binary | **T+5** | 0.7285 | **0.5059** | 0.6967 | 0.1115 | 0.5200 | 0.2255 | 0.1978 |
| RandomForest_Binary | **T+1** | 0.7784 | **0.4919** | 0.7114 | 0.0293 | 0.6631 | 0.3566 | 0.1696 |
| RandomForest_Binary | **T+2** | 0.7770 | **0.4904** | 0.7093 | 0.0292 | 0.6588 | 0.3577 | 0.1713 |
| RandomForest_Binary | **T+3** | 0.7794 | **0.4967** | 0.7141 | 0.0301 | 0.6581 | 0.3521 | 0.1722 |
| RandomForest_Binary | **T+4** | 0.7763 | **0.4942** | 0.7098 | 0.0304 | 0.6541 | 0.3499 | 0.1763 |
| RandomForest_Binary | **T+5** | 0.7788 | **0.4961** | 0.7132 | 0.0301 | 0.6601 | 0.3521 | 0.1705 |
| GradientBoosting_Binary | **T+1** | 0.7781 | **0.4951** | 0.7125 | 0.0308 | 0.6522 | 0.3356 | 0.1823 |
| GradientBoosting_Binary | **T+2** | 0.7767 | **0.4935** | 0.7105 | 0.0308 | 0.6413 | 0.3254 | 0.1829 |
| GradientBoosting_Binary | **T+3** | 0.7782 | **0.4936** | 0.7124 | 0.0307 | 0.6502 | 0.3358 | 0.1805 |
| GradientBoosting_Binary | **T+4** | 0.7761 | **0.4937** | 0.7095 | 0.0305 | 0.6453 | 0.3330 | 0.1834 |
| GradientBoosting_Binary | **T+5** | 0.7783 | **0.4944** | 0.7123 | 0.0303 | 0.6435 | 0.3302 | 0.1803 |
| LSTM_Binary | **T+1** | 0.7832 | **0.4560** | 0.6987 | 0.0105 | 0.3999 | 0.1977 | 0.2087 |
| LSTM_Binary | **T+2** | 0.7820 | **0.4555** | 0.6971 | 0.0104 | 0.3991 | 0.1992 | 0.2107 |
| LSTM_Binary | **T+3** | 0.7843 | **0.4573** | 0.7001 | 0.0100 | 0.3901 | 0.1963 | 0.2077 |
| LSTM_Binary | **T+4** | 0.7816 | **0.4570** | 0.6966 | 0.0102 | 0.3926 | 0.1991 | 0.2100 |
| LSTM_Binary | **T+5** | 0.7839 | **0.4560** | 0.6989 | 0.0095 | 0.3970 | 0.1987 | 0.2088 |
| GRU_Binary | **T+1** | 0.7844 | **0.4515** | 0.6973 | 0.0075 | 0.4519 | 0.2105 | 0.2092 |
| GRU_Binary | **T+2** | 0.7834 | **0.4512** | 0.6958 | 0.0072 | 0.4368 | 0.2064 | 0.2099 |
| GRU_Binary | **T+3** | 0.7855 | **0.4529** | 0.6988 | 0.0071 | 0.4385 | 0.2068 | 0.2084 |
| GRU_Binary | **T+4** | 0.7825 | **0.4528** | 0.6952 | 0.0077 | 0.4431 | 0.2101 | 0.2107 |
| GRU_Binary | **T+5** | 0.7848 | **0.4521** | 0.6977 | 0.0071 | 0.4491 | 0.2104 | 0.2087 |

---

## 3. Horizon-Wise Performance Degradation Summary (T+1 -> T+5)

| Model | T+1 F1 | T+2 F1 | T+3 F1 | T+4 F1 | T+5 F1 | Total F1 Degradation (Δ) |
|---|---:|---:|---:|---:|---:|---:|
| **LogisticRegression_Binary** | 0.4422 | 0.4416 | 0.4404 | 0.4413 | 0.4422 | +0.0000 |
| **LogisticRegression_Balanced_Binary** | 0.5053 | 0.5047 | 0.5083 | 0.5044 | 0.5059 | -0.0006 |
| **RandomForest_Binary** | 0.4919 | 0.4904 | 0.4967 | 0.4942 | 0.4961 | -0.0042 |
| **GradientBoosting_Binary** | 0.4951 | 0.4935 | 0.4936 | 0.4937 | 0.4944 | +0.0007 |
| **LSTM_Binary** | 0.4560 | 0.4555 | 0.4573 | 0.4570 | 0.4560 | -0.0000 |
| **GRU_Binary** | 0.4515 | 0.4512 | 0.4529 | 0.4528 | 0.4521 | -0.0007 |