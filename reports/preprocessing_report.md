# Preprocessing Report — Phase 1

**Project:** NetForecaster AI  
**Dataset:** CIC-IDS2018  
**Date:** 2026-09-06  

---

## 1. Input

| Item | Value |
|------|-------|
| Input files | 10 |
| Total rows (raw) | 16,232,943 |
| Original columns | 80 |

### Rows per file

| File | Rows (post-cleaning) |
|------|--------------------:|
| `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv` | 1,043,116 |
| `Friday-16-02-2018_TrafficForML_CICFlowMeter.csv` | 900,988 |
| `Friday-23-02-2018_TrafficForML_CICFlowMeter.csv` | 1,045,961 |
| `Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv` | 7,926,258 |
| `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv` | 331,027 |
| `Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv` | 1,046,154 |
| `Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv` | 1,045,288 |
| `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv` | 822,942 |
| `Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv` | 1,031,018 |
| `Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv` | 606,982 |

---

## 2. Cleaning Summary

| Step | Count |
|------|------:|
| Header-leak rows removed | 0 |
| Epoch-anomaly rows removed | 14 |
| Inf values replaced → NaN | 131,799 |
| NaN values filled | 191,520 |
| Duplicate rows removed | 433,195 |
| Constant-zero cols dropped | 10 |

**Constant-zero columns dropped:** `Bwd PSH Flags`, `Fwd URG Flags`, `Bwd URG Flags`, `CWE Flag Count`, `Fwd Byts/b Avg`, `Fwd Pkts/b Avg`, `Fwd Blk Rate Avg`, `Bwd Byts/b Avg`, `Bwd Pkts/b Avg`, `Bwd Blk Rate Avg`

**Rows before cleaning:** 16,232,943  
**Rows after cleaning:** 15,799,734  
**Rows removed total:** 433,209

---

## 3. Feature Columns

| Item | Value |
|------|-------|
| Original columns | 80 |
| Constant-zero removed | 10 |
| Tuesday-only ID cols removed | 4 |
| Non-feature cols (label, ts, source) | 5 |
| **Final feature count** | **68** |

---

## 4. Label Distribution (After Cleaning)

| Label | Count | % |
|-------|------:|--:|
| Benign | 13,445,484 | 85.10% |
| DDOS attack-HOIC | 668,461 | 4.23% |
| DDoS attacks-LOIC-HTTP | 576,175 | 3.65% |
| DoS attacks-Hulk | 434,873 | 2.75% |
| Bot | 282,310 | 1.79% |
| Infilteration | 161,897 | 1.02% |
| SSH-Bruteforce | 117,322 | 0.74% |
| DoS attacks-GoldenEye | 41,455 | 0.26% |
| FTP-BruteForce | 39,352 | 0.25% |
| DoS attacks-SlowHTTPTest | 19,462 | 0.12% |
| DoS attacks-Slowloris | 10,285 | 0.07% |
| DDOS attack-LOIC-UDP | 1,730 | 0.01% |
| Brute Force -Web | 611 | 0.00% |
| Brute Force -XSS | 230 | 0.00% |
| SQL Injection | 87 | 0.00% |

---

## 5. Timestamp Coverage

| Item | Value |
|------|-------|
| Min timestamp | 2018-02-14 01:00:00 |
| Max timestamp | 2018-03-02 12:59:59 |
| Span | 16.50 days |

---

## 6. Chronological Split

| Split | Rows | Start | End |
|-------|-----:|-------|-----|
| Train | 11,727,360 | 2018-02-14 01:00:00 | 2018-02-21 10:43:21 |
| Validation | 2,091,249 | 2018-02-22 01:00:00 | 2018-02-23 12:59:59 |
| Test | 1,981,125 | 2018-02-28 01:00:00 | 2018-03-02 12:59:59 |

---

## 7. Temporal Windowing

| Parameter | Value |
|-----------|-------|
| Window size (W) | 20 |
| Forecast horizon (K) | 5 |
| Stride | 1 |
| Train windows | 11,727,336 |
| Validation windows | 2,091,225 |
| Test windows | 1,981,101 |
| Input tensor shape | `(20, 68)` |
| Target tensor shape | `(5,)` |

---

## 8. Preprocessing Artifacts

| Artifact | Path |
|----------|------|
| Train Parquet | `data/processed/train.parquet` |
| Validation Parquet | `data/processed/validation.parquet` |
| Test Parquet | `data/processed/test.parquet` |
| Label encoder | `models/preprocessing/label_encoder.json` |
| Feature scaler | `models/preprocessing/feature_scaler.pkl` |
| Feature metadata | `models/preprocessing/feature_metadata.json` |
| Train window indices | `data/processed/windows/train_window_indices.npz` |
| Val window indices | `data/processed/windows/validation_window_indices.npz` |
| Test window indices | `data/processed/windows/test_window_indices.npz` |

---

*Total preprocessing time: 1626.7 seconds*