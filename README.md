# NetForecaster AI

**AI-Based Network Attack Forecasting from Network Traffic Data**  
*Smart India Hackathon 2024 — Problem Statement: SIH26153*

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-125%20Passing-brightgreen.svg)]()
[![Phase Status](https://img.shields.io/badge/Status-Phase%207%20Operational-success.svg)]()

---

## 1. Executive Summary & Objective

**NetForecaster AI** is an offline, research-grade network attack forecasting system that moves beyond standard single-step intrusion detection. Rather than simply classifying past traffic flows, NetForecaster AI learns temporal traffic dynamics and models future network state transitions:

$$\mathcal{S}[t] \xrightarrow{\text{Temporal Latent World Model}} \mathcal{S}[t+1] \rightarrow \mathcal{S}[t+2] \rightarrow \dots \rightarrow \mathcal{S}[t+K]$$

The system predicts future attack probabilities $P(\text{Attack} \mid \mathcal{S}_{t+k})$, future attack types, and MITRE-aligned threat stages across a **5-step forecasting horizon ($K=5$)**, backed by an **enterprise threat risk scoring engine**, **temporal feature attribution**, and an **offline SOC cybersecurity dashboard**.

---

## 2. Phase-by-Phase Roadmap & Status

| Phase | Milestone | Status | Key Deliverables |
|---|---|:---:|---|
| **Phase 0** | **Dataset Inspection & Feasibility** | **COMPLETED** | [`docs/DATASET_REPORT.md`](docs/DATASET_REPORT.md), schema harmonization, Inf/NaN handling, Unix-epoch anomaly resolution. |
| **Phase 1** | **Data Ingestion, Cleaning & Preprocessing** | **COMPLETED & AUDITED** | 15.8M rows cleaned, RobustScaler fitted on train, non-overlapping day-boundary chronological split, window indexing ($W=20, K=5$). |
| **Phase 2** | **Baseline Models & Evaluation** | **COMPLETED & AUDITED** | Standard/Balanced Logistic Regression, Random Forest, Gradient Boosting, PyTorch LSTM, and GRU baselines. |
| **Phase 2.5** | **Scientific Validation & Baseline Audit** | **COMPLETED & AUDITED** | [`docs/PHASE2_5_VALIDATION_REPORT.md`](docs/PHASE2_5_VALIDATION_REPORT.md), [`docs/PHASE2_5_LEAKAGE_AUDIT.md`](docs/PHASE2_5_LEAKAGE_AUDIT.md), 74 unit tests passing. |
| **Phase 3** | **Temporal Sequence Modeling** | **COMPLETED & AUDITED** | [`docs/TEMPORAL_MODEL_REPORT.md`](docs/TEMPORAL_MODEL_REPORT.md), Causal TCN & Transformer sequence forecasters, Weighted BCE & Focal Loss, 91 unit tests passing. |
| **Phase 4** | **Latent State World Model** | **COMPLETED & AUDITED** | [`docs/WORLD_MODEL_REPORT.md`](docs/WORLD_MODEL_REPORT.md), State transition MLP $S_{t+1} = S_t + f(S_t)$, autoregressive latent rollout ($K=5$), 101 unit tests passing. |
| **Phase 5** | **Explainability & Threat Stage Layer** | **COMPLETED & AUDITED** | [`docs/EXPLAINABILITY_REPORT.md`](docs/EXPLAINABILITY_REPORT.md), [`reports/results/PHASE5_RESULTS.md`](reports/results/PHASE5_RESULTS.md), threat taxonomy mapping (Stages 0–5), Integrated Gradients attribution, enterprise risk score ($0–100$), SOC JSON pipeline, 111 unit tests passing. |
| **Phase 6** | **Spatial Graph Neural Network (GNN/GAT)** | *Optional Research Track* | Dynamic bipartite IP/Service host interaction graph with GATv2 attention. |
| **Phase 7** | **Offline SOC Dashboard & System Integration** | **COMPLETED & AUDITED** | [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md), [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md), [`reports/results/PHASE7_RESULTS.md`](reports/results/PHASE7_RESULTS.md), standalone FastAPI + React/Vite SOC dashboard, traffic replay simulator, 125 unit & integration tests passing. |

---

## 3. Dataset & Preprocessing Protocol

* **Dataset:** CSE-CIC-IDS2018 (10 CSV files, ~6.57 GB total).
* **Raw Rows:** **16,233,002** $\longrightarrow$ **Cleaned Rows:** **15,799,734** (433,268 duplicates/header leaks removed).
* **Feature Dimension:** **68 unified numerical flow features** (zero identifier memorization; IP/ports dropped from tabular features).
* **Chronological Split:**
  * **Train:** Feb 14, 15, 16, 20, 21 (**11,727,360 rows** / 74.2%)
  * **Validation:** Feb 22, 23 (**2,091,249 rows** / 13.2%)
  * **Test:** Feb 28, Mar 01, 02 (**1,981,125 rows** / 12.5%)
* **Temporal Window Specification:**
  * Input window $W = 20$ consecutive flow records (not seconds).
  * Forecast horizon $K = 5$ future flow steps ($T+1, T+2, T+3, T+4, T+5$).
  * Sequential representation: $(B, 20, 68)$.

---

## 4. Master Model Benchmark Comparison (Phases 2, 3, and 4)

Evaluated on held-out test data $(N=50,000)$ across forecasting horizons $T+1 \dots T+5$:

| Model | Architecture | Input / Mechanism | T+1 ROC-AUC | T+1 PR-AUC | T+1 Macro F1 | T+5 ROC-AUC | T+5 PR-AUC | T+5 Macro F1 | Brier Score | FPR |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Random Forest** | Tree Ensemble | Aggregated $[\text{Last, Mean, Std}]$ | **0.6631** | **0.3566** | 0.4919 | **0.6601** | **0.3521** | 0.4961 | **0.1696** | 0.0293 |
| **Gradient Boosting** | Tree Ensemble | Aggregated $[\text{Last, Mean, Std}]$ | 0.6522 | 0.3356 | 0.4951 | 0.6435 | 0.3302 | 0.4944 | 0.1823 | 0.0308 |
| **Balanced Logistic** | Linear Classifier | Aggregated $[\text{Last, Mean, Std}]$ | 0.5200 | 0.2250 | **0.5053** | 0.5200 | 0.2255 | 0.5059 | 0.1979 | 0.1121 |
| **Standard Logistic** | Linear Classifier | Aggregated $[\text{Last, Mean, Std}]$ | 0.5564 | 0.2752 | 0.4422 | 0.5585 | 0.2771 | 0.4422 | 0.1841 | **0.0042** |
| **LSTM Forecaster** | Recurrent NN | Raw Sequence $(20, 68)$ | 0.3999 | 0.1977 | 0.4560 | 0.3970 | 0.1987 | 0.4560 | 0.2087 | 0.0105 |
| **GRU Forecaster** | Recurrent NN | Raw Sequence $(20, 68)$ | 0.4519 | 0.2105 | 0.4515 | 0.4491 | 0.2104 | 0.4521 | 0.2092 | 0.0075 |
| **TCN Direct Head** | Causal ConvNet | Direct Multi-Head $(B, 5)$ | 0.5665 | 0.2487 | 0.4667 | 0.5537 | 0.2419 | 0.4500 | 0.1868 | 0.0255 |
| **Transformer Direct** | Self-Attention | Direct Multi-Head $(B, 5)$ | 0.4632 | 0.2187 | **0.5031** | 0.4661 | 0.2182 | **0.5066** | 0.2200 | 0.0476 |
| **World Model (Transition)** | Latent SSM | Recursive Rollout $S_t \dots S_{t+5}$ | 0.5007 | 0.2277 | 0.4771 | 0.4910 | 0.2221 | 0.4762 | 0.1949 | 0.0669 |
| **World Model (+ Recon)** | Latent SSM + Recon | Recursive Rollout $S_t \dots S_{t+5}$ | 0.4930 | 0.2254 | 0.4771 | 0.4998 | 0.2263 | 0.4681 | 0.1910 | 0.0583 |

---

## 5. Threat Interpretation & Explainability Layer (Phase 5)

* **Enterprise Risk Scoring:** Simplex horizon-decayed risk formula $\text{Risk} = 100 \times \sum w_k P_{t+k}$ ($w = [0.30, 0.25, 0.20, 0.15, 0.10]$) classifying threats into `LOW`, `MEDIUM`, `HIGH`, and `CRITICAL`.
* **Integrated Gradients Attribution:** Flow-level and feature-level attribution identifying top influential features with directional impact tags (`increases_attack_risk` vs `decreases_attack_risk`).
* **Heuristic Threat Stages (Stages 0–5):** Mappings from raw CIC-IDS2018 classes to standard threat stages (Initial Access, Web Exploitation, DoS/Impact, C2, Lateral Movement).
* **SOC-Ready JSON API:** Structured JSON incident schema designed for direct ingestion into SIEM platforms and interactive SOC dashboards.

---

## 6. Offline SOC Dashboard & Live Replay Simulator (Phase 7)

* **Unified Offline Stack:** FastAPI backend serving REST API, bi-directional WebSockets (`/ws/replay`), and the bundled React 18 + Vite cyber dashboard on a single port (`8000`).
* **Traffic Replay Simulator:** Offline chronological player streaming authentic CIC-IDS2018 test partitions with 4 curated scenarios (*Infiltration Attack*, *Botnet C2 Burst*, *Benign Baseline*, *Mixed Transitions*).
* **Forecast vs Ground Truth:** Interactive chart comparing predicted attack probabilities against true future labels as playback advances.
* **100% Offline / Air-Gapped:** Zero cloud API requirements, zero remote database connections.

---

## 7. Repository Architecture

```
NetForecaster-AI/
├── config.yaml                     # Master centralized configuration
├── README.md                       # Main project documentation & benchmark summary
├── mappings/
│   └── attack_mapping.yaml         # Threat stage taxonomy mapping (Stages 0-5)
├── docs/
│   ├── DATASET_REPORT.md           # Comprehensive exploratory data analysis
│   ├── BASELINE_MODEL_REPORT.md    # Baseline scientific methodology and evaluation
│   ├── PHASE2_5_LEAKAGE_AUDIT.md   # Formal code-level data leakage audit
│   ├── PHASE2_5_VALIDATION_REPORT.md # Phase 2.5 deep validation & distribution audit
│   ├── TEMPORAL_MODEL_REPORT.md    # Phase 3 Temporal Modeling Scientific Report
│   ├── WORLD_MODEL_REPORT.md       # Phase 4 Latent World Model Scientific Report
│   ├── EXPLAINABILITY_REPORT.md    # Phase 5 Explainability & Risk Scoring Report
│   ├── DEMO_GUIDE.md               # Step-by-step hackathon jury presentation guide
│   ├── DEPLOYMENT.md               # Standalone offline deployment instructions
│   └── PROJECT_STATUS.md           # Live progress tracker & phase roadmap
├── app/
│   ├── backend/                    # FastAPI application, REST endpoints & WebSocket replay engine
│   └── frontend/                   # React 18 + Vite + Tailwind CSS SOC Cybersecurity Dashboard
├── data/processed/                 # Cleaned Parquet partitions (train, validation, test)
├── models/
│   ├── preprocessing/              # FeatureScaler (RobustScaler), LabelEncoder, metadata
│   ├── baseline/                   # Trained Phase 2 checkpoints (Logistic, RF, GB, LSTM, GRU)
│   ├── temporal/                   # Trained Phase 3 checkpoints (TCN, Transformer, Focal variants)
│   └── world_model/                # Trained Phase 4 Latent World Model checkpoints
│       ├── best_model.pt           # Best Latent World Model weights
│       ├── config.json             # Model architecture hyperparameters
│       └── training_history.json   # Training & validation trajectories
├── reports/
│   ├── figures/                    # Diagnostic figures (Phases 2, 3, 4)
│   ├── metrics/                    # JSON benchmark metrics (Phases 2, 2.5, 3, 4)
│   └── results/                    # Markdown benchmark reports
│       ├── BASELINE_RESULTS.md
│       ├── PHASE3_RESULTS.md
│       ├── PHASE4_RESULTS.md
│       ├── PHASE5_RESULTS.md
│       └── PHASE7_RESULTS.md       # Phase 7 Full System Integration Report
├── src/
│   ├── data/                       # CSV Loader, Cleaning, Label Encoding, Validation
│   ├── preprocessing/              # Scaler, Chronological Splitter, Sequence Representation
│   ├── models/                     # Logistic, Tree, LSTM, GRU, TCN, Transformer, World Model, Losses
│   ├── evaluation/                 # Metric computation suite & visualization engines
│   ├── forecasting/                # WorldModelRolloutEngine, RiskScorer, ForecastResult, Pipeline
│   ├── explainability/             # TemporalFeatureExplainer (Integrated Gradients)
│   ├── inference/                  # ModelArtifactLoader (Singleton) & SOCInferenceService
│   ├── graph/                      # Dynamic Graph & GAT Builders (Phase 6)
│   └── utils/                      # Configuration, Logging, Random Seed utilities
├── training/                       # Baseline, Temporal, and World Model training runners
├── scripts/                        # Data prep, evaluation, and audit scripts
└── tests/
    ├── test_phase1.py              # 39 Phase 1 data pipeline tests
    ├── test_phase2.py              # 29 Phase 2 baseline model tests
    ├── test_phase2_5.py            # 6 Phase 2.5 validation and causality tests
    ├── test_phase3.py              # 17 Phase 3 TCN, Transformer & Loss tests
    ├── test_phase4.py              # 10 Phase 4 World Model & Rollout tests
    ├── test_phase5.py              # 10 Phase 5 Explainability & Risk Scoring tests
    └── test_phase7_integration.py  # 14 Phase 7 Full Stack Integration tests
```

---

## 8. Quickstart & Demonstration

### 1. Launch Offline SOC Dashboard (Single Command)
```bash
# Start backend server & frontend static bundle
uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser.

### 2. Run in Development Mode (Hot Reloading)
```bash
# Terminal 1: Backend API & WebSocket Stream
uvicorn app.backend.main:app --reload --port 8000

# Terminal 2: React 18 + Vite Frontend
cd app/frontend
npm run dev
```
Open **`http://localhost:5173`** in your browser.

### 3. Curated Demonstration Scenarios
The dashboard includes 4 pre-indexed offline traffic replay scenarios from the held-out test partition:
1. **Infiltration Attack Progression (Stage 5):** Multi-step reconnaissance leading into internal infiltration and lateral movement.
2. **Botnet Command & Control Burst (Stage 4):** Periodic botnet beaconing and orchestration traffic.
3. **Benign Enterprise Baseline (Stage 0):** Normal enterprise traffic conditions confirming low False Positive Rate ($< 3\%$).
4. **Mixed Threat Horizon Transition:** Baseline traffic transitioning into active attack bursts.

### 4. Run All Automated Unit & Integration Tests (125/125 Passing)
```bash
pytest tests/ -q
```
*(All 125 unit and integration tests execute and pass in ~15 seconds).*

---

## 9. Key Documentation & Reference Links

* **Hackathon Presentation Script:** [`docs/DEMO_GUIDE.md`](docs/DEMO_GUIDE.md)
* **Offline Deployment Guide:** [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md)
* **Phase 7 Integration Results:** [`reports/results/PHASE7_RESULTS.md`](reports/results/PHASE7_RESULTS.md)
* **Phase 5 Explainability Report:** [`docs/EXPLAINABILITY_REPORT.md`](docs/EXPLAINABILITY_REPORT.md)
* **Phase 4 Latent World Model Report:** [`docs/WORLD_MODEL_REPORT.md`](docs/WORLD_MODEL_REPORT.md)
* **Phase 3 Temporal Modeling Report:** [`docs/TEMPORAL_MODEL_REPORT.md`](docs/TEMPORAL_MODEL_REPORT.md)
* **Phase 2.5 Validation & Leakage Audit:** [`docs/PHASE2_5_VALIDATION_REPORT.md`](docs/PHASE2_5_VALIDATION_REPORT.md)
* **Phase 2 Baseline Model Report:** [`docs/BASELINE_MODEL_REPORT.md`](docs/BASELINE_MODEL_REPORT.md)
* **Phase 0 Dataset Report:** [`docs/DATASET_REPORT.md`](docs/DATASET_REPORT.md)

---

## 10. License & Citation
Developed for the Smart India Hackathon (SIH26153). All research code, model checkpoints, and documentation are strictly partitioned for reproducibility and offline security evaluation.

