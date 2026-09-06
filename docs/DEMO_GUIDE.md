# NetForecaster AI — Hackathon Presentation & Demonstration Guide
## Problem Statement: SIH26153 — AI-Based Network Attack Forecasting from Network Traffic Data

---

## 1. Executive Summary & The Core SIH Differentiator

Most existing network intrusion detection systems (NIDS) are **reactive**: they classify an attack only after malicious packets or flows have already traversed the network perimeter.

**NetForecaster AI** is **predictive**:
> *"NetForecaster AI does not only detect an attack after it happens. It forecasts how the network threat will evolve over the next five consecutive flow steps ($T+1 \dots T+5$) before those future flows are observed."*

```
Observed Traffic History (20 Flows)
               ↓
    Temporal TCN Encoder
               ↓
    Latent State S_t ∈ R^64
               ↓
Recursive Latent Rollout Transition S_{t+k} = S_{t+k-1} + f(S_{t+k-1})
               ↓
5-Step Multi-Horizon Attack Probability [P_{t+1}, P_{t+2}, P_{t+3}, P_{t+4}, P_{t+5}]
               ↓
Horizon-Decayed Enterprise Threat Risk Score (0–100) & Threat Stage (0–5)
               ↓
Integrated Gradients Temporal Explainability & Flow Saliency
```

---

## 2. Launching the Offline SOC Application

### Option A: Single-Command Full Stack (Recommended)
FastAPI serves both the REST API, WebSocket stream, and the bundled React SOC Dashboard on a single port:

```bash
uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```
Open your browser at: **`http://127.0.0.1:8000`**

### Option B: Development Mode (Vite Hot-Reload)
Run the backend and frontend development servers concurrently:

```bash
# Terminal 1: Backend API & WebSocket
uvicorn app.backend.main:app --reload --port 8000

# Terminal 2: Vite React Frontend
cd app/frontend
npm run dev
```
Open your browser at: **`http://localhost:5173`**

---

## 3. Step-by-Step Jury Demonstration Walkthrough

### Step 1: System Health & Zero-Cloud Verification
* **Action:** Point out the top header bar:
  - `● SYSTEM ONLINE`
  - `MODE: OFFLINE TRAFFIC REPLAY`
  - `DATASET: CSE-CIC-IDS2018 (Cleaned Chronological Partitions)`
* **Key Talking Point:** *"The entire machine learning inference and temporal rollout run 100% locally and offline without external API dependencies or cloud connections."*

---

### Step 2: Select Scenario 1 — "Infiltration Attack Progression (Stage 5)"
* **Action:** In the Scenario dropdown, select **`Infiltration Attack Progression (Stage 5)`** and click **`▶ PLAY REPLAY`** (or use **`STEP`** for frame-by-frame analysis).
* **Observe:**
  1. **Flow Progress:** Watch the scrubber advance from Flow 20 through Flow 150.
  2. **Threat Risk Score:** Watch the central radial gauge surge from `LOW` ($< 25$) to `HIGH` / `CRITICAL` ($> 75$).
  3. **Threat Lifecycle Stage:** The active stage advances from *Stage 0 (Normal)* to *Stage 5 (Lateral Movement / Infiltration)*.
* **Key Talking Point:** *"Notice how the model detects anomalous reconnaissance patterns in the 20 historical flows and forecasts elevated attack risk across the next 5 horizons before the infiltration payload completes."*

---

### Step 3: Demonstrate Multi-Horizon Forecasting ($T+1 \dots T+5$) vs Ground Truth
* **Action:** Focus on the central **`Recursive 5-Step Forecast vs Ground-Truth State`** chart.
* **Observe:**
  - The cyan gradient bars show the forecasted probabilities $P(\text{Attack at } T+1) \dots P(\text{Attack at } T+5)$.
  - The lower status badges show the **ACTUAL GROUND TRUTH** observed as the replay sequentially reaches those future timesteps.
* **Key Talking Point:** *"Here is the scientific differentiator: the model predicts future attack risk across $T+1 \dots T+5$. When the replay advances, the actual ground truth verifies whether an attack occurred at each step."*

---

### Step 4: Explainability & "Why is the Risk Elevated?"
* **Action:** Direct attention to the bottom-left **`Why is the Threat Risk Elevated?`** card and bottom-right **`20-Flow Temporal Memory Saliency`** timeline.
* **Observe:**
  - **Feature Attribution:** Top drivers (e.g. `Flow Duration`, `Tot Bwd Pkts`, `Init Bwd Win Byts`) are tagged with directional influence ($\uparrow$ *Increases Attack Risk* vs $\downarrow$ *Lowers Attack Risk*).
  - **Temporal Memory:** The 20-bar timeline shows which specific historical flow (e.g. $t-0, t-3, t-18$) triggered the transition.
* **Key Talking Point:** *"This is not a black-box detector. Integrated Gradients isolates the exact flow features and temporal steps responsible for the forecast, providing actionable intelligence to SOC analysts."*

---

### Step 5: Select Scenario 2 — "Botnet Command & Control Burst (Stage 4)"
* **Action:** Switch the dropdown to **`Botnet Command & Control Burst (Stage 4)`**.
* **Observe:**
  - Threat Stage immediately identifies **Stage 4 (Command & Control)**.
  - Risk scores update with high forecast certainty.
* **Key Talking Point:** *"Here we observe periodic botnet beaconing. The temporal encoder captures the rhythmic cadence of the flow sequence, attributing risk to Stage 4 C2 operations."*

---

### Step 6: Select Scenario 3 — "Benign Enterprise Traffic Baseline (Stage 0)"
* **Action:** Switch the dropdown to **`Benign Enterprise Traffic Baseline (Stage 0)`**.
* **Observe:**
  - Risk score drops into the safe green zone ($< 15 / 100$).
  - Ground truth indicates **BENIGN** across all horizons.
* **Key Talking Point:** *"The model maintains a very low false positive rate (FPR < 3%) under normal enterprise network operations."*

---

### Step 7: Open Benchmarks & Architecture Modals
* **Action:** Click the **`Benchmarks`** button in the header.
* **Observe:**
  - The master comparison table showing Random Forest, Gradient Boosting, Balanced Logistic, TCN, Transformer, and Latent World Model.
* **Scientific Integrity Note to Jury:**
  > *"We maintain full scientific honesty: Random Forest provides the strongest static benchmark discrimination on this test set. The Latent World Model provides the continuous 5-step recursive state transition and trajectory generation architecture that makes multi-horizon rollout possible."*

---

## 4. Summary of Presentation Checklist

| Feature Demo | Expected Visual Output | SIH Impact |
|---|---|---|
| **Traffic Replay Simulator** | Smooth flow playback at 0.5x–5x speed | Demonstrates operational sequence processing |
| **Central Threat Gauge** | Dynamic 0–100 score + `CRITICAL` / `HIGH` badges | Real-time SOC alert triage |
| **5-Step Forecast Chart** | Horizon probabilities ($T+1 \dots T+5$) vs Ground Truth | Proves multi-step forecasting capability |
| **Threat Stage Breadcrumbs** | Stages 0 through 5 visual progression | Kill-chain / lifecycle situational awareness |
| **Integrated Gradients** | Top-5 directional features + 20-step saliency | Explains *why* the model predicted an attack |
| **Model Comparison Table** | Full benchmark metrics across all phases | Rigorous scientific validation |
