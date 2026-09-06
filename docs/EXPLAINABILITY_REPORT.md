# NetForecaster AI — Phase 5 Scientific Research Report
## Explainability, Threat Stage Mapping & Enterprise Risk Scoring

**SIH Problem Statement:** SIH26153 — AI-Based Network Attack Forecasting from Network Traffic Data  
**Objective:** Transform raw deep model forecasts into human-interpretable, actionable Security Operations Center (SOC) threat intelligence.  
**Components:** Attack Stage Taxonomy, Risk Scoring Engine, Decision Margin Confidence, and Integrated Gradients Temporal Explainer.  
**Evaluation Status:** Completed, Audited & Verified (111/111 Unit Tests Passing).

---

## 1. Executive Summary

While deep temporal and world models (Phases 3 and 4) generate multi-horizon numerical probabilities $\hat{Y}_{T+1:T+5} \in [0, 1]^5$, raw probabilities alone are insufficient for operational network security analysts. Security analysts require:
1. **Aggregated Threat Prioritization:** A single, transparent risk metric ($0–100$) reflecting urgency across the 5-step forecasting window.
2. **Operational Threat Context:** High-level categorization of what stage in an attack lifecycle the network is entering (e.g., Initial Access vs. Lateral Movement vs. Volumetric Impact).
3. **Actionable Root-Cause Attribution:** Precise identification of which packet/flow features and historical time steps drove the forecast.

Phase 5 introduces the production-ready interpretation layer that converts model states into structured SOC intelligence:
$$\text{Observed Traffic } X_t \longrightarrow \text{Latent World Model } \mathcal{S}_t \longrightarrow \text{Multi-Horizon Forecast} \longrightarrow \text{SOC Forecast Object}$$

---

## 2. Attack Stage Taxonomy Mapping

### 2.1 Threat Lifecycle Stages
All 15 classes in the CSE-CIC-IDS2018 dataset are mapped to 6 high-level operational threat stages in [`mappings/attack_mapping.yaml`](file:///c:/Users/sanja/OneDrive/Desktop/Documents/Projects/NetForecaster-AI/mappings/attack_mapping.yaml):

| Stage ID | Stage Name | Dataset Attack Classes | MITRE ATT&CK Tactic Alignment | Severity |
|:---:|---|---|---|:---:|
| **Stage 0** | **Normal Network Traffic** | `Benign` | Normal Enterprise Baseline | Informational |
| **Stage 1** | **Initial Access & Credential Access** | `SSH-Bruteforce`, `FTP-BruteForce`, `Brute Force -Web` | Credential Access (TA0006), Initial Access (TA0001) | Medium |
| **Stage 2** | **Web Application Exploitation** | `Brute Force -XSS`, `SQL Injection` | Execution (TA0002), Initial Access (TA0001) | High |
| **Stage 3** | **Service Disruption & DoS/DDoS** | `DDOS attack-HOIC`, `DDoS attacks-LOIC-HTTP`, `DDOS attack-LOIC-UDP`, `DoS attacks-Hulk`, `DoS attacks-GoldenEye`, `DoS attacks-SlowHTTPTest`, `DoS attacks-Slowloris` | Impact (TA0040) | High |
| **Stage 4** | **Command and Control (C2)** | `Bot` | Command and Control (TA0011) | Critical |
| **Stage 5** | **Infiltration & Lateral Movement** | `Infilteration` | Lateral Movement (TA0008), Exfiltration (TA0010) | Critical |

### 2.2 Important Scientific Limitation & Boundary
> [!IMPORTANT]
> **Heuristic Taxonomy Disclaimer:**
> * This mapping is an operational heuristic translation designed to assist SOC analysts in triaging alerts.
> * It does **not** constitute ground-truth multi-stage MITRE ATT&CK sensor telemetry.
> * The dataset labels represent standalone attack campaigns captured during individual testbed days.

---

## 3. Threat Risk Scoring & Model Confidence

### 3.1 Horizon-Weighted Risk Scoring
The enterprise risk score aggregates future attack probabilities with exponential temporal decay, placing higher priority on immediate near-term threats ($T+1$) while factoring in compounding future risks ($T+2 \dots T+5$):
$$\text{Risk Score} = 100 \times \sum_{k=1}^5 w_k \cdot P(\text{Attack at } t+k)$$
where the normalized decay weights are:
$$w = [0.30, 0.25, 0.20, 0.15, 0.10]$$

### 3.2 Categorical Severity Levels
The continuous $0–100$ risk score is mapped to standardized operational alert tiers:
* **`0.0 – 24.9`:** **`LOW`** (Routine baseline traffic; standard monitoring).
* **`25.0 – 49.9`:** **`MEDIUM`** (Elevated flow anomalies; early alert flagging).
* **`50.0 – 74.9`:** **`HIGH`** (Substantial attack probability; automated rate limiting recommended).
* **`75.0 – 100.0`:** **`CRITICAL`** (Imminent multi-horizon breach or DoS; automated isolation recommended).

### 3.3 Model Decision Margin Confidence
To distinguish high-certainty predictions from boundary cases without claiming uncalibrated Bayesian posterior truth:
$$\text{Model Confidence} = \frac{1}{K} \sum_{k=1}^K 2 \cdot |P_{t+k} - 0.5| \in [0.0, 1.0]$$
* A confidence of `1.0` indicates all horizon predictions are near $0.0$ or $1.0$ (maximum model decisiveness).
* A confidence of `0.0` indicates all horizon predictions are at the maximum uncertainty boundary ($0.5$).

---

## 4. Temporal Feature Attribution Engine

The explainability engine ([`src/explainability/temporal_explainer.py`](file:///c:/Users/sanja/OneDrive/Desktop/Documents/Projects/NetForecaster-AI/src/explainability/temporal_explainer.py)) utilizes **Integrated Gradients** over the observed $(20, 68)$ input tensor:
$$\text{Attribution}(X) = (X - X_{\text{base}}) \odot \frac{1}{M} \sum_{m=1}^M \nabla_X \text{Logit}_{T+1}\left(X_{\text{base}} + \frac{m}{M}(X - X_{\text{base}})\right)$$
where $X_{\text{base}} = \mathbf{0} \in \mathbb{R}^{20 \times 68}$ and $M=20$ interpolation steps.

### Attribution Outputs:
1. **Top Feature Importance:** Ranked list of the most influential flow features contributing to the forecast.
2. **Directional Risk Influence:** Explicit direction tagging:
   * `increases_attack_risk` (positive gradient attribution driving prediction toward attack).
   * `decreases_attack_risk` (negative gradient attribution suppressing attack probability).
3. **Temporal Flow Importance Profile:** Normalized weight vector across the 20 historical flow steps ($t-19 \dots t$), highlighting whether immediate recent flows ($t-2 \dots t$) or earlier burst initiations ($t-15 \dots t-10$) triggered the forecast.

---

## 5. Standardized SOC Forecast Result Object

The pipeline outputs a self-contained, JSON-serializable [`ForecastResult`](file:///c:/Users/sanja/OneDrive/Desktop/Documents/Projects/NetForecaster-AI/src/forecasting/forecast_result.py) object ready for SIEM ingestion, WebSocket dispatch, and dashboard visualizers:

```json
{
  "timestamp": "2026-09-06 23:45:00 UTC",
  "current_state": "Normal Network Traffic",
  "risk_score": 38.45,
  "risk_level": "MEDIUM",
  "attack_probability": {
    "T+1": 0.4215,
    "T+2": 0.3980,
    "T+3": 0.3650,
    "T+4": 0.3420,
    "T+5": 0.3150
  },
  "predicted_attack_type": "Generic Attack (Anomaly Detected)",
  "predicted_stage": "Stage 0 — Normal Network Traffic",
  "top_features": [
    {
      "feature": "Flow Pkts/s",
      "importance": 0.2845,
      "raw_attribution": 0.041250,
      "direction": "increases_attack_risk"
    },
    {
      "feature": "TotLen Fwd Pkts",
      "importance": 0.2110,
      "raw_attribution": 0.030612,
      "direction": "increases_attack_risk"
    },
    {
      "feature": "Flow Duration",
      "importance": 0.1652,
      "raw_attribution": -0.024010,
      "direction": "decreases_attack_risk"
    },
    {
      "feature": "Fwd Pkt Len Max",
      "importance": 0.1230,
      "raw_attribution": 0.017845,
      "direction": "increases_attack_risk"
    },
    {
      "feature": "Bwd IAT Mean",
      "importance": 0.0982,
      "raw_attribution": 0.014230,
      "direction": "increases_attack_risk"
    }
  ],
  "confidence": 0.2834,
  "latent_state_norm": 5.4821,
  "forecast_horizon": 5
}
```

---

## 6. Verification & Test Suite Status

* **Unit Test Status:** **111 / 111 passing (100%)**
  * `tests/test_phase1.py`: 39 tests (Data cleaning, scaling, windowing)
  * `tests/test_phase2.py`: 29 tests (Classical and recurrent baselines)
  * `tests/test_phase2_5.py`: 6 tests (Leakage and autocorrelation validation)
  * `tests/test_phase3.py`: 17 tests (TCN and Transformer forecasters)
  * `tests/test_phase4.py`: 10 tests (Latent World Model & Rollouts)
  * `tests/test_phase5.py`: 10 tests (Stage mapping, Risk scoring, Explainer, SOC results)
