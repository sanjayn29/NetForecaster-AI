# NetForecaster AI — Phase 5 Results Report
## Explainability, Threat Stage Mapping & Enterprise Risk Scoring

**SIH Problem Statement:** SIH26153 — AI-Based Network Attack Forecasting from Network Traffic Data  
**Dataset:** CIC-IDS2018 Cleaned Chronological Test Partition ($N=50,000$)  
**Input Window:** $(B, 20, 68)$ historical flows  
**Underlying Model:** Latent Network State World Model (Phase 4)  
**Target:** Threat Stage, Multi-Horizon Probability ($T+1 \dots T+5$), Enterprise Risk Score ($0 \dots 100$), and Top Feature Attributions  

---

## 1. Executive Summary

Phase 5 introduces the **Explainability and Threat Interpretation Layer** on top of the Latent Network State World Model, bridging raw probabilistic machine learning outputs and actionable Security Operations Center (SOC) intelligence.

The operational pipeline transforms raw network flow sequences into standardized SOC-ready incident intelligence:

```
Observed 20 Flows (B, 20, 68)
          ↓
Latent World Model Rollout (S_t → S_{t+1} → ... → S_{t+5})
          ↓
Future Attack Probabilities [P_{t+1}, P_{t+2}, P_{t+3}, P_{t+4}, P_{t+5}]
          ↓
Enterprise Threat Risk Scorer (Decay-Weighted 0–100) & Alert Level (LOW/MED/HIGH/CRITICAL)
          ↓
Threat Stage Translation (Stages 0–5 Heuristic Taxonomy)
          ↓
Integrated Gradients Temporal Explainer (Feature Attribution & Flow Saliency)
          ↓
SOC-Ready Incident JSON Object
```

---

## 2. Threat Stage Taxonomy Translation

| Stage ID | Threat Stage Name | Target Attack Classes (CIC-IDS2018) | Heuristic Mapping Rationale |
|:---:|:---|:---|:---|
| **0** | **Normal Baseline** | `Benign` | Normal non-malicious enterprise network traffic. |
| **1** | **Initial Access / Credential** | `FTP-BruteForce`, `SSH-Bruteforce` | Authentication-layer dictionary and brute-force intrusion attempts. |
| **2** | **Web Exploitation / Injection** | `Brute Force -Web`, `Brute Force -XSS`, `SQL Injection` | Application-layer web exploit and parameter injection probes. |
| **3** | **Impact / Denial of Service** | `DoS-GoldenEye`, `DoS-Slowloris`, `DoS-SlowHTTPTest`, `DoS-Hulk`, `DDoS-LOIC-HTTP`, `DDoS-HOIC`, `DDoS-LOIC-UDP` | High-volume resource exhaustion and network disruption floods. |
| **4** | **Command & Control** | `Bot` | Botnet beaconing and periodic orchestration traffic. |
| **5** | **Lateral Movement / Infiltration** | `Infiltration` | Post-compromise internal network reconnaissance and privilege pivot. |

> [!NOTE]
> *Operational Taxonomy Disclaimer:* The mapped threat stage provides a standardized taxonomy translation of dataset attack classes to support triage workflows. In production environments, stage classification should be supplemented with endpoint telemetry and SIEM alert correlation.

---

## 3. Horizon-Decayed Enterprise Threat Risk Formulation

Near-term threat emergence poses higher immediate operational risk than distant predictions. The enterprise risk score applies a decaying simplex weighting over the 5 forecasting horizons:

$$\text{Risk Score} = 100 \times \sum_{k=1}^{5} w_k \cdot P_{t+k}, \quad w = [0.30, 0.25, 0.20, 0.15, 0.10]$$

$$\text{Forecast Confidence} = \frac{1}{K} \sum_{k=1}^{K} 2 \cdot |P_{t+k} - 0.5| \in [0.0, 1.0]$$

### Risk Level Categorization

| Risk Score Range | Alert Level | Recommended SOC Playbook Action |
|:---:|:---:|:---|
| **$0 \le \text{Score} < 25$** | **`LOW`** | Routine monitoring; baseline traffic conditions. |
| **$25 \le \text{Score} < 50$** | **`MEDIUM`** | Advisory warning; monitor connection rate anomalies. |
| **$50 \le \text{Score} < 75$** | **`HIGH`** | Heightened vigilance; stage firewall rate-limiting and ACL rules. |
| **$75 \le \text{Score} \le 100$** | **`CRITICAL`** | Active mitigation; isolate target endpoint and notify incident response. |

---

## 4. Temporal Explainability & Feature Attribution

The explainability engine uses **Integrated Gradients** computed across the $(20, 68)$ input sequence:

$$IG_i(x) = (x_i - x_i') \times \int_{0}^{1} \frac{\partial F(x' + \alpha (x - x'))}{\partial x_i} d\alpha$$

### Explainability Deliverables
1. **Top-K Feature Attribution:** Identifies top influential network flow features (e.g., `Flow Duration`, `Tot Bwd Pkts`, `Dst Port`, `Init Bwd Win Byts`).
2. **Directional Risk Influence:** Classifies whether each feature perturbation `increases_attack_risk` ($\Delta > 0$) or `decreases_attack_risk` ($\Delta < 0$).
3. **Temporal Flow Saliency:** Ranks the 20 historical timesteps ($t-19 \dots t$) by cumulative gradient magnitude to pinpoint the exact historical flow triggering the forecast.

---

## 5. Standardized SOC Output Payload

Each forecast generates a validated JSON payload ready for SIEM / SOC dashboard consumption:

```json
{
  "timestamp": "2018-02-28T09:15:32",
  "prediction_type": "Multi-Horizon Attack Forecast",
  "forecast_horizons": {
    "T+1": 0.8421,
    "T+2": 0.7915,
    "T+3": 0.7102,
    "T+4": 0.6350,
    "T+5": 0.5480
  },
  "enterprise_risk_score": 73.84,
  "alert_level": "HIGH",
  "forecast_confidence": 0.521,
  "threat_stage": {
    "stage_id": 3,
    "stage_name": "Impact / Denial of Service",
    "mapped_attack_type": "DoS-Hulk"
  },
  "top_explanatory_features": [
    {
      "feature_name": "Flow Duration",
      "attribution_score": 0.1842,
      "directional_influence": "increases_attack_risk"
    },
    {
      "feature_name": "Tot Bwd Pkts",
      "attribution_score": 0.1290,
      "directional_influence": "increases_attack_risk"
    }
  ],
  "temporal_flow_saliency": {
    "most_influential_timestep": "t-0",
    "recent_flow_weight_pct": 34.2
  },
  "taxonomy_disclaimer": "Threat stage mapping is an operational heuristic translation. Verification via SIEM/endpoint telemetry recommended."
}
```

---

## 6. Verification & Test Coverage

All Phase 5 components are validated with 10 dedicated unit tests (`tests/test_phase5.py`), maintaining **111 / 111 total tests passing (100%)** across the repository.
