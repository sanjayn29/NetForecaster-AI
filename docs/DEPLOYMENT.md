# NetForecaster AI — Offline Deployment & Operations Manual
## Smart India Hackathon (SIH26153)

This document provides instructions for deploying and running NetForecaster AI in an offline, air-gapped environment.

---

## 1. System Requirements

* **Operating System:** Windows 10/11, Ubuntu 20.04+, macOS 12+
* **Python Version:** Python 3.10 to 3.13
* **Node.js (Optional, for frontend development):** Node 18+ and npm 9+
* **Hardware Requirements:**
  * RAM: Minimum 8 GB (16 GB recommended for full dataset exploration)
  * CPU: 4 cores or higher
  * GPU: Optional (CPU inference executes in < 15ms per 20-flow window)
  * Disk Space: ~500 MB for checkpoints, dependencies, and demo test slices

---

## 2. Standalone Offline Startup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Build the Frontend (Pre-built in `app/frontend/dist/`)
If modifying the frontend:
```bash
cd app/frontend
npm install
npm run build
cd ../..
```

### Step 3: Launch Unified Application Server
```bash
uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in any modern web browser (Chrome, Edge, Firefox).

---

## 3. REST API Endpoint Reference

| Method | Endpoint | Description |
|:---:|:---|:---|
| `GET` | `/api/health` | System health check and model loading confirmation. |
| `GET` | `/api/model-info` | Architectural specs, feature dimensionality, and scientific notes. |
| `GET` | `/api/benchmarks` | Benchmark metrics comparison (RF, GB, Logistic, TCN, Transformer, World Model). |
| `POST` | `/api/forecast` | Direct inference endpoint accepting a $(20, 68)$ flow array. |
| `GET` | `/api/replay/status` | Current playback state, flow index, and active segment. |
| `GET` | `/api/replay/segments` | List curated demonstration scenarios. |
| `POST` | `/api/replay/select-segment` | Switch active scenario (`infiltration_attack`, `botnet_c2_burst`, `benign_baseline`, `mixed_transition`). |
| `POST` | `/api/replay/start` | Start/resume continuous replay streaming. |
| `POST` | `/api/replay/pause` | Pause continuous replay streaming. |
| `POST` | `/api/replay/reset` | Reset playback index to the start of the sequence. |
| `POST` | `/api/replay/step` | Advance playback by exactly one flow record. |
| `POST` | `/api/replay/speed` | Set playback speed ($0.1\times \dots 10.0\times$). |
| `WS` | `/ws/replay` | Bi-directional WebSocket stream for real-time telemetry. |

---

## 4. Verification & Testing

Execute the complete 125-test suite locally:

```bash
python -m pytest tests/ -q -p no:cacheprovider
```

Expected result:
```text
======================= 125 passed, 4 warnings in ~15s =======================
```
