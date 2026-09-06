"""
app/backend/main.py
-------------------
FastAPI Application Entrypoint for NetForecaster AI SOC Dashboard (Phase 7).
Provides REST endpoints, WebSocket streaming for Traffic Replay, and static asset serving.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.inference.model_loader import ModelArtifactLoader
from src.inference.pipeline import SOCInferenceService
from app.backend.replay import TrafficReplayEngine

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("netforecaster.api")

# Global instances
loader: Optional[ModelArtifactLoader] = None
inference_service: Optional[SOCInferenceService] = None
replay_engine: Optional[TrafficReplayEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan manager: initialize models once at startup."""
    global loader, inference_service, replay_engine
    logger.info("Initializing NetForecaster AI model artifacts and replay engine...")
    loader = ModelArtifactLoader.get_instance()
    inference_service = SOCInferenceService(loader=loader)
    replay_engine = TrafficReplayEngine(inference_service=inference_service)
    logger.info("Initialization complete. All models and artifacts ready in memory.")
    yield
    logger.info("Shutting down NetForecaster AI backend...")
    if replay_engine:
        replay_engine.pause()


app = FastAPI(
    title="NetForecaster AI — SOC Threat Forecasting API",
    description="Offline AI-Based Multi-Horizon Network Attack Forecasting and Threat Intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local dev servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response Schemas ---

class ForecastRequest(BaseModel):
    sequence: List[List[float]] = Field(
        ...,
        description="20 historical flows with 68 features each (shape: 20x68)",
    )
    timestamp: Optional[str] = Field(None, description="Optional ISO timestamp string")
    is_scaled: bool = Field(True, description="Whether features are already RobustScaled")


class SelectSegmentRequest(BaseModel):
    segment_id: str = Field(..., description="Target segment key (e.g., 'infiltration_attack', 'botnet_c2_burst')")


class SetSpeedRequest(BaseModel):
    speed: float = Field(..., ge=0.1, le=10.0, description="Playback speed multiplier (0.1x to 10.0x)")


# --- REST API Endpoints ---

@app.get("/api/health")
async def get_health() -> Dict[str, Any]:
    """Health check endpoint and artifact verification."""
    global loader, inference_service
    world_model_loaded = loader.world_model is not None if loader else False
    scaler_loaded = loader.scaler is not None if loader else False

    return {
        "status": "healthy",
        "system": "NetForecaster AI SOC Backend",
        "mode": "OFFLINE TRAFFIC REPLAY",
        "artifacts": {
            "world_model": "loaded" if world_model_loaded else "unavailable",
            "feature_scaler": "loaded" if scaler_loaded else "unavailable",
            "features_dimension": 68,
            "window_size": 20,
            "forecast_horizon": 5,
        },
    }


@app.get("/api/model-info")
async def get_model_info() -> Dict[str, Any]:
    """Return model architecture hyperparameters and operational specifications."""
    global loader
    config = loader.world_model.get_config() if (loader and loader.world_model) else {}
    return {
        "dataset": "CSE-CIC-IDS2018 (Cleaned Chronological Partitions)",
        "input_specification": {
            "window_size": "20 consecutive flow records (W=20, not seconds)",
            "features_per_flow": 68,
            "scaler": "RobustScaler (fitted strictly on training partition)",
        },
        "forecasting_architecture": {
            "type": "Latent Network State World Model",
            "latent_dim": config.get("latent_dim", 64),
            "encoder": "Temporal Convolutional Network (TCN)",
            "transition": "Residual MLP: S_{t+1} = S_t + f(S_t)",
            "rollout_mechanism": "Autoregressive recursive 5-step rollout without teacher forcing",
        },
        "explainability": {
            "method": "Integrated Gradients (5-step linear path interpolation)",
            "attribution_targets": "Top-5 features + 20-timestep historical flow saliency",
        },
        "risk_scoring": {
            "method": "Horizon-decayed weighted simplex: [0.30, 0.25, 0.20, 0.15, 0.10]",
            "alert_levels": ["LOW (0-25)", "MEDIUM (25-50)", "HIGH (50-75)", "CRITICAL (75-100)"],
        },
        "methodology_disclaimers": [
            "Current deployment operates on offline CIC-IDS2018 flow replay (not live packet capture).",
            "Forecast horizons are measured in flow records (T+1 ... T+5).",
            "Threat stages are heuristic taxonomy translations and not ground-truth lifecycle events.",
            "Random Forest provides the highest discriminative benchmark on this test partition; the World Model demonstrates recursive latent state forecasting.",
        ],
    }


@app.get("/api/benchmarks")
async def get_benchmarks() -> Dict[str, Any]:
    """Return master model benchmark comparison table."""
    global loader
    if loader:
        return loader.benchmarks
    return {"models": []}


@app.post("/api/forecast")
async def run_forecast(req: ForecastRequest) -> Dict[str, Any]:
    """Run real-time multi-horizon forecast on custom 20-flow input sequence."""
    global inference_service
    if not inference_service:
        raise HTTPException(status_code=503, detail="Inference service not ready")

    seq = req.sequence
    if len(seq) != 20 or any(len(row) != 68 for row in seq):
        raise HTTPException(
            status_code=400,
            detail=f"Input sequence must have shape (20, 68), received ({len(seq)}, {len(seq[0]) if seq else 0})",
        )

    import numpy as np
    seq_arr = np.array(seq, dtype=np.float32)

    try:
        result = inference_service.forecast_window(
            sequence=seq_arr,
            timestamp=req.timestamp,
            is_scaled=req.is_scaled,
        )
        return result
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/replay/status")
async def get_replay_status() -> Dict[str, Any]:
    """Get status of the traffic replay engine."""
    global replay_engine
    if not replay_engine:
        raise HTTPException(status_code=503, detail="Replay engine not ready")
    return replay_engine.get_status()


@app.get("/api/replay/segments")
async def get_replay_segments() -> Dict[str, Any]:
    """List all curated demonstration segments."""
    global replay_engine
    if not replay_engine:
        raise HTTPException(status_code=503, detail="Replay engine not ready")
    return replay_engine.get_segments()


@app.post("/api/replay/select-segment")
async def select_replay_segment(req: SelectSegmentRequest) -> Dict[str, Any]:
    """Switch active replay demonstration segment."""
    global replay_engine
    if not replay_engine:
        raise HTTPException(status_code=503, detail="Replay engine not ready")

    success = replay_engine.select_segment(req.segment_id)
    if not success:
        raise HTTPException(status_code=400, detail=f"Invalid segment_id: {req.segment_id}")
    return replay_engine.get_status()


@app.post("/api/replay/start")
async def start_replay() -> Dict[str, Any]:
    """Start traffic replay."""
    global replay_engine
    if not replay_engine:
        raise HTTPException(status_code=503, detail="Replay engine not ready")
    replay_engine.start()
    return {"status": "started", "replay": replay_engine.get_status()}


@app.post("/api/replay/pause")
async def pause_replay() -> Dict[str, Any]:
    """Pause traffic replay."""
    global replay_engine
    if not replay_engine:
        raise HTTPException(status_code=503, detail="Replay engine not ready")
    replay_engine.pause()
    return {"status": "paused", "replay": replay_engine.get_status()}


@app.post("/api/replay/reset")
async def reset_replay() -> Dict[str, Any]:
    """Reset traffic replay position."""
    global replay_engine
    if not replay_engine:
        raise HTTPException(status_code=503, detail="Replay engine not ready")
    replay_engine.reset()
    return {"status": "reset", "replay": replay_engine.get_status()}


@app.post("/api/replay/step")
async def step_replay() -> Dict[str, Any]:
    """Step traffic replay forward by exactly 1 flow record."""
    global replay_engine
    if not replay_engine:
        raise HTTPException(status_code=503, detail="Replay engine not ready")
    payload = replay_engine.step()
    if payload is None:
        raise HTTPException(status_code=400, detail="Unable to advance replay step")
    return payload


@app.post("/api/replay/speed")
async def set_replay_speed(req: SetSpeedRequest) -> Dict[str, Any]:
    """Set playback speed multiplier."""
    global replay_engine
    if not replay_engine:
        raise HTTPException(status_code=503, detail="Replay engine not ready")
    replay_engine.set_speed(req.speed)
    return {"status": "speed_updated", "speed": replay_engine.speed}


# --- WebSocket Replay Stream ---

@app.websocket("/ws/replay")
async def websocket_replay_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time traffic replay telemetry."""
    global replay_engine
    await websocket.accept()
    if replay_engine:
        replay_engine.register_socket(websocket)
        # Send initial frame immediately upon connection
        init_frame = replay_engine.step()
        if init_frame:
            await websocket.send_json(init_frame)

    try:
        while True:
            data = await websocket.receive_json()
            # Support control commands over WS
            action = data.get("action")
            if replay_engine and action == "start":
                replay_engine.start()
            elif replay_engine and action == "pause":
                replay_engine.pause()
            elif replay_engine and action == "reset":
                replay_engine.reset()
            elif replay_engine and action == "step":
                frame = replay_engine.step()
                if frame:
                    await websocket.send_json(frame)
            elif replay_engine and action == "speed":
                speed = float(data.get("speed", 1.0))
                replay_engine.set_speed(speed)
            elif replay_engine and action == "select_segment":
                seg = data.get("segment_id", "infiltration_attack")
                replay_engine.select_segment(seg)
    except WebSocketDisconnect:
        if replay_engine:
            replay_engine.unregister_socket(websocket)
    except Exception as e:
        logger.warning(f"WebSocket client error: {e}")
        if replay_engine:
            replay_engine.unregister_socket(websocket)


# --- Static Frontend Serving ---

frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str) -> FileResponse:
        """Serve SPA index.html for frontend routes."""
        file_target = frontend_dist / full_path
        if file_target.exists() and file_target.is_file():
            return FileResponse(file_target)
        return FileResponse(frontend_dist / "index.html")
