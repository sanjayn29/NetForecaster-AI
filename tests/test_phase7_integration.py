"""
tests/test_phase7_integration.py
--------------------------------
Integration test suite for Phase 7 End-to-End SOC System, Inference Service, and FastAPI backend.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from src.inference.model_loader import ModelArtifactLoader
from src.inference.pipeline import SOCInferenceService
from app.backend.replay import TrafficReplayEngine
from app.backend.main import app


@pytest.fixture(scope="module")
def artifact_loader():
    """Load model artifact singleton."""
    return ModelArtifactLoader.get_instance()


@pytest.fixture(scope="module")
def inference_service(artifact_loader):
    """Initialize SOC inference service."""
    return SOCInferenceService(loader=artifact_loader)


@pytest.fixture(scope="module")
def test_client():
    """FastAPI TestClient with lifespan context."""
    with TestClient(app) as client:
        yield client


class TestInferencePipelineIntegration:
    """Validate inference service and artifact loading."""

    def test_artifact_loader_singleton(self, artifact_loader: ModelArtifactLoader):
        assert artifact_loader is not None
        assert artifact_loader.world_model is not None
        assert artifact_loader.scaler is not None
        assert len(artifact_loader.feature_names) == 68
        assert "models" in artifact_loader.benchmarks

    def test_soc_inference_service_forecast(self, inference_service: SOCInferenceService):
        # Create random normalized 20x68 window
        rng = np.random.RandomState(42)
        window = rng.randn(20, 68).astype(np.float32)

        res = inference_service.forecast_window(window, timestamp="2018-02-28 09:15:00", is_scaled=True)

        assert "timestamp" in res
        assert "risk_score" in res
        assert 0.0 <= res["risk_score"] <= 100.0
        assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

        # Check 5 horizons
        forecasts = res["forecast_horizons"]
        assert len(forecasts) == 5
        for k in range(1, 6):
            key = f"T+{k}"
            assert key in forecasts
            assert 0.0 <= forecasts[key] <= 1.0

        # Check threat stage
        stage = res["threat_stage"]
        assert "stage_id" in stage
        assert "stage_name" in stage
        assert "disclaimer" in stage

        # Check explainability
        assert "top_explanatory_features" in res
        assert "temporal_flow_importance" in res
        assert len(res["temporal_flow_importance"]) == 20
        assert "forecast_certainty" in res
        assert "decision_margin" in res

    def test_unscaled_inference(self, inference_service: SOCInferenceService):
        raw_window = np.abs(np.random.randn(20, 68) * 1000).astype(np.float32)
        res = inference_service.forecast_window(raw_window, is_scaled=False)
        assert 0.0 <= res["risk_score"] <= 100.0


class TestTrafficReplayEngine:
    """Validate offline traffic replay engine."""

    def test_replay_initialization(self, inference_service: SOCInferenceService):
        replay = TrafficReplayEngine(inference_service=inference_service)
        status = replay.get_status()
        assert status["current_step"] == 20
        assert status["total_steps"] >= 20
        assert status["is_running"] is False

    def test_replay_step_execution(self, inference_service: SOCInferenceService):
        replay = TrafficReplayEngine(inference_service=inference_service)
        payload = replay.step()
        assert payload is not None
        assert payload["type"] == "replay_update"
        assert "forecast" in payload
        assert "actual_ground_truth" in payload

        ground_truth = payload["actual_ground_truth"]
        assert "labels" in ground_truth
        assert "binary" in ground_truth
        assert len(ground_truth["binary"]) == 5

    def test_segment_selection(self, inference_service: SOCInferenceService):
        replay = TrafficReplayEngine(inference_service=inference_service)
        success = replay.select_segment("botnet_c2_burst")
        assert success is True
        assert replay.active_segment_id == "botnet_c2_burst"
        assert replay.current_step == 20


class TestFastAPIEndpoints:
    """Validate FastAPI REST and WebSocket API endpoints."""

    def test_health_endpoint(self, test_client: TestClient):
        resp = test_client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["artifacts"]["world_model"] == "loaded"

    def test_model_info_endpoint(self, test_client: TestClient):
        resp = test_client.get("/api/model-info")
        assert resp.status_code == 200
        data = resp.json()
        assert "forecasting_architecture" in data
        assert "methodology_disclaimers" in data
        assert len(data["methodology_disclaimers"]) >= 3

    def test_benchmarks_endpoint(self, test_client: TestClient):
        resp = test_client.get("/api/benchmarks")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) >= 5

    def test_forecast_endpoint(self, test_client: TestClient):
        # 20x68 valid input
        dummy_seq = [[0.0] * 68 for _ in range(20)]
        resp = test_client.post("/api/forecast", json={"sequence": dummy_seq, "is_scaled": True})
        assert resp.status_code == 200
        data = resp.json()
        assert "risk_score" in data
        assert "forecast_horizons" in data
        assert len(data["forecast_horizons"]) == 5

    def test_forecast_endpoint_invalid_shape(self, test_client: TestClient):
        invalid_seq = [[0.0] * 10 for _ in range(5)]
        resp = test_client.post("/api/forecast", json={"sequence": invalid_seq})
        assert resp.status_code == 400

    def test_replay_control_endpoints(self, test_client: TestClient):
        # Get segments
        resp = test_client.get("/api/replay/segments")
        assert resp.status_code == 200
        segments = resp.json()
        assert "infiltration_attack" in segments["segments"]

        # Select segment
        resp = test_client.post("/api/replay/select-segment", json={"segment_id": "benign_baseline"})
        assert resp.status_code == 200
        assert resp.json()["active_segment"] == "benign_baseline"

        # Step
        resp = test_client.post("/api/replay/step")
        assert resp.status_code == 200
        data = resp.json()
        assert "forecast" in data
        assert "actual_ground_truth" in data

        # Set speed
        resp = test_client.post("/api/replay/speed", json={"speed": 2.0})
        assert resp.status_code == 200
        assert resp.json()["speed"] == 2.0

        # Start and pause
        resp = test_client.post("/api/replay/start")
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

        resp = test_client.post("/api/replay/pause")
        assert resp.status_code == 200
        assert resp.json()["status"] == "paused"

        # Reset
        resp = test_client.post("/api/replay/reset")
        assert resp.status_code == 200
        assert resp.json()["status"] == "reset"

    def test_websocket_replay_stream(self, test_client: TestClient):
        with test_client.websocket_connect("/ws/replay") as ws:
            # First message received on connect
            data = ws.receive_json()
            assert data["type"] == "replay_update"
            assert "forecast" in data

            # Send a step command over WebSocket
            ws.send_json({"action": "step"})
            step_data = ws.receive_json()
            assert step_data["type"] == "replay_update"

    def test_spa_static_serving(self, test_client: TestClient):
        resp = test_client.get("/")
        assert resp.status_code == 200
        assert "NetForecaster AI" in resp.text
