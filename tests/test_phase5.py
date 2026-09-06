"""
tests/test_phase5.py
--------------------
Comprehensive Unit Test Suite for Phase 5:
Explainability, Threat Stage Mapping, Risk Scoring & SOC Forecast Result.
"""

from pathlib import Path
import json
import yaml
import numpy as np
import pytest
import torch

from src.forecasting.risk_score import (
    calculate_risk_score,
    get_risk_level,
    calculate_forecast_confidence,
    RiskScorer,
)
from src.explainability.temporal_explainer import TemporalFeatureExplainer
from src.forecasting.forecast_result import ForecastResult, SOCForecastPipeline
from src.models.world_model import LatentNetworkWorldModel


class TestAttackMapping:
    """Test suite for attack stage taxonomy mapping."""

    @pytest.fixture
    def mapping_file(self) -> Path:
        p = Path(__file__).resolve().parents[1] / "mappings" / "attack_mapping.yaml"
        assert p.exists(), f"Mapping file not found at {p}"
        return p

    def test_attack_mapping_yaml_structure(self, mapping_file: Path) -> None:
        with open(mapping_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        assert "stages" in data
        assert "attack_to_stage" in data

        # Check all 6 stages (0 through 5)
        for stage_id in range(6):
            assert stage_id in data["stages"]
            assert "name" in data["stages"][stage_id]
            assert "severity" in data["stages"][stage_id]

    def test_all_15_dataset_classes_mapped(self, mapping_file: Path) -> None:
        with open(mapping_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        expected_classes = [
            "Benign",
            "Bot",
            "Brute Force -Web",
            "Brute Force -XSS",
            "DDOS attack-HOIC",
            "DDOS attack-LOIC-UDP",
            "DDoS attacks-LOIC-HTTP",
            "DoS attacks-GoldenEye",
            "DoS attacks-Hulk",
            "DoS attacks-SlowHTTPTest",
            "DoS attacks-Slowloris",
            "FTP-BruteForce",
            "Infilteration",
            "SQL Injection",
            "SSH-Bruteforce",
        ]

        mapped = data["attack_to_stage"]
        for cls_name in expected_classes:
            assert cls_name in mapped, f"Missing class mapping for: {cls_name}"
            stage_idx = mapped[cls_name]
            assert 0 <= stage_idx <= 5


class TestRiskScoring:
    """Test suite for risk scoring and confidence metrics."""

    def test_zero_probabilities(self) -> None:
        probs = [0.0, 0.0, 0.0, 0.0, 0.0]
        score = calculate_risk_score(probs)
        level = get_risk_level(score)
        assert score == 0.0
        assert level == "LOW"

    def test_max_probabilities(self) -> None:
        probs = [1.0, 1.0, 1.0, 1.0, 1.0]
        score = calculate_risk_score(probs)
        level = get_risk_level(score)
        assert score == 100.0
        assert level == "CRITICAL"

    def test_weighted_decay(self) -> None:
        # Near term (T+1=1.0) should yield higher risk than distant term (T+5=1.0)
        p_near = [1.0, 0.0, 0.0, 0.0, 0.0]
        p_far = [0.0, 0.0, 0.0, 0.0, 1.0]

        score_near = calculate_risk_score(p_near)
        score_far = calculate_risk_score(p_far)

        assert score_near > score_far
        assert score_near == 30.0
        assert score_far == 10.0

    def test_risk_level_boundaries(self) -> None:
        assert get_risk_level(0.0) == "LOW"
        assert get_risk_level(24.9) == "LOW"
        assert get_risk_level(25.0) == "MEDIUM"
        assert get_risk_level(49.9) == "MEDIUM"
        assert get_risk_level(50.0) == "HIGH"
        assert get_risk_level(74.9) == "HIGH"
        assert get_risk_level(75.0) == "CRITICAL"
        assert get_risk_level(100.0) == "CRITICAL"

    def test_forecast_confidence(self) -> None:
        # P = 0.5 (maximum uncertainty) -> Confidence = 0.0
        assert calculate_forecast_confidence([0.5, 0.5, 0.5, 0.5, 0.5]) == 0.0

        # P = 1.0 or 0.0 (maximum certainty) -> Confidence = 1.0
        assert calculate_forecast_confidence([1.0, 0.0, 1.0, 0.0, 1.0]) == 1.0

        # Intermediate certainty
        conf = calculate_forecast_confidence([0.8, 0.2, 0.7, 0.3, 0.9])
        assert 0.0 < conf < 1.0


class TestTemporalExplainer:
    """Test suite for TemporalFeatureExplainer."""

    @pytest.fixture
    def dummy_model(self) -> LatentNetworkWorldModel:
        return LatentNetworkWorldModel(
            num_features=68,
            latent_dim=32,
            encoder_channels=[16, 16],
            encoder_dilations=[1, 2],
            kernel_size=3,
            transition_hidden_dim=32,
            forecast_horizon_k=5,
        )

    def test_explain_sequence(self, dummy_model: LatentNetworkWorldModel) -> None:
        feat_names = [f"F_{i}" for i in range(68)]
        explainer = TemporalFeatureExplainer(dummy_model, feature_names=feat_names)

        x = np.random.randn(20, 68).astype(np.float32)
        res = explainer.explain_sequence(x, target_horizon=0, top_n=5, num_steps=10)

        assert res["target_horizon"] == "T+1"
        assert len(res["top_features"]) == 5
        assert len(res["temporal_flow_importance"]) == 20
        assert 0 <= res["most_influential_timestep"] < 20

        for f_info in res["top_features"]:
            assert "feature" in f_info
            assert "importance" in f_info
            assert "direction" in f_info
            assert f_info["direction"] in ["increases_attack_risk", "decreases_attack_risk"]


class TestSOCForecastResult:
    """Test suite for ForecastResult dataclass and SOCForecastPipeline."""

    @pytest.fixture
    def dummy_model(self) -> LatentNetworkWorldModel:
        return LatentNetworkWorldModel(
            num_features=68,
            latent_dim=32,
            encoder_channels=[16, 16],
            encoder_dilations=[1, 2],
            kernel_size=3,
            transition_hidden_dim=32,
            forecast_horizon_k=5,
        )

    def test_forecast_result_serialization(self) -> None:
        fr = ForecastResult(
            timestamp="2026-09-06 23:45:00 UTC",
            current_state="Normal Network Traffic",
            risk_score=15.5,
            risk_level="LOW",
            attack_probability={"T+1": 0.12, "T+2": 0.15, "T+3": 0.18, "T+4": 0.20, "T+5": 0.22},
            predicted_attack_type="Benign",
            predicted_stage="Stage 0 — Normal Network Traffic",
            top_features=[{"feature": "Flow Pkts/s", "importance": 0.35, "direction": "increases_attack_risk"}],
            confidence=0.72,
            latent_state_norm=5.45,
            forecast_horizon=5,
        )

        d = fr.to_dict()
        assert isinstance(d, dict)
        assert d["risk_score"] == 15.5
        assert d["risk_level"] == "LOW"

        json_str = fr.to_json()
        parsed = json.loads(json_str)
        assert parsed["timestamp"] == "2026-09-06 23:45:00 UTC"
        assert parsed["predicted_stage"] == "Stage 0 — Normal Network Traffic"

    def test_soc_pipeline_end_to_end(self, dummy_model: LatentNetworkWorldModel) -> None:
        pipeline = SOCForecastPipeline(world_model=dummy_model)
        x = np.random.randn(20, 68).astype(np.float32)

        result = pipeline.forecast(x)
        assert isinstance(result, ForecastResult)
        assert 0.0 <= result.risk_score <= 100.0
        assert result.risk_level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert len(result.attack_probability) == 5
        assert 0.0 <= result.confidence <= 1.0
        assert result.latent_state_norm >= 0.0
