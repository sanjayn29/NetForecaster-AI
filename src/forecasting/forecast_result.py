"""
src/forecasting/forecast_result.py
----------------------------------
Standardized Security Operations Center (SOC) Forecast Result Object and Pipeline (Phase 5).

Transforms deep World Model outputs into structured, explainable threat intelligence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
import numpy as np
import torch

from src.forecasting.risk_score import RiskScorer
from src.explainability.temporal_explainer import TemporalFeatureExplainer
from src.models.world_model import LatentNetworkWorldModel


@dataclass
class ForecastResult:
    """
    Standardized SOC Forecast Object.
    """
    timestamp: str
    current_state: str
    risk_score: float
    risk_level: str
    attack_probability: Dict[str, float]
    predicted_attack_type: str
    predicted_stage: str
    top_features: List[Dict[str, Any]]
    confidence: float
    latent_state_norm: float
    forecast_horizon: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert object to native dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize object to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class SOCForecastPipeline:
    """
    End-to-End SOC Inference, Risk Scoring, and Explanation Pipeline.
    """

    def __init__(
        self,
        world_model: LatentNetworkWorldModel,
        feature_names: Optional[List[str]] = None,
        mapping_path: Optional[Union[str, Path]] = None,
        multiclass_model: Optional[Any] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.world_model = world_model
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(68)]
        self.device = device or torch.device("cpu")
        self.world_model.to(self.device)
        self.world_model.eval()

        self.risk_scorer = RiskScorer()
        self.explainer = TemporalFeatureExplainer(
            model=self.world_model,
            feature_names=self.feature_names,
            device=self.device,
        )
        self.multiclass_model = multiclass_model

        # Load attack stage taxonomy mappings
        if mapping_path is None:
            default_map = Path(__file__).resolve().parents[2] / "mappings" / "attack_mapping.yaml"
            mapping_path = default_map if default_map.exists() else None

        self.taxonomy = {}
        if mapping_path and Path(mapping_path).exists():
            with open(mapping_path, "r", encoding="utf-8") as f:
                self.taxonomy = yaml.safe_load(f)

    def _get_stage_for_attack(self, attack_name: str) -> str:
        """Resolve attack category to high-level threat stage."""
        if not self.taxonomy or "attack_to_stage" not in self.taxonomy:
            return "Stage mapping unavailable"

        stage_id = self.taxonomy["attack_to_stage"].get(attack_name, 0)
        stages_dict = self.taxonomy.get("stages", {})
        stage_info = stages_dict.get(stage_id, {})
        return stage_info.get("name", f"Stage {stage_id}")

    def forecast(
        self,
        sequence: Union[np.ndarray, torch.Tensor],
        timestamp: Optional[str] = None,
        target_horizon_idx: int = 0,
    ) -> ForecastResult:
        """
        Generate a complete SOC-ready ForecastResult object from an observed sequence (20, 68).
        """
        ts_str = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        if isinstance(sequence, np.ndarray):
            x_tensor = torch.from_numpy(sequence).float()
        else:
            x_tensor = sequence.float()

        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)  # (1, 20, 68)

        x_tensor = x_tensor.to(self.device)

        # 1. World Model Inference
        with torch.no_grad():
            out = self.world_model(x_tensor)
            logits = out["logits"].squeeze(0).cpu().numpy()  # (K,)
            probs = 1.0 / (1.0 + np.exp(-logits))  # (K,)
            initial_state = out["initial_state"].squeeze(0).cpu().numpy()
            latent_norm = float(np.linalg.norm(initial_state))

        # 2. Multi-Horizon Probability Map
        horizon_k = len(probs)
        attack_prob_map = {f"T+{k + 1}": round(float(probs[k]), 4) for k in range(horizon_k)}

        # 3. Risk Scoring & Confidence
        risk_profile = self.risk_scorer.evaluate(probs)
        risk_score = risk_profile["risk_score"]
        risk_level = risk_profile["risk_level"]
        confidence = risk_profile["confidence"]

        # Current State Determination (at T)
        current_state = "Attack Traffic Observed" if probs[0] >= 0.5 else "Normal Network Traffic"

        # 4. Attack Type & Threat Stage Resolution
        if self.multiclass_model is not None:
            try:
                # If multiclass model exists, predict class label
                pred_class_idx = self.multiclass_model.predict(sequence.reshape(1, -1))[0]
                attack_type = str(pred_class_idx)
            except Exception:
                attack_type = "Attack type prediction unavailable"
        else:
            if probs[0] >= 0.5:
                attack_type = "Generic Attack (Anomaly Detected)"
            else:
                attack_type = "Benign"

        predicted_stage = self._get_stage_for_attack(attack_type)

        # 5. Temporal Feature Attribution
        try:
            explanation = self.explainer.explain_sequence(
                sequence, target_horizon=target_horizon_idx, top_n=5
            )
            top_features = explanation["top_features"]
        except Exception:
            top_features = []

        return ForecastResult(
            timestamp=ts_str,
            current_state=current_state,
            risk_score=risk_score,
            risk_level=risk_level,
            attack_probability=attack_prob_map,
            predicted_attack_type=attack_type,
            predicted_stage=predicted_stage,
            top_features=top_features,
            confidence=confidence,
            latent_state_norm=round(latent_norm, 4),
            forecast_horizon=horizon_k,
        )
