"""
src/inference/pipeline.py
-------------------------
Production SOC Inference & Threat Forecasting Service (Phase 7).
Wraps Latent World Model, Preprocessors, Risk Scorer, and Temporal Explainer.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch

from src.inference.model_loader import ModelArtifactLoader
from src.forecasting.risk_score import RiskScorer
from src.explainability.temporal_explainer import TemporalFeatureExplainer


class SOCInferenceService:
    """
    Production inference service for NetForecaster AI.
    """

    def __init__(self, loader: Optional[ModelArtifactLoader] = None) -> None:
        self.loader = loader or ModelArtifactLoader.get_instance()
        self.world_model = self.loader.world_model
        self.device = self.loader.device
        self.feature_names = self.loader.feature_names
        self.scaler = self.loader.scaler
        self.taxonomy = self.loader.taxonomy

        self.risk_scorer = RiskScorer()

        if self.world_model is not None:
            self.explainer = TemporalFeatureExplainer(
                model=self.world_model,
                feature_names=self.feature_names,
                device=self.device,
            )
        else:
            self.explainer = None

    def get_stage_info(self, attack_name: str) -> Dict[str, Any]:
        """Resolve attack label to threat stage metadata."""
        if not self.taxonomy or "attack_to_stage" not in self.taxonomy:
            return {
                "stage_id": 0,
                "stage_name": "Normal Baseline",
                "description": "Baseline non-malicious network traffic.",
                "disclaimer": "Heuristic stage taxonomy translation.",
            }

        stage_id = int(self.taxonomy["attack_to_stage"].get(attack_name, 0))
        stages_dict = self.taxonomy.get("stages", {})
        info = stages_dict.get(stage_id, {})

        return {
            "stage_id": stage_id,
            "stage_name": info.get("name", f"Stage {stage_id}"),
            "description": info.get("description", "Threat stage metadata."),
            "disclaimer": self.taxonomy.get("metadata", {}).get(
                "disclaimer",
                "Threat stages are heuristic interpretations mapped from the CIC-IDS2018 taxonomy."
            ),
        }

    def forecast_window(
        self,
        sequence: Union[np.ndarray, torch.Tensor],
        timestamp: Optional[str] = None,
        is_scaled: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute full end-to-end inference over a 20-flow input window.

        Args:
            sequence: Array of shape (20, 68) or (1, 20, 68)
            timestamp: Optional timestamp string
            is_scaled: Whether features are already RobustScaled

        Returns:
            Structured dictionary matching SOC dashboard and SIEM specification.
        """
        ts_str = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        if isinstance(sequence, np.ndarray):
            seq_np = sequence.copy()
        else:
            seq_np = sequence.detach().cpu().numpy()

        if seq_np.ndim == 3 and seq_np.shape[0] == 1:
            seq_np = seq_np.squeeze(0)  # (20, 68)

        if not is_scaled and self.scaler is not None:
            # Scale features using RobustScaler
            orig_shape = seq_np.shape
            seq_flat = seq_np.reshape(-1, orig_shape[-1])
            seq_flat = self.scaler.transform(seq_flat)
            seq_np = seq_flat.reshape(orig_shape)

        x_tensor = torch.from_numpy(seq_np).float().unsqueeze(0).to(self.device)  # (1, 20, 68)

        if self.world_model is None:
            raise RuntimeError("Latent World Model checkpoint is not loaded.")

        # 1. Run Latent World Model Forward Rollout
        with torch.no_grad():
            out = self.world_model(x_tensor)
            logits = out["logits"].squeeze(0).cpu().numpy()  # (5,)
            probs = 1.0 / (1.0 + np.exp(-logits))  # (5,)
            initial_state = out["initial_state"].squeeze(0).cpu().numpy()
            latent_norm = float(np.linalg.norm(initial_state))

        # 2. Multi-Horizon Forecast Probabilities
        horizon_k = len(probs)
        forecast_map = {f"T+{k + 1}": round(float(probs[k]), 4) for k in range(horizon_k)}

        # 3. Risk Profile & Confidence
        risk_profile = self.risk_scorer.evaluate(probs)
        risk_score = float(risk_profile["risk_score"])
        risk_level = str(risk_profile["risk_level"])
        confidence = float(risk_profile["confidence"])
        decision_margin = round(float(np.mean(np.abs(probs - 0.5)) * 2), 4)

        # 4. Attack Label & Threat Stage
        if probs[0] >= 0.5:
            pred_attack_label = "Generic Attack (Anomaly Detected)"
            current_state = "Attack Threat Identified"
        else:
            pred_attack_label = "Benign"
            current_state = "Normal Baseline Traffic"

        stage_info = self.get_stage_info(pred_attack_label)

        # 5. Temporal Feature Attribution (Integrated Gradients)
        top_features: List[Dict[str, Any]] = []
        temporal_flow_importance: List[float] = []
        most_influential_step = 19

        if self.explainer is not None:
            try:
                explanation = self.explainer.explain_sequence(seq_np, target_horizon=0, top_n=5)
                top_features = explanation.get("top_features", [])
                temporal_flow_importance = explanation.get("temporal_flow_importance", [])
                most_influential_step = explanation.get("most_influential_timestep", 19)
            except Exception:
                pass

        return {
            "timestamp": ts_str,
            "current_state": current_state,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "forecast_horizons": forecast_map,
            "predicted_attack_type": pred_attack_label,
            "threat_stage": stage_info,
            "top_explanatory_features": top_features,
            "temporal_flow_importance": temporal_flow_importance,
            "most_influential_timestep": f"t-{19 - most_influential_step}" if most_influential_step is not None else "t-0",
            "forecast_certainty": confidence,
            "decision_margin": decision_margin,
            "latent_state_norm": round(latent_norm, 4),
            "window_size": 20,
            "forecast_horizon_steps": horizon_k,
        }
