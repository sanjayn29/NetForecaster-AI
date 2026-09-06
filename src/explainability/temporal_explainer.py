"""
src/explainability/temporal_explainer.py
---------------------------------------
Temporal Feature Attribution & Explainability Engine (Phase 5).

Computes:
  1. Integrated Gradients / Saliency feature attribution over (20, 68) input sequence
  2. Directional risk influence ('increases_attack_risk' vs 'decreases_attack_risk')
  3. Per-timestep temporal flow importance (t-19 ... t)
  4. Top-K explanatory features for SOC inspection
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class TemporalFeatureExplainer:
    """
    Feature attribution and temporal saliency explainer for PyTorch sequence forecasters.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(68)]
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def attribute(
        self,
        x: Union[np.ndarray, torch.Tensor],
        target_horizon: int = 0,
        num_steps: int = 20,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Integrated Gradients feature attribution.

        Args:
            x: Input sequence of shape (1, 20, 68) or (20, 68)
            target_horizon: Forecasting horizon index (0 for T+1)
            num_steps: Number of linear interpolation steps

        Returns:
            Tuple of (feature_attributions_2d, temporal_step_attributions_1d)
        """
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x.float()

        if x_tensor.ndim == 2:
            x_tensor = x_tensor.unsqueeze(0)  # (1, 20, 68)

        x_tensor = x_tensor.to(self.device)
        baseline = torch.zeros_like(x_tensor)

        # Generate interpolated path
        alphas = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device)
        total_grads = torch.zeros_like(x_tensor)

        for alpha in alphas[1:]:
            interpolated = baseline + alpha * (x_tensor - baseline)
            interpolated.requires_grad_(True)

            out = self.model(interpolated)
            if isinstance(out, dict):
                logits = out["logits"]
            else:
                logits = out

            target_logit = logits[0, target_horizon]
            grads = torch.autograd.grad(target_logit, interpolated)[0]
            total_grads += grads

        avg_grads = total_grads / num_steps
        integrated_grad = (x_tensor - baseline) * avg_grads  # (1, 20, 68)
        attr_2d = integrated_grad.squeeze(0).detach().cpu().numpy()  # (20, 68)

        # Temporal profile: L1 norm of attributions per flow step
        temporal_profile = np.sum(np.abs(attr_2d), axis=1)  # (20,)
        return attr_2d, temporal_profile

    def explain_sequence(
        self,
        x: Union[np.ndarray, torch.Tensor],
        target_horizon: int = 0,
        top_n: int = 5,
        num_steps: int = 20,
    ) -> Dict[str, Any]:
        """
        Generate human-readable SOC explanation dictionary.
        """
        attr_2d, temporal_profile = self.attribute(
            x, target_horizon=target_horizon, num_steps=num_steps
        )

        # Aggregate feature importance across the 20 historical timesteps
        feat_importance_raw = np.mean(attr_2d, axis=0)  # (68,) signed
        feat_importance_abs = np.mean(np.abs(attr_2d), axis=0)  # (68,) magnitude

        # Rank features
        ranked_indices = np.argsort(feat_importance_abs)[::-1]
        top_features: List[Dict[str, Any]] = []

        total_mag = float(np.sum(feat_importance_abs)) + 1e-8

        for idx in ranked_indices[:top_n]:
            f_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
            signed_val = float(feat_importance_raw[idx])
            rel_importance = float(feat_importance_abs[idx] / total_mag)
            direction = "increases_attack_risk" if signed_val >= 0 else "decreases_attack_risk"

            top_features.append({
                "feature": f_name,
                "importance": round(rel_importance, 4),
                "raw_attribution": round(signed_val, 6),
                "direction": direction,
            })

        # Normalize temporal profile
        total_temp = float(np.sum(temporal_profile)) + 1e-8
        norm_temporal = [round(float(v / total_temp), 4) for v in temporal_profile]

        return {
            "target_horizon": f"T+{target_horizon + 1}",
            "top_features": top_features,
            "temporal_flow_importance": norm_temporal,
            "most_influential_timestep": int(np.argmax(temporal_profile)),
        }
