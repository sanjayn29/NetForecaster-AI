"""
src/forecasting/world_rollout.py
--------------------------------
Rollout Engine and Diagnostic Metrics for Latent Network State World Model (Phase 4).

Computes:
  1. Recursive latent state drift: ||S_{t+k} - S_t||_2 for k=1..5
  2. Latent state norms: ||S_{t+k}||_2 for k=1..5
  3. Step-by-step feature reconstruction error
  4. Average and per-sample predicted attack trajectories
  5. 2D PCA projection of latent network states
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn

from src.models.world_model import LatentNetworkWorldModel


class WorldModelRolloutEngine:
    """
    Inference, rollout diagnosis, and trajectory analysis engine for LatentNetworkWorldModel.
    """

    def __init__(self, model: LatentNetworkWorldModel, device: Optional[torch.device] = None) -> None:
        self.model = model
        self.device = device or torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate_rollout(
        self,
        x: Union[np.ndarray, torch.Tensor],
        y_attack: Optional[Union[np.ndarray, torch.Tensor]] = None,
        x_future_target: Optional[Union[np.ndarray, torch.Tensor]] = None,
        batch_size: int = 512,
    ) -> Dict[str, Any]:
        """
        Evaluate full recursive rollout over input sequences.

        Args:
            x: Input sequences (N, 20, 68)
            y_attack: Optional ground-truth attack targets (N, 5)
            x_future_target: Optional ground-truth future normalized flows (N, 5, 68)
            batch_size: Batch size for batched evaluation

        Returns:
            Dictionary containing rollout analysis, drift, norms, trajectories, and PCA.
        """
        if isinstance(x, np.ndarray):
            x_tensor = torch.from_numpy(x).float()
        else:
            x_tensor = x.float()

        n_samples = len(x_tensor)
        k_steps = self.model.forecast_horizon_k

        all_logits: List[torch.Tensor] = []
        all_initial_states: List[torch.Tensor] = []
        all_states: List[torch.Tensor] = []
        all_recons: List[torch.Tensor] = []

        for i in range(0, n_samples, batch_size):
            batch_x = x_tensor[i : i + batch_size].to(self.device)
            out = self.model(batch_x)
            all_logits.append(out["logits"].cpu())
            all_initial_states.append(out["initial_state"].cpu())
            all_states.append(out["states"].cpu())
            all_recons.append(out["reconstructions"].cpu())

        logits = torch.cat(all_logits, dim=0).numpy()  # (N, K)
        probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid probabilities (N, K)
        initial_states = torch.cat(all_initial_states, dim=0).numpy()  # (N, latent_dim)
        states = torch.cat(all_states, dim=0).numpy()  # (N, K, latent_dim)
        reconstructions = torch.cat(all_recons, dim=0).numpy()  # (N, K, num_features)

        # 1. Latent State Norms across rollout steps
        # S_0 norm
        initial_norm = np.mean(np.linalg.norm(initial_states, axis=-1))
        step_norms = [float(np.mean(np.linalg.norm(states[:, k, :], axis=-1))) for k in range(k_steps)]

        # 2. Latent State Drift ||S_{t+k} - S_t||
        step_drifts = [
            float(np.mean(np.linalg.norm(states[:, k, :] - initial_states, axis=-1)))
            for k in range(k_steps)
        ]

        # 3. Predicted Attack Probability Trajectory
        mean_pred_probs = [float(np.mean(probs[:, k])) for k in range(k_steps)]
        actual_pos_prevalences = None
        if y_attack is not None:
            y_arr = np.asarray(y_attack)
            actual_pos_prevalences = [float(np.mean(y_arr[:, k])) for k in range(k_steps)]

        # 4. Feature Reconstruction Error (Smooth L1 approximation: MAE / MSE)
        step_recon_mae = None
        step_recon_mse = None
        if x_future_target is not None:
            x_fut = np.asarray(x_future_target)
            step_recon_mae = [
                float(np.mean(np.abs(reconstructions[:, k, :] - x_fut[:, k, :])))
                for k in range(k_steps)
            ]
            step_recon_mse = [
                float(np.mean((reconstructions[:, k, :] - x_fut[:, k, :]) ** 2))
                for k in range(k_steps)
            ]

        # 5. 2D PCA Projection of Initial Latent States
        pca_coords = None
        pca_explained_var = None
        if n_samples >= 10:
            pca = PCA(n_components=2, random_state=42)
            sub_n = min(n_samples, 5000)
            pca_res = pca.fit_transform(initial_states[:sub_n])
            pca_coords = pca_res.tolist()
            pca_explained_var = [float(v) for v in pca.explained_variance_ratio_]

        # 6. Sample Trajectories (Benign stable vs Attack escalating)
        sample_trajectories = {}
        if y_attack is not None:
            y_arr = np.asarray(y_attack)
            # Find an all-benign sequence target
            benign_idx = np.where(np.all(y_arr == 0, axis=1))[0]
            if len(benign_idx) > 0:
                sample_trajectories["benign_sample"] = {
                    "true_labels": y_arr[benign_idx[0]].tolist(),
                    "predicted_probs": probs[benign_idx[0]].tolist(),
                }
            # Find an attack-escalating sequence target
            attack_idx = np.where(y_arr[:, -1] == 1)[0]
            if len(attack_idx) > 0:
                sample_trajectories["attack_sample"] = {
                    "true_labels": y_arr[attack_idx[0]].tolist(),
                    "predicted_probs": probs[attack_idx[0]].tolist(),
                }

        return {
            "num_samples": n_samples,
            "forecast_horizon_k": k_steps,
            "latent_dim": self.model.latent_dim,
            "initial_state_norm": float(initial_norm),
            "latent_state_norms": step_norms,
            "latent_state_drifts": step_drifts,
            "mean_predicted_probabilities": mean_pred_probs,
            "actual_positive_prevalences": actual_pos_prevalences,
            "reconstruction_mae": step_recon_mae,
            "reconstruction_mse": step_recon_mse,
            "pca_explained_variance_ratio": pca_explained_var,
            "sample_trajectories": sample_trajectories,
            "probabilities": probs,
        }
