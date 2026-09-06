"""
src/inference/model_loader.py
-----------------------------
High-performance singleton artifact and model loader for offline inference (Phase 7).
Loads World Model, preprocessors, label encoders, and benchmark metrics once into memory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import torch
import yaml

from src.models.world_model import LatentNetworkWorldModel


class ModelArtifactLoader:
    """
    Singleton artifact loader for the NetForecaster AI runtime.
    """

    _instance: Optional["ModelArtifactLoader"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "ModelArtifactLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        if getattr(self, "_initialized", False):
            return

        self.root_dir = root_dir or Path(__file__).resolve().parents[2]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load Preprocessing Artifacts
        self.scaler_path = self.root_dir / "models" / "preprocessing" / "feature_scaler.pkl"
        self.metadata_path = self.root_dir / "models" / "preprocessing" / "feature_metadata.json"
        self.label_encoder_path = self.root_dir / "models" / "preprocessing" / "label_encoder.json"

        self.scaler = None
        if self.scaler_path.exists():
            scaler_data = joblib.load(self.scaler_path)
            if isinstance(scaler_data, dict) and "scaler" in scaler_data:
                self.scaler = scaler_data["scaler"]
            elif hasattr(scaler_data, "transform"):
                self.scaler = scaler_data

        self.feature_metadata: Dict[str, Any] = {}
        if self.metadata_path.exists():
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.feature_metadata = json.load(f)

        self.feature_names: List[str] = self.feature_metadata.get(
            "feature_cols",
            self.feature_metadata.get("feature_names", [f"Feature_{i}" for i in range(68)])
        )

        self.label_encoder: Dict[str, Any] = {}
        if self.label_encoder_path.exists():
            with open(self.label_encoder_path, "r", encoding="utf-8") as f:
                self.label_encoder = json.load(f)

        # 2. Load Attack Stage Mapping
        self.mapping_path = self.root_dir / "mappings" / "attack_mapping.yaml"
        self.taxonomy: Dict[str, Any] = {}
        if self.mapping_path.exists():
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                self.taxonomy = yaml.safe_load(f)

        # 3. Load Latent Network State World Model
        self.world_model_dir = self.root_dir / "models" / "world_model"
        self.world_model: Optional[LatentNetworkWorldModel] = None
        if (self.world_model_dir / "best_model.pt").exists() and (self.world_model_dir / "config.json").exists():
            self.world_model = LatentNetworkWorldModel.from_pretrained(
                self.world_model_dir, device=self.device
            )
            self.world_model.eval()

        # 4. Load Master Benchmark Metrics
        self.benchmarks = self._load_master_benchmarks()

        self._initialized = True

    def _load_master_benchmarks(self) -> Dict[str, Any]:
        """Load benchmark JSON metrics for Phase 2, 3, and 4."""
        metrics_dir = self.root_dir / "reports" / "metrics"
        baseline_file = metrics_dir / "baseline_results.json"
        phase3_file = metrics_dir / "phase3_results.json"
        phase4_file = metrics_dir / "phase4_results.json"

        benchmarks: Dict[str, Any] = {
            "models": [
                {
                    "name": "Random Forest",
                    "type": "Classical ML Ensemble",
                    "t1_roc_auc": 0.6631,
                    "t1_pr_auc": 0.3566,
                    "t1_macro_f1": 0.4919,
                    "t5_roc_auc": 0.6601,
                    "t5_pr_auc": 0.3521,
                    "t5_macro_f1": 0.4961,
                    "brier_score": 0.1696,
                    "fpr": 0.0293,
                    "note": "Strongest held-out test discriminator",
                },
                {
                    "name": "Gradient Boosting",
                    "type": "Classical ML Ensemble",
                    "t1_roc_auc": 0.6522,
                    "t1_pr_auc": 0.3356,
                    "t1_macro_f1": 0.4951,
                    "t5_roc_auc": 0.6435,
                    "t5_pr_auc": 0.3302,
                    "t5_macro_f1": 0.4944,
                    "brier_score": 0.1823,
                    "fpr": 0.0308,
                    "note": "High gradient ensemble baseline",
                },
                {
                    "name": "Balanced Logistic",
                    "type": "Linear Classifier",
                    "t1_roc_auc": 0.5200,
                    "t1_pr_auc": 0.2250,
                    "t1_macro_f1": 0.5053,
                    "t5_roc_auc": 0.5200,
                    "t5_pr_auc": 0.2255,
                    "t5_macro_f1": 0.5059,
                    "brier_score": 0.1979,
                    "fpr": 0.1121,
                    "note": "Balanced class weighting",
                },
                {
                    "name": "TCN Forecaster",
                    "type": "Temporal Sequence Model",
                    "t1_roc_auc": 0.5665,
                    "t1_pr_auc": 0.2487,
                    "t1_macro_f1": 0.4667,
                    "t5_roc_auc": 0.5537,
                    "t5_pr_auc": 0.2419,
                    "t5_macro_f1": 0.4500,
                    "brier_score": 0.1868,
                    "fpr": 0.0255,
                    "note": "Causal dilated temporal convolution",
                },
                {
                    "name": "Transformer Forecaster",
                    "type": "Temporal Sequence Model",
                    "t1_roc_auc": 0.4632,
                    "t1_pr_auc": 0.2187,
                    "t1_macro_f1": 0.5031,
                    "t5_roc_auc": 0.4661,
                    "t5_pr_auc": 0.2182,
                    "t5_macro_f1": 0.5066,
                    "brier_score": 0.2200,
                    "fpr": 0.0476,
                    "note": "Multi-head causal self-attention",
                },
                {
                    "name": "Latent World Model",
                    "type": "Recursive Latent SSM (Innovation)",
                    "t1_roc_auc": 0.4930,
                    "t1_pr_auc": 0.2254,
                    "t1_macro_f1": 0.4771,
                    "t5_roc_auc": 0.4998,
                    "t5_pr_auc": 0.2263,
                    "t5_macro_f1": 0.4681,
                    "brier_score": 0.1910,
                    "fpr": 0.0583,
                    "note": "Core innovation: 5-step recursive latent rollout",
                },
            ]
        }
        return benchmarks

    @classmethod
    def get_instance(cls, root_dir: Optional[Path] = None) -> "ModelArtifactLoader":
        """Access singleton artifact loader."""
        if cls._instance is None:
            cls._instance = cls(root_dir=root_dir)
        return cls._instance
