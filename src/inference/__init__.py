"""
src/inference
-------------
Inference service and model artifact loaders (Phase 7).
"""

from src.inference.model_loader import ModelArtifactLoader
from src.inference.pipeline import SOCInferenceService

__all__ = ["ModelArtifactLoader", "SOCInferenceService"]
