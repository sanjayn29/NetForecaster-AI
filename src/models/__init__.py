"""
src/models package
------------------
Baseline and sequence modeling architectures for NetForecaster AI.
"""

from src.models.logistic_baseline import LogisticRegressionBaseline
from src.models.tree_baselines import RandomForestBaseline, GradientBoostingBaseline
from src.models.lstm_model import LSTMForecaster
from src.models.gru_model import GRUForecaster
from src.models.tcn_model import TCNForecaster, TemporalConvNet, TemporalBlock
from src.models.transformer_model import TransformerForecaster, PositionalEncoding
from src.models.state_transition import ResidualStateTransition
from src.models.world_model import LatentNetworkWorldModel
from src.models.loss import (
    compute_pos_weight,
    MultiHorizonWeightedBCEWithLogitsLoss,
    BinaryFocalLossWithLogits,
)

__all__ = [
    "LogisticRegressionBaseline",
    "RandomForestBaseline",
    "GradientBoostingBaseline",
    "LSTMForecaster",
    "GRUForecaster",
    "TCNForecaster",
    "TemporalConvNet",
    "TemporalBlock",
    "TransformerForecaster",
    "PositionalEncoding",
    "ResidualStateTransition",
    "LatentNetworkWorldModel",
    "compute_pos_weight",
    "MultiHorizonWeightedBCEWithLogitsLoss",
    "BinaryFocalLossWithLogits",
]

