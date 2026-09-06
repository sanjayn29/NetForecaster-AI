"""
training/train_temporal.py
--------------------------
Master PyTorch Training Pipeline for Phase 3: Temporal Sequence Modeling.

Models:
  1. Experiment A: Temporal Convolutional Network (TCN) + Weighted BCE
  2. Experiment B: Transformer Sequence Encoder + Weighted BCE
  3. Experiment C: TCN + Binary Focal Loss
  4. Experiment D: Transformer + Binary Focal Loss

Features:
  - Strict chronological window generation (W=20, K=5)
  - pos_weight computed STRICTLY from training targets (pos_weight = N_negative / N_positive)
  - AdamW optimizer + Plateau scheduler + Gradient clipping (norm=1.0)
  - Validation Macro F1 model selection & Early Stopping
  - Hardware & CPU thread optimization
  - Serialization to models/temporal/
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import CFG
from src.utils.logger import get_logger, setup_root_logger
from src.utils.seed import set_seed
from src.models.tcn_model import TCNForecaster
from src.models.transformer_model import TransformerForecaster
from src.models.loss import (
    compute_pos_weight,
    MultiHorizonWeightedBCEWithLogitsLoss,
    BinaryFocalLossWithLogits,
)

# ── Setup ──────────────────────────────────────────────────────────────────
LOG_FILE = PROJECT_ROOT / "logs" / "temporal_training.log"
(PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
setup_root_logger(level="INFO", log_file=LOG_FILE)
logger = get_logger("train_temporal")

set_seed(CFG.project.seed)

PROCESSED_DIR = PROJECT_ROOT / CFG.data.processed_dir
MODELS_TEMPORAL_DIR = PROJECT_ROOT / "models" / "temporal"
MODELS_TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Detect hardware and configure optimal CPU/GPU threads."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        threads = torch.get_num_threads()
        logger.info("Using CPU with %d PyTorch threads", threads)
    return device


def load_partition_data(
    split_name: str,
    feature_cols: list[str],
    max_rows: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load feature matrix and binary/encoded targets from Parquet partition."""
    path = PROCESSED_DIR / f"{split_name}.parquet"
    logger.info("Loading partition %s from %s...", split_name, path)
    cols_to_load = feature_cols + ["label_binary", "label_encoded"]
    df = pd.read_parquet(path, columns=cols_to_load)

    if max_rows and len(df) > max_rows:
        step = max(1, len(df) // max_rows)
        df = df.iloc[::step].iloc[:max_rows].reset_index(drop=True)

    X = df[feature_cols].values.astype(np.float32)
    y_bin = df["label_binary"].values.astype(int)
    y_multi = df["label_encoded"].values.astype(int)
    logger.info("  %s loaded: %d rows x %d features", split_name, len(X), X.shape[1])
    return X, y_bin, y_multi


def build_window_matrices(
    X: np.ndarray,
    y_bin: np.ndarray,
    y_multi: np.ndarray,
    window_size: int = 20,
    forecast_horizon: int = 5,
    stride: int = 1,
    max_samples: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct temporal window arrays (N, W, D) and future target matrices (N, K)."""
    N_total = len(X)
    total_windows = (N_total - window_size - forecast_horizon) // stride + 1
    if total_windows <= 0:
        raise ValueError("Not enough rows for windowing.")

    if max_samples and total_windows > max_samples:
        indices = np.linspace(0, total_windows - 1, max_samples, dtype=int) * stride
    else:
        indices = np.arange(0, total_windows * stride, stride)

    n_out = len(indices)
    D = X.shape[1]
    K = forecast_horizon

    X_seq = np.empty((n_out, window_size, D), dtype=np.float32)
    y_bin_seq = np.empty((n_out, K), dtype=np.float32)
    y_multi_seq = np.empty((n_out, K), dtype=int)

    for i, idx in enumerate(indices):
        X_seq[i] = X[idx : idx + window_size]
        y_bin_seq[i] = y_bin[idx + window_size : idx + window_size + K]
        y_multi_seq[i] = y_multi[idx + window_size : idx + window_size + K]

    return X_seq, y_bin_seq, y_multi_seq


def fast_evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Fast validation pass for checkpointing and early stopping."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += loss.item() * len(batch_x)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_targets.append(batch_y.int().cpu().numpy())

    n_samples = len(loader.dataset)
    avg_loss = total_loss / max(1, n_samples)
    y_true = np.vstack(all_targets)
    y_pred = np.vstack(all_preds)

    f1_list = [f1_score(y_true[:, k], y_pred[:, k], average="macro", zero_division=0) for k in range(y_true.shape[1])]
    avg_macro_f1 = float(np.mean(f1_list))
    t1_f1 = float(f1_list[0])

    return avg_loss, avg_macro_f1, t1_f1


def train_single_model(
    model_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    save_dir: Path,
    epochs: int = 5,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 2,
) -> Dict[str, Any]:
    """Train a temporal model with AdamW, lr scheduler, and early stopping on Validation Macro F1."""
    logger.info("\n" + "=" * 60)
    logger.info("Training %s -> Save to %s", model_name, save_dir)
    logger.info("=" * 60)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_t1_macro_f1": [],
        "epoch_times": [],
    }

    best_val_macro_f1 = -1.0
    best_epoch = -1
    no_improve_count = 0
    t_start_train = time.time()

    for epoch in range(1, epochs + 1):
        t_epoch_start = time.time()
        model.train()
        total_train_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item() * len(batch_x)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        val_loss, val_macro_f1, val_t1_f1 = fast_evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_macro_f1)

        epoch_time = time.time() - t_epoch_start
        history["train_loss"].append(round(avg_train_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        history["val_macro_f1"].append(round(val_macro_f1, 5))
        history["val_t1_macro_f1"].append(round(val_t1_f1, 5))
        history["epoch_times"].append(round(epoch_time, 2))

        logger.info(
            "  Epoch %2d/%2d | Train Loss: %.4f | Val Loss: %.4f | Val Macro F1: %.4f (T+1: %.4f) | Time: %.1fs",
            epoch, epochs, avg_train_loss, val_loss, val_macro_f1, val_t1_f1, epoch_time,
        )

        # Checkpoint if best validation Macro F1
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            no_improve_count = 0
            model.save(save_dir)
            logger.info("    -> Checkpoint saved (New Best Val Macro F1: %.4f)", best_val_macro_f1)
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                logger.info("  Early stopping triggered after %d epochs without improvement.", no_improve_count)
                break

    total_training_time = round(time.time() - t_start_train, 2)
    logger.info("Finished %s in %.1fs. Best Epoch: %d with Val Macro F1: %.4f",
                model_name, total_training_time, best_epoch, best_val_macro_f1)

    # Save training history
    with open(save_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "total_time_seconds": total_training_time,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_macro_f1,
            "history": history,
        }, f, indent=2)

    return {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "training_time": total_training_time,
        "history": history,
    }


def main() -> None:
    t_start_all = time.time()
    logger.info("=" * 80)
    logger.info("NetForecaster AI — Phase 3: Temporal Sequence Modeling (TCN & Transformer)")
    logger.info("=" * 80)

    device = get_device()

    # Load feature metadata & label encoder
    meta_path = PROJECT_ROOT / "models" / "preprocessing" / "feature_metadata.json"
    with open(meta_path, "r") as f:
        feature_meta = json.load(f)
    feature_cols = feature_meta["feature_cols"]
    W = CFG.windowing.window_size
    K = CFG.windowing.forecast_horizon
    D = len(feature_cols)

    logger.info("Features: %d | Window: %d | Forecast Horizon: %d", D, W, K)

    # Load partitioned datasets
    TRAIN_SAMPLE = 100_000
    VAL_SAMPLE = 25_000
    TEST_SAMPLE = 50_000

    X_tr_raw, y_tr_bin_raw, y_tr_multi_raw = load_partition_data("train", feature_cols, max_rows=350_000)
    X_va_raw, y_va_bin_raw, y_va_multi_raw = load_partition_data("validation", feature_cols, max_rows=100_000)
    X_te_raw, y_te_bin_raw, y_te_multi_raw = load_partition_data("test", feature_cols, max_rows=150_000)

    logger.info("\n--- Constructing Temporal Sequences (W=%d, K=%d) ---", W, K)
    X_seq_tr, y_seq_bin_tr, _ = build_window_matrices(
        X_tr_raw, y_tr_bin_raw, y_tr_multi_raw, W, K, max_samples=TRAIN_SAMPLE
    )
    X_seq_va, y_seq_bin_va, _ = build_window_matrices(
        X_va_raw, y_va_bin_raw, y_va_multi_raw, W, K, max_samples=VAL_SAMPLE
    )
    X_seq_te, y_seq_bin_te, _ = build_window_matrices(
        X_te_raw, y_te_bin_raw, y_te_multi_raw, W, K, max_samples=TEST_SAMPLE
    )

    logger.info("Sequence Shapes: Train=%s, Val=%s, Test=%s", X_seq_tr.shape, X_seq_va.shape, X_seq_te.shape)

    # Calculate class imbalance weights strictly from training targets
    pos_weights_horizon = compute_pos_weight(y_seq_bin_tr, horizon_specific=True)
    pos_weight_common = compute_pos_weight(y_seq_bin_tr, horizon_specific=False)
    logger.info("Training Pos-Weights per horizon: %s (Common: %.4f)",
                pos_weights_horizon.numpy().round(3).tolist(), pos_weight_common.item())

    # Build PyTorch DataLoaders with high batch parallelism
    BATCH_SIZE = 512
    EVAL_BATCH_SIZE = 1024
    train_dataset = TensorDataset(torch.from_numpy(X_seq_tr), torch.from_numpy(y_seq_bin_tr))
    val_dataset = TensorDataset(torch.from_numpy(X_seq_va), torch.from_numpy(y_seq_bin_va))
    test_dataset = TensorDataset(torch.from_numpy(X_seq_te), torch.from_numpy(y_seq_bin_te))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    # Mini-batch shape verification
    sample_x, sample_y = next(iter(train_loader))
    logger.info("DataLoader Check: batch_x=%s, batch_y=%s", sample_x.shape, sample_y.shape)

    # Multi-Horizon Loss Functions
    weighted_bce_loss = MultiHorizonWeightedBCEWithLogitsLoss(pos_weight=pos_weights_horizon.to(device))
    focal_loss = BinaryFocalLossWithLogits(alpha=0.75, gamma=2.0)

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT A: TCN + WEIGHTED BCE
    # ══════════════════════════════════════════════════════════════════
    tcn_model = TCNForecaster(
        num_features=D,
        num_channels=[64, 64, 64, 64],
        kernel_size=3,
        dilations=[1, 2, 4, 8],
        forecast_horizon_k=K,
        num_classes=2,
        dropout=0.1,
    )
    logger.info("TCN Receptive Field: %d steps (Sequence Window: %d)", tcn_model.receptive_field, W)
    tcn_res = train_single_model(
        model_name="TCN_WeightedBCE",
        model=tcn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=weighted_bce_loss,
        device=device,
        save_dir=MODELS_TEMPORAL_DIR / "tcn",
        epochs=5,
        lr=1e-3,
    )

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT B: TRANSFORMER + WEIGHTED BCE
    # ══════════════════════════════════════════════════════════════════
    transformer_model = TransformerForecaster(
        num_features=D,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        forecast_horizon_k=K,
        num_classes=2,
        dropout=0.1,
    )
    trans_res = train_single_model(
        model_name="Transformer_WeightedBCE",
        model=transformer_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=weighted_bce_loss,
        device=device,
        save_dir=MODELS_TEMPORAL_DIR / "transformer",
        epochs=5,
        lr=1e-3,
    )

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT C: TCN + FOCAL LOSS
    # ══════════════════════════════════════════════════════════════════
    tcn_focal_model = TCNForecaster(
        num_features=D,
        num_channels=[64, 64, 64, 64],
        kernel_size=3,
        dilations=[1, 2, 4, 8],
        forecast_horizon_k=K,
        num_classes=2,
        dropout=0.1,
    )
    tcn_focal_res = train_single_model(
        model_name="TCN_FocalLoss",
        model=tcn_focal_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=focal_loss,
        device=device,
        save_dir=MODELS_TEMPORAL_DIR / "tcn_focal",
        epochs=5,
        lr=1e-3,
    )

    # ══════════════════════════════════════════════════════════════════
    # EXPERIMENT D: TRANSFORMER + FOCAL LOSS
    # ══════════════════════════════════════════════════════════════════
    trans_focal_model = TransformerForecaster(
        num_features=D,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        forecast_horizon_k=K,
        num_classes=2,
        dropout=0.1,
    )
    trans_focal_res = train_single_model(
        model_name="Transformer_FocalLoss",
        model=trans_focal_model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=focal_loss,
        device=device,
        save_dir=MODELS_TEMPORAL_DIR / "transformer_focal",
        epochs=5,
        lr=1e-3,
    )

    logger.info("\n" + "=" * 80)
    logger.info("All Temporal Model Trainings Completed in %.1f seconds!", time.time() - t_start_all)
    logger.info("  TCN (Weighted BCE) Best Val Macro F1: %.4f (Time: %.1fs)", tcn_res["best_val_macro_f1"], tcn_res["training_time"])
    logger.info("  Transformer (Weighted BCE) Best Val Macro F1: %.4f (Time: %.1fs)", trans_res["best_val_macro_f1"], trans_res["training_time"])
    logger.info("  TCN (Focal Loss) Best Val Macro F1: %.4f (Time: %.1fs)", tcn_focal_res["best_val_macro_f1"], tcn_focal_res["training_time"])
    logger.info("  Transformer (Focal Loss) Best Val Macro F1: %.4f (Time: %.1fs)", trans_focal_res["best_val_macro_f1"], trans_focal_res["training_time"])
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
