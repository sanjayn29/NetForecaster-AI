"""
training/train_world_model.py
-----------------------------
Master PyTorch Training Pipeline for Phase 4: Latent Network State World Model.

Experiments:
  1. Experiment A: Latent Network World Model (TCN Backbone + Transition MLP + Weighted BCE)
  2. Experiment B: Latent Network World Model (TCN Backbone + Transition MLP + Weighted BCE + State Reconstruction)

Features:
  - Multi-horizon recursive rollout (K=5) without future teacher-forcing leakage.
  - Multi-task loss: L_attack + lambda_state * L_state + lambda_consistency * L_consistency
  - Training-only pos_weight computation (pos_weight = N_neg / N_pos)
  - AdamW optimizer + Plateau scheduler + Gradient clipping (norm=1.0)
  - Validation Macro F1 model selection & Early Stopping (patience=2)
  - Serialization to models/world_model/
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import CFG
from src.utils.logger import get_logger, setup_root_logger
from src.utils.seed import set_seed
from src.models.world_model import LatentNetworkWorldModel
from src.models.loss import compute_pos_weight, MultiHorizonWeightedBCEWithLogitsLoss

# ── Setup ──────────────────────────────────────────────────────────────────
LOG_FILE = PROJECT_ROOT / "logs" / "world_model_training.log"
(PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
setup_root_logger(level="INFO", log_file=LOG_FILE)
logger = get_logger("train_world_model")

set_seed(CFG.project.seed)

PROCESSED_DIR = PROJECT_ROOT / CFG.data.processed_dir
MODELS_WM_DIR = PROJECT_ROOT / "models" / "world_model"
MODELS_WM_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    """Detect hardware and configure optimal PyTorch compute device."""
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
    feature_cols: List[str],
    max_rows: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load feature matrix and binary targets from Parquet partition."""
    path = PROCESSED_DIR / f"{split_name}.parquet"
    logger.info("Loading partition %s from %s...", split_name, path)
    cols_to_load = feature_cols + ["label_binary"]
    df = pd.read_parquet(path, columns=cols_to_load)

    if max_rows and len(df) > max_rows:
        step = max(1, len(df) // max_rows)
        df = df.iloc[::step].iloc[:max_rows].reset_index(drop=True)

    X = df[feature_cols].values.astype(np.float32)
    y_bin = df["label_binary"].values.astype(np.float32)
    logger.info("  %s loaded: %d rows x %d features", split_name, len(X), X.shape[1])
    return X, y_bin


def build_world_model_windows(
    X: np.ndarray,
    y_bin: np.ndarray,
    window_size: int = 20,
    forecast_horizon: int = 5,
    stride: int = 1,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct historical sequences (N, W, D), future attack targets (N, K),
    and future feature targets (N, K, D) for state reconstruction loss.
    """
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
    X_fut_seq = np.empty((n_out, K, D), dtype=np.float32)

    for i, idx in enumerate(indices):
        X_seq[i] = X[idx : idx + window_size]
        y_bin_seq[i] = y_bin[idx + window_size : idx + window_size + K]
        X_fut_seq[i] = X[idx + window_size : idx + window_size + K]

    return X_seq, y_bin_seq, X_fut_seq


def fast_evaluate_world_model(
    model: LatentNetworkWorldModel,
    loader: DataLoader,
    attack_criterion: nn.Module,
    lambda_state: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Fast validation pass for checkpointing and early stopping."""
    model.eval()
    total_loss = 0.0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            bx = batch[0].to(device)
            by = batch[1].to(device)
            b_fut = batch[2].to(device)

            out = model(bx)
            loss_attack = attack_criterion(out["logits"], by)

            loss_recon = torch.tensor(0.0, device=device)
            if lambda_state > 0.0:
                loss_recon = F.smooth_l1_loss(out["reconstructions"], b_fut)

            loss = loss_attack + lambda_state * loss_recon
            total_loss += loss.item() * len(bx)

            probs = torch.sigmoid(out["logits"]).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            all_preds.append(preds)
            all_targets.append(by.cpu().numpy().astype(int))

    n = len(loader.dataset)
    avg_loss = total_loss / n
    preds_cat = np.concatenate(all_preds, axis=0)
    targets_cat = np.concatenate(all_targets, axis=0)

    # Average Macro F1 across all K horizons
    k_f1s = [
        f1_score(targets_cat[:, k], preds_cat[:, k], average="macro", zero_division=0)
        for k in range(targets_cat.shape[1])
    ]
    macro_f1 = float(np.mean(k_f1s))
    t1_f1 = float(k_f1s[0])

    return avg_loss, macro_f1, t1_f1


def train_world_model_experiment(
    exp_name: str,
    model: LatentNetworkWorldModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weight: float,
    lambda_state: float = 0.0,
    lambda_consistency: float = 0.01,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    epochs: int = 10,
    patience: int = 2,
    device: torch.device = torch.device("cpu"),
) -> Tuple[LatentNetworkWorldModel, Dict[str, Any]]:
    """Train World Model with recursive latent rollout and multi-task loss."""
    logger.info("=" * 70)
    logger.info("Starting Training: %s", exp_name)
    logger.info("  lambda_attack=1.0, lambda_state=%.2f, lambda_consistency=%.3f", lambda_state, lambda_consistency)
    pw_str = str(np.round(pos_weight.cpu().numpy(), 4)) if isinstance(pos_weight, torch.Tensor) else f"{pos_weight:.4f}"
    logger.info("  pos_weight: %s | lr: %.1e | epochs: %d | patience: %d", pw_str, learning_rate, epochs, patience)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )
    attack_criterion = MultiHorizonWeightedBCEWithLogitsLoss(pos_weight=pos_weight)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_t1_f1": [],
        "epoch_times": [],
    }

    best_val_macro_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    save_dir = MODELS_WM_DIR / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    start_total = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            bx = batch[0].to(device)
            by = batch[1].to(device)
            b_fut = batch[2].to(device)

            optimizer.zero_grad()
            out = model(bx)

            # 1. Attack Loss (Weighted BCE)
            l_attack = attack_criterion(out["logits"], by)

            # 2. State Reconstruction Loss (Smooth L1)
            l_state = torch.tensor(0.0, device=device)
            if lambda_state > 0.0:
                l_state = F.smooth_l1_loss(out["reconstructions"], b_fut)

            # 3. State Regularization / Smoothness Loss
            # Penalize sudden explosive drift in latent state transitions
            states = out["states"]  # (B, K, latent_dim)
            s_0 = out["initial_state"].unsqueeze(1)  # (B, 1, latent_dim)
            all_s = torch.cat([s_0, states], dim=1)  # (B, K+1, latent_dim)
            step_diffs = all_s[:, 1:, :] - all_s[:, :-1, :]
            l_consistency = torch.mean(torch.norm(step_diffs, dim=-1))

            loss = l_attack + lambda_state * l_state + lambda_consistency * l_consistency

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(bx)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss, val_macro_f1, val_t1_f1 = fast_evaluate_world_model(
            model, val_loader, attack_criterion, lambda_state, device
        )
        scheduler.step(val_macro_f1)
        epoch_time = time.time() - epoch_start

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_macro_f1"].append(float(val_macro_f1))
        history["val_t1_f1"].append(float(val_t1_f1))
        history["epoch_times"].append(float(epoch_time))

        logger.info(
            "  Epoch [%2d/%2d] | Train Loss: %.4f | Val Loss: %.4f | Val Macro F1: %.4f | Val T+1 F1: %.4f | (%.1fs)",
            epoch,
            epochs,
            train_loss,
            val_loss,
            val_macro_f1,
            val_t1_f1,
            epoch_time,
        )

        # Checkpointing
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            patience_counter = 0
            model.save_pretrained(save_dir)
            logger.info("    -> Checkpoint saved (Val Macro F1: %.4f)", val_macro_f1)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("    Early stopping triggered after %d epochs without improvement.", patience)
                break

    total_time = time.time() - start_total
    logger.info(
        "Finished %s in %.1fs | Best Epoch: %d | Best Val Macro F1: %.4f",
        exp_name,
        total_time,
        best_epoch,
        best_val_macro_f1,
    )

    # Save training history
    history["total_training_time_seconds"] = float(total_time)
    history["best_epoch"] = int(best_epoch)
    history["best_val_macro_f1"] = float(best_val_macro_f1)
    with open(save_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Load best model
    best_model = LatentNetworkWorldModel.from_pretrained(save_dir, device=device)
    return best_model, history


def main() -> None:
    """Master Phase 4 Training Pipeline Execution."""
    logger.info("=" * 80)
    logger.info("PHASE 4: LATENT NETWORK STATE WORLD MODEL TRAINING")
    logger.info("=" * 80)

    device = get_device()

    # Load feature metadata
    meta_path = PROJECT_ROOT / "models" / "preprocessing" / "feature_metadata.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta.get("feature_cols") or meta.get("feature_names")
    num_features = len(feature_cols)
    logger.info("Dataset feature count: %d", num_features)

    # 1. Load Data Partitions
    # Train: 100k windows, Val: 20k windows
    X_train_raw, y_train_raw = load_partition_data("train", feature_cols, max_rows=150_000)
    X_val_raw, y_val_raw = load_partition_data("validation", feature_cols, max_rows=50_000)

    # 2. Build Windows
    logger.info("Constructing chronological sequences (W=20, K=5)...")
    X_train_seq, y_train_seq, X_train_fut = build_world_model_windows(
        X_train_raw, y_train_raw, window_size=20, forecast_horizon=5, stride=1, max_samples=100_000
    )
    X_val_seq, y_val_seq, X_val_fut = build_world_model_windows(
        X_val_raw, y_val_raw, window_size=20, forecast_horizon=5, stride=1, max_samples=20_000
    )

    logger.info("  Train shapes: X=%s, y=%s, X_fut=%s", X_train_seq.shape, y_train_seq.shape, X_train_fut.shape)
    logger.info("  Val shapes:   X=%s, y=%s, X_fut=%s", X_val_seq.shape, y_val_seq.shape, X_val_fut.shape)

    # Compute training pos_weight
    pos_weight = compute_pos_weight(y_train_seq)
    pw_display = str(np.round(pos_weight.cpu().numpy(), 4)) if isinstance(pos_weight, torch.Tensor) else f"{pos_weight:.4f}"
    logger.info("Computed training pos_weight = %s", pw_display)

    # DataLoaders
    batch_size = 256
    train_dataset = TensorDataset(
        torch.from_numpy(X_train_seq),
        torch.from_numpy(y_train_seq),
        torch.from_numpy(X_train_fut),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(X_val_seq),
        torch.from_numpy(y_val_seq),
        torch.from_numpy(X_val_fut),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ── Experiment A: Latent Network World Model (Transition + Weighted BCE) ────
    model_a = LatentNetworkWorldModel(
        num_features=num_features,
        latent_dim=64,
        encoder_channels=[64, 64, 64, 64],
        encoder_dilations=[1, 2, 4, 8],
        kernel_size=3,
        transition_hidden_dim=128,
        forecast_horizon_k=5,
        dropout=0.1,
        enable_reconstruction=False,
    )
    best_model_a, hist_a = train_world_model_experiment(
        exp_name="world_model_transition",
        model=model_a,
        train_loader=train_loader,
        val_loader=val_loader,
        pos_weight=pos_weight,
        lambda_state=0.0,
        lambda_consistency=0.01,
        learning_rate=1e-3,
        epochs=10,
        patience=2,
        device=device,
    )

    # ── Experiment B: Latent Network World Model + Reconstruction ──────────────
    model_b = LatentNetworkWorldModel(
        num_features=num_features,
        latent_dim=64,
        encoder_channels=[64, 64, 64, 64],
        encoder_dilations=[1, 2, 4, 8],
        kernel_size=3,
        transition_hidden_dim=128,
        forecast_horizon_k=5,
        dropout=0.1,
        enable_reconstruction=True,
    )
    best_model_b, hist_b = train_world_model_experiment(
        exp_name="world_model_reconstruction",
        model=model_b,
        train_loader=train_loader,
        val_loader=val_loader,
        pos_weight=pos_weight,
        lambda_state=0.1,
        lambda_consistency=0.01,
        learning_rate=1e-3,
        epochs=10,
        patience=2,
        device=device,
    )

    # Select Primary Best Model (based on Validation Macro F1)
    primary_exp = "world_model_transition" if hist_a["best_val_macro_f1"] >= hist_b["best_val_macro_f1"] else "world_model_reconstruction"
    logger.info("=" * 80)
    logger.info("Selected Primary Phase 4 Model: %s", primary_exp)

    src_dir = MODELS_WM_DIR / primary_exp
    for item in ["best_model.pt", "config.json", "training_history.json"]:
        shutil.copy2(src_dir / item, MODELS_WM_DIR / item)
    logger.info("Copied best model files to %s", MODELS_WM_DIR)
    logger.info("Phase 4 Training Complete!")


if __name__ == "__main__":
    main()
