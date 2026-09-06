"""
app/backend/replay.py
---------------------
Offline Traffic Replay Simulator Engine (Phase 7).
Sequentially replays authentic CIC-IDS2018 flows, runs World Model inference at step T,
and compares forecasted future attack probabilities with actual future ground truth.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd
from fastapi import WebSocket

from src.inference.pipeline import SOCInferenceService
from src.inference.model_loader import ModelArtifactLoader

logger = logging.getLogger("netforecaster.replay")


class TrafficReplayEngine:
    """
    Offline chronological traffic replay and WebSocket broadcast engine.
    """

    CURATED_SEGMENTS = {
        "infiltration_attack": {
            "name": "Infiltration Attack Progression (Stage 5)",
            "description": "Port scanning, reconnaissance, and post-exploit lateral movement.",
            "start_row": 41945,
            "length": 150,
        },
        "botnet_c2_burst": {
            "name": "Botnet Command & Control Burst (Stage 4)",
            "description": "High-frequency botnet C2 beaconing and orchestration traffic.",
            "start_row": 949980,
            "length": 150,
        },
        "benign_baseline": {
            "name": "Benign Enterprise Traffic Baseline (Stage 0)",
            "description": "Normal enterprise network operations without malicious activity.",
            "start_row": 1500,
            "length": 150,
        },
        "mixed_transition": {
            "name": "Mixed Threat Horizon Transition",
            "description": "Baseline traffic transitioning into multi-stage attack activity.",
            "start_row": 41920,
            "length": 180,
        },
    }

    def __init__(
        self,
        inference_service: Optional[SOCInferenceService] = None,
        data_path: Optional[Path] = None,
    ) -> None:
        self.inference_service = inference_service or SOCInferenceService()
        root_dir = self.inference_service.loader.root_dir
        self.data_path = data_path or (root_dir / "data" / "processed" / "test.parquet")

        self.active_segment_id = "infiltration_attack"
        self.current_step = 20  # needs 20 flows historical window
        self.is_running = False
        self.speed = 1.0  # 1.0x = 1 update per second
        self.connections: Set[WebSocket] = set()

        self._task: Optional[asyncio.Task[None]] = None
        self._df: Optional[pd.DataFrame] = None
        self._features: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._timestamps: Optional[List[str]] = None

        self._load_active_segment()

    def _load_active_segment(self) -> None:
        """Load and prepare features for the currently active segment."""
        meta = self.CURATED_SEGMENTS.get(self.active_segment_id, self.CURATED_SEGMENTS["infiltration_attack"])
        start = meta["start_row"]
        length = meta["length"]

        feature_cols = self.inference_service.feature_names

        if self.data_path.exists():
            try:
                # Read parquet slice
                df = pd.read_parquet(self.data_path)
                segment_df = df.iloc[start : start + length].copy()
                self._df = segment_df
                self._timestamps = (
                    segment_df["Timestamp"].astype(str).tolist()
                    if "Timestamp" in segment_df.columns
                    else [f"2018-02-28 09:15:{i:02d}" for i in range(len(segment_df))]
                )
                self._labels = (
                    segment_df["Label"].to_numpy()
                    if "Label" in segment_df.columns
                    else np.array(["Benign"] * len(segment_df))
                )

                # Extract features
                avail_feats = [c for c in feature_cols if c in segment_df.columns]
                if len(avail_feats) == 68:
                    raw_feats = segment_df[avail_feats].to_numpy(dtype=np.float32)
                else:
                    # Fallback to float cols
                    numeric_cols = segment_df.select_dtypes(include=[np.number]).columns.tolist()[:68]
                    raw_feats = segment_df[numeric_cols].to_numpy(dtype=np.float32)

                # Scale using RobustScaler
                if self.inference_service.scaler is not None:
                    self._features = self.inference_service.scaler.transform(raw_feats).astype(np.float32)
                else:
                    self._features = raw_feats
            except Exception as e:
                logger.warning(f"Error reading parquet segment: {e}. Generating fallback buffer.")
                self._init_fallback_buffer(length)
        else:
            logger.info("Test parquet not found. Generating deterministic demonstration segment.")
            self._init_fallback_buffer(length)

        self.current_step = 20

    def _init_fallback_buffer(self, length: int) -> None:
        """Initialize a fallback synthetic test segment if dataset path is missing."""
        rng = np.random.RandomState(42)
        self._features = rng.randn(length, 68).astype(np.float32)
        self._labels = np.array(["Benign"] * 30 + ["Infilteration"] * (length - 30))
        self._timestamps = [f"2018-02-28 09:15:{i:02d}" for i in range(length)]

    def get_segments(self) -> Dict[str, Any]:
        """Return metadata for all available curated replay segments."""
        return {
            "active_segment": self.active_segment_id,
            "segments": self.CURATED_SEGMENTS,
        }

    def select_segment(self, segment_id: str) -> bool:
        """Switch active replay segment and reset playback index."""
        if segment_id not in self.CURATED_SEGMENTS:
            return False
        self.is_running = False
        self.active_segment_id = segment_id
        self._load_active_segment()
        return True

    def get_status(self) -> Dict[str, Any]:
        """Return current status of the replay simulator."""
        total_steps = len(self._features) if self._features is not None else 0
        return {
            "active_segment": self.active_segment_id,
            "segment_meta": self.CURATED_SEGMENTS.get(self.active_segment_id),
            "current_step": self.current_step,
            "total_steps": total_steps,
            "is_running": self.is_running,
            "speed": self.speed,
            "progress_pct": round((self.current_step / max(total_steps, 1)) * 100, 1),
            "active_connections": len(self.connections),
        }

    def start(self) -> None:
        """Start or resume traffic replay."""
        self.is_running = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_loop())

    def pause(self) -> None:
        """Pause traffic replay."""
        self.is_running = False

    def reset(self) -> None:
        """Reset replay position to the start of the window."""
        self.is_running = False
        self.current_step = 20

    def set_speed(self, speed: float) -> None:
        """Set replay playback speed multiplier."""
        self.speed = max(0.2, min(speed, 10.0))

    def step(self) -> Optional[Dict[str, Any]]:
        """Advance replay by exactly one flow step and compute forecast."""
        if self._features is None or len(self._features) < 25:
            return None

        max_step = len(self._features) - 5
        if self.current_step >= max_step:
            self.current_step = 20  # loop around

        window_feats = self._features[self.current_step - 20 : self.current_step]  # (20, 68)
        current_ts = self._timestamps[self.current_step - 1] if self._timestamps else "2018-02-28 09:15:00"
        current_label = str(self._labels[self.current_step - 1]) if self._labels is not None else "Benign"

        # Ground truth future states at T+1 ... T+5
        actual_future_labels = {}
        actual_future_binary = {}
        for k in range(5):
            idx = self.current_step + k
            lbl = str(self._labels[idx]) if self._labels is not None and idx < len(self._labels) else "Benign"
            is_attack = 0 if lbl.lower() == "benign" else 1
            actual_future_labels[f"T+{k + 1}"] = lbl
            actual_future_binary[f"T+{k + 1}"] = is_attack

        # Run World Model inference
        forecast_result = self.inference_service.forecast_window(
            sequence=window_feats,
            timestamp=current_ts,
            is_scaled=True,
        )

        # Attach ground truth comparison
        payload = {
            "type": "replay_update",
            "segment_id": self.active_segment_id,
            "step": self.current_step,
            "total_steps": len(self._features),
            "progress_pct": round((self.current_step / len(self._features)) * 100, 1),
            "current_flow_label": current_label,
            "forecast": forecast_result,
            "actual_ground_truth": {
                "labels": actual_future_labels,
                "binary": actual_future_binary,
            },
        }

        self.current_step += 1
        return payload

    async def _run_loop(self) -> None:
        """Background async loop broadcasting replay frames over WebSockets."""
        while self.is_running:
            payload = self.step()
            if payload and self.connections:
                # Broadcast to all active websockets
                dead_conns = set()
                for ws in self.connections:
                    try:
                        await ws.send_json(payload)
                    except Exception:
                        dead_conns.add(ws)
                self.connections.difference_update(dead_conns)

            # Delay based on speed (default: 1.0s / speed)
            interval = 1.0 / self.speed
            await asyncio.sleep(interval)

    def register_socket(self, ws: WebSocket) -> None:
        """Register active WebSocket client."""
        self.connections.add(ws)

    def unregister_socket(self, ws: WebSocket) -> None:
        """Unregister closed WebSocket client."""
        self.connections.discard(ws)
