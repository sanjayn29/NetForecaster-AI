"""
src/data/label_encoder.py
--------------------------
Label encoding for NetForecaster AI.

Encodes the actual CIC-IDS2018 dataset labels into integer indices.

IMPORTANT: This encoder preserves the ORIGINAL dataset labels.
MITRE ATT&CK stage mapping is handled separately (Phase 11).

Labels (from Phase 0 inspection)
----------------------------------
 0  Benign
 1  Bot
 2  Brute Force -Web
 3  Brute Force -XSS
 4  DDoS attacks-LOIC-HTTP
 5  DDOS attack-HOIC
 6  DDOS attack-LOIC-UDP
 7  DoS attacks-GoldenEye
 8  DoS attacks-Hulk
 9  DoS attacks-SlowHTTPTest
10  DoS attacks-Slowloris
11  FTP-BruteForce
12  Infilteration
13  SQL Injection
14  SSH-Bruteforce
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Canonical label order (alphabetical, with Benign=0 as the first class)
# Benign is placed first intentionally — most forecasting metrics are
# defined relative to the "normal" class.
CANONICAL_LABELS = [
    "Benign",
    "Bot",
    "Brute Force -Web",
    "Brute Force -XSS",
    "DDoS attacks-LOIC-HTTP",
    "DDOS attack-HOIC",
    "DDOS attack-LOIC-UDP",
    "DoS attacks-GoldenEye",
    "DoS attacks-Hulk",
    "DoS attacks-SlowHTTPTest",
    "DoS attacks-Slowloris",
    "FTP-BruteForce",
    "Infilteration",
    "SQL Injection",
    "SSH-Bruteforce",
]

# Binary mapping: 0 = Benign, 1 = Attack (any attack type)
BINARY_MAP = {lbl: (0 if lbl == "Benign" else 1) for lbl in CANONICAL_LABELS}


class CICLabelEncoder:
    """Label encoder for CIC-IDS2018 dataset.

    Provides:
    - Multi-class encoding (0–14)
    - Binary encoding (0=Benign, 1=Attack)
    - Inverse transform
    - Serialization to/from JSON

    Attributes
    ----------
    label_to_int : dict[str, int]
        Mapping from label string to integer index.
    int_to_label : dict[int, str]
        Inverse mapping.
    n_classes : int
        Total number of classes.
    classes_ : list[str]
        Ordered list of class names.
    """

    def __init__(self, labels: Optional[list[str]] = None) -> None:
        if labels is None:
            labels = CANONICAL_LABELS
        self.classes_: list[str] = list(labels)
        self.label_to_int: dict[str, int] = {lbl: i for i, lbl in enumerate(self.classes_)}
        self.int_to_label: dict[int, str] = {i: lbl for lbl, i in self.label_to_int.items()}
        self.n_classes: int = len(self.classes_)

    def fit(self, labels: pd.Series | list[str]) -> "CICLabelEncoder":
        """Fit from actual label series — extends canonical labels with any unseen ones.

        Parameters
        ----------
        labels : pd.Series or list
            The Label column from the dataset.

        Returns
        -------
        self
        """
        seen = set(str(l) for l in labels if pd.notna(l) and str(l) != "Label")
        known = set(self.classes_)
        new_labels = sorted(seen - known)
        if new_labels:
            logger.warning(
                "[LabelEncoder] Found %d unseen label(s): %s. Adding to end of encoder.",
                len(new_labels),
                new_labels,
            )
            for lbl in new_labels:
                idx = len(self.classes_)
                self.classes_.append(lbl)
                self.label_to_int[lbl] = idx
                self.int_to_label[idx] = lbl
            self.n_classes = len(self.classes_)
        return self

    def transform(self, labels: pd.Series | list[str]) -> np.ndarray:
        """Encode labels to integer indices.

        Unknown labels are encoded as -1 and logged.

        Parameters
        ----------
        labels : pd.Series or list
            Labels to encode.

        Returns
        -------
        np.ndarray of int64, shape (n,)
        """
        encoded = np.array(
            [self.label_to_int.get(str(l), -1) for l in labels],
            dtype=np.int64,
        )
        n_unknown = int((encoded == -1).sum())
        if n_unknown > 0:
            logger.warning("[LabelEncoder] %d unknown label(s) encoded as -1.", n_unknown)
        return encoded

    def fit_transform(self, labels: pd.Series | list[str]) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(labels)
        return self.transform(labels)

    def inverse_transform(self, indices: np.ndarray | list[int]) -> list[str]:
        """Decode integer indices back to label strings.

        Parameters
        ----------
        indices : array-like of int

        Returns
        -------
        list[str]
        """
        return [self.int_to_label.get(int(i), f"UNKNOWN({i})") for i in indices]

    def to_binary(self, labels: pd.Series | list[str]) -> np.ndarray:
        """Encode to binary attack indicator (0=Benign, 1=Attack).

        Parameters
        ----------
        labels : pd.Series or list

        Returns
        -------
        np.ndarray of int64, shape (n,)
        """
        return np.array([BINARY_MAP.get(str(l), 1) for l in labels], dtype=np.int64)

    def save(self, path: str | Path) -> None:
        """Save encoder mapping to JSON.

        Parameters
        ----------
        path : str or Path
            Output file path (e.g., ``models/preprocessing/label_encoder.json``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "classes": self.classes_,
            "label_to_int": self.label_to_int,
            "n_classes": self.n_classes,
            "binary_map": BINARY_MAP,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("[LabelEncoder] Saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "CICLabelEncoder":
        """Load encoder from a saved JSON file.

        Parameters
        ----------
        path : str or Path

        Returns
        -------
        CICLabelEncoder
        """
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        encoder = cls(labels=payload["classes"])
        encoder.label_to_int = payload["label_to_int"]
        encoder.int_to_label = {int(k): v for k, v in payload.get("int_to_label", {}).items()}
        if not encoder.int_to_label:
            encoder.int_to_label = {v: k for k, v in encoder.label_to_int.items()}
        encoder.n_classes = payload["n_classes"]
        logger.info("[LabelEncoder] Loaded from %s (%d classes)", path, encoder.n_classes)
        return encoder

    def summary(self) -> str:
        lines = [f"CICLabelEncoder — {self.n_classes} classes:"]
        for i, lbl in enumerate(self.classes_):
            binary = BINARY_MAP.get(lbl, 1)
            lines.append(f"  {i:2d}  {lbl:<40}  (binary={binary})")
        return "\n".join(lines)
