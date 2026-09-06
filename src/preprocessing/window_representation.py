r"""
src/preprocessing/window_representation.py
-----------------------------------------
Deterministic statistical window aggregation for tabular baseline models.

Given an input temporal window sequence of shape (W, D) = (20, 68),
creates a deterministic summary representation of shape (3 * D,) = (204,):
  - Last timestep X[t]          (68 features) -> captures instantaneous state
  - Window Mean \mu             (68 features) -> captures traffic rate / volume intensity
  - Window Std \sigma           (68 features) -> captures traffic burstiness & volatility

This avoids naive raw flattening (1360 features) which suffers from collinearity,
overfitting, and lack of temporal aggregation.
"""

from __future__ import annotations

import numpy as np


def aggregate_window(
    window: np.ndarray,
    include_last: bool = True,
    include_mean: bool = True,
    include_std: bool = True,
    include_min: bool = False,
    include_max: bool = False,
) -> np.ndarray:
    """Aggregate a single window (W, D) into a 1D feature vector.

    Parameters
    ----------
    window : np.ndarray
        Array of shape (W, D) where W is window size and D is feature dimension.
    include_last : bool, default=True
        Include the last timestep X[-1].
    include_mean : bool, default=True
        Include the mean across the window along time axis.
    include_std : bool, default=True
        Include the standard deviation across the window along time axis.
    include_min : bool, default=False
        Include min across window.
    include_max : bool, default=False
        Include max across window.

    Returns
    -------
    np.ndarray
        1D feature vector of shape (k * D,).
    """
    parts: list[np.ndarray] = []

    if include_last:
        parts.append(window[-1])
    if include_mean:
        parts.append(np.mean(window, axis=0))
    if include_std:
        parts.append(np.std(window, axis=0))
    if include_min:
        parts.append(np.min(window, axis=0))
    if include_max:
        parts.append(np.max(window, axis=0))

    if not parts:
        raise ValueError("At least one aggregation component must be selected.")

    return np.concatenate(parts, axis=0)


def aggregate_windows_batch(
    windows: np.ndarray,
    include_last: bool = True,
    include_mean: bool = True,
    include_std: bool = True,
    include_min: bool = False,
    include_max: bool = False,
) -> np.ndarray:
    """Aggregate a batch of windows (N, W, D) into (N, k * D).

    Parameters
    ----------
    windows : np.ndarray
        Array of shape (N, W, D).

    Returns
    -------
    np.ndarray
        Array of shape (N, k * D).
    """
    parts: list[np.ndarray] = []

    if include_last:
        parts.append(windows[:, -1, :])
    if include_mean:
        parts.append(np.mean(windows, axis=1))
    if include_std:
        parts.append(np.std(windows, axis=1))
    if include_min:
        parts.append(np.min(windows, axis=1))
    if include_max:
        parts.append(np.max(windows, axis=1))

    if not parts:
        raise ValueError("At least one aggregation component must be selected.")

    return np.concatenate(parts, axis=1)


def get_aggregated_feature_names(
    base_feature_names: list[str],
    include_last: bool = True,
    include_mean: bool = True,
    include_std: bool = True,
    include_min: bool = False,
    include_max: bool = False,
) -> list[str]:
    """Generate human-readable names for the aggregated features."""
    names: list[str] = []
    if include_last:
        names.extend([f"{f}_last" for f in base_feature_names])
    if include_mean:
        names.extend([f"{f}_mean" for f in base_feature_names])
    if include_std:
        names.extend([f"{f}_std" for f in base_feature_names])
    if include_min:
        names.extend([f"{f}_min" for f in base_feature_names])
    if include_max:
        names.extend([f"{f}_max" for f in base_feature_names])
    return names
