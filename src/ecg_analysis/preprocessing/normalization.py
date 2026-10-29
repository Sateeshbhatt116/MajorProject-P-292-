"""Amplitude normalization routines for ECG signals and segmented heartbeats."""
from __future__ import annotations

import numpy as np

from src.ecg_analysis.utils.exceptions import PreprocessingError


def zscore_normalize(
    signal: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Standardize signal to zero mean and unit variance (Z-score).

    Args:
        signal: 1D or 2D array of ECG amplitudes.
        eps: Small constant to prevent division by zero for flat segments.

    Returns:
        Normalized signal with mean ≈ 0 and std ≈ 1.
    """
    if signal is None or len(signal) == 0:
        raise PreprocessingError("Signal array is empty or None")

    mean = np.mean(signal, axis=0, keepdims=True)
    std = np.std(signal, axis=0, keepdims=True)
    std = np.where(std < eps, eps, std)
    return ((signal - mean) / std).astype(np.float64)


def minmax_normalize(
    signal: np.ndarray,
    feature_range: tuple[float, float] = (-1.0, 1.0),
    eps: float = 1e-8,
) -> np.ndarray:
    """Scale signal amplitudes into a specified interval [a, b].

    Args:
        signal: 1D or 2D array of ECG amplitudes.
        feature_range: Tuple (min, max) desired range (default: (-1.0, 1.0)).
        eps: Small constant to prevent division by zero.

    Returns:
        Rescaled signal array within target range.
    """
    if signal is None or len(signal) == 0:
        raise PreprocessingError("Signal array is empty or None")

    a, b = feature_range
    s_min = np.min(signal, axis=0, keepdims=True)
    s_max = np.max(signal, axis=0, keepdims=True)
    denom = s_max - s_min
    denom = np.where(denom < eps, eps, denom)

    scaled_0_1 = (signal - s_min) / denom
    return (scaled_0_1 * (b - a) + a).astype(np.float64)


def robust_normalize(
    signal: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize signal using median and Interquartile Range (IQR).

    Highly resistant to sudden electrode disconnect spikes and pacing spikes.

    Args:
        signal: 1D or 2D array of ECG amplitudes.
        eps: Small constant to avoid zero division.

    Returns:
        Robustly centered and scaled signal.
    """
    if signal is None or len(signal) == 0:
        raise PreprocessingError("Signal array is empty or None")

    median = np.median(signal, axis=0, keepdims=True)
    q75, q25 = np.percentile(signal, [75, 25], axis=0, keepdims=True)
    iqr = q75 - q25
    iqr = np.where(iqr < eps, eps, iqr)
    return ((signal - median) / iqr).astype(np.float64)


def normalize_signal(
    signal: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    """Convenience dispatcher for signal normalization."""
    if method == "zscore":
        return zscore_normalize(signal)
    elif method == "minmax":
        return minmax_normalize(signal)
    elif method == "robust":
        return robust_normalize(signal)
    else:
        raise ValueError(f"Unknown normalization method '{method}'. Choose 'zscore', 'minmax', or 'robust'.")
