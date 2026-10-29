"""Baseline wander removal routines for electrocardiogram signals.

Implements:
1. Two-stage cascaded median filtering (Meyer / de Chazal standard).
2. Multi-level Discrete Wavelet Transform (DWT) baseline suppression.
"""
from __future__ import annotations

import numpy as np
import pywt
from scipy.ndimage import median_filter

from src.ecg_analysis.utils.exceptions import PreprocessingError
from src.ecg_analysis.utils.logging_config import setup_logger

logger = setup_logger("ecg_analysis.preprocessing.baseline_correction")


def median_baseline_correction(
    ecg_signal: np.ndarray,
    fs: float = 360.0,
    window1_ms: float = 200.0,
    window2_ms: float = 600.0,
) -> np.ndarray:
    """Remove baseline wander using cascaded two-stage median filtering.

    The first median filter (default 200 ms) eliminates P-waves and QRS complexes.
    The second median filter (default 600 ms) eliminates T-waves. The resulting
    estimate is the pure baseline drift, which is subtracted from the original ECG.
    This preserves ST-segment morphology without phase distortion or ringing.

    Args:
        ecg_signal: 1D or 2D ECG signal array.
        fs: Sampling rate in Hz (default: 360 Hz).
        window1_ms: Window length of the first median filter in ms (default: 200 ms).
        window2_ms: Window length of the second median filter in ms (default: 600 ms).

    Returns:
        Baseline-corrected ECG signal.
    """
    if ecg_signal is None or len(ecg_signal) == 0:
        raise PreprocessingError("Input signal is empty or None")

    # Compute window sizes in samples (must be odd integers)
    w1 = int(round((window1_ms / 1000.0) * fs))
    if w1 % 2 == 0:
        w1 += 1
    w2 = int(round((window2_ms / 1000.0) * fs))
    if w2 % 2 == 0:
        w2 += 1

    try:
        if ecg_signal.ndim == 1:
            stage1 = median_filter(ecg_signal, size=w1, mode="nearest")
            baseline = median_filter(stage1, size=w2, mode="nearest")
            return (ecg_signal - baseline).astype(np.float64)
        else:
            corrected = np.empty_like(ecg_signal, dtype=np.float64)
            for ch in range(ecg_signal.shape[1]):
                col = ecg_signal[:, ch]
                stage1 = median_filter(col, size=w1, mode="nearest")
                baseline = median_filter(stage1, size=w2, mode="nearest")
                corrected[:, ch] = col - baseline
            return corrected
    except Exception as exc:
        raise PreprocessingError(f"Median baseline correction failed: {exc}") from exc


def wavelet_baseline_correction(
    ecg_signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 8,
) -> np.ndarray:
    """Remove baseline wander using Discrete Wavelet Transform (DWT).

    Decomposes signal to specified level, sets approximation coefficients (low-frequency baseline)
    to zero, and reconstructs the signal.

    Args:
        ecg_signal: 1D numpy array of ECG amplitudes.
        wavelet: Wavelet family (default: 'db4').
        level: Decomposition level (default: 8, covering < 0.7 Hz at 360 Hz).

    Returns:
        Baseline-corrected 1D ECG signal.
    """
    if ecg_signal is None or len(ecg_signal) == 0:
        raise PreprocessingError("Input signal is empty or None")

    if ecg_signal.ndim > 1:
        # Process each channel
        corrected = np.empty_like(ecg_signal, dtype=np.float64)
        for ch in range(ecg_signal.shape[1]):
            corrected[:, ch] = wavelet_baseline_correction(
                ecg_signal[:, ch], wavelet=wavelet, level=level
            )
        return corrected

    try:
        max_level = pywt.dwt_max_level(data_len=len(ecg_signal), filter_len=pywt.Wavelet(wavelet).dec_len)
        actual_level = min(level, max_level)

        coeffs = pywt.wavedec(ecg_signal, wavelet=wavelet, level=actual_level)
        # Zero out approximation coefficients cA_level (lowest frequency band)
        coeffs[0] = np.zeros_like(coeffs[0])
        reconstructed = pywt.waverec(coeffs, wavelet=wavelet)

        # Match original length in case of padding
        return reconstructed[: len(ecg_signal)].astype(np.float64)
    except Exception as exc:
        raise PreprocessingError(f"Wavelet baseline correction failed: {exc}") from exc


def remove_baseline_wander(
    ecg_signal: np.ndarray,
    fs: float = 360.0,
    method: str = "median",
) -> np.ndarray:
    """Convenience dispatcher for baseline wander removal.

    Args:
        ecg_signal: 1D or 2D ECG signal array.
        fs: Sampling rate in Hz.
        method: 'median' (recommended for clinical fidelity) or 'wavelet'.

    Returns:
        Baseline-corrected ECG signal.
    """
    if method == "median":
        return median_baseline_correction(ecg_signal, fs=fs)
    elif method == "wavelet":
        return wavelet_baseline_correction(ecg_signal)
    else:
        raise ValueError(f"Unknown baseline correction method '{method}'. Choose 'median' or 'wavelet'.")
