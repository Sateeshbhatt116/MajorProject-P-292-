"""Digital filtering routines for ECG noise reduction and artifact removal.

Implements zero-phase Butterworth bandpass filtering, powerline interference notch
filtering, and high-frequency EMG noise suppression conforming to AAMI/AHA guidelines.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal

from src.ecg_analysis.utils.exceptions import PreprocessingError
from src.ecg_analysis.utils.logging_config import setup_logger

logger = setup_logger("ecg_analysis.preprocessing.filtering")


def _get_butter_bandpass_coeffs(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Butterworth bandpass filter coefficients (b, a)."""
    nyq = 0.5 * fs
    if lowcut <= 0 or lowcut >= nyq:
        raise ValueError(f"lowcut ({lowcut}) must be between 0 and Nyquist frequency ({nyq})")
    if highcut <= lowcut or highcut >= nyq:
        raise ValueError(f"highcut ({highcut}) must be between lowcut ({lowcut}) and Nyquist ({nyq})")

    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="bandpass")
    return b, a


def bandpass_filter(
    ecg_signal: np.ndarray,
    lowcut: float = 0.5,
    highcut: float = 45.0,
    fs: float = 360.0,
    order: int = 3,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter to ECG signal.

    Attenuates baseline drift (<0.5 Hz) and muscle EMG artifact (>45 Hz)
    while preserving the critical QRS morphology (5-30 Hz) and P/T waves.

    Args:
        ecg_signal: 1D or 2D array of ECG signal amplitudes.
        lowcut: Lower cutoff frequency in Hz (default: 0.5 Hz).
        highcut: Upper cutoff frequency in Hz (default: 45.0 Hz).
        fs: Sampling frequency in Hz (default: 360 Hz).
        order: Butterworth filter order (default: 3).

    Returns:
        Filtered ECG signal with identical shape and zero phase distortion.
    """
    if ecg_signal is None or len(ecg_signal) == 0:
        raise PreprocessingError("Input signal is empty or None")

    sig_len = ecg_signal.shape[0]
    min_len = 3 * max(2 * order + 1, 15)
    if sig_len < min_len:
        raise PreprocessingError(
            f"Signal length ({sig_len}) is too short for zero-phase filter (min {min_len})"
        )

    try:
        b, a = _get_butter_bandpass_coeffs(lowcut, highcut, fs, order)
        # Apply forward-backward filtering along axis 0 for zero phase delay
        filtered = signal.filtfilt(b, a, ecg_signal, axis=0)
        return filtered.astype(np.float64)
    except Exception as exc:
        raise PreprocessingError(f"Bandpass filtering failed: {exc}") from exc


def notch_filter(
    ecg_signal: np.ndarray,
    notch_freq: float = 60.0,
    quality_factor: float = 30.0,
    fs: float = 360.0,
) -> np.ndarray:
    """Apply zero-phase IIR notch filter to remove powerline interference.

    Args:
        ecg_signal: 1D or 2D array of ECG signal amplitudes.
        notch_freq: Center frequency of the notch in Hz (default: 60 Hz for US/MIT-BIH).
        quality_factor: Quality factor Q (default: 30.0 for narrow notch).
        fs: Sampling frequency in Hz (default: 360 Hz).

    Returns:
        Notch-filtered ECG signal.
    """
    if ecg_signal is None or len(ecg_signal) == 0:
        raise PreprocessingError("Input signal is empty or None")

    nyq = 0.5 * fs
    if notch_freq >= nyq:
        logger.warning(
            f"Notch frequency ({notch_freq} Hz) exceeds Nyquist ({nyq} Hz). Skipping notch."
        )
        return ecg_signal

    try:
        b, a = signal.iirnotch(notch_freq, quality_factor, fs)
        filtered = signal.filtfilt(b, a, ecg_signal, axis=0)
        return filtered.astype(np.float64)
    except Exception as exc:
        raise PreprocessingError(f"Notch filtering failed: {exc}") from exc


def lowpass_filter(
    ecg_signal: np.ndarray,
    cutoff_freq: float = 45.0,
    fs: float = 360.0,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth low-pass filter to remove high-frequency noise."""
    if ecg_signal is None or len(ecg_signal) == 0:
        raise PreprocessingError("Input signal is empty or None")

    nyq = 0.5 * fs
    if cutoff_freq >= nyq:
        return ecg_signal

    try:
        norm_cut = cutoff_freq / nyq
        b, a = signal.butter(order, norm_cut, btype="lowpass")
        filtered = signal.filtfilt(b, a, ecg_signal, axis=0)
        return filtered.astype(np.float64)
    except Exception as exc:
        raise PreprocessingError(f"Lowpass filtering failed: {exc}") from exc
