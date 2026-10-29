"""Shared test fixtures and synthetic ECG generators for deterministic testing."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root and src are on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from src.ecg_analysis.data.dataset_loader import ECGRecord


def generate_synthetic_ecg(
    duration_sec: float = 10.0,
    sampling_rate: int = 360,
    heart_rate_bpm: float = 72.0,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate a realistic synthetic ECG signal with known R-peak locations and annotations.

    Returns:
        signal: 1D numpy array of ECG amplitudes (mV).
        r_peaks: 1D numpy array of sample indices where R-peaks occur.
        labels: List of AAMI labels ('N').
    """
    rng = np.random.RandomState(seed)
    total_samples = int(duration_sec * sampling_rate)
    time = np.linspace(0, duration_sec, total_samples, endpoint=False)

    rr_interval = 60.0 / heart_rate_bpm
    num_beats = int(duration_sec / rr_interval)

    signal = np.zeros(total_samples)
    r_peaks = []
    labels = []

    for i in range(num_beats):
        t_center = (i + 0.5) * rr_interval
        r_idx = int(t_center * sampling_rate)
        if 0 <= r_idx < total_samples:
            r_peaks.append(r_idx)
            labels.append("N")

        # Gaussian wavelets for P, Q, R, S, T waves relative to R-peak
        # P wave: amplitude 0.25 mV, width 0.04s, -0.2s before R
        t_p = t_center - 0.20
        signal += 0.25 * np.exp(-((time - t_p) ** 2) / (2 * (0.035**2)))

        # Q wave: amplitude -0.15 mV, width 0.02s, -0.04s before R
        t_q = t_center - 0.04
        signal -= 0.15 * np.exp(-((time - t_q) ** 2) / (2 * (0.015**2)))

        # R wave: amplitude 1.2 mV, width 0.025s, at t_center
        signal += 1.2 * np.exp(-((time - t_center) ** 2) / (2 * (0.025**2)))

        # S wave: amplitude -0.35 mV, width 0.02s, +0.04s after R
        t_s = t_center + 0.04
        signal -= 0.35 * np.exp(-((time - t_s) ** 2) / (2 * (0.015**2)))

        # T wave: amplitude 0.35 mV, width 0.06s, +0.25s after R
        t_t = t_center + 0.25
        signal += 0.35 * np.exp(-((time - t_t) ** 2) / (2 * (0.055**2)))

    # Add Gaussian high-frequency noise and slight baseline wander
    noise = rng.normal(0, noise_std, total_samples)
    baseline_wander = 0.15 * np.sin(2 * np.pi * 0.25 * time)  # 0.25 Hz respiration
    signal += noise + baseline_wander

    return signal, np.array(r_peaks, dtype=int), labels


@pytest.fixture
def synthetic_record() -> ECGRecord:
    """Fixture providing a valid 10-second synthetic ECGRecord."""
    sig, r_peaks, labels = generate_synthetic_ecg(duration_sec=10.0, sampling_rate=360)
    signals_2d = np.column_stack([sig, sig * 0.8])  # Two leads: MLII, V1
    return ECGRecord(
        record_name="synth_100",
        sampling_rate=360,
        lead_names=["MLII", "V1"],
        signals=signals_2d,
        annotation_samples=r_peaks,
        annotation_symbols=["N"] * len(r_peaks),
        aami_labels=labels,
        comments=["Synthetic test record"],
    )
