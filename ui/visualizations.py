"""Interactive Matplotlib and Seaborn visualization functions for ECG Cardian."""
from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pywt

from src.ecg_analysis.data.label_mapping import AAMI_CLASS_COLORS, AAMI_CLASSES
from src.ecg_analysis.preprocessing.segmentation import ECGBeat


def plot_continuous_ecg(
    raw_signal: np.ndarray,
    clean_signal: np.ndarray,
    r_peaks: Optional[np.ndarray] = None,
    fs: int = 360,
    start_sec: float = 0.0,
    duration_sec: float = 6.0,
    lead_name: str = "MLII",
) -> plt.Figure:
    """Plot multi-panel continuous ECG signal comparing raw vs. preprocessed waveforms."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, dpi=120)

    start_idx = int(start_sec * fs)
    end_idx = min(len(raw_signal), int((start_sec + duration_sec) * fs))

    time_vec = np.linspace(start_sec, start_sec + (end_idx - start_idx) / fs, end_idx - start_idx)

    # 1. Raw ECG with baseline drift / artifacts
    ax1.plot(time_vec, raw_signal[start_idx:end_idx], color="#64748b", linewidth=1.1, label="Raw Unfiltered Signal")
    ax1.set_ylabel("Amplitude (mV)", fontsize=10)
    ax1.set_title(f"Raw ECG Acquisition — Lead: {lead_name}", fontsize=11, fontweight="bold", pad=8)
    ax1.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax1.legend(loc="upper right", framealpha=0.9)

    # 2. Filtered & Baseline-Corrected ECG
    ax2.plot(time_vec, clean_signal[start_idx:end_idx], color="#2563eb", linewidth=1.2, label="Preprocessed (Bandpass + Notch + Median)")

    # Overlay detected R-peaks in window
    if r_peaks is not None and len(r_peaks) > 0:
        window_peaks = r_peaks[(r_peaks >= start_idx) & (r_peaks < end_idx)]
        if len(window_peaks) > 0:
            peak_times = (window_peaks - start_idx) / fs + start_sec
            peak_amps = clean_signal[window_peaks]
            ax2.scatter(
                peak_times,
                peak_amps,
                color="#ef4444",
                s=40,
                zorder=5,
                label=f"Detected R-Peaks ({len(window_peaks)})",
            )

    ax2.set_xlabel("Time (seconds)", fontsize=10)
    ax2.set_ylabel("Amplitude (mV)", fontsize=10)
    ax2.set_title("Preprocessed ECG with QRS Alignments", fontsize=11, fontweight="bold", pad=8)
    ax2.grid(True, linestyle="--", alpha=0.5, color="#cbd5e1")
    ax2.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    return fig


def plot_beat_morphology(
    beat: ECGBeat,
    r_peak_rel_idx: int = 90,
    fs: int = 360,
) -> plt.Figure:
    """Plot detailed individual heartbeat waveform with P-QRS-T morphological regions."""
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)

    n_samples = len(beat.signal)
    time_ms = (np.arange(n_samples) - r_peak_rel_idx) / fs * 1000.0  # ms relative to R-peak

    color = AAMI_CLASS_COLORS.get(beat.aami_label, "#2563eb")

    # Shaded physiological segments:
    # P-wave: [-200 ms, -80 ms]
    ax.axvspan(-220, -70, color="#3b82f6", alpha=0.10, label="P-Wave (Atrial)")
    # QRS-complex: [-60 ms, +60 ms]
    ax.axvspan(-60, 60, color="#ef4444", alpha=0.12, label="QRS Complex (Ventricular)")
    # ST-T segment: [+80 ms, +350 ms]
    ax.axvspan(80, 360, color="#10b981", alpha=0.10, label="T-Wave (Repolarization)")

    # Plot beat waveform
    ax.plot(time_ms, beat.signal, color=color, linewidth=2.0, label=f"Beat Waveform (Class {beat.aami_label})")

    # Mark R-peak
    ax.scatter([0.0], [beat.signal[r_peak_rel_idx]], color="#dc2626", s=60, zorder=6, label="R-Peak Reference")

    ax.set_title(
        f"Segmented Beat Morphology (Label: {beat.aami_label}) — Length: {n_samples} Samples",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("Time Relative to R-Peak (ms)", fontsize=10)
    ax.set_ylabel("Normalized Amplitude", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    return fig


def plot_wavelet_decomposition(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
) -> plt.Figure:
    """Plot multi-level Discrete Wavelet Transform decomposition subbands."""
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    # coeffs: [cA4, cD4, cD3, cD2, cD1]

    fig, axes = plt.subplots(len(coeffs), 1, figsize=(9, 6.5), sharex=False, dpi=120)

    band_names = [f"cA{level} (Approximation 0-11 Hz)"] + [
        f"cD{level - i} (Detail ~{11 * (2**(i-1))}-{22 * (2**(i-1))} Hz)"
        for i in range(1, len(coeffs))
    ]

    palette = ["#1e293b", "#0284c7", "#0d9488", "#ca8a04", "#e11d48"]

    for i, (coeff, name, col) in enumerate(zip(coeffs, band_names, palette)):
        axes[i].plot(coeff, color=col, linewidth=1.2)
        energy = np.sum(coeff**2)
        axes[i].set_ylabel(f"Band {i}", fontsize=9)
        axes[i].set_title(f"{name} — Energy: {energy:.2f}", fontsize=9.5, fontweight="600", loc="left", pad=4)
        axes[i].grid(True, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Wavelet Coefficient Index", fontsize=10)
    plt.tight_layout()
    return fig


def plot_class_probabilities(
    probabilities: Dict[str, float],
    predicted_class: str,
) -> plt.Figure:
    """Horizontal bar chart of 5-class AAMI probability distribution."""
    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=120)

    classes = list(AAMI_CLASSES)
    probs = [probabilities.get(c, 0.0) * 100.0 for c in classes]
    colors = [AAMI_CLASS_COLORS.get(c, "#64748b") for c in classes]

    bars = ax.barh(classes, probs, color=colors, height=0.55, edgecolor="#334155", linewidth=0.6)

    # Highlight predicted class
    for bar, cls in zip(bars, classes):
        if cls == predicted_class:
            bar.set_edgecolor("#000000")
            bar.set_linewidth(1.8)

    ax.set_xlim(0, 105)
    ax.set_xlabel("Classification Confidence (%)", fontsize=10)
    ax.set_title("AAMI EC57 Class Probability Breakdown", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Annotate percentage on bars
    for bar, p in zip(bars, probs):
        ax.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{p:.1f}%",
            va="center",
            fontsize=9,
            fontweight="bold" if p > 50 else "normal",
        )

    plt.tight_layout()
    return fig


def plot_feature_importances(
    importances: Dict[str, float],
    top_n: int = 12,
) -> plt.Figure:
    """Bar chart of top biomedical feature importances from Random Forest."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)

    top_items = list(importances.items())[:top_n]
    names = [k for k, _ in reversed(top_items)]
    vals = [v for _, v in reversed(top_items)]

    ax.barh(names, vals, color="#3b82f6", height=0.6)
    ax.set_xlabel("Relative Gini Feature Importance", fontsize=10)
    ax.set_title(f"Top {top_n} Biomedical Features (Random Forest)", fontsize=11, fontweight="bold", pad=10)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    return fig
