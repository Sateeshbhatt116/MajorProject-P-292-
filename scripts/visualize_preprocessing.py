"""Generate before/after preprocessing comparison figures and diagnostic reports."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "src"))

from config.settings import get_settings
from src.ecg_analysis.data.dataset_loader import ECGDataLoader, ECGRecord
from src.ecg_analysis.preprocessing.pipeline import ECGPreprocessingPipeline
from src.ecg_analysis.utils.logging_config import setup_logger

logger = setup_logger("scripts.visualize_preprocessing")


def generate_preprocessing_comparison_plot(
    record_name: str = "100",
    lead_name: str = "MLII",
    duration_sec: float = 6.0,
    output_path: Path | None = None,
) -> Path:
    """Generate and save publication-grade before/after preprocessing figure."""
    settings = get_settings()
    out_file = output_path or (settings.reports_figures_dir / "preprocessing_comparison.png")
    out_file.parent.mkdir(parents=True, exist_ok=True)

    loader = ECGDataLoader()
    pipeline = ECGPreprocessingPipeline()

    if loader.record_exists(record_name):
        logger.info(f"Loading real PhysioNet record {record_name}...")
        rec = loader.load_record(record_name)
    else:
        logger.warning(
            f"Record {record_name} not found in {loader.raw_data_dir}. Generating synthetic signal for plot."
        )
        fs = 360
        t = np.linspace(0, duration_sec, int(duration_sec * fs))
        raw_sig = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 0.15 * t) + 0.15 * np.sin(2 * np.pi * 60 * t)
        rec = ECGRecord(
            record_name=f"synth_{record_name}",
            sampling_rate=fs,
            lead_names=[lead_name],
            signals=raw_sig.reshape(-1, 1),
            annotation_samples=np.array([int(i * 0.8 * fs) for i in range(1, int(duration_sec / 0.8))]),
            annotation_symbols=["N"] * int(duration_sec / 0.8),
        )

    fs = rec.sampling_rate
    num_samples = min(int(duration_sec * fs), rec.num_samples)
    time_axis = np.arange(num_samples) / fs

    raw_lead = rec.get_lead_signal(lead_name)[:num_samples]
    clean_lead, beats = pipeline.preprocess_record(rec, lead_name=lead_name)
    clean_lead_crop = clean_lead[:num_samples]

    # Create 4-panel diagnostic plot
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=False)
    plt.subplots_adjust(hspace=0.40)

    # Panel 1: Raw ECG with Baseline Drift and Powerline
    axes[0].plot(time_axis, raw_lead, color="#2c3e50", lw=1.2, label=f"Raw Signal (Lead {lead_name})")
    axes[0].set_title(
        f"Raw Clinical ECG Signal - PhysioNet Record {rec.record_name} (Sampling: {fs} Hz)",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].set_ylabel("Amplitude (mV)", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="upper right", framealpha=0.9)

    # Panel 2: Preprocessed ECG (Notch + Butterworth + Median Baseline Removal)
    axes[1].plot(time_axis, clean_lead_crop, color="#2980b9", lw=1.2, label="Preprocessed & Cleaned ECG")
    # Mark beat R-peaks within crop
    crop_peaks = [b.r_peak_index for b in beats if b.r_peak_index < num_samples]
    if crop_peaks:
        axes[1].scatter(
            [p / fs for p in crop_peaks],
            [clean_lead[p] for p in crop_peaks],
            color="#e74c3c",
            s=40,
            zorder=5,
            label=f"Detected R-peaks (N={len(crop_peaks)})",
        )
    axes[1].set_title(
        "Processed Signal (0.5–45 Hz Bandpass, 60 Hz Notch, 2-Stage Cascaded Median Baseline Correction)",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].set_ylabel("Amplitude (mV)", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="upper right", framealpha=0.9)

    # Panel 3: Power Spectral Density (PSD)
    f_raw, psd_raw = signal.welch(raw_lead, fs=fs, nperseg=min(len(raw_lead), 512))
    f_clean, psd_clean = signal.welch(clean_lead_crop, fs=fs, nperseg=min(len(clean_lead_crop), 512))
    axes[2].semilogy(f_raw, psd_raw, color="#7f8c8d", lw=1.2, label="Raw Spectrum (Drift + Noise)")
    axes[2].semilogy(f_clean, psd_clean, color="#27ae60", lw=1.5, label="Cleaned Spectrum (0.5–45 Hz Band)")
    axes[2].axvspan(0.5, 45.0, color="#2ecc71", alpha=0.12, label="Clinical Diagnostic Band")
    axes[2].set_xlim(0, 100)
    axes[2].set_title("Power Spectral Density (Welch Periodogram)", fontsize=11, fontweight="bold")
    axes[2].set_xlabel("Frequency (Hz)", fontsize=9)
    axes[2].set_ylabel("Power Density (mV²/Hz)", fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend(loc="upper right", framealpha=0.9)

    # Panel 4: Overlay of Segmented Beats (250 Samples)
    if beats:
        beat_t = np.linspace(-90 / fs, 160 / fs, 250) * 1000  # in ms
        sample_beats = beats[: min(30, len(beats))]
        for b in sample_beats:
            axes[3].plot(beat_t, b.signal, color="#3498db", alpha=0.25, lw=1.0)
        mean_beat = np.mean([b.signal for b in sample_beats], axis=0)
        axes[3].plot(beat_t, mean_beat, color="#c0392b", lw=2.2, label=f"Ensemble Mean Beat (N={len(sample_beats)})")
        axes[3].axvline(0, color="#e74c3c", linestyle="--", lw=1.2, label="R-Peak Reference (t=0 ms)")
        axes[3].set_title(
            f"Segmented Heartbeat Morphology (250 Samples = 694 ms @ {fs} Hz)",
            fontsize=11,
            fontweight="bold",
        )
        axes[3].set_xlabel("Time Relative to R-Peak (ms)", fontsize=9)
        axes[3].set_ylabel("Normalized Amplitude", fontsize=9)
        axes[3].grid(True, linestyle="--", alpha=0.5)
        axes[3].legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved preprocessing comparison figure to {out_file}")
    return out_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ECG preprocessing comparison visualization.")
    parser.add_argument("--record", type=str, default="100", help="WFDB record ID (default: '100')")
    parser.add_argument("--lead", type=str, default="MLII", help="ECG Lead name (default: 'MLII')")
    parser.add_argument("--duration", type=float, default=6.0, help="Window duration in seconds (default: 6.0)")
    args = parser.parse_args()

    out_fig = generate_preprocessing_comparison_plot(
        record_name=args.record,
        lead_name=args.lead,
        duration_sec=args.duration,
    )
    print(f"[SUCCESS] Preprocessing comparison visualization generated: {out_fig}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
