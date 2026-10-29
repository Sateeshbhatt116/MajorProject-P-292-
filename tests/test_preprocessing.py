"""Unit tests for biomedical filtering, baseline correction, and normalization."""
from __future__ import annotations

import numpy as np
import pytest

from src.ecg_analysis.preprocessing.baseline_correction import (
    median_baseline_correction,
    remove_baseline_wander,
    wavelet_baseline_correction,
)
from src.ecg_analysis.preprocessing.filtering import (
    bandpass_filter,
    lowpass_filter,
    notch_filter,
)
from src.ecg_analysis.preprocessing.normalization import (
    minmax_normalize,
    normalize_signal,
    robust_normalize,
    zscore_normalize,
)
from src.ecg_analysis.preprocessing.pipeline import (
    ECGPreprocessingPipeline,
)
from src.ecg_analysis.utils.exceptions import PreprocessingError


class TestFiltering:
    """Test suite for digital filters."""

    def test_bandpass_attenuates_noise(self) -> None:
        fs = 360.0
        t = np.linspace(0, 5, int(5 * fs), endpoint=False)
        # 10 Hz cardiac signal + 0.1 Hz baseline drift + 120 Hz EMG noise
        clean_comp = np.sin(2 * np.pi * 10 * t)
        low_comp = 2.0 * np.sin(2 * np.pi * 0.1 * t)
        high_comp = 1.0 * np.sin(2 * np.pi * 120 * t)
        noisy = clean_comp + low_comp + high_comp

        filtered = bandpass_filter(noisy, lowcut=0.5, highcut=45.0, fs=fs, order=3)

        # Baseline (0.1 Hz) and high freq (120 Hz) should be strongly attenuated
        assert len(filtered) == len(noisy)
        assert np.std(filtered) < np.std(noisy)
        # Power of filtered signal should be close to 10 Hz component power (0.5 for unit sin)
        assert abs(np.std(filtered) - np.std(clean_comp)) < 0.25

    def test_notch_filter_attenuates_powerline(self) -> None:
        fs = 360.0
        t = np.linspace(0, 3, int(3 * fs), endpoint=False)
        # 60 Hz pure powerline artifact
        powerline = np.sin(2 * np.pi * 60.0 * t)
        filtered = notch_filter(powerline, notch_freq=60.0, quality_factor=30.0, fs=fs)

        # Powerline amplitude should be attenuated by > 80%
        assert np.max(np.abs(filtered[100:-100])) < 0.20

    def test_lowpass_filter(self) -> None:
        fs = 360.0
        t = np.linspace(0, 2, int(2 * fs), endpoint=False)
        high_freq = np.sin(2 * np.pi * 90.0 * t)
        filtered = lowpass_filter(high_freq, cutoff_freq=45.0, fs=fs)
        assert np.std(filtered[50:-50]) < 0.15

    def test_short_signal_raises_preprocessing_error(self) -> None:
        short_sig = np.array([1.0, 2.0, 3.0])
        with pytest.raises(PreprocessingError):
            bandpass_filter(short_sig)


class TestBaselineCorrection:
    """Test suite for baseline wander removal."""

    def test_median_baseline_correction(self) -> None:
        fs = 360.0
        t = np.linspace(0, 4, int(4 * fs), endpoint=False)
        # Artificial drift + rapid pulse
        drift = 1.5 * np.sin(2 * np.pi * 0.2 * t)
        pulse = np.zeros_like(t)
        pulse[int(2 * fs)] = 2.0  # sharp QRS-like spike

        corrected = median_baseline_correction(drift + pulse, fs=fs)
        # Drift should be removed; pulse should remain near index
        assert abs(np.mean(corrected)) < 0.1
        assert corrected[int(2 * fs)] > 1.0

    def test_wavelet_baseline_correction(self) -> None:
        fs = 360.0
        t = np.linspace(0, 4, int(4 * fs), endpoint=False)
        drift = 2.0 * np.sin(2 * np.pi * 0.15 * t)
        corrected = wavelet_baseline_correction(drift)
        assert np.std(corrected[100:-100]) < 0.40

    def test_remove_baseline_wander_dispatcher(self) -> None:
        sig = np.ones(500)
        out_med = remove_baseline_wander(sig, method="median")
        out_wav = remove_baseline_wander(sig, method="wavelet")
        assert len(out_med) == 500
        assert len(out_wav) == 500
        with pytest.raises(ValueError):
            remove_baseline_wander(sig, method="unknown")


class TestNormalization:
    """Test suite for signal normalization."""

    def test_zscore_normalize(self) -> None:
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        norm = zscore_normalize(arr)
        assert pytest.approx(float(np.mean(norm)), abs=1e-6) == 0.0
        assert pytest.approx(float(np.std(norm)), abs=1e-4) == 1.0

    def test_minmax_normalize(self) -> None:
        arr = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        norm = minmax_normalize(arr, feature_range=(-1.0, 1.0))
        assert pytest.approx(float(np.min(norm)), abs=1e-5) == -1.0
        assert pytest.approx(float(np.max(norm)), abs=1e-5) == 1.0

    def test_robust_normalize(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 1000.0])  # has extreme outlier
        norm = robust_normalize(arr)
        assert pytest.approx(float(np.median(norm)), abs=1e-5) == 0.0

    def test_normalize_dispatcher(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert len(normalize_signal(arr, "zscore")) == 5
        assert len(normalize_signal(arr, "minmax")) == 5
        assert len(normalize_signal(arr, "robust")) == 5
        with pytest.raises(ValueError):
            normalize_signal(arr, "invalid")


class TestPipeline:
    """Test suite for ECGPreprocessingPipeline."""

    def test_pipeline_continuous_signal(self) -> None:
        pipeline = ECGPreprocessingPipeline()
        fs = 360
        t = np.linspace(0, 3, 3 * fs)
        raw = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 60 * t)
        clean = pipeline.process_continuous_signal(raw, fs=fs)
        assert len(clean) == len(raw)
        assert not np.isnan(clean).any()

    def test_pipeline_output_dimensions(self) -> None:
        """Verify output dimensions match input dimensions across diverse signal lengths."""
        pipeline = ECGPreprocessingPipeline()
        for length in [720, 1440, 3600]:
            raw = np.random.randn(length)
            clean = pipeline.process_continuous_signal(raw, fs=360.0)
            assert clean.shape == (length,)

    def test_pipeline_reproducibility(self) -> None:
        """Verify deterministic preprocessing behavior on identical inputs."""
        pipeline = ECGPreprocessingPipeline()
        rng = np.random.RandomState(42)
        raw = rng.randn(1800)
        clean1 = pipeline.process_continuous_signal(raw, fs=360.0)
        clean2 = pipeline.process_continuous_signal(raw, fs=360.0)
        assert np.array_equal(clean1, clean2)

    def test_segment_length_consistency(self) -> None:
        """Verify beat segment lengths remain strictly 250 samples."""
        from tests.conftest import generate_synthetic_ecg
        from src.ecg_analysis.data.dataset_loader import ECGRecord

        sig, r_peaks, _ = generate_synthetic_ecg(duration_sec=8.0, sampling_rate=360)
        rec = ECGRecord(
            record_name="synth_len_test",
            sampling_rate=360,
            lead_names=["MLII"],
            signals=sig.reshape(-1, 1),
            annotation_samples=r_peaks,
            annotation_symbols=["N"] * len(r_peaks),
        )

        pipeline = ECGPreprocessingPipeline()
        _, beats = pipeline.preprocess_record(rec)
        assert len(beats) > 0
        for b in beats:
            assert len(b.signal) == 250
            assert b.r_peak_rel_index == 90

