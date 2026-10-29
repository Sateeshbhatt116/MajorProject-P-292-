"""Unit tests for beat segmentation and Pan-Tompkins QRS detector."""
from __future__ import annotations

import numpy as np
import pytest

from src.ecg_analysis.data.dataset_loader import ECGRecord
from src.ecg_analysis.preprocessing.segmentation import (
    pan_tompkins_qrs_detector,
    segment_beats_from_annotations,
    segment_beats_unannotated,
)
from tests.conftest import generate_synthetic_ecg


class TestPanTompkins:
    """Test suite for the Pan-Tompkins QRS detection algorithm."""

    def test_r_peak_detection_accuracy(self) -> None:
        # Generate 15-second clean synthetic ECG with known R-peaks (at ~72 bpm)
        duration_sec = 15.0
        fs = 360
        sig, true_peaks, _ = generate_synthetic_ecg(
            duration_sec=duration_sec,
            sampling_rate=fs,
            heart_rate_bpm=72.0,
            noise_std=0.01,
            seed=42,
        )

        detected_peaks = pan_tompkins_qrs_detector(sig, fs=fs)

        assert len(detected_peaks) > 0

        # Check sensitivity: for each true peak, is there a detected peak within ±25 samples (±70 ms)?
        matched_count = 0
        tolerance_samples = int(0.070 * fs)

        for tp in true_peaks:
            diffs = np.abs(detected_peaks - tp)
            if np.min(diffs) <= tolerance_samples:
                matched_count += 1

        sensitivity = matched_count / len(true_peaks)
        assert sensitivity >= 0.85, f"Sensitivity {sensitivity:.2f} is below 85%"

    def test_empty_signal_error(self) -> None:
        with pytest.raises(Exception):
            pan_tompkins_qrs_detector(np.array([]))


class TestBeatSegmentation:
    """Test suite for beat extraction."""

    def test_annotated_segmentation(self, synthetic_record: ECGRecord) -> None:
        beats = segment_beats_from_annotations(
            ecg_signal=synthetic_record.get_lead_signal("MLII"),
            annotation_samples=synthetic_record.annotation_samples,
            annotation_symbols=synthetic_record.annotation_symbols,
            fs=360,
            pre_r_samples=90,
            post_r_samples=160,
            record_name=synthetic_record.record_name,
        )

        assert len(beats) > 0
        for b in beats:
            assert b.length == 250
            assert b.r_peak_rel_index == 90
            assert b.aami_label == "N"
            assert b.numeric_label == 0
            assert b.record_name == "synth_100"

    def test_unannotated_segmentation(self) -> None:
        sig, _, _ = generate_synthetic_ecg(duration_sec=6.0, sampling_rate=360)
        beats, r_peaks = segment_beats_unannotated(sig, fs=360, pre_r_samples=90, post_r_samples=160)

        assert len(beats) > 0
        assert len(r_peaks) == len(beats)
        for b in beats:
            assert b.length == 250
            assert b.aami_label == "UNKNOWN"
            assert b.numeric_label == -1
