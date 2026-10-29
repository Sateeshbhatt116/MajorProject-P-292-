"""Unit tests for statistical, morphological, and wavelet feature extraction."""
from __future__ import annotations

import numpy as np
import pytest

from src.ecg_analysis.features.feature_pipeline import ECGFeatureExtractor
from src.ecg_analysis.features.statistical_features import extract_statistical_features
from src.ecg_analysis.features.wavelet_features import extract_wavelet_features
from src.ecg_analysis.preprocessing.segmentation import ECGBeat
from tests.conftest import generate_synthetic_ecg


class TestStatisticalFeatures:
    """Test suite for time-domain features."""

    def test_statistical_feature_keys_and_values(self) -> None:
        # Synthetic single beat of 250 samples
        beat = np.zeros(250)
        beat[90] = 1.5  # R peak
        beat[80] = -0.2  # Q
        beat[100] = -0.3  # S

        feats = extract_statistical_features(beat, r_peak_rel_idx=90)

        required_keys = [
            "stat_mean",
            "stat_std",
            "stat_var",
            "stat_min",
            "stat_max",
            "stat_ptp",
            "stat_skewness",
            "stat_kurtosis",
            "stat_energy",
            "stat_rms",
            "stat_zcr",
            "stat_r_amplitude",
            "stat_pre_energy",
            "stat_post_energy",
            "stat_energy_ratio",
            "stat_left_slope",
            "stat_right_slope",
        ]

        for k in required_keys:
            assert k in feats, f"Missing feature key: {k}"
            assert np.isfinite(feats[k]), f"Feature {k} is not finite: {feats[k]}"

        assert feats["stat_r_amplitude"] == 1.5
        assert feats["stat_ptp"] == pytest.approx(1.8, abs=1e-5)
        assert feats["stat_energy"] > 0


class TestWaveletFeatures:
    """Test suite for DWT features."""

    def test_wavelet_feature_keys_and_entropy(self) -> None:
        beat = np.sin(np.linspace(0, 4 * np.pi, 250))
        feats = extract_wavelet_features(beat, wavelet="db4", level=4)

        assert "dwt_total_energy" in feats
        assert "dwt_wavelet_entropy" in feats
        assert feats["dwt_total_energy"] > 0.0
        assert feats["dwt_wavelet_entropy"] >= 0.0

        for key, val in feats.items():
            assert np.isfinite(val), f"Wavelet feature {key} is not finite: {val}"


class TestFeaturePipeline:
    """Test suite for the combined ECGFeatureExtractor."""

    def test_extract_feature_matrix(self) -> None:
        sig, r_peaks, labels = generate_synthetic_ecg(duration_sec=6.0, sampling_rate=360)
        # Create a few beats
        beats = [
            ECGBeat(
                signal=sig[p - 90 : p + 160],
                r_peak_index=p,
                r_peak_rel_index=90,
                aami_label="N",
                numeric_label=0,
                lead_name="MLII",
                sampling_rate=360,
                record_name="synth_100",
            )
            for p in r_peaks
            if p >= 90 and (p + 160) <= len(sig)
        ]

        assert len(beats) > 0

        extractor = ECGFeatureExtractor(wavelet="db4", level=4)
        X_df, y_num, y_aami = extractor.extract_feature_matrix(beats)

        assert X_df.shape[0] == len(beats)
        assert X_df.shape[1] > 20  # Has > 20 statistical & wavelet features
        assert len(y_num) == len(beats)
        assert len(y_aami) == len(beats)
        assert not X_df.isna().any().any()
        assert not np.isinf(X_df.values).any()

    def test_feature_ordering_consistency(self) -> None:
        """Verify feature ordering is deterministic across different beats."""
        extractor = ECGFeatureExtractor(wavelet="db4", level=4)
        b1 = np.random.randn(250)
        b2 = np.random.randn(250)
        f1 = list(extractor.extract_beat_features_dict(b1, 90).keys())
        f2 = list(extractor.extract_beat_features_dict(b2, 90).keys())
        assert f1 == f2
        assert f1 == extractor.feature_names

    def test_label_feature_alignment(self) -> None:
        """Verify no row-label misalignment when extracting mixed-class beats."""
        labels = ["N", "S", "V", "F", "Q"]
        numeric = [0, 1, 2, 3, 4]
        beats = [
            ECGBeat(
                signal=np.sin(np.linspace(0, i + 1, 250)),
                r_peak_index=100 * (i + 1),
                r_peak_rel_index=90,
                aami_label=lbl,
                numeric_label=num,
                lead_name="MLII",
                sampling_rate=360,
                record_name=f"rec_{i}",
            )
            for i, (lbl, num) in enumerate(zip(labels, numeric))
        ]

        extractor = ECGFeatureExtractor(wavelet="db4", level=4)
        X_df, y_num, y_aami = extractor.extract_feature_matrix(beats)

        assert len(X_df) == 5
        assert list(y_num) == numeric
        assert list(y_aami) == labels

    def test_reproducible_and_leak_free_splitting(self) -> None:
        """Verify inter-patient splitting is reproducible and eliminates data leakage."""
        from src.ecg_analysis.data.dataset_splitter import (
            split_beats_inter_patient,
            verify_zero_leakage,
        )

        # Create beats spanning 4 distinct patient records
        records = ["101", "106", "118", "200"]
        beats = []
        for rec in records:
            for b_idx in range(25):
                beats.append(
                    ECGBeat(
                        signal=np.random.randn(250),
                        r_peak_index=b_idx * 250 + 90,
                        r_peak_rel_index=90,
                        aami_label="N",
                        numeric_label=0,
                        lead_name="MLII",
                        sampling_rate=360,
                        record_name=rec,
                    )
                )

        # Test reproducibility
        split1 = split_beats_inter_patient(beats, test_size=0.25, val_size=0.25, random_state=42)
        split2 = split_beats_inter_patient(beats, test_size=0.25, val_size=0.25, random_state=42)

        assert split1.total_beats == len(beats)
        assert [b.record_name for b in split1.train_beats] == [b.record_name for b in split2.train_beats]
        assert [b.record_name for b in split1.test_beats] == [b.record_name for b in split2.test_beats]

        # Verify zero leakage
        assert verify_zero_leakage(split1) is True
        assert len(split1.train_records.intersection(split1.test_records)) == 0
        assert len(split1.train_records.intersection(split1.val_records)) == 0

