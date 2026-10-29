"""Phase 8 End-to-end integration tests covering data loading through prediction and error recovery."""
from __future__ import annotations

import numpy as np
import pytest

from config.settings import get_settings
from src.ecg_analysis.data.dataset_loader import ECGDataLoader, ECGRecord
from src.ecg_analysis.data.label_mapping import AAMI_CLASSES, aami_to_int, int_to_aami
from src.ecg_analysis.features.feature_pipeline import ECGFeatureExtractor
from src.ecg_analysis.models.model_registry import ModelRegistry
from src.ecg_analysis.models.predict import ECGPredictor
from src.ecg_analysis.preprocessing.pipeline import ECGPreprocessingPipeline
from src.ecg_analysis.preprocessing.segmentation import ECGBeat
from src.ecg_analysis.utils.exceptions import (
    DatasetNotFoundError,
    ModelNotFoundError,
)
from tests.conftest import generate_synthetic_ecg


class TestEndToEndPipeline:
    """Validate full end-to-end signal processing and inference workflows."""

    def test_end_to_end_annotated_pipeline(self) -> None:
        """Verify pipeline from raw record loading through preprocessing, feature extraction, and prediction."""
        loader = ECGDataLoader()
        pipeline = ECGPreprocessingPipeline()
        extractor = ECGFeatureExtractor(wavelet="db4", level=4)

        # 1. Load record (real record 100 or synthetic fallback)
        if loader.record_exists("100"):
            record = loader.load_record("100", sampfrom=0, sampto=3600)
        else:
            sig, r_peaks, _ = generate_synthetic_ecg(duration_sec=10.0, sampling_rate=360)
            record = ECGRecord(
                record_name="synth_e2e",
                sampling_rate=360,
                lead_names=["MLII"],
                signals=sig.reshape(-1, 1),
                annotation_samples=r_peaks,
                annotation_symbols=["N"] * len(r_peaks),
            )

        # 2. Preprocess continuous signal and segment beats
        clean_sig, beats = pipeline.preprocess_record(record, lead_name="MLII")
        assert len(clean_sig) == record.num_samples
        assert len(beats) > 0

        # 3. Extract feature matrix
        X_df, y_num, y_aami = extractor.extract_feature_matrix(beats)
        assert len(X_df) == len(beats)
        assert X_df.shape[1] == 39
        assert not X_df.isna().any().any()

        # 4. Predict using ECGPredictor
        registry = ModelRegistry()
        available = registry.list_available_models()
        if available:
            model_name = available[0]["name"]
            predictor = ECGPredictor(model_name=model_name)
            summary = predictor.predict_beats(beats)

            assert summary.total_beats == len(beats)
            assert summary.normal_beats + summary.abnormal_beats == summary.total_beats
            assert summary.overall_risk in [
                "LOW / NORMAL RISK",
                "MODERATE CARDIAC RISK",
                "HIGH ARRHYTHMIA RISK",
            ]

    def test_end_to_end_unannotated_stream_prediction(self) -> None:
        """Verify processing and prediction of raw continuous signal without ground-truth annotations."""
        fs = 360
        sig, _, _ = generate_synthetic_ecg(duration_sec=8.0, sampling_rate=fs, heart_rate_bpm=72.0)

        pipeline = ECGPreprocessingPipeline()
        clean_sig, beats, r_peaks = pipeline.preprocess_unannotated_signal(
            sig, fs=fs, lead_name="MLII", record_name="stream_test"
        )

        assert len(clean_sig) == len(sig)
        assert len(beats) > 0
        assert len(r_peaks) == len(beats)

        # Predict with mock classical model
        from src.ecg_analysis.models.classical_ml import ClassicalMLModel

        extractor = ECGFeatureExtractor()
        X = np.random.randn(40, 39)
        y = np.array([0] * 30 + [1] * 10)
        mock_model = ClassicalMLModel(model_type="random_forest", n_estimators=5, random_state=42)
        mock_model.fit(X, y)

        predictor = ECGPredictor(model=mock_model, feature_extractor=extractor)
        summary = predictor.predict_beats(beats)

        assert summary.total_beats == len(beats)
        for res in summary.results:
            assert res.predicted_class in AAMI_CLASSES
            assert 0.0 <= res.confidence <= 1.0


class TestModelLoadingIntegrity:
    """Validate saved model checkpoints and registry persistence."""

    def test_saved_model_registry_loading(self) -> None:
        """Verify saved models in registry can be loaded and produce consistent predictions."""
        registry = ModelRegistry()
        models = registry.list_available_models()
        assert len(models) >= 1, "At least one model must be registered in models/saved/"

        test_beat = ECGBeat(
            signal=np.sin(np.linspace(0, 4, 250)),
            r_peak_index=90,
            r_peak_rel_index=90,
            aami_label="N",
            numeric_label=0,
        )

        for m_info in models:
            m_name = m_info["name"]
            predictor = ECGPredictor(model_name=m_name)
            res = predictor.predict_beat(test_beat)

            assert res.predicted_class in AAMI_CLASSES
            assert 0.0 <= res.confidence <= 1.0
            assert res.severity_level in [
                "Normal",
                "Moderate Risk",
                "Critical Attention",
                "Indeterminate / Paced",
            ]

    def test_request_missing_model_raises_model_not_found(self) -> None:
        """Verify requesting a non-existent model name raises ModelNotFoundError."""
        with pytest.raises(ModelNotFoundError):
            ECGPredictor(model_name="completely_unregistered_model_xyz")


class TestMalformedInputAndErrorRecovery:
    """Validate graceful error recovery on malformed or invalid inputs."""

    def test_empty_signal_rejected(self) -> None:
        """Verify empty signal input raises appropriate exception."""
        pipeline = ECGPreprocessingPipeline()
        with pytest.raises(Exception):
            pipeline.process_continuous_signal(np.array([]))

    def test_nan_corrupted_signal_detected(self) -> None:
        """Verify signal containing NaNs fails dataset validation."""
        from src.ecg_analysis.data.dataset_validator import DatasetValidator

        nan_signal = np.ones((500, 1))
        nan_signal[250, 0] = np.nan
        rec = ECGRecord(
            record_name="nan_test",
            sampling_rate=360,
            lead_names=["MLII"],
            signals=nan_signal,
        )
        validator = DatasetValidator()
        result = validator.validate_record(rec)
        assert result.is_valid is False
        assert any("nan" in err.lower() for err in result.issues)

    def test_missing_dataset_file_raises_dataset_not_found(self) -> None:
        """Verify attempting to load a missing record file raises DatasetNotFoundError."""
        loader = ECGDataLoader()
        with pytest.raises(DatasetNotFoundError):
            loader.load_record("non_existent_record_404")


class TestSystemConsistency:
    """Validate consistency across project components."""

    def test_aami_class_mapping_bijectivity(self) -> None:
        """Verify bidirectional AAMI label mapping is consistent across all classes."""
        for idx, cls in enumerate(AAMI_CLASSES):
            assert aami_to_int(cls) == idx
            assert int_to_aami(idx) == cls

    def test_settings_match_project_constants(self) -> None:
        """Verify AppSettings parameters align with pipeline constants."""
        settings = get_settings()
        assert settings.default_sampling_rate == 360
        assert settings.beat_length == 250
        assert settings.pre_r_samples == 90
        assert settings.post_r_samples == 160
        assert settings.aami_classes == ("N", "S", "V", "F", "Q")
