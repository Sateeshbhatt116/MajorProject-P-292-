"""Unit tests for inference and risk assessment."""
from __future__ import annotations

import numpy as np

from src.ecg_analysis.models.classical_ml import ClassicalMLModel
from src.ecg_analysis.models.predict import (
    ECGPredictor,
    assess_cardiac_severity,
)
from src.ecg_analysis.preprocessing.segmentation import ECGBeat


class TestPrediction:
    """Test suite for ECGPredictor and clinical severity logic."""

    def test_assess_cardiac_severity(self) -> None:
        sev_n, _ = assess_cardiac_severity("N")
        assert "Normal" in sev_n

        sev_v, _ = assess_cardiac_severity("V")
        assert "Critical" in sev_v

        sev_s, _ = assess_cardiac_severity("S")
        assert "Moderate" in sev_s

    def test_predictor_with_classical_model(self) -> None:
        # Train mock classical model
        from src.ecg_analysis.features.feature_pipeline import ECGFeatureExtractor

        extractor = ECGFeatureExtractor()
        dummy_beat = np.zeros(250)
        feats = extractor.extract_beat_features_dict(dummy_beat)
        num_features = len(feats)

        X = np.random.normal(size=(50, num_features))
        y = np.array([0] * 35 + [2] * 15)  # 0: N, 2: V
        model = ClassicalMLModel(model_type="random_forest", n_estimators=5, random_state=42)
        model.fit(X, y)

        predictor = ECGPredictor(model=model, feature_extractor=extractor)

        beat1 = ECGBeat(
            signal=np.random.normal(size=250),
            r_peak_index=100,
            r_peak_rel_index=90,
            aami_label="N",
            numeric_label=0,
        )

        res = predictor.predict_beat(beat1)
        assert res.predicted_class in ["N", "V", "S", "F", "Q"]
        assert 0.0 <= res.confidence <= 1.0
        assert isinstance(res.is_abnormal, bool)
        assert res.severity_level in [
            "Normal",
            "Moderate Risk",
            "Critical Attention",
            "Indeterminate / Paced",
        ]

        # Batch prediction
        batch_summary = predictor.predict_beats([beat1, beat1])
        assert batch_summary.total_beats == 2
        assert "RISK" in batch_summary.overall_risk
