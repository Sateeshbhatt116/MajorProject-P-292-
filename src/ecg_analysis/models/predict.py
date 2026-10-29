"""Unified prediction engine and clinical abnormality risk assessment."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.ecg_analysis.data.label_mapping import (
    AAMI_CLASSES,
    int_to_aami,
)
from src.ecg_analysis.features.feature_pipeline import ECGFeatureExtractor
from src.ecg_analysis.models.classical_ml import ClassicalMLModel
from src.ecg_analysis.models.cnn_model import ECG1DCNN
from src.ecg_analysis.models.model_registry import ModelRegistry
from src.ecg_analysis.preprocessing.pipeline import ECGPreprocessingPipeline
from src.ecg_analysis.preprocessing.segmentation import ECGBeat
from src.ecg_analysis.utils.exceptions import ModelInferenceError


@dataclass
class PredictionResult:
    """Individual heartbeat classification and clinical risk assessment."""

    predicted_class: str  # 'N', 'S', 'V', 'F', 'Q'
    confidence: float
    probabilities: Dict[str, float]
    is_abnormal: bool
    severity_level: str  # 'Normal', 'Moderate Risk', 'Critical Attention', 'Indeterminate'
    clinical_description: str
    beat_index: Optional[int] = None
    r_peak_sample: Optional[int] = None


@dataclass
class BatchPredictionSummary:
    """Aggregated clinical summary for a full ECG recording."""

    total_beats: int
    normal_beats: int
    abnormal_beats: int
    class_counts: Dict[str, int]
    abnormal_percentage: float
    predominant_abnormality: Optional[str]
    overall_risk: str
    results: List[PredictionResult] = field(default_factory=list)


def assess_cardiac_severity(aami_class: str) -> Tuple[str, str]:
    """Map AAMI classification to clinical severity tier and risk guidance."""
    if aami_class == "N":
        return "Normal", "Normal sinus or typical conduction pattern."
    elif aami_class == "S":
        return "Moderate Risk", "Supraventricular ectopic beat (atrial/junctional premature contraction)."
    elif aami_class == "V":
        return (
            "Critical Attention",
            "Ventricular ectopic beat (PVC / ventricular escape). Elevated arrhythmia risk.",
        )
    elif aami_class == "F":
        return "Moderate Risk", "Fusion beat (intermediate ventricular depolarization)."
    elif aami_class == "Q":
        return "Indeterminate / Paced", "Paced or unclassifiable heartbeat morphology."
    return "Unknown", "Unclassified pattern."


class ECGPredictor:
    """High-level prediction service supporting both Classical ML and Deep Learning."""

    def __init__(
        self,
        model: Optional[Union[ClassicalMLModel, ECG1DCNN]] = None,
        model_name: Optional[str] = None,
        feature_extractor: Optional[ECGFeatureExtractor] = None,
        preprocessing_pipeline: Optional[ECGPreprocessingPipeline] = None,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.registry = ModelRegistry()
        self.feature_extractor = feature_extractor or ECGFeatureExtractor()
        self.preprocessing_pipeline = preprocessing_pipeline or ECGPreprocessingPipeline()

        if model is not None:
            self.model = model
            self.model_type = "cnn_1d" if isinstance(model, ECG1DCNN) else "classical_ml"
        elif model_name is not None:
            self._load_by_name(model_name)
        else:
            self.model = None
            self.model_type = "none"

    def _load_by_name(self, model_name: str) -> None:
        """Attempt loading as classical model or CNN from registry."""
        try:
            self.model, _ = self.registry.load_classical_model(model_name)
            self.model_type = "classical_ml"
        except Exception:
            self.model, _ = self.registry.load_cnn_model(model_name, device=self.device)
            self.model_type = "cnn_1d"

    def predict_beat(self, beat: ECGBeat) -> PredictionResult:
        """Classify a single segmented ECGBeat."""
        if self.model is None:
            raise ModelInferenceError("No model loaded in ECGPredictor.")

        if self.model_type == "cnn_1d":
            # CNN directly accepts 1D raw normalized beat tensor
            assert isinstance(self.model, ECG1DCNN)
            tensor_x = torch.from_numpy(beat.signal).float().unsqueeze(0).unsqueeze(0).to(self.device)
            probs_t = self.model.predict_proba(tensor_x).cpu().numpy().ravel()
            pred_idx = int(np.argmax(probs_t))
            pred_class = int_to_aami(pred_idx)
            confidence = float(probs_t[pred_idx])
            probs_dict = {int_to_aami(i): float(p) for i, p in enumerate(probs_t)}
        else:
            # Classical ML model requires extracted features
            assert isinstance(self.model, ClassicalMLModel)
            feat_dict = self.feature_extractor.extract_from_beat(beat)
            probs = self.model.predict_proba([list(feat_dict.values())])[0]
            pred_idx = int(np.argmax(probs))
            pred_class = int_to_aami(pred_idx)
            confidence = float(probs[pred_idx])
            probs_dict = {int_to_aami(i): float(p) for i, p in enumerate(probs)}

        severity, desc = assess_cardiac_severity(pred_class)

        return PredictionResult(
            predicted_class=pred_class,
            confidence=confidence,
            probabilities=probs_dict,
            is_abnormal=(pred_class != "N"),
            severity_level=severity,
            clinical_description=desc,
            beat_index=beat.r_peak_index,
            r_peak_sample=beat.r_peak_index,
        )

    def predict_beats(self, beats: List[ECGBeat]) -> BatchPredictionSummary:
        """Classify a batch of beats using vectorized batch inference and compute clinical risk."""
        if not beats:
            return BatchPredictionSummary(
                total_beats=0,
                normal_beats=0,
                abnormal_beats=0,
                class_counts={c: 0 for c in AAMI_CLASSES},
                abnormal_percentage=0.0,
                predominant_abnormality=None,
                overall_risk="NO DATA",
                results=[],
            )

        total = len(beats)

        # High-performance batch inference
        if self.model_type == "cnn_1d":
            assert isinstance(self.model, ECG1DCNN)
            signals = np.array([b.signal for b in beats], dtype=np.float32)
            tensor_x = torch.from_numpy(signals).unsqueeze(1).to(self.device)
            probs_matrix = self.model.predict_proba(tensor_x).cpu().numpy()
        else:
            assert isinstance(self.model, ClassicalMLModel)
            X_df, _, _ = self.feature_extractor.extract_feature_matrix(beats)
            probs_matrix = self.model.predict_proba(X_df)

        results: List[PredictionResult] = []
        class_counts = {c: 0 for c in AAMI_CLASSES}

        for i, beat in enumerate(beats):
            probs_row = probs_matrix[i]
            pred_idx = int(np.argmax(probs_row))
            pred_class = int_to_aami(pred_idx)
            confidence = float(probs_row[pred_idx])
            probs_dict = {int_to_aami(j): float(p) for j, p in enumerate(probs_row)}

            severity, desc = assess_cardiac_severity(pred_class)
            res = PredictionResult(
                predicted_class=pred_class,
                confidence=confidence,
                probabilities=probs_dict,
                is_abnormal=(pred_class != "N"),
                severity_level=severity,
                clinical_description=desc,
                beat_index=beat.r_peak_index,
                r_peak_sample=beat.r_peak_index,
            )
            results.append(res)
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1

        total = len(beats)
        normal_count = class_counts.get("N", 0)
        abnormal_count = total - normal_count
        abnormal_pct = (abnormal_count / total * 100.0) if total > 0 else 0.0

        # Determine predominant abnormality
        abnormal_only = {k: v for k, v in class_counts.items() if k != "N" and v > 0}
        predom = max(abnormal_only, key=abnormal_only.get) if abnormal_only else None

        # Determine overall patient risk
        if class_counts.get("V", 0) >= 3 or abnormal_pct > 25.0:
            overall_risk = "HIGH ARRHYTHMIA RISK"
        elif abnormal_pct > 5.0 or class_counts.get("S", 0) >= 5:
            overall_risk = "MODERATE CARDIAC RISK"
        else:
            overall_risk = "LOW / NORMAL RISK"

        return BatchPredictionSummary(
            total_beats=total,
            normal_beats=normal_count,
            abnormal_beats=abnormal_count,
            class_counts=class_counts,
            abnormal_percentage=abnormal_pct,
            predominant_abnormality=predom,
            overall_risk=overall_risk,
            results=results,
        )

    def predict_continuous_signal(
        self,
        raw_signal: np.ndarray,
        fs: int = 360,
    ) -> Tuple[np.ndarray, BatchPredictionSummary]:
        """Preprocess continuous ECG signal, detect R-peaks, and classify all beats."""
        clean_sig, beats, r_peaks = self.preprocessing_pipeline.preprocess_unannotated_signal(
            raw_signal, fs=fs
        )
        summary = self.predict_beats(beats)
        return clean_sig, summary
