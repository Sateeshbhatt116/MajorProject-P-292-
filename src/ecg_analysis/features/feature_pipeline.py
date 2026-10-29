"""Combined feature extraction pipeline for machine learning classification."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ecg_analysis.features.statistical_features import extract_statistical_features
from src.ecg_analysis.features.wavelet_features import extract_wavelet_features
from src.ecg_analysis.preprocessing.segmentation import ECGBeat
from src.ecg_analysis.utils.logging_config import setup_logger

logger = setup_logger("ecg_analysis.features.feature_pipeline")


class ECGFeatureExtractor:
    """Extracts hybrid statistical and wavelet feature vectors from segmented ECG beats."""

    def __init__(
        self,
        wavelet: str = "db4",
        wavelet_level: int = 4,
        level: Optional[int] = None,
        include_raw_samples: bool = False,
    ) -> None:
        self.wavelet = wavelet
        self.wavelet_level = level if level is not None else wavelet_level
        self.include_raw_samples = include_raw_samples
        self._feature_names: Optional[List[str]] = None

    @property
    def feature_names(self) -> List[str]:
        """Return cached list of feature column names."""
        if self._feature_names is None:
            # Generate dummy beat to deduce feature names
            dummy_sig = np.zeros(250)
            feat_dict = self.extract_beat_features_dict(dummy_sig, r_peak_rel_idx=90)
            self._feature_names = list(feat_dict.keys())
        return self._feature_names

    def extract_beat_features_dict(
        self,
        beat_signal: np.ndarray,
        r_peak_rel_idx: int = 90,
    ) -> dict[str, float]:
        """Extract all hybrid features from a single beat signal as a dictionary."""
        stat_feats = extract_statistical_features(beat_signal, r_peak_rel_idx=r_peak_rel_idx)
        wav_feats = extract_wavelet_features(
            beat_signal, wavelet=self.wavelet, level=self.wavelet_level
        )

        combined: dict[str, float] = {**stat_feats, **wav_feats}

        if self.include_raw_samples:
            for idx, val in enumerate(beat_signal):
                combined[f"raw_sample_{idx}"] = float(val)

        return combined

    def extract_from_beat(self, beat: ECGBeat) -> dict[str, float]:
        """Extract features from an ECGBeat dataclass."""
        return self.extract_beat_features_dict(
            beat.signal, r_peak_rel_idx=beat.r_peak_rel_index
        )

    def extract_feature_matrix(
        self,
        beats: List[ECGBeat],
    ) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """Transform a list of ECGBeats into tabular feature matrix X and target labels y.

        Args:
            beats: List of ECGBeat objects.

        Returns:
            Tuple of (X_df: pd.DataFrame, y_numeric: np.ndarray, y_aami: List[str]).
        """
        if not beats:
            raise ValueError("Empty beats list provided for feature extraction.")

        rows: List[dict[str, float]] = []
        y_numeric: List[int] = []
        y_aami: List[str] = []

        for beat in beats:
            feat_dict = self.extract_from_beat(beat)
            rows.append(feat_dict)
            y_numeric.append(beat.numeric_label)
            y_aami.append(beat.aami_label)

        X_df = pd.DataFrame(rows)
        self._feature_names = list(X_df.columns)

        logger.info(
            f"Extracted feature matrix: {X_df.shape[0]} beats x {X_df.shape[1]} features."
        )
        return X_df, np.array(y_numeric, dtype=int), y_aami
