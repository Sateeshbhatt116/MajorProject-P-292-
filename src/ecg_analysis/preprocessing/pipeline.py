"""Unified biomedical signal preprocessing pipeline for ECG Cardian."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config.settings import get_settings
from src.ecg_analysis.data.dataset_loader import ECGRecord
from src.ecg_analysis.preprocessing.baseline_correction import remove_baseline_wander
from src.ecg_analysis.preprocessing.filtering import bandpass_filter, notch_filter
from src.ecg_analysis.preprocessing.normalization import normalize_signal
from src.ecg_analysis.preprocessing.segmentation import (
    ECGBeat,
    segment_beats_from_annotations,
    segment_beats_unannotated,
)
from src.ecg_analysis.utils.logging_config import setup_logger

logger = setup_logger("ecg_analysis.preprocessing.pipeline")


@dataclass
class PreprocessingConfig:
    """Configurable hyperparameters for biomedical signal preprocessing."""

    notch_enabled: bool = True
    notch_freq: float = 60.0
    notch_quality_factor: float = 30.0

    bandpass_enabled: bool = True
    lowcut: float = 0.5
    highcut: float = 45.0
    bandpass_order: int = 3

    baseline_correction_method: str = "median"  # 'median', 'wavelet', or 'none'

    normalization_method: str = "zscore"  # 'zscore', 'minmax', 'robust', or 'none'

    pre_r_samples: int = 90  # ~250 ms @ 360 Hz
    post_r_samples: int = 160  # ~444 ms @ 360 Hz

    normalize_per_beat: bool = True

    @property
    def beat_length(self) -> int:
        return self.pre_r_samples + self.post_r_samples


class ECGPreprocessingPipeline:
    """Production-grade biomedical signal processing pipeline for ECG signals."""

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        if config is None:
            settings = get_settings()
            config = PreprocessingConfig(
                lowcut=settings.bandpass_lowcut,
                highcut=settings.bandpass_highcut,
                bandpass_order=settings.bandpass_order,
                notch_freq=settings.notch_frequency,
                notch_quality_factor=settings.notch_quality_factor,
                pre_r_samples=settings.pre_r_samples,
                post_r_samples=settings.post_r_samples,
            )
        self.config = config

    def process_continuous_signal(
        self,
        raw_signal: np.ndarray,
        fs: float = 360.0,
    ) -> np.ndarray:
        """Apply sequential biomedical filters to continuous ECG signal.

        Order of execution:
        1. Powerline notch filter (50/60 Hz)
        2. Zero-phase Butterworth bandpass filter (0.5 - 45 Hz)
        3. Baseline wander removal (cascaded median or DWT)
        4. Optional continuous normalization
        """
        signal_clean = np.asarray(raw_signal, dtype=np.float64).copy()

        # Step 1: Notch filter
        if self.config.notch_enabled:
            signal_clean = notch_filter(
                signal_clean,
                notch_freq=self.config.notch_freq,
                quality_factor=self.config.notch_quality_factor,
                fs=fs,
            )

        # Step 2: Bandpass filter
        if self.config.bandpass_enabled:
            signal_clean = bandpass_filter(
                signal_clean,
                lowcut=self.config.lowcut,
                highcut=self.config.highcut,
                fs=fs,
                order=self.config.bandpass_order,
            )

        # Step 3: Baseline drift suppression
        if self.config.baseline_correction_method in ("median", "wavelet"):
            signal_clean = remove_baseline_wander(
                signal_clean,
                fs=fs,
                method=self.config.baseline_correction_method,
            )

        # Step 4: Optional global normalization
        if not self.config.normalize_per_beat and self.config.normalization_method != "none":
            signal_clean = normalize_signal(
                signal_clean,
                method=self.config.normalization_method,
            )

        return signal_clean

    def preprocess_record(
        self,
        record: ECGRecord,
        lead_name: Optional[str] = "MLII",
    ) -> Tuple[np.ndarray, List[ECGBeat]]:
        """Preprocess an annotated ECGRecord and extract standardized beats.

        Args:
            record: ECGRecord instance.
            lead_name: Lead to extract and process (defaults to 'MLII').

        Returns:
            Tuple of (clean_continuous_lead_signal, list_of_segmented_beats).
        """
        raw_lead = record.get_lead_signal(lead_name)
        fs = float(record.sampling_rate)

        clean_signal = self.process_continuous_signal(raw_lead, fs=fs)

        beats = segment_beats_from_annotations(
            ecg_signal=clean_signal,
            annotation_samples=record.annotation_samples,
            annotation_symbols=record.annotation_symbols,
            fs=record.sampling_rate,
            pre_r_samples=self.config.pre_r_samples,
            post_r_samples=self.config.post_r_samples,
            lead_name=lead_name or record.lead_names[0],
            record_name=record.record_name,
        )

        # Apply per-beat normalization if configured
        if self.config.normalize_per_beat and self.config.normalization_method != "none":
            normalized_beats: List[ECGBeat] = []
            for b in beats:
                norm_sig = normalize_signal(b.signal, method=self.config.normalization_method)
                norm_beat = ECGBeat(
                    signal=norm_sig,
                    r_peak_index=b.r_peak_index,
                    r_peak_rel_index=b.r_peak_rel_index,
                    aami_label=b.aami_label,
                    numeric_label=b.numeric_label,
                    lead_name=b.lead_name,
                    sampling_rate=b.sampling_rate,
                    record_name=b.record_name,
                )
                normalized_beats.append(norm_beat)
            beats = normalized_beats

        logger.info(
            f"Preprocessed Record {record.record_name} (Lead {lead_name}): extracted {len(beats)} beats."
        )
        return clean_signal, beats

    def preprocess_unannotated_signal(
        self,
        raw_signal: np.ndarray,
        fs: int = 360,
        lead_name: str = "Lead",
        record_name: str = "sample",
    ) -> Tuple[np.ndarray, List[ECGBeat], np.ndarray]:
        """Preprocess raw signal without annotations using Pan-Tompkins QRS detector.

        Returns:
            Tuple of (clean_signal, segmented_beats, detected_r_peaks).
        """
        clean_signal = self.process_continuous_signal(raw_signal, fs=float(fs))

        beats, r_peaks = segment_beats_unannotated(
            ecg_signal=clean_signal,
            fs=fs,
            pre_r_samples=self.config.pre_r_samples,
            post_r_samples=self.config.post_r_samples,
            lead_name=lead_name,
            record_name=record_name,
        )

        if self.config.normalize_per_beat and self.config.normalization_method != "none":
            normalized_beats: List[ECGBeat] = []
            for b in beats:
                norm_sig = normalize_signal(b.signal, method=self.config.normalization_method)
                norm_beat = ECGBeat(
                    signal=norm_sig,
                    r_peak_index=b.r_peak_index,
                    r_peak_rel_index=b.r_peak_rel_index,
                    aami_label=b.aami_label,
                    numeric_label=b.numeric_label,
                    lead_name=b.lead_name,
                    sampling_rate=b.sampling_rate,
                    record_name=b.record_name,
                )
                normalized_beats.append(norm_beat)
            beats = normalized_beats

        return clean_signal, beats, r_peaks
