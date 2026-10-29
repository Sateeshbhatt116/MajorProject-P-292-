"""Integrity, format, and clinical sanity validator for ECG signals and annotations."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.ecg_analysis.data.dataset_loader import ECGDataLoader, ECGRecord
from src.ecg_analysis.utils.logging_config import setup_logger

logger = setup_logger("ecg_analysis.data.dataset_validator")


@dataclass
class ValidationResult:
    """Detailed validation diagnostics for an ECG recording."""

    record_name: str
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    sampling_rate: int = 0
    duration_sec: float = 0.0
    num_samples: int = 0
    num_leads: int = 0
    num_valid_beats: int = 0
    lead_names: List[str] = field(default_factory=list)
    amplitude_range_mv: Dict[str, tuple[float, float]] = field(default_factory=dict)
    class_distribution: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASSED" if self.is_valid else "FAILED"
        summary_lines = [
            f"Validation [{status}] for Record '{self.record_name}':",
            f"  - Duration: {self.duration_sec:.2f} s ({self.num_samples} samples @ {self.sampling_rate} Hz)",
            f"  - Leads: {', '.join(self.lead_names)}",
            f"  - Valid Beats: {self.num_valid_beats}",
            f"  - Class Dist: {dict(self.class_distribution)}",
        ]
        if self.issues:
            summary_lines.append(f"  - Critical Issues: {'; '.join(self.issues)}")
        if self.warnings:
            summary_lines.append(f"  - Warnings: {'; '.join(self.warnings)}")
        return "\n".join(summary_lines)


class DatasetValidator:
    """Validates ECG records against biomedical and technical constraints."""

    def __init__(
        self,
        expected_fs: int = 360,
        max_amplitude_mv: float = 10.0,
        min_duration_sec: float = 5.0,
    ) -> None:
        self.expected_fs = expected_fs
        self.max_amplitude_mv = max_amplitude_mv
        self.min_duration_sec = min_duration_sec

    def validate_record(self, record: ECGRecord) -> ValidationResult:
        """Execute validation checks on an ECGRecord instance."""
        issues: List[str] = []
        warnings: List[str] = []

        # 1. Check signal existence and dimensionality
        if record.signals is None or len(record.signals) == 0:
            issues.append("Signal array is empty or None")
            return ValidationResult(
                record_name=record.record_name,
                is_valid=False,
                issues=issues,
            )

        num_samples = len(record.signals)
        sampling_rate = record.sampling_rate

        # 2. Check sampling frequency
        if sampling_rate <= 0:
            issues.append(f"Invalid sampling rate: {sampling_rate} Hz")
        elif sampling_rate != self.expected_fs:
            warnings.append(
                f"Sampling rate is {sampling_rate} Hz (expected default: {self.expected_fs} Hz)"
            )

        duration_sec = num_samples / sampling_rate if sampling_rate > 0 else 0.0
        if duration_sec < self.min_duration_sec:
            issues.append(
                f"Signal duration ({duration_sec:.2f}s) is shorter than minimum required ({self.min_duration_sec}s)"
            )

        # 3. Check for NaNs or Infinities
        if np.isnan(record.signals).any():
            issues.append("Signal contains NaN (Not a Number) values")
        if np.isinf(record.signals).any():
            issues.append("Signal contains infinite values")

        # 4. Check amplitude range and flatline
        amplitude_ranges: Dict[str, tuple[float, float]] = {}
        for lead_idx, lead_name in enumerate(record.lead_names):
            lead_sig = record.signals[:, lead_idx] if record.signals.ndim > 1 else record.signals
            sig_min = float(np.min(lead_sig))
            sig_max = float(np.max(lead_sig))
            amplitude_ranges[lead_name] = (sig_min, sig_max)

            # Flatline check (zero variance)
            if float(np.var(lead_sig)) < 1e-6:
                issues.append(f"Lead '{lead_name}' appears to be a flatline / disconnected lead")

            # Extreme saturation
            if abs(sig_min) > self.max_amplitude_mv or abs(sig_max) > self.max_amplitude_mv:
                warnings.append(
                    f"Lead '{lead_name}' amplitude ({sig_min:.2f} to {sig_max:.2f} mV) exceeds standard range (±{self.max_amplitude_mv} mV)"
                )

        # 5. Check annotations
        beat_indices, aami_labels = record.get_valid_beat_annotations()
        num_valid_beats = len(beat_indices)

        if len(record.annotation_samples) == 0:
            warnings.append("No annotation markers found for this record")
        else:
            # Check annotation bounds
            out_of_bounds = (record.annotation_samples < 0) | (record.annotation_samples >= num_samples)
            if np.any(out_of_bounds):
                issues.append(
                    f"{np.sum(out_of_bounds)} annotation markers are outside signal sample range [0, {num_samples-1}]"
                )

        class_distribution = dict(Counter(aami_labels))

        is_valid = len(issues) == 0
        return ValidationResult(
            record_name=record.record_name,
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            sampling_rate=sampling_rate,
            duration_sec=duration_sec,
            num_samples=num_samples,
            num_leads=len(record.lead_names),
            num_valid_beats=num_valid_beats,
            lead_names=record.lead_names,
            amplitude_range_mv=amplitude_ranges,
            class_distribution=class_distribution,
        )

    def validate_dataset(self, data_dir: Optional[Path] = None) -> List[ValidationResult]:
        """Validate all available records in a directory."""
        loader = ECGDataLoader(raw_data_dir=data_dir)
        records = loader.list_available_records()
        results: List[ValidationResult] = []

        if not records:
            logger.warning(f"No records found to validate in {loader.raw_data_dir}")
            return results

        for rec_name in records:
            try:
                rec = loader.load_record(rec_name)
                res = self.validate_record(rec)
                results.append(res)
                logger.info(res.summary())
            except Exception as exc:
                logger.error(f"Failed to load/validate record {rec_name}: {exc}")
                results.append(
                    ValidationResult(
                        record_name=rec_name,
                        is_valid=False,
                        issues=[f"Loading failed: {str(exc)}"],
                    )
                )

        return results
