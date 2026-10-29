"""WFDB ECG dataset loader for PhysioNet databases (MIT-BIH Arrhythmia)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import wfdb

from config.settings import get_settings
from src.ecg_analysis.data.label_mapping import map_symbol_to_aami
from src.ecg_analysis.utils.exceptions import (
    DatasetNotFoundError,
    InvalidSignalError,
)
from src.ecg_analysis.utils.logging_config import setup_logger

logger = setup_logger("ecg_analysis.data.dataset_loader")


@dataclass
class ECGRecord:
    """Represents a loaded multi-lead electrocardiogram recording and annotations."""

    record_name: str
    sampling_rate: int
    lead_names: List[str]
    signals: np.ndarray  # Shape: (N_samples, N_leads), physical units (mV)
    annotation_samples: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    annotation_symbols: List[str] = field(default_factory=list)
    aami_labels: List[str] = field(default_factory=list)
    comments: List[str] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        """Total duration of recording in seconds."""
        if self.sampling_rate <= 0 or len(self.signals) == 0:
            return 0.0
        return len(self.signals) / self.sampling_rate

    @property
    def num_samples(self) -> int:
        return len(self.signals)

    @property
    def num_leads(self) -> int:
        return len(self.lead_names)

    def get_lead_signal(self, lead_name: Optional[str] = None) -> np.ndarray:
        """Extract a single-lead 1D continuous signal array.

        Args:
            lead_name: Specific lead name (e.g. 'MLII', 'V1'). If None or not found,
                       returns the primary (first) lead.

        Returns:
            1D numpy array of signal amplitudes in mV.
        """
        if self.signals.ndim == 1:
            return self.signals
        if lead_name is not None and lead_name in self.lead_names:
            idx = self.lead_names.index(lead_name)
            return self.signals[:, idx]
        # Default to lead 0
        return self.signals[:, 0]

    def get_valid_beat_annotations(self) -> Tuple[np.ndarray, List[str]]:
        """Filter annotations to include only valid cardiac beats and their AAMI classes.

        Returns:
            Tuple of (sample_indices, aami_class_labels).
        """
        valid_indices: List[int] = []
        valid_labels: List[str] = []

        for idx, sym in zip(self.annotation_samples, self.annotation_symbols):
            aami = map_symbol_to_aami(sym)
            if aami is not None:
                valid_indices.append(int(idx))
                valid_labels.append(aami)

        return np.array(valid_indices, dtype=int), valid_labels


class ECGDataLoader:
    """Robust loader and manager for PhysioNet ECG datasets."""

    def __init__(self, raw_data_dir: Optional[Union[Path, str]] = None) -> None:
        self.raw_data_dir = (
            Path(raw_data_dir).resolve() if raw_data_dir else get_settings().data_raw_dir
        )
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)

    def record_exists(self, record_name: str) -> bool:
        """Check if header file for record exists locally."""
        hea_file = self.raw_data_dir / f"{record_name}.hea"
        return hea_file.is_file()

    def list_available_records(self) -> List[str]:
        """List all available record identifiers in the raw data directory."""
        records = sorted(
            [f.stem for f in self.raw_data_dir.glob("*.hea") if f.is_file() and not f.stem.startswith(".")]
        )
        return records

    def load_record(
        self,
        record_name: str,
        sampfrom: int = 0,
        sampto: Optional[int] = None,
        download_if_missing: bool = False,
    ) -> ECGRecord:
        """Load an ECG record and its annotations using WFDB.

        Args:
            record_name: Record ID (e.g., '100', '101', '200').
            sampfrom: Starting sample index (default: 0).
            sampto: Ending sample index (default: None for full signal).
            download_if_missing: If True and record is missing, download from PhysioNet.

        Returns:
            ECGRecord object containing signals, leads, and mapped annotations.
        """
        clean_name = Path(record_name).stem
        record_path = self.raw_data_dir / clean_name

        if not self.record_exists(clean_name):
            if download_if_missing:
                logger.info(f"Record {clean_name} not found locally. Downloading from PhysioNet mitdb...")
                try:
                    wfdb.dl_database("mitdb", dl_dir=str(self.raw_data_dir), records=[clean_name])
                except Exception as exc:
                    raise DatasetNotFoundError(
                        f"Failed to download record {clean_name} from PhysioNet: {exc}"
                    ) from exc
            else:
                raise DatasetNotFoundError(
                    f"Record '{clean_name}' not found in {self.raw_data_dir}. "
                    f"Please run the download script or pass download_if_missing=True."
                )

        try:
            # Read WFDB signal record
            record = wfdb.rdrecord(
                record_name=str(record_path),
                sampfrom=sampfrom,
                sampto=sampto,
                physical=True,  # Convert to physical mV units
            )
        except Exception as exc:
            raise InvalidSignalError(f"Error reading signal for record {clean_name}: {exc}") from exc

        if record.p_signal is None or len(record.p_signal) == 0:
            raise InvalidSignalError(f"Signal array for record {clean_name} is empty.")

        # Read annotations if .atr file exists
        ann_samples: np.ndarray = np.array([], dtype=int)
        ann_symbols: List[str] = []
        aami_labels: List[str] = []

        atr_path = self.raw_data_dir / f"{clean_name}.atr"
        if atr_path.is_file():
            try:
                annotation = wfdb.rdann(
                    record_name=str(record_path),
                    extension="atr",
                    sampfrom=sampfrom,
                    sampto=sampto,
                )
                ann_samples = annotation.sample
                ann_symbols = [str(s) for s in annotation.symbol]
                aami_labels = [map_symbol_to_aami(s) or "NON_BEAT" for s in ann_symbols]
            except Exception as exc:
                logger.warning(f"Could not load annotations for {clean_name}: {exc}")

        return ECGRecord(
            record_name=clean_name,
            sampling_rate=int(record.fs),
            lead_names=list(record.sig_name),
            signals=record.p_signal,
            annotation_samples=ann_samples,
            annotation_symbols=ann_symbols,
            aami_labels=aami_labels,
            comments=list(getattr(record, "comments", [])),
        )
