"""Data ingestion, validation, and label mapping package."""
from src.ecg_analysis.data.dataset_loader import ECGDataLoader, ECGRecord
from src.ecg_analysis.data.dataset_validator import DatasetValidator, ValidationResult
from src.ecg_analysis.data.label_mapping import (
    AAMI_CLASSES,
    AAMI_CLASS_COLORS,
    AAMI_CLASS_DESCRIPTIONS,
    AAMI_CLASS_TO_INT,
    INT_TO_AAMI_CLASS,
    MITBIH_TO_AAMI,
    NON_BEAT_SYMBOLS,
    aami_to_int,
    get_class_description,
    int_to_aami,
    is_valid_beat,
    map_symbol_to_aami,
)

__all__ = [
    "ECGDataLoader",
    "ECGRecord",
    "DatasetValidator",
    "ValidationResult",
    "AAMI_CLASSES",
    "AAMI_CLASS_COLORS",
    "AAMI_CLASS_DESCRIPTIONS",
    "AAMI_CLASS_TO_INT",
    "INT_TO_AAMI_CLASS",
    "MITBIH_TO_AAMI",
    "NON_BEAT_SYMBOLS",
    "map_symbol_to_aami",
    "is_valid_beat",
    "aami_to_int",
    "int_to_aami",
    "get_class_description",
]
