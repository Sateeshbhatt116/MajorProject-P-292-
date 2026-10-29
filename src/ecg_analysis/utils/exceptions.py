"""Custom exceptions for the ECG Analysis domain."""


class ECGAnalysisError(Exception):
    """Base exception for all domain-specific errors in ECG Cardian."""


class DatasetNotFoundError(ECGAnalysisError):
    """Raised when an ECG record or dataset file is missing or inaccessible."""


class InvalidSignalError(ECGAnalysisError):
    """Raised when an input ECG signal fails validation (empty, NaN, invalid duration/frequency)."""


class PreprocessingError(ECGAnalysisError):
    """Raised when filtering, baseline correction, or segmentation fails."""


class FeatureExtractionError(ECGAnalysisError):
    """Raised when statistical, morphological, or wavelet feature extraction fails."""


class ModelNotFoundError(ECGAnalysisError):
    """Raised when a requested model checkpoint or weights file is not found."""


class ModelInferenceError(ECGAnalysisError):
    """Raised when forward pass or inference fails on an input signal or feature vector."""
