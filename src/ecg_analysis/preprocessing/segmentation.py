"""Beat segmentation and automated QRS / R-peak detection.

Implements:
1. Realization of the Pan-Tompkins QRS detection algorithm (1985).
2. Windowed heartbeat extraction around R-peaks (90 samples pre-R, 160 post-R @ 360 Hz).
3. Data structures for segmented beats with AAMI labels and clinical metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import signal

from src.ecg_analysis.data.label_mapping import (
    aami_to_int,
    map_symbol_to_aami,
)
from src.ecg_analysis.utils.exceptions import PreprocessingError
from src.ecg_analysis.utils.logging_config import setup_logger

logger = setup_logger("ecg_analysis.preprocessing.segmentation")


@dataclass
class ECGBeat:
    """Represents a single windowed heartbeat segment."""

    signal: np.ndarray  # 1D array of shape (beat_length,)
    r_peak_index: int  # Absolute sample index in continuous recording
    r_peak_rel_index: int  # Relative sample index within the beat window
    aami_label: str  # 'N', 'S', 'V', 'F', 'Q' or 'UNKNOWN'
    numeric_label: int  # 0, 1, 2, 3, 4 or -1
    lead_name: str = "MLII"
    sampling_rate: int = 360
    record_name: str = "unknown"

    @property
    def length(self) -> int:
        return len(self.signal)


def pan_tompkins_qrs_detector(
    ecg_signal: np.ndarray,
    fs: float = 360.0,
) -> np.ndarray:
    """Automated R-peak detection using the classic Pan-Tompkins algorithm (1985).

    Pipeline:
    1. Bandpass filter (5 - 15 Hz) to maximize QRS energy and minimize T-wave/muscle noise.
    2. Five-point derivative to obtain slope information.
    3. Non-linear squaring operation to amplify high-frequency components.
    4. Moving window integration (~150 ms) to extract waveform envelope.
    5. Adaptive thresholding with physiological refractory period (200 ms).

    Args:
        ecg_signal: 1D array of raw ECG amplitudes.
        fs: Sampling frequency in Hz.

    Returns:
        1D integer array of detected R-peak sample indices.
    """
    if ecg_signal is None or len(ecg_signal) == 0:
        raise PreprocessingError("Signal is empty or None")

    sig = np.asarray(ecg_signal, dtype=np.float64).ravel()
    num_samples = len(sig)

    # 1. Bandpass filter: 5 - 15 Hz
    nyq = 0.5 * fs
    low = 5.0 / nyq
    high = 15.0 / nyq
    b, a = signal.butter(2, [low, high], btype="bandpass")
    filtered = signal.filtfilt(b, a, sig)

    # 2. Derivative filter: y[n] = (1/8) * (2*x[n] + x[n-1] - x[n-3] - 2*x[n-4])
    diff_kernel = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) * (1.0 / 8.0) * fs
    differentiated = np.convolve(filtered, diff_kernel, mode="same")

    # 3. Squaring: non-linear rectification
    squared = differentiated**2

    # 4. Moving window integration: ~150 ms
    window_width = int(round(0.150 * fs))
    if window_width < 1:
        window_width = 1
    mwi_kernel = np.ones(window_width) / float(window_width)
    integrated = np.convolve(squared, mwi_kernel, mode="same")

    # 5. Peak finding on integrated signal with refractory period (200 ms)
    min_distance = int(round(0.200 * fs))
    # Adaptive threshold: 30% of 90th percentile to ignore extreme pacing spikes
    thresh = 0.25 * np.percentile(integrated, 90)

    peaks, _ = signal.find_peaks(integrated, height=thresh, distance=min_distance)

    # 6. Refine R-peak location on original signal by searching local max in ±50 ms window
    search_half_window = int(round(0.060 * fs))
    refined_peaks: List[int] = []

    for p in peaks:
        start = max(0, p - search_half_window)
        end = min(num_samples, p + search_half_window + 1)
        # Search for highest absolute amplitude or max peak
        local_segment = np.abs(filtered[start:end])
        if len(local_segment) > 0:
            local_max_idx = int(np.argmax(local_segment))
            refined_peaks.append(start + local_max_idx)

    return np.unique(np.array(refined_peaks, dtype=int))


def segment_beats_from_annotations(
    ecg_signal: np.ndarray,
    annotation_samples: np.ndarray,
    annotation_symbols: List[str],
    fs: int = 360,
    pre_r_samples: int = 90,
    post_r_samples: int = 160,
    lead_name: str = "MLII",
    record_name: str = "unknown",
) -> List[ECGBeat]:
    """Segment ECG signal into fixed-length windows around annotated R-peaks.

    Following biomedical literature (e.g., de Chazal et al.):
    - pre_r_samples = 90 (~250 ms at 360 Hz): covers P-wave and PR interval
    - post_r_samples = 160 (~444 ms at 360 Hz): covers ST-segment and T-wave
    - Total beat length = 250 samples (~694 ms)

    Args:
        ecg_signal: 1D array of ECG signal amplitudes.
        annotation_samples: Sample indices of R-peaks.
        annotation_symbols: WFDB annotation symbols.
        fs: Sampling rate.
        pre_r_samples: Samples before R-peak.
        post_r_samples: Samples after R-peak.
        lead_name: ECG lead name.
        record_name: Source record ID.

    Returns:
        List of ECGBeat instances.
    """
    beats: List[ECGBeat] = []
    sig = np.asarray(ecg_signal, dtype=np.float64).ravel()
    num_samples = len(sig)

    for r_idx, sym in zip(annotation_samples, annotation_symbols):
        aami_cls = map_symbol_to_aami(sym)
        if aami_cls is None:
            # Skip non-beat annotations (e.g. rhythm markers '+', '~')
            continue

        start = int(r_idx) - pre_r_samples
        end = int(r_idx) + post_r_samples

        # Exclude beats cut off by recording boundaries
        if start < 0 or end > num_samples:
            continue

        beat_segment = sig[start:end]
        if len(beat_segment) != (pre_r_samples + post_r_samples):
            continue

        numeric_lbl = aami_to_int(aami_cls)

        beat = ECGBeat(
            signal=beat_segment,
            r_peak_index=int(r_idx),
            r_peak_rel_index=pre_r_samples,
            aami_label=aami_cls,
            numeric_label=numeric_lbl,
            lead_name=lead_name,
            sampling_rate=fs,
            record_name=record_name,
        )
        beats.append(beat)

    return beats


def segment_beats_unannotated(
    ecg_signal: np.ndarray,
    fs: int = 360,
    pre_r_samples: int = 90,
    post_r_samples: int = 160,
    lead_name: str = "Lead",
    record_name: str = "unannotated",
) -> Tuple[List[ECGBeat], np.ndarray]:
    """Segment unannotated ECG signal using automated Pan-Tompkins R-peak detector.

    Returns:
        Tuple of (list_of_beats, detected_r_peak_indices).
    """
    sig = np.asarray(ecg_signal, dtype=np.float64).ravel()
    num_samples = len(sig)
    r_peaks = pan_tompkins_qrs_detector(sig, fs=float(fs))

    beats: List[ECGBeat] = []
    valid_r_peaks: List[int] = []

    for r_idx in r_peaks:
        start = r_idx - pre_r_samples
        end = r_idx + post_r_samples

        if start < 0 or end > num_samples:
            continue

        beat_segment = sig[start:end]
        if len(beat_segment) != (pre_r_samples + post_r_samples):
            continue

        valid_r_peaks.append(r_idx)
        beat = ECGBeat(
            signal=beat_segment,
            r_peak_index=int(r_idx),
            r_peak_rel_index=pre_r_samples,
            aami_label="UNKNOWN",
            numeric_label=-1,
            lead_name=lead_name,
            sampling_rate=fs,
            record_name=record_name,
        )
        beats.append(beat)

    return beats, np.array(valid_r_peaks, dtype=int)
