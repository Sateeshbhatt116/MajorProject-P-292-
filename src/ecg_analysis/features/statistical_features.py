"""Time-domain statistical and morphological feature extraction for ECG heartbeats."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats

from src.ecg_analysis.utils.exceptions import FeatureExtractionError


def extract_statistical_features(
    signal: np.ndarray,
    r_peak_rel_idx: int = 90,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Extract comprehensive time-domain statistical and morphological features.

    Features extracted:
    - Mean, Standard Deviation, Variance
    - Skewness (asymmetry) & Kurtosis (peakedness)
    - Minimum, Maximum, Peak-to-Peak Amplitude
    - Root Mean Square (RMS) & Signal Energy
    - Zero Crossing Rate (ZCR)
    - Pre-R vs Post-R energy ratio
    - Local QRS slopes (left slope, right slope)

    Args:
        signal: 1D array of a single segmented beat.
        r_peak_rel_idx: Sample index of the R-peak within the beat window.
        eps: Small constant for numerical stability.

    Returns:
        Dictionary of named scalar features.
    """
    if signal is None or len(signal) == 0:
        raise FeatureExtractionError("Beat signal array is empty or None")

    sig = np.asarray(signal, dtype=np.float64).ravel()
    n = len(sig)

    # 1. Basic moments
    mean_val = float(np.mean(sig))
    std_val = float(np.std(sig))
    var_val = float(np.var(sig))
    min_val = float(np.min(sig))
    max_val = float(np.max(sig))
    ptp_val = max_val - min_val

    # 2. Higher-order moments
    skew_val = float(stats.skew(sig)) if std_val > eps else 0.0
    kurt_val = float(stats.kurtosis(sig)) if std_val > eps else 0.0

    # 3. Energy & Power
    energy = float(np.sum(sig**2))
    rms_val = float(np.sqrt(np.mean(sig**2)))

    # 4. Zero-crossing rate
    zero_crossings = np.sum(np.diff(np.signbit(sig - mean_val)).astype(bool))
    zcr = float(zero_crossings) / float(n - 1) if n > 1 else 0.0

    # 5. Morphological asymmetry around R-peak
    r_idx = min(max(0, r_peak_rel_idx), n - 1)
    r_amp = float(sig[r_idx])

    pre_r = sig[:r_idx]
    post_r = sig[r_idx:]

    pre_energy = float(np.sum(pre_r**2)) if len(pre_r) > 0 else 0.0
    post_energy = float(np.sum(post_r**2)) if len(post_r) > 0 else 0.0
    energy_ratio = pre_energy / (post_energy + eps)

    # 6. QRS slope estimation (±15 samples around R-peak)
    left_start = max(0, r_idx - 15)
    right_end = min(n, r_idx + 15)

    left_slope = (r_amp - sig[left_start]) / float(r_idx - left_start + eps)
    right_slope = (sig[right_end - 1] - r_amp) / float(right_end - r_idx + eps)

    return {
        "stat_mean": mean_val,
        "stat_std": std_val,
        "stat_var": var_val,
        "stat_min": min_val,
        "stat_max": max_val,
        "stat_ptp": ptp_val,
        "stat_skewness": skew_val,
        "stat_kurtosis": kurt_val,
        "stat_energy": energy,
        "stat_rms": rms_val,
        "stat_zcr": zcr,
        "stat_r_amplitude": r_amp,
        "stat_pre_energy": pre_energy,
        "stat_post_energy": post_energy,
        "stat_energy_ratio": energy_ratio,
        "stat_left_slope": float(left_slope),
        "stat_right_slope": float(right_slope),
    }
