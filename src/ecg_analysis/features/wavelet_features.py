"""Discrete Wavelet Transform (DWT) multi-resolution feature extraction."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pywt

from src.ecg_analysis.utils.exceptions import FeatureExtractionError


def extract_wavelet_features(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 4,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """Extract multi-resolution DWT features from an ECG heartbeat.

    Decomposes beat using Daubechies 4 (db4) wavelet, which closely mirrors
    the shape of human QRS complexes. Extracts sub-band energies, relative
    wavelet energy, and Shannon wavelet entropy.

    Args:
        signal: 1D array of single segmented beat.
        wavelet: Wavelet family (default: 'db4').
        level: Decomposition level (default: 4).
        eps: Small constant for numerical stability.

    Returns:
        Dictionary of wavelet sub-band energies and entropy features.
    """
    if signal is None or len(signal) == 0:
        raise FeatureExtractionError("Signal array is empty or None")

    sig = np.asarray(signal, dtype=np.float64).ravel()

    try:
        max_level = pywt.dwt_max_level(data_len=len(sig), filter_len=pywt.Wavelet(wavelet).dec_len)
        actual_level = min(level, max_level)

        # coeffs: [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        coeffs = pywt.wavedec(sig, wavelet=wavelet, level=actual_level)
    except Exception as exc:
        raise FeatureExtractionError(f"DWT decomposition failed: {exc}") from exc

    features: Dict[str, float] = {}
    subband_energies: list[float] = []

    # Process approximation coefficients cA_level
    ca = coeffs[0]
    ca_energy = float(np.sum(ca**2))
    subband_energies.append(ca_energy)
    features[f"dwt_cA{actual_level}_energy"] = ca_energy
    features[f"dwt_cA{actual_level}_mean"] = float(np.mean(np.abs(ca)))
    features[f"dwt_cA{actual_level}_std"] = float(np.std(ca))

    # Process detail coefficients cD_i
    for i, cd in enumerate(coeffs[1:], 1):
        band_num = actual_level - i + 1
        cd_energy = float(np.sum(cd**2))
        subband_energies.append(cd_energy)
        features[f"dwt_cD{band_num}_energy"] = cd_energy
        features[f"dwt_cD{band_num}_mean"] = float(np.mean(np.abs(cd)))
        features[f"dwt_cD{band_num}_std"] = float(np.std(cd))

    # Total Wavelet Energy and Relative Wavelet Energy
    total_energy = float(np.sum(subband_energies))
    features["dwt_total_energy"] = total_energy

    # Wavelet Entropy: H = -sum(p_i * log2(p_i))
    entropy = 0.0
    for idx, e in enumerate(subband_energies):
        p_i = e / (total_energy + eps)
        features[f"dwt_rel_energy_{idx}"] = float(p_i)
        if p_i > eps:
            entropy -= p_i * np.log2(p_i)

    features["dwt_wavelet_entropy"] = float(entropy)

    return features
