"""ANSI/AAMI EC57 standard heartbeat classification and label mapping.

The Association for the Advancement of Medical Instrumentation (AAMI) standard
recommends grouping MIT-BIH arrhythmia database beat types into 5 primary classes:
- N: Normal beat (Normal sinus, LBBB, RBBB, nodal/atrial escape)
- S: Supraventricular ectopic beat (SVEB: APC, Aberrant APC, Nodal premature)
- V: Ventricular ectopic beat (VEB: PVC, Ventricular escape, R-on-T)
- F: Fusion beat (Ventricular + Normal)
- Q: Unknown / Paced / Unclassifiable beat
"""
from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

# Mapping from PhysioNet MIT-BIH symbol to AAMI superclass
MITBIH_TO_AAMI: Dict[str, str] = {
    # N Class: Normal and Bundle Branch Block beats
    "N": "N",  # Normal beat
    "L": "N",  # Left bundle branch block beat
    "R": "N",  # Right bundle branch block beat
    "B": "N",  # Bundle branch block beat (unspecified)
    "e": "N",  # Atrial escape beat
    "j": "N",  # Nodal (junctional) escape beat
    # S Class: Supraventricular Ectopic Beats (SVEB)
    "A": "S",  # Atrial premature beat
    "a": "S",  # Aberrated atrial premature beat
    "J": "S",  # Nodal (junctional) premature beat
    "S": "S",  # Supraventricular premature or ectopic beat
    # V Class: Ventricular Ectopic Beats (VEB)
    "V": "V",  # Premature ventricular contraction
    "E": "V",  # Ventricular escape beat
    "r": "V",  # R-on-T premature ventricular contraction
    # F Class: Fusion Beats
    "F": "F",  # Fusion of ventricular and normal beat
    # Q Class: Paced, Unknown, Unclassifiable beats
    "/": "Q",  # Paced beat
    "f": "Q",  # Fusion of paced and normal beat
    "Q": "Q",  # Unclassifiable beat
    "?": "Q",  # Beat not classified
}

# Non-beat annotation symbols in MIT-BIH (rhythm changes, artifacts, comments)
NON_BEAT_SYMBOLS: Set[str] = {
    "+",  # Rhythm change
    "~",  # Signal quality change
    "|",  # Isolated QRS-like artifact
    "[",  # Start of ventricular flutter/fibrillation
    "]",  # End of ventricular flutter/fibrillation
    "!",  # Ventricular flutter wave
    '"',  # Comment annotation
    "x",  # Non-conducted P-wave (blocked APC)
    "p",  # Peak of P-wave
    "t",  # Peak of T-wave
    "u",  # Peak of U-wave
    "`",  # PQ junction
    "'",  # J-point
    "^",  # Non-conducted PAC
}

# Standard 5 AAMI classes in fixed order
AAMI_CLASSES: Tuple[str, ...] = ("N", "S", "V", "F", "Q")

AAMI_CLASS_TO_INT: Dict[str, int] = {cls_name: idx for idx, cls_name in enumerate(AAMI_CLASSES)}
INT_TO_AAMI_CLASS: Dict[int, str] = {idx: cls_name for idx, cls_name in enumerate(AAMI_CLASSES)}

AAMI_CLASS_DESCRIPTIONS: Dict[str, str] = {
    "N": "Normal Beat (Normal sinus, LBBB, RBBB, escape)",
    "S": "Supraventricular Ectopic Beat (Atrial/Nodal premature)",
    "V": "Ventricular Ectopic Beat (PVC, Ventricular escape)",
    "F": "Fusion Beat (Ventricular & Normal fusion)",
    "Q": "Unknown / Paced / Unclassifiable Beat",
}

AAMI_CLASS_COLORS: Dict[str, str] = {
    "N": "#10B981",  # Green
    "S": "#3B82F6",  # Blue
    "V": "#EF4444",  # Red / Warning
    "F": "#F59E0B",  # Amber
    "Q": "#8B5CF6",  # Purple
}


def map_symbol_to_aami(symbol: str) -> Optional[str]:
    """Map a PhysioNet MIT-BIH annotation symbol to an AAMI class.

    Args:
        symbol: Raw annotation symbol string from WFDB.

    Returns:
        One of 'N', 'S', 'V', 'F', 'Q', or None if it is a non-beat annotation.
    """
    clean_sym = symbol.strip()
    if clean_sym in NON_BEAT_SYMBOLS:
        return None
    return MITBIH_TO_AAMI.get(clean_sym, None)


def is_valid_beat(symbol: str) -> bool:
    """Return True if the annotation symbol represents a valid cardiac beat."""
    return map_symbol_to_aami(symbol) is not None


def aami_to_int(aami_class: str) -> int:
    """Convert an AAMI class string ('N', 'S', 'V', 'F', 'Q') to integer index [0..4]."""
    if aami_class not in AAMI_CLASS_TO_INT:
        raise ValueError(f"Unknown AAMI class: '{aami_class}'. Expected one of {AAMI_CLASSES}")
    return AAMI_CLASS_TO_INT[aami_class]


def int_to_aami(class_idx: int) -> str:
    """Convert integer index [0..4] to AAMI class string."""
    if class_idx not in INT_TO_AAMI_CLASS:
        raise ValueError(f"Unknown class index: {class_idx}. Expected [0..{len(AAMI_CLASSES)-1}]")
    return INT_TO_AAMI_CLASS[class_idx]


def get_class_description(aami_class: str) -> str:
    """Get full human-readable clinical description of an AAMI class."""
    return AAMI_CLASS_DESCRIPTIONS.get(aami_class, "Unknown Class")
