"""Patient-aware and record-level dataset splitting to prevent data leakage."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.ecg_analysis.preprocessing.segmentation import ECGBeat
from src.ecg_analysis.utils.logging_config import setup_logger

logger = setup_logger("ecg_analysis.data.dataset_splitter")

# Standard de Chazal et al. (2004) Inter-Patient Benchmark Records
DS1_TRAINING_RECORDS: Tuple[str, ...] = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230"
)

DS2_TESTING_RECORDS: Tuple[str, ...] = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234"
)


@dataclass
class DatasetSplit:
    """Container holding partitioned beats and audit metadata."""

    train_beats: List[ECGBeat]
    val_beats: List[ECGBeat]
    test_beats: List[ECGBeat]
    train_records: Set[str] = field(default_factory=set)
    val_records: Set[str] = field(default_factory=set)
    test_records: Set[str] = field(default_factory=set)
    split_strategy: str = "inter_patient"

    @property
    def total_beats(self) -> int:
        return len(self.train_beats) + len(self.val_beats) + len(self.test_beats)

    def summary(self) -> Dict[str, int]:
        return {
            "train_count": len(self.train_beats),
            "val_count": len(self.val_beats),
            "test_count": len(self.test_beats),
            "total_count": self.total_beats,
        }


def split_beats_inter_patient(
    beats: Sequence[ECGBeat],
    test_size: float = 0.20,
    val_size: float = 0.10,
    random_state: int = 42,
) -> DatasetSplit:
    """Partition ECG beats into train, val, and test subsets with zero record/patient leakage.

    Strategy:
    1. If multiple records are present:
       Applies GroupShuffleSplit based on record_name. All beats from any individual
       patient/record belong exclusively to either train, validation, or test.
    2. If only one record is present:
       Partitions the signal chronologically into discrete blocks (e.g. first 70% time,
       next 15% time, last 15% time). This prevents adjacent-beat autocorrelation leakage
       inherent in random shuffling of contiguous cardiac rhythms.
    """
    if not beats:
        raise ValueError("Cannot split empty beats list.")

    records = [b.record_name for b in beats]
    unique_records = sorted(list(set(records)))

    if len(unique_records) > 1:
        # Multi-record inter-patient split
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        idx_arr = np.arange(len(beats))
        train_val_idx, test_idx = next(gss_test.split(idx_arr, groups=records))

        train_val_beats = [beats[i] for i in train_val_idx]
        train_val_records = [records[i] for i in train_val_idx]

        val_adjusted_size = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(
            n_splits=1, test_size=val_adjusted_size, random_state=random_state + 1
        )
        sub_idx = np.arange(len(train_val_beats))
        train_sub_idx, val_sub_idx = next(gss_val.split(sub_idx, groups=train_val_records))

        train_beats = [train_val_beats[i] for i in train_sub_idx]
        val_beats = [train_val_beats[i] for i in val_sub_idx]
        test_beats = [beats[i] for i in test_idx]

        train_recs = set(b.record_name for b in train_beats)
        val_recs = set(b.record_name for b in val_beats)
        test_recs = set(b.record_name for b in test_beats)

        strategy = "group_inter_patient"

    else:
        # Single-record chronological block split
        n = len(beats)
        n_test = max(1, int(n * test_size))
        n_val = max(1, int(n * val_size))
        n_train = n - n_test - n_val

        # Sort strictly by chronological R-peak sample index
        sorted_beats = sorted(beats, key=lambda b: b.r_peak_index)

        train_beats = sorted_beats[:n_train]
        val_beats = sorted_beats[n_train : n_train + n_val]
        test_beats = sorted_beats[n_train + n_val :]

        train_recs = set([unique_records[0]])
        val_recs = set([unique_records[0]])
        test_recs = set([unique_records[0]])
        strategy = "single_record_chronological_block"

    split = DatasetSplit(
        train_beats=train_beats,
        val_beats=val_beats,
        test_beats=test_beats,
        train_records=train_recs,
        val_records=val_recs,
        test_records=test_recs,
        split_strategy=strategy,
    )

    logger.info(
        f"Dataset split ({strategy}): "
        f"Train={len(train_beats)}, Val={len(val_beats)}, Test={len(test_beats)}"
    )
    return split


def verify_zero_leakage(split: DatasetSplit) -> bool:
    """Audit a dataset split for cross-partition data leakage.

    Checks:
    1. Multi-record: disjoint sets of record identifiers.
    2. Single-record: disjoint, chronological time ranges of R-peak indices.
    """
    if split.split_strategy == "group_inter_patient":
        # Disjoint record sets
        train_val_overlap = split.train_records & split.val_records
        train_test_overlap = split.train_records & split.test_records
        val_test_overlap = split.val_records & split.test_records

        if train_val_overlap or train_test_overlap or val_test_overlap:
            logger.error(
                f"Data leakage detected! Overlapping records: "
                f"Train-Val={train_val_overlap}, Train-Test={train_test_overlap}"
            )
            return False
        return True

    elif split.split_strategy == "single_record_chronological_block":
        # Time-ordered non-overlapping intervals
        max_train_peak = max(b.r_peak_index for b in split.train_beats)
        min_val_peak = min(b.r_peak_index for b in split.val_beats)
        max_val_peak = max(b.r_peak_index for b in split.val_beats)
        min_test_peak = min(b.r_peak_index for b in split.test_beats)

        if max_train_peak >= min_val_peak or max_val_peak >= min_test_peak:
            logger.error(
                f"Temporal leakage detected! R-peaks overlap across chronological blocks: "
                f"train_max={max_train_peak}, val_min={min_val_peak}, test_min={min_test_peak}"
            )
            return False
        return True

    return True
