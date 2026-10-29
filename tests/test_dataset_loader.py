"""Unit tests for dataset loading, AAMI mapping, and validation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ecg_analysis.data.dataset_loader import ECGRecord
from src.ecg_analysis.data.dataset_validator import DatasetValidator
from src.ecg_analysis.data.label_mapping import (
    AAMI_CLASSES,
    aami_to_int,
    get_class_description,
    int_to_aami,
    is_valid_beat,
    map_symbol_to_aami,
)


class TestLabelMapping:
    """Test suite for ANSI/AAMI EC57 label mapping."""

    def test_normal_beat_mapping(self) -> None:
        normal_symbols = ["N", "L", "R", "e", "j"]
        for sym in normal_symbols:
            assert map_symbol_to_aami(sym) == "N"
            assert is_valid_beat(sym) is True

    def test_supraventricular_ectopic_mapping(self) -> None:
        sveb_symbols = ["A", "a", "J", "S"]
        for sym in sveb_symbols:
            assert map_symbol_to_aami(sym) == "S"
            assert is_valid_beat(sym) is True

    def test_ventricular_ectopic_mapping(self) -> None:
        veb_symbols = ["V", "E", "r"]
        for sym in veb_symbols:
            assert map_symbol_to_aami(sym) == "V"
            assert is_valid_beat(sym) is True

    def test_fusion_beat_mapping(self) -> None:
        assert map_symbol_to_aami("F") == "F"
        assert is_valid_beat("F") is True

    def test_paced_and_unclassifiable_mapping(self) -> None:
        q_symbols = ["/", "f", "Q", "?"]
        for sym in q_symbols:
            assert map_symbol_to_aami(sym) == "Q"
            assert is_valid_beat(sym) is True

    def test_non_beat_annotations(self) -> None:
        non_beats = ["+", "~", "|", "[", "]", "!", '"', "x"]
        for sym in non_beats:
            assert map_symbol_to_aami(sym) is None
            assert is_valid_beat(sym) is False

    def test_bidirectional_integer_mapping(self) -> None:
        for idx, cls_name in enumerate(AAMI_CLASSES):
            assert aami_to_int(cls_name) == idx
            assert int_to_aami(idx) == cls_name

        with pytest.raises(ValueError):
            aami_to_int("INVALID")

        with pytest.raises(ValueError):
            int_to_aami(99)

    def test_class_descriptions(self) -> None:
        for cls_name in AAMI_CLASSES:
            desc = get_class_description(cls_name)
            assert isinstance(desc, str)
            assert len(desc) > 5


class TestECGRecord:
    """Test suite for ECGRecord container."""

    def test_record_properties(self, synthetic_record: ECGRecord) -> None:
        assert synthetic_record.record_name == "synth_100"
        assert synthetic_record.sampling_rate == 360
        assert synthetic_record.num_samples == 3600
        assert pytest.approx(synthetic_record.duration_seconds, 0.01) == 10.0
        assert synthetic_record.num_leads == 2

    def test_lead_signal_retrieval(self, synthetic_record: ECGRecord) -> None:
        mlii = synthetic_record.get_lead_signal("MLII")
        v1 = synthetic_record.get_lead_signal("V1")
        default_lead = synthetic_record.get_lead_signal()

        assert len(mlii) == 3600
        assert len(v1) == 3600
        assert np.array_equal(mlii, default_lead)
        assert np.allclose(v1, mlii * 0.8)

    def test_valid_beat_filtering(self) -> None:
        # Create record with a mix of beat and non-beat annotations
        samples = np.array([100, 200, 300, 400])
        symbols = ["N", "+", "V", "~"]
        record = ECGRecord(
            record_name="test_filter",
            sampling_rate=360,
            lead_names=["MLII"],
            signals=np.zeros((500, 1)),
            annotation_samples=samples,
            annotation_symbols=symbols,
        )

        valid_indices, valid_labels = record.get_valid_beat_annotations()
        assert len(valid_indices) == 2
        assert list(valid_indices) == [100, 300]
        assert valid_labels == ["N", "V"]


class TestDatasetValidator:
    """Test suite for dataset validation rules."""

    def test_valid_record_passes(self, synthetic_record: ECGRecord) -> None:
        validator = DatasetValidator(expected_fs=360, min_duration_sec=5.0)
        result = validator.validate_record(synthetic_record)

        assert result.is_valid is True
        assert len(result.issues) == 0
        assert result.num_valid_beats > 0
        assert "N" in result.class_distribution

    def test_empty_signal_fails(self) -> None:
        empty_rec = ECGRecord(
            record_name="empty",
            sampling_rate=360,
            lead_names=["MLII"],
            signals=np.array([]),
        )
        validator = DatasetValidator()
        result = validator.validate_record(empty_rec)
        assert result.is_valid is False
        assert any("empty" in issue.lower() for issue in result.issues)

    def test_nan_detection(self, synthetic_record: ECGRecord) -> None:
        corrupted = np.copy(synthetic_record.signals)
        corrupted[100, 0] = np.nan
        bad_rec = ECGRecord(
            record_name="nan_rec",
            sampling_rate=360,
            lead_names=["MLII", "V1"],
            signals=corrupted,
        )
        validator = DatasetValidator()
        result = validator.validate_record(bad_rec)
        assert result.is_valid is False
        assert any("nan" in issue.lower() for issue in result.issues)

    def test_out_of_bounds_annotations(self) -> None:
        bad_ann_rec = ECGRecord(
            record_name="oob_ann",
            sampling_rate=360,
            lead_names=["MLII"],
            signals=np.ones((500, 1)),
            annotation_samples=np.array([250, 9999]),  # 9999 is out of bounds
            annotation_symbols=["N", "N"],
        )
        validator = DatasetValidator()
        result = validator.validate_record(bad_ann_rec)
        assert result.is_valid is False
        assert any("outside signal" in issue.lower() for issue in result.issues)


class TestDatasetLoaderOperations:
    """Test suite for ECGDataLoader filesystem and loading operations."""

    def test_missing_dataset_directory_handled(self, tmp_path: Path) -> None:
        """Verify loader handles non-existent raw directory gracefully by creating it."""
        missing_dir = tmp_path / "non_existent_raw_dir"
        assert not missing_dir.exists()
        from src.ecg_analysis.data.dataset_loader import ECGDataLoader

        loader = ECGDataLoader(raw_data_dir=missing_dir)
        assert missing_dir.is_dir()
        assert loader.list_available_records() == []

    def test_reject_missing_record_gracefully(self) -> None:
        """Verify requesting a non-existent record raises DatasetNotFoundError."""
        from src.ecg_analysis.data.dataset_loader import ECGDataLoader
        from src.ecg_analysis.utils.exceptions import DatasetNotFoundError

        loader = ECGDataLoader()
        with pytest.raises(DatasetNotFoundError):
            loader.load_record("completely_fictional_record_999")

    def test_load_valid_physionet_record_if_present(self) -> None:
        """Verify loading an authentic MIT-BIH record from disk when present."""
        from src.ecg_analysis.data.dataset_loader import ECGDataLoader

        loader = ECGDataLoader()
        if loader.record_exists("100"):
            rec = loader.load_record("100", sampfrom=0, sampto=3600)
            assert rec.record_name == "100"
            assert rec.sampling_rate == 360
            assert rec.num_samples == 3600
            assert rec.num_leads >= 1
            assert "MLII" in rec.lead_names
            valid_idx, valid_labels = rec.get_valid_beat_annotations()
            assert len(valid_idx) > 0
            for lbl in valid_labels:
                assert lbl in AAMI_CLASSES

