"""Phase 1 Foundation unit tests covering environment, configuration, paths, and imports."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from config.settings import AppSettings, get_settings
from src.ecg_analysis.utils.exceptions import (
    DatasetNotFoundError,
    ECGAnalysisError,
    FeatureExtractionError,
    InvalidSignalError,
    ModelInferenceError,
    ModelNotFoundError,
    PreprocessingError,
)
from src.ecg_analysis.utils.logging_config import setup_logger
from src.ecg_analysis.utils.paths import ProjectPaths


class TestEnvironmentChecks:
    """Validate runtime environment and core dependency availability."""

    def test_python_version_compatibility(self) -> None:
        """Verify Python runtime meets or exceeds Python 3.10 requirement."""
        assert sys.version_info >= (3, 10), f"Python version {sys.version} is below 3.10"

    def test_core_dependencies_importable(self) -> None:
        """Verify all critical numerical, scientific, ML, and UI libraries import cleanly."""
        required_packages = [
            "numpy",
            "pandas",
            "scipy",
            "pywt",
            "sklearn",
            "wfdb",
            "torch",
            "matplotlib",
            "streamlit",
            "pytest",
        ]
        for pkg in required_packages:
            mod = importlib.import_module(pkg)
            assert mod is not None, f"Failed to import core package: {pkg}"

    def test_essential_workspace_directories_exist(self) -> None:
        """Verify key workspace directories exist and are accessible."""
        settings = get_settings()
        assert settings.data_raw_dir.is_dir(), f"Missing: {settings.data_raw_dir}"
        assert settings.data_processed_dir.is_dir(), f"Missing: {settings.data_processed_dir}"
        assert settings.models_saved_dir.is_dir(), f"Missing: {settings.models_saved_dir}"
        assert settings.reports_figures_dir.is_dir(), f"Missing: {settings.reports_figures_dir}"
        assert settings.reports_metrics_dir.is_dir(), f"Missing: {settings.reports_metrics_dir}"


class TestConfigurationLoading:
    """Validate configuration defaults, singleton behavior, and environment overrides."""

    def test_settings_singleton_and_defaults(self) -> None:
        """Verify get_settings returns singleton with correct biomedical defaults."""
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2, "get_settings() should return a singleton instance"
        assert s1.default_sampling_rate == 360
        assert s1.beat_length == 250
        assert s1.pre_r_samples == 90
        assert s1.post_r_samples == 160
        assert s1.aami_classes == ("N", "S", "V", "F", "Q")
        assert s1.bandpass_lowcut == 0.5
        assert s1.bandpass_highcut == 45.0
        assert s1.notch_frequency == 60.0

    def test_settings_custom_instance(self, tmp_path: Path) -> None:
        """Verify custom AppSettings instance initializes and creates directories."""
        custom_root = tmp_path / "custom_proj"
        custom_root.mkdir()
        custom = AppSettings(project_root=custom_root)
        assert custom.project_root == custom_root
        assert (custom_root / "data" / "raw").is_dir()
        assert (custom_root / "models" / "saved").is_dir()


class TestPathResolution:
    """Validate ProjectPaths utility resolution."""

    def test_project_root_contains_core_files(self) -> None:
        """Verify ProjectPaths.root() locates root files (pyproject.toml, README.md)."""
        root = ProjectPaths.root()
        assert (root / "pyproject.toml").is_file()
        assert (root / "README.md").is_file()
        assert (root / "requirements.txt").is_file()

    def test_paths_match_settings(self) -> None:
        """Verify ProjectPaths methods map to AppSettings directories."""
        settings = get_settings()
        assert ProjectPaths.raw_data_dir() == settings.data_raw_dir
        assert ProjectPaths.processed_data_dir() == settings.data_processed_dir
        assert ProjectPaths.models_saved_dir() == settings.models_saved_dir
        assert ProjectPaths.reports_figures_dir() == settings.reports_figures_dir
        assert ProjectPaths.reports_metrics_dir() == settings.reports_metrics_dir

    def test_resolve_record_path(self) -> None:
        """Verify resolve_record_path strips file extensions and anchors to raw dir."""
        res_plain = ProjectPaths.resolve_record_path("100")
        res_hea = ProjectPaths.resolve_record_path("100.hea")
        res_dat = ProjectPaths.resolve_record_path("100.dat")
        assert res_plain.name == "100"
        assert res_hea.name == "100"
        assert res_dat.name == "100"
        assert res_plain.parent == ProjectPaths.raw_data_dir()


class TestImportChecks:
    """Validate clean imports across the entire src.ecg_analysis namespace."""

    def test_src_module_imports(self) -> None:
        """Verify all submodules import with zero syntax or circular dependency errors."""
        modules_to_test = [
            "config.settings",
            "src.ecg_analysis.utils.paths",
            "src.ecg_analysis.utils.logging_config",
            "src.ecg_analysis.utils.exceptions",
            "src.ecg_analysis.data.label_mapping",
            "src.ecg_analysis.data.dataset_loader",
            "src.ecg_analysis.data.dataset_validator",
            "src.ecg_analysis.preprocessing.filtering",
            "src.ecg_analysis.preprocessing.baseline_correction",
            "src.ecg_analysis.preprocessing.normalization",
            "src.ecg_analysis.preprocessing.segmentation",
            "src.ecg_analysis.preprocessing.pipeline",
            "src.ecg_analysis.features.statistical_features",
            "src.ecg_analysis.features.wavelet_features",
            "src.ecg_analysis.features.feature_pipeline",
            "src.ecg_analysis.models.classical_ml",
            "src.ecg_analysis.models.cnn_model",
            "src.ecg_analysis.models.model_registry",
            "src.ecg_analysis.models.predict",
            "src.ecg_analysis.models.train",
            "src.ecg_analysis.evaluation.metrics",
            "src.ecg_analysis.evaluation.confusion_matrix",
            "src.ecg_analysis.evaluation.evaluation_report",
        ]
        for mod_name in modules_to_test:
            mod = importlib.import_module(mod_name)
            assert mod is not None, f"Could not import {mod_name}"


class TestLoggingAndExceptions:
    """Validate logging initialization and exception hierarchy."""

    def test_setup_logger(self) -> None:
        """Verify logger instantiation, naming, and level assignment."""
        log = setup_logger(name="test_foundation_logger", level="DEBUG")
        assert log.name == "test_foundation_logger"
        assert log.level == 10  # DEBUG

    def test_exception_hierarchy(self) -> None:
        """Verify all domain exceptions inherit from ECGAnalysisError."""
        subclasses = [
            DatasetNotFoundError,
            InvalidSignalError,
            PreprocessingError,
            FeatureExtractionError,
            ModelNotFoundError,
            ModelInferenceError,
        ]
        for exc in subclasses:
            assert issubclass(exc, ECGAnalysisError)

        # Test catching a child exception as ECGAnalysisError
        with pytest.raises(ECGAnalysisError):
            raise DatasetNotFoundError("Record not found in raw data")
