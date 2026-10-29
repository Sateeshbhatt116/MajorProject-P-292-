"""Global settings and configuration for the ECG analysis application."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # config/ is one level below project root
    return Path(__file__).resolve().parent.parent


@dataclass
class AppSettings:
    """Application and pipeline settings with sensible biomedical defaults."""

    # Project paths
    project_root: Path = field(default_factory=get_project_root)
    data_dir: Path = field(init=False)
    data_raw_dir: Path = field(init=False)
    data_interim_dir: Path = field(init=False)
    data_processed_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    models_saved_dir: Path = field(init=False)
    models_metadata_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    reports_figures_dir: Path = field(init=False)
    reports_metrics_dir: Path = field(init=False)

    # General configuration
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    random_seed: int = field(default_factory=lambda: int(os.getenv("RANDOM_SEED", "42")))
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "cpu"))

    # ECG Signal acquisition & sampling parameters
    default_sampling_rate: int = 360  # MIT-BIH Arrhythmia standard frequency is 360 Hz
    default_leads: List[str] = field(default_factory=lambda: ["MLII", "V1"])

    # Preprocessing parameters
    bandpass_lowcut: float = 0.5  # High-pass cut to remove respiration / baseline wander (Hz)
    bandpass_highcut: float = 45.0  # Low-pass cut to remove high-frequency EMG noise (Hz)
    bandpass_order: int = 3
    notch_frequency: float = 60.0  # US powerline frequency (MIT-BIH was recorded in Boston, USA)
    notch_quality_factor: float = 30.0

    # Beat segmentation parameters (centered around R-peak)
    # 360 Hz sampling rate: 250 samples ≈ 694 ms (90 samples pre-R, 160 samples post-R)
    pre_r_samples: int = 90
    post_r_samples: int = 160
    beat_length: int = 250  # 90 + 160

    # AAMI EC57 5-class standard
    aami_classes: Tuple[str, ...] = ("N", "S", "V", "F", "Q")
    aami_class_descriptions: Dict[str, str] = field(
        default_factory=lambda: {
            "N": "Normal / Bundle branch block beat",
            "S": "Supraventricular ectopic beat (SVEB)",
            "V": "Ventricular ectopic beat (VEB)",
            "F": "Fusion beat",
            "Q": "Paced / Unclassifiable beat",
        }
    )

    def __post_init__(self) -> None:
        """Resolve and guarantee directories exist."""
        self.data_dir = self.project_root / "data"
        self.data_raw_dir = Path(os.getenv("DATA_RAW_DIR", str(self.data_dir / "raw"))).resolve()
        self.data_interim_dir = self.data_dir / "interim"
        self.data_processed_dir = Path(
            os.getenv("DATA_PROCESSED_DIR", str(self.data_dir / "processed"))
        ).resolve()

        self.models_dir = self.project_root / "models"
        self.models_saved_dir = Path(
            os.getenv("MODELS_SAVED_DIR", str(self.models_dir / "saved"))
        ).resolve()
        self.models_metadata_dir = self.models_dir / "metadata"

        self.reports_dir = Path(os.getenv("REPORTS_DIR", str(self.project_root / "reports"))).resolve()
        self.reports_figures_dir = self.reports_dir / "figures"
        self.reports_metrics_dir = self.reports_dir / "metrics"

        # Ensure directories exist
        for d in [
            self.data_raw_dir,
            self.data_interim_dir,
            self.data_processed_dir,
            self.models_saved_dir,
            self.models_metadata_dir,
            self.reports_figures_dir,
            self.reports_metrics_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


_SETTINGS_INSTANCE: AppSettings | None = None


def get_settings() -> AppSettings:
    """Singleton getter for application settings."""
    global _SETTINGS_INSTANCE
    if _SETTINGS_INSTANCE is None:
        _SETTINGS_INSTANCE = AppSettings()
    return _SETTINGS_INSTANCE
