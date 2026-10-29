"""Path resolution utilities using pathlib."""
from __future__ import annotations

from pathlib import Path
from typing import Union

from config.settings import get_settings


class ProjectPaths:
    """Convenient path resolver based on application settings."""

    @staticmethod
    def root() -> Path:
        return get_settings().project_root

    @staticmethod
    def raw_data_dir() -> Path:
        return get_settings().data_raw_dir

    @staticmethod
    def processed_data_dir() -> Path:
        return get_settings().data_processed_dir

    @staticmethod
    def models_saved_dir() -> Path:
        return get_settings().models_saved_dir

    @staticmethod
    def reports_figures_dir() -> Path:
        return get_settings().reports_figures_dir

    @staticmethod
    def reports_metrics_dir() -> Path:
        return get_settings().reports_metrics_dir

    @staticmethod
    def resolve_record_path(record_name: str, base_dir: Union[Path, str, None] = None) -> Path:
        """Resolve a WFDB record path (without extension) in the raw data directory."""
        base = Path(base_dir) if base_dir else get_settings().data_raw_dir
        # If record_name has extension like .dat or .hea, strip it
        clean_name = Path(record_name).stem
        return base / clean_name
