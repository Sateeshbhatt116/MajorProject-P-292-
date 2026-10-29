"""Standardized logging configuration for the project."""
from __future__ import annotations

import logging
import sys
from typing import Optional

from config.settings import get_settings


def setup_logger(
    name: str = "ecg_analysis",
    level: Optional[str] = None,
    log_to_stdout: bool = True,
) -> logging.Logger:
    """Configure and return a structured logger."""
    settings = get_settings()
    log_level_name = level or settings.log_level
    numeric_level = getattr(logging, log_level_name.upper(), logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers if logger was already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        if log_to_stdout:
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setLevel(numeric_level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)

    return logger
