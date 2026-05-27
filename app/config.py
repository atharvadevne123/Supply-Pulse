"""Application configuration loaded from environment variables."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

logger = logging.getLogger(__name__)


class Settings:
    """Immutable application settings derived from environment variables."""

    def __init__(self) -> None:
        self.database_url: str = os.getenv("DATABASE_URL", "sqlite:///./supply_pulse.db")
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0")
        self.api_port: int = int(os.getenv("API_PORT", "8000"))
        self.api_workers: int = int(os.getenv("API_WORKERS", "2"))
        self.model_version: str = os.getenv("MODEL_VERSION", "1.0.0")
        self.rate_limit: int = int(os.getenv("RATE_LIMIT", "300"))
        self.drift_p_value_threshold: float = float(os.getenv("DRIFT_P_VALUE_THRESHOLD", "0.05"))
        self.drift_min_sample_size: int = int(os.getenv("DRIFT_MIN_SAMPLE_SIZE", "30"))
        self.default_service_level: float = float(os.getenv("DEFAULT_SERVICE_LEVEL", "0.95"))
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def __repr__(self) -> str:
        return (
            f"Settings(host={self.api_host}, port={self.api_port}, "
            f"workers={self.api_workers}, log_level={self.log_level})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    settings = Settings()
    logger.info("Settings loaded: %s", settings)
    return settings
