from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCAL_CONFIG_ROOT = Path(os.getenv("LOCALAPPDATA", PROJECT_ROOT / ".local")) / "AurumPilot"


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local .env file."""

    app_name: str = "AurumPilot"
    app_env: str = "development"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_origin: str = "http://localhost:8501"
    storage_dir: Path = PROJECT_ROOT / "storage"
    model_dir: Path = PROJECT_ROOT / "storage" / "models"
    databento_dataset: str = "GLBX.MDP3"
    databento_schema: str = "ohlcv-1d"
    databento_symbol: str = "GC.v.0"
    databento_stype_in: str = "continuous"
    market_history_calendar_days: int = 800
    market_max_staleness_calendar_days: int = 4
    market_sync_overlap_days: int = 2
    market_sync_chunk_days: int = 370
    technical_sidecar_url: str = "http://model-sidecar:8100"
    technical_sidecar_timeout_seconds: float = 120.0
    technical_history_start: date = date(2010, 6, 7)
    technical_display_timezone: str = "Asia/Singapore"
    technical_strategy_min_gold_weight: float = 0.5
    technical_strategy_max_gold_weight: float = 1.0
    technical_strategy_bounds_source: str = (
        "MN18_RECEIVER_50_TO_100_PERCENT"
    )
    user_allocation_stale_days: int = 30
    demo_user_id: str = "demo"
    local_config_dir: Path = DEFAULT_LOCAL_CONFIG_ROOT
    local_config_display_path: str | None = None

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        for relative in (
            "market", "models", "macro", "technical", "users",
            "predictions", "forecasts", "cycles", "audits",
        ):
            (self.storage_dir / relative).mkdir(parents=True, exist_ok=True)

    @property
    def secrets_file_path(self) -> Path:
        return self.local_config_dir / "secrets.json"

    @property
    def secrets_display_path(self) -> str:
        return self.local_config_display_path or str(self.secrets_file_path)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
