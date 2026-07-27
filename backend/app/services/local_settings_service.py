"""Local test-version data-source settings with non-echoing plaintext credentials."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Literal

from pydantic import SecretStr

from backend.app.core.config import Settings
from backend.app.repositories.databento_market_repository import DatabentoMarketRepository
from backend.app.schemas.runtime_settings import (
    CredentialDeleteResponse,
    DataSourceBatchItem,
    DataSourceSettingsResponse,
    DataSourceStatus,
    DataSourceUpdate,
)


@dataclass(frozen=True)
class DataSourceSpec:
    source_id: str
    display_name: str
    category: Literal["MARKET", "MACRO"]
    purpose: str
    cost_notice: str
    required: bool
    can_disable: bool
    credential_supported: bool
    credential_required: bool
    default_enabled: bool


SOURCE_SPECS = (
    DataSourceSpec("databento", "Databento", "MARKET", "Official gold market prices and formal model input.", "Account and plan required.", True, False, True, True, True),
    DataSourceSpec("federal_reserve", "Federal Reserve", "MACRO", "FOMC calendar and policy releases.", "Official public source; normally no key.", True, True, False, False, True),
    DataSourceSpec("new_york_fed", "New York Fed", "MACRO", "EFFR and official money-market rates.", "Official public source; normally no key.", True, True, False, False, True),
    DataSourceSpec("us_treasury", "U.S. Treasury", "MACRO", "Nominal and real Treasury yields.", "Official public source; normally no key.", True, True, False, False, True),
    DataSourceSpec("bls", "BLS", "MACRO", "CPI, payrolls, unemployment and wages.", "Official API; registration key increases quota.", True, True, True, False, True),
    DataSourceSpec("bea", "BEA", "MACRO", "PCE and personal income/outlays releases.", "Official API and RSS.", True, True, True, False, True),
    DataSourceSpec("fred", "FRED / ALFRED", "MACRO", "Optional official cross-check for rates and initial values.", "Optional official API; key required for API calls.", False, True, True, True, True),
)
SOURCE_SPEC_BY_ID = {item.source_id: item for item in SOURCE_SPECS}


class LocalSettingsService:
    """Persist only current local settings; credentials are never returned by public methods."""

    _write_lock = Lock()

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.path = settings.secrets_file_path
        self.display_path = settings.secrets_display_path
        self.market_repository = DatabentoMarketRepository(settings.storage_dir / "market")

    @staticmethod
    def _default_payload() -> dict:
        return {
            "schema_version": "aurumpilot.local_settings.v1",
            "sources": {
                spec.source_id: {"enabled": spec.default_enabled}
                for spec in SOURCE_SPECS
            },
        }

    def _read(self) -> dict:
        if not self.path.is_file():
            return self._default_payload()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Local settings file is unreadable or invalid") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("sources"), dict):
            raise ValueError("Local settings file has an invalid shape")
        merged = self._default_payload()
        for source_id, record in payload["sources"].items():
            if source_id in SOURCE_SPEC_BY_ID and isinstance(record, dict):
                merged["sources"][source_id].update(record)
        return merged

    def _write(self, payload: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload["updated_at_utc"] = datetime.now(UTC).isoformat()
        serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        with self._write_lock:
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=self.path.parent, prefix=".secrets.", suffix=".tmp", delete=False
                ) as handle:
                    handle.write(serialized)
                    handle.flush()
                    os.fsync(handle.fileno())
                    temporary_path = Path(handle.name)
                os.chmod(temporary_path, stat.S_IRUSR | stat.S_IWUSR)
                os.replace(temporary_path, self.path)
                os.chmod(self.path, stat.S_IRUSR | stat.S_IWUSR)
            finally:
                if temporary_path is not None and temporary_path.exists():
                    temporary_path.unlink()

    @staticmethod
    def _record(payload: dict, source_id: str) -> dict:
        if source_id not in SOURCE_SPEC_BY_ID:
            raise ValueError(f"Unknown data source: {source_id}")
        return payload["sources"][source_id]

    def list_sources(self) -> DataSourceSettingsResponse:
        payload = self._read()
        statuses: list[DataSourceStatus] = []
        for spec in SOURCE_SPECS:
            record = self._record(payload, spec.source_id)
            enabled = bool(record.get("enabled", spec.default_enabled))
            configured = bool(record.get("api_key"))
            credential_status = "DISABLED" if not enabled else "CONFIGURED" if configured else "NOT_CONFIGURED"
            statuses.append(DataSourceStatus(
                source_id=spec.source_id,
                display_name=spec.display_name,
                category=spec.category,
                purpose=spec.purpose,
                cost_notice=spec.cost_notice,
                required=spec.required,
                can_disable=spec.can_disable,
                credential_supported=spec.credential_supported,
                credential_required=spec.credential_required,
                enabled=enabled,
                credential_status=credential_status,
            ))
        databento = next(item for item in statuses if item.source_id == "databento")
        return DataSourceSettingsResponse(
            sources=statuses,
            market_data_ready=(
                databento.enabled
                and databento.credential_status == "CONFIGURED"
                and self._market_cache_ready()
            ),
            secrets_file_path=self.display_path,
            plaintext_warning="Test version: API keys are stored as plaintext in this local file.",
        )

    def _market_cache_ready(self) -> bool:
        try:
            frame, _ = self.market_repository.read()
            latest = frame["ts_event_utc"].iloc[-1].date()
        except (OSError, ValueError, FileNotFoundError, KeyError, IndexError):
            return False
        return (datetime.now(UTC).date() - latest).days <= self.settings.market_max_staleness_calendar_days

    def update_source(self, source_id: str, update: DataSourceUpdate) -> DataSourceSettingsResponse:
        return self.update_sources([
            DataSourceBatchItem(
                source_id=source_id,
                enabled=update.enabled,
                api_key=update.api_key,
            )
        ])

    def update_sources(
        self,
        updates: list[DataSourceBatchItem],
    ) -> DataSourceSettingsResponse:
        """Validate the full editor payload before one atomic settings write."""

        validated: list[tuple[DataSourceSpec, DataSourceBatchItem]] = []
        for update in updates:
            spec = SOURCE_SPEC_BY_ID.get(update.source_id)
            if spec is None:
                raise ValueError(f"Unknown data source: {update.source_id}")
            if update.enabled is False and not spec.can_disable:
                raise ValueError(
                    f"{spec.display_name} is required and cannot be disabled"
                )
            if update.api_key is not None and not spec.credential_supported:
                raise ValueError(f"{spec.display_name} does not accept an API key")
            validated.append((spec, update))

        payload = self._read()
        updated_at = datetime.now(UTC).isoformat()
        for spec, update in validated:
            record = self._record(payload, spec.source_id)
            if update.enabled is not None:
                record["enabled"] = update.enabled
                record["enabled_updated_at_utc"] = updated_at
            if update.api_key is not None:
                record["api_key"] = update.api_key.get_secret_value()
        self._write(payload)
        return self.list_sources()

    def delete_credential(self, source_id: str) -> CredentialDeleteResponse:
        spec = SOURCE_SPEC_BY_ID.get(source_id)
        if spec is None:
            raise ValueError(f"Unknown data source: {source_id}")
        if not spec.credential_supported:
            raise ValueError(f"{spec.display_name} does not accept an API key")
        payload = self._read()
        record = self._record(payload, source_id)
        deleted = record.pop("api_key", None) is not None
        self._write(payload)
        status = "DISABLED" if not bool(record.get("enabled", spec.default_enabled)) else "NOT_CONFIGURED"
        return CredentialDeleteResponse(source_id=source_id, deleted=deleted, credential_status=status)

    def is_enabled(self, source_id: str) -> bool:
        payload = self._read()
        spec = SOURCE_SPEC_BY_ID.get(source_id)
        if spec is None:
            raise ValueError(f"Unknown data source: {source_id}")
        return bool(self._record(payload, source_id).get("enabled", spec.default_enabled))

    def credential(self, source_id: str) -> SecretStr | None:
        payload = self._read()
        value = self._record(payload, source_id).get("api_key")
        return SecretStr(value) if isinstance(value, str) and value else None
