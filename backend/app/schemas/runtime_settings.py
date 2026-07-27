"""Public contracts for local data-source settings; secret values are request-only."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator


CredentialStatus = Literal["NOT_CONFIGURED", "CONFIGURED", "DISABLED"]
SourceCategory = Literal["MARKET", "MACRO"]


class DataSourceUpdate(BaseModel):
    enabled: bool | None = None
    api_key: SecretStr | None = Field(default=None, repr=False)

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        normalized = value.get_secret_value().strip()
        if not normalized:
            raise ValueError("api_key cannot be blank")
        if len(normalized) > 4096:
            raise ValueError("api_key is too long")
        return SecretStr(normalized)


class DataSourceBatchItem(DataSourceUpdate):
    source_id: str


class DataSourceBatchUpdate(BaseModel):
    sources: list[DataSourceBatchItem] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_sources(self) -> "DataSourceBatchUpdate":
        source_ids = [item.source_id for item in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source_id values must be unique")
        return self


class DataSourceStatus(BaseModel):
    source_id: str
    display_name: str
    category: SourceCategory
    purpose: str
    cost_notice: str
    required: bool
    can_disable: bool
    credential_supported: bool
    credential_required: bool
    enabled: bool
    credential_status: CredentialStatus


class DataSourceSettingsResponse(BaseModel):
    schema_version: str = "aurumpilot.local_settings.v1"
    sources: list[DataSourceStatus]
    market_data_ready: bool
    secrets_file_path: str
    plaintext_warning: str


class DataSourceVerificationResponse(BaseModel):
    source_id: Literal["databento"] = "databento"
    verified: bool = True
    metadata_only: bool = True
    dataset: str
    schema_name: str
    symbol: str
    stype_in: str
    symbol_resolved: bool
    dataset_available_from_utc: str | None = None
    dataset_available_until_utc: str | None = None


class CredentialDeleteResponse(BaseModel):
    source_id: str
    deleted: bool
    credential_status: CredentialStatus
