"""Schemas for the read-only Today Advice composition layer."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PortfolioAllocationSnapshotCreate(BaseModel):
    """One user-reported gold allocation; percentages use 0..100 units."""

    model_config = ConfigDict(extra="forbid")

    gold_weight_pct: float = Field(ge=0, le=100)
    as_of_utc: datetime
    source: Literal["USER_REPORTED"] = "USER_REPORTED"

    @field_validator("as_of_utc")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of_utc must include an explicit timezone")
        return value.astimezone(UTC)


class PortfolioAllocationSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    snapshot_id: str
    user_id: str
    current_gold_weight_pct: float = Field(ge=0, le=100)
    as_of_utc: datetime
    created_at_utc: datetime
    source: Literal["USER_REPORTED"]
    supersedes_snapshot_id: str | None = None

