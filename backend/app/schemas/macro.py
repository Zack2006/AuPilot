"""Independent five-level official macro-risk API models."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aupilot.core.enums import MacroRiskLevel
from aupilot.core.macro_policy import FROZEN_MACRO_RISK_SCALE
from aupilot.macro_gate.schemas import (
    ALLOWED_EVENT_TYPES,
    ALLOWED_EVIDENCE_DOMAINS,
    OFFICIAL_DOMAINS,
    MacroClaim,
    MacroCoverageResult,
    MacroScoreComponents,
    MacroSummaryFact,
)


class MacroEvidenceCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    event_type: str
    title: str
    canonical_url: str
    eligible_from_utc: datetime
    content_sha256: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")
    score: float
    provider_id: str | None = None
    metric_id: str | None = None
    source_tier: Literal["A", "B", "C"] = "A"
    published_at_utc: datetime | None = None
    retrieved_at_utc: datetime | None = None
    first_seen_at_utc: datetime | None = None
    revision_status: Literal["INITIAL", "REVISED", "UNKNOWN"] = "UNKNOWN"
    evidence_kind: Literal["DOCUMENT", "OBSERVATION"] = "DOCUMENT"
    series_id: str | None = None
    observation_date: date | None = None
    observation_value: float | None = None
    claim_ids: list[str] = Field(default_factory=list)

    @field_validator("eligible_from_utc")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("macro citation timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("event_type")
    @classmethod
    def require_allowed_event_type(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in ALLOWED_EVENT_TYPES:
            raise ValueError(f"Unsupported macro citation event type: {value}")
        return normalized

    @field_validator("canonical_url")
    @classmethod
    def require_official_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_EVIDENCE_DOMAINS:
            raise ValueError("Macro citations must use an approved evidence HTTPS domain")
        return value

    @model_validator(mode="after")
    def validate_observation_fields(self) -> MacroEvidenceCitation:
        fields = (self.series_id, self.observation_date, self.observation_value)
        if self.evidence_kind == "OBSERVATION" and any(value is None for value in fields):
            raise ValueError("Observation citations require series, date, and value")
        if self.evidence_kind == "DOCUMENT" and any(value is not None for value in fields):
            raise ValueError("Document citations cannot contain structured observation fields")
        return self


class MacroRiskScaleBand(BaseModel):
    """One immutable band in the public five-level event-risk scale."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    risk_level: MacroRiskLevel = Field(serialization_alias="level")
    risk_score: int = Field(ge=1, le=5)
    timing: Literal[
        "OUTSIDE_MONITORING_WINDOW",
        "BEFORE_RELEASE",
        "AFTER_RELEASE",
    ]
    lower_bound_hours: int = Field(ge=0)
    upper_bound_hours: int | None = Field(default=None, ge=0)
    lower_bound_inclusive: bool
    upper_bound_inclusive: bool
    assessment_unavailable_also: bool = False
    event_factor_adjustment_allowed: bool = False

    @model_validator(mode="after")
    def validate_band(self) -> MacroRiskScaleBand:
        if self.risk_score != self.risk_level.score:
            raise ValueError("risk scale score must match its level")
        if self.upper_bound_hours is not None and self.lower_bound_hours > self.upper_bound_hours:
            raise ValueError("risk scale bounds are reversed")
        return self


class MacroInterpretation(BaseModel):
    """Auditable plain-language explanation derived only from accepted claims."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    interpretation_id: str = Field(min_length=1)
    event_type: str
    topic_key: Literal[
        "FOMC_POLICY",
        "EFFR_ALIGNMENT",
        "TREASURY_YIELD_CURVE",
        "REAL_YIELD_10Y",
        "BREAKEVEN_PROXY_10Y",
        "CPI_RELEASE",
        "PCE_RELEASE",
        "NFP_RELEASE",
    ]
    official_fact: str = Field(min_length=1)
    analysis: str = Field(min_length=1)
    interpretation_status: Literal[
        "EXPLAINED",
        "CONTEXT_ONLY",
        "INSUFFICIENT_COMPARISON",
    ]
    claim_ids: list[str] = Field(min_length=1)
    method: Literal["DETERMINISTIC_CLAIM_EXPLANATION_V1"] = (
        "DETERMINISTIC_CLAIM_EXPLANATION_V1"
    )
    affects_risk_score: Literal[False] = False
    affects_technical_model: Literal[False] = False
    affects_trade_permission: Literal[False] = False

    @field_validator("event_type")
    @classmethod
    def require_allowed_event_type(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in ALLOWED_EVENT_TYPES:
            raise ValueError(f"Unsupported macro interpretation event type: {value}")
        return normalized


def _default_risk_scale() -> list[MacroRiskScaleBand]:
    return [
        MacroRiskScaleBand(
            risk_level=band.risk_level,
            risk_score=band.risk_score,
            timing=band.timing,
            lower_bound_hours=band.lower_bound_hours,
            upper_bound_hours=band.upper_bound_hours,
            lower_bound_inclusive=band.lower_bound_inclusive,
            upper_bound_inclusive=band.upper_bound_inclusive,
            assessment_unavailable_also=band.assessment_unavailable_also,
            event_factor_adjustment_allowed=band.event_factor_adjustment_allowed,
        )
        for band in FROZEN_MACRO_RISK_SCALE
    ]


class MacroRiskResponse(BaseModel):
    """Frozen public contract; it cannot carry a direction or an action."""

    model_config = ConfigDict(extra="forbid")

    risk_level: MacroRiskLevel
    risk_score: int = Field(ge=1, le=5)
    risk_scale_basis: Literal[
        "OFFICIAL_EVENT_PROXIMITY_IMPACT_RELEVANCE_AND_ASSESSMENT_RELIABILITY"
    ] = (
        "OFFICIAL_EVENT_PROXIMITY_IMPACT_RELEVANCE_AND_ASSESSMENT_RELIABILITY"
    )
    event_severity_dynamic: Literal[False] = False
    event_impact_basis: Literal["DETERMINISTIC_EVENT_TYPE_POLICY"] = (
        "DETERMINISTIC_EVENT_TYPE_POLICY"
    )
    risk_scale: list[MacroRiskScaleBand] = Field(default_factory=_default_risk_scale, min_length=5, max_length=5)
    assessment_id: str | None = None
    decision_as_of_utc: datetime
    reason_codes: list[str]
    news_summary: list[str] = Field(min_length=1)
    citations: list[MacroEvidenceCitation] = Field(default_factory=list)
    claims: list[MacroClaim] = Field(default_factory=list)
    coverage: list[MacroCoverageResult] = Field(default_factory=list)
    summary_facts: list[MacroSummaryFact] = Field(default_factory=list)
    interpretations: list[MacroInterpretation] = Field(default_factory=list)
    score_components: MacroScoreComponents | None = None
    source_degraded: bool = False
    market_probability_supported: bool = False
    assessment_supported: bool
    informational_only: Literal[True] = True
    trade_permission: Literal[False] = False
    technical_model_input_allowed: Literal[False] = False
    decision_engine_input_allowed: Literal[False] = False
    historical_profit_tuning_allowed: Literal[False] = False

    @field_validator("decision_as_of_utc")
    @classmethod
    def require_utc_as_of(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decision_as_of_utc must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("risk_score")
    @classmethod
    def risk_score_matches_level(cls, value: int, info) -> int:
        level = info.data.get("risk_level")
        if level is not None and value != level.score:
            raise ValueError("risk_score must match the frozen five-level risk scale")
        return value

    @model_validator(mode="after")
    def validate_complete_risk_scale(self) -> MacroRiskResponse:
        actual = [(band.risk_level, band.risk_score) for band in self.risk_scale]
        expected = [(band.risk_level, band.risk_score) for band in FROZEN_MACRO_RISK_SCALE]
        if actual != expected:
            raise ValueError("risk_scale must contain the frozen five levels in score order")
        return self


class MacroEventResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: str
    scheduled_release_at_utc: datetime
    actual_release_at_utc: datetime | None = None
    high_impact: bool = True
    source: str | None = None
    source_url: str
    source_date_label: str | None = None

    @field_validator("event_type")
    @classmethod
    def require_allowed_event_type(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in ALLOWED_EVENT_TYPES:
            raise ValueError(f"Unsupported macro event type: {value}")
        return normalized

    @field_validator("scheduled_release_at_utc", "actual_release_at_utc")
    @classmethod
    def require_timezone_aware_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("macro event timestamps must be timezone-aware")
        return None if value is None else value.astimezone(UTC)

    @field_validator("source_url")
    @classmethod
    def require_official_source_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_DOMAINS:
            raise ValueError("Macro events must use an approved official HTTPS source URL")
        return value


class MacroEventsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[MacroEventResponse]
    fetch_succeeded: bool
    retrieved_at_utc: datetime | None = None
    fresh_until_utc: datetime | None = None
    snapshot_sha256: str | None = None


class MacroProviderStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_id: str
    required: bool
    status: Literal["AVAILABLE", "UNAVAILABLE", "STALE", "INVALID"]
    reason_code: str | None = None
    updated_at_utc: datetime | None = None


class MacroCoverageSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot: str
    required: bool = True
    status: Literal["COVERED", "DEGRADED", "MISSING", "CONFLICTING", "STALE"]
    supporting_claim_ids: list[str] = Field(default_factory=list)
    latest_observation_at_utc: datetime | None = None
    reason_codes: list[str] = Field(default_factory=list)


class MacroStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: Literal["macro_rag"] = "macro_rag"
    status: Literal["OK", "DEGRADED", "UNAVAILABLE"]
    assessment_supported: bool
    coverage_complete: bool
    providers: list[MacroProviderStatus]
    coverage: list[MacroCoverageSlot]
    last_success_at_utc: datetime | None = None


class MacroEvidenceResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment_id: str
    content_hash: str = Field(pattern=r"^[A-Fa-f0-9]{64}$")
    decision_as_of_utc: datetime
    citations: list[MacroEvidenceCitation]
    claims: list[MacroClaim] = Field(default_factory=list)
    coverage: list[MacroCoverageResult] = Field(default_factory=list)
    summary_facts: list[MacroSummaryFact] = Field(default_factory=list)
    interpretations: list[MacroInterpretation] = Field(default_factory=list)
    score_components: MacroScoreComponents | None = None
    assessment_supported: bool
