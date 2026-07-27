from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aupilot.core.enums import MacroRiskLevel

OFFICIAL_DOMAINS = {
    "federalreserve.gov",
    "www.federalreserve.gov",
    "newyorkfed.org",
    "www.newyorkfed.org",
    "markets.newyorkfed.org",
    "home.treasury.gov",
    "www.treasury.gov",
    "bls.gov",
    "www.bls.gov",
    "api.bls.gov",
    "bea.gov",
    "www.bea.gov",
    "apps.bea.gov",
    "cmegroup.com",
    "www.cmegroup.com",
    "fred.stlouisfed.org",
    "api.stlouisfed.org",
}
TIER_B_DOMAINS = {
    "reuters.com",
    "www.reuters.com",
    "apnews.com",
    "www.apnews.com",
    "ft.com",
    "www.ft.com",
    "wsj.com",
    "www.wsj.com",
    "bloomberg.com",
    "www.bloomberg.com",
    "cnbc.com",
    "www.cnbc.com",
}
ALLOWED_EVIDENCE_DOMAINS = OFFICIAL_DOMAINS | TIER_B_DOMAINS
ALLOWED_EVENT_TYPES = {"FOMC", "RATES", "CPI", "PCE", "NFP"}
COVERAGE_SLOTS = {
    "FOMC.calendar",
    "FOMC.latest_official_decision",
    "FOMC.market_expectation",
    "RATES.effr",
    "RATES.nominal_2y",
    "RATES.nominal_10y",
    "RATES.real_10y",
    "RATES.breakeven_proxy_10y",
    "CPI.next_release_time",
    "CPI.latest_initial_release",
    "PCE.next_release_time",
    "PCE.latest_initial_release",
    "NFP.next_release_time",
    "NFP.latest_initial_release",
}


class MacroDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    source: str
    provider_id: str | None = None
    metric_id: str | None = None
    source_tier: Literal["A", "B", "C"] = "A"
    event_type: str
    title: str
    canonical_url: str
    content: str = Field(min_length=20)
    published_at_utc: datetime | None = None
    retrieved_at_utc: datetime
    eligible_from_utc: datetime
    content_sha256: str
    replay_eligible: bool = False
    first_seen_at_utc: datetime | None = None
    retrieval_method: Literal["API", "RSS", "ICS", "HTML", "SEARCH"] = "HTML"
    official_primary: bool = True
    revision_status: Literal["INITIAL", "REVISED", "UNKNOWN"] = "UNKNOWN"
    independence_key: str | None = None
    near_duplicate_group: str | None = None

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in ALLOWED_EVENT_TYPES:
            raise ValueError(f"Unsupported macro event type: {value}")
        return normalized

    @field_validator("canonical_url")
    @classmethod
    def validate_evidence_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_EVIDENCE_DOMAINS:
            raise ValueError("Macro documents must use an approved evidence HTTPS domain")
        return value

    @field_validator(
        "retrieved_at_utc", "eligible_from_utc", "published_at_utc", "first_seen_at_utc"
    )
    @classmethod
    def validate_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("Macro timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_source_tier(self) -> MacroDocument:
        hostname = urlparse(self.canonical_url).hostname
        if self.source_tier == "A" and hostname not in OFFICIAL_DOMAINS:
            raise ValueError("Tier A documents must use an official or raw-market domain")
        if self.source_tier == "B" and hostname not in TIER_B_DOMAINS:
            raise ValueError("Tier B documents must use an allowlisted editorial domain")
        if self.official_primary is not (self.source_tier == "A"):
            raise ValueError("official_primary must agree with source_tier")
        return self


class EvidenceCitation(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    event_type: str
    title: str
    canonical_url: str
    eligible_from_utc: datetime
    content_sha256: str
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
    claim_ids: tuple[str, ...] = ()

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in ALLOWED_EVENT_TYPES:
            raise ValueError(f"Unsupported macro citation event type: {value}")
        return normalized

    @field_validator("canonical_url")
    @classmethod
    def validate_official_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_EVIDENCE_DOMAINS:
            raise ValueError("Macro citations must use an approved evidence HTTPS domain")
        return value

    @field_validator("content_sha256")
    @classmethod
    def validate_content_hash(cls, value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 64 or any(character not in "0123456789ABCDEF" for character in normalized):
            raise ValueError("Macro citation content_sha256 must be a SHA-256 hex digest")
        return normalized

    @model_validator(mode="after")
    def validate_observation_citation(self) -> EvidenceCitation:
        fields = (self.series_id, self.observation_date, self.observation_value)
        if self.evidence_kind == "OBSERVATION" and any(value is None for value in fields):
            raise ValueError("Observation citations require series, date, and value")
        if self.evidence_kind == "DOCUMENT" and any(value is not None for value in fields):
            raise ValueError("Document citations cannot contain structured observation fields")
        return self


class MacroClaim(BaseModel):
    """One immutable atomic assertion from one independently counted source."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    slot: str
    event_type: str
    claim_type: str
    normalized_value: str = Field(min_length=1)
    display_text: str = Field(min_length=1)
    value: float | str | dict[str, Any] | list[Any] | None = None
    unit: str | None = None
    reference_period: str | None = None
    observed_at_utc: datetime | None = None
    source_record_id: str
    provider_id: str
    source_tier: Literal["A", "B", "C"]
    canonical_url: str
    published_at_utc: datetime | None = None
    first_seen_at_utc: datetime
    retrieved_at_utc: datetime
    eligible_from_utc: datetime
    content_sha256: str = Field(pattern=r"^[A-F0-9]{64}$")
    retrieval_method: Literal["API", "RSS", "ICS", "HTML", "SEARCH"]
    official_primary: bool
    revision_status: Literal["INITIAL", "REVISED", "UNKNOWN"] = "UNKNOWN"
    independence_key: str = Field(min_length=1)
    near_duplicate_group: str | None = None

    @field_validator("slot")
    @classmethod
    def validate_slot(cls, value: str) -> str:
        if value not in COVERAGE_SLOTS:
            raise ValueError(f"Unsupported coverage slot: {value}")
        return value

    @field_validator("event_type")
    @classmethod
    def validate_claim_event_type(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in ALLOWED_EVENT_TYPES:
            raise ValueError(f"Unsupported macro claim event type: {value}")
        return normalized

    @field_validator(
        "observed_at_utc",
        "published_at_utc",
        "first_seen_at_utc",
        "retrieved_at_utc",
        "eligible_from_utc",
    )
    @classmethod
    def validate_claim_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("Macro claim timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_claim_source(self) -> MacroClaim:
        hostname = urlparse(self.canonical_url).hostname
        if urlparse(self.canonical_url).scheme != "https":
            raise ValueError("Macro claim sources must use HTTPS")
        if self.source_tier == "A" and hostname not in OFFICIAL_DOMAINS:
            raise ValueError("Tier A claims require an official or raw-market source")
        if self.source_tier == "B" and hostname not in TIER_B_DOMAINS:
            raise ValueError("Tier B claims require an allowlisted editorial source")
        if self.source_tier == "C" or self.retrieval_method == "SEARCH":
            raise ValueError("Tier C/search discoveries cannot become rating claims")
        if self.official_primary is not (self.source_tier == "A"):
            raise ValueError("official_primary must agree with source_tier")
        if self.published_at_utc and self.eligible_from_utc < self.published_at_utc:
            raise ValueError("A claim cannot become eligible before publication")
        if self.eligible_from_utc < self.first_seen_at_utc:
            raise ValueError("A claim cannot become eligible before first_seen_at_utc")
        return self


class MacroCoverageResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    slot: str
    required: bool
    status: Literal["COVERED", "DEGRADED", "MISSING", "CONFLICTING", "STALE"]
    supporting_claim_ids: tuple[str, ...] = ()
    latest_observation_at_utc: datetime | None = None
    reason_codes: tuple[str, ...] = ()


class MacroSummaryFact(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    claim_ids: tuple[str, ...] = Field(min_length=1)


class MacroScoreComponents(BaseModel):
    model_config = ConfigDict(frozen=True)

    calendar_score: int = Field(ge=1, le=5)
    proximity_score: int = Field(default=1, ge=1, le=5)
    impact_strength_score: int = Field(default=1, ge=1, le=5)
    rate_relevance_score: int = Field(default=1, ge=1, le=5)
    release_cooldown_score: int = Field(ge=1, le=5)
    expectation_uncertainty_score: int = Field(ge=1, le=5)
    evidence_quality_score: int = Field(ge=1, le=5)


class MacroAssessment(BaseModel):
    model_config = ConfigDict(frozen=True)

    risk_level: MacroRiskLevel
    risk_score: int = Field(ge=1, le=5)
    decision_as_of_utc: datetime
    reason_codes: tuple[str, ...]
    news_summary: tuple[str, ...] = Field(min_length=1)
    citations: tuple[EvidenceCitation, ...] = ()
    claims: tuple[MacroClaim, ...] = ()
    coverage: tuple[MacroCoverageResult, ...] = ()
    summary_facts: tuple[MacroSummaryFact, ...] = ()
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
    def validate_as_of(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("decision_as_of_utc must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_risk_score(self) -> MacroAssessment:
        if self.risk_score != self.risk_level.score:
            raise ValueError("risk_score must match the frozen five-level risk scale")
        if any(not value.strip() for value in self.news_summary):
            raise ValueError("news_summary entries cannot be blank")
        return self


class MacroObservation(BaseModel):
    """One immutable provider observation with a point-in-time cutoff."""

    model_config = ConfigDict(frozen=True)

    observation_id: str
    series_id: str
    observation_date: date
    value: float
    realtime_start: date
    realtime_end: date
    eligible_from_utc: datetime
    retrieved_at_utc: datetime
    source_url: str
    source_payload_sha256: str = Field(pattern=r"^[A-F0-9]{64}$")
    initial_release_only: bool
    provider_id: str = "fred_alfred"
    metric_id: str | None = None
    source_tier: Literal["A", "B", "C"] = "A"
    published_at_utc: datetime | None = None
    first_seen_at_utc: datetime | None = None
    retrieval_method: Literal["API", "RSS", "ICS", "HTML", "SEARCH"] = "API"
    official_primary: bool = True
    revision_status: Literal["INITIAL", "REVISED", "UNKNOWN"] = "UNKNOWN"
    unit: str | None = None
    independence_key: str | None = None

    @field_validator("series_id")
    @classmethod
    def validate_series_id(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized or not normalized.replace("_", "").isalnum():
            raise ValueError("Invalid provider series or metric id")
        return normalized

    @field_validator("source_url")
    @classmethod
    def validate_source_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_EVIDENCE_DOMAINS:
            raise ValueError("Macro observations must come from an approved evidence HTTPS domain")
        if parsed.query:
            raise ValueError("Macro observation source_url must not persist API parameters")
        return value

    @field_validator(
        "eligible_from_utc", "retrieved_at_utc", "published_at_utc", "first_seen_at_utc"
    )
    @classmethod
    def validate_observation_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("Macro observation timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_observation_source_tier(self) -> MacroObservation:
        hostname = urlparse(self.source_url).hostname
        if self.source_tier == "A" and hostname not in OFFICIAL_DOMAINS:
            raise ValueError("Tier A observations require an official source")
        if self.source_tier == "B" and hostname not in TIER_B_DOMAINS:
            raise ValueError("Tier B observations require an allowlisted source")
        if self.official_primary is not (self.source_tier == "A"):
            raise ValueError("official_primary must agree with source_tier")
        return self
