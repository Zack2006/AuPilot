"""Formal dual-model manifest and append-only prediction-log contracts."""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SHA256_PATTERN = r"^[A-Fa-f0-9]{64}$"


class FormalModelArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["turning_point", "price"]
    relative_path: str = Field(min_length=1, max_length=240)
    sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("sha256")
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        return value.upper()


class TurningPointEvaluationRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(min_length=1, max_length=120)
    method: Literal["FORWARD_CLOSE_MOVE"] = "FORWARD_CLOSE_MOVE"
    evaluation_window_sessions: int = Field(ge=1, le=60)
    minimum_move_fraction: float = Field(gt=0, le=0.5)
    minimum_signal_persistence_sessions: int = Field(ge=1, le=20)


class FormalModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["aurumpilot.formal_model_manifest.v1"]
    service_mode: Literal["formal-prediction"]
    model_version: str = Field(min_length=1, max_length=120)
    feature_schema_version: str = Field(min_length=1, max_length=120)
    training_data_cutoff_utc: datetime
    activated_at_utc: datetime
    out_of_sample_start_date: date
    code_sha: str = Field(pattern=r"^[0-9a-f]{40}([0-9a-f]{24})?$")
    artifacts: list[FormalModelArtifact] = Field(min_length=2, max_length=2)
    turning_point_evaluation: TurningPointEvaluationRule

    @field_validator("code_sha")
    @classmethod
    def normalize_code_sha(cls, value: str) -> str:
        return value.lower()

    @field_validator("training_data_cutoff_utc", "activated_at_utc")
    @classmethod
    def manifest_timestamps_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("formal model manifest timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_dual_model_boundary(self) -> "FormalModelManifest":
        if {artifact.role for artifact in self.artifacts} != {"turning_point", "price"}:
            raise ValueError("manifest must contain exactly one turning_point and one price model")
        if self.out_of_sample_start_date <= self.training_data_cutoff_utc.date():
            raise ValueError("out_of_sample_start_date must be after the training cutoff date")
        if self.activated_at_utc <= self.training_data_cutoff_utc:
            raise ValueError("activated_at_utc must be after the training cutoff")
        if self.activated_at_utc.date() > self.out_of_sample_start_date:
            raise ValueError("activated_at_utc cannot be after the out-of-sample start date")
        return self

    def artifact(self, role: Literal["turning_point", "price"]) -> FormalModelArtifact:
        return next(item for item in self.artifacts if item.role == role)


class FormalPredictionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: str = Field(pattern=r"^pred_[A-Za-z0-9_-]{8,80}$")
    generated_at_utc: datetime
    prediction_for_date: date
    market_data_as_of_utc: datetime
    training_data_cutoff_utc: datetime
    model_version: str = Field(min_length=1, max_length=120)
    feature_schema_version: str = Field(min_length=1, max_length=120)
    model_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    turning_point_model_sha256: str = Field(pattern=SHA256_PATTERN)
    price_model_sha256: str = Field(pattern=SHA256_PATTERN)
    market_data_sha256: str = Field(pattern=SHA256_PATTERN)
    code_sha: str = Field(pattern=r"^[0-9a-f]{40}([0-9a-f]{24})?$")
    predicted_close: float = Field(gt=0)
    predicted_low: float = Field(gt=0)
    predicted_high: float = Field(gt=0)
    expected_direction: Literal["UP", "DOWN", "NEUTRAL"]
    direction_confidence: float = Field(ge=0, le=1)
    turning_point: Literal["NONE", "UP_TURN", "DOWN_TURN"]
    turning_point_confidence: float = Field(ge=0, le=1)
    target_exposure: float = Field(ge=0, le=1)
    execution_rule: Literal["NEXT_SESSION_OPEN"] = "NEXT_SESSION_OPEN"
    supersedes_prediction_id: str | None = Field(default=None, pattern=r"^pred_[A-Za-z0-9_-]{8,80}$")
    revision_reason: str | None = Field(default=None, min_length=1, max_length=500)
    service_mode: Literal["formal-prediction"] = "formal-prediction"
    published: Literal[True] = True

    @field_validator(
        "model_manifest_sha256",
        "turning_point_model_sha256",
        "price_model_sha256",
        "market_data_sha256",
    )
    @classmethod
    def normalize_sha256(cls, value: str) -> str:
        return value.upper()

    @field_validator("code_sha")
    @classmethod
    def normalize_code_sha(cls, value: str) -> str:
        return value.lower()

    @field_validator("revision_reason")
    @classmethod
    def normalize_revision_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("revision_reason cannot be blank")
        return normalized

    @field_validator("generated_at_utc", "market_data_as_of_utc", "training_data_cutoff_utc")
    @classmethod
    def timestamps_must_be_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("formal prediction timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_timepoint_and_range(self) -> "FormalPredictionCreate":
        if self.training_data_cutoff_utc > self.market_data_as_of_utc:
            raise ValueError("training cutoff cannot be after market data as-of")
        if self.market_data_as_of_utc > self.generated_at_utc:
            raise ValueError("prediction cannot be generated before its market data as-of")
        if self.prediction_for_date <= self.market_data_as_of_utc.date():
            raise ValueError("prediction_for_date must be after market data as-of")
        if self.generated_at_utc.date() >= self.prediction_for_date:
            raise ValueError("formal predictions must be generated before the target UTC date")
        if not self.predicted_low <= self.predicted_close <= self.predicted_high:
            raise ValueError("predicted_close must be inside predicted_low/high")
        if (self.supersedes_prediction_id is None) != (self.revision_reason is None):
            raise ValueError("revision_reason is required exactly when a prediction is superseded")
        return self


class FormalPredictionRecord(FormalPredictionCreate):
    previous_record_sha256: str = Field(pattern=SHA256_PATTERN)
    record_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator("previous_record_sha256", "record_sha256")
    @classmethod
    def normalize_chain_sha256(cls, value: str) -> str:
        return value.upper()


class FormalDualModelOutput(BaseModel):
    """Business output accepted from a dual-model adapter before audit fields are attached."""

    model_config = ConfigDict(extra="forbid")

    prediction_for_date: date
    predicted_close: float = Field(gt=0)
    predicted_low: float = Field(gt=0)
    predicted_high: float = Field(gt=0)
    expected_direction: Literal["UP", "DOWN", "NEUTRAL"]
    direction_confidence: float = Field(ge=0, le=1)
    turning_point: Literal["NONE", "UP_TURN", "DOWN_TURN"]
    turning_point_confidence: float = Field(ge=0, le=1)
    target_exposure: float = Field(ge=0, le=1)
    revision_reason: str | None = Field(default=None, min_length=1, max_length=500)

    @field_validator("revision_reason")
    @classmethod
    def normalize_revision_reason(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("revision_reason cannot be blank")
        return normalized

    @model_validator(mode="after")
    def validate_price_range(self) -> "FormalDualModelOutput":
        if not self.predicted_low <= self.predicted_close <= self.predicted_high:
            raise ValueError("predicted_close must be inside predicted_low/high")
        return self
