"""Receiver schemas for the hash-gated MN18 + PN02 dual-model contract."""

from __future__ import annotations

import os
from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SHA256_PATTERN = r"^[A-Fa-f0-9]{64}$"
TECHNICAL_SCHEMA_VERSION = "aupilot.mn18_pn02.dual_outlook.v1"
DAILY_UNIT_ID = "CANONICAL_GC_UTC_DAILY_BUCKET_V1"
MN18_MODEL_ID = "MN18_THREE_TOP_EXPERT_FORWARD_SHADOW_CANDIDATE_V1"
MN18_MODEL_VERSION = os.getenv(
    "AUPILOT_MN18_MODEL_VERSION",
    "MN18-FULL-DEVELOPMENT-FORWARD-20260727T114605Z",
)
PN02_MODEL_ID = "PN02_RAW_MEDIAN_CAUSAL_GEOMETRY_RESIDUAL_LIGHTGBM_V1"
PN02_MODEL_VERSION = os.getenv(
    "AUPILOT_PN02_MODEL_VERSION",
    "PN02-FULL-DEVELOPMENT-20260727T131932366431Z",
)
EXPECTED_COMPONENT_HASHES = {
    "dual_package_manifest_sha256": os.getenv(
        "AUPILOT_DUAL_PACKAGE_MANIFEST_SHA256",
        "D584DA5206C12D332984160CC6D97178EDE062E5CB6263120446ED8B7A635B9E",
    ).upper(),
    "mn18_bundle_manifest_sha256": os.getenv(
        "AUPILOT_MN18_BUNDLE_MANIFEST_SHA256",
        "7325C34BFFF0CD77D658ABED413A23C799F6AE10F693012CB918C1D391702D3D",
    ).upper(),
    "mn18_joblib_sha256": os.getenv(
        "AUPILOT_MN18_JOBLIB_SHA256",
        "C497E8251BF7A740023CE64C0CFC8CA10CEC20C40AC2BADB9A49AF4D380716A6",
    ).upper(),
    "pn02_bundle_sha256": os.getenv(
        "AUPILOT_PN02_BUNDLE_SHA256",
        "E1FC8FC6849C014D61E284504EBE4070730539B919482EF7178A7AFD08F3E9F8",
    ).upper(),
}
SEVEN_CLASSES = (
    "NORMAL",
    "TOP_L1",
    "TOP_L2",
    "TOP_L3",
    "BOTTOM_L1",
    "BOTTOM_L2",
    "BOTTOM_L3",
)


class TechnicalDailyOHLC(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trade_date: date
    open: float = Field(gt=0, allow_inf_nan=False)
    high: float = Field(gt=0, allow_inf_nan=False)
    low: float = Field(gt=0, allow_inf_nan=False)
    close: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_candle(self) -> "TechnicalDailyOHLC":
        if self.high < max(self.open, self.close):
            raise ValueError("technical input high violates OHLC")
        if self.low > min(self.open, self.close):
            raise ValueError("technical input low violates OHLC")
        return self


class TechnicalEvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    daily_history: list[TechnicalDailyOHLC] = Field(min_length=60)
    current_gold_weight: float = Field(ge=0.5, le=1.0)
    outstanding_top_inventory_pp: float = Field(ge=0, le=50)
    as_of_utc: datetime
    display_timezone: str = Field(default="Asia/Singapore", min_length=1)

    @field_validator("as_of_utc")
    @classmethod
    def as_of_must_be_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("as_of_utc must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def history_must_be_unique_and_increasing(
        self,
    ) -> "TechnicalEvaluationRequest":
        dates = [bar.trade_date for bar in self.daily_history]
        if dates != sorted(set(dates)):
            raise ValueError(
                "technical input dates must be unique and strictly increasing"
            )
        if dates[0] != date(2010, 6, 7):
            raise ValueError("technical input must start at 2010-06-07")
        return self


class TechnicalInterval(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lower: float = Field(gt=0, allow_inf_nan=False)
    upper: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_order(self) -> "TechnicalInterval":
        if self.lower > self.upper:
            raise ValueError("conditional interval lower exceeds upper")
        return self


class TechnicalOHLCPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    open: float = Field(gt=0, allow_inf_nan=False)
    high: float = Field(gt=0, allow_inf_nan=False)
    low: float = Field(gt=0, allow_inf_nan=False)
    close: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_candle(self) -> "TechnicalOHLCPoint":
        if self.high < max(self.open, self.close):
            raise ValueError("conditional high violates OHLC")
        if self.low > min(self.open, self.close):
            raise ValueError("conditional low violates OHLC")
        return self


class TechnicalMarginalIntervals(BaseModel):
    model_config = ConfigDict(extra="forbid")
    open: TechnicalInterval
    high: TechnicalInterval
    low: TechnicalInterval
    close: TechnicalInterval


class TechnicalConditionalScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_label: Literal["TOP_ACTION_ZONE", "BOTTOM_ACTION_ZONE"]
    point: TechnicalOHLCPoint
    marginal_80pct_intervals: TechnicalMarginalIntervals | None
    controls_trading: Literal[False]


class TechnicalProbabilityRow(BaseModel):
    model_config = ConfigDict(extra="forbid")
    horizon_index: int = Field(ge=1, le=21)
    target_bucket: date
    p_normal: float = Field(ge=0, le=1)
    p_top_l1: float = Field(ge=0, le=1)
    p_top_l2: float = Field(ge=0, le=1)
    p_top_l3: float = Field(ge=0, le=1)
    p_bottom_l1: float = Field(ge=0, le=1)
    p_bottom_l2: float = Field(ge=0, le=1)
    p_bottom_l3: float = Field(ge=0, le=1)
    p_top: float = Field(ge=0, le=1)
    p_bottom: float = Field(ge=0, le=1)
    display_class: Literal[
        "NORMAL",
        "TOP_L1",
        "TOP_L2",
        "TOP_L3",
        "BOTTOM_L1",
        "BOTTOM_L2",
        "BOTTOM_L3",
    ]
    controls_trading: bool
    fixed_role_permission: Literal["TOP_ONLY", "BOTTOM_ONLY", "DISPLAY_ONLY"]

    @model_validator(mode="after")
    def validate_probabilities(self) -> "TechnicalProbabilityRow":
        values = (
            self.p_normal,
            self.p_top_l1,
            self.p_top_l2,
            self.p_top_l3,
            self.p_bottom_l1,
            self.p_bottom_l2,
            self.p_bottom_l3,
        )
        if abs(sum(values) - 1.0) > 1e-8:
            raise ValueError("seven-class probabilities do not sum to one")
        if abs(self.p_top - sum(values[1:4])) > 1e-8:
            raise ValueError("p_top differs from top class sum")
        if abs(self.p_bottom - sum(values[4:7])) > 1e-8:
            raise ValueError("p_bottom differs from bottom class sum")
        return self


class TechnicalPriceSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    horizon_index: int = Field(ge=1, le=21)
    target_bucket: date
    top_conditional: TechnicalConditionalScenario
    bottom_conditional: TechnicalConditionalScenario


class TechnicalPriceOutlook(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: Literal["AuPilot"]
    model_id: Literal["PN02_RAW_MEDIAN_CAUSAL_GEOMETRY_RESIDUAL_LIGHTGBM_V1"]
    model_version: str
    model_status: list[str]
    as_of_utc: datetime
    source_bucket: date
    daily_unit_id: Literal["CANONICAL_GC_UTC_DAILY_BUCKET_V1"]
    slot_count: Literal[21]
    slots: list[TechnicalPriceSlot] = Field(min_length=21, max_length=21)
    controls_trading: Literal[False]
    automatic_execution: Literal[False]
    advisory_only: Literal[True]
    input_contract: dict[str, Any]
    turning_integration_contract: dict[str, Any]
    turning_model_id: Literal[
        "MN18_THREE_TOP_EXPERT_FORWARD_SHADOW_CANDIDATE_V1"
    ]


class TechnicalAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["HOLD", "REDUCE_GOLD_WEIGHT"]
    automatic_execution: Literal[False]
    current_target_gold_weight: float = Field(ge=0.5, le=1.0)
    recommended_target_gold_weight: float = Field(ge=0.5, le=1.0)
    requested_delta_pp: float = Field(ge=-50, le=0)
    executed_delta_pp: float = Field(ge=-50, le=0)
    outstanding_top_inventory_pp_before: float = Field(ge=0, le=50)
    outstanding_top_inventory_pp_after: float = Field(ge=0, le=50)
    reason_code: str = Field(min_length=1)
    requires_actual_databento_bucket_open: Literal[True]

    @model_validator(mode="after")
    def validate_action(self) -> "TechnicalAction":
        delta = self.current_target_gold_weight - self.recommended_target_gold_weight
        if self.action == "HOLD" and abs(delta) > 1e-12:
            raise ValueError("HOLD must preserve target weight")
        if self.action == "REDUCE_GOLD_WEIGHT" and delta <= 0:
            raise ValueError("REDUCE_GOLD_WEIGHT must reduce target weight")
        if self.action == "REDUCE_GOLD_WEIGHT" and self.executed_delta_pp >= 0:
            raise ValueError("REDUCE_GOLD_WEIGHT requires negative executed delta")
        return self


class TechnicalHistory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rows: int = Field(ge=60)
    source_bucket: date
    source_bucket_complete: Literal[True]
    start_bucket: date


class TechnicalOutlook(BaseModel):
    """Exact native response from the official joint adapter."""

    model_config = ConfigDict(extra="forbid")
    project: Literal["AuPilot"]
    model_id: Literal["MN18_THREE_TOP_EXPERT_FORWARD_SHADOW_CANDIDATE_V1"]
    model_version: str
    model_status: list[str]
    as_of_utc: datetime
    history: TechnicalHistory
    input_contract: dict[str, Any]
    forecast_contract: dict[str, Any]
    probability_rows: list[TechnicalProbabilityRow] = Field(
        min_length=21, max_length=21
    )
    action: TechnicalAction
    scheduled_action_requests: list[dict[str, Any]]
    price_outlook: TechnicalPriceOutlook
    warnings: list[str]

    @model_validator(mode="after")
    def validate_dual_model_contract(self) -> "TechnicalOutlook":
        if self.model_version != MN18_MODEL_VERSION:
            raise ValueError("MN18 model version mismatch")
        if self.price_outlook.model_version != PN02_MODEL_VERSION:
            raise ValueError("PN02 model version mismatch")
        horizons = [row.horizon_index for row in self.probability_rows]
        targets = [row.target_bucket for row in self.probability_rows]
        if horizons != list(range(1, 22)) or targets != sorted(set(targets)):
            raise ValueError("MN18 21-slot order is invalid")
        permissions = [row.fixed_role_permission for row in self.probability_rows]
        if permissions != ["TOP_ONLY", "BOTTOM_ONLY"] + ["DISPLAY_ONLY"] * 19:
            raise ValueError("MN18 fixed role permissions changed")
        price_pairs = [
            (slot.horizon_index, slot.target_bucket)
            for slot in self.price_outlook.slots
        ]
        if price_pairs != list(zip(horizons, targets, strict=True)):
            raise ValueError("MN18/PN02 source-target-horizon alignment failed")
        if self.price_outlook.source_bucket != self.history.source_bucket:
            raise ValueError("MN18/PN02 source bucket mismatch")
        if self.price_outlook.as_of_utc != self.as_of_utc:
            raise ValueError("MN18/PN02 as_of mismatch")
        if self.action.current_target_gold_weight < self.action.recommended_target_gold_weight:
            raise ValueError("MN18 H1 cannot increase gold weight")
        return self


class TechnicalIssuanceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    issuance_id: str
    idempotency_key: str = Field(pattern=SHA256_PATTERN)
    status: Literal["SUCCESS", "FAILED"]
    issuance_kind: Literal[
        "PRE_FORWARD_PRODUCT_PREVIEW",
        "COMPLIANT_FORWARD",
        "FAILED_ATTEMPT",
    ]
    created_at_utc: datetime
    source_bucket: date | None
    source_bucket_end_utc: datetime | None
    issued_at_utc: datetime | None
    input_history_start: date | None
    input_history_end: date | None
    input_rows: int | None = Field(default=None, ge=1)
    current_gold_weight: float = Field(ge=0.5, le=1.0)
    outstanding_top_inventory_pp: float = Field(ge=0, le=50)
    dual_package_manifest_hash: str = Field(pattern=SHA256_PATTERN)
    mn18_manifest_hash: str = Field(pattern=SHA256_PATTERN)
    mn18_joblib_hash: str = Field(pattern=SHA256_PATTERN)
    pn02_bundle_hash: str = Field(pattern=SHA256_PATTERN)
    market_data_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    input_request_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    output: dict[str, Any] | None
    output_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    error_reason: str | None


class TechnicalEvaluationResult(BaseModel):
    issuance: TechnicalIssuanceRecord
    created: bool


class TechnicalRuntimeState(BaseModel):
    current_gold_weight: float = Field(ge=0.5, le=1.0)
    outstanding_top_inventory_pp: float = Field(ge=0, le=50)
    revision: int = Field(ge=0)
    latest_issuance_id: str | None
    updated_at_utc: datetime


class TechnicalFillRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    issuance_id: str = Field(min_length=1)
    request_id: str | None = Field(default=None, min_length=1)
    idempotency_key: str = Field(min_length=8, max_length=128)
    expected_revision: int = Field(ge=0)
    filled_at_utc: datetime
    fill_price: float = Field(gt=0)
    actual_delta_pp: float | None = Field(default=None, gt=0, le=50)
    confirmed: Literal[True]

    @field_validator("filled_at_utc")
    @classmethod
    def filled_at_must_be_aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("filled_at_utc must be timezone-aware")
        return value.astimezone(UTC)


class TechnicalFillRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fill_id: str
    issuance_id: str
    request_id: str | None = None
    side: Literal["TOP", "BOTTOM"] | None = None
    action: Literal["REDUCE_GOLD_WEIGHT", "REENTER_GOLD_WEIGHT"] | None = None
    idempotency_key: str
    revision_before: int = Field(ge=0)
    revision_after: int = Field(ge=1)
    weight_before: float = Field(ge=0.5, le=1.0)
    weight_after: float = Field(ge=0.5, le=1.0)
    inventory_before_pp: float = Field(ge=0, le=50)
    inventory_after_pp: float = Field(ge=0, le=50)
    actual_delta_pp: float = Field(gt=0, le=50)
    filled_at_utc: datetime
    fill_price: float = Field(gt=0)
    recorded_at_utc: datetime
