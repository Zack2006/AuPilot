"""Contracts for formal out-of-sample model-validation readiness."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class ModelValidationStatus(BaseModel):
    formal_evaluation_available: bool
    status: Literal["READY", "NOT_READY"]
    reason_codes: list[str]
    market_data_ready: bool
    model_manifest_available: bool
    prediction_record_count: int
    training_data_cutoff_utc: datetime | None = None
    model_activated_at_utc: datetime | None = None
    model_version: str | None = None
    feature_schema_version: str | None = None
    out_of_sample_start_date: date | None = None
    model_manifest_sha256: str | None = None
    prediction_log_head_sha256: str | None = None
    prediction_from_date: date | None = None
    prediction_until_date: date | None = None
    service_mode: Literal["formal-prediction", "unavailable"]
    mock_results_included: bool = False


class PredictionEvaluationRecord(BaseModel):
    prediction_id: str
    prediction_for_date: date
    generated_at_utc: datetime
    market_data_as_of_utc: datetime
    model_version: str
    predicted_close: float
    predicted_low: float
    predicted_high: float
    expected_direction: Literal["UP", "DOWN", "NEUTRAL"]
    direction_confidence: float
    turning_point: Literal["NONE", "UP_TURN", "DOWN_TURN"]
    turning_point_confidence: float
    target_exposure: float
    actual_open: float | None = None
    actual_high: float | None = None
    actual_low: float | None = None
    actual_close: float | None = None
    actual_direction: Literal["UP", "DOWN", "NEUTRAL"] | None = None
    close_error: float | None = None
    absolute_percentage_error: float | None = None
    direction_hit: bool | None = None
    interval_covered: bool | None = None
    turning_point_hit: bool | None = None
    evaluation_status: Literal["EVALUATED", "MARKET_BAR_MISSING", "PRIOR_BAR_MISSING"]
    market_data_sha256: str
    record_sha256: str


class PredictionRecordsResponse(BaseModel):
    schema_version: Literal["aurumpilot.model_validation_records.v1"] = "aurumpilot.model_validation_records.v1"
    model_version: str
    model_manifest_sha256: str
    prediction_log_head_sha256: str
    market_data_sha256: str
    data_from_date: date
    data_until_date: date
    count: int = Field(ge=1)
    items: list[PredictionEvaluationRecord]
    mock_results_included: Literal[False] = False


class EquityCurvePoint(BaseModel):
    date: date
    model_equity: float
    buy_hold_equity: float
    model_cash: float
    buy_hold_cash: float
    model_units: float = Field(ge=0)
    buy_hold_units: float = Field(ge=0)
    target_exposure: float
    prediction_id: str | None = None
    prediction_status: Literal["PUBLISHED", "MISSING_CARRIED_FORWARD"]


class StrategyPerformance(BaseModel):
    initial_capital: float
    final_equity: float
    cumulative_return: float
    max_drawdown: float
    trade_count: int = Field(ge=0)
    turnover: float = Field(ge=0)
    transaction_cost: float = Field(ge=0)
    slippage_cost: float = Field(ge=0)
    ending_cash: float
    ending_position_units: float = Field(ge=0)


class PredictionAccuracy(BaseModel):
    evaluated_predictions: int = Field(ge=0)
    direction_eligible: int = Field(ge=0)
    direction_hits: int = Field(ge=0)
    direction_hit_rate: float | None = None
    mean_absolute_error: float | None = None
    mean_absolute_percentage_error: float | None = None
    interval_coverage_rate: float | None = None
    turning_point_eligible: int = Field(ge=0)
    turning_point_hits: int = Field(ge=0)
    turning_point_hit_rate: float | None = None
    daily_prediction_coverage_rate: float = Field(ge=0, le=1)


class ModelValidationReport(BaseModel):
    schema_version: Literal["aurumpilot.model_validation_report.v1"] = "aurumpilot.model_validation_report.v1"
    requested_start_date: date
    requested_end_date: date
    effective_start_date: date
    effective_end_date: date
    initial_capital: float = Field(gt=0)
    transaction_cost_bps: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)
    model_strategy: StrategyPerformance
    buy_and_hold: StrategyPerformance
    excess_final_equity: float
    excess_return: float
    drawdown_difference: float
    accuracy: PredictionAccuracy
    equity_curve: list[EquityCurvePoint]
    prediction_records: list[PredictionEvaluationRecord]
    missing_prediction_dates: list[date]
    market_data_sha256: str
    model_manifest_sha256: str
    prediction_log_head_sha256: str
    model_version: str
    training_data_cutoff_utc: datetime
    model_activated_at_utc: datetime
    out_of_sample_start_date: date
    execution_rule: Literal["NEXT_SESSION_OPEN"] = "NEXT_SESSION_OPEN"
    macro_rag_used: Literal[False] = False
    informational_only: Literal[True] = True
    mock_results_included: Literal[False] = False
