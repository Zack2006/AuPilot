"""Formal model-validation readiness endpoint."""

from datetime import date

from fastapi import APIRouter, Query

from backend.app.api.dependencies import (
    mn18_validation_evidence_service,
    mn18_validation_timeline_service,
    model_validation_service,
)
from backend.app.schemas.model_validation import (
    ModelValidationReport,
    ModelValidationStatus,
    PredictionRecordsResponse,
)


router = APIRouter(prefix="/model-validation", tags=["model-validation"])


@router.get("/mn18-oos")
def mn18_development_oos() -> dict:
    """Return hash-gated MN18 evidence without historical recomputation."""
    return mn18_validation_evidence_service().evidence()


@router.get("/mn18-timeline")
def mn18_validation_timeline() -> dict:
    """Return the verified MN18 OOS/bridge/forward timeline."""
    return mn18_validation_timeline_service().timeline()


@router.post("/mn18-timeline/refresh")
def refresh_mn18_validation_timeline() -> dict:
    """Refresh Databento and MN18 forward state idempotently."""
    return mn18_validation_timeline_service().refresh()


@router.get("/status", response_model=ModelValidationStatus)
def status() -> ModelValidationStatus:
    return model_validation_service().status()


@router.get("/records", response_model=PredictionRecordsResponse)
def records(
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
) -> PredictionRecordsResponse:
    return model_validation_service().records(start_date=start_date, end_date=end_date)


@router.get("/report", response_model=ModelValidationReport)
def report(
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
    initial_capital: float = Query(default=100_000.0, gt=0, le=1_000_000_000),
    transaction_cost_bps: float = Query(default=2.0, ge=0, le=1000),
    slippage_bps: float = Query(default=0.0, ge=0, le=1000),
) -> ModelValidationReport:
    return model_validation_service().report(
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
    )
