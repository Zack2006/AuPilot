from datetime import UTC, datetime

from fastapi import APIRouter, HTTPException, Query

from backend.app.api.dependencies import macro_service
from backend.app.schemas.macro import (
    MacroEvidenceResponse,
    MacroEventsResponse,
    MacroRiskResponse,
    MacroStatusResponse,
)

router = APIRouter(prefix="/macro", tags=["macro"])


@router.get("/events", response_model=MacroEventsResponse)
def events() -> MacroEventsResponse:
    return macro_service().events()


@router.get("/calendar", response_model=MacroEventsResponse)
def calendar(days: int = Query(default=15, ge=1, le=90)) -> MacroEventsResponse:
    return macro_service().events(days=days)


@router.get("/risk", response_model=MacroRiskResponse)
def risk(as_of: datetime | None = Query(default=None)) -> MacroRiskResponse:
    now = datetime.now(UTC)
    if as_of is not None:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must include an explicit timezone")
        if as_of.astimezone(UTC) > now:
            raise ValueError("future as_of is not allowed")
    return macro_service().assess(as_of)


@router.get("/assessment/latest", response_model=MacroRiskResponse)
def latest_assessment() -> MacroRiskResponse:
    return macro_service().assess()


@router.get("/status", response_model=MacroStatusResponse)
def status() -> MacroStatusResponse:
    return macro_service().status()


@router.get("/evidence/{assessment_id}", response_model=MacroEvidenceResponse)
def evidence(assessment_id: str) -> MacroEvidenceResponse:
    try:
        return macro_service().evidence(assessment_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
