"""Databento-only formal gold daily market endpoints."""

from __future__ import annotations

import json
from datetime import date

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.app.api.dependencies import market_service
from backend.app.services.feature_service import FeatureService


router = APIRouter(prefix="/market", tags=["market"])


def _records(frame: pd.DataFrame) -> list[dict]:
    return json.loads(frame.to_json(orient="records", date_format="iso", date_unit="ms"))


@router.get("/latest")
def latest():
    return market_service().latest(refresh=False)


@router.get("/daily")
def daily(limit: int = Query(365, ge=20, le=1000)) -> dict:
    return gold_daily(start=None, end=None, limit=limit)


@router.get("/gold/daily")
def gold_daily(
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
    limit: int | None = Query(default=None, ge=20, le=10000),
) -> dict:
    if start is not None and end is not None and start > end:
        raise HTTPException(status_code=422, detail="MARKET_DATE_RANGE_INVALID")
    result = market_service().get_history(refresh=False)
    # Calculate the shared display indicators from the complete causal history
    # before applying a presentation range. This keeps the first row in a
    # requested window from losing its rolling context.
    frame = FeatureService().calculate(result.frame.copy())
    dates = frame["ts_event_utc"].dt.date
    if start is not None:
        frame = frame.loc[dates >= start]
        dates = frame["ts_event_utc"].dt.date
    if end is not None:
        frame = frame.loc[dates <= end]
    if limit is not None:
        frame = frame.tail(limit)
    return {
        "metadata": result.metadata.model_dump(mode="json"),
        "count": len(frame),
        "items": _records(frame),
    }


@router.get("/gold/status")
def gold_status() -> dict:
    """Cheap cache identity used to invalidate the shared frontend payload."""
    return market_service().cache_metadata().model_dump(mode="json")


@router.get("/history")
def history(limit: int = Query(365, ge=20, le=1000)) -> dict:
    """Compatibility route with the same canonical Databento contract as /daily."""
    return gold_daily(start=None, end=None, limit=limit)


@router.post("/refresh")
def refresh():
    """The only HTTP endpoint allowed to incur a formal Databento data request."""
    return market_service().latest(refresh=True)


@router.get("/technical-history")
def technical_history(days: int = Query(365, ge=20, le=1000)) -> dict:
    """Compatibility alias backed by the same Databento-only product route."""
    return gold_daily(start=None, end=None, limit=days)
