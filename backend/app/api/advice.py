"""Page-level Today Advice API and append-only user allocation snapshots."""

from fastapi import APIRouter, Query

from backend.app.api.dependencies import today_advice_service
from backend.app.schemas.advice import PortfolioAllocationSnapshotCreate


router = APIRouter(tags=["advice"])


@router.get("/advice/today")
def today_advice(
    timezone: str = Query(default="Asia/Shanghai", min_length=1, max_length=64),
) -> dict:
    return today_advice_service().compose(timezone)


@router.get("/advice/history")
def today_advice_history() -> dict:
    """Read immutable H1-only Today Advice audit records."""
    return today_advice_service().history()


@router.get("/user/portfolio-snapshots")
def portfolio_snapshot_history() -> dict:
    return today_advice_service().allocation_history()


@router.post("/user/portfolio-snapshots", status_code=201)
def create_portfolio_snapshot(
    payload: PortfolioAllocationSnapshotCreate,
) -> dict:
    service = today_advice_service()
    result = service.create_allocation_snapshot(payload)
    result["current_advice"] = service.compose("Asia/Shanghai")
    return result
