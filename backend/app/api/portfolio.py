"""Portfolio settings and derived-state routes. / 持仓设置与派生状态路由。"""

from fastapi import APIRouter

from backend.app.api.dependencies import portfolio_settings_service, position_accounting_service
from backend.app.schemas.portfolio import PortfolioInput
from backend.app.schemas.portfolio_settings import PortfolioSettingsCreate, PortfolioSettingsVersion
from backend.app.schemas.transaction import PerformancePoint, PortfolioSnapshot

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


@router.post("/validate")
def validate_portfolio(portfolio: PortfolioInput) -> dict:
    return {"valid": True, "portfolio": portfolio}


@router.get("/settings/current", response_model=PortfolioSettingsVersion)
def current_settings() -> PortfolioSettingsVersion:
    return portfolio_settings_service().current()


@router.get("/settings/history", response_model=list[PortfolioSettingsVersion])
def settings_history() -> list[PortfolioSettingsVersion]:
    return portfolio_settings_service().history()


@router.post("/settings", response_model=PortfolioSettingsVersion, status_code=201)
def create_settings(payload: PortfolioSettingsCreate) -> PortfolioSettingsVersion:
    return portfolio_settings_service().create(payload)


@router.get("/snapshot", response_model=PortfolioSnapshot)
def portfolio_snapshot() -> PortfolioSnapshot:
    return position_accounting_service().snapshot()


@router.get("/performance", response_model=list[PerformancePoint])
def portfolio_performance() -> list[PerformancePoint]:
    return position_accounting_service().performance()
