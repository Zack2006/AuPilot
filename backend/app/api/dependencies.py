"""Application dependency graph. / 应用依赖图。

Purpose / 文件用途: construct one module-level service graph for the modular monolith.
Inputs / 输入: environment-backed Settings and fixed demo user id.
Outputs / 输出: cached repositories and domain services used by thin FastAPI routes.
Business invariants / 业务约束: DecisionService is injected into recommendation, forecast and analytics;
no other service owns action thresholds.
Side effects / 副作用: factories may create required storage directories; migration occurs on service access.
Fallback behavior / 降级: prediction and RAG factories retain their explicit Mock adapters.
"""

from functools import lru_cache

from backend.app.core.config import get_settings
from backend.app.repositories.json_repository import JsonRepository
from backend.app.repositories.portfolio_repository import PortfolioRepository
from backend.app.repositories.technical_issuance_repository import (
    TechnicalIssuanceRepository,
)
from backend.app.services.cycle_service import TacticalCycleService
from backend.app.services.formal_prediction_publication_service import FormalPredictionPublicationService
from backend.app.services.mn18_validation_service import (
    MN18ValidationEvidenceService,
    MN18ValidationTimelineService,
)
from backend.app.services.macro_rag_service import MacroRAGServiceFactory
from backend.app.services.local_settings_service import LocalSettingsService
from backend.app.services.model_validation_service import ModelValidationService
from backend.app.services.market_data_service import MarketDataService
from backend.app.services.position_accounting_service import PositionAccountingService
from backend.app.services.portfolio_settings_service import PortfolioSettingsService
from backend.app.services.technical_issuance_service import TechnicalIssuanceService
from backend.app.services.today_advice_service import TodayAdviceService
from backend.app.services.transaction_ledger_service import TransactionLedgerService


def current_user_id() -> str:
    """Return the fixed MVP identity. / 返回当前 MVP 固定演示身份。"""
    return get_settings().demo_user_id


@lru_cache
def market_service() -> MarketDataService:
    return MarketDataService(get_settings(), local_settings_service())


@lru_cache
def macro_service():
    return MacroRAGServiceFactory.create(get_settings())


@lru_cache
def local_settings_service() -> LocalSettingsService:
    return LocalSettingsService(get_settings())


@lru_cache
def model_validation_service() -> ModelValidationService:
    return ModelValidationService(get_settings(), local_settings_service(), market_service())


@lru_cache
def mn18_validation_evidence_service() -> MN18ValidationEvidenceService:
    """Return the MN18 hash-gated historical evidence service."""
    return MN18ValidationEvidenceService(get_settings())


@lru_cache
def mn18_validation_timeline_service() -> MN18ValidationTimelineService:
    """Return the MN18 OOS/bridge/forward timeline service."""
    return MN18ValidationTimelineService(
        get_settings(),
        mn18_validation_evidence_service(),
        technical_issuance_service(),
        market_service(),
    )


@lru_cache
def formal_prediction_publication_service() -> FormalPredictionPublicationService:
    """Internal-only publisher for a future formal dual-model runner."""
    return FormalPredictionPublicationService(get_settings(), market_service())


@lru_cache
def technical_issuance_service() -> TechnicalIssuanceService:
    settings = get_settings()
    return TechnicalIssuanceService(
        settings,
        market_service(),
        repository=TechnicalIssuanceRepository(
            settings.storage_dir
            / "technical"
            / "mn18_pn02_issuances.sqlite"
        ),
    )


@lru_cache
def portfolio_repository() -> PortfolioRepository:
    return PortfolioRepository(get_settings().storage_dir, current_user_id())


@lru_cache
def today_advice_service() -> TodayAdviceService:
    return TodayAdviceService(
        get_settings(),
        technical_issuance_service(),
        portfolio_repository(),
        macro_service(),
    )


@lru_cache
def portfolio_settings_service() -> PortfolioSettingsService:
    return PortfolioSettingsService(portfolio_repository())


@lru_cache
def transaction_ledger_service() -> TransactionLedgerService:
    return TransactionLedgerService(portfolio_repository(), portfolio_settings_service())


@lru_cache
def position_accounting_service() -> PositionAccountingService:
    return PositionAccountingService(transaction_ledger_service(), portfolio_settings_service(), market_service())


@lru_cache
def cycle_service() -> TacticalCycleService:
    settings = get_settings()
    repository = JsonRepository(settings.storage_dir / "cycles" / "tactical_cycles.json", [])
    return TacticalCycleService(repository, market_service())


