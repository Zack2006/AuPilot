"""Weighted-average position accounting. / 加权平均法持仓核算。

Purpose / 文件用途: derive snapshots and performance from immutable transactions.
Inputs / 输入: transaction quantities, trade prices, fees, contract multiplier and market price.
Outputs / 输出: current quantities, cost, realized/unrealized P&L and dated performance.
Business invariants / 业务约束: ``PnL = price difference × quantity × multiplier - fees``; no inferred orders.
Side effects / 副作用: none; all results are calculated on demand.
Fallback behavior / 降级: empty ledgers return a zero snapshot, never invented positions.
"""

from dataclasses import dataclass
from datetime import datetime, time, timezone

from backend.app.core.constants import PositionBucket
from backend.app.schemas.transaction import PerformancePoint, PortfolioSnapshot, PositionTransaction
from backend.app.services.market_data_service import MarketDataService
from backend.app.services.portfolio_settings_service import PortfolioSettingsService
from backend.app.services.transaction_ledger_service import TransactionLedgerService


@dataclass
class _BucketState:
    quantity: float = 0.0
    average_cost: float = 0.0


class PositionAccountingService:
    def __init__(self, ledger: TransactionLedgerService, settings: PortfolioSettingsService,
                 market: MarketDataService) -> None:
        self.ledger = ledger
        self.settings = settings
        self.market = market

    def snapshot(self, market_price: float | None = None, as_of: datetime | None = None) -> PortfolioSnapshot:
        """Calculate a snapshot with monetary values in settings currency. / 以设置币种计算当前快照。"""
        effective = self.settings.current()
        price = float(market_price if market_price is not None else self.market.latest().close)
        cutoff = as_of or datetime.now(timezone.utc)
        states, realized, fees = self._replay(self.ledger.list(), cutoff)
        core, tactical = states[PositionBucket.CORE], states[PositionBucket.TACTICAL]
        total = core.quantity + tactical.quantity
        average = ((core.average_cost * core.quantity + tactical.average_cost * tactical.quantity) / total) if total else 0
        market_value = total * price * effective.contract_multiplier
        unrealized = sum((price - state.average_cost) * state.quantity * effective.contract_multiplier for state in states.values())
        invested = average * total * effective.contract_multiplier
        total_return = (realized + unrealized - fees) / invested if invested else 0
        return PortfolioSnapshot(
            as_of=cutoff, total_quantity=round(total, 8), core_quantity=round(core.quantity, 8),
            tactical_quantity=round(tactical.quantity, 8), average_cost=round(average, 4), market_price=price,
            market_value=round(market_value, 2), realized_pnl=round(realized - fees, 2),
            unrealized_pnl=round(unrealized, 2), total_fees=round(fees, 2), total_return=round(total_return, 8),
            core_ratio=round(core.quantity / total, 8) if total else 0,
            tactical_ratio=round(tactical.quantity / total, 8) if total else 0,
            settings_id=effective.settings_id, settings_version=effective.version, currency=effective.currency,
            quantity_unit=effective.quantity_unit, contract_multiplier=effective.contract_multiplier,
        )

    def performance(self) -> list[PerformancePoint]:
        """Replay the ledger at every available market date. / 在每个可用行情日期重放账本。"""
        frame = self.market.get_history().frame
        points: list[PerformancePoint] = []
        for row in frame.itertuples(index=False):
            cutoff = datetime.combine(row.ts_event_utc.date(), time.max, tzinfo=timezone.utc)
            snapshot = self.snapshot(float(row.close), cutoff)
            points.append(PerformancePoint(
                date=row.ts_event_utc.date(), market_price=float(row.close), total_quantity=snapshot.total_quantity,
                average_cost=snapshot.average_cost, realized_pnl=snapshot.realized_pnl,
                unrealized_pnl=snapshot.unrealized_pnl, cumulative_fees=snapshot.total_fees,
                portfolio_value=snapshot.market_value,
            ))
        return points

    def transaction_impacts(self) -> list[dict]:
        """Expose traceable per-record cost and realized-P&L changes. / 返回每条记录对成本和已实现盈亏的影响。"""
        states = {PositionBucket.CORE: _BucketState(), PositionBucket.TACTICAL: _BucketState()}
        impacts = []
        for item in self.ledger.list():
            if item.voided_at:
                continue
            before = states[item.bucket].average_cost
            realized = self._apply(states[item.bucket], item)
            impacts.append({
                **item.model_dump(mode="json"), "average_cost_impact": round(states[item.bucket].average_cost - before, 4),
                "realized_pnl_impact": round(realized - item.fee, 2),
            })
        return impacts

    def _replay(self, transactions: list[PositionTransaction], cutoff: datetime) -> tuple[dict, float, float]:
        states = {PositionBucket.CORE: _BucketState(), PositionBucket.TACTICAL: _BucketState()}
        realized = fees = 0.0
        for item in transactions:
            executed = item.executed_at if item.executed_at.tzinfo else item.executed_at.replace(tzinfo=timezone.utc)
            if item.voided_at or executed > cutoff:
                continue
            realized += self._apply(states[item.bucket], item)
            fees += item.fee
        return states, realized, fees

    @staticmethod
    def _apply(state: _BucketState, item: PositionTransaction) -> float:
        """Apply one fact using weighted average; returns gross realized P&L. / 按加权平均法应用一条事实并返回毛已实现盈亏。"""
        is_buy = item.side is None or item.side.value == "BUY"
        if is_buy:
            new_quantity = state.quantity + item.quantity
            state.average_cost = ((state.average_cost * state.quantity) + item.price * item.quantity) / new_quantity
            state.quantity = new_quantity
            return 0.0
        realized = (item.price - state.average_cost) * item.quantity * item.contract_multiplier
        state.quantity = max(0.0, state.quantity - item.quantity)
        if state.quantity <= 1e-9:
            state.average_cost = 0.0
        return realized

