"""Reduction-to-rebuy tactical cycle tracking. / 减仓—回补战术周期追踪。

Purpose / 文件用途: preserve the original MVP cycle feature beneath tactical analytics.
Inputs / 输入: recorded reduction/rebuy ratios and quote prices; ratios use decimals.
Outputs / 输出: current/history cycle states, realized cost reduction and missed-upside cost.
Business invariants / 业务约束: one active cycle; closed cycles cannot be mutated; no order execution.
Side effects / 副作用: writes cycle JSON through the repository.
Fallback behavior / 降级: missing records raise explicit NotFoundError.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from backend.app.core.constants import CycleStatus
from backend.app.core.exceptions import NotFoundError
from backend.app.repositories.json_repository import JsonRepository
from backend.app.schemas.cycle import TacticalCycle, TacticalCycleCreate, TacticalCycleUpdate
from backend.app.services.market_data_service import MarketDataService


class TacticalCycleService:
    """Manage file-backed reduce-to-rebuy tactical cycles."""

    def __init__(self, repository: JsonRepository, market: MarketDataService) -> None:
        self.repository = repository
        self.market = market

    def list_cycles(self) -> list[TacticalCycle]:
        return [TacticalCycle.model_validate(item) for item in self.repository.read()]

    def current(self) -> TacticalCycle | None:
        for cycle in reversed(self.list_cycles()):
            if cycle.status in {CycleStatus.OPEN, CycleStatus.PARTIALLY_REBOUGHT}:
                return cycle
        return None

    def create(self, payload: TacticalCycleCreate) -> TacticalCycle:
        if self.current() is not None:
            raise ValueError("An active tactical cycle already exists")
        now = datetime.now(timezone.utc)
        market_result = self.market.get_history(refresh=False)
        observed_row = market_result.frame.iloc[-1]
        cycle = TacticalCycle(
            cycle_id=f"cycle_{uuid4().hex[:12]}", status=CycleStatus.OPEN,
            created_at=now, updated_at=now, reduction_price=float(observed_row["close"]),
            price_source_id=market_result.metadata.source_id,
            price_as_of_utc=observed_row["ts_event_utc"].to_pydatetime(),
            market_data_sha256=market_result.metadata.content_sha256,
            **payload.model_dump(),
        )
        records = self.repository.read()
        records.append(cycle.model_dump(mode="json"))
        self.repository.write(records)
        return cycle

    def update(self, cycle_id: str, payload: TacticalCycleUpdate) -> TacticalCycle:
        records = self.repository.read()
        for index, item in enumerate(records):
            if item["cycle_id"] != cycle_id:
                continue
            cycle = TacticalCycle.model_validate(item)
            if cycle.status in {CycleStatus.COMPLETED, CycleStatus.INVALIDATED}:
                raise ValueError("A closed tactical cycle cannot be updated")
            update = payload.model_dump(exclude_none=True)
            added_ratio = float(update.get("rebought_ratio", 0))
            total_rebought = min(cycle.reduction_ratio, cycle.rebought_ratio + added_ratio)
            average_rebuy = cycle.average_rebuy_price
            if added_ratio and payload.rebuy_price:
                previous_value = (cycle.average_rebuy_price or 0) * cycle.rebought_ratio
                average_rebuy = (previous_value + payload.rebuy_price * added_ratio) / max(total_rebought, 1e-9)
            status = CycleStatus.PARTIALLY_REBOUGHT if total_rebought > 0 else CycleStatus.OPEN
            if payload.complete or total_rebought >= cycle.reduction_ratio - 1e-9:
                status = CycleStatus.COMPLETED
            current_price = float(self.market.latest(refresh=False).close)
            if payload.invalidate or current_price >= cycle.invalidation_price:
                status = CycleStatus.INVALIDATED
            realized = max(cycle.reduction_price - (average_rebuy or cycle.reduction_price), 0) * total_rebought
            missed = max(current_price - cycle.reduction_price, 0) * max(cycle.reduction_ratio - total_rebought, 0)
            cycle = cycle.model_copy(update={
                "status": status, "updated_at": datetime.now(timezone.utc), "rebought_ratio": total_rebought,
                "average_rebuy_price": round(average_rebuy, 2) if average_rebuy else None,
                "realized_cost_reduction": round(realized, 4), "missed_upside_cost": round(missed, 4),
                "notes": payload.notes if payload.notes is not None else cycle.notes,
            })
            records[index] = cycle.model_dump(mode="json")
            self.repository.write(records)
            return cycle
        raise NotFoundError(f"Tactical cycle not found: {cycle_id}")

    def is_invalidated(self, cycle: TacticalCycle, current_price: float) -> bool:
        return current_price >= cycle.invalidation_price

