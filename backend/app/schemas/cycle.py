from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, model_validator

from backend.app.core.constants import CycleStatus


class TacticalCycleCreate(BaseModel):
    reduction_ratio: float = Field(gt=0, le=1)
    rebuy_price_range: tuple[float, float]
    invalidation_price: float = Field(gt=0)
    notes: str = ""

    @model_validator(mode="after")
    def validate_range(self) -> "TacticalCycleCreate":
        if self.rebuy_price_range[0] <= 0 or self.rebuy_price_range[0] > self.rebuy_price_range[1]:
            raise ValueError("rebuy_price_range must contain a valid ascending positive range")
        return self


class TacticalCycleUpdate(BaseModel):
    rebought_ratio: float | None = Field(default=None, ge=0, le=1)
    rebuy_price: float | None = Field(default=None, gt=0)
    complete: bool = False
    invalidate: bool = False
    notes: str | None = None


class TacticalCycle(BaseModel):
    cycle_id: str
    status: CycleStatus
    created_at: datetime
    updated_at: datetime
    reduction_price: float
    reduction_ratio: float
    rebuy_price_range: tuple[float, float]
    rebought_ratio: float = 0
    average_rebuy_price: float | None = None
    realized_cost_reduction: float = 0
    missed_upside_cost: float = 0
    invalidation_price: float
    price_source_id: str | None = None
    price_as_of_utc: datetime | None = None
    market_data_sha256: str | None = None
    notes: str = ""
