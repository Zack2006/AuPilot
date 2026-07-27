"""Append-only transaction and derived snapshot contracts. / 追加式成交与派生快照契约。"""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.core.constants import PositionBucket, TransactionEventType, TransactionSide, TransactionSource


class PositionTransactionCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instrument: str = "GC.v.0"
    event_type: TransactionEventType = TransactionEventType.TRADE
    side: TransactionSide | None = None
    bucket: PositionBucket
    executed_at: datetime
    quantity: float = Field(gt=0)
    price: float = Field(gt=0)
    fee: float = Field(default=0, ge=0)
    currency: str = "USD"
    quantity_unit: str = "contract"
    contract_multiplier: float = Field(default=100.0, gt=0)
    source: TransactionSource = TransactionSource.MANUAL
    notes: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def validate_event_side(self) -> "PositionTransactionCreate":
        if self.event_type == TransactionEventType.TRADE and self.side is None:
            raise ValueError("TRADE records require BUY or SELL")
        if self.event_type == TransactionEventType.OPENING_BALANCE and self.side not in {None, TransactionSide.BUY}:
            raise ValueError("OPENING_BALANCE cannot be a sell")
        return self


class PositionTransaction(PositionTransactionCreate):
    transaction_id: str
    user_id: str
    created_at: datetime
    voided_at: datetime | None = None
    void_reason: str | None = None
    policy_violation: str | None = None


class VoidTransactionRequest(BaseModel):
    reason: str = Field(min_length=2, max_length=500)


class PortfolioSnapshot(BaseModel):
    as_of: datetime
    total_quantity: float
    core_quantity: float
    tactical_quantity: float
    average_cost: float
    market_price: float
    market_value: float
    realized_pnl: float
    unrealized_pnl: float
    total_fees: float
    total_return: float
    core_ratio: float
    tactical_ratio: float
    settings_id: str
    settings_version: int
    currency: str
    quantity_unit: str
    contract_multiplier: float
    cost_method: str = "WEIGHTED_AVERAGE"


class PerformancePoint(BaseModel):
    date: date
    market_price: float
    total_quantity: float
    average_cost: float
    realized_pnl: float
    unrealized_pnl: float
    cumulative_fees: float
    portfolio_value: float
