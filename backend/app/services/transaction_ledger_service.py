"""Append-only transaction ledger. / 追加式历史成交账本。

Purpose / 文件用途: record historical facts, void mistakes, and migrate one opening balance.
Business invariants / 业务约束: no order execution, no direct overwrite, and no sell beyond a bucket balance.
Side effects / 副作用: appends JSON; voiding only adds void metadata to the original fact.
Fallback behavior / 降级: invalid/corrupt files fail visibly.
"""

from datetime import datetime, timezone
from uuid import uuid4

from backend.app.core.constants import PositionBucket, TransactionEventType, TransactionSource
from backend.app.core.exceptions import NotFoundError
from backend.app.repositories.portfolio_repository import PortfolioRepository
from backend.app.schemas.transaction import PositionTransaction, PositionTransactionCreate
from backend.app.services.portfolio_settings_service import PortfolioSettingsService


class TransactionLedgerService:
    def __init__(self, repository: PortfolioRepository, settings: PortfolioSettingsService) -> None:
        self.repository = repository
        self.settings = settings

    def list(self, start: datetime | None = None, end: datetime | None = None, side: str | None = None,
             bucket: str | None = None, event_type: str | None = None) -> list[PositionTransaction]:
        self.ensure_migrated()
        items = [PositionTransaction.model_validate(item) for item in self.repository.read_transactions()]
        if start: items = [item for item in items if item.executed_at >= start]
        if end: items = [item for item in items if item.executed_at <= end]
        if side: items = [item for item in items if item.side and item.side.value == side]
        if bucket: items = [item for item in items if item.bucket.value == bucket]
        if event_type: items = [item for item in items if item.event_type.value == event_type]
        return sorted(items, key=lambda item: (item.executed_at, item.created_at))

    def create(self, payload: PositionTransactionCreate) -> PositionTransaction:
        """Record a completed trade fact; this function never contacts a broker. / 记录已发生事实，不连接券商。"""
        self.ensure_migrated()
        if payload.side and payload.side.value == "SELL":
            available = self._bucket_quantity(payload.bucket)
            if payload.quantity > available + 1e-9:
                raise ValueError(f"Sell quantity {payload.quantity} exceeds available {available}; short positions are forbidden")
        settings = self.settings.current()
        policy = None
        if payload.side and payload.side.value == "SELL" and payload.bucket == PositionBucket.CORE:
            policy = "Recorded historical fact: core-position sale violates the current core-protection policy."
        now = datetime.now(timezone.utc)
        record = PositionTransaction(
            **payload.model_dump(), transaction_id=f"txn_{uuid4().hex[:16]}", user_id=self.repository.user_id,
            created_at=now, policy_violation=policy,
        )
        if abs(record.contract_multiplier - settings.contract_multiplier) > 1e-9:
            raise ValueError("Transaction contract_multiplier must match the effective portfolio settings")
        self.repository.append_transaction(record.model_dump(mode="json"))
        return record

    def void(self, transaction_id: str, reason: str) -> PositionTransaction:
        for record in self.list():
            if record.transaction_id != transaction_id:
                continue
            if record.voided_at is not None:
                raise ValueError("Transaction is already voided")
            updated = record.model_copy(update={"voided_at": datetime.now(timezone.utc), "void_reason": reason})
            self.repository.replace_transaction(transaction_id, updated.model_dump(mode="json"))
            return updated
        raise NotFoundError(f"Transaction not found: {transaction_id}")

    def ensure_migrated(self) -> None:
        """Create exactly one opening-balance fact from legacy holdings. / 从旧持仓恰好创建一条期初余额。"""
        self.settings.ensure_migrated()
        if self.repository.read_transactions():
            return
        legacy = self.settings._legacy()
        if not legacy or float(legacy.get("total_position", 0)) <= 0:
            return
        now = datetime.now(timezone.utc)
        settings = self.settings.current()
        opening = PositionTransaction(
            transaction_id="txn_demo_opening_balance", user_id=self.repository.user_id,
            instrument="GC.v.0", event_type=TransactionEventType.OPENING_BALANCE, side=None,
            bucket=PositionBucket.CORE, executed_at=now, quantity=float(legacy["total_position"]),
            price=float(legacy["average_cost"]), fee=0, currency=settings.currency,
            quantity_unit=settings.quantity_unit, contract_multiplier=settings.contract_multiplier,
            source=TransactionSource.MIGRATION,
            notes="Migration opening balance only; no historical purchase order was fabricated.", created_at=now,
        )
        self.repository.append_transaction(opening.model_dump(mode="json"))

    def _bucket_quantity(self, bucket: PositionBucket) -> float:
        quantity = 0.0
        for item in self.list():
            if item.voided_at or item.bucket != bucket:
                continue
            quantity += item.quantity if item.side is None or item.side.value == "BUY" else -item.quantity
        return quantity

