"""Historical transaction ledger routes. / 历史成交账本路由。"""

from datetime import datetime

from fastapi import APIRouter, Query

from backend.app.api.dependencies import position_accounting_service, transaction_ledger_service
from backend.app.schemas.transaction import PositionTransaction, PositionTransactionCreate, VoidTransactionRequest

router = APIRouter(prefix="/portfolio/transactions", tags=["transactions"])


@router.get("", response_model=list[PositionTransaction])
def list_transactions(start: datetime | None = None, end: datetime | None = None,
                      side: str | None = Query(None), bucket: str | None = Query(None),
                      event_type: str | None = Query(None)) -> list[PositionTransaction]:
    return transaction_ledger_service().list(start, end, side, bucket, event_type)


@router.get("/impacts")
def transaction_impacts() -> list[dict]:
    return position_accounting_service().transaction_impacts()


@router.post("", response_model=PositionTransaction, status_code=201)
def create_transaction(payload: PositionTransactionCreate) -> PositionTransaction:
    return transaction_ledger_service().create(payload)


@router.post("/{transaction_id}/void", response_model=PositionTransaction)
def void_transaction(transaction_id: str, payload: VoidTransactionRequest) -> PositionTransaction:
    return transaction_ledger_service().void(transaction_id, payload.reason)
