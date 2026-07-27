from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel


class AuditService:
    """Append immutable recommendation evidence to a local JSONL audit trail."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def append(
        self,
        portfolio: BaseModel,
        technical: BaseModel,
        recommendation: BaseModel,
        market_metadata: BaseModel,
    ) -> str:
        audit_id = f"audit_{uuid4().hex}"
        record = {
            "audit_id": audit_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "portfolio_input": portfolio.model_dump(mode="json"),
            "technical_prediction": technical.model_dump(mode="json"),
            "recommendation": recommendation.model_dump(mode="json"),
            "market_metadata": market_metadata.model_dump(mode="json"),
            "model_version": getattr(technical, "model_version", "unknown"),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return audit_id
