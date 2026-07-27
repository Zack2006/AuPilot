"""Idempotent scheduler for new complete MN18 + PN02 UTC source buckets."""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.api.dependencies import technical_issuance_service  # noqa: E402


STATUS_PATH = ROOT / "storage" / "technical" / "worker_status.json"
INTERVAL_SECONDS = max(
    300, int(os.getenv("TECHNICAL_REFRESH_INTERVAL_SECONDS", "21600"))
)


def _write_status(payload: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(STATUS_PATH)


def _run_once() -> dict:
    started = datetime.now(UTC)
    results = technical_issuance_service().evaluate_pending(
        refresh_market=True
    )
    return {
        "state": "running",
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "last_run_started_at_utc": started.isoformat(),
        "last_run_succeeded": True,
        "new_issuance_count": sum(result.created for result in results),
        "issuance_ids": [
            result.issuance.issuance_id for result in results
        ],
        "minimum_forward_source_bucket": "2026-07-27",
        "automatic_execution": False,
        "error": None,
    }


def main() -> None:
    while True:
        try:
            status = _run_once()
        except Exception as exc:  # fail closed; retain real reason code
            status = {
                "state": "running",
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "last_run_succeeded": False,
                "new_issuance_count": 0,
                "issuance_ids": [],
                "minimum_forward_source_bucket": "2026-07-27",
                "automatic_execution": False,
                "error": f"{type(exc).__name__}:{str(exc)[:500]}",
            }
        _write_status(status)
        deadline = time.monotonic() + INTERVAL_SECONDS
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(60.0, remaining))
            status = {
                **status,
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "heartbeat_only": True,
            }
            _write_status(status)


if __name__ == "__main__":
    main()
