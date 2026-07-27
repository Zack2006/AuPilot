"""Explicitly fetch and atomically publish formal Databento daily gold bars."""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.api.dependencies import market_service


def main() -> None:
    result = market_service().get_history(refresh=True)
    metadata = result.metadata
    print(
        f"rows={len(result.frame)} source={metadata.source_id} dataset={metadata.dataset} "
        f"symbol={metadata.symbol} schema={metadata.schema_name} "
        f"as_of={metadata.data_until_utc.isoformat()} sha256={metadata.content_sha256}"
    )


if __name__ == "__main__":
    main()
