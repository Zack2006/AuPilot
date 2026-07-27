"""Run one idempotent MN18 + PN02 issuance from the latest formal cache."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.api.dependencies import technical_issuance_service  # noqa: E402


def main() -> None:
    result = technical_issuance_service().evaluate_latest()
    issuance = result.issuance
    print(
        json.dumps(
            {
                "created": result.created,
                "issuance_id": issuance.issuance_id,
                "source_bucket": issuance.source_bucket.isoformat(),
                "input_rows": issuance.input_rows,
                "action": issuance.output["action"]["action"],
                "turning_model_id": issuance.output["model_id"],
                "price_model_id": issuance.output["price_outlook"]["model_id"],
                "slot_count": len(issuance.output["probability_rows"]),
                "output_sha256": issuance.output_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
