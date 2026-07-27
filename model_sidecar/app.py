"""Private ASGI runtime for the hash-gated MN18 + PN02 dual model."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import lightgbm
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aupilot.deployment.mn18_pn02_dual_model import (
    load_verified_mn18_pn02_dual_model,
)


SCHEMA_VERSION = "aupilot.mn18_pn02.dual_outlook.v1"
REQUIRED_HISTORY_START = date(2010, 6, 7)


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _verify_hash(path: Path, expected: str, reason: str) -> str:
    actual = _sha256(path)
    if actual != expected.upper():
        raise RuntimeError(f"{reason}:{actual}")
    return actual


def _verify_package(root: Path, expected_manifest_sha256: str) -> dict:
    manifest_path = root / "PACKAGE_MANIFEST.json"
    manifest_sha = _verify_hash(
        manifest_path,
        expected_manifest_sha256,
        "DUAL_PACKAGE_MANIFEST_SHA256_MISMATCH",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for relative, record in manifest["files"].items():
        path = root / relative
        if not path.is_file():
            errors.append(f"MISSING:{relative}")
        elif path.stat().st_size != int(record["bytes"]):
            errors.append(f"BYTES:{relative}")
        elif _sha256(path) != str(record["sha256"]).upper():
            errors.append(f"SHA256:{relative}")
    if errors:
        raise RuntimeError(
            "DUAL_PACKAGE_VERIFICATION_FAILED:" + ",".join(errors[:10])
        )
    return {
        "files_verified": len(manifest["files"]),
        "manifest_sha256": manifest_sha,
        "error_count": 0,
    }


class DailyOHLC(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trade_date: date
    open: float = Field(gt=0, allow_inf_nan=False)
    high: float = Field(gt=0, allow_inf_nan=False)
    low: float = Field(gt=0, allow_inf_nan=False)
    close: float = Field(gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_candle(self) -> "DailyOHLC":
        if self.high < max(self.open, self.close):
            raise ValueError("high is below open/close")
        if self.low > min(self.open, self.close):
            raise ValueError("low is above open/close")
        return self


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    daily_history: list[DailyOHLC] = Field(min_length=60)
    current_gold_weight: float = Field(ge=0.5, le=1.0)
    outstanding_top_inventory_pp: float = Field(ge=0, le=50)
    as_of_utc: datetime
    display_timezone: str | None = None

    @model_validator(mode="after")
    def validate_request(self) -> "EvaluationRequest":
        if self.as_of_utc.tzinfo is None or self.as_of_utc.utcoffset() is None:
            raise ValueError("as_of_utc must be timezone-aware")
        dates = [row.trade_date for row in self.daily_history]
        if dates != sorted(set(dates)):
            raise ValueError("daily_history must be unique and increasing")
        if dates[0] != REQUIRED_HISTORY_START:
            raise ValueError("daily_history start differs from 2010-06-07")
        latest_bucket_end = datetime.combine(
            dates[-1] + timedelta(days=1),
            time.min,
            tzinfo=UTC,
        )
        if self.as_of_utc.astimezone(UTC) < latest_bucket_end:
            raise ValueError(
                "daily_history contains an incomplete UTC daily bucket"
            )
        return self


root = Path(_required_environment("AUPILOT_DUAL_MODEL_ROOT")).resolve()
turning_manifest = Path(
    _required_environment("AUPILOT_MN18_BUNDLE_MANIFEST")
).resolve()
price_bundle = Path(
    _required_environment("AUPILOT_PN02_BUNDLE")
).resolve()

package_verification = _verify_package(
    root,
    _required_environment("AUPILOT_DUAL_PACKAGE_MANIFEST_SHA256"),
)
turning_manifest_sha256 = _verify_hash(
    turning_manifest,
    _required_environment("AUPILOT_MN18_BUNDLE_MANIFEST_SHA256"),
    "MN18_BUNDLE_MANIFEST_SHA256_MISMATCH",
)
price_bundle_sha256 = _verify_hash(
    price_bundle,
    _required_environment("AUPILOT_PN02_BUNDLE_SHA256"),
    "PN02_BUNDLE_SHA256_MISMATCH",
)

runtime = load_verified_mn18_pn02_dual_model(
    turning_manifest_path=turning_manifest,
    price_bundle_path=price_bundle,
    root=root,
)
if (
    runtime.turning_bundle.model_version
    != _required_environment("AUPILOT_MN18_MODEL_VERSION")
):
    raise RuntimeError("MN18_MODEL_VERSION_MISMATCH")
if (
    runtime.price_bundle.model_version
    != _required_environment("AUPILOT_PN02_MODEL_VERSION")
):
    raise RuntimeError("PN02_MODEL_VERSION_MISMATCH")


app = FastAPI(title="AuPilot MN18 + PN02 Model Sidecar")


def _component_status() -> dict[str, Any]:
    return {
        "turning": {
            "loaded": True,
            "model_id": (
                "MN18_THREE_TOP_EXPERT_FORWARD_SHADOW_CANDIDATE_V1"
            ),
            "model_version": runtime.turning_bundle.model_version,
            "bundle_manifest_sha256": turning_manifest_sha256,
            "joblib_sha256": (
                "C497E8251BF7A740023CE64C0CFC8CA10CEC20C40AC2BADB9A49AF4D380716A6"
            ),
            "controls_trading": True,
        },
        "conditional_ohlc": {
            "loaded": True,
            "model_id": runtime.price_bundle.model_id,
            "model_version": runtime.price_bundle.model_version,
            "bundle_sha256": price_bundle_sha256,
            "controls_trading": False,
            "advisory_only": True,
        },
    }


@app.get("/api/v1/technical/health")
def health() -> dict[str, Any]:
    return {
        "service": "AuPilot",
        "status": "READY_DEVELOPMENT_REQUIRES_FORWARD_SHADOW",
        "schema_version": SCHEMA_VERSION,
        "bundle_verified": True,
        "package_verification": package_verification,
        "components": _component_status(),
        "exactly_21_slots": True,
        "seven_class_probabilities": True,
        "mn18_is_sole_action_authority": True,
        "pn02_controls_trading": False,
        "rag_controls_trading": False,
        "automatic_execution": False,
        "broker_or_order_endpoint": False,
        "python_version": sys.version.split()[0],
        "lightgbm_version": lightgbm.__version__,
    }


@app.get("/api/v1/technical/status")
def status() -> dict[str, Any]:
    return {
        **health(),
        "contract_id": "MN18_PN02_DUAL_MODEL_UTC_DAILY_V1",
        "history_start": REQUIRED_HISTORY_START.isoformat(),
        "history_ohlc_only": True,
        "technical_features_internal": True,
        "calendar_semantics": (
            "FUTURE_VALID_CANONICAL_UTC_DAILY_BUCKETS"
        ),
        "not_comex_sessions": True,
        "h1_permission": "TOP_ONLY",
        "h2_permission": "BOTTOM_ONLY",
        "h3_to_h21_permission": "DISPLAY_ONLY",
        "join_keys": ["horizon_index", "target_bucket"],
    }


@app.post("/api/v1/internal/technical/evaluate-latest-daily")
def evaluate(request: EvaluationRequest) -> dict[str, Any]:
    frame = pd.DataFrame(
        [row.model_dump(mode="python") for row in request.daily_history]
    )
    try:
        result = runtime.predict_from_history(
            frame,
            current_gold_weight=request.current_gold_weight,
            outstanding_top_inventory_pp=(
                request.outstanding_top_inventory_pp
            ),
            as_of_utc=request.as_of_utc.astimezone(UTC),
        )
    except (AssertionError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"DUAL_MODEL_INPUT_OR_CONTRACT_ERROR:{type(exc).__name__}:{exc}",
        ) from exc
    return result
