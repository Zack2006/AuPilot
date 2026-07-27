"""Product API for persisted AuPilot MN18 + PN02 technical issuances."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from fastapi import APIRouter, HTTPException, Query

from backend.app.api.dependencies import technical_issuance_service
from backend.app.core.exceptions import TechnicalIssuanceUnavailableError
from backend.app.schemas.technical import (
    TechnicalEvaluationRequest,
    TechnicalFillRequest,
)
from backend.app.services.technical_composition import (
    compose_product_forecast,
)


router = APIRouter(tags=["technical"])


def _latest():
    service = technical_issuance_service()
    record = service.repository.latest_success()
    if record is None:
        raise TechnicalIssuanceUnavailableError("TECHNICAL_ISSUANCE_NOT_READY")
    return service, record


def _staleness(service, record) -> tuple[bool, str | None, dict | None]:
    try:
        market = service.market_data.get_history(refresh=False)
    except Exception as exc:
        return True, f"MARKET_DATA_UNAVAILABLE:{type(exc).__name__}", None
    latest_bucket = market.frame["ts_event_utc"].iloc[-1].date()
    if latest_bucket != record.source_bucket:
        return True, "LATEST_MARKET_BUCKET_DIFFERS_FROM_ISSUANCE", market.metadata.model_dump(mode="json")
    return False, None, market.metadata.model_dump(mode="json")


@router.get("/technical/health")
def technical_health() -> dict:
    service = technical_issuance_service()
    return service.sidecar.health()


@router.get("/technical/status")
def technical_status() -> dict:
    service = technical_issuance_service()
    payload = dict(service.sidecar.status())
    state = service.repository.current_state()
    latest = service.repository.latest_success()
    latest_failure = service.repository.latest_failure()
    if latest is None:
        stale, reason = True, "TECHNICAL_ISSUANCE_NOT_READY"
        try:
            market = service.market_data.get_history(refresh=False)
            market_metadata = market.metadata.model_dump(mode="json")
        except Exception:
            market_metadata = None
    else:
        stale, reason, market_metadata = _staleness(service, latest)
    latest_audit = None
    audit_reader = getattr(service.repository, "composition_audit", None)
    if latest is not None and callable(audit_reader):
        latest_audit = audit_reader(latest.issuance_id)
    component_status = payload.get("components") or {}
    turning_component = component_status.get("turning") or {}
    price_component = component_status.get("conditional_ohlc") or {}
    payload.update(
        {
            "cache_status": "READY" if market_metadata is not None else "UNAVAILABLE",
            "cache": market_metadata,
            "latest_source_bucket": latest.source_bucket.isoformat() if latest else None,
            "latest_issuance_id": latest.issuance_id if latest else None,
            "latest_issuance_kind": latest.issuance_kind if latest else None,
            "latest_issuance_time_utc": latest.issued_at_utc.isoformat() if latest else None,
            "latest_output_sha256": latest.output_sha256 if latest else None,
            "latest_composition_audit_id": (
                latest_audit["audit_id"] if latest_audit else None
            ),
            "latest_composition_audit_sha256": (
                latest_audit["audit_sha256"] if latest_audit else None
            ),
            "latest_inference_duration_ms": (
                latest_audit["audit"].get("inference_duration_ms")
                if latest_audit
                else None
            ),
            "current_gold_weight": state.current_gold_weight,
            "outstanding_top_inventory_pp": (
                state.outstanding_top_inventory_pp
            ),
            "state_revision": state.revision,
            "state_updated_at_utc": state.updated_at_utc.isoformat(),
            "fill_count": len(service.repository.list_fills()),
            "scheduled_request_count": len(
                service.repository.list_scheduled_requests()
            ),
            "receiver_policy": "MN18_RECEIVER_POLICY_V1",
            "receiver_event_count": len(service.receiver.list_events()),
            "forward_shadow_state": service.receiver.shadow_state(),
            "forward_shadow_daily_action_count": len(
                service.receiver.list_shadow_actions()
            ),
            "minimum_forward_source_bucket": "2026-07-27",
            "stale": stale,
            "stale_reason": reason,
            "action_probability_model_loaded": bool(
                turning_component.get("loaded", True)
            ),
            "conditional_ohlc_model_loaded": bool(
                price_component.get("loaded", True)
            ),
            "latest_complete_bucket": (
                market_metadata.get("data_until_utc")
                if market_metadata
                else None
            ),
            "source_bucket": (
                latest.source_bucket.isoformat() if latest else None
            ),
            "forecast_21_slots_ready": bool(
                latest is not None
                and not stale
                and turning_component.get("loaded", True)
            ),
            "conditional_price_forecast_ready": bool(
                latest is not None
                and isinstance((latest.output or {}).get("price_outlook"), dict)
                and price_component.get("loaded", True)
            ),
            "development_candidate": True,
            "latest_failed_attempt": (
                {
                    "issuance_id": latest_failure.issuance_id,
                    "created_at_utc": latest_failure.created_at_utc.isoformat(),
                    "error_reason": latest_failure.error_reason,
                }
                if latest_failure
                else None
            ),
        }
    )
    return payload


@router.post("/internal/technical/evaluate-latest-daily")
def evaluate_latest_daily(request: TechnicalEvaluationRequest) -> dict:
    service = technical_issuance_service()
    if request.daily_history[0].trade_date != service.settings.technical_history_start:
        raise HTTPException(status_code=422, detail="TECHNICAL_FULL_HISTORY_START_MISMATCH")
    source_bucket_end = datetime.combine(
        request.daily_history[-1].trade_date + timedelta(days=1),
        datetime.min.time(),
        UTC,
    )
    if request.as_of_utc.astimezone(UTC) < source_bucket_end:
        raise HTTPException(status_code=422, detail="TECHNICAL_AS_OF_PRECEDES_SOURCE_AVAILABILITY")
    market = service.market_data.get_history(refresh=False)
    if len(request.daily_history) != len(market.frame):
        raise HTTPException(status_code=422, detail="TECHNICAL_FULL_HISTORY_ROW_COUNT_MISMATCH")
    for supplied, expected in zip(request.daily_history, market.frame.itertuples(index=False), strict=True):
        expected_values = (
            expected.ts_event_utc.date(),
            float(expected.open),
            float(expected.high),
            float(expected.low),
            float(expected.close),
        )
        supplied_values = (
            supplied.trade_date,
            supplied.open,
            supplied.high,
            supplied.low,
            supplied.close,
        )
        if supplied_values != expected_values:
            raise HTTPException(status_code=422, detail="TECHNICAL_HISTORY_DIFFERS_FROM_FORMAL_CACHE")
    _, raw = service.sidecar.evaluate(request)
    return raw


@router.get("/technical/advice/today")
def advice_today() -> dict:
    service, record = _latest()
    stale, stale_reason, _ = _staleness(service, record)
    output = record.output
    product = compose_product_forecast(output)
    first_slot = product["slots"][0]
    action = service.receiver.resolve_user_target(
        first_slot["target_bucket"]
    )
    return {
        "schema_version": product["schema_version"],
        "issuance_id": record.issuance_id,
        "issuance_kind": record.issuance_kind,
        "source_bucket": record.source_bucket.isoformat(),
        "source_bucket_end_utc": record.source_bucket_end_utc.isoformat(),
        "issued_at_utc": record.issued_at_utc.isoformat(),
        "stale": stale,
        "stale_reason": stale_reason,
        "action": action["action"],
        "authority": output["model_id"],
        "reason_code": action["reason_code"],
        "current_target_gold_weight": action["current_target_gold_weight"],
        "recommended_target_gold_weight": action["recommended_target_gold_weight"],
        "requested_delta_pp": action["requested_delta_pp"],
        "executed_delta_pp": action["executed_delta_pp"],
        "outstanding_top_inventory_pp_before": action[
            "outstanding_top_inventory_pp_before"
        ],
        "outstanding_top_inventory_pp_after": action[
            "outstanding_top_inventory_pp_after"
        ],
        "predicted_action_class": (
            first_slot["display_class"]
            if first_slot["display_class"].startswith("TOP_")
            else None
        ),
        "predicted_side": (
            "TOP"
            if first_slot["display_class"].startswith("TOP_")
            else None
        ),
        "model_probability": first_slot["p_top"],
        "top_lift": None,
        "bottom_lift": None,
        "probability_lift": None,
        "selected_lift_threshold": None,
        "expected_delta_pp": action["requested_delta_pp"],
        "expected_execution_bucket": first_slot["target_bucket"],
        "action_eligible_horizon_indices": [1],
        "fill_required_before_state_change": True,
        "models": product["models"],
        "output_sha256": record.output_sha256,
        "release_status": product["release_status"],
        "today_probabilities": first_slot["probabilities"],
        "today_display_signal": first_slot["display_signal"],
        "today_conditional_price_outlook": first_slot[
            "conditional_price_outlook"
        ],
        "automatic_execution": False,
        "broker_or_order_endpoint": False,
        "receiver_policy": action["receiver_policy"],
        "winning_request_id": action["winning_request_id"],
        "receiver_lifecycle_preview": action["lifecycle_preview"],
        "scheduled_action_requests": output["scheduled_action_requests"],
    }


@router.get("/technical/forecast/21-slots")
def forecast_21_slots() -> dict:
    service, record = _latest()
    stale, stale_reason, _ = _staleness(service, record)
    payload = compose_product_forecast(record.output)
    payload["stale"] = stale
    payload["stale_reason"] = stale_reason
    payload["issuance_id"] = record.issuance_id
    payload["issuance_kind"] = record.issuance_kind
    payload["output_sha256"] = record.output_sha256
    payload["persisted"] = True
    return payload


@router.get("/technical/issuances/latest")
def latest_issuance() -> dict:
    """Read the latest verified persisted issuance without running a model."""
    _, record = _latest()
    return {
        "schema_version": "aurumpilot.task4.technical_issuance_latest.v1",
        "issuance": record.model_dump(mode="json"),
        "product_forecast": compose_product_forecast(record.output),
        "persisted": True,
        "page_get_runs_models": False,
    }


@router.get("/technical/issuances/{issuance_id}/audit")
def issuance_composition_audit(issuance_id: str) -> dict:
    service = technical_issuance_service()
    audit = service.repository.composition_audit(issuance_id)
    if audit is None:
        raise HTTPException(
            status_code=404,
            detail="TECHNICAL_COMPOSITION_AUDIT_NOT_READY",
        )
    return {
        "schema_version": "aupilot.mn18_pn02.audit_response.v1",
        **audit,
        "immutable": True,
    }


@router.get("/technical/issuances")
def issuances(
    start: date | None = Query(default=None),
    end: date | None = Query(default=None),
    include_failures: bool = Query(default=False),
) -> dict:
    if start is not None and end is not None and start > end:
        raise HTTPException(status_code=422, detail="ISSUANCE_DATE_RANGE_INVALID")
    service = technical_issuance_service()
    records = service.repository.list_records(include_failures=include_failures)
    selected = [
        record
        for record in records
        if record.source_bucket is not None
        and (start is None or record.source_bucket >= start)
        and (end is None or record.source_bucket <= end)
    ]
    return {
        "schema_version": "aurumpilot.task4.technical_issuance_list.v1",
        "count": len(selected),
        "items": [record.model_dump(mode="json") for record in selected],
        "historical_backfill_used": False,
        "current_model_recalculation_used": False,
    }


@router.get("/technical/scheduled-requests")
def scheduled_requests() -> dict:
    service = technical_issuance_service()
    items = service.repository.list_scheduled_requests()
    return {
        "schema_version": "aupilot.mn18.scheduled_request_ledger.v1",
        "count": len(items),
        "items": items,
        "immutable_native_requests": True,
        "automatic_execution": False,
    }


@router.get("/technical/receiver-ledger")
def receiver_ledger() -> dict:
    service = technical_issuance_service()
    return {
        "schema_version": "aupilot.mn18.receiver_ledger.v1",
        "receiver_policy": "MN18_RECEIVER_POLICY_V1",
        "requests": service.repository.list_scheduled_requests(),
        "events": service.receiver.list_events(),
        "user_state": service.repository.current_state().model_dump(
            mode="json"
        ),
        "user_fifo_lots": service.receiver.fifo_lots("USER"),
        "forward_shadow_state": service.receiver.shadow_state(),
        "forward_shadow_fifo_lots": service.receiver.fifo_lots("SHADOW"),
        "forward_shadow_daily_actions": (
            service.receiver.list_shadow_actions()
        ),
        "user_and_shadow_state_isolated": True,
        "automatic_execution": False,
    }


@router.get("/technical/fills")
def fills() -> dict:
    service = technical_issuance_service()
    state = service.repository.current_state()
    items = service.repository.list_fills()
    return {
        "schema_version": "aupilot.mn18_pn02.technical_fill_list.v1",
        "count": len(items),
        "items": [item.model_dump(mode="json") for item in items],
        "state": state.model_dump(mode="json"),
        "broker_or_order_endpoint": False,
    }


@router.post("/technical/fills")
def record_fill(request: TechnicalFillRequest) -> dict:
    """Persist a user-confirmed historical fill; this does not execute a trade."""
    service = technical_issuance_service()
    if request.request_id is None:
        record, created = service.repository.record_fill(request)
    else:
        record, created = service.receiver.record_user_fill(request)
    return {
        "schema_version": "aupilot.mn18_pn02.technical_fill_result.v1",
        "fill": record.model_dump(mode="json"),
        "created": created,
        "state": service.repository.current_state().model_dump(mode="json"),
        "automatic_execution": False,
        "broker_or_order_endpoint": False,
    }
