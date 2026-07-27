"""Product projection for the official MN18 + PN02 joint-model response."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, time, timedelta
from typing import Any

from backend.app.schemas.technical import (
    EXPECTED_COMPONENT_HASHES,
    TECHNICAL_SCHEMA_VERSION,
)


PROBABILITY_FIELDS = {
    "NORMAL": "p_normal",
    "TOP_L1": "p_top_l1",
    "TOP_L2": "p_top_l2",
    "TOP_L3": "p_top_l3",
    "BOTTOM_L1": "p_bottom_l1",
    "BOTTOM_L2": "p_bottom_l2",
    "BOTTOM_L3": "p_bottom_l3",
}


def _display_signal(
    row: dict[str, Any],
    action: dict[str, Any],
) -> dict[str, Any]:
    label = str(row.get("display_class") or "NORMAL")
    if label.startswith("TOP_"):
        side = "TOP"
        strength: str | None = label.removeprefix("TOP_")
    elif label.startswith("BOTTOM_"):
        side = "BOTTOM"
        strength = label.removeprefix("BOTTOM_")
    else:
        side = "NORMAL"
        strength = None
    actionable = bool(
        row.get("horizon_index") == 1
        and side == "TOP"
        and action.get("action") == "REDUCE_GOLD_WEIGHT"
    )
    return {
        "side": side,
        "strength": strength,
        "actionable": actionable,
    }


def _selected_price_outlook(
    price_slot: dict[str, Any],
    signal: dict[str, Any],
) -> dict[str, Any] | None:
    side = signal["side"]
    if side == "NORMAL":
        return None
    scenario = price_slot.get(
        "top_conditional" if side == "TOP" else "bottom_conditional"
    )
    if not isinstance(scenario, dict):
        return None
    expected = f"{side}_ACTION_ZONE"
    if scenario.get("scenario_label") != expected:
        return None
    return {
        "scenario": expected,
        "point": deepcopy(scenario.get("point")),
        "marginal_80pct_intervals": deepcopy(
            scenario.get("marginal_80pct_intervals")
        ),
        "advisory_only": True,
        "controls_trading": False,
    }


def compose_product_forecast(output: dict[str, Any]) -> dict[str, Any]:
    """Add the existing product view without mutating the raw audit payload."""

    payload = deepcopy(output)
    rows = payload.get("probability_rows") or []
    price = payload.get("price_outlook") or {}
    price_slots = price.get("slots") or []
    if len(rows) != 21 or len(price_slots) != 21:
        raise ValueError("MN18_PN02_EXPECTED_EXACTLY_21_ALIGNED_SLOTS")
    action = payload.get("action") or {}
    product_slots: list[dict[str, Any]] = []
    for row, price_slot in zip(rows, price_slots, strict=True):
        if (
            row.get("horizon_index") != price_slot.get("horizon_index")
            or row.get("target_bucket") != price_slot.get("target_bucket")
        ):
            raise ValueError("MN18_PN02_SLOT_ALIGNMENT_FAILED")
        probabilities = {
            label: float(row[field])
            for label, field in PROBABILITY_FIELDS.items()
        }
        signal = _display_signal(row, action)
        target = datetime.fromisoformat(str(row["target_bucket"])).date()
        bucket_start = datetime.combine(target, time.min, UTC)
        product_slots.append(
            {
                **deepcopy(row),
                "bucket_start_utc": bucket_start.isoformat(),
                "bucket_end_utc": (bucket_start + timedelta(days=1)).isoformat(),
                "display_timezone": "UTC",
                "display_start": bucket_start.isoformat(),
                "display_end": (bucket_start + timedelta(days=1)).isoformat(),
                "daily_unit_id": "CANONICAL_GC_UTC_DAILY_BUCKET_V1",
                "is_comex_session_bar": False,
                "boundary_action_eligible": bool(row.get("controls_trading")),
                "action_eligible": row.get("horizon_index") == 1,
                "observation_only": row.get("horizon_index") >= 3,
                "probabilities": probabilities,
                "aggregate_probabilities": {
                    "NORMAL": probabilities["NORMAL"],
                    "TOP": float(row["p_top"]),
                    "BOTTOM": float(row["p_bottom"]),
                },
                "display_signal": signal,
                "conditional_top_ohlc": deepcopy(
                    price_slot.get("top_conditional")
                ),
                "conditional_bottom_ohlc": deepcopy(
                    price_slot.get("bottom_conditional")
                ),
                "conditional_price_outlook": _selected_price_outlook(
                    price_slot, signal
                ),
                "price_scenarios_control_trading": False,
            }
        )
    payload.update(
        {
            "schema_version": TECHNICAL_SCHEMA_VERSION,
            "source_bucket": payload["history"]["source_bucket"],
            "source_bucket_end_utc": (
                datetime.combine(
                    datetime.fromisoformat(
                        payload["history"]["source_bucket"]
                    ).date()
                    + timedelta(days=1),
                    time.min,
                    UTC,
                ).isoformat()
            ),
            "issued_at_utc": payload["as_of_utc"],
            "slot_count": 21,
            "slots": product_slots,
            "models": {
                "action_probability_model_id": payload["model_id"],
                "action_probability_model_version": payload["model_version"],
                "conditional_ohlc_model_id": price["model_id"],
                "conditional_ohlc_model_version": price["model_version"],
                "conditional_ohlc_model_status": price["model_status"],
                "conditional_ohlc_advisory_only": True,
                "conditional_ohlc_controls_trading": False,
                **deepcopy(EXPECTED_COMPONENT_HASHES),
            },
            "release_status": "PENDING_NEW_FORWARD_SHADOW_EVIDENCE",
            "development_candidate": True,
            "conditional_price_component_status": "READY",
            "conditional_price_error": None,
            "automatic_execution": False,
            "broker_or_order_endpoint": False,
            "composition": {
                "source_target_horizon_alignment": "VERIFIED",
                "turning_model_status": "READY",
                "conditional_price_model_status": "READY",
                "normal_slots_hide_conditional_prices": True,
                "pn02_controls_trading": False,
                "macro_input_used": False,
            },
        }
    )
    return payload


def build_composition_audit(
    *,
    issuance_id: str,
    output: dict[str, Any],
    output_sha256: str,
    market_data_sha256: str | None,
    input_request_sha256: str | None,
    data_quality_status: str | None,
    inference_duration_ms: float | None,
    created_at_utc: datetime,
) -> dict[str, Any]:
    """Create an append-only audit projection for one immutable issuance."""

    product = compose_product_forecast(output)
    return {
        "schema_version": "aupilot.mn18_pn02.composition_audit.v1",
        "issuance_id": issuance_id,
        "created_at_utc": created_at_utc.isoformat(),
        "source_bucket": output["history"]["source_bucket"],
        "as_of_utc": output["as_of_utc"],
        "data_quality_status": data_quality_status,
        "inference_duration_ms": inference_duration_ms,
        "market_data_sha256": market_data_sha256,
        "input_request_sha256": input_request_sha256,
        "raw_output_sha256": output_sha256,
        "component_hashes": deepcopy(EXPECTED_COMPONENT_HASHES),
        "mn18": {
            "model_id": output["model_id"],
            "model_version": output["model_version"],
            "action": deepcopy(output["action"]),
            "probability_rows": deepcopy(output["probability_rows"]),
            "scheduled_action_requests": deepcopy(
                output["scheduled_action_requests"]
            ),
        },
        "pn02": {
            "model_id": output["price_outlook"]["model_id"],
            "model_version": output["price_outlook"]["model_version"],
            "controls_trading": False,
            "raw_price_outlook": deepcopy(output["price_outlook"]),
        },
        "source_target_horizon_alignment": "VERIFIED",
        "normal_slots_hide_conditional_prices": True,
        "product_slots": deepcopy(product["slots"]),
    }
