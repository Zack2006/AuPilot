"""Deterministic MN18 H1/H2 receiver arbitration.

This module implements ``MN18_RECEIVER_POLICY_V1`` only.  It does not
calculate model probabilities, thresholds, expert weights, or price
outlooks.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any


RECEIVER_POLICY_ID = "MN18_RECEIVER_POLICY_V1"
MIN_GOLD_WEIGHT = 0.5
MAX_GOLD_WEIGHT = 1.0
EPSILON = 1.0e-12


def _latest(requests: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    items = list(requests)
    if not items:
        return None
    return max(
        items,
        key=lambda item: (
            str(item["source_bucket"]),
            str(item.get("created_at_utc") or ""),
            str(item["request_id"]),
        ),
    )


def _base_result(
    *,
    target_bucket: str,
    current_gold_weight: float,
    inventory_pp: float,
) -> dict[str, Any]:
    return {
        "receiver_policy": RECEIVER_POLICY_ID,
        "target_bucket": target_bucket,
        "action": "HOLD",
        "reason_code": "NO_QUALIFIED_REQUEST_FOR_TARGET_BUCKET",
        "winning_request_id": None,
        "source_bucket": None,
        "source_issuance_id": None,
        "source_record_issuance_id": None,
        "side": None,
        "horizon_index": None,
        "requested_delta_pp": 0.0,
        "executed_delta_pp": 0.0,
        "current_target_gold_weight": current_gold_weight,
        "recommended_target_gold_weight": current_gold_weight,
        "outstanding_top_inventory_pp_before": inventory_pp,
        "outstanding_top_inventory_pp_after": inventory_pp,
        "position_room_pp": (MAX_GOLD_WEIGHT - current_gold_weight) * 100.0,
        "fifo_consumptions": [],
        "fifo_lot_to_add_pp": 0.0,
        "automatic_execution": False,
        "requires_actual_databento_bucket_open": True,
    }


def resolve_target_bucket(
    *,
    target_bucket: str,
    requests: Iterable[dict[str, Any]],
    current_gold_weight: float,
    fifo_lots: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Resolve one target bucket without mutating persistence.

    Every input request is already a qualified native MN18 request.  The
    receiver compares recency only; it never compares cross-expert
    probabilities or threshold margins.
    """

    if not MIN_GOLD_WEIGHT - EPSILON <= current_gold_weight <= MAX_GOLD_WEIGHT + EPSILON:
        raise ValueError("MN18_RECEIVER_GOLD_WEIGHT_OUT_OF_BOUNDS")

    request_rows = [
        deepcopy(item)
        for item in requests
        if str(item["target_bucket"]) == target_bucket
    ]
    top_rows = [
        item
        for item in request_rows
        if item["side"] == "TOP" and int(item["horizon_index"]) == 1
    ]
    bottom_rows = [
        item
        for item in request_rows
        if item["side"] == "BOTTOM" and int(item["horizon_index"]) == 2
    ]
    latest_top = _latest(top_rows)
    latest_bottom = _latest(bottom_rows)
    events: list[dict[str, Any]] = []

    for rows, winner in ((top_rows, latest_top), (bottom_rows, latest_bottom)):
        if winner is None:
            continue
        for item in rows:
            if item["request_id"] != winner["request_id"]:
                events.append(
                    {
                        "request_id": item["request_id"],
                        "status": (
                            "CANCELLED_SUPERSEDED_BY_NEWER_SAME_SIDE_REQUEST"
                        ),
                        "superseded_by_request_id": winner["request_id"],
                    }
                )

    normalized_lots = []
    for lot in fifo_lots:
        remaining = float(lot["remaining_pp"])
        if remaining > EPSILON:
            normalized_lots.append(
                {
                    "lot_id": str(lot["lot_id"]),
                    "remaining_pp": remaining,
                    "created_at_utc": str(lot.get("created_at_utc") or ""),
                }
            )
    normalized_lots.sort(
        key=lambda item: (item["created_at_utc"], item["lot_id"])
    )
    inventory_pp = sum(item["remaining_pp"] for item in normalized_lots)
    result = _base_result(
        target_bucket=target_bucket,
        current_gold_weight=current_gold_weight,
        inventory_pp=inventory_pp,
    )

    if latest_top is not None:
        for item in bottom_rows:
            events.append(
                {
                    "request_id": item["request_id"],
                    "status": "CANCELLED_SUPERSEDED_BY_NEWER_H1_TOP",
                    "superseded_by_request_id": latest_top["request_id"],
                }
            )
        requested = float(latest_top["requested_delta_pp"])
        if requested >= -EPSILON:
            raise ValueError("MN18_H1_TOP_REQUEST_MUST_BE_NEGATIVE")
        executable_abs = min(
            abs(requested),
            max(0.0, (current_gold_weight - MIN_GOLD_WEIGHT) * 100.0),
        )
        executed = -executable_abs
        result.update(
            {
                "action": (
                    "REDUCE_GOLD_WEIGHT"
                    if executable_abs > EPSILON
                    else "HOLD"
                ),
                "reason_code": (
                    "QUALIFIED_MN18_H1_TOP"
                    if executable_abs > EPSILON
                    else "QUALIFIED_TOP_AT_MINIMUM_WEIGHT"
                ),
                "winning_request_id": latest_top["request_id"],
                "source_bucket": latest_top["source_bucket"],
                "source_issuance_id": (
                    "MN18:"
                    + str(
                        latest_top.get("model_version")
                        or "MN18-FULL-DEVELOPMENT-FORWARD-20260727T114605Z"
                    )
                    + ":SOURCE:"
                    + str(latest_top["source_bucket"])
                ),
                "source_record_issuance_id": latest_top["issuance_id"],
                "side": "TOP",
                "horizon_index": 1,
                "requested_delta_pp": requested,
                "executed_delta_pp": executed,
                "recommended_target_gold_weight": (
                    current_gold_weight + executed / 100.0
                ),
                "outstanding_top_inventory_pp_after": (
                    inventory_pp + executable_abs
                ),
                "fifo_lot_to_add_pp": executable_abs,
            }
        )
        events.append(
            {
                "request_id": latest_top["request_id"],
                "status": (
                    "AWAITING_USER_CONFIRMATION"
                    if executable_abs > EPSILON
                    else "REJECTED_POSITION_BOUND"
                ),
                "reason_code": result["reason_code"],
            }
        )
        return result, events

    if latest_bottom is not None:
        requested = float(latest_bottom["requested_delta_pp"])
        if requested <= EPSILON:
            raise ValueError("MN18_H2_BOTTOM_REQUEST_MUST_BE_POSITIVE")
        position_room_pp = max(
            0.0, (MAX_GOLD_WEIGHT - current_gold_weight) * 100.0
        )
        executable = min(requested, inventory_pp, position_room_pp)
        consumptions: list[dict[str, Any]] = []
        remaining = executable
        for lot in normalized_lots:
            if remaining <= EPSILON:
                break
            consumed = min(remaining, lot["remaining_pp"])
            consumptions.append(
                {"lot_id": lot["lot_id"], "consumed_pp": consumed}
            )
            remaining -= consumed
        if executable > EPSILON:
            reason = "QUALIFIED_MN18_H2_BOTTOM_FIFO_REENTER"
            action = "REENTER_GOLD_WEIGHT"
            event_status = "AWAITING_USER_CONFIRMATION"
        elif inventory_pp <= EPSILON:
            reason = "REJECTED_NO_FIFO_INVENTORY"
            action = "HOLD"
            event_status = "REJECTED_NO_FIFO_INVENTORY"
        else:
            reason = "REJECTED_POSITION_BOUND"
            action = "HOLD"
            event_status = "REJECTED_POSITION_BOUND"
        result.update(
            {
                "action": action,
                "reason_code": reason,
                "winning_request_id": latest_bottom["request_id"],
                "source_bucket": latest_bottom["source_bucket"],
                "source_issuance_id": (
                    "MN18:"
                    + str(
                        latest_bottom.get("model_version")
                        or "MN18-FULL-DEVELOPMENT-FORWARD-20260727T114605Z"
                    )
                    + ":SOURCE:"
                    + str(latest_bottom["source_bucket"])
                ),
                "source_record_issuance_id": latest_bottom["issuance_id"],
                "side": "BOTTOM",
                "horizon_index": 2,
                "requested_delta_pp": requested,
                "executed_delta_pp": executable,
                "recommended_target_gold_weight": (
                    current_gold_weight + executable / 100.0
                ),
                "outstanding_top_inventory_pp_after": (
                    inventory_pp - executable
                ),
                "position_room_pp": position_room_pp,
                "fifo_consumptions": consumptions,
            }
        )
        events.append(
            {
                "request_id": latest_bottom["request_id"],
                "status": event_status,
                "reason_code": reason,
            }
        )
    return result, events
