"""Deterministic page composition for the persisted Today Advice product."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aupilot.core.hashing import canonical_json_sha256

from backend.app.core.config import Settings
from backend.app.core.exceptions import TechnicalIssuanceUnavailableError
from backend.app.repositories.portfolio_repository import PortfolioRepository
from backend.app.schemas.advice import (
    PortfolioAllocationSnapshot,
    PortfolioAllocationSnapshotCreate,
)
from backend.app.services.technical_composition import compose_product_forecast


ACTION_NAMES = {
    "HOLD": "HOLD",
    "REDUCE_GOLD_WEIGHT": "REDUCE",
    "INCREASE_GOLD_WEIGHT": "ADD",
}


def calculate_personalized_target(
    current_gold_weight_pct: float | None,
    requested_delta_pp: float,
    strategy_min_weight_pct: float,
    strategy_max_weight_pct: float,
) -> dict[str, float | bool | None | str]:
    """Apply a signed percentage-point delta without changing model output."""

    if not (
        0
        <= strategy_min_weight_pct
        <= strategy_max_weight_pct
        <= 100
    ):
        raise ValueError("TECHNICAL_STRATEGY_WEIGHT_BOUNDS_INVALID")
    if current_gold_weight_pct is None:
        return {
            "effective_delta_pp": None,
            "target_gold_weight_pct": None,
            "was_clamped": None,
            "status": "USER_ALLOCATION_REQUIRED",
        }
    if not 0 <= current_gold_weight_pct <= 100:
        raise ValueError("USER_GOLD_WEIGHT_INVALID")
    raw_target = current_gold_weight_pct + requested_delta_pp
    target = min(
        max(raw_target, strategy_min_weight_pct),
        strategy_max_weight_pct,
    )
    return {
        "effective_delta_pp": target - current_gold_weight_pct,
        "target_gold_weight_pct": target,
        "was_clamped": abs(target - raw_target) > 1e-9,
        "status": "READY",
    }


def _iso_utc(value: Any) -> str:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("TECHNICAL_BUCKET_TIMEZONE_MISSING")
    return parsed.astimezone(UTC).isoformat()


def _price_dto(outlook: dict[str, Any] | None, side: str) -> dict | None:
    if not isinstance(outlook, dict):
        return None
    point = outlook.get("point")
    intervals = outlook.get("marginal_80pct_intervals")
    if not isinstance(point, dict):
        return None
    fields: dict[str, dict[str, float | None]] = {}
    for field in ("open", "high", "low", "close"):
        interval = intervals.get(field) if isinstance(intervals, dict) else None
        fields[field] = {
            "point": float(point[field]),
            "p10": (
                float(interval["lower"])
                if isinstance(interval, dict) and interval.get("lower") is not None
                else None
            ),
            "p90": (
                float(interval["upper"])
                if isinstance(interval, dict) and interval.get("upper") is not None
                else None
            ),
        }
    return {
        "side": side,
        "scenario": outlook["scenario"],
        "model_status": "DEVELOPMENT_CANDIDATE",
        **fields,
        "advisory_only": True,
        "controls_trading": False,
        "interval_semantics": "MARGINAL_80PCT_PER_FIELD",
    }


class TodayAdviceService:
    """Compose persisted technical, user-allocation and macro snapshots."""

    def __init__(
        self,
        settings: Settings,
        technical,
        portfolio: PortfolioRepository,
        macro,
    ) -> None:
        self.settings = settings
        self.technical = technical
        self.portfolio = portfolio
        self.macro = macro

    def _latest_allocation(self) -> dict | None:
        records = self.portfolio.read_allocation_snapshots()
        return records[-1] if records else None

    def create_allocation_snapshot(
        self,
        payload: PortfolioAllocationSnapshotCreate,
    ) -> dict:
        now = datetime.now(UTC)
        previous = self._latest_allocation()
        identity = {
            "user_id": self.portfolio.user_id,
            "current_gold_weight_pct": float(payload.gold_weight_pct),
            "as_of_utc": payload.as_of_utc.isoformat(),
            "source": payload.source,
            "supersedes_snapshot_id": (
                previous.get("snapshot_id") if previous else None
            ),
        }
        snapshot = PortfolioAllocationSnapshot(
            snapshot_id=(
                "portfolio_" + canonical_json_sha256(identity)[:24].lower()
            ),
            user_id=self.portfolio.user_id,
            current_gold_weight_pct=float(payload.gold_weight_pct),
            as_of_utc=payload.as_of_utc,
            created_at_utc=now,
            source=payload.source,
            supersedes_snapshot_id=(
                previous.get("snapshot_id") if previous else None
            ),
        ).model_dump(mode="json")
        created = self.portfolio.append_allocation_snapshot(snapshot)
        return {
            "snapshot": snapshot,
            "created": created,
            "automatic_execution": False,
            "broker_or_order_endpoint": False,
        }

    def allocation_history(self) -> dict:
        items = self.portfolio.read_allocation_snapshots()
        return {
            "schema_version": "aurumpilot.user_allocation_history.v1",
            "count": len(items),
            "items": items,
            "append_only": True,
        }

    def history(self) -> dict:
        """Return immutable H1-only Today Advice records, newest first."""
        items = self.portfolio.read_today_advice_h1_logs()
        return {
            "schema_version": "aurumpilot.today_advice_h1_history.v1",
            "count": len(items),
            "items": list(reversed(items)),
            "append_only": True,
            "immutable": True,
            "logged_horizon_indices": [1],
            "h2_h21_excluded": True,
        }

    def _macro_snapshot(self) -> dict:
        try:
            raw = self.macro.assess().model_dump(mode="json")
        except Exception as exc:
            return {
                "snapshot_id": None,
                "label": "CAUTION",
                "summary": "今日宏观提示不可用",
                "as_of_utc": None,
                "status": "UNAVAILABLE",
                "reason_code": f"MACRO_UNAVAILABLE:{type(exc).__name__}",
                "assessment_supported": False,
                "informational_only": True,
                "trade_permission": False,
                "technical_model_input_allowed": False,
                "decision_engine_input_allowed": False,
            }
        summaries = raw.get("news_summary") or []
        supported = bool(raw.get("assessment_supported"))
        label = str(raw.get("risk_level") or "CAUTION").upper()
        return {
            "snapshot_id": raw.get("assessment_id"),
            "label": label,
            "summary": (
                str(summaries[0])
                if summaries
                else (
                    "宏观数据缺失"
                    if not supported
                    else "官方宏观数据已完成独立评估"
                )
            ),
            "as_of_utc": raw.get("decision_as_of_utc"),
            "status": "AVAILABLE" if supported else "DATA_MISSING",
            "reason_code": (
                None if supported else "MACRO_ASSESSMENT_UNSUPPORTED"
            ),
            "assessment_supported": supported,
            "informational_only": True,
            "trade_permission": False,
            "technical_model_input_allowed": False,
            "decision_engine_input_allowed": False,
        }

    def _personalized_target(
        self,
        allocation: dict | None,
        requested_delta_pp: float,
    ) -> dict:
        minimum = float(self.settings.technical_strategy_min_gold_weight) * 100
        maximum = float(self.settings.technical_strategy_max_gold_weight) * 100
        current = (
            float(allocation["current_gold_weight_pct"])
            if allocation is not None
            else None
        )
        base = {
            "strategy_min_weight_pct": minimum,
            "strategy_max_weight_pct": maximum,
            "strategy_bounds_source": (
                self.settings.technical_strategy_bounds_source
            ),
            "requested_delta_pp": requested_delta_pp,
        }
        base.update(
            calculate_personalized_target(
                current,
                requested_delta_pp,
                minimum,
                maximum,
            )
        )
        return base

    def compose(self, timezone_name: str) -> dict:
        try:
            timezone = ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("DISPLAY_TIMEZONE_INVALID") from exc

        record = self.technical.repository.latest_success()
        if record is None or record.output is None:
            raise TechnicalIssuanceUnavailableError(
                "TECHNICAL_ISSUANCE_NOT_READY"
            )
        product = compose_product_forecast(record.output)
        raw_slots = product.get("slots") or []
        if product.get("slot_count") != 21 or len(raw_slots) != 21:
            raise ValueError("TECHNICAL_SLOT_COUNT_INVALID")

        action = product.get("action") or {}
        action_name = ACTION_NAMES.get(str(action.get("action")))
        if action_name is None:
            raise ValueError("TECHNICAL_ACTION_INVALID")
        weight_before_pct = float(action["current_target_gold_weight"]) * 100
        weight_after_pct = float(action["recommended_target_gold_weight"]) * 100
        requested_delta_pp = weight_after_pct - weight_before_pct
        if action_name == "HOLD" and abs(requested_delta_pp) > 1e-9:
            raise ValueError("TECHNICAL_HOLD_DELTA_MISMATCH")
        if action_name == "ADD" and requested_delta_pp < 0:
            raise ValueError("TECHNICAL_ADD_DELTA_MISMATCH")
        if action_name == "REDUCE" and requested_delta_pp > 0:
            raise ValueError("TECHNICAL_REDUCE_DELTA_MISMATCH")

        slots = []
        for raw in raw_slots:
            index = int(raw["horizon_index"])
            probabilities = {
                key: float(value)
                for key, value in raw["probabilities"].items()
            }
            total = sum(probabilities.values())
            if abs(total - 1.0) > 1e-6:
                raise ValueError("TECHNICAL_PROBABILITY_SUM_INVALID")
            top = sum(
                probabilities[key]
                for key in ("TOP_L1", "TOP_L2", "TOP_L3")
            )
            bottom = sum(
                probabilities[key]
                for key in ("BOTTOM_L1", "BOTTOM_L2", "BOTTOM_L3")
            )
            label = str(raw.get("display_class") or "NORMAL")
            side = (
                "TOP"
                if label.startswith("TOP_")
                else "BOTTOM"
                if label.startswith("BOTTOM_")
                else "NORMAL"
            )
            start_utc = _iso_utc(raw["bucket_start_utc"])
            end_utc = _iso_utc(raw["bucket_end_utc"])
            local_start = datetime.fromisoformat(start_utc).astimezone(timezone)
            slots.append(
                {
                    "slot_id": f"{record.issuance_id}:H{index}",
                    "slot_index": index,
                    "horizon_index": index,
                    "target_bucket": raw["target_bucket"],
                    "target_bucket_start_utc": start_utc,
                    "target_bucket_end_utc": end_utc,
                    "local_date": local_start.date().isoformat(),
                    "local_start": local_start.isoformat(),
                    "display_timezone": timezone_name,
                    "predicted_label": label,
                    "probabilities": probabilities,
                    "aggregate_probabilities": {
                        "NORMAL": probabilities["NORMAL"],
                        "TOP": top,
                        "BOTTOM": bottom,
                    },
                    "conditional_price": _price_dto(
                        raw.get("conditional_price_outlook"), side
                    ),
                    "is_current_action_slot": index == 1,
                    "forecast_only": index != 1,
                    "actionable": bool(
                        index == 1 and action_name != "HOLD"
                    ),
                    "recommended_action": (
                        action_name if index == 1 else None
                    ),
                    "requested_delta_pp": (
                        requested_delta_pp if index == 1 else None
                    ),
                    "advisory_only": True,
                    "controls_trading": index == 1,
                }
            )

        first = slots[0]
        now = datetime.now(UTC)
        market_stale = False
        stale_reason = None
        try:
            market = self.technical.market_data.get_history(refresh=False)
            latest_bucket = market.frame["ts_event_utc"].iloc[-1].date()
            market_stale = latest_bucket != record.source_bucket
            if market_stale:
                stale_reason = "LATEST_MARKET_BUCKET_DIFFERS_FROM_ISSUANCE"
        except Exception as exc:
            market_stale = True
            stale_reason = f"MARKET_DATA_UNAVAILABLE:{type(exc).__name__}"
        h1_end = datetime.fromisoformat(first["target_bucket_end_utc"])
        h1_expired = now >= h1_end
        stale = market_stale or h1_expired
        if h1_expired and stale_reason is None:
            stale_reason = "H1_TARGET_BUCKET_EXPIRED"
        freshness_status = "STALE" if stale else "CURRENT"

        allocation = self._latest_allocation()
        user_allocation = None
        if allocation is not None:
            updated = datetime.fromisoformat(
                str(allocation["as_of_utc"]).replace("Z", "+00:00")
            ).astimezone(UTC)
            age_days = max(0, (now - updated).days)
            user_allocation = {
                "snapshot_id": allocation["snapshot_id"],
                "current_gold_weight_pct": float(
                    allocation["current_gold_weight_pct"]
                ),
                "updated_at_utc": updated.isoformat(),
                "age_days": age_days,
                "is_stale": (
                    age_days > self.settings.user_allocation_stale_days
                ),
                "source": allocation["source"],
            }
        personalized = self._personalized_target(
            allocation, requested_delta_pp
        )
        macro = self._macro_snapshot()

        price_identity = {
            "technical_issuance_id": record.issuance_id,
            "model_sha256": product["models"]["pn02_bundle_sha256"],
            "slots": [
                {
                    "horizon_index": item["horizon_index"],
                    "top": item.get("conditional_top_ohlc"),
                    "bottom": item.get("conditional_bottom_ohlc"),
                }
                for item in raw_slots
            ],
        }
        price_output_sha256 = canonical_json_sha256(price_identity)
        composite_identity = {
            "technical_issuance_id": record.issuance_id,
            "technical_output_sha256": record.output_sha256,
            "price_output_sha256": price_output_sha256,
            "macro_snapshot_id": macro["snapshot_id"],
            "portfolio_snapshot_id": (
                allocation.get("snapshot_id") if allocation else None
            ),
            "requested_delta_pp": requested_delta_pp,
            "effective_delta_pp": personalized["effective_delta_pp"],
            "personalized_target_weight_pct": personalized[
                "target_gold_weight_pct"
            ],
        }
        composite_id = (
            "composite_"
            + canonical_json_sha256(composite_identity)[:24].lower()
        )
        previous_composites = self.portfolio.read_composite_snapshots()
        composite_audit = {
            "schema_version": "aurumpilot.today_advice_composite_audit.v1",
            "composite_snapshot_id": composite_id,
            "created_at_utc": now.isoformat(),
            "supersedes_snapshot_id": (
                previous_composites[-1].get("composite_snapshot_id")
                if previous_composites
                and previous_composites[-1].get("composite_snapshot_id")
                != composite_id
                else None
            ),
            **composite_identity,
            "technical_model_manifest_sha256": product["models"][
                "mn18_bundle_manifest_sha256"
            ],
            "technical_model_artifact_sha256": product["models"][
                "mn18_joblib_sha256"
            ],
            "price_issuance_id": f"{record.issuance_id}:pn02",
            "price_model_artifact_sha256": product["models"][
                "pn02_bundle_sha256"
            ],
            "technical_output_unchanged": True,
            "automatic_execution": False,
        }
        self.portfolio.append_composite_snapshot(composite_audit)

        today = {
            **first,
            "recommended_action": action_name,
            "action_reason_code": action.get("reason_code"),
            "action_reason_text": action.get("reason_code"),
            "requested_delta_pp": requested_delta_pp,
            "model_expected_delta_pp_diagnostic": action.get(
                "expected_delta_pp"
            ),
            "model_tactical_weight_before_pct": weight_before_pct,
            "model_tactical_weight_after_pct": weight_after_pct,
            "execution_mode": "ADVISORY_ONLY",
            "execution_bucket_start_utc": first[
                "target_bucket_start_utc"
            ],
            "execution_bucket_end_utc": first["target_bucket_end_utc"],
        }
        response = {
            "schema_version": "aurumpilot.today_advice_page.v1",
            "composite_snapshot_id": composite_id,
            "freshness_status": freshness_status,
            "display_title": (
                "今日建议" if not stale else "最近可用建议"
            ),
            "stale_reason": stale_reason,
            "technical_status": "DEVELOPMENT_CANDIDATE",
            "price_status": product["composition"][
                "conditional_price_model_status"
            ],
            "macro_status": macro["status"],
            "technical_issuance": {
                "issuance_id": record.issuance_id,
                "issuance_kind": record.issuance_kind,
                "issued_at_utc": record.issued_at_utc.isoformat(),
                "source_bucket_utc": record.source_bucket.isoformat(),
                "source_bucket_end_utc": (
                    record.source_bucket_end_utc.isoformat()
                ),
                "output_sha256": record.output_sha256,
                "model_id": product["models"][
                    "action_probability_model_id"
                ],
                "model_manifest_sha256": product["models"][
                    "mn18_bundle_manifest_sha256"
                ],
                "model_artifact_sha256": product["models"][
                    "mn18_joblib_sha256"
                ],
                "persisted": True,
                "integrity_verified": True,
            },
            "price_issuance": {
                "issuance_id": f"{record.issuance_id}:pn02",
                "output_sha256": price_output_sha256,
                "model_id": product["models"][
                    "conditional_ohlc_model_id"
                ],
                "model_artifact_sha256": product["models"][
                    "pn02_bundle_sha256"
                ],
                "raw_top_and_bottom_preserved": True,
                "page_direction_gated": True,
            },
            "today_advice": today,
            "user_allocation": user_allocation,
            "personalized_target": personalized,
            "macro": macro,
            "slots": slots,
            "slot_count": 21,
            "page_get_runs_models": False,
            "macro_changes_technical_action": False,
            "conditional_price_controls_trading": False,
            "automatic_execution": False,
            "broker_or_order_endpoint": False,
        }
        log_identity = {
            "technical_issuance_id": record.issuance_id,
            "source_bucket_utc": record.source_bucket.isoformat(),
            "h1_slot_id": today["slot_id"],
            "technical_output_sha256": record.output_sha256,
        }
        h1_log = {
            "schema_version": "aurumpilot.today_advice_h1_log.v1",
            "log_id": (
                "h1_advice_"
                + canonical_json_sha256(log_identity)[:24].lower()
            ),
            "technical_issuance_id": record.issuance_id,
            "source_bucket_utc": record.source_bucket.isoformat(),
            "source_bucket_end_utc": (
                record.source_bucket_end_utc.isoformat()
            ),
            "issued_at_utc": record.issued_at_utc.isoformat(),
            "logged_at_utc": now.isoformat(),
            "freshness_status_at_log_time": freshness_status,
            "stale_reason_at_log_time": stale_reason,
            "today_advice": today,
            "technical_issuance": response["technical_issuance"],
            "price_issuance": response["price_issuance"],
            "user_allocation": user_allocation,
            "personalized_target": personalized,
            "macro": macro,
            "composite_snapshot_id": composite_id,
            "slot_count_logged": 1,
            "logged_horizon_indices": [1],
            "h2_h21_excluded": True,
            "immutable": True,
            "page_get_runs_models": False,
            "macro_changes_technical_action": False,
            "conditional_price_controls_trading": False,
            "automatic_execution": False,
            "broker_or_order_endpoint": False,
        }
        created = self.portfolio.append_today_advice_h1_log(h1_log)
        response["h1_log"] = {
            "log_id": h1_log["log_id"],
            "created": created,
            "persisted": True,
            "immutable": True,
            "logged_horizon_indices": [1],
            "h2_h21_excluded": True,
        }
        return response
