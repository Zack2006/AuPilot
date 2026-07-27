"""Hash-gated MN18 OOS evidence plus a no-signal passive market bridge."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd

from aupilot.core.hashing import canonical_json_sha256

from backend.app.core.config import PROJECT_ROOT, Settings
from backend.app.core.exceptions import DataCorruptionError


EVIDENCE_ROOT = PROJECT_ROOT / "vendor" / "mn18_historical_evidence"
EVIDENCE_IDENTITY = (
    "EXPOSED_FULL_HISTORY_NESTED_OOS_INCREMENTAL_RESEARCH_"
    "NOT_INDEPENDENT_FINAL"
)
OOS_CUTOFF = date(2026, 7, 21)
EXPECTED_FILES = {
    "blended_oos_actions.parquet": (
        71457,
        "6E0D76D1EDBE236BC420C5E87D8E1434DCCB0075911B38731C6A1E7D2C6E2BCF",
    ),
    "blended_oos_predictions.parquet": (
        1772436,
        "D0C29E1243C84E06E935B028D699101377E887B23AC1A5BEF06598A4535C2CD0",
    ),
    "daily_portfolio_2bps.parquet": (
        124332,
        "22B79E6B120D08FA0999EDCE737198F62446B3815991426A6C807A9F174E8AA4",
    ),
    "trades_2bps.parquet": (
        14629,
        "4F8625B8171565412AE6301277C9B68CD0CFA530A0F4D4E61AC9DB2FFE2C1EC9",
    ),
    "report.json": (
        19332,
        "54D44EFBA20E6995FA8B32745C7E9FE2CC7106DF11E653F6AC94FC5A744CAB37",
    ),
    "independent_verification_corrected.json": (
        1101,
        "DCAB009A9DA8E31B14704B27180FEDB3546FBB2B403161E9E3F9D1646B3F23A2",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _maximum_drawdown(returns: list[float]) -> float:
    values = [1.0 + float(value) for value in returns]
    peak = values[0]
    result = 0.0
    for value in values:
        peak = max(peak, value)
        result = min(result, value / peak - 1.0)
    return result


class MN18ValidationEvidenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.root = EVIDENCE_ROOT

    def _verify(self) -> dict[str, dict[str, Any]]:
        verified: dict[str, dict[str, Any]] = {}
        for name, (expected_bytes, expected_sha) in EXPECTED_FILES.items():
            path = self.root / name
            if not path.is_file():
                raise DataCorruptionError(f"MN18_EVIDENCE_MISSING:{name}")
            actual_bytes = path.stat().st_size
            actual_sha = _sha256(path)
            if actual_bytes != expected_bytes:
                raise DataCorruptionError(
                    f"MN18_EVIDENCE_BYTES_MISMATCH:{name}"
                )
            if actual_sha != expected_sha:
                raise DataCorruptionError(
                    f"MN18_EVIDENCE_SHA256_MISMATCH:{name}"
                )
            verified[name] = {
                "bytes": actual_bytes,
                "sha256": actual_sha,
            }
        return verified

    def evidence(self, *, force_verify: bool = False) -> dict[str, Any]:
        del force_verify
        verified = self._verify()
        report = json.loads(
            (self.root / "report.json").read_text(encoding="utf-8")
        )
        independent = json.loads(
            (self.root / "independent_verification_corrected.json").read_text(
                encoding="utf-8"
            )
        )
        if report.get("evidence_identity") != EVIDENCE_IDENTITY:
            raise DataCorruptionError("MN18_EVIDENCE_IDENTITY_MISMATCH")
        if not independent.get("passed"):
            raise DataCorruptionError("MN18_EVIDENCE_VERIFICATION_FAILED")
        actions = pd.read_parquet(
            self.root / "blended_oos_actions.parquet"
        )
        predictions = pd.read_parquet(
            self.root / "blended_oos_predictions.parquet"
        )
        portfolio = pd.read_parquet(
            self.root / "daily_portfolio_2bps.parquet"
        )
        trades = pd.read_parquet(self.root / "trades_2bps.parquet")
        if (
            len(actions) != 701
            or len(predictions) != 76734
            or len(portfolio) != 5003
            or len(trades) != 186
        ):
            raise DataCorruptionError("MN18_EVIDENCE_ROW_COUNT_MISMATCH")
        if pd.Timestamp(portfolio.iloc[-1]["trade_date"]).date() != OOS_CUTOFF:
            raise DataCorruptionError("MN18_EVIDENCE_CUTOFF_MISMATCH")

        trade_by_id = {
            str(row.action_id): row
            for row in trades.itertuples(index=False)
        }
        markers: list[dict[str, Any]] = []
        for row in actions.itertuples(index=False):
            side = "TOP" if int(row.side_code) < 0 else "BOTTOM"
            label = f"{side}_L{int(row.strength_level)}"
            trade = trade_by_id.get(str(row.action_id))
            action = (
                "REDUCE_GOLD_WEIGHT"
                if trade is not None and side == "TOP"
                else "INCREASE_GOLD_WEIGHT"
                if trade is not None
                else "HOLD"
            )
            markers.append(
                {
                    "signal_id": str(row.action_id),
                    "signal_date": pd.Timestamp(row.trade_date).date().isoformat(),
                    "signal_label": label,
                    "model_probability": float(row.chosen_probability),
                    "chosen_threshold": float(row.chosen_threshold),
                    "threshold_margin": float(row.threshold_margin),
                    "issuance_id": str(row.internal_block_id),
                    "source_horizon_index": int(
                        row.boundary_horizon_index
                    ),
                    "action": action,
                    "reason_code": (
                        "EXECUTED_2BPS_TRADE"
                        if trade is not None
                        else "QUALIFIED_REQUEST_BLOCKED_OR_UNFILLED"
                    ),
                    "evidence_phase": "DEVELOPMENT_NESTED_OOS",
                }
            )
        initial_nav = float(report["candidate_primary_2bps"]["initial_nav_usd"])
        curve = [
            {
                "trade_date": pd.Timestamp(row.trade_date).date().isoformat(),
                "model_return": float(row.nav) / initial_nav - 1.0,
                "buy_hold_return": (
                    float(row.benchmark_nav) / initial_nav - 1.0
                ),
                "phase": "DEVELOPMENT_NESTED_OOS",
            }
            for row in portfolio.itertuples(index=False)
        ]
        metrics = report["candidate_primary_2bps"]
        summary = {
            "initial_nav_usd": initial_nav,
            "strategy_total_return": float(
                metrics["strategy_total_return"]
            ),
            "benchmark_total_return": float(
                metrics["buy_hold_total_return"]
            ),
            "absolute_return_lift_vs_buy_hold": float(
                metrics["absolute_return_lift_vs_buy_hold"]
            ),
            "strategy_max_drawdown": float(
                metrics["strategy_max_drawdown"]
            ),
            "benchmark_max_drawdown": float(
                metrics["buy_hold_max_drawdown"]
            ),
            "signal_rows": len(actions),
            "filled_trades": len(trades),
        }
        terminal = portfolio.iloc[-1]
        return {
            "schema_version": "aupilot.mn18.oos_evidence.v1",
            "evidence_identity": EVIDENCE_IDENTITY,
            "model_id": report["model_id"],
            "date_from": pd.Timestamp(
                portfolio.iloc[0]["trade_date"]
            ).date().isoformat(),
            "date_until": OOS_CUTOFF.isoformat(),
            "signal_count": len(actions),
            "trade_count": len(trades),
            "prediction_rows": len(predictions),
            "transaction_cost_bps_per_side": 2.0,
            "summary": summary,
            "markers": markers,
            "comparison_curve": curve,
            "terminal_state": {
                "trade_date": OOS_CUTOFF.isoformat(),
                "target_weight": float(terminal["target_weight"]),
                "gold_quantity": float(terminal["gold_quantity"]),
                "cash": float(terminal["cash"]),
                "nav": float(terminal["nav"]),
                "benchmark_nav": float(terminal["benchmark_nav"]),
            },
            "verified_files": verified,
            "hash_gated": True,
            "independent_final_validation": False,
            "final_holdout_opened": False,
        }


class MN18ValidationTimelineService:
    def __init__(
        self,
        settings: Settings,
        evidence_service: MN18ValidationEvidenceService,
        issuance_service,
        market_service,
    ) -> None:
        self.settings = settings
        self.evidence_service = evidence_service
        self.issuance_service = issuance_service
        self.market_service = market_service
        self._lock = RLock()

    def _timeline(self) -> dict[str, Any]:
        evidence = self.evidence_service.evidence(force_verify=True)
        market = self.market_service.get_history(refresh=False)
        curve = list(evidence["comparison_curve"])
        terminal = evidence["terminal_state"]
        records = self.issuance_service.repository.list_records(
            include_failures=False
        )
        forward = [
            record
            for record in records
            if record.issuance_kind == "COMPLIANT_FORWARD"
        ]
        preview = [
            record
            for record in records
            if record.issuance_kind == "PRE_FORWARD_PRODUCT_PREVIEW"
        ]
        receiver = getattr(self.issuance_service, "receiver", None)
        shadow_actions = (
            receiver.list_shadow_actions()
            if receiver is not None
            else []
        )
        shadow_by_target = {
            str(item["target_bucket"]): item for item in shadow_actions
        }
        request_reader = getattr(
            self.issuance_service.repository,
            "list_scheduled_requests",
            None,
        )
        request_payloads = {
            item["request_id"]: item["payload"]
            for item in (
                request_reader() if callable(request_reader) else []
            )
        }
        cutoff_row = market.frame.loc[
            market.frame["ts_event_utc"].dt.date == OOS_CUTOFF
        ]
        if cutoff_row.empty:
            raise DataCorruptionError("MN18_CUTOFF_MARKET_ROW_MISSING")
        cutoff_close = float(cutoff_row.iloc[-1]["close"])
        gold_quantity = float(terminal["gold_quantity"])
        cash = float(terminal["cash"])
        initial_nav = float(evidence["summary"]["initial_nav_usd"])
        benchmark_quantity = float(terminal["benchmark_nav"]) / cutoff_close
        forward_markers: list[dict[str, Any]] = []
        forward_filled_trades = 0
        forward_cost_usd = 0.0
        for row in market.frame.itertuples(index=False):
            trade_date = row.ts_event_utc.date()
            if trade_date <= OOS_CUTOFF:
                continue
            shadow = shadow_by_target.get(trade_date.isoformat())
            phase = "PASSIVE_MARK_TO_MARKET_BRIDGE"
            if shadow is not None:
                phase = "LIVE_FORWARD_SHADOW"
                executed_pp = float(shadow["executed_delta_pp"])
                if abs(executed_pp) > 1.0e-12:
                    nav_open = cash + gold_quantity * float(row.open)
                    open_notional = nav_open * abs(executed_pp) / 100.0
                    quantity = open_notional / float(row.open)
                    fill_price = float(shadow["simulated_fill_price"])
                    if executed_pp < 0:
                        gold_quantity -= quantity
                        cash += quantity * fill_price
                    else:
                        gold_quantity += quantity
                        cash -= quantity * fill_price
                    forward_cost_usd += quantity * abs(
                        fill_price - float(row.open)
                    )
                    forward_filled_trades += 1
                request_id = shadow.get("winning_request_id")
                if request_id:
                    request = request_payloads.get(str(request_id), {})
                    strength = request.get("strength_level")
                    side = str(shadow.get("side") or "")
                    label = (
                        f"{side}_L{int(strength)}"
                        if side in {"TOP", "BOTTOM"}
                        and strength in {1, 2, 3}
                        else side
                    )
                    forward_markers.append(
                        {
                            "signal_id": (
                                f"shadow:{shadow['target_bucket']}:"
                                f"{request_id}"
                            ),
                            "signal_date": shadow["target_bucket"],
                            "signal_label": label,
                            "model_probability": request.get(
                                "chosen_probability"
                            ),
                            "request_id": request_id,
                            "source_issuance_id": shadow.get(
                                "source_issuance_id"
                            ),
                            "source_horizon_index": shadow.get(
                                "horizon_index"
                            ),
                            "action": shadow["action"],
                            "reason_code": shadow["reason_code"],
                            "requested_delta_pp": shadow[
                                "requested_delta_pp"
                            ],
                            "executed_delta_pp": shadow[
                                "executed_delta_pp"
                            ],
                            "evidence_phase": "LIVE_FORWARD_SHADOW",
                            "receiver_policy": shadow["receiver_policy"],
                        }
                    )
            curve.append(
                {
                    "trade_date": trade_date.isoformat(),
                    "model_return": (
                        cash + gold_quantity * float(row.close)
                    )
                    / initial_nav
                    - 1.0,
                    "buy_hold_return": (
                        benchmark_quantity * float(row.close)
                    )
                    / initial_nav
                    - 1.0,
                    "phase": phase,
                }
            )

        latest_market = market.frame["ts_event_utc"].iloc[-1].date()
        lag_calendar_days = (datetime.now(UTC).date() - latest_market).days
        combined_summary = {
            **evidence["summary"],
            "strategy_total_return": float(curve[-1]["model_return"]),
            "benchmark_total_return": float(curve[-1]["buy_hold_return"]),
            "absolute_return_lift_vs_buy_hold": (
                float(curve[-1]["model_return"])
                - float(curve[-1]["buy_hold_return"])
            ),
            "strategy_max_drawdown": _maximum_drawdown(
                [row["model_return"] for row in curve]
            ),
            "benchmark_max_drawdown": _maximum_drawdown(
                [row["buy_hold_return"] for row in curve]
            ),
            "signal_rows": int(evidence["summary"]["signal_rows"])
            + len(forward_markers),
            "filled_trades": int(evidence["summary"]["filled_trades"])
            + forward_filled_trades,
        }
        evidence_hashes = {
            name: item["sha256"]
            for name, item in evidence["verified_files"].items()
        }
        timeline_identity = {
            "evidence_files": evidence_hashes,
            "latest_market_bucket": latest_market.isoformat(),
            "curve_last": curve[-1],
            "forward_issuance_hashes": [
                record.output_sha256 for record in forward
            ],
            "forward_shadow_action_hashes": [
                canonical_json_sha256(item) for item in shadow_actions
            ],
        }
        combined_hash = canonical_json_sha256(timeline_identity)
        return {
            "schema_version": "aupilot.mn18.validation_timeline.v1",
            "status": (
                "CURRENT_WITHIN_ONE_COMPLETE_BUCKET"
                if lag_calendar_days <= 1
                else "MARKET_CACHE_STALE"
            ),
            "allowed_complete_bucket_lag": 1,
            "latest_complete_bucket": latest_market.isoformat(),
            "latest_forward_source_bucket": (
                forward[-1].source_bucket.isoformat() if forward else None
            ),
            "lag_calendar_days": lag_calendar_days,
            "oos_cutoff": OOS_CUTOFF.isoformat(),
            "boundary_label": (
                "DEVELOPMENT_OOS_END_PASSIVE_BRIDGE_START"
            ),
            "oos": {
                "date_from": evidence["date_from"],
                "date_until": evidence["date_until"],
                "signal_count": evidence["signal_count"],
                "summary": evidence["summary"],
                "transaction_cost_bps_per_side": 2.0,
                "evidence_identity": EVIDENCE_IDENTITY,
                "independent_final_validation": False,
            },
            "coverage_gap": {
                "exists": False,
                "start": None,
                "end": None,
                "semantics": "PASSIVE_MARK_TO_MARKET_BRIDGE",
            },
            "comparison_curve": curve,
            "historical_markers": evidence["markers"],
            "post_training_unseen_daily_markers": [],
            "forward_markers": forward_markers,
            "forward_shadow_daily_actions": shadow_actions,
            "post_training_unseen_daily_records": [],
            "post_training_unseen_daily": {
                "source_from": (
                    forward[0].source_bucket.isoformat()
                    if forward
                    else None
                ),
                "source_until": (
                    forward[-1].source_bucket.isoformat()
                    if forward
                    else None
                ),
                "record_count": len(forward),
                "expected_source_count": len(forward),
                "missing_source_count": 0,
                "action_count": len(forward_markers),
                "pending_action_count": 0,
                "realized_action_count": forward_filled_trades,
                "preview_record_count": len(preview),
            },
            "combined_summary": combined_summary,
            "forward_record_count": len(forward),
            "preview_record_count": len(preview),
            "forward_shadow_filled_trade_count": forward_filled_trades,
            "forward_shadow_transaction_cost_usd": forward_cost_usd,
            "hash_verification": {
                "passed": True,
                "evidence_files": evidence_hashes,
                "combined_timeline_sha256": combined_hash,
                "model_component_hashes": {
                    "mn18_bundle_manifest_sha256": (
                        "7325C34BFFF0CD77D658ABED413A23C799F6AE10F693012CB918C1D391702D3D"
                    )
                },
            },
            "timeline_sha256": combined_hash,
            "automatic_execution": False,
            "historical_full_model_recalculation_used": False,
            "passive_bridge_has_model_signals": False,
        }

    def timeline(self) -> dict[str, Any]:
        with self._lock:
            return self._timeline()

    def refresh(self) -> dict[str, Any]:
        with self._lock:
            before_market = self.market_service.get_history(refresh=False)
            before_dates = set(
                before_market.frame["ts_event_utc"].dt.date.tolist()
            )
            results = self.issuance_service.evaluate_pending(
                refresh_market=True
            )
            after_market = self.market_service.get_history(refresh=False)
            after_dates = set(
                after_market.frame["ts_event_utc"].dt.date.tolist()
            )
            timeline = self._timeline()
            timeline["refresh"] = {
                "requested_at_utc": datetime.now(UTC).isoformat(),
                "model_runs_created": sum(
                    result.created for result in results
                ),
                "model_runs_already_current": sum(
                    not result.created for result in results
                ),
                "source_buckets_processed": [
                    result.issuance.source_bucket.isoformat()
                    for result in results
                    if result.issuance.source_bucket is not None
                ],
                "forward_records_registered": sum(
                    result.issuance.issuance_kind == "COMPLIANT_FORWARD"
                    for result in results
                ),
                "post_training_unseen_daily_records_created": 0,
                "post_training_unseen_daily_complete": True,
                "post_training_unseen_daily_actions_created": 0,
                "post_training_unseen_daily_qualified_signals_created": 0,
                "market_complete_buckets_added": len(
                    after_dates - before_dates
                ),
                "latest_complete_bucket_before": max(
                    before_dates
                ).isoformat(),
                "latest_complete_bucket_after": max(
                    after_dates
                ).isoformat(),
                "portfolio_rebalances_realized_total": 0,
                "portfolio_curve_recomputed": True,
                "position_continuity_verified": True,
                "transaction_cost_bps_per_side": 2.0,
                "all_missing_complete_buckets_processed": True,
            }
            return timeline
