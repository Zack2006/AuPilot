"""Create immutable, idempotent MN18 + PN02 issuances from formal daily data."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any, Protocol
from uuid import uuid4

from aupilot.core.hashing import canonical_json_sha256

from backend.app.core.config import Settings
from backend.app.repositories.mn18_receiver_ledger import MN18ReceiverLedger
from backend.app.repositories.technical_issuance_repository import TechnicalIssuanceRepository
from backend.app.schemas.technical import (
    EXPECTED_COMPONENT_HASHES,
    TechnicalDailyOHLC,
    TechnicalEvaluationRequest,
    TechnicalEvaluationResult,
    TechnicalIssuanceRecord,
    TechnicalOutlook,
)
from backend.app.services.market_data_service import MarketDataService
from backend.app.services.technical_composition import build_composition_audit
from backend.app.services.technical_sidecar_client import TechnicalSidecarClient


MINIMUM_FORWARD_SOURCE_BUCKET = date(2026, 7, 27)
MINIMUM_PREVIEW_SOURCE_BUCKET = date(2026, 7, 22)


class TechnicalSidecar(Protocol):
    def health(self) -> dict[str, Any]: ...

    def evaluate(self, request: TechnicalEvaluationRequest) -> tuple[TechnicalOutlook, dict[str, Any]]: ...


class TechnicalIssuanceService:
    def __init__(
        self,
        settings: Settings,
        market_data: MarketDataService,
        *,
        repository: TechnicalIssuanceRepository | None = None,
        sidecar: TechnicalSidecar | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.settings = settings
        self.market_data = market_data
        self.repository = repository or TechnicalIssuanceRepository(
            settings.storage_dir
            / "technical"
            / "mn18_pn02_issuances.sqlite"
        )
        self.receiver = MN18ReceiverLedger(self.repository.path)
        self.sidecar = sidecar or TechnicalSidecarClient(
            settings.technical_sidecar_url,
            settings.technical_sidecar_timeout_seconds,
        )
        self.clock = clock or (lambda: datetime.now(UTC))
        self.id_factory = id_factory or (lambda: uuid4().hex[:16])

    @staticmethod
    def _aware_utc(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Technical issuance clock must be timezone-aware")
        return value.astimezone(UTC)

    @staticmethod
    def _contains_prohibited_key(value: object) -> bool:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).casefold()
                if "macro" in normalized or "rag" in normalized:
                    if child not in (False, None, [], {}):
                        return True
                    continue
                if TechnicalIssuanceService._contains_prohibited_key(child):
                    return True
        elif isinstance(value, list):
            return any(TechnicalIssuanceService._contains_prohibited_key(child) for child in value)
        return False

    @staticmethod
    def _request_from_market(
        frame,
        *,
        current_gold_weight: float,
        outstanding_top_inventory_pp: float,
        as_of: datetime,
        timezone: str,
    ):
        bars = [
            TechnicalDailyOHLC(
                trade_date=row.ts_event_utc.date(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
            )
            for row in frame.itertuples(index=False)
        ]
        return TechnicalEvaluationRequest(
            daily_history=bars,
            current_gold_weight=current_gold_weight,
            outstanding_top_inventory_pp=outstanding_top_inventory_pp,
            as_of_utc=as_of,
            display_timezone=timezone,
        )

    @staticmethod
    def _success_identity(
        *,
        source_bucket: date,
        issuance_kind: str,
        current_gold_weight: float,
        outstanding_top_inventory_pp: float,
        market_data_sha256: str,
        history_start: date,
        history_end: date,
        input_rows: int,
    ) -> str:
        return canonical_json_sha256(
            {
                "schema_version": "aupilot.mn18_pn02.issuance_identity.v1",
                "source_bucket": source_bucket.isoformat(),
                "issuance_kind": issuance_kind,
                "current_gold_weight": current_gold_weight,
                "outstanding_top_inventory_pp": (
                    outstanding_top_inventory_pp
                ),
                "market_data_sha256": market_data_sha256,
                "history_start": history_start.isoformat(),
                "history_end": history_end.isoformat(),
                "input_rows": input_rows,
                "component_hashes": EXPECTED_COMPONENT_HASHES,
            }
        )

    @staticmethod
    def _history_sha256(frame) -> str:
        """Hash exactly the causal OHLC rows supplied to the sidecar."""
        return canonical_json_sha256(
            [
                {
                    "trade_date": row.ts_event_utc.date().isoformat(),
                    "open": float(row.open),
                    "high": float(row.high),
                    "low": float(row.low),
                    "close": float(row.close),
                }
                for row in frame[
                    ["ts_event_utc", "open", "high", "low", "close"]
                ].itertuples(index=False)
            ]
        )

    def _record_failure(
        self,
        *,
        error: Exception,
        created_at: datetime,
        current_gold_weight: float,
        outstanding_top_inventory_pp: float,
        source_bucket: date | None = None,
        source_bucket_end: datetime | None = None,
        history_start: date | None = None,
        history_end: date | None = None,
        input_rows: int | None = None,
        market_data_sha256: str | None = None,
        input_request_sha256: str | None = None,
    ) -> None:
        attempt_id = self.id_factory()
        reason = f"{type(error).__name__}:{str(error)[:300]}"
        failure = TechnicalIssuanceRecord(
            issuance_id=f"fail_{created_at:%Y%m%dT%H%M%S}_{attempt_id}",
            idempotency_key=canonical_json_sha256(
                {
                    "status": "FAILED",
                    "created_at_utc": created_at.isoformat(),
                    "attempt_id": attempt_id,
                    "reason": reason,
                }
            ),
            status="FAILED",
            issuance_kind="FAILED_ATTEMPT",
            created_at_utc=created_at,
            source_bucket=source_bucket,
            source_bucket_end_utc=source_bucket_end,
            issued_at_utc=None,
            input_history_start=history_start,
            input_history_end=history_end,
            input_rows=input_rows,
            current_gold_weight=current_gold_weight,
            outstanding_top_inventory_pp=outstanding_top_inventory_pp,
            dual_package_manifest_hash=EXPECTED_COMPONENT_HASHES[
                "dual_package_manifest_sha256"
            ],
            mn18_manifest_hash=EXPECTED_COMPONENT_HASHES[
                "mn18_bundle_manifest_sha256"
            ],
            mn18_joblib_hash=EXPECTED_COMPONENT_HASHES[
                "mn18_joblib_sha256"
            ],
            pn02_bundle_hash=EXPECTED_COMPONENT_HASHES[
                "pn02_bundle_sha256"
            ],
            market_data_sha256=market_data_sha256,
            input_request_sha256=input_request_sha256,
            output=None,
            output_sha256=None,
            error_reason=reason,
        )
        self.repository.append(failure)

    def _ensure_composition_audit(
        self,
        record: TechnicalIssuanceRecord,
        *,
        data_quality_status: str | None,
        inference_duration_ms: float | None,
        created_at: datetime,
    ) -> None:
        if record.output is None or record.output_sha256 is None:
            return
        audit = build_composition_audit(
            issuance_id=record.issuance_id,
            output=record.output,
            output_sha256=record.output_sha256,
            market_data_sha256=record.market_data_sha256,
            input_request_sha256=record.input_request_sha256,
            data_quality_status=data_quality_status,
            inference_duration_ms=inference_duration_ms,
            created_at_utc=created_at,
        )
        self.repository.append_composition_audit(
            record.issuance_id,
            audit,
        )

    def _ensure_scheduled_requests(
        self,
        record: TechnicalIssuanceRecord,
    ) -> None:
        if record.output is None:
            return
        self.repository.append_scheduled_requests(
            issuance=record,
            model_version=str(record.output["model_version"]),
            requests=list(record.output["scheduled_action_requests"]),
        )

    def _evaluate_market(
        self,
        market,
        *,
        source_bucket: date | None = None,
    ) -> TechnicalEvaluationResult:
        created_at = self._aware_utc(self.clock())
        user_state = self.repository.current_state()
        current_gold_weight = user_state.current_gold_weight
        outstanding_top_inventory_pp = user_state.outstanding_top_inventory_pp
        context: dict[str, Any] = {}
        try:
            frame = market.frame
            if source_bucket is not None:
                frame = frame.loc[
                    frame["ts_event_utc"].dt.date <= source_bucket
                ].copy()
                if (
                    frame.empty
                    or frame["ts_event_utc"].iloc[-1].date()
                    != source_bucket
                ):
                    raise ValueError(
                        "TECHNICAL_SOURCE_BUCKET_NOT_IN_FORMAL_MARKET"
                    )
            source_bucket = frame["ts_event_utc"].iloc[-1].date()
            if source_bucket < MINIMUM_PREVIEW_SOURCE_BUCKET:
                raise ValueError(
                    "MN18_SOURCE_BEFORE_TRAINING_CUTOFF_FORBIDDEN"
                )
            issuance_kind = (
                "COMPLIANT_FORWARD"
                if source_bucket >= MINIMUM_FORWARD_SOURCE_BUCKET
                else "PRE_FORWARD_PRODUCT_PREVIEW"
            )
            if issuance_kind == "COMPLIANT_FORWARD":
                shadow_state = self.receiver.shadow_state()
                current_gold_weight = float(
                    shadow_state["current_gold_weight"]
                )
                outstanding_top_inventory_pp = float(
                    shadow_state["outstanding_top_inventory_pp"]
                )
            source_bucket_end = datetime.combine(
                source_bucket + timedelta(days=1), datetime.min.time(), UTC
            )
            history_start = frame["ts_event_utc"].iloc[0].date()
            history_end = source_bucket
            market_data_sha256 = (
                market.metadata.content_sha256
                if len(frame) == len(market.frame)
                else self._history_sha256(frame)
            )
            context.update(
                source_bucket=source_bucket,
                source_bucket_end=source_bucket_end,
                history_start=history_start,
                history_end=history_end,
                input_rows=len(frame),
                market_data_sha256=market_data_sha256,
            )
            data_quality_status = getattr(
                market.metadata,
                "quality_status",
                None,
            )
            if created_at < source_bucket_end:
                raise ValueError("TECHNICAL_AS_OF_PRECEDES_SOURCE_BUCKET_AVAILABILITY")
            if history_start != self.settings.technical_history_start:
                raise ValueError("TECHNICAL_FULL_HISTORY_START_MISMATCH")

            request = self._request_from_market(
                frame,
                current_gold_weight=current_gold_weight,
                outstanding_top_inventory_pp=outstanding_top_inventory_pp,
                as_of=created_at,
                timezone=self.settings.technical_display_timezone,
            )
            request_payload = request.model_dump(mode="json")
            request_sha256 = canonical_json_sha256(request_payload)
            context["input_request_sha256"] = request_sha256

            self.sidecar.health()
            idempotency_key = self._success_identity(
                source_bucket=source_bucket,
                issuance_kind=issuance_kind,
                current_gold_weight=current_gold_weight,
                outstanding_top_inventory_pp=(
                    outstanding_top_inventory_pp
                ),
                market_data_sha256=market_data_sha256,
                history_start=history_start,
                history_end=history_end,
                input_rows=len(frame),
            )
            existing = self.repository.find_success(idempotency_key)
            if existing is not None:
                self._ensure_scheduled_requests(existing)
                self._ensure_composition_audit(
                    existing,
                    data_quality_status=data_quality_status,
                    inference_duration_ms=None,
                    created_at=created_at,
                )
                return TechnicalEvaluationResult(issuance=existing, created=False)

            inference_started = time.perf_counter()
            outlook, raw_output = self.sidecar.evaluate(request)
            inference_duration_ms = round(
                (time.perf_counter() - inference_started) * 1000.0,
                3,
            )
            if self._contains_prohibited_key(raw_output):
                raise ValueError("TECHNICAL_OUTPUT_CONTAINS_MACRO_OR_RAG")
            if outlook.history.source_bucket != source_bucket:
                raise ValueError("TECHNICAL_SOURCE_BUCKET_MISMATCH")
            if outlook.as_of_utc.astimezone(UTC) != created_at:
                raise ValueError("TECHNICAL_ISSUED_AT_MISMATCH")
            if outlook.history.start_bucket != history_start:
                raise ValueError("TECHNICAL_INPUT_HISTORY_START_MISMATCH")
            if (
                abs(
                    outlook.action.current_target_gold_weight
                    - current_gold_weight
                )
                > 1e-12
            ):
                raise ValueError("TECHNICAL_CURRENT_GOLD_WEIGHT_MISMATCH")

            output_sha256 = canonical_json_sha256(raw_output)
            issuance = TechnicalIssuanceRecord(
                issuance_id=(
                    (
                        "iss_"
                        if issuance_kind == "COMPLIANT_FORWARD"
                        else "preview_"
                    )
                    + f"{source_bucket:%Y%m%d}_{idempotency_key[:16].lower()}"
                ),
                idempotency_key=idempotency_key,
                status="SUCCESS",
                issuance_kind=issuance_kind,
                created_at_utc=created_at,
                source_bucket=source_bucket,
                source_bucket_end_utc=source_bucket_end,
                issued_at_utc=outlook.as_of_utc,
                input_history_start=history_start,
                input_history_end=history_end,
                input_rows=len(frame),
                current_gold_weight=current_gold_weight,
                outstanding_top_inventory_pp=(
                    outstanding_top_inventory_pp
                ),
                dual_package_manifest_hash=EXPECTED_COMPONENT_HASHES[
                    "dual_package_manifest_sha256"
                ],
                mn18_manifest_hash=EXPECTED_COMPONENT_HASHES[
                    "mn18_bundle_manifest_sha256"
                ],
                mn18_joblib_hash=EXPECTED_COMPONENT_HASHES[
                    "mn18_joblib_sha256"
                ],
                pn02_bundle_hash=EXPECTED_COMPONENT_HASHES[
                    "pn02_bundle_sha256"
                ],
                market_data_sha256=market_data_sha256,
                input_request_sha256=request_sha256,
                output=raw_output,
                output_sha256=output_sha256,
                error_reason=None,
            )
            try:
                self.repository.append(issuance)
            except sqlite3.IntegrityError:
                concurrent = self.repository.find_success(idempotency_key)
                if concurrent is None:
                    raise
                self._ensure_scheduled_requests(concurrent)
                self._ensure_composition_audit(
                    concurrent,
                    data_quality_status=data_quality_status,
                    inference_duration_ms=None,
                    created_at=created_at,
                )
                return TechnicalEvaluationResult(issuance=concurrent, created=False)
            self._ensure_scheduled_requests(issuance)
            self._ensure_composition_audit(
                issuance,
                data_quality_status=data_quality_status,
                inference_duration_ms=inference_duration_ms,
                created_at=created_at,
            )
            return TechnicalEvaluationResult(issuance=issuance, created=True)
        except Exception as exc:
            try:
                self._record_failure(
                    error=exc,
                    created_at=created_at,
                    current_gold_weight=current_gold_weight,
                    outstanding_top_inventory_pp=(
                        outstanding_top_inventory_pp
                    ),
                    **context,
                )
            except Exception:
                pass
            raise

    def evaluate_latest(
        self, *, refresh_market: bool = True
    ) -> TechnicalEvaluationResult:
        # A formal model request is one of the explicit Databento refresh
        # boundaries: initial cache creation, scheduled issuance, or the
        # user's manual model-validation refresh action.
        market = self.market_data.get_history(refresh=refresh_market)
        return self._evaluate_market(market)

    def evaluate_pending(
        self, *, refresh_market: bool = True
    ) -> list[TechnicalEvaluationResult]:
        """Catch up every newly completed formal bucket, once and in order.

        The first run never reconstructs history: without an existing formal
        issuance it evaluates only the latest complete bucket. Later runs crop
        the verified cache at each missing bucket, so no later OHLC row enters
        that bucket's model request.
        """
        market = self.market_data.get_history(refresh=refresh_market)
        successes = self.repository.list_records(include_failures=False)
        existing_sources = {
            record.source_bucket
            for record in successes
            if record.source_bucket is not None
        }
        market_dates = market.frame["ts_event_utc"].dt.date.tolist()
        eligible_market_dates = [
            value
            for value in market_dates
            if value >= MINIMUM_FORWARD_SOURCE_BUCKET
        ]
        if not existing_sources:
            latest = market_dates[-1]
            pending = (
                [latest]
                if latest >= MINIMUM_PREVIEW_SOURCE_BUCKET
                else []
            )
        else:
            pending = [
                value
                for value in eligible_market_dates
                if value not in existing_sources
            ]
        if not pending:
            return []
        results: list[TechnicalEvaluationResult] = []
        for value in pending:
            prior_forward_exists = any(
                record.issuance_kind == "COMPLIANT_FORWARD"
                and record.source_bucket is not None
                and record.source_bucket < value
                for record in self.repository.list_records(
                    include_failures=False
                )
            )
            if prior_forward_exists:
                target_row = market.frame.loc[
                    market.frame["ts_event_utc"].dt.date == value
                ].iloc[0]
                self.receiver.settle_shadow_target(
                    target_bucket=value,
                    target_open=float(target_row["open"]),
                    completed_at_utc=self._aware_utc(self.clock()),
                    transaction_cost_bps_per_side=2.0,
                )
            results.append(
                self._evaluate_market(market, source_bucket=value)
            )
        return results
