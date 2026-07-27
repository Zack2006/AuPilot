"""Readiness gate for formal, immutable out-of-sample prediction evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from pydantic import ValidationError

from backend.app.core.config import Settings
from backend.app.core.exceptions import FormalValidationUnavailableError, MarketDataUnavailableError
from backend.app.repositories.formal_prediction_repository import FormalPredictionRepository
from backend.app.schemas.formal_model import FormalModelManifest, FormalPredictionRecord
from backend.app.schemas.model_validation import (
    EquityCurvePoint,
    ModelValidationReport,
    ModelValidationStatus,
    PredictionAccuracy,
    PredictionEvaluationRecord,
    PredictionRecordsResponse,
    StrategyPerformance,
)
from backend.app.services.formal_model_registry import FormalModelRegistry
from backend.app.services.local_settings_service import LocalSettingsService
from backend.app.services.market_data_service import MarketDataService


@dataclass(frozen=True)
class _FormalContext:
    manifest: FormalModelManifest
    manifest_sha256: str
    records: list[FormalPredictionRecord]
    prediction_log_head_sha256: str


class ModelValidationService:
    """Reject Mock/sample records and expose only fully audited formal inputs."""

    def __init__(
        self,
        settings: Settings,
        local_settings: LocalSettingsService,
        market_data: MarketDataService | None = None,
    ) -> None:
        self.registry = FormalModelRegistry(settings)
        self.manifest_path = self.registry.manifest_path
        self.prediction_log_path = settings.storage_dir / "predictions" / "formal_predictions.jsonl"
        self.predictions = FormalPredictionRepository(self.prediction_log_path)
        self.local_settings = local_settings
        self.market_data = market_data or MarketDataService(settings, local_settings)

    def _validated_manifest(self) -> tuple[FormalModelManifest | None, str | None, str | None]:
        return self.registry.validated_manifest()

    @staticmethod
    def _compatible_records(
        records: list[FormalPredictionRecord],
        manifest: FormalModelManifest,
        manifest_sha256: str,
    ) -> list[FormalPredictionRecord]:
        turning_sha = manifest.artifact("turning_point").sha256
        price_sha = manifest.artifact("price").sha256
        compatible: list[FormalPredictionRecord] = []
        for record in records:
            if (
                record.model_manifest_sha256 != manifest_sha256
                or record.model_version != manifest.model_version
                or record.feature_schema_version != manifest.feature_schema_version
                or record.training_data_cutoff_utc != manifest.training_data_cutoff_utc
                or record.generated_at_utc < manifest.activated_at_utc
                or record.code_sha != manifest.code_sha
                or record.turning_point_model_sha256 != turning_sha
                or record.price_model_sha256 != price_sha
                or record.prediction_for_date < manifest.out_of_sample_start_date
            ):
                raise ValueError("Formal prediction record is incompatible with the active manifest")
            compatible.append(record)
        return compatible

    @staticmethod
    def _active_records(records: list[FormalPredictionRecord]) -> list[FormalPredictionRecord]:
        superseded = {item.supersedes_prediction_id for item in records if item.supersedes_prediction_id}
        return [item for item in records if item.prediction_id not in superseded]

    def _formal_context(self) -> _FormalContext:
        manifest, manifest_sha256, reason = self._validated_manifest()
        if manifest is None or manifest_sha256 is None:
            raise FormalValidationUnavailableError(reason or "FORMAL_MODEL_MANIFEST_INVALID")
        try:
            all_records = self.predictions.read_all()
            compatible = self._compatible_records(all_records, manifest, manifest_sha256)
            active = self._active_records(compatible)
        except (OSError, ValueError, ValidationError) as exc:
            raise FormalValidationUnavailableError("FORMAL_PREDICTION_LOG_INVALID") from exc
        if not active:
            raise FormalValidationUnavailableError("FORMAL_PREDICTION_LOG_EMPTY")
        return _FormalContext(
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            records=active,
            prediction_log_head_sha256=all_records[-1].record_sha256,
        )

    @staticmethod
    def _selected_records(
        records: list[FormalPredictionRecord], start_date: date | None, end_date: date | None
    ) -> tuple[date, date, list[FormalPredictionRecord]]:
        available_from = min(item.prediction_for_date for item in records)
        available_until = max(item.prediction_for_date for item in records)
        selected_from = start_date or available_from
        selected_until = end_date or available_until
        if selected_from > selected_until:
            raise ValueError("start_date must not be after end_date")
        if selected_from < available_from or selected_until > available_until:
            raise ValueError("Requested range must stay inside the formal prediction record range")
        selected = [
            item for item in records if selected_from <= item.prediction_for_date <= selected_until
        ]
        if not selected:
            raise ValueError("Requested range contains no formal predictions")
        return selected_from, selected_until, selected

    @staticmethod
    def _actual_direction(prior_close: float, actual_close: float) -> str:
        difference = actual_close - prior_close
        if abs(difference) <= max(abs(prior_close), 1.0) * 1e-12:
            return "NEUTRAL"
        return "UP" if difference > 0 else "DOWN"

    @staticmethod
    def _turning_point_hit(
        frame: pd.DataFrame,
        target_index: int,
        record: FormalPredictionRecord,
        manifest: FormalModelManifest,
    ) -> bool | None:
        if record.turning_point == "NONE":
            return None
        rule = manifest.turning_point_evaluation
        future = frame.iloc[target_index + 1 : target_index + 1 + rule.evaluation_window_sessions]
        if len(future) < rule.evaluation_window_sessions:
            return None
        anchor = float(frame.iloc[target_index]["close"])
        if record.turning_point == "UP_TURN":
            return float(future["close"].max()) / anchor - 1 >= rule.minimum_move_fraction
        return float(future["close"].min()) / anchor - 1 <= -rule.minimum_move_fraction

    def _evaluation_records(
        self,
        frame: pd.DataFrame,
        records: list[FormalPredictionRecord],
        manifest: FormalModelManifest,
    ) -> list[PredictionEvaluationRecord]:
        timestamps = pd.to_datetime(frame["ts_event_utc"], utc=True)
        date_to_index = {timestamp.date(): index for index, timestamp in enumerate(timestamps)}
        evaluated: list[PredictionEvaluationRecord] = []
        for record in records:
            common = {
                "prediction_id": record.prediction_id,
                "prediction_for_date": record.prediction_for_date,
                "generated_at_utc": record.generated_at_utc,
                "market_data_as_of_utc": record.market_data_as_of_utc,
                "model_version": record.model_version,
                "predicted_close": record.predicted_close,
                "predicted_low": record.predicted_low,
                "predicted_high": record.predicted_high,
                "expected_direction": record.expected_direction,
                "direction_confidence": record.direction_confidence,
                "turning_point": record.turning_point,
                "turning_point_confidence": record.turning_point_confidence,
                "target_exposure": record.target_exposure,
                "market_data_sha256": record.market_data_sha256,
                "record_sha256": record.record_sha256,
            }
            target_index = date_to_index.get(record.prediction_for_date)
            if target_index is None:
                evaluated.append(PredictionEvaluationRecord(**common, evaluation_status="MARKET_BAR_MISSING"))
                continue
            prior_indexes = frame.index[timestamps <= record.market_data_as_of_utc].tolist()
            if not prior_indexes:
                evaluated.append(PredictionEvaluationRecord(**common, evaluation_status="PRIOR_BAR_MISSING"))
                continue
            target = frame.iloc[target_index]
            prior_close = float(frame.loc[prior_indexes[-1], "close"])
            actual_close = float(target["close"])
            actual_direction = self._actual_direction(prior_close, actual_close)
            evaluated.append(
                PredictionEvaluationRecord(
                    **common,
                    actual_open=float(target["open"]),
                    actual_high=float(target["high"]),
                    actual_low=float(target["low"]),
                    actual_close=actual_close,
                    actual_direction=actual_direction,
                    close_error=record.predicted_close - actual_close,
                    absolute_percentage_error=abs(record.predicted_close - actual_close) / actual_close,
                    direction_hit=record.expected_direction == actual_direction,
                    interval_covered=record.predicted_low <= actual_close <= record.predicted_high,
                    turning_point_hit=self._turning_point_hit(frame, target_index, record, manifest),
                    evaluation_status="EVALUATED",
                )
            )
        return evaluated

    def records(self, start_date: date | None = None, end_date: date | None = None) -> PredictionRecordsResponse:
        context = self._formal_context()
        selected_from, selected_until, selected = self._selected_records(
            context.records, start_date, end_date
        )
        market = self.market_data.get_history(refresh=False)
        items = self._evaluation_records(market.frame, selected, context.manifest)
        return PredictionRecordsResponse(
            model_version=context.manifest.model_version,
            model_manifest_sha256=context.manifest_sha256,
            prediction_log_head_sha256=context.prediction_log_head_sha256,
            market_data_sha256=market.metadata.content_sha256,
            data_from_date=selected_from,
            data_until_date=selected_until,
            count=len(items),
            items=items,
        )

    @staticmethod
    def _max_drawdown(values: list[float]) -> float:
        peak = values[0]
        result = 0.0
        for value in values:
            peak = max(peak, value)
            result = min(result, value / peak - 1)
        return result

    @classmethod
    def _performance(
        cls,
        initial_capital: float,
        values: list[float],
        trade_count: int,
        traded_notional: float,
        transaction_cost: float,
        slippage_cost: float,
        ending_cash: float,
        ending_position_units: float,
    ) -> StrategyPerformance:
        return StrategyPerformance(
            initial_capital=initial_capital,
            final_equity=values[-1],
            cumulative_return=values[-1] / initial_capital - 1,
            max_drawdown=cls._max_drawdown([initial_capital, *values]),
            trade_count=trade_count,
            turnover=traded_notional / initial_capital,
            transaction_cost=transaction_cost,
            slippage_cost=slippage_cost,
            ending_cash=ending_cash,
            ending_position_units=ending_position_units,
        )

    @staticmethod
    def _accuracy(
        records: list[PredictionEvaluationRecord], daily_coverage_rate: float
    ) -> PredictionAccuracy:
        evaluated = [item for item in records if item.evaluation_status == "EVALUATED"]
        direction = [item for item in evaluated if item.direction_hit is not None]
        turning = [item for item in evaluated if item.turning_point_hit is not None]
        absolute_errors = [abs(item.close_error) for item in evaluated if item.close_error is not None]
        percentage_errors = [
            item.absolute_percentage_error for item in evaluated if item.absolute_percentage_error is not None
        ]
        intervals = [item.interval_covered for item in evaluated if item.interval_covered is not None]
        return PredictionAccuracy(
            evaluated_predictions=len(evaluated),
            direction_eligible=len(direction),
            direction_hits=sum(item.direction_hit is True for item in direction),
            direction_hit_rate=(sum(item.direction_hit is True for item in direction) / len(direction) if direction else None),
            mean_absolute_error=(sum(absolute_errors) / len(absolute_errors) if absolute_errors else None),
            mean_absolute_percentage_error=(
                sum(percentage_errors) / len(percentage_errors) if percentage_errors else None
            ),
            interval_coverage_rate=(sum(item is True for item in intervals) / len(intervals) if intervals else None),
            turning_point_eligible=len(turning),
            turning_point_hits=sum(item.turning_point_hit is True for item in turning),
            turning_point_hit_rate=(
                sum(item.turning_point_hit is True for item in turning) / len(turning) if turning else None
            ),
            daily_prediction_coverage_rate=daily_coverage_rate,
        )

    def report(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 2.0,
        slippage_bps: float = 0.0,
    ) -> ModelValidationReport:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0 <= transaction_cost_bps <= 1000 or not 0 <= slippage_bps <= 1000:
            raise ValueError("cost assumptions must be between 0 and 1000 basis points")

        context = self._formal_context()
        market = self.market_data.get_history(refresh=False)
        frame = market.frame.copy()
        market_dates = pd.to_datetime(frame["ts_event_utc"], utc=True).dt.date
        market_date_set = set(market_dates)
        latest_market_date = max(market_date_set)
        missing_realized_dates = [
            item.prediction_for_date
            for item in context.records
            if item.prediction_for_date <= latest_market_date and item.prediction_for_date not in market_date_set
        ]
        if missing_realized_dates:
            raise FormalValidationUnavailableError("FORMAL_MARKET_COVERAGE_INCOMPLETE")
        realized_records = [item for item in context.records if item.prediction_for_date in market_date_set]
        if not realized_records:
            raise FormalValidationUnavailableError("FORMAL_PREDICTIONS_NOT_YET_REALIZED")
        requested_start, requested_end, selected = self._selected_records(
            realized_records, start_date, end_date
        )
        bars = frame.loc[(market_dates >= requested_start) & (market_dates <= requested_end)].reset_index(drop=True)
        if bars.empty:
            raise ValueError("Requested range contains no Databento trading sessions")

        predictions_by_date = {item.prediction_for_date: item for item in selected}
        evaluations = self._evaluation_records(frame, selected, context.manifest)
        transaction_rate = transaction_cost_bps / 10_000
        slippage_rate = slippage_bps / 10_000
        total_cost_rate = transaction_rate + slippage_rate

        model_cash = initial_capital
        model_units = 0.0
        target_exposure = 0.0
        model_trade_count = 0
        model_traded_notional = 0.0
        model_transaction_cost = 0.0
        model_slippage_cost = 0.0
        model_values: list[float] = []

        first_open = float(bars.iloc[0]["open"])
        hold_notional = initial_capital / (1 + total_cost_rate)
        hold_units = hold_notional / first_open
        hold_transaction_cost = hold_notional * transaction_rate
        hold_slippage_cost = hold_notional * slippage_rate
        hold_cash = initial_capital - hold_notional - hold_transaction_cost - hold_slippage_cost
        hold_values: list[float] = []
        equity_curve: list[EquityCurvePoint] = []
        missing_dates: list[date] = []

        for _, bar in bars.iterrows():
            trading_date = pd.Timestamp(bar["ts_event_utc"]).date()
            open_price = float(bar["open"])
            close_price = float(bar["close"])
            prediction = predictions_by_date.get(trading_date)
            if prediction is not None:
                target_exposure = prediction.target_exposure
                equity_at_open = model_cash + model_units * open_price
                desired_units = equity_at_open * target_exposure / open_price
                delta_units = desired_units - model_units
                if delta_units > 0:
                    affordable_units = model_cash / (open_price * (1 + total_cost_rate))
                    delta_units = min(delta_units, affordable_units)
                traded_notional = abs(delta_units) * open_price
                if traded_notional > 1e-8:
                    transaction_cost = traded_notional * transaction_rate
                    slippage_cost = traded_notional * slippage_rate
                    model_cash -= delta_units * open_price + transaction_cost + slippage_cost
                    model_units += delta_units
                    model_trade_count += 1
                    model_traded_notional += traded_notional
                    model_transaction_cost += transaction_cost
                    model_slippage_cost += slippage_cost
            else:
                missing_dates.append(trading_date)

            model_equity = model_cash + model_units * close_price
            hold_equity = hold_cash + hold_units * close_price
            model_values.append(model_equity)
            hold_values.append(hold_equity)
            equity_curve.append(
                EquityCurvePoint(
                    date=trading_date,
                    model_equity=model_equity,
                    buy_hold_equity=hold_equity,
                    model_cash=model_cash,
                    buy_hold_cash=hold_cash,
                    model_units=model_units,
                    buy_hold_units=hold_units,
                    target_exposure=target_exposure,
                    prediction_id=None if prediction is None else prediction.prediction_id,
                    prediction_status="MISSING_CARRIED_FORWARD" if prediction is None else "PUBLISHED",
                )
            )

        model_performance = self._performance(
            initial_capital,
            model_values,
            model_trade_count,
            model_traded_notional,
            model_transaction_cost,
            model_slippage_cost,
            model_cash,
            model_units,
        )
        hold_performance = self._performance(
            initial_capital,
            hold_values,
            1,
            hold_notional,
            hold_transaction_cost,
            hold_slippage_cost,
            hold_cash,
            hold_units,
        )
        coverage_rate = (len(bars) - len(missing_dates)) / len(bars)
        return ModelValidationReport(
            requested_start_date=requested_start,
            requested_end_date=requested_end,
            effective_start_date=equity_curve[0].date,
            effective_end_date=equity_curve[-1].date,
            initial_capital=initial_capital,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
            model_strategy=model_performance,
            buy_and_hold=hold_performance,
            excess_final_equity=model_performance.final_equity - hold_performance.final_equity,
            excess_return=model_performance.cumulative_return - hold_performance.cumulative_return,
            drawdown_difference=model_performance.max_drawdown - hold_performance.max_drawdown,
            accuracy=self._accuracy(evaluations, coverage_rate),
            equity_curve=equity_curve,
            prediction_records=evaluations,
            missing_prediction_dates=missing_dates,
            market_data_sha256=market.metadata.content_sha256,
            model_manifest_sha256=context.manifest_sha256,
            prediction_log_head_sha256=context.prediction_log_head_sha256,
            model_version=context.manifest.model_version,
            training_data_cutoff_utc=context.manifest.training_data_cutoff_utc,
            model_activated_at_utc=context.manifest.activated_at_utc,
            out_of_sample_start_date=context.manifest.out_of_sample_start_date,
        )

    def status(self) -> ModelValidationStatus:
        source_status = self.local_settings.list_sources()
        databento = next(item for item in source_status.sources if item.source_id == "databento")
        manifest, manifest_sha256, manifest_reason = self._validated_manifest()
        reasons: list[str] = []
        if databento.credential_status != "CONFIGURED":
            reasons.append("DATABENTO_CREDENTIAL_REQUIRED")
        elif not source_status.market_data_ready:
            reasons.append("DATABENTO_MARKET_CACHE_UNAVAILABLE")
        if manifest_reason is not None:
            reasons.append(manifest_reason)

        records: list[FormalPredictionRecord] = []
        log_head_sha256: str | None = None
        if not self.prediction_log_path.is_file() or self.prediction_log_path.stat().st_size == 0:
            reasons.append("FORMAL_PREDICTION_LOG_EMPTY")
        else:
            try:
                records = self.predictions.read_all()
                if manifest is not None and manifest_sha256 is not None:
                    records = self._compatible_records(records, manifest, manifest_sha256)
                elif records:
                    records = []
                active = self._active_records(records)
                log_head_sha256 = records[-1].record_sha256 if records else None
                records = active
                if not records:
                    reasons.append("FORMAL_PREDICTION_LOG_EMPTY")
            except (OSError, ValueError, ValidationError):
                records = []
                reasons.append("FORMAL_PREDICTION_LOG_INVALID")

        realized_records = records
        if records and source_status.market_data_ready:
            try:
                market = self.market_data.get_history(refresh=False)
                market_dates = set(pd.to_datetime(market.frame["ts_event_utc"], utc=True).dt.date)
                latest_market_date = max(market_dates)
                if any(
                    item.prediction_for_date <= latest_market_date
                    and item.prediction_for_date not in market_dates
                    for item in records
                ):
                    reasons.append("FORMAL_MARKET_COVERAGE_INCOMPLETE")
                realized_records = [item for item in records if item.prediction_for_date in market_dates]
                if not realized_records:
                    reasons.append("FORMAL_PREDICTIONS_NOT_YET_REALIZED")
            except (OSError, ValueError, ValidationError, MarketDataUnavailableError):
                realized_records = []
                reasons.append("DATABENTO_MARKET_CACHE_UNAVAILABLE")

        reasons = list(dict.fromkeys(reasons))
        ready = not reasons
        return ModelValidationStatus(
            formal_evaluation_available=ready,
            status="READY" if ready else "NOT_READY",
            reason_codes=reasons,
            market_data_ready=source_status.market_data_ready,
            model_manifest_available=manifest is not None,
            prediction_record_count=len(records),
            training_data_cutoff_utc=None if manifest is None else manifest.training_data_cutoff_utc,
            model_activated_at_utc=None if manifest is None else manifest.activated_at_utc,
            model_version=None if manifest is None else manifest.model_version,
            feature_schema_version=None if manifest is None else manifest.feature_schema_version,
            out_of_sample_start_date=None if manifest is None else manifest.out_of_sample_start_date,
            model_manifest_sha256=manifest_sha256,
            prediction_log_head_sha256=log_head_sha256,
            prediction_from_date=min((item.prediction_for_date for item in realized_records), default=None),
            prediction_until_date=max((item.prediction_for_date for item in realized_records), default=None),
            service_mode="formal-prediction" if ready else "unavailable",
        )
