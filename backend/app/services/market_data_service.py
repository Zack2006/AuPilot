"""Databento-only UTC daily COMEX gold market-data service."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from threading import RLock
from typing import Callable, Protocol

import pandas as pd
from databento import BentoError

from backend.app.core.config import Settings
from backend.app.core.constants import MarketRegime
from backend.app.core.exceptions import MarketDataUnavailableError
from backend.app.repositories.databento_market_repository import DatabentoMarketRepository
from backend.app.schemas.market import MarketSeriesMetadata, MarketSnapshot
from backend.app.services.local_settings_service import LocalSettingsService


class DatabentoHistoricalClient(Protocol):
    metadata: object
    symbology: object
    timeseries: object


@dataclass
class MarketDataResult:
    frame: pd.DataFrame
    metadata: MarketSeriesMetadata


class MarketDataService:
    """Serve only verified Databento ohlcv-1d data; no Yahoo or sample fallback."""

    def __init__(
        self,
        settings: Settings,
        local_settings: LocalSettingsService,
        client_factory: Callable[[str], DatabentoHistoricalClient] | None = None,
    ) -> None:
        self.settings = settings
        self.local_settings = local_settings
        self.repository = DatabentoMarketRepository(settings.storage_dir / "market")
        self.client_factory = client_factory or self._official_client
        self._refresh_lock = RLock()

    @staticmethod
    def _official_client(key: str):
        import databento as db

        return db.Historical(key)

    def _credential(self) -> str:
        value = self.local_settings.credential("databento")
        if value is None:
            raise MarketDataUnavailableError("DATABENTO_CREDENTIAL_REQUIRED")
        return value.get_secret_value()

    def _metadata(self, manifest: dict) -> MarketSeriesMetadata:
        return MarketSeriesMetadata(
            dataset=manifest["dataset"],
            schema_name=manifest["schema"],
            symbol=manifest["symbol"],
            stype_in=manifest["stype_in"],
            retrieved_at_utc=manifest["retrieved_at_utc"],
            data_from_utc=manifest["data_from_utc"],
            data_until_utc=manifest["data_until_utc"],
            row_count=manifest["row_count"],
            content_sha256=manifest["content_sha256"],
            quality_status=manifest.get("quality_status", "VALID"),
        )

    def get_history(self, refresh: bool = False) -> MarketDataResult:
        try:
            if refresh:
                with self._refresh_lock:
                    frame, manifest = self._download_and_publish()
            else:
                frame, manifest = self.repository.read()
        except MarketDataUnavailableError:
            raise
        except (OSError, ValueError, FileNotFoundError) as exc:
            raise MarketDataUnavailableError(str(exc)) from exc
        self._validate_formal_contract(frame, manifest)
        latest = frame["ts_event_utc"].iloc[-1].to_pydatetime()
        age_days = (datetime.now(UTC).date() - latest.date()).days
        if age_days > self.settings.market_max_staleness_calendar_days:
            raise MarketDataUnavailableError("DATABENTO_DAILY_CACHE_STALE")
        return MarketDataResult(frame=frame, metadata=self._metadata(manifest))

    def cache_metadata(self) -> MarketSeriesMetadata:
        """Return the local cache identity without reading bars or contacting Databento."""
        try:
            manifest = self.repository.read_manifest()
            self._validate_source_identity(manifest)
            latest = pd.Timestamp(manifest["data_until_utc"])
        except (OSError, ValueError, FileNotFoundError) as exc:
            raise MarketDataUnavailableError(str(exc)) from exc
        age_days = (datetime.now(UTC).date() - latest.date()).days
        if age_days > self.settings.market_max_staleness_calendar_days:
            raise MarketDataUnavailableError("DATABENTO_DAILY_CACHE_STALE")
        return self._metadata(manifest)

    def ensure_initial_cache(self) -> bool:
        """Populate the cache only when a deployment has never published one."""
        if self.repository.manifest_path.is_file():
            return False
        self.get_history(refresh=True)
        return True

    def verify_access(self) -> dict:
        """Validate key, dataset, schema and symbol through metadata-only calls."""
        key = self._credential()
        try:
            client = self.client_factory(key)
            dataset = self.settings.databento_dataset
            schema = self.settings.databento_schema
            symbol = self.settings.databento_symbol
            stype_in = self.settings.databento_stype_in
            schemas = client.metadata.list_schemas(dataset=dataset)
            if schema not in schemas:
                raise MarketDataUnavailableError("DATABENTO_OHLCV_1D_SCHEMA_UNAVAILABLE")
            dataset_range = client.metadata.get_dataset_range(dataset=dataset)
            available_from, available_until = self._schema_range(dataset_range, schema)
            resolution_end = available_until.date()
            resolution_start = max(available_from.date(), resolution_end - timedelta(days=7))
            if resolution_start >= resolution_end:
                raise MarketDataUnavailableError("DATABENTO_ACCESSIBLE_RANGE_TOO_SHORT")
            resolution = client.symbology.resolve(
                dataset=dataset,
                symbols=symbol,
                stype_in=stype_in,
                stype_out="instrument_id",
                start_date=resolution_start,
                end_date=resolution_end,
            )
            if not self._symbol_resolved(resolution, symbol):
                raise MarketDataUnavailableError("DATABENTO_SYMBOL_RESOLUTION_FAILED")
        except MarketDataUnavailableError:
            raise
        except BentoError as exc:
            raise MarketDataUnavailableError("DATABENTO_ACCESS_VERIFICATION_FAILED") from exc
        return {
            "dataset": dataset,
            "schema_name": schema,
            "symbol": symbol,
            "stype_in": stype_in,
            "symbol_resolved": True,
            "dataset_available_from_utc": available_from.isoformat(),
            "dataset_available_until_utc": available_until.isoformat(),
        }

    def latest(self, refresh: bool = False) -> MarketSnapshot:
        result = self.get_history(refresh=refresh)
        row = result.frame.iloc[-1]
        close = float(row["close"])
        ma20 = float(result.frame["close"].tail(20).mean())
        ma60 = float(result.frame["close"].tail(60).mean())
        if close > ma20 > ma60:
            regime = MarketRegime.UPTREND
        elif close < ma20 < ma60:
            regime = MarketRegime.DOWNTREND
        else:
            regime = MarketRegime.RANGE_BOUND
        metadata = result.metadata
        return MarketSnapshot(
            ts_event_utc=row["ts_event_utc"], open=row["open"], high=row["high"], low=row["low"],
            close=row["close"], volume=row["volume"], market_regime=regime,
            dataset=metadata.dataset, schema_name=metadata.schema_name, symbol=metadata.symbol,
            retrieved_at_utc=metadata.retrieved_at_utc,
            content_sha256=metadata.content_sha256,
            quality_status=metadata.quality_status,
        )

    def _download_and_publish(self) -> tuple[pd.DataFrame, dict]:
        key = self._credential()
        client = self.client_factory(key)
        dataset = self.settings.databento_dataset
        schema = self.settings.databento_schema
        symbol = self.settings.databento_symbol
        stype_in = self.settings.databento_stype_in
        try:
            available_schemas = client.metadata.list_schemas(dataset=dataset)
            if schema not in available_schemas:
                raise MarketDataUnavailableError("DATABENTO_OHLCV_1D_SCHEMA_UNAVAILABLE")
            dataset_range = client.metadata.get_dataset_range(dataset=dataset)
            available_from, available_until = self._schema_range(dataset_range, schema)
            now = datetime.now(UTC)
            end = min(available_until.date(), now.date())
            existing: pd.DataFrame | None = None
            try:
                existing, existing_manifest = self.repository.read()
                self._validate_source_identity(existing_manifest)
            except FileNotFoundError:
                existing = None
            required_start = self.settings.technical_history_start
            if required_start < available_from.date():
                raise MarketDataUnavailableError("DATABENTO_REQUIRED_HISTORY_UNAVAILABLE")
            start = required_start
            if existing is not None:
                existing_last = existing["ts_event_utc"].iloc[-1].date()
                start = max(required_start, existing_last - timedelta(days=self.settings.market_sync_overlap_days))
            if start >= end:
                raise MarketDataUnavailableError("DATABENTO_ACCESSIBLE_RANGE_TOO_SHORT")
            resolution_start = max(start, end - timedelta(days=7))
            resolution = client.symbology.resolve(
                dataset=dataset,
                symbols=symbol,
                stype_in=stype_in,
                stype_out="instrument_id",
                start_date=resolution_start,
                end_date=end,
            )
            if not self._symbol_resolved(resolution, symbol):
                raise MarketDataUnavailableError("DATABENTO_SYMBOL_RESOLUTION_FAILED")
            condition_end = end - timedelta(days=1)
            dataset_conditions = client.metadata.get_dataset_condition(
                dataset=dataset,
                start_date=required_start,
                end_date=condition_end,
            )
            condition_audit = self._provider_condition_audit(
                dataset_conditions,
                start_date=required_start,
                end_date=condition_end,
            )
            chunks: list[pd.DataFrame] = []
            chunk_start = start
            chunk_count = 0
            while chunk_start < end:
                chunk_end = min(
                    chunk_start + timedelta(days=self.settings.market_sync_chunk_days),
                    end,
                )
                store = client.timeseries.get_range(
                    dataset=dataset,
                    start=chunk_start,
                    end=chunk_end,
                    symbols=symbol,
                    schema=schema,
                    stype_in=stype_in,
                    stype_out="instrument_id",
                )
                chunk = store.to_df(
                    price_type="float",
                    pretty_ts=True,
                    map_symbols=True,
                    tz="UTC",
                ).reset_index()
                if not chunk.empty:
                    timestamp_column = "ts_event" if "ts_event" in chunk.columns else chunk.columns[0]
                    chunks.append(chunk.rename(columns={timestamp_column: "ts_event_utc"}))
                chunk_count += 1
                chunk_start = chunk_end
            if not chunks:
                raise MarketDataUnavailableError("DATABENTO_NO_DAILY_ROWS_RETURNED")
            frame = pd.concat(chunks, ignore_index=True)
        except MarketDataUnavailableError:
            raise
        except BentoError as exc:
            raise MarketDataUnavailableError("DATABENTO_PROVIDER_REQUEST_FAILED") from exc
        frame["ts_event_utc"] = pd.to_datetime(frame["ts_event_utc"], utc=True, errors="raise")
        frame = frame.sort_values("ts_event_utc").drop_duplicates("ts_event_utc", keep="last")
        normalized = self.repository.validate(frame, minimum_rows=1)
        bucket_end = normalized["ts_event_utc"] + pd.Timedelta(1, unit="D")
        normalized = normalized.loc[bucket_end <= pd.Timestamp(now)].copy()
        if normalized.empty:
            raise MarketDataUnavailableError("DATABENTO_NO_COMPLETE_UTC_DAILY_BUCKETS")
        if existing is not None:
            normalized = pd.concat([existing, normalized], ignore_index=True)
            normalized = normalized.sort_values("ts_event_utc").drop_duplicates(
                "ts_event_utc", keep="last"
            )
        normalized = self.repository.validate(normalized)
        if normalized["ts_event_utc"].iloc[0].date() != required_start:
            raise MarketDataUnavailableError("DATABENTO_FULL_HISTORY_START_MISMATCH")
        manifest = self.repository.write(
            normalized,
            {
                "source_id": "databento",
                "dataset": dataset,
                "schema": schema,
                "symbol": symbol,
                "stype_in": stype_in,
                "daily_unit_id": "CANONICAL_GC_UTC_DAILY_BUCKET_V1",
                "retrieved_at_utc": now.isoformat(),
                "sync_mode": "FULL" if existing is None else "INCREMENTAL",
                "request_start_date": start.isoformat(),
                "request_end_date_exclusive": end.isoformat(),
                "request_chunk_count": chunk_count,
                **condition_audit,
            },
        )
        return normalized, manifest

    def _validate_source_identity(self, manifest: dict) -> None:
        expected = {
            "source_id": "databento",
            "dataset": self.settings.databento_dataset,
            "schema": self.settings.databento_schema,
            "symbol": self.settings.databento_symbol,
            "stype_in": self.settings.databento_stype_in,
        }
        if any(manifest.get(key) != value for key, value in expected.items()):
            raise MarketDataUnavailableError("DATABENTO_CACHE_IDENTITY_MISMATCH")

    def _validate_formal_contract(self, frame: pd.DataFrame, manifest: dict) -> None:
        self._validate_source_identity(manifest)
        if manifest.get("daily_unit_id") != "CANONICAL_GC_UTC_DAILY_BUCKET_V1":
            raise MarketDataUnavailableError("DATABENTO_DAILY_UNIT_ID_MISMATCH")
        if frame["ts_event_utc"].iloc[0].date() != self.settings.technical_history_start:
            raise MarketDataUnavailableError("DATABENTO_FULL_HISTORY_START_MISMATCH")
        latest = frame["ts_event_utc"].iloc[-1]
        if latest + pd.Timedelta(1, unit="D") > pd.Timestamp.now(tz="UTC"):
            raise MarketDataUnavailableError("DATABENTO_INCOMPLETE_UTC_BUCKET_PRESENT")

    @staticmethod
    def _symbol_resolved(resolution: object, symbol: str) -> bool:
        if not isinstance(resolution, dict):
            return False
        result = resolution.get("result")
        return isinstance(result, dict) and bool(result.get(symbol))

    @staticmethod
    def _provider_condition_audit(
        conditions: object,
        *,
        start_date: date,
        end_date: date,
    ) -> dict:
        if not isinstance(conditions, list) or not conditions:
            raise MarketDataUnavailableError("DATABENTO_DATASET_CONDITION_INVALID")

        counts: Counter[str] = Counter()
        exceptions: list[dict[str, str]] = []
        seen_dates: set[date] = set()
        for item in conditions:
            if not isinstance(item, dict):
                raise MarketDataUnavailableError("DATABENTO_DATASET_CONDITION_INVALID")
            raw_date = item.get("date")
            raw_condition = item.get("condition")
            if not isinstance(raw_date, str) or not isinstance(raw_condition, str):
                raise MarketDataUnavailableError("DATABENTO_DATASET_CONDITION_INVALID")
            try:
                condition_date = date.fromisoformat(raw_date)
            except ValueError as exc:
                raise MarketDataUnavailableError("DATABENTO_DATASET_CONDITION_INVALID") from exc
            condition = raw_condition.strip().lower()
            if (
                not condition
                or condition_date < start_date
                or condition_date > end_date
                or condition_date in seen_dates
            ):
                raise MarketDataUnavailableError("DATABENTO_DATASET_CONDITION_INVALID")
            seen_dates.add(condition_date)
            counts[condition] += 1
            if condition != "available":
                exceptions.append({"date": condition_date.isoformat(), "condition": condition})

        return {
            "provider_condition_range": {
                "start_date": start_date.isoformat(),
                "end_date_inclusive": end_date.isoformat(),
            },
            "provider_condition_counts": dict(sorted(counts.items())),
            "provider_condition_exceptions": sorted(exceptions, key=lambda item: item["date"]),
            "quality_status": (
                "VALID_WITH_PROVIDER_CONDITION_WARNINGS" if exceptions else "VALID"
            ),
        }

    @staticmethod
    def _schema_range(dataset_range: object, schema: str) -> tuple[pd.Timestamp, pd.Timestamp]:
        if not isinstance(dataset_range, dict):
            raise MarketDataUnavailableError("DATABENTO_DATASET_RANGE_INVALID")
        schema_ranges = dataset_range.get("schema")
        selected = schema_ranges.get(schema) if isinstance(schema_ranges, dict) else None
        source = selected if isinstance(selected, dict) else dataset_range
        try:
            available_from = pd.Timestamp(source["start"])
            available_until = pd.Timestamp(source["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MarketDataUnavailableError("DATABENTO_DATASET_RANGE_INVALID") from exc
        if available_from.tzinfo is None or available_until.tzinfo is None:
            raise MarketDataUnavailableError("DATABENTO_DATASET_RANGE_INVALID")
        available_from = available_from.tz_convert(UTC)
        available_until = available_until.tz_convert(UTC)
        if available_from >= available_until:
            raise MarketDataUnavailableError("DATABENTO_DATASET_RANGE_INVALID")
        return available_from, available_until
