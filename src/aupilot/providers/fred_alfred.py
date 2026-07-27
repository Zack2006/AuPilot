from __future__ import annotations

import json
import sqlite3
import time as time_module
from collections.abc import Iterable, Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Protocol

import httpx

from aupilot.core.config import resolve_project_path
from aupilot.core.hashing import canonical_json_sha256, sha256_file
from aupilot.core.manifest import write_json_atomic
from aupilot.macro_gate.evidence_store import MacroEvidenceStore
from aupilot.macro_gate.schemas import MacroObservation

FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
FRED_INITIAL_RELEASE_OUTPUT_TYPE = 4
MAX_OBSERVATIONS_PER_REQUEST = 100_000
FRED_OBSERVATION_WINDOW_YEARS = 5
FRED_REALTIME_WINDOW_YEARS = 2
FRED_RETRY_ATTEMPTS = 3
FRED_RETRY_DELAY_SECONDS = 1


class HTTPResponse(Protocol):
    def raise_for_status(self) -> None: ...

    def json(self) -> Any: ...


class HTTPClient(Protocol):
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object],
    ) -> HTTPResponse: ...


@dataclass(frozen=True)
class FredSeriesSnapshot:
    series_id: str
    payload: dict[str, Any]
    payload_sha256: str
    observations: tuple[MacroObservation, ...]
    missing_value_rows: int

    def to_report(self) -> dict[str, object]:
        return {
            "series_id": self.series_id,
            "payload_sha256": self.payload_sha256,
            "observation_rows": len(self.observations),
            "missing_value_rows": self.missing_value_rows,
            "initial_release_only": True,
        }


def _parse_iso_date(value: object, *, name: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"Invalid FRED {name}: {value}") from error


def _validate_bounds(start: date, end: date, *, name: str) -> None:
    if start > end:
        raise ValueError(f"FRED {name} start must not exceed end")


def _add_years(value: date, years: int) -> date:
    """Add calendar years while keeping February 29 ranges valid."""

    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(year=value.year + years, day=28)


def _date_windows(
    start: date,
    end: date,
    *,
    window_years: int = FRED_OBSERVATION_WINDOW_YEARS,
) -> tuple[tuple[date, date], ...]:
    """Split large FRED date ranges into API-friendly calendar windows."""

    if window_years <= 0:
        raise ValueError("FRED window years must be positive")
    windows: list[tuple[date, date]] = []
    window_start = start
    while window_start <= end:
        window_end = min(
            end,
            _add_years(window_start, window_years) - timedelta(days=1),
        )
        windows.append((window_start, window_end))
        window_start = window_end + timedelta(days=1)
    return tuple(windows)


def normalize_fred_initial_release_payload(
    payload: Mapping[str, Any],
    *,
    series_id: str,
    retrieved_at_utc: datetime,
    allow_empty: bool = False,
) -> FredSeriesSnapshot:
    """Normalize ALFRED initial releases with a conservative availability timestamp.

    FRED exposes real-time *dates* for this endpoint, not guaranteed intraday release
    timestamps. AuPilot therefore makes an observation replay-eligible at 00:00 UTC on the
    next calendar day. Exact release schedules may later tighten this delay, but never move a
    value earlier without separately hashed evidence.
    """

    if retrieved_at_utc.tzinfo is None or retrieved_at_utc.utcoffset() is None:
        raise ValueError("retrieved_at_utc must be timezone-aware")
    normalized_series = series_id.strip().upper()
    output_type = int(payload.get("output_type", -1))
    if output_type != FRED_INITIAL_RELEASE_OUTPUT_TYPE:
        raise ValueError("FRED payload is not initial-release-only output_type=4")
    rows = payload.get("observations")
    if not isinstance(rows, list):
        raise ValueError("FRED payload observations must be a list")
    count = int(payload.get("count", len(rows)))
    offset = int(payload.get("offset", 0))
    limit = int(payload.get("limit", MAX_OBSERVATIONS_PER_REQUEST))
    if offset != 0 or count > limit or len(rows) < count:
        raise RuntimeError("FRED response was truncated or paginated")
    payload_body = json.loads(json.dumps(payload, ensure_ascii=False))
    payload_sha = canonical_json_sha256(payload_body)
    observations: list[MacroObservation] = []
    missing = 0
    seen: set[tuple[date, date]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("FRED observation row must be a mapping")
        value_text = str(row.get("value", "")).strip()
        if value_text in {"", "."}:
            missing += 1
            continue
        observation_date = _parse_iso_date(row.get("date"), name="observation date")
        realtime_start = _parse_iso_date(row.get("realtime_start"), name="realtime_start")
        realtime_end = _parse_iso_date(row.get("realtime_end"), name="realtime_end")
        _validate_bounds(realtime_start, realtime_end, name="real-time period")
        key = (observation_date, realtime_start)
        if key in seen:
            raise ValueError(f"Duplicate FRED initial release row: {key}")
        seen.add(key)
        try:
            value = float(value_text)
        except ValueError as error:
            raise ValueError(f"Invalid FRED numeric value: {value_text}") from error
        eligible = datetime.combine(realtime_start + timedelta(days=1), time.min, tzinfo=UTC)
        identity = {
            "series_id": normalized_series,
            "observation_date": observation_date.isoformat(),
            "value": value,
            "realtime_start": realtime_start.isoformat(),
            "realtime_end": realtime_end.isoformat(),
            "eligibility_rule": "NEXT_UTC_DAY_AFTER_REALTIME_START",
        }
        observations.append(
            MacroObservation(
                observation_id=(
                    f"fred:{normalized_series}:{observation_date.isoformat()}:"
                    f"{realtime_start.isoformat()}:{canonical_json_sha256(identity)[:20]}"
                ),
                series_id=normalized_series,
                observation_date=observation_date,
                value=value,
                realtime_start=realtime_start,
                realtime_end=realtime_end,
                eligible_from_utc=eligible,
                retrieved_at_utc=retrieved_at_utc.astimezone(UTC),
                source_url=FRED_OBSERVATIONS_URL,
                source_payload_sha256=payload_sha,
                initial_release_only=True,
            )
        )
    if not observations and not allow_empty:
        raise RuntimeError(f"FRED returned no numeric initial releases for {normalized_series}")
    return FredSeriesSnapshot(
        series_id=normalized_series,
        payload=payload_body,
        payload_sha256=payload_sha,
        observations=tuple(observations),
        missing_value_rows=missing,
    )


def _fetch_with_client(
    client: HTTPClient,
    *,
    api_key: str,
    series_id: str,
    observation_start: date,
    observation_end: date,
    realtime_start: date,
    realtime_end: date,
    retrieved_at_utc: datetime,
    allow_empty: bool = False,
) -> FredSeriesSnapshot:
    response = client.get(
        FRED_OBSERVATIONS_URL,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": observation_start.isoformat(),
            "observation_end": observation_end.isoformat(),
            "realtime_start": realtime_start.isoformat(),
            "realtime_end": realtime_end.isoformat(),
            "output_type": FRED_INITIAL_RELEASE_OUTPUT_TYPE,
            "sort_order": "asc",
            "limit": MAX_OBSERVATIONS_PER_REQUEST,
            "offset": 0,
        },
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("FRED JSON response must be a mapping")
    return normalize_fred_initial_release_payload(
        payload,
        series_id=series_id,
        retrieved_at_utc=retrieved_at_utc,
        allow_empty=allow_empty,
    )


def _is_vintage_date_limit_error(error: httpx.HTTPStatusError) -> bool:
    response = getattr(error, "response", None)
    return (
        getattr(response, "status_code", None) == 400
        and "vintage dates" in str(getattr(response, "text", "")).lower()
        and "2000" in str(getattr(response, "text", ""))
    )


def _is_retryable_http_error(error: httpx.HTTPStatusError) -> bool:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code == 429 or (status_code is not None and 500 <= status_code <= 599)


def _fetch_with_retries(client: HTTPClient, **kwargs: Any) -> FredSeriesSnapshot:
    for attempt in range(FRED_RETRY_ATTEMPTS):
        try:
            return _fetch_with_client(client, **kwargs)
        except httpx.TimeoutException as error:
            if attempt == FRED_RETRY_ATTEMPTS - 1:
                context = ", ".join(
                    f"{name}={kwargs[name]}"
                    for name in (
                        "series_id",
                        "observation_start",
                        "observation_end",
                        "realtime_start",
                        "realtime_end",
                    )
                    if name in kwargs
                )
                request = getattr(error, "_request", None)
                raise httpx.ReadTimeout(
                    f"FRED request timed out after retries ({context})",
                    request=request,
                ) from error
            time_module.sleep(FRED_RETRY_DELAY_SECONDS * (attempt + 1))
        except httpx.HTTPStatusError as error:
            if not _is_retryable_http_error(error) or attempt == FRED_RETRY_ATTEMPTS - 1:
                raise
            time_module.sleep(FRED_RETRY_DELAY_SECONDS * (attempt + 1))
    raise AssertionError("unreachable retry loop")


def _merge_series_snapshots(
    snapshots: Iterable[FredSeriesSnapshot],
    *,
    series_id: str,
    observation_start: date,
    observation_end: date,
    realtime_start: date,
    realtime_end: date,
    request_windows: Iterable[Mapping[str, str]],
) -> FredSeriesSnapshot:
    """Merge non-overlapping range responses into one auditable series payload."""

    parts = tuple(snapshots)
    if not parts:
        raise ValueError("Cannot merge an empty FRED snapshot sequence")
    raw_rows: dict[tuple[str, str], dict[str, Any]] = {}
    observations: dict[tuple[date, date], MacroObservation] = {}
    for snapshot in parts:
        rows = snapshot.payload.get("observations")
        if not isinstance(rows, list):
            raise ValueError("FRED snapshot observations must be a list")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("FRED observation row must be a mapping")
            key = (str(row.get("date", "")), str(row.get("realtime_start", "")))
            raw_rows[key] = row
        for observation in snapshot.observations:
            observations[(observation.observation_date, observation.realtime_start)] = observation

    merged_rows = [raw_rows[key] for key in sorted(raw_rows)]
    payload = dict(parts[0].payload)
    payload.update(
        {
            "observation_start": observation_start.isoformat(),
            "observation_end": observation_end.isoformat(),
            "realtime_start": realtime_start.isoformat(),
            "realtime_end": realtime_end.isoformat(),
            "output_type": FRED_INITIAL_RELEASE_OUTPUT_TYPE,
            "count": len(merged_rows),
            "offset": 0,
            "limit": MAX_OBSERVATIONS_PER_REQUEST,
            "observations": merged_rows,
            "request_windows": list(request_windows),
        }
    )
    missing = sum(
        1
        for row in merged_rows
        if str(row.get("value", "")).strip() in {"", "."}
    )
    return FredSeriesSnapshot(
        series_id=series_id.strip().upper(),
        payload=payload,
        payload_sha256=canonical_json_sha256(payload),
        observations=tuple(
            observations[key]
            for key in sorted(observations, key=lambda value: (value[0], value[1]))
        ),
        missing_value_rows=missing,
    )


def fetch_fred_initial_releases(
    *,
    api_key: str,
    series_ids: Iterable[str],
    allowed_series_ids: Iterable[str],
    observation_start: date,
    observation_end: date,
    realtime_start: date,
    realtime_end: date,
    client: HTTPClient | None = None,
    retrieved_at_utc: datetime | None = None,
    paired_realtime_windows: bool = False,
) -> tuple[FredSeriesSnapshot, ...]:
    """Fetch only allowlisted initial releases; the API key is never returned or persisted."""

    if not api_key.strip():
        raise ValueError("FRED_API_KEY is missing")
    requested = tuple(dict.fromkeys(value.strip().upper() for value in series_ids))
    allowed = {value.strip().upper() for value in allowed_series_ids}
    if not requested or any(not value for value in requested):
        raise ValueError("At least one non-empty FRED series id is required")
    forbidden = sorted(set(requested) - allowed)
    if forbidden:
        raise PermissionError(f"FRED series are outside the AuPilot whitelist: {forbidden}")
    _validate_bounds(observation_start, observation_end, name="observation period")
    _validate_bounds(realtime_start, realtime_end, name="real-time period")
    retrieved = datetime.now(UTC) if retrieved_at_utc is None else retrieved_at_utc
    owns_client = client is None
    active_client: HTTPClient = (
        httpx.Client(
            timeout=30.0,
            follow_redirects=False,
            # FRED may close an idle keep-alive connection between sequential
            # series requests; a fresh connection avoids stale-pool timeouts.
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        )
        if client is None
        else client
    )
    try:
        snapshots: list[FredSeriesSnapshot] = []
        for series_id in requested:
            parts: list[FredSeriesSnapshot] = []
            request_windows: list[dict[str, str]] = []
            for window_start, window_end in _date_windows(observation_start, observation_end):
                if paired_realtime_windows:
                    paired_start = max(window_start, realtime_start)
                    paired_end = min(window_end, realtime_end)
                    if paired_start > paired_end:
                        continue
                    parts.append(
                        _fetch_with_retries(
                            active_client,
                            api_key=api_key,
                            series_id=series_id,
                            observation_start=window_start,
                            observation_end=window_end,
                            realtime_start=paired_start,
                            realtime_end=paired_end,
                            retrieved_at_utc=retrieved,
                            allow_empty=True,
                        )
                    )
                    request_windows.append(
                        {
                            "observation_start": window_start.isoformat(),
                            "observation_end": window_end.isoformat(),
                            "realtime_start": paired_start.isoformat(),
                            "realtime_end": paired_end.isoformat(),
                        }
                    )
                    continue
                try:
                    parts.append(
                        _fetch_with_retries(
                            active_client,
                            api_key=api_key,
                            series_id=series_id,
                            observation_start=window_start,
                            observation_end=window_end,
                            realtime_start=realtime_start,
                            realtime_end=realtime_end,
                            retrieved_at_utc=retrieved,
                            allow_empty=True,
                        )
                    )
                    request_windows.append(
                        {
                            "observation_start": window_start.isoformat(),
                            "observation_end": window_end.isoformat(),
                            "realtime_start": realtime_start.isoformat(),
                            "realtime_end": realtime_end.isoformat(),
                        }
                    )
                except (httpx.HTTPStatusError, httpx.TimeoutException) as error:
                    if isinstance(error, httpx.HTTPStatusError) and not _is_vintage_date_limit_error(error):
                        raise
                    for realtime_window_start, realtime_window_end in _date_windows(
                        realtime_start,
                        realtime_end,
                        window_years=FRED_REALTIME_WINDOW_YEARS,
                    ):
                        parts.append(
                            _fetch_with_retries(
                                active_client,
                                api_key=api_key,
                                series_id=series_id,
                                observation_start=window_start,
                                observation_end=window_end,
                                realtime_start=realtime_window_start,
                                realtime_end=realtime_window_end,
                                retrieved_at_utc=retrieved,
                                allow_empty=True,
                            )
                        )
                        request_windows.append(
                            {
                                "observation_start": window_start.isoformat(),
                                "observation_end": window_end.isoformat(),
                                "realtime_start": realtime_window_start.isoformat(),
                                "realtime_end": realtime_window_end.isoformat(),
                            }
                        )
            merged = _merge_series_snapshots(
                parts,
                series_id=series_id,
                observation_start=observation_start,
                observation_end=observation_end,
                realtime_start=realtime_start,
                realtime_end=realtime_end,
                request_windows=request_windows,
            )
            if not merged.observations:
                raise RuntimeError(f"FRED returned no numeric initial releases for {series_id}")
            snapshots.append(merged)
        return tuple(snapshots)
    finally:
        if owns_client:
            close = getattr(active_client, "close", None)
            if close is not None:
                close()


def persist_fred_initial_release_refresh(
    snapshots: tuple[FredSeriesSnapshot, ...],
    *,
    evidence_store: MacroEvidenceStore,
    destination_root: str | Path,
    root: Path,
    run_id: str | None = None,
    seed_database_path: str | Path | None = None,
) -> dict[str, Any]:
    """Persist redacted raw payloads, normalized rows, and one immutable refresh manifest."""

    if not snapshots:
        raise ValueError("Cannot persist an empty FRED refresh")
    root = root.resolve()
    identifier = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = resolve_project_path(Path(destination_root) / identifier, root=root)
    if run_root.exists():
        raise FileExistsError(f"Refusing to overwrite a FRED refresh: {run_root}")
    run_root.mkdir(parents=True)
    refresh_database = run_root / "evidence.sqlite"
    refresh_store = evidence_store
    if seed_database_path is not None:
        seed_path = resolve_project_path(seed_database_path, root=root)
        if not seed_path.is_file():
            raise FileNotFoundError(f"FRED refresh seed database is missing: {seed_path}")
        source_uri = f"file:{seed_path.as_posix()}?mode=ro"
        with closing(sqlite3.connect(source_uri, uri=True)) as source_connection:
            with closing(sqlite3.connect(refresh_database)) as destination_connection:
                source_connection.backup(destination_connection)
                destination_connection.commit()
        refresh_store = MacroEvidenceStore(refresh_database)
    artifacts = []
    inserted = 0
    series_reports = []
    for snapshot in snapshots:
        raw_path = run_root / f"{snapshot.series_id}_initial_release.json"
        write_json_atomic(raw_path, snapshot.payload)
        artifacts.append(
            {
                "series_id": snapshot.series_id,
                "path": raw_path.resolve().relative_to(root).as_posix(),
                "bytes": raw_path.stat().st_size,
                "sha256": sha256_file(raw_path),
                "canonical_payload_sha256": snapshot.payload_sha256,
            }
        )
        inserted += refresh_store.ingest_observations(snapshot.observations)
        series_reports.append(snapshot.to_report())
    if refresh_store.path == refresh_database:
        with sqlite3.connect(refresh_database) as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()
            if mode is None or str(mode[0]).lower() != "delete":
                raise RuntimeError("Failed to seal FRED refresh SQLite database")
    manifest = {
        "schema_version": "aupilot.fred_refresh_manifest.v1",
        "project": "AuPilot",
        "operation": "FRED_ALFRED_INITIAL_RELEASE_REFRESH",
        "run_id": identifier,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "fetch_succeeded": True,
        "published": True,
        "source": FRED_OBSERVATIONS_URL,
        "series": series_reports,
        "artifacts": artifacts,
        "observation_rows": sum(len(value.observations) for value in snapshots),
        "inserted_rows": inserted,
        "store_observation_count": refresh_store.observation_count(),
        "point_in_time_rule": "NEXT_UTC_DAY_AFTER_REALTIME_START",
        "api_output_type": FRED_INITIAL_RELEASE_OUTPUT_TYPE,
        "historical_profit_tuning_allowed": False,
        "macro_model_selection_allowed": False,
        "rag_is_independent": True,
        "secret_recorded": False,
    }
    if refresh_database.is_file():
        manifest["database"] = {
            "path": refresh_database.resolve().relative_to(root).as_posix(),
            "bytes": refresh_database.stat().st_size,
            "sha256": sha256_file(refresh_database),
        }
    manifest_path = run_root / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return manifest
