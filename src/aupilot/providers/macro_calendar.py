"""Official macro event-calendar refresh with strict failure semantics."""

from __future__ import annotations

import hashlib
import html
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta, time
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx

from aupilot.core.hashing import sha256_file
from aupilot.core.manifest import write_json_atomic
from aupilot.macro_gate.schemas import ALLOWED_EVENT_TYPES, OFFICIAL_DOMAINS

from .macro_official import extract_official_text

PARSER_VERSION = "official-calendar-v2"
EASTERN = ZoneInfo("America/New_York")
_MONTHS = {
    value.lower(): index
    for index, value in enumerate(
        ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"),
        start=1,
    )
}
_MONTHS.update({key[:3]: value for key, value in tuple(_MONTHS.items()) if len(key) > 3})
_MONTHS["sept"] = 9
_DATE_RE = re.compile(
    r"\b(?P<month>January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"\.?"
    r"\s+(?P<day>\d{1,2})(?:\s*[-\u2013\u2014]\s*(?P<end_day>\d{1,2}))?"
    r"(?:,|\s)+(?P<year>20\d{2})\b",
    re.IGNORECASE,
)
_NUMERIC_DATE_RE = re.compile(r"\b(?P<month>\d{1,2})/(?P<day>\d{1,2})/(?P<year>20\d{2})\b")
_NO_YEAR_DATE_RE = re.compile(
    r"\b(?P<month>January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"\.?"
    r"\s+(?P<day>\d{1,2})(?:\s*[-\u2013\u2014]\s*(?P<end_day>\d{1,2}))?\b",
    re.IGNORECASE,
)
_YEAR_HEADER_RE = re.compile(
    r"(?:\bYear\s+(?P<year_label>20\d{2})\b|\b(?P<year_release>20\d{2})(?=\s+Release\b)|"
    r"\bENTIRE\s+YEAR,\s*(?P<year_entire>20\d{2})\b)",
    re.IGNORECASE,
)
_FOMC_SECTION_RE = re.compile(r"\b(?P<year>20\d{2})\s+FOMC Meetings(?P<body>.*?)(?=\b20\d{2}\s+FOMC Meetings|\Z)", re.IGNORECASE | re.DOTALL)
_BLS_MONTH_HEADER_RE = re.compile(
    r"Schedule\s+of\s+Selected\s+Releases\s+for\s+"
    r"(?P<month>January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"(?P<year>20\d{2})",
    re.IGNORECASE,
)
_BLS_DAY_CELL_RE = re.compile(
    r"<td\b[^>]*\bid=[\"']d(?P<month>\d{2})(?P<day>\d{2})[\"'][^>]*>"
    r"(?P<body>.*?)</td>",
    re.IGNORECASE | re.DOTALL,
)


class CalendarHTTPResponse(Protocol):
    url: Any
    content: bytes
    text: str

    def raise_for_status(self) -> None: ...


class CalendarHTTPClient(Protocol):
    def get(self, url: str) -> CalendarHTTPResponse: ...


class CalendarFormatError(ValueError):
    """Raised when an official page cannot produce a trustworthy event list."""


def _failure_reason_code(error: Exception) -> str:
    if isinstance(error, (httpx.TimeoutException, TimeoutError)):
        return "UPSTREAM_TIMEOUT"
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        match = re.search(r"\bHTTP[ _:=]*(\d{3})\b", str(error), flags=re.IGNORECASE)
        status_code = None if match is None else int(match.group(1))
    if status_code is not None and 400 <= int(status_code) <= 599:
        return f"HTTP_{int(status_code)}"
    if isinstance(error, CalendarFormatError):
        return "CALENDAR_FORMAT_ERROR"
    return "UPSTREAM_REQUEST_FAILED"


def _failure_report(
    *,
    source: str,
    event_type: str,
    requested_url: str,
    error: Exception,
) -> dict[str, str]:
    return {
        "source": source,
        "event_type": event_type,
        "requested_url": requested_url,
        "error_type": type(error).__name__,
        "reason_code": _failure_reason_code(error),
    }


def _validate_source(source: Mapping[str, object]) -> tuple[str, str, str]:
    source_name = str(source.get("source", "")).strip()
    event_type = str(source.get("event_type", "")).strip().upper()
    url = str(source.get("url", "")).strip()
    parsed = urlparse(url)
    if event_type not in ALLOWED_EVENT_TYPES:
        raise CalendarFormatError(f"unsupported calendar event type: {event_type}")
    if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_DOMAINS:
        raise CalendarFormatError("calendar source must be an approved official HTTPS URL")
    if not source_name:
        raise CalendarFormatError("calendar source name is required")
    return source_name, event_type, url


def _parse_candidates(content: str, *, keywords: tuple[str, ...] = ()) -> list[tuple[datetime, str]]:
    candidates: list[tuple[datetime, str]] = []
    for match in _DATE_RE.finditer(content):
        snippet = content[max(0, match.start() - 180) : match.end() + 180].lower()
        if keywords and not any(keyword.lower() in snippet for keyword in keywords):
            continue
        month = _MONTHS.get(match.group("month").lower().rstrip("."))
        if month is None:
            continue
        day = int(match.group("end_day") or match.group("day"))
        try:
            candidates.append((datetime(int(match.group("year")), month, day), match.group(0)))
        except ValueError as error:
            raise CalendarFormatError(f"invalid official calendar date: {match.group(0)}") from error
    for match in _NUMERIC_DATE_RE.finditer(content):
        snippet = content[max(0, match.start() - 180) : match.end() + 180].lower()
        if keywords and not any(keyword.lower() in snippet for keyword in keywords):
            continue
        try:
            candidates.append((datetime(int(match.group("year")), int(match.group("month")), int(match.group("day"))), match.group(0)))
        except ValueError as error:
            raise CalendarFormatError(f"invalid official calendar date: {match.group(0)}") from error
    for header in _YEAR_HEADER_RE.finditer(content):
        year = int(header.group("year_label") or header.group("year_release") or header.group("year_entire"))
        body_end = _YEAR_HEADER_RE.search(content, header.end())
        body = content[header.end() : body_end.start() if body_end else len(content)]
        for match in _NO_YEAR_DATE_RE.finditer(body):
            snippet = body[max(0, match.start() - 180) : match.end() + 180].lower()
            if keywords and not any(keyword.lower() in snippet for keyword in keywords):
                continue
            month = _MONTHS.get(match.group("month").lower().rstrip("."))
            if month is None:
                continue
            day = int(match.group("end_day") or match.group("day"))
            try:
                candidates.append((datetime(year, month, day), match.group(0)))
            except ValueError as error:
                raise CalendarFormatError(f"invalid official calendar date: {match.group(0)}") from error
    return candidates


def _parse_fomc_candidates(content: str) -> list[tuple[datetime, str]]:
    candidates: list[tuple[datetime, str]] = []
    for section in _FOMC_SECTION_RE.finditer(content):
        year = int(section.group("year"))
        body = section.group("body").split("Note:", 1)[0]
        for match in _NO_YEAR_DATE_RE.finditer(body):
            if not match.group("end_day"):
                continue
            month = _MONTHS.get(match.group("month").lower().rstrip("."))
            if month is None:
                continue
            try:
                candidates.append((datetime(year, month, int(match.group("end_day"))), match.group(0)))
            except ValueError as error:
                raise CalendarFormatError(f"invalid official FOMC date: {match.group(0)}") from error
    return candidates


def _parse_bls_monthly_candidates(
    content: str,
    *,
    release_name: str,
) -> list[tuple[datetime, str]]:
    header = _BLS_MONTH_HEADER_RE.search(content)
    if header is None:
        raise CalendarFormatError("BLS monthly calendar header is missing")
    page_month = _MONTHS[header.group("month").lower()]
    page_year = int(header.group("year"))
    expected = release_name.casefold()
    candidates: list[tuple[datetime, str]] = []
    for cell in _BLS_DAY_CELL_RE.finditer(content):
        cell_text = html.unescape(re.sub(r"<[^>]+>", " ", cell.group("body")))
        normalized = " ".join(cell_text.split()).casefold()
        if expected not in normalized:
            continue
        month = int(cell.group("month"))
        day = int(cell.group("day"))
        year = page_year
        if page_month == 12 and month == 1:
            year += 1
        elif page_month == 1 and month == 12:
            year -= 1
        try:
            candidates.append(
                (datetime(year, month, day), f"{month:02d}/{day:02d}/{year}")
            )
        except ValueError as error:
            raise CalendarFormatError(
                f"invalid BLS monthly calendar cell: d{month:02d}{day:02d}"
            ) from error
    return candidates


def _calendar_request_urls(
    source: Mapping[str, object],
    *,
    retrieved_at_utc: datetime,
) -> tuple[str, ...]:
    template = source.get("monthly_url_template")
    if template is None:
        return (str(source["url"]),)
    template_value = str(template)
    months = max(int(source.get("monthly_schedule_months", 2)), 1)
    urls: list[str] = []
    for offset in range(months):
        absolute_month = retrieved_at_utc.year * 12 + retrieved_at_utc.month - 1 + offset
        year, zero_based_month = divmod(absolute_month, 12)
        url = template_value.format(year=year, month=zero_based_month + 1)
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_DOMAINS:
            raise CalendarFormatError("monthly calendar URL must be approved official HTTPS")
        urls.append(url)
    return tuple(urls)


def _event_time(value: datetime, *, release_hour_et: int, release_minute_et: int) -> datetime:
    local = datetime.combine(value.date(), time(release_hour_et, release_minute_et), tzinfo=EASTERN)
    return local.astimezone(UTC)


def parse_official_calendar_events(
    *,
    source: Mapping[str, object],
    content: str,
    retrieved_at_utc: datetime,
) -> tuple[dict[str, object], ...]:
    """Parse one known official schedule using source-specific keyword rules."""

    source_name, event_type, _ = _validate_source(source)
    if retrieved_at_utc.tzinfo is None or retrieved_at_utc.utcoffset() is None:
        raise ValueError("retrieved_at_utc must be timezone-aware")
    retrieved = retrieved_at_utc.astimezone(UTC)
    parser_keywords = tuple(map(str, source.get("date_keywords", [])))
    release_name = source.get("release_name")
    candidates = (
        _parse_bls_monthly_candidates(content, release_name=str(release_name))
        if release_name is not None
        else _parse_fomc_candidates(content)
        if event_type == "FOMC"
        else []
    )
    if not candidates:
        candidates = _parse_candidates(content, keywords=parser_keywords)
    if not candidates:
        raise CalendarFormatError(f"official {event_type} calendar yielded no dates")
    past_days = int(source.get("past_days", 2))
    horizon_days = int(source.get("horizon_days", 370))
    release_hour = int(source.get("release_hour_et", 8))
    release_minute = int(source.get("release_minute_et", 0 if event_type == "FOMC" else 30))
    events: dict[str, dict[str, object]] = {}
    for candidate, label in candidates:
        scheduled = _event_time(candidate, release_hour_et=release_hour, release_minute_et=release_minute)
        if not (retrieved - timedelta(days=past_days) <= scheduled <= retrieved + timedelta(days=horizon_days)):
            continue
        event_id = f"{source_name}:{event_type}:{scheduled.date().isoformat()}"
        events[event_id] = {
            "event_id": event_id,
            "event_type": event_type,
            "scheduled_release_at_utc": scheduled.isoformat().replace("+00:00", "Z"),
            "actual_release_at_utc": scheduled.isoformat().replace("+00:00", "Z") if scheduled <= retrieved else None,
            "high_impact": True,
            "source": source_name,
            "source_url": str(source["url"]),
            "source_date_label": label,
        }
    if not events:
        raise CalendarFormatError(f"official {event_type} calendar has no dates in refresh window")
    return tuple(events[key] for key in sorted(events))


def fetch_official_calendar(
    *,
    sources: Iterable[Mapping[str, object]],
    retrieved_at_utc: datetime | None = None,
    freshness_hours: int = 6,
    client: CalendarHTTPClient | None = None,
) -> dict[str, object]:
    """Fetch official calendars and publish any usable first-party subset."""

    source_list = tuple(sources)
    if retrieved_at_utc is None:
        retrieved = datetime.now(UTC)
    else:
        if retrieved_at_utc.tzinfo is None or retrieved_at_utc.utcoffset() is None:
            raise ValueError("retrieved_at_utc must be timezone-aware")
        retrieved = retrieved_at_utc.astimezone(UTC)
    owns_client = client is None
    active_client: CalendarHTTPClient = (
        httpx.Client(
            timeout=20.0,
            follow_redirects=True,
            headers={
                "User-Agent": "AuPilot/0.1 read-only official macro calendar collector",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        if client is None
        else client
    )
    events: dict[str, dict[str, object]] = {}
    source_reports: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    try:
        for source in source_list:
            source_name = str(source.get("source", ""))
            event_type = str(source.get("event_type", "")).upper()
            url = str(source.get("url", ""))
            try:
                _validate_source(source)
                source_event_count = 0
                for requested_url in _calendar_request_urls(
                    source,
                    retrieved_at_utc=retrieved,
                ):
                    response = active_client.get(requested_url)
                    response.raise_for_status()
                    final_url = str(response.url)
                    final_parsed = urlparse(final_url)
                    if final_parsed.scheme != "https" or final_parsed.hostname not in OFFICIAL_DOMAINS:
                        raise CalendarFormatError("official calendar redirected outside the allowlist")
                    body = response.content if isinstance(response.content, bytes) else response.text.encode("utf-8")
                    content_sha256 = hashlib.sha256(body).hexdigest().upper()
                    content = response.text
                    if source.get("release_name") is None:
                        content, _ = extract_official_text(response.text)
                    try:
                        parsed_events = parse_official_calendar_events(
                            source={**source, "url": final_url},
                            content=content,
                            retrieved_at_utc=retrieved,
                        )
                    except CalendarFormatError as error:
                        if (
                            source.get("release_name") is None
                            or "has no dates in refresh window" not in str(error)
                        ):
                            raise
                        parsed_events = ()
                    source_event_count += len(parsed_events)
                    for event in parsed_events:
                        events[str(event["event_id"])] = event
                    source_reports.append(
                        {
                            "source": source_name,
                            "event_type": event_type,
                            "requested_url": requested_url,
                            "canonical_url": final_url,
                            "http_status": int(getattr(response, "status_code", 200)),
                            "content_sha256": content_sha256,
                            "event_count": len(parsed_events),
                            "parser_version": PARSER_VERSION,
                        }
                    )
                if source_event_count == 0:
                    raise CalendarFormatError(
                        f"official {event_type} monthly calendars have no dates in refresh window"
                    )
            except Exception as error:
                failures.append(
                    _failure_report(
                        source=source_name,
                        event_type=event_type,
                        requested_url=url,
                        error=error,
                    )
                )
    finally:
        if owns_client:
            close = getattr(active_client, "close", None)
            if close is not None:
                close()
    freshness_hours = max(int(freshness_hours), 1)
    return {
        "schema_version": "aupilot.macro_calendar.v1",
        "retrieved_at_utc": retrieved.isoformat().replace("+00:00", "Z"),
        "fresh_until_utc": (retrieved + timedelta(hours=freshness_hours)).isoformat().replace("+00:00", "Z"),
        "fetch_succeeded": bool(events),
        "source_degraded": bool(failures),
        "events": [events[key] for key in sorted(events)],
        "sources": source_reports,
        "failures": failures,
        "parser_version": PARSER_VERSION,
    }


def publish_calendar_snapshot(
    snapshot: Mapping[str, object],
    *,
    storage_root: str | Path,
    active_filename: str = "calendar_snapshot.json",
    run_id: str | None = None,
) -> dict[str, object]:
    """Write an immutable refresh run and atomically replace the active snapshot."""

    root = Path(storage_root).resolve()
    identifier = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    run_root = root / "refresh" / "calendar" / identifier
    run_root.mkdir(parents=True, exist_ok=False)
    snapshot_path = run_root / "calendar_snapshot.json"
    write_json_atomic(snapshot_path, dict(snapshot))
    manifest = {
        "schema_version": "aupilot.macro_calendar_refresh.v1",
        "operation": "OFFICIAL_MACRO_CALENDAR_REFRESH",
        "run_id": identifier,
        "snapshot": {
            "path": snapshot_path.relative_to(root).as_posix(),
            "bytes": snapshot_path.stat().st_size,
            "sha256": sha256_file(snapshot_path),
        },
        "fetch_succeeded": snapshot.get("fetch_succeeded") is True,
        "event_count": len(snapshot.get("events", [])),
        "source_count": len(snapshot.get("sources", [])),
        "failure_count": len(snapshot.get("failures", [])),
        "historical_profit_tuning_allowed": False,
        "trade_permission": False,
        "technical_model_input_allowed": False,
        "decision_engine_input_allowed": False,
    }
    manifest_path = run_root / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    active_path = root / active_filename
    write_json_atomic(active_path, dict(snapshot), allow_overwrite=True)
    return {
        **manifest,
        "active_snapshot": {
            "path": active_path.relative_to(root).as_posix(),
            "bytes": active_path.stat().st_size,
            "sha256": sha256_file(active_path),
        },
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
    }
