from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aupilot.core.config import load_yaml
from aupilot.core.enums import MacroRiskLevel
from aupilot.core.hashing import sha256_file
from aupilot.core.macro_policy import (
    CAUTION_HOURS,
    HOLD_HOURS,
    MONITORING_HOURS,
    POST_RELEASE_CANCEL_HOURS,
    validate_event_factor_policy,
    validate_frozen_calendar_windows,
)

from .calendar_gate import MacroEvent, MacroGateResult, evaluate_macro_calendar
from .evidence_rag import EvidenceRAG
from .evidence_store import MacroEvidenceStore
from .schemas import ALLOWED_EVENT_TYPES, OFFICIAL_DOMAINS, MacroAssessment, MacroClaim


def _parse_utc(value: object, *, field: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include an explicit timezone")
    return parsed.astimezone(UTC)


def _unavailable_calendar(*reason_codes: str) -> MacroGateResult:
    return MacroGateResult(
        MacroRiskLevel.CAUTION,
        (*reason_codes, "ASSESSMENT_UNAVAILABLE"),
        news_summary=("The official macro event calendar is currently unavailable.",),
    )


def _calendar_source_id(value: object, source_url: object = "") -> str:
    source = str(value or "")
    if source.startswith("bls"):
        return "bls"
    if source.startswith("bea"):
        return "bea"
    if source:
        return source
    hostname = urlparse(str(source_url)).hostname
    return {
        "www.federalreserve.gov": "federal_reserve",
        "www.bls.gov": "bls",
        "www.bea.gov": "bea",
    }.get(hostname or "", hostname or "")


def load_calendar_result(
    snapshot_path: str | Path,
    *,
    decision_as_of_utc: datetime,
    pre_release_window_hours: int = MONITORING_HOURS,
    caution_window_hours: int = CAUTION_HOURS,
    hold_window_hours: int = HOLD_HOURS,
    post_release_cooldown_hours: int = POST_RELEASE_CANCEL_HOURS,
    enabled_source_ids: frozenset[str] | None = None,
) -> MacroGateResult:
    """Load a bounded official-event snapshot and fail closed on every invalid state."""

    path = Path(snapshot_path)
    if not path.is_file():
        return _unavailable_calendar("MACRO_CALENDAR_SNAPSHOT_MISSING")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("calendar snapshot must be a JSON object")
        fetch_succeeded = payload.get("fetch_succeeded") is True
        fresh_until = _parse_utc(payload["fresh_until_utc"], field="fresh_until_utc")
        raw_events = payload.get("events")
        if not isinstance(raw_events, list):
            raise ValueError("calendar snapshot events must be a list")
        events = []
        for value in raw_events:
            if not isinstance(value, dict):
                raise ValueError("calendar event must be an object")
            if (
                enabled_source_ids is not None
                and _calendar_source_id(value.get("source"), value.get("source_url"))
                not in enabled_source_ids
            ):
                continue
            event_type = str(value["event_type"]).upper()
            if event_type not in ALLOWED_EVENT_TYPES:
                raise ValueError(f"unsupported macro event type: {event_type}")
            source_url = str(value["source_url"])
            source = urlparse(source_url)
            if source.scheme != "https" or source.hostname not in OFFICIAL_DOMAINS:
                raise ValueError("calendar event source must be an approved official HTTPS URL")
            actual = value.get("actual_release_at_utc")
            events.append(
                MacroEvent(
                    event_id=str(value["event_id"]),
                    event_type=event_type,
                    scheduled_release_at_utc=_parse_utc(
                        value["scheduled_release_at_utc"],
                        field="scheduled_release_at_utc",
                    ),
                    actual_release_at_utc=(
                        None
                        if actual in (None, "")
                        else _parse_utc(actual, field="actual_release_at_utc")
                    ),
                    high_impact=bool(value.get("high_impact", True)),
                )
            )
        return evaluate_macro_calendar(
            decision_as_of_utc=decision_as_of_utc,
            events=tuple(events),
            calendar_is_fresh=decision_as_of_utc <= fresh_until,
            fetch_succeeded=(
                fetch_succeeded
                and (enabled_source_ids is None or bool(events))
            ),
            pre_release_window=timedelta(hours=pre_release_window_hours),
            caution_window=timedelta(hours=caution_window_hours),
            hold_window=timedelta(hours=hold_window_hours),
            post_release_cooldown=timedelta(hours=post_release_cooldown_hours),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return _unavailable_calendar("MACRO_CALENDAR_SNAPSHOT_INVALID")


def load_calendar_claims(
    snapshot_path: str | Path,
    *,
    decision_as_of_utc: datetime,
    enabled_source_ids: frozenset[str] | None = None,
) -> tuple[MacroClaim, ...]:
    path = Path(snapshot_path)
    if not path.is_file():
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        retrieved = _parse_utc(payload["retrieved_at_utc"], field="retrieved_at_utc")
        if retrieved > decision_as_of_utc:
            return ()
        digest = sha256_file(path)
        events = payload["events"]
        if not isinstance(events, list):
            return ()
        slots = {
            "FOMC": "FOMC.calendar",
            "CPI": "CPI.next_release_time",
            "PCE": "PCE.next_release_time",
            "NFP": "NFP.next_release_time",
        }
        claims = []
        for event_type, slot in slots.items():
            upcoming = [
                value
                for value in events
                if isinstance(value, dict)
                and (
                    enabled_source_ids is None
                    or _calendar_source_id(value.get("source"), value.get("source_url"))
                    in enabled_source_ids
                )
                and str(value.get("event_type", "")).upper() == event_type
                and _parse_utc(
                    value["scheduled_release_at_utc"],
                    field="scheduled_release_at_utc",
                )
                >= decision_as_of_utc
            ]
            if not upcoming:
                continue
            event = min(
                upcoming,
                key=lambda value: _parse_utc(
                    value["scheduled_release_at_utc"],
                    field="scheduled_release_at_utc",
                ),
            )
            scheduled = _parse_utc(
                event["scheduled_release_at_utc"],
                field="scheduled_release_at_utc",
            )
            source_url = str(event["source_url"])
            parsed = urlparse(source_url)
            if parsed.scheme != "https" or parsed.hostname not in OFFICIAL_DOMAINS:
                return ()
            event_id = str(event["event_id"])
            provider_id = str(event.get("source") or parsed.hostname)
            claim_hash = digest[:32]
            claims.append(
                MacroClaim(
                    claim_id=f"claim:calendar:{event_id}:{claim_hash}",
                    slot=slot,
                    event_type=event_type,
                    claim_type=slot.split(".", 1)[1],
                    normalized_value=scheduled.isoformat(),
                    display_text=(
                        f"The official calendar schedules {event_type} for "
                        f"{scheduled.isoformat()}."
                    ),
                    value={"scheduled_release_at_utc": scheduled.isoformat()},
                    reference_period=scheduled.date().isoformat(),
                    observed_at_utc=scheduled,
                    source_record_id=event_id,
                    provider_id=provider_id,
                    source_tier="A",
                    canonical_url=source_url,
                    first_seen_at_utc=retrieved,
                    retrieved_at_utc=retrieved,
                    eligible_from_utc=retrieved,
                    content_sha256=digest,
                    retrieval_method="HTML",
                    official_primary=True,
                    revision_status="INITIAL",
                    independence_key=provider_id,
                )
            )
        return tuple(claims)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return ()


def assess_macro_risk(
    *,
    database_path: str | Path,
    calendar_snapshot_path: str | Path,
    config_path: str | Path,
    decision_as_of_utc: datetime,
    replay_only: bool = False,
    enabled_source_ids: frozenset[str] | None = None,
    allowed_provider_ids: frozenset[str] | None = None,
    read_stored_evidence: bool = True,
    calendar_integrity_error: str | None = None,
) -> MacroAssessment:
    """Produce one independent five-level assessment without any technical input."""

    if decision_as_of_utc.tzinfo is None or decision_as_of_utc.utcoffset() is None:
        raise ValueError("decision_as_of_utc must be timezone-aware")
    decision_as_of_utc = decision_as_of_utc.astimezone(UTC)
    config: dict[str, Any] = load_yaml(config_path)
    rate_config = config["formal_rate_rag"]
    windows = config["calendar_windows"]
    validate_frozen_calendar_windows(windows)
    validate_event_factor_policy(config["event_factor_policy"])
    if calendar_integrity_error is None:
        calendar_result = load_calendar_result(
            calendar_snapshot_path,
            decision_as_of_utc=decision_as_of_utc,
            pre_release_window_hours=int(windows["monitoring_hours"]),
            caution_window_hours=int(windows["caution_hours"]),
            hold_window_hours=int(windows["hold_hours"]),
            post_release_cooldown_hours=int(windows["post_release_cancel_hours"]),
            enabled_source_ids=enabled_source_ids,
        )
        calendar_claims = load_calendar_claims(
            calendar_snapshot_path,
            decision_as_of_utc=decision_as_of_utc,
            enabled_source_ids=enabled_source_ids,
        )
    else:
        calendar_result = _unavailable_calendar(calendar_integrity_error)
        calendar_claims = ()
    return EvidenceRAG(
        MacroEvidenceStore(database_path),
        top_k=int(config["retrieval"]["top_k"]),
        rate_series_ids=tuple(map(str, rate_config["series_ids"])),
        minimum_rate_series=1,
        maximum_rate_staleness_days=int(
            rate_config["maximum_staleness_calendar_days"]
        ),
        require_rate_observations=bool(rate_config["require_for_formal_assessment"]),
        required_coverage=config["required_coverage"],
        expectation_config=config["expectation_uncertainty"],
        release_cooldown_hours=int(windows["post_release_cancel_hours"]),
        allowed_provider_ids=allowed_provider_ids,
        read_stored_evidence=read_stored_evidence,
    ).assess(
        decision_as_of_utc=decision_as_of_utc,
        calendar_result=calendar_result,
        replay_only=replay_only,
        calendar_claims=calendar_claims,
    )
