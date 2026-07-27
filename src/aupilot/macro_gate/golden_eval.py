from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from aupilot.core.config import load_yaml, project_root, resolve_project_path
from aupilot.core.enums import MacroRiskLevel
from aupilot.core.hashing import canonical_json_sha256, sha256_file
from aupilot.core.manifest import write_json_atomic

from .calendar_gate import MacroGateResult
from .evidence_rag import EvidenceRAG
from .evidence_store import MacroEvidenceStore
from .schemas import MacroClaim


CASE_ID_PATTERN = re.compile(r"^[a-z0-9_]+$")
IMPLEMENTATION_SOURCES = {
    "calendar-risk": "src/aupilot/macro_gate/calendar_gate.py",
    "coverage-engine": "src/aupilot/macro_gate/coverage.py",
    "evidence-rag": "src/aupilot/macro_gate/evidence_rag.py",
    "evidence-store": "src/aupilot/macro_gate/evidence_store.py",
    "assessment-schema": "src/aupilot/macro_gate/schemas.py",
    "multisource-providers": "src/aupilot/providers/macro_multi_source.py",
    "golden-evaluator": "src/aupilot/macro_gate/golden_eval.py",
}


def _as_utc(value: object, *, name: str) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must include an explicit timezone")
    return timestamp.tz_convert("UTC").to_pydatetime()


def _validated_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "aupilot.macro_rag_golden.v4":
        raise ValueError("Unknown macro RAG golden fixture schema")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Macro RAG golden fixture has no cases")
    case_ids = [str(item.get("case_id", "")) for item in cases if isinstance(item, dict)]
    if (
        len(case_ids) != len(cases)
        or len(case_ids) != len(set(case_ids))
        or any(CASE_ID_PATTERN.fullmatch(value) is None for value in case_ids)
    ):
        raise ValueError("Macro RAG golden case ids must be unique safe identifiers")
    expected_levels = {str(item.get("expected_risk_level")) for item in cases}
    if expected_levels != {value.value for value in MacroRiskLevel}:
        raise ValueError("Macro RAG golden fixture must cover all five risk levels")
    return {**payload, "as_of_utc": _as_utc(payload.get("as_of_utc"), name="as_of_utc")}


def _claim(
    *,
    case_id: str,
    slot: str,
    event_type: str,
    normalized_value: str,
    value: object,
    provider_id: str,
    url: str,
    observed_at_utc: datetime,
    first_seen_at_utc: datetime,
    source_tier: str = "A",
    revision_status: str = "INITIAL",
    near_duplicate_group: str | None = None,
) -> MacroClaim:
    identity = {
        "case_id": case_id,
        "slot": slot,
        "value": value,
        "provider": provider_id,
        "first_seen": first_seen_at_utc.isoformat(),
    }
    digest = canonical_json_sha256(identity)
    published = observed_at_utc if slot in {
        "FOMC.latest_official_decision",
        "CPI.latest_initial_release",
        "PCE.latest_initial_release",
        "NFP.latest_initial_release",
    } else None
    return MacroClaim(
        claim_id=f"golden-claim:{case_id}:{digest[:20]}",
        slot=slot,
        event_type=event_type,
        claim_type=slot.split(".", 1)[1],
        normalized_value=normalized_value,
        display_text=f"Golden fact for {slot}: {normalized_value}.",
        value=value,
        reference_period=observed_at_utc.date().isoformat(),
        observed_at_utc=observed_at_utc,
        source_record_id=f"golden-source:{case_id}:{digest[:20]}",
        provider_id=provider_id,
        source_tier=source_tier,
        canonical_url=url,
        published_at_utc=published,
        first_seen_at_utc=first_seen_at_utc,
        retrieved_at_utc=first_seen_at_utc,
        eligible_from_utc=first_seen_at_utc,
        content_sha256=digest,
        retrieval_method="HTML" if published else "API",
        official_primary=source_tier == "A",
        revision_status=revision_status,
        independence_key=provider_id,
        near_duplicate_group=near_duplicate_group,
    )


def build_complete_fixture_claims(case_id: str, as_of: datetime) -> list[MacroClaim]:
    seen = as_of - timedelta(hours=1)
    rows = (
        ("FOMC.calendar", "FOMC", "2026-03-18T18:00:00Z", {"scheduled_release_at_utc": "2026-03-18T18:00:00Z"}, "fed_calendar", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", as_of + timedelta(days=62)),
        ("FOMC.latest_official_decision", "FOMC", "2025-12-10:3.5:3.75", {"target_lower": "3.5", "target_upper": "3.75"}, "fed_release", "https://www.federalreserve.gov/newsevents/pressreleases/monetary20251210a.htm", as_of - timedelta(days=36)),
        ("FOMC.market_expectation", "FOMC", "hold:0.85", {"max_outcome_probability": 0.85}, "cme_authorized", "https://www.cmegroup.com/fedwatch", as_of - timedelta(hours=1)),
        ("RATES.effr", "RATES", "2026-01-14:3.63", 3.63, "new_york_fed", "https://markets.newyorkfed.org/api/rates/unsecured/effr/last/1.json", as_of - timedelta(days=1)),
        ("RATES.nominal_2y", "RATES", "2026-01-14:4.26", 4.26, "treasury", "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml", as_of - timedelta(days=1)),
        ("RATES.nominal_10y", "RATES", "2026-01-14:4.63", 4.63, "treasury", "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml", as_of - timedelta(days=1)),
        ("RATES.real_10y", "RATES", "2026-01-14:2.37", 2.37, "treasury", "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml", as_of - timedelta(days=1)),
        ("RATES.breakeven_proxy_10y", "RATES", "2026-01-14:2.26", 2.26, "treasury", "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml", as_of - timedelta(days=1)),
        ("CPI.next_release_time", "CPI", "2026-02-11T13:30:00Z", {"scheduled_release_at_utc": "2026-02-11T13:30:00Z"}, "bls_calendar", "https://www.bls.gov/schedule/news_release/cpi.htm", as_of + timedelta(days=27)),
        ("CPI.latest_initial_release", "CPI", "2025-12:324.1", 324.1, "bls_api", "https://api.bls.gov/publicAPI/v2/timeseries/data/CUUR0000SA0", as_of - timedelta(days=20)),
        ("PCE.next_release_time", "PCE", "2026-01-30T13:30:00Z", {"scheduled_release_at_utc": "2026-01-30T13:30:00Z"}, "bea_calendar", "https://www.bea.gov/news/schedule", as_of + timedelta(days=15)),
        ("PCE.latest_initial_release", "PCE", "2025-11", {"release_title": "Personal Income and Outlays, November 2025"}, "bea_release", "https://www.bea.gov/news/2025/personal-income-and-outlays-november-2025", as_of - timedelta(days=20)),
        ("NFP.next_release_time", "NFP", "2026-02-06T13:30:00Z", {"scheduled_release_at_utc": "2026-02-06T13:30:00Z"}, "bls_calendar", "https://www.bls.gov/schedule/news_release/empsit.htm", as_of + timedelta(days=22)),
        ("NFP.latest_initial_release", "NFP", "2025-12:159000", 159000, "bls_api", "https://api.bls.gov/publicAPI/v2/timeseries/data/CES0000000001", as_of - timedelta(days=20)),
    )
    return [
        _claim(
            case_id=case_id,
            slot=slot,
            event_type=event_type,
            normalized_value=normalized,
            value=value,
            provider_id=provider,
            url=url,
            observed_at_utc=observed,
            first_seen_at_utc=seen,
        )
        for slot, event_type, normalized, value, provider, url, observed in rows
    ]


def _mutated_claims(case: dict[str, Any], as_of: datetime) -> tuple[MacroClaim, ...]:
    case_id = str(case["case_id"])
    claims = build_complete_fixture_claims(case_id, as_of)
    mutation = case.get("mutation")
    if mutation == "missing_pce":
        claims = [item for item in claims if item.slot != "PCE.latest_initial_release"]
    elif mutation == "future_fomc":
        index = next(i for i, item in enumerate(claims) if item.slot == "FOMC.latest_official_decision")
        old = claims[index]
        claims[index] = old.model_copy(
            update={
                "claim_id": old.claim_id + ":future",
                "first_seen_at_utc": as_of + timedelta(days=1),
                "retrieved_at_utc": as_of + timedelta(days=1),
                "eligible_from_utc": as_of + timedelta(days=1),
            }
        )
    elif mutation == "stale_rates":
        claims = [
            item.model_copy(update={"observed_at_utc": as_of - timedelta(days=20)})
            if item.slot.startswith("RATES.")
            else item
            for item in claims
        ]
    elif mutation in {"tier_b_quorum", "duplicate_reprints"}:
        claims = [item for item in claims if item.slot != "CPI.latest_initial_release"]
        duplicate = "reuters-wire-1" if mutation == "duplicate_reprints" else None
        for provider, url in (
            ("reuters", "https://www.reuters.com/world/us/cpi-release"),
            ("cnbc", "https://www.cnbc.com/2026/01/14/cpi-release.html"),
        ):
            claims.append(
                _claim(
                    case_id=case_id,
                    slot="CPI.latest_initial_release",
                    event_type="CPI",
                    normalized_value="2025-12:324.1",
                    value=324.1,
                    provider_id=provider,
                    url=url,
                    observed_at_utc=as_of - timedelta(days=1),
                    first_seen_at_utc=as_of - timedelta(hours=1),
                    source_tier="B",
                    near_duplicate_group=duplicate,
                )
            )
    elif mutation == "conflicting_fomc_dates":
        claims = [item for item in claims if item.slot != "FOMC.calendar"]
        for provider, url, value in (
            ("reuters", "https://www.reuters.com/world/us/fomc-calendar", "2026-03-18T18:00:00Z"),
            ("cnbc", "https://www.cnbc.com/2026/01/14/fomc-calendar.html", "2026-03-19T18:00:00Z"),
        ):
            claims.append(
                _claim(
                    case_id=case_id,
                    slot="FOMC.calendar",
                    event_type="FOMC",
                    normalized_value=value,
                    value={"scheduled_release_at_utc": value},
                    provider_id=provider,
                    url=url,
                    observed_at_utc=as_of + timedelta(days=62),
                    first_seen_at_utc=as_of - timedelta(hours=1),
                    source_tier="B",
                )
            )
    elif mutation == "revised_nfp":
        claims = [
            item.model_copy(update={"revision_status": "REVISED"})
            if item.slot == "NFP.latest_initial_release"
            else item
            for item in claims
        ]
    return tuple(claims)


def _case_calendar(case: dict[str, Any]) -> MacroGateResult:
    reasons = tuple(map(str, case.get("calendar_reason_codes", ())))
    if not reasons:
        raise ValueError("Golden case calendar reasons cannot be empty")
    return MacroGateResult(
        MacroRiskLevel(str(case["calendar_risk_level"])),
        reasons,
        news_summary=("Golden official event-calendar result.",),
    )


def _build_case_store(
    case: dict[str, Any], fixture: dict[str, Any], database: Path
) -> MacroEvidenceStore:
    store = MacroEvidenceStore(database)
    store.ingest_claims(_mutated_claims(case, fixture["as_of_utc"]))
    return store


def _evaluate_case(
    case: dict[str, Any],
    fixture: dict[str, Any],
    store: MacroEvidenceStore,
    macro_config: dict[str, Any],
) -> dict[str, Any]:
    assessment = EvidenceRAG(
        store,
        required_coverage=macro_config["required_coverage"],
        expectation_config=macro_config["expectation_uncertainty"],
        release_cooldown_hours=int(
            macro_config["calendar_windows"]["post_release_cancel_hours"]
        ),
    ).assess(
        decision_as_of_utc=fixture["as_of_utc"],
        calendar_result=_case_calendar(case),
    )
    expected_risk = MacroRiskLevel(str(case["expected_risk_level"]))
    expected_supported = bool(case["expected_supported"])
    expected_reasons = set(map(str, case.get("expected_reason_codes_contains", ())))
    actual_reasons = set(assessment.reason_codes)
    future_citations = sum(
        item.eligible_from_utc > fixture["as_of_utc"] for item in assessment.citations
    )
    calendar_floor_respected = assessment.risk_score >= _case_calendar(case).risk_level.score
    required_complete = all(
        not item.required or item.status in {"COVERED", "DEGRADED"}
        for item in assessment.coverage
    )
    passed = bool(
        assessment.risk_level is expected_risk
        and assessment.assessment_supported is expected_supported
        and expected_reasons.issubset(actual_reasons)
        and future_citations == 0
        and required_complete is expected_supported
        and calendar_floor_respected
        and all(item.claim_ids for item in assessment.summary_facts)
        and assessment.informational_only is True
        and assessment.trade_permission is False
        and assessment.technical_model_input_allowed is False
        and assessment.decision_engine_input_allowed is False
        and assessment.historical_profit_tuning_allowed is False
    )
    return {
        "case_id": case["case_id"],
        "expected_risk_level": expected_risk.value,
        "actual_risk_level": assessment.risk_level.value,
        "risk_score": assessment.risk_score,
        "expected_supported": expected_supported,
        "assessment_supported": assessment.assessment_supported,
        "expected_reason_codes": sorted(expected_reasons),
        "actual_reason_codes": list(assessment.reason_codes),
        "citation_count": len(assessment.citations),
        "claim_count": len(assessment.claims),
        "coverage": [item.model_dump(mode="json") for item in assessment.coverage],
        "future_citation_count": future_citations,
        "calendar_risk_floor_respected": calendar_floor_respected,
        "passed": passed,
    }


def _seal_database(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    with closing(sqlite3.connect(path)) as connection:
        mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()
    if mode is None or str(mode[0]).lower() != "delete":
        raise RuntimeError("Failed to seal macro RAG golden SQLite journal")


def _implementation_records(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "role": role,
            "path": relative,
            "bytes": (root / relative).stat().st_size,
            "sha256": sha256_file(root / relative),
        }
        for role, relative in IMPLEMENTATION_SOURCES.items()
    ]


def evaluate_macro_rag_golden(
    *,
    fixture_path: str | Path = "configs/macro_rag_golden.yaml",
    output: str | Path,
    root: Path | None = None,
) -> dict[str, Any]:
    workspace = project_root() if root is None else root.resolve()
    fixture_source = resolve_project_path(fixture_path, root=workspace)
    fixture = _validated_fixture(load_yaml(fixture_source))
    macro_config_path = workspace / "configs" / "macro.yaml"
    macro_config = load_yaml(macro_config_path)
    destination = resolve_project_path(output, root=workspace)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite macro RAG golden evidence: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []
    databases: list[dict[str, Any]] = []
    for case in fixture["cases"]:
        case_id = str(case["case_id"])
        case_root = destination / "cases" / case_id
        case_root.mkdir(parents=True, exist_ok=False)
        database = case_root / "evidence.sqlite"
        store = _build_case_store(case, fixture, database)
        results.append(_evaluate_case(case, fixture, store, macro_config))
        claim_count = store.claim_count()
        _seal_database(database)
        databases.append(
            {
                "case_id": case_id,
                "path": database.relative_to(workspace).as_posix(),
                "bytes": database.stat().st_size,
                "sha256": sha256_file(database),
                "claim_count": claim_count,
            }
        )
    frame = pd.DataFrame(results)
    result_path = destination / "case_results.parquet"
    frame.to_parquet(result_path, index=False)
    result_record = {
        "path": result_path.relative_to(workspace).as_posix(),
        "rows": len(frame),
        "bytes": result_path.stat().st_size,
        "sha256": sha256_file(result_path),
    }
    manifest = {
        "schema_version": "aupilot.macro_rag_golden_evaluation.v4",
        "operation": "EVALUATE_MULTISOURCE_MACRO_CLAIM_GOLDEN_SET",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "fixture": {
            "path": fixture_source.relative_to(workspace).as_posix(),
            "bytes": fixture_source.stat().st_size,
            "sha256": sha256_file(fixture_source),
        },
        "macro_config_sha256": sha256_file(macro_config_path),
        "implementation": _implementation_records(workspace),
        "cases_total": len(frame),
        "cases_passed": int(frame["passed"].sum()),
        "risk_level_counts": {
            level.value: int(frame["actual_risk_level"].eq(level.value).sum())
            for level in MacroRiskLevel
        },
        "future_citation_count": int(frame["future_citation_count"].sum()),
        "unsupported_assessment_count": int((~frame["assessment_supported"]).sum()),
        "calendar_risk_floor_violation_count": int(
            (~frame["calendar_risk_floor_respected"]).sum()
        ),
        "passed": bool(frame["passed"].all()),
        "case_results": result_record,
        "case_databases": databases,
        "historical_profit_tuning_allowed": False,
        "model_selection_allowed": False,
        "trade_permission": False,
        "formal_performance_claim_allowed": False,
    }
    manifest_path = destination / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    return {
        **manifest,
        "manifest_path": manifest_path.relative_to(workspace).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
    }


def _verified_database(record: dict[str, Any], *, root: Path) -> MacroEvidenceStore:
    path = resolve_project_path(str(record.get("path", "")), root=root)
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(path) != str(record.get("sha256", "")).upper()
    ):
        raise RuntimeError(f"Macro RAG golden database changed: {record.get('case_id')}")
    store = MacroEvidenceStore(path)
    if store.claim_count() != int(record.get("claim_count", -1)):
        raise RuntimeError(f"Macro RAG golden claim count changed: {record.get('case_id')}")
    return store


def load_verified_macro_rag_golden_evaluation(
    manifest_path: str | Path,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    workspace = project_root() if root is None else root.resolve()
    path = resolve_project_path(manifest_path, root=workspace)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "aupilot.macro_rag_golden_evaluation.v4"
        or manifest.get("operation") != "EVALUATE_MULTISOURCE_MACRO_CLAIM_GOLDEN_SET"
        or manifest.get("historical_profit_tuning_allowed") is not False
        or manifest.get("model_selection_allowed") is not False
        or manifest.get("trade_permission") is not False
        or manifest.get("formal_performance_claim_allowed") is not False
    ):
        raise ValueError("Unknown or unsafe macro RAG golden evaluation")
    fixture_record = manifest.get("fixture")
    if not isinstance(fixture_record, dict):
        raise ValueError("Macro RAG golden fixture record is missing")
    fixture_path = resolve_project_path(str(fixture_record.get("path", "")), root=workspace)
    if (
        not fixture_path.is_file()
        or fixture_path.stat().st_size != int(fixture_record.get("bytes", -1))
        or sha256_file(fixture_path) != str(fixture_record.get("sha256", "")).upper()
    ):
        raise RuntimeError("Macro RAG golden fixture changed")
    macro_config_path = workspace / "configs" / "macro.yaml"
    if sha256_file(macro_config_path) != str(manifest.get("macro_config_sha256", "")).upper():
        raise RuntimeError("Macro RAG config changed")
    records = manifest.get("implementation")
    if not isinstance(records, list):
        raise ValueError("Macro RAG implementation records are missing")
    by_role = {str(item.get("role")): item for item in records if isinstance(item, dict)}
    if set(by_role) != set(IMPLEMENTATION_SOURCES):
        raise ValueError("Macro RAG implementation role set is incomplete")
    for role, relative in IMPLEMENTATION_SOURCES.items():
        source = workspace / relative
        record = by_role[role]
        if (
            record.get("path") != relative
            or source.stat().st_size != int(record.get("bytes", -1))
            or sha256_file(source) != str(record.get("sha256", "")).upper()
        ):
            raise RuntimeError(f"Macro RAG implementation changed: {role}")
    fixture = _validated_fixture(load_yaml(fixture_path))
    macro_config = load_yaml(macro_config_path)
    database_records = manifest.get("case_databases")
    if not isinstance(database_records, list):
        raise ValueError("Macro RAG case databases are missing")
    by_case = {str(item.get("case_id")): item for item in database_records}
    if set(by_case) != {str(item["case_id"]) for item in fixture["cases"]}:
        raise ValueError("Macro RAG case database set is incomplete")
    recomputed = [
        _evaluate_case(
            case,
            fixture,
            _verified_database(by_case[str(case["case_id"])], root=workspace),
            macro_config,
        )
        for case in fixture["cases"]
    ]
    result_record = manifest.get("case_results")
    if not isinstance(result_record, dict):
        raise ValueError("Macro RAG case result record is missing")
    result_path = resolve_project_path(str(result_record.get("path", "")), root=workspace)
    if (
        not result_path.is_file()
        or result_path.stat().st_size != int(result_record.get("bytes", -1))
        or sha256_file(result_path) != str(result_record.get("sha256", "")).upper()
    ):
        raise RuntimeError("Macro RAG case result table changed")
    stored = json.loads(pd.read_parquet(result_path).to_json(orient="records"))
    if stored != json.loads(pd.DataFrame(recomputed).to_json(orient="records")):
        raise RuntimeError("Macro RAG golden results disagree with verified databases")
    if (
        manifest.get("cases_total") != len(recomputed)
        or manifest.get("cases_passed") != sum(bool(item["passed"]) for item in recomputed)
        or manifest.get("passed") is not all(bool(item["passed"]) for item in recomputed)
    ):
        raise RuntimeError("Macro RAG golden manifest summary disagrees with recomputation")
    return manifest
