from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aupilot.core.config import load_yaml, project_root, resolve_project_path
from aupilot.core.hashing import sha256_file
from aupilot.core.manifest import write_json_atomic

from .evidence_rag import EvidenceRAG
from .evidence_store import MacroEvidenceStore
from .golden_eval import load_verified_macro_rag_golden_evaluation
from .runtime import load_calendar_claims, load_calendar_result


def _as_utc(value: str | datetime) -> datetime:
    timestamp = (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, str)
        else value
    )
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Formal macro evidence as_of must include an explicit timezone")
    return timestamp.astimezone(UTC)


def _snapshot_sqlite(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"Formal macro evidence database is missing: {source}")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite macro evidence snapshot: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_uri = f"file:{source.as_posix()}?mode=ro"
    with closing(sqlite3.connect(source_uri, uri=True)) as source_connection:
        with closing(sqlite3.connect(destination)) as destination_connection:
            source_connection.backup(destination_connection)
            destination_connection.commit()
    with closing(sqlite3.connect(destination)) as connection:
        mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()
    if mode is None or str(mode[0]).lower() != "delete":
        raise RuntimeError("Failed to seal formal macro evidence SQLite snapshot")


def _assessment_report(
    store: MacroEvidenceStore,
    *,
    calendar_snapshot_path: Path,
    as_of_utc: datetime,
    replay_only: bool,
    macro_config: dict[str, Any],
) -> dict[str, Any]:
    windows = macro_config["calendar_windows"]
    calendar = load_calendar_result(
        calendar_snapshot_path,
        decision_as_of_utc=as_of_utc,
        pre_release_window_hours=int(windows["monitoring_hours"]),
        caution_window_hours=int(windows["caution_hours"]),
        hold_window_hours=int(windows["hold_hours"]),
        post_release_cooldown_hours=int(windows["post_release_cancel_hours"]),
    )
    assessment = EvidenceRAG(
        store,
        top_k=int(macro_config["retrieval"]["top_k"]),
        required_coverage=macro_config["required_coverage"],
        expectation_config=macro_config["expectation_uncertainty"],
        release_cooldown_hours=int(windows["post_release_cancel_hours"]),
    ).assess(
        decision_as_of_utc=as_of_utc,
        calendar_result=calendar,
        replay_only=replay_only,
        calendar_claims=load_calendar_claims(
            calendar_snapshot_path,
            decision_as_of_utc=as_of_utc,
        ),
    )
    future_citations = sum(
        value.eligible_from_utc > as_of_utc for value in assessment.citations
    )
    coverage_complete = bool(assessment.coverage) and all(
        not value.required or value.status in {"COVERED", "DEGRADED"}
        for value in assessment.coverage
    )
    passed = bool(
        assessment.assessment_supported is True
        and future_citations == 0
        and coverage_complete
        and all(value.claim_ids for value in assessment.summary_facts)
    )
    return {
        "as_of_utc": as_of_utc.isoformat(),
        "replay_only": replay_only,
        "document_count": store.document_count(),
        "observation_count": store.observation_count(),
        "claim_count": store.claim_count(),
        "assessment": assessment.model_dump(mode="json"),
        "future_citation_count": future_citations,
        "assessment_supported": assessment.assessment_supported,
        "coverage_complete": coverage_complete,
        "source_degraded": assessment.source_degraded,
        "passed": passed,
        "historical_profit_tuning_allowed": False,
        "model_selection_allowed": False,
        "trade_permission": False,
        "formal_performance_claim_allowed": False,
    }


def freeze_formal_macro_evidence(
    *,
    source_database_path: str | Path,
    golden_manifest_path: str | Path,
    as_of_utc: str | datetime,
    replay_only: bool,
    output: str | Path,
    calendar_snapshot_path: str | Path = "storage/macro/calendar_snapshot.json",
    root: Path | None = None,
) -> dict[str, Any]:
    workspace = project_root() if root is None else root.resolve()
    source_path = resolve_project_path(source_database_path, root=workspace)
    calendar_source = resolve_project_path(calendar_snapshot_path, root=workspace)
    golden_path = resolve_project_path(golden_manifest_path, root=workspace)
    golden = load_verified_macro_rag_golden_evaluation(golden_path, root=workspace)
    if golden.get("passed") is not True:
        raise PermissionError("Macro RAG golden evidence did not pass")
    if not calendar_source.is_file():
        raise FileNotFoundError("Formal macro calendar snapshot is missing")
    as_of = _as_utc(as_of_utc)
    destination = resolve_project_path(output, root=workspace)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite formal macro evidence: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    source_sha_at_snapshot = sha256_file(source_path)
    calendar_sha_at_snapshot = sha256_file(calendar_source)
    snapshot_path = destination / "evidence_snapshot.sqlite"
    calendar_snapshot = destination / "calendar_snapshot.json"
    _snapshot_sqlite(source_path, snapshot_path)
    shutil.copy2(calendar_source, calendar_snapshot)
    macro_config_path = workspace / "configs" / "macro.yaml"
    macro_config = load_yaml(macro_config_path)
    report = _assessment_report(
        MacroEvidenceStore(snapshot_path),
        calendar_snapshot_path=calendar_snapshot,
        as_of_utc=as_of,
        replay_only=replay_only,
        macro_config=macro_config,
    )
    report_path = destination / "formal_macro_evidence_report.json"
    write_json_atomic(report_path, report)
    manifest = {
        "schema_version": "aupilot.formal_macro_evidence.v3",
        "operation": "FREEZE_POINT_IN_TIME_MULTISOURCE_MACRO_EVIDENCE",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "as_of_utc": as_of.isoformat(),
        "replay_only": replay_only,
        "source_database": {
            "path": source_path.relative_to(workspace).as_posix(),
            "sha256_at_snapshot": source_sha_at_snapshot,
            "may_append_after_snapshot": True,
        },
        "source_calendar": {
            "path": calendar_source.relative_to(workspace).as_posix(),
            "sha256_at_snapshot": calendar_sha_at_snapshot,
            "may_refresh_after_snapshot": True,
        },
        "snapshot": {
            "path": snapshot_path.relative_to(workspace).as_posix(),
            "bytes": snapshot_path.stat().st_size,
            "sha256": sha256_file(snapshot_path),
        },
        "calendar_snapshot": {
            "path": calendar_snapshot.relative_to(workspace).as_posix(),
            "bytes": calendar_snapshot.stat().st_size,
            "sha256": sha256_file(calendar_snapshot),
        },
        "golden_evidence": {
            "path": golden_path.relative_to(workspace).as_posix(),
            "bytes": golden_path.stat().st_size,
            "sha256": sha256_file(golden_path),
        },
        "macro_config": {
            "path": macro_config_path.relative_to(workspace).as_posix(),
            "bytes": macro_config_path.stat().st_size,
            "sha256": sha256_file(macro_config_path),
        },
        "report": {
            "path": report_path.relative_to(workspace).as_posix(),
            "bytes": report_path.stat().st_size,
            "sha256": sha256_file(report_path),
        },
        "assessment_risk_level": report["assessment"]["risk_level"],
        "future_citation_count": report["future_citation_count"],
        "coverage_complete": report["coverage_complete"],
        "passed": report["passed"],
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


def _verified_record(record: dict[str, Any], *, root: Path, role: str) -> Path:
    path = resolve_project_path(str(record.get("path", "")), root=root)
    if (
        not path.is_file()
        or path.stat().st_size != int(record.get("bytes", -1))
        or sha256_file(path) != str(record.get("sha256", "")).upper()
    ):
        raise RuntimeError(f"Formal macro {role} changed")
    return path


def load_verified_formal_macro_evidence(
    manifest_path: str | Path,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    workspace = project_root() if root is None else root.resolve()
    path = resolve_project_path(manifest_path, root=workspace)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "aupilot.formal_macro_evidence.v3"
        or manifest.get("operation")
        != "FREEZE_POINT_IN_TIME_MULTISOURCE_MACRO_EVIDENCE"
        or manifest.get("historical_profit_tuning_allowed") is not False
        or manifest.get("model_selection_allowed") is not False
        or manifest.get("trade_permission") is not False
        or manifest.get("formal_performance_claim_allowed") is not False
    ):
        raise ValueError("Unknown or unsafe formal macro evidence manifest")
    snapshot_record = manifest.get("snapshot")
    calendar_record = manifest.get("calendar_snapshot")
    golden_record = manifest.get("golden_evidence")
    config_record = manifest.get("macro_config")
    report_record = manifest.get("report")
    if not all(
        isinstance(value, dict)
        for value in (
            snapshot_record,
            calendar_record,
            golden_record,
            config_record,
            report_record,
        )
    ):
        raise ValueError("Formal macro evidence records are incomplete")
    snapshot_path = _verified_record(snapshot_record, root=workspace, role="snapshot")
    calendar_path = _verified_record(calendar_record, root=workspace, role="calendar")
    golden_path = _verified_record(golden_record, root=workspace, role="golden evidence")
    config_path = _verified_record(config_record, root=workspace, role="config")
    report_path = _verified_record(report_record, root=workspace, role="report")
    golden = load_verified_macro_rag_golden_evaluation(golden_path, root=workspace)
    if golden.get("passed") is not True:
        raise PermissionError("Macro RAG golden evidence no longer passes")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recomputed = _assessment_report(
        MacroEvidenceStore(snapshot_path),
        calendar_snapshot_path=calendar_path,
        as_of_utc=_as_utc(str(manifest.get("as_of_utc", ""))),
        replay_only=bool(manifest.get("replay_only")),
        macro_config=load_yaml(config_path),
    )
    if report != recomputed:
        raise RuntimeError("Formal macro report disagrees with its immutable snapshots")
    summary = {
        "assessment_risk_level": recomputed["assessment"]["risk_level"],
        "future_citation_count": recomputed["future_citation_count"],
        "coverage_complete": recomputed["coverage_complete"],
        "passed": recomputed["passed"],
    }
    if any(manifest.get(key) != value for key, value in summary.items()):
        raise RuntimeError("Formal macro manifest flags disagree with recomputation")
    return manifest
