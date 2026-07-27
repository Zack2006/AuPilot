from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

import httpx

from aupilot.core.config import load_yaml, project_root, resolve_project_path
from aupilot.core.hashing import sha256_file
from aupilot.core.manifest import write_json_atomic
from aupilot.macro_gate.evidence_store import MacroEvidenceStore
from aupilot.macro_gate.golden_eval import evaluate_macro_rag_golden
from aupilot.macro_gate.runtime import assess_macro_risk
from aupilot.providers.fred_alfred import (
    fetch_fred_initial_releases,
    persist_fred_initial_release_refresh,
)
from aupilot.providers.macro_calendar import fetch_official_calendar, publish_calendar_snapshot
from aupilot.providers.macro_multi_source import RetrievalRequest, default_coordinator
from aupilot.providers.macro_official import PARSER_VERSION, fetch_official_document


def _as_of(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("--as-of must include an explicit timezone")
    return parsed.astimezone(UTC)


def _utc_today() -> str:
    return datetime.now(UTC).date().isoformat()


def _seal_sqlite(path: Path) -> None:
    """Checkpoint WAL and seal a refresh database before hashing/publishing."""

    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()
        if mode is None or str(mode[0]).lower() != "delete":
            raise RuntimeError("Could not seal the macro evidence database")


def _copy_sqlite_database(source: Path, destination: Path) -> None:
    """Seed a refresh database without losing observations from the active store."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(source) as source_connection:
        with sqlite3.connect(destination) as destination_connection:
            source_connection.backup(destination_connection)
            destination_connection.commit()


def _fred_failure_reason_code(error: Exception) -> str:
    if isinstance(error, (httpx.TimeoutException, TimeoutError)):
        return "UPSTREAM_TIMEOUT"
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code is None:
        match = re.search(r"\bHTTP[ _:=]*(\d{3})\b", str(error), flags=re.IGNORECASE)
        status_code = None if match is None else int(match.group(1))
    if status_code is not None and 400 <= int(status_code) <= 599:
        return f"HTTP_{int(status_code)}"
    if isinstance(error, httpx.RemoteProtocolError):
        return "UPSTREAM_PROTOCOL_ERROR"
    return "FRED_REFRESH_FAILED"


def _write_fred_failure(args: argparse.Namespace, *, root: Path, error: Exception) -> int:
    report = {
        "schema_version": "aupilot.fred_refresh.v1",
        "operation": "FRED_ALFRED_INITIAL_RELEASE_REFRESH",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "fetch_succeeded": False,
        "published": False,
        "passed": False,
        "reason_code": _fred_failure_reason_code(error),
        "error_type": type(error).__name__,
        "message": str(error)[:500],
        "secret_recorded": False,
        "trade_permission": False,
        "technical_model_input_allowed": False,
        "decision_engine_input_allowed": False,
        "historical_profit_tuning_allowed": False,
    }
    report_path_value = None
    if args.report:
        report_path = resolve_project_path(args.report, root=root)
        write_json_atomic(report_path, report, allow_overwrite=True)
        report_path_value = report_path.relative_to(root).as_posix()
    database_path = resolve_project_path(args.database, root=root)
    previous_status: dict[str, object] = {}
    try:
        previous_payload = json.loads(
            (database_path.parent / "fred_refresh_status.json").read_text(encoding="utf-8")
        )
        if isinstance(previous_payload, dict):
            previous_status = previous_payload
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    write_json_atomic(
        database_path.parent / "fred_refresh_status.json",
        {
            "schema_version": "aupilot.fred_refresh_status.v1",
            "updated_at_utc": report["completed_at_utc"],
            "fetch_succeeded": False,
            "published": False,
            "reason_code": report["reason_code"],
            "error_type": report["error_type"],
            "report_path": report_path_value,
            "active_database_sha256": previous_status.get("active_database_sha256"),
            "active_database_path": previous_status.get("active_database_path"),
            "manifest_path": previous_status.get("manifest_path"),
        },
        allow_overwrite=True,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 2


def _run_fred_refresh(args: argparse.Namespace, *, root: Path, key: str) -> int:
    try:
        config = load_yaml(root / "configs" / "macro.yaml")
        allowed = tuple(map(str, config["series_whitelist"]))
        formal_rate_config = config.get("formal_rate_rag", {})
        requested_series = tuple(map(str, formal_rate_config.get("series_ids", allowed)))
        active_database = resolve_project_path(args.database, root=root)
        seed_database = resolve_project_path(
            args.seed_database or args.database,
            root=root,
        )
        if not seed_database.is_file():
            raise FileNotFoundError(
                "FRED refresh requires an existing seed evidence database"
            )
        fred_run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        snapshots = fetch_fred_initial_releases(
            api_key=key,
            series_ids=requested_series,
            allowed_series_ids=allowed,
            observation_start=date.fromisoformat(args.observation_start),
            observation_end=date.fromisoformat(args.observation_end),
            realtime_start=date.fromisoformat(args.realtime_start),
            realtime_end=date.fromisoformat(args.realtime_end),
            paired_realtime_windows=True,
        )
        report = persist_fred_initial_release_refresh(
            snapshots,
            evidence_store=MacroEvidenceStore(seed_database),
            destination_root=args.output_root,
            root=root,
            run_id=fred_run_id,
            seed_database_path=seed_database,
        )
        refresh_database = resolve_project_path(str(report["database"]["path"]), root=root)
        active_tmp = active_database.with_name(f".{active_database.name}.{fred_run_id}.tmp")
        shutil.copy2(refresh_database, active_tmp)
        os.replace(active_tmp, active_database)
        report["published"] = True
        report["active_database"] = {
            "path": active_database.relative_to(root).as_posix(),
            "bytes": active_database.stat().st_size,
            "sha256": sha256_file(active_database),
        }
        fred_manifest_path = resolve_project_path(args.output_root, root=root) / str(report["run_id"]) / "manifest.json"
        write_json_atomic(
            active_database.parent / "fred_refresh_status.json",
            {
                "schema_version": "aupilot.fred_refresh_status.v1",
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "fetch_succeeded": True,
                "published": True,
                "manifest_path": fred_manifest_path.relative_to(root).as_posix(),
                "active_database_sha256": sha256_file(active_database),
                "active_database_path": active_database.relative_to(root).as_posix(),
                "secret_recorded": False,
            },
            allow_overwrite=True,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        return _write_fred_failure(args, root=root, error=error)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aupilot-rag")
    commands = parser.add_subparsers(dest="command", required=True)

    golden = commands.add_parser("evaluate-golden")
    golden.add_argument("--fixture", default="configs/macro_rag_golden.yaml")
    golden.add_argument("--output", required=True)

    shadow = commands.add_parser("refresh-official-documents")
    shadow.add_argument("--database", default="data/macro/shadow/evidence.sqlite")
    shadow.add_argument("--report", required=True)
    shadow.add_argument("--clean", action="store_true")
    shadow.add_argument("--disabled-source", action="append", default=[])

    calendar = commands.add_parser("refresh-calendar")
    calendar.add_argument("--snapshot", default="storage/macro/calendar_snapshot.json")
    calendar.add_argument("--report", required=True)
    calendar.add_argument("--config", default="configs/macro.yaml")
    calendar.add_argument("--disabled-source", action="append", default=[])

    fred = commands.add_parser("refresh-fred")
    fred.add_argument("--database", default="data/macro/formal/evidence.sqlite")
    fred.add_argument("--seed-database", default=None)
    fred.add_argument("--output-root", default="data/macro/formal/fred")
    fred.add_argument("--observation-start", default="2010-01-01")
    fred.add_argument("--observation-end", default=_utc_today())
    fred.add_argument("--realtime-start", default="2010-01-01")
    fred.add_argument("--realtime-end", default=_utc_today())
    fred.add_argument("--report", default=None)

    assess = commands.add_parser("assess")
    assess.add_argument("--database", default="data/macro/formal/evidence.sqlite")
    assess.add_argument("--calendar-snapshot", required=True)
    assess.add_argument("--config", default="configs/macro.yaml")
    assess.add_argument("--as-of", default=None)
    assess.add_argument("--replay-only", action="store_true")
    assess.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = project_root()
    if args.command == "evaluate-golden":
        report = evaluate_macro_rag_golden(
            fixture_path=args.fixture,
            output=args.output,
            root=root,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 2

    if args.command == "refresh-official-documents":
        config = load_yaml(root / "configs" / "macro.yaml")
        database = resolve_project_path(args.database, root=root)
        report_path = resolve_project_path(args.report, root=root)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
        run_root = database.parent / "refresh" / "documents" / run_id
        run_root.mkdir(parents=True, exist_ok=False)
        staging_database = run_root / "evidence.sqlite"
        seed_database_record: dict[str, object] | None = None
        if database.is_file() and not args.clean:
            _copy_sqlite_database(database, staging_database)
            seed_database_record = {
                "path": database.relative_to(root).as_posix(),
                "bytes": database.stat().st_size,
                "sha256": sha256_file(database),
            }
        store = MacroEvidenceStore(staging_database)
        fetched: list[dict[str, object]] = []
        failures: list[dict[str, str]] = []
        disabled_sources = set(args.disabled_source)
        for source in config["shadow_source_pages"]:
            source_id = str(source["source"])
            if source_id in disabled_sources:
                continue
            try:
                document = fetch_official_document(**source)
                fetched.append(
                    {
                        "doc_id": document.doc_id,
                        "event_type": document.event_type,
                        "canonical_url": document.canonical_url,
                        "content_sha256": document.content_sha256,
                        "retrieved_at_utc": document.retrieved_at_utc.isoformat(),
                        "eligible_from_utc": document.eligible_from_utc.isoformat(),
                        "replay_eligible": document.replay_eligible,
                        "inserted": store.ingest(document),
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "source": str(source["source"]),
                        "event_type": str(source["event_type"]),
                        "error_type": type(error).__name__,
                    }
                )
        batches = default_coordinator(config, disabled_sources=disabled_sources).fetch(
            RetrievalRequest(datetime.now(UTC))
        )
        provider_records: list[dict[str, object]] = []
        required_ingestion_failed = False
        for batch in batches:
            record = batch.manifest_record()
            if batch.fetch_succeeded:
                try:
                    inserted_documents = sum(store.ingest(value) for value in batch.documents)
                    inserted_observations = store.ingest_observations(batch.observations)
                    inserted_claims = store.ingest_claims(batch.claims)
                    record.update(
                        {
                            "inserted_documents": inserted_documents,
                            "inserted_observations": inserted_observations,
                            "inserted_claims": inserted_claims,
                        }
                    )
                except Exception as error:
                    record.update(
                        {
                            "fetch_succeeded": False,
                            "error_type": type(error).__name__,
                            "reason_code": "PROVIDER_BATCH_INGESTION_FAILED",
                        }
                    )
                    if batch.required:
                        required_ingestion_failed = True
            provider_records.append(record)
        usable_provider_records = [
            record
            for record in provider_records
            if record.get("fetch_succeeded") is True
            and int(record.get("claim_count", 0)) > 0
        ]
        fetch_succeeded = not required_ingestion_failed and bool(usable_provider_records)
        source_degraded = len(usable_provider_records) < len(provider_records)
        database_record: dict[str, object] | None = None
        active_database_record: dict[str, object] | None = None
        published = False
        if staging_database.is_file():
            _seal_sqlite(staging_database)
            database_record = {
                "path": staging_database.relative_to(root).as_posix(),
                "bytes": staging_database.stat().st_size,
                "sha256": sha256_file(staging_database),
            }
        if fetch_succeeded and database_record is not None:
            active_tmp = database.with_name(f".{database.name}.{run_id}.tmp")
            shutil.copy2(staging_database, active_tmp)
            os.replace(active_tmp, database)
            active_database_record = {
                "path": database.relative_to(root).as_posix(),
                "bytes": database.stat().st_size,
                "sha256": sha256_file(database),
            }
            published = True
        report = {
            "schema_version": "aupilot.official_macro_multisource_refresh.v2",
            "operation": "OFFICIAL_MACRO_MULTISOURCE_REFRESH",
            "run_id": run_id,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "mode": "MULTISOURCE_REQUIRED_WITH_OPTIONAL_FRED",
            "parser_version": PARSER_VERSION,
            "fetch_succeeded": fetch_succeeded,
            "published": published,
            "fetched": fetched,
            "legacy_shadow_failures": failures,
            "provider_batches": provider_records,
            "source_degraded": source_degraded,
            "available_provider_ids": [
                str(record["provider_id"]) for record in usable_provider_records
            ],
            "document_count": store.document_count(),
            "observation_count": store.observation_count(),
            "claim_count": store.claim_count(),
            "seed_database": seed_database_record,
            "run_database": database_record,
            "active_database": active_database_record,
            "trade_permission": False,
            "technical_model_input_allowed": False,
            "decision_engine_input_allowed": False,
            "historical_profit_tuning_allowed": False,
            "secret_recorded": False,
        }
        write_json_atomic(report_path, report)
        manifest = {
            "schema_version": "aupilot.official_macro_multisource_refresh_manifest.v2",
            "operation": "OFFICIAL_MACRO_MULTISOURCE_REFRESH",
            "run_id": run_id,
            "report": {
                "path": report_path.relative_to(root).as_posix(),
                "bytes": report_path.stat().st_size,
                "sha256": sha256_file(report_path),
            },
            "run_database": database_record,
            "active_database": active_database_record,
            "fetch_succeeded": fetch_succeeded,
            "published": published,
            "document_count": store.document_count(),
            "observation_count": store.observation_count(),
            "claim_count": store.claim_count(),
            "provider_batches": provider_records,
            "source_degraded": source_degraded,
            "available_provider_ids": [
                str(record["provider_id"]) for record in usable_provider_records
            ],
            "seed_database": seed_database_record,
            "historical_profit_tuning_allowed": False,
            "trade_permission": False,
            "technical_model_input_allowed": False,
            "decision_engine_input_allowed": False,
            "secret_recorded": False,
        }
        manifest_path = run_root / "manifest.json"
        write_json_atomic(manifest_path, manifest)
        status_path = database.parent / "evidence_refresh_status.json"
        write_json_atomic(
            status_path,
            {
                "schema_version": "aupilot.official_macro_evidence_status.v1",
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "run_id": run_id,
                "fetch_succeeded": fetch_succeeded,
                "published": published,
                "report_path": report_path.relative_to(root).as_posix(),
                "manifest_path": manifest_path.relative_to(root).as_posix(),
                "active_database_sha256": (
                    None if active_database_record is None else active_database_record["sha256"]
                ),
                "provider_batches": provider_records,
                "source_degraded": source_degraded,
                "available_provider_ids": [
                    str(record["provider_id"]) for record in usable_provider_records
                ],
            },
            allow_overwrite=True,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if fetch_succeeded else 2

    if args.command == "refresh-calendar":
        config = load_yaml(resolve_project_path(args.config, root=root))
        calendar_config = config["calendar_sources"]
        report_path = resolve_project_path(args.report, root=root)
        if report_path.exists():
            raise FileExistsError(f"Refusing to overwrite immutable refresh report: {report_path}")
        disabled_sources = set(args.disabled_source)

        def calendar_source_id(source: dict) -> str:
            value = str(source.get("source", ""))
            if value.startswith("bls"):
                return "bls"
            if value.startswith("bea"):
                return "bea"
            return value

        enabled_calendar_sources = [
            source for source in calendar_config["sources"]
            if calendar_source_id(source) not in disabled_sources
        ]
        snapshot = fetch_official_calendar(
            sources=enabled_calendar_sources,
            freshness_hours=int(calendar_config["freshness_hours"]),
        )
        snapshot_path = resolve_project_path(args.snapshot, root=root)
        publication = publish_calendar_snapshot(
            snapshot,
            storage_root=snapshot_path.parent,
            active_filename=snapshot_path.name,
        )
        report = {
            "operation": "OFFICIAL_MACRO_CALENDAR_REFRESH",
            "snapshot": snapshot,
            "publication": publication,
            "source_degraded": bool(snapshot.get("source_degraded", False)),
            "trade_permission": False,
            "technical_model_input_allowed": False,
            "decision_engine_input_allowed": False,
            "historical_profit_tuning_allowed": False,
        }
        write_json_atomic(report_path, report)
        snapshot_status_path = snapshot_path.parent / "calendar_refresh_status.json"
        failure_reason_codes = sorted(
            {
                str(item["reason_code"])
                for item in snapshot.get("failures", [])
                if isinstance(item, dict) and item.get("reason_code")
            }
        )
        write_json_atomic(
            snapshot_status_path,
            {
                "schema_version": "aupilot.macro_calendar_status.v1",
                "updated_at_utc": datetime.now(UTC).isoformat(),
                "fetch_succeeded": snapshot["fetch_succeeded"] is True,
                "published": True,
                "active_snapshot_sha256": publication["active_snapshot"]["sha256"],
                "manifest_path": publication["manifest_path"],
                "report_path": report_path.relative_to(root).as_posix(),
                "failure_reason_codes": failure_reason_codes,
                "source_degraded": bool(snapshot.get("source_degraded", False)),
                "reason_code": (None if not failure_reason_codes else failure_reason_codes[0]),
            },
            allow_overwrite=True,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if snapshot["fetch_succeeded"] is True else 2

    if args.command == "refresh-fred":
        key = os.environ.get("FRED_API_KEY", "")
        if not key.strip():
            previous_status: dict[str, object] = {}
            previous_path = resolve_project_path(args.database, root=root).parent / "fred_refresh_status.json"
            try:
                previous_payload = json.loads(previous_path.read_text(encoding="utf-8"))
                if isinstance(previous_payload, dict):
                    previous_status = previous_payload
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
            report = {
                "schema_version": "aupilot.fred_refresh.v1",
                "operation": "FRED_ALFRED_INITIAL_RELEASE_REFRESH",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "fetch_succeeded": False,
                "published": False,
                "passed": False,
                "reason_code": "FRED_API_KEY_MISSING",
                "secret_recorded": False,
                "trade_permission": False,
                "technical_model_input_allowed": False,
                "decision_engine_input_allowed": False,
                "historical_profit_tuning_allowed": False,
            }
            if args.report:
                report_path = resolve_project_path(args.report, root=root)
                write_json_atomic(report_path, report, allow_overwrite=True)
            write_json_atomic(
                resolve_project_path(args.database, root=root).parent / "fred_refresh_status.json",
                {
                    "schema_version": "aupilot.fred_refresh_status.v1",
                    "updated_at_utc": report["completed_at_utc"],
                    "fetch_succeeded": False,
                    "published": False,
                    "reason_code": "FRED_API_KEY_MISSING",
                    "report_path": None if not args.report else report_path.relative_to(root).as_posix(),
                    # Keep the last successful immutable optional artifact
                    # pointer while reporting the current fetch failure.
                    "active_database_sha256": previous_status.get("active_database_sha256"),
                    "active_database_path": previous_status.get("active_database_path"),
                    "manifest_path": previous_status.get("manifest_path"),
                },
                allow_overwrite=True,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 2
        return _run_fred_refresh(args, root=root, key=key)

    if args.command == "assess":
        assessment = assess_macro_risk(
            database_path=resolve_project_path(args.database, root=root),
            calendar_snapshot_path=resolve_project_path(
                args.calendar_snapshot,
                root=root,
            ),
            config_path=resolve_project_path(args.config, root=root),
            decision_as_of_utc=_as_of(args.as_of),
            replay_only=bool(args.replay_only),
        )
        output = resolve_project_path(args.output, root=root)
        write_json_atomic(output, assessment.model_dump(mode="json"))
        print(assessment.model_dump_json(indent=2))
        return 0 if assessment.assessment_supported else 2
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
