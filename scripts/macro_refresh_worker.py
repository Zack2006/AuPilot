"""Scheduled official macro refresh worker; it never serves API requests."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
STORAGE = ROOT / "storage" / "macro"
REPORTS = STORAGE / "reports"
LOCK = STORAGE / "refresh" / ".macro_refresh.lock"
WORKER_STATUS = STORAGE / "worker_status.json"
CALENDAR_SNAPSHOT = STORAGE / "calendar_snapshot.json"
STALE_LOCK_GRACE_SECONDS = 60
DEFAULT_REFRESH_INTERVAL_SECONDS = 4 * 60 * 60
DEFAULT_RETRY_INTERVAL_SECONDS = 10 * 60
DEFAULT_REFRESH_AHEAD_SECONDS = 30 * 60
MACRO_SOURCE_IDS = (
    "federal_reserve", "new_york_fed", "us_treasury", "bls", "bea", "fred",
)
sys.path.insert(0, str(ROOT / "src"))

from aupilot.core.manifest import write_json_atomic  # noqa: E402


def _local_source_records() -> dict[str, dict[str, object]]:
    config_dir = Path(os.environ.get("LOCAL_CONFIG_DIR", ROOT / ".local" / "AurumPilot"))
    path = config_dir / "secrets.json"
    if not path.is_file():
        return {source_id: {"enabled": True} for source_id in MACRO_SOURCE_IDS}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources = payload["sources"]
        if not isinstance(sources, dict):
            raise ValueError("invalid sources")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {source_id: {"enabled": False} for source_id in MACRO_SOURCE_IDS}
    return {
        source_id: dict(sources.get(source_id, {"enabled": True}))
        if isinstance(sources.get(source_id, {"enabled": True}), dict)
        else {"enabled": False}
        for source_id in MACRO_SOURCE_IDS
    }


def _source_enabled(source_id: str) -> bool:
    return bool(_local_source_records().get(source_id, {}).get("enabled", True))


def _source_credential(source_id: str, legacy_environment_name: str) -> str:
    if not _source_enabled(source_id):
        return ""
    value = _local_source_records().get(source_id, {}).get("api_key")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return os.environ.get(legacy_environment_name, "").strip()


def _disabled_source_arguments() -> list[str]:
    arguments: list[str] = []
    for source_id in MACRO_SOURCE_IDS:
        if source_id != "fred" and not _source_enabled(source_id):
            arguments.extend(["--disabled-source", source_id])
    return arguments


def _run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]


def _lock_pid(path: Path) -> int | None:
    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        if not first_line.startswith("pid="):
            return None
        pid = int(first_line[4:])
        return pid if pid > 0 else None
    except (OSError, IndexError, ValueError):
        return None


def _lock_host(path: Path) -> str | None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("host="):
                value = line[5:].strip()
                return value or None
    except OSError:
        return None
    return None


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _acquire_lock() -> int | None:
    """Acquire the lock and recover only an old lock from a dead process."""

    LOCK.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            return os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                age_seconds = max(0.0, time.time() - LOCK.stat().st_mtime)
            except OSError:
                return None
            pid = _lock_pid(LOCK)
            host = _lock_host(LOCK)
            different_runtime = host is not None and host != socket.gethostname()
            # Docker can restart a container in place, preserving both its
            # hostname and PID 1 while the previous process is already gone.
            same_pid_after_restart = (
                pid == os.getpid() and age_seconds >= STALE_LOCK_GRACE_SECONDS
            )
            if (
                not different_runtime
                and not same_pid_after_restart
                and ((pid is not None and _process_is_alive(pid)) or age_seconds < STALE_LOCK_GRACE_SECONDS)
            ):
                return None
            try:
                LOCK.unlink()
            except (FileNotFoundError, OSError):
                return None
    return None


def _write_worker_status(state: str, **fields: object) -> None:
    write_json_atomic(
        WORKER_STATUS,
        {
            "schema_version": "aupilot.macro_refresh_worker_status.v1",
            "state": state,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            **fields,
            "informational_only": True,
            "trade_permission": False,
            "technical_model_input_allowed": False,
            "decision_engine_input_allowed": False,
            "historical_profit_tuning_allowed": False,
            "secret_recorded": False,
        },
        allow_overwrite=True,
    )


def _calendar_refresh_due(
    *,
    now_utc: datetime | None = None,
    refresh_ahead_seconds: int = DEFAULT_REFRESH_AHEAD_SECONDS,
) -> bool:
    """Return true when the published calendar is missing or near expiry."""

    now = now_utc or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now_utc must be timezone-aware")
    try:
        payload = json.loads(CALENDAR_SNAPSHOT.read_text(encoding="utf-8"))
        raw_fresh_until = str(payload["fresh_until_utc"])
        fresh_until = datetime.fromisoformat(raw_fresh_until.replace("Z", "+00:00"))
        if fresh_until.tzinfo is None or fresh_until.utcoffset() is None:
            raise ValueError("fresh_until_utc must be timezone-aware")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return True
    margin = timedelta(seconds=max(refresh_ahead_seconds, 0))
    return fresh_until <= now + margin


def _sleep_with_heartbeat(
    seconds: int,
    *,
    pid: int,
    interval_seconds: int,
    retry_seconds: int,
    refresh_ahead_seconds: int,
    last_run_fields: dict[str, object] | None = None,
    wake_condition: Callable[[], bool] | None = None,
) -> None:
    deadline = time.monotonic() + seconds
    while True:
        if wake_condition is not None and wake_condition():
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(30, remaining))
        _write_worker_status(
            "running",
            pid=pid,
            interval_seconds=interval_seconds,
            retry_seconds=retry_seconds,
            refresh_ahead_seconds=refresh_ahead_seconds,
            heartbeat_at_utc=datetime.now(UTC).isoformat(),
            **(last_run_fields or {}),
        )


def _next_wait_seconds(
    report: dict[str, object],
    *,
    interval_seconds: int,
    retry_seconds: int,
) -> int:
    return interval_seconds if report.get("succeeded") is True else retry_seconds


def _fred_refresh_arguments() -> list[str]:
    """Backfill once, then keep the active FRED store fresh incrementally."""

    arguments = [
        "refresh-fred", "--database", "storage/macro/fred_evidence.sqlite",
        "--output-root", "storage/macro/refresh/fred",
    ]
    fred_database = STORAGE / "fred_evidence.sqlite"
    seed_database = fred_database if fred_database.is_file() else STORAGE / "evidence.sqlite"
    try:
        seed_argument = seed_database.relative_to(ROOT).as_posix()
    except ValueError:
        seed_argument = str(seed_database)
    arguments.extend([
        "--seed-database",
        seed_argument,
    ])
    incremental_days = max(int(os.environ.get("FRED_INCREMENTAL_DAYS", "0")), 0)
    bootstrap_days = max(int(os.environ.get("FRED_BOOTSTRAP_DAYS", "0")), 0)
    try:
        status = json.loads((STORAGE / "fred_refresh_status.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        status = {}
    published = status.get("fetch_succeeded") is True and status.get("published") is True
    window_days = incremental_days if published else bootstrap_days
    if window_days <= 0:
        return arguments
    start = date.today() - timedelta(days=window_days)
    arguments.extend([
        "--observation-start", start.isoformat(),
        "--realtime-start", start.isoformat(),
    ])
    return arguments


def _run_command(
    run_id: str,
    name: str,
    arguments: list[str],
    *,
    required: bool,
) -> dict[str, object]:
    report_path = REPORTS / f"{name}_{run_id}.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    fred_key = _source_credential("fred", "FRED_API_KEY")
    bls_key = _source_credential("bls", "BLS_API_KEY")
    if fred_key:
        environment["FRED_API_KEY"] = fred_key
    else:
        environment.pop("FRED_API_KEY", None)
    if bls_key:
        environment["BLS_API_KEY"] = bls_key
    else:
        environment.pop("BLS_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "-m", "aupilot.rag_cli", *arguments, "--report", str(report_path.relative_to(ROOT))],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("MACRO_REFRESH_COMMAND_TIMEOUT_SECONDS", "180")),
        check=False,
    )
    stdout, stderr = result.stdout, result.stderr
    for secret in (fred_key, bls_key):
        if secret:
            stdout = stdout.replace(secret, "[REDACTED]")
            stderr = stderr.replace(secret, "[REDACTED]")
    command_report: dict[str, object] = {}
    try:
        loaded_report = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(loaded_report, dict):
            command_report = loaded_report
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return {
        "name": name,
        "required": required,
        "return_code": result.returncode,
        "succeeded": result.returncode == 0,
        "report_path": report_path.relative_to(ROOT).as_posix(),
        "stdout_sha256": hashlib.sha256(stdout.encode("utf-8")).hexdigest().upper(),
        "stderr_sha256": hashlib.sha256(stderr.encode("utf-8")).hexdigest().upper(),
        "source_degraded": bool(command_report.get("source_degraded", False)),
        "secret_recorded": False,
    }


def run_once() -> tuple[int, dict[str, object]]:
    run_id = _run_id()
    REPORTS.mkdir(parents=True, exist_ok=True)
    STORAGE.mkdir(parents=True, exist_ok=True)
    source_filter_arguments = _disabled_source_arguments()
    commands = [
        ("calendar_refresh", [
            "refresh-calendar", "--snapshot", "storage/macro/calendar_snapshot.json",
            "--config", "configs/macro.yaml", *source_filter_arguments,
        ], True),
        ("official_refresh", [
            "refresh-official-documents", "--database", "storage/macro/evidence.sqlite",
            *source_filter_arguments,
        ], True),
    ]
    # FRED is an optional cross-check. Do not turn an intentionally absent
    # credential into a failed worker run when the official document path is
    # healthy.
    if _source_credential("fred", "FRED_API_KEY"):
        commands.append(("fred_refresh", _fred_refresh_arguments(), False))
    results = []
    for name, arguments, required in commands:
        try:
            results.append(
                _run_command(run_id, name, arguments, required=required)
            )
        except Exception as error:
            results.append({
                "name": name, "required": required,
                "return_code": None, "succeeded": False,
                "error_type": type(error).__name__, "secret_recorded": False,
            })
    required_failures = [
        str(item["name"])
        for item in results
        if item.get("required") is True and item.get("succeeded") is not True
    ]
    optional_failures = [
        str(item["name"])
        for item in results
        if item.get("required") is False and item.get("succeeded") is not True
    ]
    report = {
        "schema_version": "aupilot.macro_refresh_worker.v2",
        "operation": "SCHEDULED_OFFICIAL_MACRO_REFRESH",
        "run_id": run_id,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "commands": results,
        "succeeded": not required_failures,
        "all_commands_succeeded": not required_failures and not optional_failures,
        "required_failures": required_failures,
        "optional_failures": optional_failures,
        "source_degraded": bool(
            required_failures
            or optional_failures
            or any(item.get("source_degraded") is True for item in results)
        ),
        "informational_only": True,
        "trade_permission": False,
        "technical_model_input_allowed": False,
        "decision_engine_input_allowed": False,
        "historical_profit_tuning_allowed": False,
        "secret_recorded": False,
    }
    write_json_atomic(REPORTS / f"worker_{run_id}.json", report)
    return (0 if report["succeeded"] else 2), report


def main() -> int:
    descriptor = _acquire_lock()
    if descriptor is None:
        print(json.dumps({"succeeded": False, "reason_code": "MACRO_REFRESH_ALREADY_RUNNING"}))
        return 3
    last_run_fields: dict[str, object] = {}
    try:
        try:
            os.write(
                descriptor,
                f"pid={os.getpid()}\nhost={socket.gethostname()}\n".encode("utf-8"),
            )
        finally:
            os.close(descriptor)
        interval_seconds = max(
            int(os.environ.get("MACRO_REFRESH_INTERVAL_SECONDS", str(DEFAULT_REFRESH_INTERVAL_SECONDS))),
            300,
        )
        retry_seconds = max(
            int(os.environ.get("MACRO_REFRESH_RETRY_SECONDS", str(DEFAULT_RETRY_INTERVAL_SECONDS))),
            60,
        )
        refresh_ahead_seconds = max(
            int(os.environ.get("MACRO_REFRESH_AHEAD_SECONDS", str(DEFAULT_REFRESH_AHEAD_SECONDS))),
            0,
        )
        _write_worker_status(
            "running",
            pid=os.getpid(),
            started_at_utc=datetime.now(UTC).isoformat(),
            interval_seconds=interval_seconds,
            retry_seconds=retry_seconds,
            refresh_ahead_seconds=refresh_ahead_seconds,
        )
        while True:
            _write_worker_status(
                "running",
                pid=os.getpid(),
                interval_seconds=interval_seconds,
                retry_seconds=retry_seconds,
                refresh_ahead_seconds=refresh_ahead_seconds,
                heartbeat_at_utc=datetime.now(UTC).isoformat(),
            )
            code, report = run_once()
            _write_worker_status(
                "running",
                pid=os.getpid(),
                interval_seconds=interval_seconds,
                retry_seconds=retry_seconds,
                refresh_ahead_seconds=refresh_ahead_seconds,
                heartbeat_at_utc=datetime.now(UTC).isoformat(),
                last_run_id=report["run_id"],
                last_run_completed_at_utc=report["completed_at_utc"],
                last_run_succeeded=report["succeeded"],
            )
            last_run_fields = {
                "last_run_id": report["run_id"],
                "last_run_completed_at_utc": report["completed_at_utc"],
                "last_run_succeeded": report["succeeded"],
            }
            print(json.dumps(report, ensure_ascii=False))
            if os.environ.get("MACRO_REFRESH_RUN_ONCE", "false").lower() == "true":
                return code
            wait_seconds = _next_wait_seconds(
                report,
                interval_seconds=interval_seconds,
                retry_seconds=retry_seconds,
            )
            _sleep_with_heartbeat(
                wait_seconds,
                pid=os.getpid(),
                interval_seconds=interval_seconds,
                retry_seconds=retry_seconds,
                refresh_ahead_seconds=refresh_ahead_seconds,
                last_run_fields=last_run_fields,
                wake_condition=lambda: _calendar_refresh_due(
                    refresh_ahead_seconds=refresh_ahead_seconds
                ),
            )
    finally:
        try:
            _write_worker_status(
                "stopped",
                pid=os.getpid(),
                stopped_at_utc=datetime.now(UTC).isoformat(),
                **last_run_fields,
            )
        except OSError:
            pass
        try:
            LOCK.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
