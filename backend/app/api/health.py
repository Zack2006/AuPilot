import hashlib
import json

from fastapi import APIRouter

from backend.app.api.dependencies import macro_service, technical_issuance_service
from backend.app.core.config import get_settings

router = APIRouter(tags=["health"])


def _status(path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return {"fetch_succeeded": False, "reason_code": "STATUS_INVALID"}


def _sha256(path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    macro_dir = settings.storage_dir / "macro"
    calendar = _status(macro_dir / "calendar_snapshot.json")
    calendar_status = _status(macro_dir / "calendar_refresh_status.json")
    evidence = _status(macro_dir / "evidence_refresh_status.json")
    fred = _status(macro_dir / "fred_refresh_status.json")
    worker = _status(macro_dir / "worker_status.json")
    macro_runtime = macro_service().status()
    return {
        "status": "ok",
        "prediction_service": technical_issuance_service().__class__.__name__,
        "formal_technical_path": True,
        "technical_runtime_mode": "MN18_PN02_HASH_GATED",
        "offline_ready": True,
        "macro_rag": {
            "calendar_fetch_succeeded": (
                None
                if calendar is None and calendar_status is None
                else (
                    calendar_status.get("fetch_succeeded") is True
                    if calendar_status is not None
                    else calendar.get("fetch_succeeded") is True
                )
            ),
            "calendar_snapshot_sha256": _sha256(macro_dir / "calendar_snapshot.json"),
            "evidence_refresh_succeeded": None if evidence is None else evidence.get("fetch_succeeded") is True,
            "evidence_database_sha256": _sha256(macro_dir / "evidence.sqlite"),
            "fred_refresh_succeeded": None if fred is None else fred.get("fetch_succeeded") is True,
            "assessment_supported": macro_runtime.assessment_supported,
            "status": macro_runtime.status.lower(),
            "refresh_worker": None if worker is None else {
                "state": worker.get("state"),
                "updated_at_utc": worker.get("updated_at_utc"),
                "last_run_succeeded": worker.get("last_run_succeeded"),
                "last_run_completed_at_utc": worker.get("last_run_completed_at_utc"),
            },
        },
    }
