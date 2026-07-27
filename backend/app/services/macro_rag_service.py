"""Adapter from FastAPI to the independent official macro RAG core."""

from __future__ import annotations

import json
import hashlib
from datetime import UTC, datetime, timedelta

from aupilot.core.enums import MacroRiskLevel
from aupilot.core.hashing import canonical_json_sha256, sha256_file
from aupilot.macro_gate.runtime import assess_macro_risk
from aupilot.macro_gate.schemas import MacroAssessment

from backend.app.core.config import PROJECT_ROOT, Settings
from backend.app.schemas.macro import (
    MacroCoverageSlot,
    MacroEvidenceResponse,
    MacroEventsResponse,
    MacroProviderStatus,
    MacroRiskResponse,
    MacroStatusResponse,
)
from backend.app.services.local_settings_service import LocalSettingsService
from backend.app.services.macro_interpretation_service import build_macro_interpretations


PRIMARY_MACRO_SOURCE_IDS = (
    "federal_reserve", "new_york_fed", "us_treasury", "bls", "bea",
)
PROVIDER_SOURCE_IDS = {
    "federal_reserve_fomc_release": "federal_reserve",
    "new_york_fed_effr": "new_york_fed",
    "us_treasury_curve": "us_treasury",
    "bls_public_data_api": "bls",
    "bea_personal_income_outlays": "bea",
}


class MacroRAGService:
    """Read-only official macro risk service with deterministic degraded output."""

    def __init__(self, settings: Settings) -> None:
        self.storage_dir = settings.storage_dir / "macro"
        self.database_path = self.storage_dir / "evidence.sqlite"
        self.fred_database_path = self.storage_dir / "fred_evidence.sqlite"
        self.evidence_status_path = self.storage_dir / "evidence_refresh_status.json"
        self.fred_status_path = self.storage_dir / "fred_refresh_status.json"
        self.calendar_snapshot_path = self.storage_dir / "calendar_snapshot.json"
        self.calendar_status_path = self.storage_dir / "calendar_refresh_status.json"
        self.config_path = PROJECT_ROOT / "configs" / "macro.yaml"
        self.local_settings = LocalSettingsService(settings)

    def _disabled_primary_sources(self) -> list[str]:
        return [
            source_id for source_id in PRIMARY_MACRO_SOURCE_IDS
            if not self.local_settings.is_enabled(source_id)
        ]

    def _available_evidence_provider_ids(self, status: dict | None) -> frozenset[str]:
        if status is None or not isinstance(status.get("provider_batches"), list):
            return frozenset()
        available = set()
        for item in status["provider_batches"]:
            if not isinstance(item, dict):
                continue
            provider_id = str(item.get("provider_id") or "")
            source_id = PROVIDER_SOURCE_IDS.get(provider_id)
            if (
                source_id is not None
                and self.local_settings.is_enabled(source_id)
                and item.get("fetch_succeeded") is True
                and int(item.get("claim_count") or 0) > 0
            ):
                available.add(provider_id)
        return frozenset(available)

    def _source_state_reason_codes(self, status: dict | None) -> list[str]:
        reasons = [
            f"MACRO_SOURCE_DISABLED_{source_id.upper()}"
            for source_id in self._disabled_primary_sources()
        ]
        records = {
            str(item.get("provider_id")): item
            for item in ([] if status is None else status.get("provider_batches", []))
            if isinstance(item, dict) and item.get("provider_id")
        }
        for provider_id, source_id in PROVIDER_SOURCE_IDS.items():
            if not self.local_settings.is_enabled(source_id):
                continue
            record = records.get(provider_id)
            if (
                record is None
                or record.get("fetch_succeeded") is not True
                or int(record.get("claim_count") or 0) < 1
            ):
                reasons.append(f"MACRO_SOURCE_UNAVAILABLE_{source_id.upper()}")
        return list(dict.fromkeys(reasons))

    @staticmethod
    def _fallback(
        as_of: datetime,
        error: Exception | None = None,
        reason_codes: list[str] | None = None,
    ) -> MacroRiskResponse:
        reason_codes = list(reason_codes or ["RAG_INTERNAL_FAILURE"])
        reason_codes.extend(("ALL_OFFICIAL_SOURCES_UNAVAILABLE", "ASSESSMENT_UNAVAILABLE"))
        if error is not None:
            reason_codes.insert(1, type(error).__name__)
        return MacroRiskResponse(
            risk_level=MacroRiskLevel.CAUTION,
            risk_score=MacroRiskLevel.CAUTION.score,
            decision_as_of_utc=as_of,
            reason_codes=list(dict.fromkeys(reason_codes)),
            news_summary=[
                "Official macro evidence is unavailable; a reliable assessment cannot be produced."
            ],
            citations=[],
            source_degraded=True,
            assessment_supported=False,
        )

    @staticmethod
    def _assessment_id(response: MacroRiskResponse) -> str:
        payload = response.model_dump(mode="json", exclude={"assessment_id", "decision_as_of_utc"})
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:24]

    @classmethod
    def _response(cls, assessment: MacroAssessment) -> MacroRiskResponse:
        response = MacroRiskResponse.model_validate(assessment.model_dump(mode="json"))
        response = response.model_copy(
            update={
                "interpretations": build_macro_interpretations(
                    response.claims,
                    response.summary_facts,
                )
            }
        )
        return response.model_copy(update={"assessment_id": cls._assessment_id(response)})

    @staticmethod
    def _manifest_path(status_path, value: object):
        if not isinstance(value, str) or not value.strip():
            return None
        status_root = status_path.parent.resolve()
        normalized = value.replace("\\", "/")
        candidates = [status_path.parent / value]
        storage_prefix = "storage/macro/"
        if normalized.startswith(storage_prefix):
            candidates.insert(0, status_path.parent / normalized[len(storage_prefix) :])
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved.is_relative_to(status_root) and resolved.is_file():
                return resolved
        return None

    @staticmethod
    def _manifest_artifact_sha256(manifest: dict, artifact_path) -> str | None:
        if artifact_path.name == "calendar_snapshot.json":
            record = manifest.get("snapshot")
        else:
            record = manifest.get("active_database") or manifest.get("database")
        if not isinstance(record, dict):
            return None
        value = record.get("sha256")
        return value.upper() if isinstance(value, str) and len(value) == 64 else None

    @staticmethod
    def _read_status(path) -> dict | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _verify_database_status(
        self,
        status_path,
        *,
        artifact_path,
        missing_reason: str,
    ) -> tuple[dict | None, str | None]:
        """Verify one publisher against its physically isolated SQLite artifact."""

        status = self._read_status(status_path)
        if status is None or status.get("fetch_succeeded") is not True or status.get("published") is not True:
            return status, missing_reason
        expected_hash = status.get("active_database_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            return status, "MACRO_EVIDENCE_DATABASE_HASH_MISSING"
        if not artifact_path.is_file():
            return status, f"{missing_reason}_DATABASE_MISSING"
        actual_hash = sha256_file(artifact_path)
        if expected_hash.upper() != actual_hash:
            return status, f"{missing_reason}_DATABASE_HASH_MISMATCH"
        manifest_path = self._manifest_path(status_path, status.get("manifest_path"))
        if manifest_path is None:
            return status, f"{missing_reason}_MANIFEST_MISSING"
        manifest = self._read_status(manifest_path)
        if manifest is None or manifest.get("fetch_succeeded") is not True:
            return status, f"{missing_reason}_MANIFEST_INVALID"
        if self._manifest_artifact_sha256(manifest, artifact_path) != actual_hash:
            return status, f"{missing_reason}_MANIFEST_ARTIFACT_HASH_MISMATCH"
        return status, None

    def _calendar_integrity_error(self) -> str | None:
        if not self.calendar_status_path.is_file() or not self.calendar_snapshot_path.is_file():
            return "MACRO_CALENDAR_REFRESH_FAILED"
        try:
            status = json.loads(self.calendar_status_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return "MACRO_CALENDAR_REFRESH_FAILED"
        if (
            not isinstance(status, dict)
            or status.get("fetch_succeeded") is not True
            or status.get("published") is not True
        ):
            return "MACRO_CALENDAR_REFRESH_FAILED"
        expected_hash = status.get("active_snapshot_sha256")
        actual_hash = sha256_file(self.calendar_snapshot_path)
        if not isinstance(expected_hash, str) or expected_hash.upper() != actual_hash:
            return "MACRO_CALENDAR_SNAPSHOT_HASH_MISSING" if not isinstance(expected_hash, str) else "MACRO_CALENDAR_SNAPSHOT_HASH_MISMATCH"
        manifest_path = self._manifest_path(self.calendar_status_path, status.get("manifest_path"))
        if manifest_path is None:
            return "MACRO_CALENDAR_REFRESH_FAILED_MANIFEST_MISSING"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return "MACRO_CALENDAR_REFRESH_FAILED_MANIFEST_INVALID"
        if not isinstance(manifest, dict) or manifest.get("fetch_succeeded") is not True:
            return "MACRO_CALENDAR_REFRESH_FAILED_MANIFEST_INVALID"
        if self._manifest_artifact_sha256(manifest, self.calendar_snapshot_path) != actual_hash:
            return "MACRO_CALENDAR_REFRESH_FAILED_MANIFEST_ARTIFACT_HASH_MISMATCH"
        return None

    def _calendar_integrity_verified(self) -> bool:
        return self._calendar_integrity_error() is None

    def assess(self, as_of: datetime | None = None) -> MacroRiskResponse:
        cutoff = datetime.now(UTC) if as_of is None else as_of
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("as_of must include an explicit timezone")
        cutoff = cutoff.astimezone(UTC)
        disabled_sources = self._disabled_primary_sources()
        if len(disabled_sources) == len(PRIMARY_MACRO_SOURCE_IDS):
            return self._fallback(
                cutoff,
                reason_codes=[
                    "ALL_PRIMARY_MACRO_SOURCES_DISABLED",
                    *[
                        f"MACRO_SOURCE_DISABLED_{source_id.upper()}"
                        for source_id in disabled_sources
                    ],
                ],
            )
        try:
            calendar_error = self._calendar_integrity_error()
            if calendar_error not in {None, "MACRO_CALENDAR_REFRESH_FAILED"}:
                return self._fallback(cutoff, reason_codes=[calendar_error])
            evidence_status, evidence_error = self._verify_database_status(
                self.evidence_status_path,
                artifact_path=self.database_path,
                missing_reason="MACRO_EVIDENCE_REFRESH_FAILED",
            )
            if evidence_error not in {None, "MACRO_EVIDENCE_REFRESH_FAILED"}:
                return self._fallback(cutoff, reason_codes=[evidence_error])
            _fred_status, fred_error = self._verify_database_status(
                self.fred_status_path,
                artifact_path=self.fred_database_path,
                missing_reason="FRED_REFRESH_FAILED",
            )
            fred_enabled = self.local_settings.is_enabled("fred")
            fred_available = fred_enabled and fred_error is None
            enabled_source_ids = frozenset(
                source_id
                for source_id in PRIMARY_MACRO_SOURCE_IDS
                if self.local_settings.is_enabled(source_id)
            )
            available_provider_ids = (
                self._available_evidence_provider_ids(evidence_status)
                if evidence_error is None
                else frozenset()
            )
            assessment = assess_macro_risk(
                database_path=self.database_path,
                calendar_snapshot_path=self.calendar_snapshot_path,
                config_path=self.config_path,
                decision_as_of_utc=cutoff,
                enabled_source_ids=enabled_source_ids,
                allowed_provider_ids=available_provider_ids,
                read_stored_evidence=evidence_error is None,
                calendar_integrity_error=calendar_error,
            )
        except Exception as error:
            return self._fallback(cutoff, error)
        source_state_reasons = self._source_state_reason_codes(evidence_status)
        reason_codes = [*assessment.reason_codes, *source_state_reasons]
        if calendar_error is not None:
            reason_codes.append(calendar_error)
        if evidence_error is not None:
            reason_codes.append(evidence_error)
        if not fred_available:
            reason_codes.append(
                "OPTIONAL_SOURCE_UNAVAILABLE_FRED"
                if fred_enabled
                else "OPTIONAL_SOURCE_DISABLED_FRED"
            )
        degraded = bool(
            assessment.source_degraded
            or source_state_reasons
            or calendar_error
            or evidence_error
            or not fred_available
        )
        if assessment.assessment_supported and degraded:
            reason_codes.append("PARTIAL_OFFICIAL_SOURCE_COVERAGE")
        if not assessment.assessment_supported:
            reason_codes.extend(
                ("ALL_OFFICIAL_SOURCES_UNAVAILABLE", "ASSESSMENT_UNAVAILABLE")
            )
        assessment = assessment.model_copy(
            update={
                "reason_codes": tuple(dict.fromkeys(reason_codes)),
                "source_degraded": degraded,
            }
        )
        return self._response(assessment)

    def status(self) -> MacroStatusResponse:
        calendar_error = self._calendar_integrity_error()
        evidence_status, evidence_error = self._verify_database_status(
            self.evidence_status_path,
            artifact_path=self.database_path,
            missing_reason="MACRO_EVIDENCE_REFRESH_FAILED",
        )
        fred_status, fred_error = self._verify_database_status(
            self.fred_status_path,
            artifact_path=self.fred_database_path,
            missing_reason="FRED_REFRESH_FAILED",
        )
        fred_enabled = self.local_settings.is_enabled("fred")
        fred_available = fred_enabled and fred_error is None
        assessment = self.assess()
        coverage = [
            MacroCoverageSlot.model_validate(item.model_dump(mode="json"))
            for item in assessment.coverage
        ]
        coverage_complete = bool(coverage) and all(
            not item.required or item.status in {"COVERED", "DEGRADED"}
            for item in coverage
        )
        if not assessment.assessment_supported:
            service_status = "UNAVAILABLE"
        elif (
            calendar_error is not None
            or evidence_error is not None
            or not fred_available
            or assessment.source_degraded
        ):
            service_status = "DEGRADED"
        else:
            service_status = "OK"

        def _updated(status: dict | None) -> datetime | None:
            value = None if status is None else status.get("updated_at_utc")
            try:
                parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                return None
            return parsed.astimezone(UTC) if parsed.tzinfo and parsed.utcoffset() is not None else None

        successful_updates = [
            value for value in (
                _updated(self._read_status(self.calendar_status_path)),
                _updated(evidence_status),
                _updated(fred_status) if fred_available else None,
            ) if value is not None
        ]
        providers = [
            MacroProviderStatus(
                provider_id="federal_reserve_calendar",
                required=False,
                status="AVAILABLE" if calendar_error is None else "INVALID",
                reason_code=calendar_error,
                updated_at_utc=_updated(self._read_status(self.calendar_status_path)),
            )
        ]
        provider_batches = (
            []
            if evidence_status is None
            else evidence_status.get("provider_batches", [])
        )
        if isinstance(provider_batches, list) and provider_batches:
            for item in provider_batches:
                if not isinstance(item, dict) or not item.get("provider_id"):
                    continue
                if str(item["provider_id"]).startswith("cme_"):
                    continue
                provider_id = str(item["provider_id"])
                source_id = PROVIDER_SOURCE_IDS.get(provider_id)
                enabled = source_id is None or self.local_settings.is_enabled(source_id)
                available = (
                    enabled
                    and item.get("fetch_succeeded") is True
                    and int(item.get("claim_count") or 0) > 0
                )
                providers.append(
                    MacroProviderStatus(
                        provider_id=provider_id,
                        required=False,
                        status=(
                            "AVAILABLE"
                            if available
                            else "UNAVAILABLE"
                        ),
                        reason_code=(
                            None
                            if available
                            else "SOURCE_DISABLED_BY_LOCAL_SETTINGS"
                            if not enabled
                            else str(item.get("reason_code") or "PROVIDER_UNAVAILABLE")
                        ),
                        updated_at_utc=_updated(evidence_status),
                    )
                )
        else:
            providers.append(
                MacroProviderStatus(
                    provider_id="official_documents",
                    required=False,
                    status="AVAILABLE" if evidence_error is None else "INVALID",
                    reason_code=evidence_error,
                    updated_at_utc=_updated(evidence_status),
                )
            )
        providers.append(
            MacroProviderStatus(
                provider_id="optional_fred",
                required=False,
                status="AVAILABLE" if fred_available else "UNAVAILABLE",
                reason_code=(
                    None if fred_available else
                    "SOURCE_DISABLED_BY_LOCAL_SETTINGS" if not fred_enabled else
                    (fred_error or "FRED_DATABASE_HASH_UNVERIFIED")
                ),
                updated_at_utc=_updated(fred_status),
            )
        )
        return MacroStatusResponse(
            status=service_status,
            assessment_supported=assessment.assessment_supported,
            coverage_complete=coverage_complete,
            providers=providers,
            coverage=coverage,
            last_success_at_utc=max(successful_updates) if successful_updates else None,
        )

    def evidence(self, assessment_id: str) -> MacroEvidenceResponse:
        assessment = self.assess()
        response = self._response(assessment)
        current_id = self._assessment_id(response)
        if assessment_id != current_id:
            raise ValueError("assessment_id does not match the latest immutable assessment")
        payload = {
            "assessment_id": current_id,
            "decision_as_of_utc": response.decision_as_of_utc.isoformat(),
            "citations": [item.model_dump(mode="json") for item in response.citations],
            "claims": [item.model_dump(mode="json") for item in response.claims],
            "coverage": [item.model_dump(mode="json") for item in response.coverage],
            "summary_facts": [item.model_dump(mode="json") for item in response.summary_facts],
            "interpretations": [item.model_dump(mode="json") for item in response.interpretations],
            "score_components": (
                response.score_components.model_dump(mode="json")
                if response.score_components is not None
                else None
            ),
            "assessment_supported": response.assessment_supported,
        }
        return MacroEvidenceResponse(
            content_hash=canonical_json_sha256(payload),
            **payload,
        )

    def events(self, days: int | None = None) -> MacroEventsResponse:
        if not self._calendar_integrity_verified():
            return MacroEventsResponse(items=[], fetch_succeeded=False)
        try:
            payload = json.loads(self.calendar_snapshot_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
                raise ValueError("calendar snapshot shape is invalid")
            items = payload["events"]
            source_map = {
                "federal_reserve": "federal_reserve",
                "bls_cpi_schedule": "bls",
                "bls_employment_schedule": "bls",
                "bea_schedule": "bea",
            }
            items = [
                item for item in items
                if self.local_settings.is_enabled(source_map.get(str(item.get("source")), str(item.get("source"))))
            ]
            enabled_calendar_source_available = bool(items)
            if days is not None:
                if days < 1 or days > 90:
                    raise ValueError("days must be between 1 and 90")
                now = datetime.now(UTC)
                horizon = now + timedelta(days=days)
                items = [
                    item for item in items
                    if now <= datetime.fromisoformat(
                        str(item["scheduled_release_at_utc"]).replace("Z", "+00:00")
                    ).astimezone(UTC) <= horizon
                ]
            return MacroEventsResponse(
                items=items,
                fetch_succeeded=(
                    payload.get("fetch_succeeded") is True
                    and enabled_calendar_source_available
                ),
                retrieved_at_utc=payload.get("retrieved_at_utc"),
                fresh_until_utc=payload.get("fresh_until_utc"),
                snapshot_sha256=sha256_file(self.calendar_snapshot_path),
            )
        except Exception:
            return MacroEventsResponse(items=[], fetch_succeeded=False)


class MacroRAGServiceFactory:
    @staticmethod
    def create(settings: Settings) -> MacroRAGService:
        return MacroRAGService(settings)
