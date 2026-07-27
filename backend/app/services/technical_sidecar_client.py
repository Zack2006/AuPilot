"""Strict HTTP adapter for the private hash-gated MN18 + PN02 sidecar."""

from __future__ import annotations

import os
from typing import Any

import httpx
from pydantic import ValidationError

from backend.app.core.exceptions import TechnicalSidecarUnavailableError
from backend.app.schemas.technical import (
    EXPECTED_COMPONENT_HASHES,
    MN18_MODEL_ID,
    MN18_MODEL_VERSION,
    PN02_MODEL_ID,
    PN02_MODEL_VERSION,
    TECHNICAL_SCHEMA_VERSION,
    TechnicalEvaluationRequest,
    TechnicalOutlook,
)

EXPECTED_DUAL_PACKAGE_FILE_COUNT = int(
    os.getenv("AUPILOT_DUAL_PACKAGE_FILE_COUNT", "342")
)


class TechnicalSidecarClient:
    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
    ) -> dict[str, Any]:
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.request(method, path, json=json_body)
        except httpx.HTTPError as exc:
            raise TechnicalSidecarUnavailableError(
                "TECHNICAL_SIDECAR_UNREACHABLE"
            ) from exc
        if response.status_code != 200:
            try:
                detail = str(response.json().get("detail", ""))[:300]
            except (AttributeError, ValueError):
                detail = ""
            raise TechnicalSidecarUnavailableError(
                f"TECHNICAL_SIDECAR_HTTP_{response.status_code}:{detail}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise TechnicalSidecarUnavailableError(
                "TECHNICAL_SIDECAR_RESPONSE_NOT_JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise TechnicalSidecarUnavailableError(
                "TECHNICAL_SIDECAR_RESPONSE_INVALID"
            )
        return payload

    @staticmethod
    def _validate_components(payload: dict[str, Any]) -> None:
        components = payload.get("components")
        if not isinstance(components, dict):
            raise TechnicalSidecarUnavailableError(
                "TECHNICAL_SIDECAR_COMPONENT_STATUS_MISSING"
            )
        turning = components.get("turning")
        price = components.get("conditional_ohlc")
        if (
            not isinstance(turning, dict)
            or turning.get("loaded") is not True
            or turning.get("model_id") != MN18_MODEL_ID
            or turning.get("model_version") != MN18_MODEL_VERSION
            or turning.get("bundle_manifest_sha256", "").upper()
            != EXPECTED_COMPONENT_HASHES["mn18_bundle_manifest_sha256"]
            or turning.get("joblib_sha256", "").upper()
            != EXPECTED_COMPONENT_HASHES["mn18_joblib_sha256"]
            or turning.get("controls_trading") is not True
            or not isinstance(price, dict)
            or price.get("loaded") is not True
            or price.get("model_id") != PN02_MODEL_ID
            or price.get("model_version") != PN02_MODEL_VERSION
            or price.get("bundle_sha256", "").upper()
            != EXPECTED_COMPONENT_HASHES["pn02_bundle_sha256"]
            or price.get("controls_trading") is not False
            or price.get("advisory_only") is not True
        ):
            raise TechnicalSidecarUnavailableError(
                "TECHNICAL_SIDECAR_COMPONENT_STATUS_INVALID"
            )

    def health(self) -> dict[str, Any]:
        payload = self._request("GET", "/api/v1/technical/health")
        required = {
            "service": "AuPilot",
            "status": "READY_DEVELOPMENT_REQUIRES_FORWARD_SHADOW",
            "schema_version": TECHNICAL_SCHEMA_VERSION,
            "bundle_verified": True,
            "exactly_21_slots": True,
            "seven_class_probabilities": True,
            "mn18_is_sole_action_authority": True,
            "pn02_controls_trading": False,
            "rag_controls_trading": False,
            "automatic_execution": False,
            "broker_or_order_endpoint": False,
        }
        if any(payload.get(key) != value for key, value in required.items()):
            raise TechnicalSidecarUnavailableError(
                "TECHNICAL_SIDECAR_HEALTH_CONTRACT_INVALID"
            )
        verification = payload.get("package_verification") or {}
        if (
            verification.get("files_verified")
            != EXPECTED_DUAL_PACKAGE_FILE_COUNT
            or verification.get("error_count") != 0
            or str(verification.get("manifest_sha256", "")).upper()
            != EXPECTED_COMPONENT_HASHES["dual_package_manifest_sha256"]
        ):
            raise TechnicalSidecarUnavailableError(
                "TECHNICAL_SIDECAR_PACKAGE_VERIFICATION_INVALID"
            )
        self._validate_components(payload)
        return payload

    def status(self) -> dict[str, Any]:
        payload = self._request("GET", "/api/v1/technical/status")
        required = {
            "schema_version": TECHNICAL_SCHEMA_VERSION,
            "contract_id": "MN18_PN02_DUAL_MODEL_UTC_DAILY_V1",
            "history_start": "2010-06-07",
            "history_ohlc_only": True,
            "technical_features_internal": True,
            "calendar_semantics": "FUTURE_VALID_CANONICAL_UTC_DAILY_BUCKETS",
            "not_comex_sessions": True,
            "h1_permission": "TOP_ONLY",
            "h2_permission": "BOTTOM_ONLY",
            "h3_to_h21_permission": "DISPLAY_ONLY",
        }
        if any(payload.get(key) != value for key, value in required.items()):
            raise TechnicalSidecarUnavailableError(
                "TECHNICAL_SIDECAR_STATUS_CONTRACT_INVALID"
            )
        if payload.get("join_keys") != ["horizon_index", "target_bucket"]:
            raise TechnicalSidecarUnavailableError(
                "TECHNICAL_SIDECAR_JOIN_KEYS_INVALID"
            )
        self._validate_components(payload)
        return payload

    def evaluate(
        self,
        request: TechnicalEvaluationRequest,
    ) -> tuple[TechnicalOutlook, dict[str, Any]]:
        raw = self._request(
            "POST",
            "/api/v1/internal/technical/evaluate-latest-daily",
            json_body=request.model_dump(mode="json"),
        )
        try:
            validated = TechnicalOutlook.model_validate(raw)
        except ValidationError as exc:
            raise TechnicalSidecarUnavailableError(
                "TECHNICAL_SIDECAR_OUTPUT_CONTRACT_INVALID:"
                + str(exc)[:500]
            ) from exc
        return validated, raw
