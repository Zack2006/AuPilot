"""Validated registry for the active formal dual-model manifest and artifacts."""

from __future__ import annotations

import json

from pydantic import ValidationError

from aupilot.core.hashing import sha256_file

from backend.app.core.config import Settings
from backend.app.schemas.formal_model import FormalModelManifest


class FormalModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self.model_dir = settings.model_dir
        self.manifest_path = settings.model_dir / "technical_model_manifest.json"

    def validated_manifest(self) -> tuple[FormalModelManifest | None, str | None, str | None]:
        if not self.manifest_path.is_file():
            return None, None, "FORMAL_MODEL_MANIFEST_MISSING"
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            manifest = FormalModelManifest.model_validate(payload)
            root = self.model_dir.resolve()
            for artifact in manifest.artifacts:
                candidate = (root / artifact.relative_path).resolve()
                if root not in candidate.parents or not candidate.is_file():
                    raise ValueError("Formal model artifact path is invalid")
                if sha256_file(candidate) != artifact.sha256:
                    raise ValueError("Formal model artifact hash mismatch")
        except (OSError, ValueError, json.JSONDecodeError, ValidationError):
            return None, None, "FORMAL_MODEL_MANIFEST_INVALID"
        return manifest, sha256_file(self.manifest_path), None
