from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app.core.exceptions import DataCorruptionError


class JsonRepository:
    """Small UTF-8 JSON repository with atomic replacement on writes."""

    def __init__(self, path: Path, default: Any) -> None:
        self.path = path
        self.default = default
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.write(default)

    def read(self) -> Any:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise DataCorruptionError(f"Cannot read valid JSON from {self.path}: {exc}") from exc

    def write(self, value: Any) -> None:
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        temporary.replace(self.path)
