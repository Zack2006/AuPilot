from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def load_yaml(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {source}")
    return value


def load_project_config(filename: str) -> dict[str, Any]:
    return load_yaml(project_root() / "configs" / filename)


def resolve_project_path(path: str | Path, *, root: Path | None = None) -> Path:
    """Resolve a path and reject access outside the AuPilot workspace root."""
    project = (project_root() if root is None else root).resolve()
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else project / candidate).resolve()
    try:
        resolved.relative_to(project)
    except ValueError as error:
        raise PermissionError(f"Path is outside the AuPilot workspace: {resolved}") from error
    return resolved
