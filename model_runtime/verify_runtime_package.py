from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    manifest_path = root / "PACKAGE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = set(manifest["files"])
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    errors: list[str] = []
    for relative, record in manifest["files"].items():
        path = root / relative
        if not path.is_file():
            errors.append(f"MISSING:{relative}")
            continue
        if path.stat().st_size != int(record["bytes"]):
            errors.append(f"BYTES:{relative}")
        if sha256(path) != str(record["sha256"]).upper():
            errors.append(f"SHA256:{relative}")
    for relative in sorted(expected - actual):
        errors.append(f"MISSING:{relative}")
    for relative in sorted(actual - expected):
        errors.append(f"EXTRA:{relative}")
    if errors:
        raise SystemExit("MODEL_RUNTIME_VERIFY_FAILED:" + ",".join(errors[:20]))
    print(
        json.dumps(
            {
                "passed": True,
                "files_verified": len(expected),
                "errors": 0,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
