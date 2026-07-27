"""Tamper-evident append-only JSONL repository for formal predictions."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import Lock

from aupilot.core.hashing import canonical_json_sha256, sha256_file

from backend.app.schemas.formal_model import FormalPredictionCreate, FormalPredictionRecord


GENESIS_SHA256 = "0" * 64


class FormalPredictionRepository:
    _append_lock = Lock()

    def __init__(self, path: Path) -> None:
        self.path = path
        self.checkpoint_path = path.with_suffix(path.suffix + ".head.json")

    @staticmethod
    def _record_sha256(record: FormalPredictionRecord) -> str:
        payload = record.model_dump(mode="json", exclude={"record_sha256"})
        return canonical_json_sha256(payload)

    def read_all(self) -> list[FormalPredictionRecord]:
        if not self.path.is_file():
            if self.checkpoint_path.exists():
                raise ValueError("Formal prediction checkpoint exists without its log")
            return []
        if not self.checkpoint_path.is_file():
            raise ValueError("Formal prediction checkpoint is missing")
        records: list[FormalPredictionRecord] = []
        previous_sha = GENESIS_SHA256
        seen_ids: set[str] = set()
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ValueError(f"Formal prediction log contains a blank line at {line_number}")
                try:
                    record = FormalPredictionRecord.model_validate_json(line)
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Formal prediction log record {line_number} is invalid") from exc
                if record.prediction_id in seen_ids:
                    raise ValueError("Formal prediction IDs must be unique")
                if record.previous_record_sha256 != previous_sha:
                    raise ValueError("Formal prediction hash chain is broken")
                if record.record_sha256 != self._record_sha256(record):
                    raise ValueError("Formal prediction record hash is invalid")
                if records and record.generated_at_utc < records[-1].generated_at_utc:
                    raise ValueError("Formal prediction generation timestamps must be nondecreasing")
                seen_ids.add(record.prediction_id)
                records.append(record)
                previous_sha = record.record_sha256
        try:
            checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Formal prediction checkpoint is invalid") from exc
        expected = {
            "schema_version": "aurumpilot.formal_prediction_head.v1",
            "record_count": len(records),
            "head_record_sha256": previous_sha,
            "log_file_sha256": sha256_file(self.path),
        }
        if checkpoint != expected:
            raise ValueError("Formal prediction checkpoint does not match the log")
        return records

    def _write_checkpoint(self, records: list[FormalPredictionRecord]) -> None:
        checkpoint = {
            "schema_version": "aurumpilot.formal_prediction_head.v1",
            "record_count": len(records),
            "head_record_sha256": records[-1].record_sha256 if records else GENESIS_SHA256,
            "log_file_sha256": sha256_file(self.path),
        }
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=self.path.parent, suffix=".head.tmp", delete=False
            ) as handle:
                json.dump(checkpoint, handle, ensure_ascii=True, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
                temporary_path = Path(handle.name)
            os.replace(temporary_path, self.checkpoint_path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def append(self, prediction: FormalPredictionCreate) -> FormalPredictionRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._append_lock:
            existing = self.read_all()
            if any(item.prediction_id == prediction.prediction_id for item in existing):
                raise ValueError("Formal prediction ID already exists")
            if existing and prediction.generated_at_utc < existing[-1].generated_at_utc:
                raise ValueError("Formal prediction generation timestamps must be nondecreasing")
            same_date = [item for item in existing if item.prediction_for_date == prediction.prediction_for_date]
            if same_date and prediction.supersedes_prediction_id != same_date[-1].prediction_id:
                raise ValueError("A same-date revision must supersede the latest prediction")
            if prediction.supersedes_prediction_id is not None:
                superseded = next(
                    (item for item in existing if item.prediction_id == prediction.supersedes_prediction_id), None
                )
                if superseded is None:
                    raise ValueError("supersedes_prediction_id does not exist")
                if superseded.prediction_for_date != prediction.prediction_for_date:
                    raise ValueError("A revision cannot supersede a different prediction date")
            previous_sha = existing[-1].record_sha256 if existing else GENESIS_SHA256
            unsigned = FormalPredictionRecord(
                **prediction.model_dump(),
                previous_record_sha256=previous_sha,
                record_sha256=GENESIS_SHA256,
            )
            record = unsigned.model_copy(update={"record_sha256": self._record_sha256(unsigned)})
            serialized = json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(serialized + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._write_checkpoint([*existing, record])
            return record
