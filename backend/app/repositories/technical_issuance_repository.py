"""Fresh SQLite persistence for MN18/PN02 issuances and user-confirmed fills."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from aupilot.core.hashing import canonical_json_sha256

from backend.app.schemas.technical import (
    TechnicalFillRecord,
    TechnicalFillRequest,
    TechnicalIssuanceRecord,
    TechnicalOutlook,
    TechnicalRuntimeState,
)


class TechnicalIssuanceRepository:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=30.0)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=FULL;

                CREATE TABLE IF NOT EXISTS technical_issuances (
                    issuance_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('SUCCESS', 'FAILED')),
                    issuance_kind TEXT NOT NULL CHECK (
                        issuance_kind IN (
                            'PRE_FORWARD_PRODUCT_PREVIEW',
                            'COMPLIANT_FORWARD',
                            'FAILED_ATTEMPT'
                        )
                    ),
                    created_at_utc TEXT NOT NULL,
                    source_bucket TEXT,
                    source_bucket_end_utc TEXT,
                    issued_at_utc TEXT,
                    input_history_start TEXT,
                    input_history_end TEXT,
                    input_rows INTEGER,
                    current_gold_weight REAL NOT NULL
                        CHECK (current_gold_weight BETWEEN 0.5 AND 1.0),
                    outstanding_top_inventory_pp REAL NOT NULL
                        CHECK (outstanding_top_inventory_pp BETWEEN 0 AND 50),
                    dual_package_manifest_hash TEXT NOT NULL,
                    mn18_manifest_hash TEXT NOT NULL,
                    mn18_joblib_hash TEXT NOT NULL,
                    pn02_bundle_hash TEXT NOT NULL,
                    market_data_sha256 TEXT,
                    input_request_sha256 TEXT,
                    output_json TEXT,
                    output_sha256 TEXT,
                    error_reason TEXT,
                    CHECK (
                        (status = 'SUCCESS' AND output_json IS NOT NULL
                         AND output_sha256 IS NOT NULL AND error_reason IS NULL)
                        OR
                        (status = 'FAILED' AND output_json IS NULL
                         AND output_sha256 IS NULL AND error_reason IS NOT NULL)
                    )
                );

                CREATE UNIQUE INDEX IF NOT EXISTS ux_mn18_pn02_success_idempotency
                    ON technical_issuances(idempotency_key)
                    WHERE status = 'SUCCESS';
                CREATE INDEX IF NOT EXISTS ix_mn18_pn02_issuance_source_bucket
                    ON technical_issuances(source_bucket, issued_at_utc);

                CREATE TABLE IF NOT EXISTS technical_composition_audits (
                    audit_id TEXT PRIMARY KEY,
                    issuance_id TEXT NOT NULL UNIQUE,
                    created_at_utc TEXT NOT NULL,
                    audit_json TEXT NOT NULL,
                    audit_sha256 TEXT NOT NULL,
                    FOREIGN KEY (issuance_id)
                        REFERENCES technical_issuances(issuance_id)
                );

                CREATE TABLE IF NOT EXISTS technical_scheduled_requests (
                    model_version TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    issuance_id TEXT NOT NULL,
                    source_bucket TEXT NOT NULL,
                    target_bucket TEXT NOT NULL,
                    horizon_index INTEGER NOT NULL CHECK (
                        horizon_index IN (1, 2)
                    ),
                    side TEXT NOT NULL CHECK (side IN ('TOP', 'BOTTOM')),
                    requested_delta_pp REAL NOT NULL,
                    initial_status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    PRIMARY KEY (model_version, request_id),
                    FOREIGN KEY (issuance_id)
                        REFERENCES technical_issuances(issuance_id)
                );
                CREATE INDEX IF NOT EXISTS ix_mn18_request_target
                    ON technical_scheduled_requests(
                        model_version, target_bucket, horizon_index
                    );

                CREATE TABLE IF NOT EXISTS technical_runtime_state (
                    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                    current_gold_weight REAL NOT NULL
                        CHECK (current_gold_weight BETWEEN 0.5 AND 1.0),
                    outstanding_top_inventory_pp REAL NOT NULL
                        CHECK (outstanding_top_inventory_pp BETWEEN 0 AND 50),
                    revision INTEGER NOT NULL CHECK (revision >= 0),
                    latest_issuance_id TEXT,
                    updated_at_utc TEXT NOT NULL,
                    FOREIGN KEY (latest_issuance_id)
                        REFERENCES technical_issuances(issuance_id)
                );

                CREATE TABLE IF NOT EXISTS technical_fills (
                    fill_id TEXT PRIMARY KEY,
                    issuance_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    request_sha256 TEXT NOT NULL,
                    revision_before INTEGER NOT NULL,
                    revision_after INTEGER NOT NULL,
                    weight_before REAL NOT NULL,
                    weight_after REAL NOT NULL,
                    inventory_before_pp REAL NOT NULL,
                    inventory_after_pp REAL NOT NULL,
                    actual_delta_pp REAL NOT NULL,
                    filled_at_utc TEXT NOT NULL,
                    fill_price REAL NOT NULL CHECK (fill_price > 0),
                    recorded_at_utc TEXT NOT NULL,
                    FOREIGN KEY (issuance_id)
                        REFERENCES technical_issuances(issuance_id)
                );
                """
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO technical_runtime_state (
                    singleton_id, current_gold_weight,
                    outstanding_top_inventory_pp, revision,
                    latest_issuance_id, updated_at_utc
                ) VALUES (
                    1, 1.0, 0.0, 0, NULL, '1970-01-01T00:00:00+00:00'
                )
                """
            )
            connection.commit()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> TechnicalIssuanceRecord:
        output = json.loads(row["output_json"]) if row["output_json"] else None
        record = TechnicalIssuanceRecord(
            issuance_id=row["issuance_id"],
            idempotency_key=row["idempotency_key"],
            status=row["status"],
            issuance_kind=row["issuance_kind"],
            created_at_utc=row["created_at_utc"],
            source_bucket=row["source_bucket"],
            source_bucket_end_utc=row["source_bucket_end_utc"],
            issued_at_utc=row["issued_at_utc"],
            input_history_start=row["input_history_start"],
            input_history_end=row["input_history_end"],
            input_rows=row["input_rows"],
            current_gold_weight=row["current_gold_weight"],
            outstanding_top_inventory_pp=row[
                "outstanding_top_inventory_pp"
            ],
            dual_package_manifest_hash=row["dual_package_manifest_hash"],
            mn18_manifest_hash=row["mn18_manifest_hash"],
            mn18_joblib_hash=row["mn18_joblib_hash"],
            pn02_bundle_hash=row["pn02_bundle_hash"],
            market_data_sha256=row["market_data_sha256"],
            input_request_sha256=row["input_request_sha256"],
            output=output,
            output_sha256=row["output_sha256"],
            error_reason=row["error_reason"],
        )
        if record.status == "SUCCESS":
            TechnicalOutlook.model_validate(record.output)
            if canonical_json_sha256(record.output) != record.output_sha256:
                raise ValueError("Technical issuance output SHA-256 mismatch")
        return record

    @staticmethod
    def _row_to_fill(row: sqlite3.Row) -> TechnicalFillRecord:
        columns = set(row.keys())
        return TechnicalFillRecord(
            fill_id=row["fill_id"],
            issuance_id=row["issuance_id"],
            request_id=(
                row["request_id"] if "request_id" in columns else None
            ),
            side=row["side"] if "side" in columns else None,
            action=row["action"] if "action" in columns else None,
            idempotency_key=row["idempotency_key"],
            revision_before=row["revision_before"],
            revision_after=row["revision_after"],
            weight_before=row["weight_before"],
            weight_after=row["weight_after"],
            inventory_before_pp=row["inventory_before_pp"],
            inventory_after_pp=row["inventory_after_pp"],
            actual_delta_pp=row["actual_delta_pp"],
            filled_at_utc=row["filled_at_utc"],
            fill_price=row["fill_price"],
            recorded_at_utc=row["recorded_at_utc"],
        )

    @staticmethod
    def _state_from_row(row: sqlite3.Row) -> TechnicalRuntimeState:
        return TechnicalRuntimeState(
            current_gold_weight=row["current_gold_weight"],
            outstanding_top_inventory_pp=row[
                "outstanding_top_inventory_pp"
            ],
            revision=row["revision"],
            latest_issuance_id=row["latest_issuance_id"],
            updated_at_utc=row["updated_at_utc"],
        )

    def current_state(self) -> TechnicalRuntimeState:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT current_gold_weight, outstanding_top_inventory_pp,
                       revision, latest_issuance_id, updated_at_utc
                FROM technical_runtime_state
                WHERE singleton_id = 1
                """
            ).fetchone()
        if row is None:
            raise ValueError("Technical runtime state is missing")
        return self._state_from_row(row)

    def current_gold_weight(self) -> float:
        return self.current_state().current_gold_weight

    def outstanding_top_inventory_pp(self) -> float:
        return self.current_state().outstanding_top_inventory_pp

    def find_success(
        self, idempotency_key: str
    ) -> TechnicalIssuanceRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM technical_issuances
                WHERE idempotency_key = ? AND status = 'SUCCESS'
                """,
                (idempotency_key,),
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def latest_success(self) -> TechnicalIssuanceRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT issuance.*
                FROM technical_runtime_state AS state
                JOIN technical_issuances AS issuance
                  ON issuance.issuance_id = state.latest_issuance_id
                WHERE state.singleton_id = 1
                  AND issuance.status = 'SUCCESS'
                """
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def latest_failure(self) -> TechnicalIssuanceRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM technical_issuances
                WHERE status = 'FAILED'
                ORDER BY created_at_utc DESC, issuance_id DESC
                LIMIT 1
                """
            ).fetchone()
        return self._row_to_record(row) if row is not None else None

    def list_records(
        self, *, include_failures: bool = True
    ) -> list[TechnicalIssuanceRecord]:
        query = "SELECT * FROM technical_issuances"
        if not include_failures:
            query += " WHERE status = 'SUCCESS'"
        query += " ORDER BY created_at_utc, issuance_id"
        with self._connection() as connection:
            rows = connection.execute(query).fetchall()
        return [self._row_to_record(row) for row in rows]

    def list_fills(self) -> list[TechnicalFillRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM technical_fills
                ORDER BY revision_after, fill_id
                """
            ).fetchall()
        return [self._row_to_fill(row) for row in rows]

    def composition_audit(self, issuance_id: str) -> dict | None:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT audit_id, issuance_id, created_at_utc,
                       audit_json, audit_sha256
                FROM technical_composition_audits
                WHERE issuance_id = ?
                """,
                (issuance_id,),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["audit_json"])
        if canonical_json_sha256(payload) != row["audit_sha256"]:
            raise ValueError("Technical composition audit SHA-256 mismatch")
        if payload.get("issuance_id") != issuance_id:
            raise ValueError("Technical composition audit issuance mismatch")
        return {
            "audit_id": row["audit_id"],
            "issuance_id": row["issuance_id"],
            "created_at_utc": row["created_at_utc"],
            "audit": payload,
            "audit_sha256": row["audit_sha256"],
        }

    def append_composition_audit(
        self,
        issuance_id: str,
        audit: dict,
    ) -> tuple[dict, bool]:
        audit_sha256 = canonical_json_sha256(audit)
        audit_id = f"audit_{audit_sha256[:20].lower()}"
        audit_json = json.dumps(
            audit,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT audit_id, issuance_id, created_at_utc,
                       audit_json, audit_sha256
                FROM technical_composition_audits
                WHERE issuance_id = ?
                """,
                (issuance_id,),
            ).fetchone()
            if existing is not None:
                connection.commit()
                return self.composition_audit(issuance_id), False
            issuance = connection.execute(
                """
                SELECT issuance_id FROM technical_issuances
                WHERE issuance_id = ? AND status = 'SUCCESS'
                """,
                (issuance_id,),
            ).fetchone()
            if issuance is None:
                raise ValueError("Technical audit issuance not found")
            connection.execute(
                """
                INSERT INTO technical_composition_audits (
                    audit_id, issuance_id, created_at_utc,
                    audit_json, audit_sha256
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    audit_id,
                    issuance_id,
                    str(audit["created_at_utc"]),
                    audit_json,
                    audit_sha256,
                ),
            )
            connection.commit()
        return self.composition_audit(issuance_id), True

    def append_scheduled_requests(
        self,
        *,
        issuance: TechnicalIssuanceRecord,
        model_version: str,
        requests: list[dict],
    ) -> int:
        """Persist native MN18 requests immutably and detect replay conflicts."""

        if issuance.source_bucket is None:
            raise ValueError("SCHEDULED_REQUEST_SOURCE_BUCKET_MISSING")
        inserted = 0
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for request in requests:
                payload_sha = canonical_json_sha256(request)
                payload_json = json.dumps(
                    request,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                key = (model_version, str(request["request_id"]))
                existing = connection.execute(
                    """
                    SELECT payload_sha256
                    FROM technical_scheduled_requests
                    WHERE model_version = ? AND request_id = ?
                    """,
                    key,
                ).fetchone()
                if existing is not None:
                    if existing["payload_sha256"] != payload_sha:
                        event_table = connection.execute(
                            """
                            SELECT 1 FROM sqlite_master
                            WHERE type = 'table'
                              AND name = 'technical_request_events'
                            """
                        ).fetchone()
                        if event_table is not None:
                            event = {
                                "receiver_policy": "MN18_RECEIVER_POLICY_V1",
                                "model_version": model_version,
                                "request_id": key[1],
                                "state_scope": "RECEIVER",
                                "status": (
                                    "FAIL_CLOSED_REQUEST_ID_COLLISION"
                                ),
                                "payload": {
                                    "existing_payload_sha256": existing[
                                        "payload_sha256"
                                    ],
                                    "incoming_payload_sha256": payload_sha,
                                },
                                "created_at_utc": datetime.now(
                                    UTC
                                ).isoformat(),
                            }
                            event_sha = canonical_json_sha256(event)
                            connection.execute(
                                """
                                INSERT OR IGNORE INTO
                                    technical_request_events (
                                        event_id, model_version, request_id,
                                        state_scope, status, event_json,
                                        event_sha256, created_at_utc
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    f"evt_{event_sha[:24].lower()}",
                                    model_version,
                                    key[1],
                                    "RECEIVER",
                                    (
                                        "FAIL_CLOSED_REQUEST_ID_COLLISION"
                                    ),
                                    json.dumps(
                                        event,
                                        ensure_ascii=False,
                                        sort_keys=True,
                                        separators=(",", ":"),
                                    ),
                                    event_sha,
                                    event["created_at_utc"],
                                ),
                            )
                            connection.commit()
                        raise ValueError(
                            "FAIL_CLOSED_REQUEST_ID_COLLISION"
                        )
                    continue
                connection.execute(
                    """
                    INSERT INTO technical_scheduled_requests (
                        model_version, request_id, issuance_id,
                        source_bucket, target_bucket, horizon_index, side,
                        requested_delta_pp, initial_status, payload_json,
                        payload_sha256, created_at_utc
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model_version,
                        str(request["request_id"]),
                        issuance.issuance_id,
                        issuance.source_bucket.isoformat(),
                        str(request["target_bucket"]),
                        int(request["horizon_index"]),
                        str(request["side"]),
                        float(request["requested_delta_pp"]),
                        str(request["execution_status"]),
                        payload_json,
                        payload_sha,
                        issuance.created_at_utc.isoformat(),
                    ),
                )
                inserted += 1
            connection.commit()
        return inserted

    def list_scheduled_requests(self) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM technical_scheduled_requests
                ORDER BY target_bucket, horizon_index, request_id
                """
            ).fetchall()
        items: list[dict] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            if canonical_json_sha256(payload) != row["payload_sha256"]:
                raise ValueError("MN18_SCHEDULED_REQUEST_SHA256_MISMATCH")
            items.append(
                {
                    "model_version": row["model_version"],
                    "request_id": row["request_id"],
                    "issuance_id": row["issuance_id"],
                    "source_bucket": row["source_bucket"],
                    "target_bucket": row["target_bucket"],
                    "horizon_index": row["horizon_index"],
                    "side": row["side"],
                    "requested_delta_pp": row["requested_delta_pp"],
                    "initial_status": row["initial_status"],
                    "payload": payload,
                    "payload_sha256": row["payload_sha256"],
                    "created_at_utc": row["created_at_utc"],
                }
            )
        return items

    def append(
        self, record: TechnicalIssuanceRecord
    ) -> TechnicalIssuanceRecord:
        output_json = None
        if record.output is not None:
            output_json = json.dumps(
                record.output,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        values = (
            record.issuance_id,
            record.idempotency_key,
            record.status,
            record.issuance_kind,
            record.created_at_utc.isoformat(),
            record.source_bucket.isoformat() if record.source_bucket else None,
            (
                record.source_bucket_end_utc.isoformat()
                if record.source_bucket_end_utc
                else None
            ),
            record.issued_at_utc.isoformat() if record.issued_at_utc else None,
            (
                record.input_history_start.isoformat()
                if record.input_history_start
                else None
            ),
            (
                record.input_history_end.isoformat()
                if record.input_history_end
                else None
            ),
            record.input_rows,
            record.current_gold_weight,
            record.outstanding_top_inventory_pp,
            record.dual_package_manifest_hash,
            record.mn18_manifest_hash,
            record.mn18_joblib_hash,
            record.pn02_bundle_hash,
            record.market_data_sha256,
            record.input_request_sha256,
            output_json,
            record.output_sha256,
            record.error_reason,
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO technical_issuances (
                    issuance_id, idempotency_key, status, issuance_kind,
                    created_at_utc,
                    source_bucket, source_bucket_end_utc, issued_at_utc,
                    input_history_start, input_history_end, input_rows,
                    current_gold_weight, outstanding_top_inventory_pp,
                    dual_package_manifest_hash, mn18_manifest_hash,
                    mn18_joblib_hash, pn02_bundle_hash,
                    market_data_sha256, input_request_sha256,
                    output_json, output_sha256, error_reason
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                values,
            )
            if record.status == "SUCCESS":
                connection.execute(
                    """
                    UPDATE technical_runtime_state
                    SET latest_issuance_id = ?, updated_at_utc = ?
                    WHERE singleton_id = 1
                    """,
                    (record.issuance_id, record.created_at_utc.isoformat()),
                )
            connection.commit()
        return record

    def record_fill(
        self,
        request: TechnicalFillRequest,
        *,
        recorded_at_utc: datetime | None = None,
    ) -> tuple[TechnicalFillRecord, bool]:
        recorded_at = recorded_at_utc or datetime.now(UTC)
        if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
            raise ValueError("recorded_at_utc must be timezone-aware")
        recorded_at = recorded_at.astimezone(UTC)
        request_hash = canonical_json_sha256(
            request.model_dump(mode="json")
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT * FROM technical_fills WHERE idempotency_key = ?
                """,
                (request.idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing["request_sha256"] != request_hash:
                    raise ValueError("TECHNICAL_FILL_IDEMPOTENCY_CONFLICT")
                connection.commit()
                return self._row_to_fill(existing), False

            state_row = connection.execute(
                """
                SELECT current_gold_weight, outstanding_top_inventory_pp,
                       revision, latest_issuance_id, updated_at_utc
                FROM technical_runtime_state
                WHERE singleton_id = 1
                """
            ).fetchone()
            if state_row is None:
                raise ValueError("TECHNICAL_RUNTIME_STATE_MISSING")
            state = self._state_from_row(state_row)
            if state.revision != request.expected_revision:
                raise ValueError("TECHNICAL_FILL_REVISION_CONFLICT")

            issuance_row = connection.execute(
                """
                SELECT * FROM technical_issuances
                WHERE issuance_id = ? AND status = 'SUCCESS'
                """,
                (request.issuance_id,),
            ).fetchone()
            if issuance_row is None:
                raise ValueError("TECHNICAL_FILL_ISSUANCE_NOT_FOUND")
            issuance = self._row_to_record(issuance_row)
            if issuance.issued_at_utc is None or issuance.output is None:
                raise ValueError("TECHNICAL_FILL_ISSUANCE_INVALID")
            if request.filled_at_utc < issuance.issued_at_utc:
                raise ValueError("TECHNICAL_FILL_PRECEDES_ADVICE")
            action = issuance.output["action"]
            if action["action"] == "HOLD":
                raise ValueError("TECHNICAL_FILL_NOT_ALLOWED_FOR_HOLD")
            if (
                abs(
                    float(action["current_target_gold_weight"])
                    - state.current_gold_weight
                )
                > 1e-12
            ):
                raise ValueError("TECHNICAL_FILL_STATE_NO_LONGER_MATCHES_ADVICE")

            full_delta_pp = abs(float(action["requested_delta_pp"]))
            actual_delta_pp = min(
                float(request.actual_delta_pp or full_delta_pp),
                full_delta_pp,
            )
            if actual_delta_pp <= 0:
                raise ValueError("TECHNICAL_FILL_DELTA_MUST_BE_POSITIVE")
            weight_after = max(
                0.5,
                state.current_gold_weight - actual_delta_pp / 100.0,
            )
            inventory_after_pp = min(
                50.0,
                state.outstanding_top_inventory_pp + actual_delta_pp,
            )
            revision_after = state.revision + 1
            fill_id = f"fill_{request_hash[:16].lower()}"
            connection.execute(
                """
                INSERT INTO technical_fills (
                    fill_id, issuance_id, idempotency_key, request_sha256,
                    revision_before, revision_after, weight_before,
                    weight_after, inventory_before_pp, inventory_after_pp,
                    actual_delta_pp, filled_at_utc, fill_price, recorded_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill_id,
                    request.issuance_id,
                    request.idempotency_key,
                    request_hash,
                    state.revision,
                    revision_after,
                    state.current_gold_weight,
                    weight_after,
                    state.outstanding_top_inventory_pp,
                    inventory_after_pp,
                    actual_delta_pp,
                    request.filled_at_utc.isoformat(),
                    request.fill_price,
                    recorded_at.isoformat(),
                ),
            )
            connection.execute(
                """
                UPDATE technical_runtime_state
                SET current_gold_weight = ?,
                    outstanding_top_inventory_pp = ?,
                    revision = ?, updated_at_utc = ?
                WHERE singleton_id = 1
                """,
                (
                    weight_after,
                    inventory_after_pp,
                    revision_after,
                    recorded_at.isoformat(),
                ),
            )
            row = connection.execute(
                "SELECT * FROM technical_fills WHERE fill_id = ?",
                (fill_id,),
            ).fetchone()
            connection.commit()
        return self._row_to_fill(row), True
