"""Append-audited MN18 receiver ledger and isolated state machines."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from aupilot.core.hashing import canonical_json_sha256

from backend.app.schemas.technical import (
    MN18_MODEL_VERSION,
    TechnicalFillRecord,
    TechnicalFillRequest,
    TechnicalRuntimeState,
)
from backend.app.services.mn18_receiver_policy import (
    EPSILON,
    RECEIVER_POLICY_ID,
    resolve_target_bucket,
)


class MN18ReceiverLedger:
    """Persist receiver policy facts without altering native model output."""

    def __init__(self, path: Path) -> None:
        self.path = path
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
                CREATE TABLE IF NOT EXISTS technical_request_events (
                    event_id TEXT PRIMARY KEY,
                    model_version TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    state_scope TEXT NOT NULL CHECK (
                        state_scope IN ('RECEIVER', 'USER', 'SHADOW')
                    ),
                    status TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    event_sha256 TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    FOREIGN KEY (model_version, request_id)
                        REFERENCES technical_scheduled_requests(
                            model_version, request_id
                        )
                );
                CREATE INDEX IF NOT EXISTS ix_mn18_request_events
                    ON technical_request_events(
                        model_version, request_id, state_scope, created_at_utc
                    );

                CREATE TABLE IF NOT EXISTS technical_fifo_lots (
                    lot_id TEXT PRIMARY KEY,
                    state_scope TEXT NOT NULL CHECK (
                        state_scope IN ('USER', 'SHADOW')
                    ),
                    model_version TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    original_pp REAL NOT NULL CHECK (original_pp > 0),
                    created_at_utc TEXT NOT NULL,
                    source_fill_id TEXT,
                    FOREIGN KEY (model_version, request_id)
                        REFERENCES technical_scheduled_requests(
                            model_version, request_id
                        )
                );
                CREATE INDEX IF NOT EXISTS ix_mn18_fifo_lots_scope
                    ON technical_fifo_lots(state_scope, created_at_utc, lot_id);

                CREATE TABLE IF NOT EXISTS technical_fifo_consumptions (
                    consumption_id TEXT PRIMARY KEY,
                    state_scope TEXT NOT NULL CHECK (
                        state_scope IN ('USER', 'SHADOW')
                    ),
                    lot_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    consumed_pp REAL NOT NULL CHECK (consumed_pp > 0),
                    created_at_utc TEXT NOT NULL,
                    source_fill_id TEXT,
                    FOREIGN KEY (lot_id) REFERENCES technical_fifo_lots(lot_id),
                    FOREIGN KEY (model_version, request_id)
                        REFERENCES technical_scheduled_requests(
                            model_version, request_id
                        )
                );

                CREATE TABLE IF NOT EXISTS technical_shadow_state (
                    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                    current_gold_weight REAL NOT NULL CHECK (
                        current_gold_weight BETWEEN 0.5 AND 1.0
                    ),
                    outstanding_top_inventory_pp REAL NOT NULL CHECK (
                        outstanding_top_inventory_pp BETWEEN 0 AND 50
                    ),
                    revision INTEGER NOT NULL CHECK (revision >= 0),
                    latest_target_bucket TEXT,
                    updated_at_utc TEXT NOT NULL,
                    seed_contract TEXT NOT NULL
                );
                INSERT OR IGNORE INTO technical_shadow_state (
                    singleton_id, current_gold_weight,
                    outstanding_top_inventory_pp, revision,
                    latest_target_bucket, updated_at_utc, seed_contract
                ) VALUES (
                    1, 1.0, 0.0, 0, NULL,
                    '2026-07-21T23:59:59+00:00',
                    'FROZEN_OOS_TERMINAL_STATE_2026-07-21'
                );

                CREATE TABLE IF NOT EXISTS technical_daily_shadow_actions (
                    model_version TEXT NOT NULL,
                    target_bucket TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    result_sha256 TEXT NOT NULL,
                    target_open REAL NOT NULL CHECK (target_open > 0),
                    simulated_fill_price REAL,
                    transaction_cost_bps_per_side REAL NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    PRIMARY KEY (model_version, target_bucket)
                );
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(technical_fills)"
                ).fetchall()
            }
            for name, sql_type in (
                ("request_id", "TEXT"),
                ("side", "TEXT"),
                ("action", "TEXT"),
            ):
                if name not in columns:
                    connection.execute(
                        f"ALTER TABLE technical_fills ADD COLUMN {name} {sql_type}"
                    )
            connection.commit()

    @staticmethod
    def _canonical_json(payload: dict[str, Any]) -> tuple[str, str]:
        return (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            canonical_json_sha256(payload),
        )

    def shadow_state(self) -> dict[str, Any]:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM technical_shadow_state WHERE singleton_id = 1"
            ).fetchone()
        if row is None:
            raise ValueError("MN18_SHADOW_STATE_MISSING")
        return dict(row)

    def _fifo_lots(
        self,
        connection: sqlite3.Connection,
        state_scope: str,
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            """
            SELECT lot.lot_id, lot.original_pp, lot.created_at_utc,
                   COALESCE(SUM(consumed.consumed_pp), 0) AS consumed_pp
            FROM technical_fifo_lots AS lot
            LEFT JOIN technical_fifo_consumptions AS consumed
              ON consumed.lot_id = lot.lot_id
            WHERE lot.state_scope = ?
            GROUP BY lot.lot_id, lot.original_pp, lot.created_at_utc
            ORDER BY lot.created_at_utc, lot.lot_id
            """,
            (state_scope,),
        ).fetchall()
        return [
            {
                "lot_id": row["lot_id"],
                "original_pp": row["original_pp"],
                "remaining_pp": row["original_pp"] - row["consumed_pp"],
                "created_at_utc": row["created_at_utc"],
            }
            for row in rows
            if row["original_pp"] - row["consumed_pp"] > EPSILON
        ]

    def fifo_lots(self, state_scope: str) -> list[dict[str, Any]]:
        if state_scope not in {"USER", "SHADOW"}:
            raise ValueError("MN18_FIFO_SCOPE_INVALID")
        with self._connection() as connection:
            return self._fifo_lots(connection, state_scope)

    def requests_for_target(
        self,
        target_bucket: date | str,
        *,
        compliant_forward_only: bool,
    ) -> list[dict[str, Any]]:
        target = (
            target_bucket.isoformat()
            if isinstance(target_bucket, date)
            else str(target_bucket)
        )
        query = """
            SELECT request.*, issuance.issuance_kind
            FROM technical_scheduled_requests AS request
            JOIN technical_issuances AS issuance
              ON issuance.issuance_id = request.issuance_id
            WHERE request.model_version = ?
              AND request.target_bucket = ?
        """
        parameters: list[Any] = [MN18_MODEL_VERSION, target]
        if compliant_forward_only:
            query += " AND issuance.issuance_kind = 'COMPLIANT_FORWARD'"
        query += (
            " ORDER BY request.source_bucket, request.created_at_utc,"
            " request.request_id"
        )
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [
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
                "payload": json.loads(row["payload_json"]),
                "payload_sha256": row["payload_sha256"],
                "created_at_utc": row["created_at_utc"],
                "issuance_kind": row["issuance_kind"],
            }
            for row in rows
        ]

    def append_event(
        self,
        *,
        request_id: str,
        state_scope: str,
        status: str,
        payload: dict[str, Any],
        created_at_utc: datetime,
        connection: sqlite3.Connection | None = None,
    ) -> tuple[dict[str, Any], bool]:
        if state_scope not in {"RECEIVER", "USER", "SHADOW"}:
            raise ValueError("MN18_REQUEST_EVENT_SCOPE_INVALID")
        event = {
            "receiver_policy": RECEIVER_POLICY_ID,
            "model_version": MN18_MODEL_VERSION,
            "request_id": request_id,
            "state_scope": state_scope,
            "status": status,
            "payload": payload,
            "created_at_utc": created_at_utc.astimezone(UTC).isoformat(),
        }
        event_json, event_sha = self._canonical_json(event)
        event_id = f"evt_{event_sha[:24].lower()}"

        owns_connection = connection is None
        if owns_connection:
            context = self._connection()
            connection = context.__enter__()
            connection.execute("BEGIN IMMEDIATE")
        try:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO technical_request_events (
                    event_id, model_version, request_id, state_scope,
                    status, event_json, event_sha256, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    MN18_MODEL_VERSION,
                    request_id,
                    state_scope,
                    status,
                    event_json,
                    event_sha,
                    event["created_at_utc"],
                ),
            )
            created = cursor.rowcount == 1
            if owns_connection:
                connection.commit()
            return {**event, "event_id": event_id, "event_sha256": event_sha}, created
        finally:
            if owns_connection:
                context.__exit__(None, None, None)

    def list_events(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT event_json, event_sha256
                FROM technical_request_events
                ORDER BY created_at_utc, event_id
                """
            ).fetchall()
        result = []
        for row in rows:
            payload = json.loads(row["event_json"])
            if canonical_json_sha256(payload) != row["event_sha256"]:
                raise ValueError("MN18_REQUEST_EVENT_SHA256_MISMATCH")
            result.append(payload)
        return result

    def resolve_user_target(self, target_bucket: date | str) -> dict[str, Any]:
        target = (
            target_bucket.isoformat()
            if isinstance(target_bucket, date)
            else str(target_bucket)
        )
        with self._connection() as connection:
            state_row = connection.execute(
                """
                SELECT current_gold_weight, outstanding_top_inventory_pp,
                       revision, latest_issuance_id, updated_at_utc
                FROM technical_runtime_state WHERE singleton_id = 1
                """
            ).fetchone()
            requests = self.requests_for_target(
                target, compliant_forward_only=False
            )
            lots = self._fifo_lots(connection, "USER")
        if state_row is None:
            raise ValueError("TECHNICAL_RUNTIME_STATE_MISSING")
        collision_ids = {
            event["request_id"]
            for event in self.list_events()
            if event["status"] == "FAIL_CLOSED_REQUEST_ID_COLLISION"
        }
        target_collision = any(
            request["request_id"] in collision_ids for request in requests
        )
        if target_collision:
            requests = []
        result, events = resolve_target_bucket(
            target_bucket=target,
            requests=requests,
            current_gold_weight=float(state_row["current_gold_weight"]),
            fifo_lots=lots,
        )
        result["state_scope"] = "USER"
        result["state_revision"] = int(state_row["revision"])
        result["lifecycle_preview"] = events
        if target_collision:
            result["action"] = "HOLD"
            result["reason_code"] = "FAIL_CLOSED_REQUEST_ID_COLLISION"
        return result

    def daily_shadow_action(
        self, target_bucket: date | str
    ) -> dict[str, Any] | None:
        target = (
            target_bucket.isoformat()
            if isinstance(target_bucket, date)
            else str(target_bucket)
        )
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM technical_daily_shadow_actions
                WHERE model_version = ? AND target_bucket = ?
                """,
                (MN18_MODEL_VERSION, target),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(row["result_json"])
        if canonical_json_sha256(payload) != row["result_sha256"]:
            raise ValueError("MN18_SHADOW_ACTION_SHA256_MISMATCH")
        return payload

    def list_shadow_actions(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT result_json, result_sha256
                FROM technical_daily_shadow_actions
                ORDER BY target_bucket
                """
            ).fetchall()
        payloads = []
        for row in rows:
            payload = json.loads(row["result_json"])
            if canonical_json_sha256(payload) != row["result_sha256"]:
                raise ValueError("MN18_SHADOW_ACTION_SHA256_MISMATCH")
            payloads.append(payload)
        return payloads

    def settle_shadow_target(
        self,
        *,
        target_bucket: date,
        target_open: float,
        completed_at_utc: datetime,
        transaction_cost_bps_per_side: float = 2.0,
    ) -> tuple[dict[str, Any], bool]:
        if target_open <= 0:
            raise ValueError("MN18_SHADOW_TARGET_OPEN_INVALID")
        completed_at = completed_at_utc.astimezone(UTC)
        target_end = datetime.combine(
            target_bucket + timedelta(days=1), datetime.min.time(), UTC
        )
        if completed_at < target_end:
            raise ValueError("MN18_SHADOW_TARGET_BUCKET_INCOMPLETE")
        target = target_bucket.isoformat()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                """
                SELECT result_json, result_sha256
                FROM technical_daily_shadow_actions
                WHERE model_version = ? AND target_bucket = ?
                """,
                (MN18_MODEL_VERSION, target),
            ).fetchone()
            if existing is not None:
                payload = json.loads(existing["result_json"])
                if canonical_json_sha256(payload) != existing["result_sha256"]:
                    raise ValueError("MN18_SHADOW_ACTION_SHA256_MISMATCH")
                connection.commit()
                return payload, False

            request_rows = connection.execute(
                """
                SELECT request.*, issuance.issuance_kind
                FROM technical_scheduled_requests AS request
                JOIN technical_issuances AS issuance
                  ON issuance.issuance_id = request.issuance_id
                WHERE request.model_version = ?
                  AND request.target_bucket = ?
                  AND issuance.issuance_kind = 'COMPLIANT_FORWARD'
                ORDER BY request.source_bucket, request.created_at_utc,
                         request.request_id
                """,
                (MN18_MODEL_VERSION, target),
            ).fetchall()
            requests = [
                {
                    "model_version": row["model_version"],
                    "request_id": row["request_id"],
                    "issuance_id": row["issuance_id"],
                    "source_bucket": row["source_bucket"],
                    "target_bucket": row["target_bucket"],
                    "horizon_index": row["horizon_index"],
                    "side": row["side"],
                    "requested_delta_pp": row["requested_delta_pp"],
                    "created_at_utc": row["created_at_utc"],
                }
                for row in request_rows
            ]
            collision = connection.execute(
                """
                SELECT event.request_id
                FROM technical_request_events AS event
                JOIN technical_scheduled_requests AS request
                  ON request.model_version = event.model_version
                 AND request.request_id = event.request_id
                WHERE event.model_version = ?
                  AND request.target_bucket = ?
                  AND event.status = 'FAIL_CLOSED_REQUEST_ID_COLLISION'
                LIMIT 1
                """,
                (MN18_MODEL_VERSION, target),
            ).fetchone()
            state = connection.execute(
                "SELECT * FROM technical_shadow_state WHERE singleton_id = 1"
            ).fetchone()
            if state is None:
                raise ValueError("MN18_SHADOW_STATE_MISSING")
            lots = self._fifo_lots(connection, "SHADOW")
            result, events = resolve_target_bucket(
                target_bucket=target,
                requests=[] if collision is not None else requests,
                current_gold_weight=float(state["current_gold_weight"]),
                fifo_lots=lots,
            )
            if collision is not None:
                result["reason_code"] = "FAIL_CLOSED_REQUEST_ID_COLLISION"
            executed = float(result["executed_delta_pp"])
            if result["side"] == "TOP" and executed < -EPSILON:
                fill_price = target_open * (
                    1.0 - transaction_cost_bps_per_side / 10_000.0
                )
            elif result["side"] == "BOTTOM" and executed > EPSILON:
                fill_price = target_open * (
                    1.0 + transaction_cost_bps_per_side / 10_000.0
                )
            else:
                fill_price = None
            result.update(
                {
                    "state_scope": "SHADOW",
                    "state_revision_before": int(state["revision"]),
                    "state_revision_after": (
                        int(state["revision"]) + (1 if abs(executed) > EPSILON else 0)
                    ),
                    "target_open": target_open,
                    "simulated_fill_price": fill_price,
                    "transaction_cost_bps_per_side": (
                        transaction_cost_bps_per_side
                    ),
                    "settled_at_utc": completed_at.isoformat(),
                    "request_frozen_before_target_open": True,
                    "target_hlc_used_for_execution_decision": False,
                }
            )

            winner = result["winning_request_id"]
            for event in events:
                status = event["status"]
                if (
                    event["request_id"] == winner
                    and abs(executed) > EPSILON
                ):
                    status = "EXECUTED_IN_FORWARD_SHADOW"
                self.append_event(
                    request_id=event["request_id"],
                    state_scope="SHADOW",
                    status=status,
                    payload={**event, "target_bucket": target},
                    created_at_utc=completed_at,
                    connection=connection,
                )

            if result["side"] == "TOP" and executed < -EPSILON:
                lot_payload = {
                    "scope": "SHADOW",
                    "request_id": winner,
                    "target_bucket": target,
                    "original_pp": abs(executed),
                }
                lot_id = (
                    f"lot_shadow_{canonical_json_sha256(lot_payload)[:20].lower()}"
                )
                connection.execute(
                    """
                    INSERT INTO technical_fifo_lots (
                        lot_id, state_scope, model_version, request_id,
                        original_pp, created_at_utc, source_fill_id
                    ) VALUES (?, 'SHADOW', ?, ?, ?, ?, NULL)
                    """,
                    (
                        lot_id,
                        MN18_MODEL_VERSION,
                        winner,
                        abs(executed),
                        completed_at.isoformat(),
                    ),
                )
            elif result["side"] == "BOTTOM" and executed > EPSILON:
                for index, consumption in enumerate(
                    result["fifo_consumptions"]
                ):
                    consumption_payload = {
                        "scope": "SHADOW",
                        "request_id": winner,
                        "target_bucket": target,
                        "lot_id": consumption["lot_id"],
                        "consumed_pp": consumption["consumed_pp"],
                        "index": index,
                    }
                    consumption_id = (
                        "consume_shadow_"
                        + canonical_json_sha256(consumption_payload)[:20].lower()
                    )
                    connection.execute(
                        """
                        INSERT INTO technical_fifo_consumptions (
                            consumption_id, state_scope, lot_id,
                            model_version, request_id, consumed_pp,
                            created_at_utc, source_fill_id
                        ) VALUES (?, 'SHADOW', ?, ?, ?, ?, ?, NULL)
                        """,
                        (
                            consumption_id,
                            consumption["lot_id"],
                            MN18_MODEL_VERSION,
                            winner,
                            consumption["consumed_pp"],
                            completed_at.isoformat(),
                        ),
                    )

            revision_after = result["state_revision_after"]
            connection.execute(
                """
                UPDATE technical_shadow_state
                SET current_gold_weight = ?,
                    outstanding_top_inventory_pp = ?,
                    revision = ?, latest_target_bucket = ?,
                    updated_at_utc = ?
                WHERE singleton_id = 1
                """,
                (
                    result["recommended_target_gold_weight"],
                    result["outstanding_top_inventory_pp_after"],
                    revision_after,
                    target,
                    completed_at.isoformat(),
                ),
            )
            result_json, result_sha = self._canonical_json(result)
            connection.execute(
                """
                INSERT INTO technical_daily_shadow_actions (
                    model_version, target_bucket, result_json,
                    result_sha256, target_open, simulated_fill_price,
                    transaction_cost_bps_per_side, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    MN18_MODEL_VERSION,
                    target,
                    result_json,
                    result_sha,
                    target_open,
                    fill_price,
                    transaction_cost_bps_per_side,
                    completed_at.isoformat(),
                ),
            )
            connection.commit()
        return result, True

    @staticmethod
    def _fill_from_row(row: sqlite3.Row) -> TechnicalFillRecord:
        return TechnicalFillRecord(
            fill_id=row["fill_id"],
            issuance_id=row["issuance_id"],
            request_id=row["request_id"],
            side=row["side"],
            action=row["action"],
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

    def record_user_fill(
        self,
        request: TechnicalFillRequest,
        *,
        recorded_at_utc: datetime | None = None,
    ) -> tuple[TechnicalFillRecord, bool]:
        if request.request_id is None:
            raise ValueError("MN18_USER_FILL_REQUEST_ID_REQUIRED")
        recorded_at = (recorded_at_utc or datetime.now(UTC)).astimezone(UTC)
        request_hash = canonical_json_sha256(
            request.model_dump(mode="json")
        )
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM technical_fills WHERE idempotency_key = ?",
                (request.idempotency_key,),
            ).fetchone()
            if existing is not None:
                if existing["request_sha256"] != request_hash:
                    raise ValueError("TECHNICAL_FILL_IDEMPOTENCY_CONFLICT")
                connection.commit()
                return self._fill_from_row(existing), False

            request_row = connection.execute(
                """
                SELECT request.*, issuance.issuance_kind
                FROM technical_scheduled_requests AS request
                JOIN technical_issuances AS issuance
                  ON issuance.issuance_id = request.issuance_id
                WHERE request.model_version = ? AND request.request_id = ?
                """,
                (MN18_MODEL_VERSION, request.request_id),
            ).fetchone()
            if request_row is None:
                raise ValueError("MN18_USER_FILL_REQUEST_NOT_FOUND")
            if request_row["issuance_id"] != request.issuance_id:
                raise ValueError("MN18_USER_FILL_ISSUANCE_MISMATCH")
            target = date.fromisoformat(request_row["target_bucket"])
            if request.filled_at_utc.astimezone(UTC).date() != target:
                raise ValueError("MN18_USER_FILL_OUTSIDE_TARGET_BUCKET")

            state_row = connection.execute(
                "SELECT * FROM technical_runtime_state WHERE singleton_id = 1"
            ).fetchone()
            if state_row is None:
                raise ValueError("TECHNICAL_RUNTIME_STATE_MISSING")
            if int(state_row["revision"]) != request.expected_revision:
                raise ValueError("TECHNICAL_FILL_REVISION_CONFLICT")
            request_rows = connection.execute(
                """
                SELECT request.*
                FROM technical_scheduled_requests AS request
                WHERE request.model_version = ?
                  AND request.target_bucket = ?
                ORDER BY request.source_bucket, request.created_at_utc,
                         request.request_id
                """,
                (MN18_MODEL_VERSION, target.isoformat()),
            ).fetchall()
            requests = [dict(row) for row in request_rows]
            lots = self._fifo_lots(connection, "USER")
            result, events = resolve_target_bucket(
                target_bucket=target.isoformat(),
                requests=requests,
                current_gold_weight=float(state_row["current_gold_weight"]),
                fifo_lots=lots,
            )
            if result["winning_request_id"] != request.request_id:
                raise ValueError("MN18_USER_FILL_REQUEST_NOT_DAILY_WINNER")
            executable = abs(float(result["executed_delta_pp"]))
            if executable <= EPSILON:
                raise ValueError("MN18_USER_FILL_ZERO_EXECUTABLE_DELTA")
            actual = min(
                float(request.actual_delta_pp or executable), executable
            )
            side = str(result["side"])
            signed_actual = -actual if side == "TOP" else actual
            weight_after = float(state_row["current_gold_weight"]) + (
                signed_actual / 100.0
            )
            inventory_before = sum(
                float(lot["remaining_pp"]) for lot in lots
            )
            inventory_after = (
                inventory_before + actual
                if side == "TOP"
                else inventory_before - actual
            )
            revision_after = int(state_row["revision"]) + 1
            fill_id = f"fill_{request_hash[:16].lower()}"
            action = (
                "REDUCE_GOLD_WEIGHT"
                if side == "TOP"
                else "REENTER_GOLD_WEIGHT"
            )
            connection.execute(
                """
                INSERT INTO technical_fills (
                    fill_id, issuance_id, idempotency_key, request_sha256,
                    revision_before, revision_after, weight_before,
                    weight_after, inventory_before_pp, inventory_after_pp,
                    actual_delta_pp, filled_at_utc, fill_price,
                    recorded_at_utc, request_id, side, action
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill_id,
                    request.issuance_id,
                    request.idempotency_key,
                    request_hash,
                    state_row["revision"],
                    revision_after,
                    state_row["current_gold_weight"],
                    weight_after,
                    inventory_before,
                    inventory_after,
                    actual,
                    request.filled_at_utc.isoformat(),
                    request.fill_price,
                    recorded_at.isoformat(),
                    request.request_id,
                    side,
                    action,
                ),
            )
            if side == "TOP":
                lot_payload = {
                    "scope": "USER",
                    "request_id": request.request_id,
                    "fill_id": fill_id,
                    "original_pp": actual,
                }
                lot_id = (
                    f"lot_user_{canonical_json_sha256(lot_payload)[:20].lower()}"
                )
                connection.execute(
                    """
                    INSERT INTO technical_fifo_lots (
                        lot_id, state_scope, model_version, request_id,
                        original_pp, created_at_utc, source_fill_id
                    ) VALUES (?, 'USER', ?, ?, ?, ?, ?)
                    """,
                    (
                        lot_id,
                        MN18_MODEL_VERSION,
                        request.request_id,
                        actual,
                        request.filled_at_utc.isoformat(),
                        fill_id,
                    ),
                )
            else:
                remaining = actual
                for index, lot in enumerate(lots):
                    if remaining <= EPSILON:
                        break
                    consumed = min(remaining, float(lot["remaining_pp"]))
                    consumption_payload = {
                        "scope": "USER",
                        "request_id": request.request_id,
                        "fill_id": fill_id,
                        "lot_id": lot["lot_id"],
                        "consumed_pp": consumed,
                        "index": index,
                    }
                    consumption_id = (
                        "consume_user_"
                        + canonical_json_sha256(consumption_payload)[:20].lower()
                    )
                    connection.execute(
                        """
                        INSERT INTO technical_fifo_consumptions (
                            consumption_id, state_scope, lot_id,
                            model_version, request_id, consumed_pp,
                            created_at_utc, source_fill_id
                        ) VALUES (?, 'USER', ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            consumption_id,
                            lot["lot_id"],
                            MN18_MODEL_VERSION,
                            request.request_id,
                            consumed,
                            request.filled_at_utc.isoformat(),
                            fill_id,
                        ),
                    )
                    remaining -= consumed
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
                    inventory_after,
                    revision_after,
                    recorded_at.isoformat(),
                ),
            )
            for event in events:
                if event["request_id"] == request.request_id:
                    self.append_event(
                        request_id=request.request_id,
                        state_scope="USER",
                        status="CONFIRMED_USER_FILL",
                        payload={
                            "fill_id": fill_id,
                            "actual_delta_pp": actual,
                            "target_bucket": target.isoformat(),
                            "prior_resolution": event,
                        },
                        created_at_utc=recorded_at,
                        connection=connection,
                    )
            row = connection.execute(
                "SELECT * FROM technical_fills WHERE fill_id = ?",
                (fill_id,),
            ).fetchone()
            connection.commit()
        return self._fill_from_row(row), True
