from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .schemas import EvidenceCitation, MacroClaim, MacroDocument, MacroObservation


@contextmanager
def _sqlite_connection(path: Path) -> Iterator[sqlite3.Connection]:
    """Commit or roll back and always release the Windows SQLite file handle."""

    connection = sqlite3.connect(path)
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


class MacroEvidenceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _sqlite_connection(self.path) as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT NOT NULL UNIQUE,
                    source TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    canonical_url TEXT NOT NULL,
                    content TEXT NOT NULL,
                    published_at_utc TEXT,
                    retrieved_at_utc TEXT NOT NULL,
                    eligible_from_utc TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    replay_eligible INTEGER NOT NULL,
                    provider_id TEXT,
                    metric_id TEXT,
                    source_tier TEXT NOT NULL DEFAULT 'A',
                    first_seen_at_utc TEXT,
                    retrieval_method TEXT NOT NULL DEFAULT 'HTML',
                    official_primary INTEGER NOT NULL DEFAULT 1,
                    revision_status TEXT NOT NULL DEFAULT 'UNKNOWN',
                    independence_key TEXT,
                    near_duplicate_group TEXT
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
                    title,
                    content,
                    event_type,
                    content='documents',
                    content_rowid='row_id',
                    tokenize='unicode61'
                );
                CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN
                    INSERT INTO documents_fts(rowid, title, content, event_type)
                    VALUES (new.row_id, new.title, new.content, new.event_type);
                END;
                CREATE TABLE IF NOT EXISTS observations (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    observation_id TEXT NOT NULL UNIQUE,
                    series_id TEXT NOT NULL,
                    observation_date TEXT NOT NULL,
                    value REAL NOT NULL,
                    realtime_start TEXT NOT NULL,
                    realtime_end TEXT NOT NULL,
                    eligible_from_utc TEXT NOT NULL,
                    retrieved_at_utc TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    source_payload_sha256 TEXT NOT NULL,
                    initial_release_only INTEGER NOT NULL,
                    provider_id TEXT NOT NULL DEFAULT 'fred_alfred',
                    metric_id TEXT,
                    source_tier TEXT NOT NULL DEFAULT 'A',
                    published_at_utc TEXT,
                    first_seen_at_utc TEXT,
                    retrieval_method TEXT NOT NULL DEFAULT 'API',
                    official_primary INTEGER NOT NULL DEFAULT 1,
                    revision_status TEXT NOT NULL DEFAULT 'UNKNOWN',
                    unit TEXT,
                    independence_key TEXT
                );
                CREATE INDEX IF NOT EXISTS observations_pit_idx
                ON observations(series_id, observation_date, eligible_from_utc);
                CREATE TABLE IF NOT EXISTS claims (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    claim_id TEXT NOT NULL UNIQUE,
                    slot TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    claim_type TEXT NOT NULL,
                    normalized_value TEXT NOT NULL,
                    display_text TEXT NOT NULL,
                    value_json TEXT,
                    unit TEXT,
                    reference_period TEXT,
                    observed_at_utc TEXT,
                    source_record_id TEXT NOT NULL,
                    provider_id TEXT NOT NULL,
                    source_tier TEXT NOT NULL,
                    canonical_url TEXT NOT NULL,
                    published_at_utc TEXT,
                    first_seen_at_utc TEXT NOT NULL,
                    retrieved_at_utc TEXT NOT NULL,
                    eligible_from_utc TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    retrieval_method TEXT NOT NULL,
                    official_primary INTEGER NOT NULL,
                    revision_status TEXT NOT NULL,
                    independence_key TEXT NOT NULL,
                    near_duplicate_group TEXT
                );
                CREATE INDEX IF NOT EXISTS claims_pit_idx
                ON claims(slot, eligible_from_utc, observed_at_utc);
                """
            )
            self._migrate_columns(connection)

    @staticmethod
    def _migrate_columns(connection: sqlite3.Connection) -> None:
        migrations = {
            "documents": {
                "provider_id": "TEXT",
                "metric_id": "TEXT",
                "source_tier": "TEXT NOT NULL DEFAULT 'A'",
                "first_seen_at_utc": "TEXT",
                "retrieval_method": "TEXT NOT NULL DEFAULT 'HTML'",
                "official_primary": "INTEGER NOT NULL DEFAULT 1",
                "revision_status": "TEXT NOT NULL DEFAULT 'UNKNOWN'",
                "independence_key": "TEXT",
                "near_duplicate_group": "TEXT",
            },
            "observations": {
                "provider_id": "TEXT NOT NULL DEFAULT 'fred_alfred'",
                "metric_id": "TEXT",
                "source_tier": "TEXT NOT NULL DEFAULT 'A'",
                "published_at_utc": "TEXT",
                "first_seen_at_utc": "TEXT",
                "retrieval_method": "TEXT NOT NULL DEFAULT 'API'",
                "official_primary": "INTEGER NOT NULL DEFAULT 1",
                "revision_status": "TEXT NOT NULL DEFAULT 'UNKNOWN'",
                "unit": "TEXT",
                "independence_key": "TEXT",
            },
        }
        for table, columns in migrations.items():
            existing = {
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for name, declaration in columns.items():
                if name not in existing:
                    connection.execute(
                        f"ALTER TABLE {table} ADD COLUMN {name} {declaration}"
                    )

    def ingest(self, document: MacroDocument) -> bool:
        self.initialize()
        with _sqlite_connection(self.path) as connection:
            existing = connection.execute(
                "SELECT content_sha256 FROM documents WHERE doc_id = ?", (document.doc_id,)
            ).fetchone()
            if existing is not None:
                if existing[0] != document.content_sha256:
                    raise ValueError("Immutable doc_id was reused for different content")
                return False
            connection.execute(
                """
                INSERT INTO documents (
                    doc_id, source, event_type, title, canonical_url, content,
                    published_at_utc, retrieved_at_utc, eligible_from_utc,
                    content_sha256, replay_eligible, provider_id, metric_id,
                    source_tier, first_seen_at_utc, retrieval_method,
                    official_primary, revision_status, independence_key,
                    near_duplicate_group
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.doc_id,
                    document.source,
                    document.event_type,
                    document.title,
                    document.canonical_url,
                    document.content,
                    document.published_at_utc.isoformat() if document.published_at_utc else None,
                    document.retrieved_at_utc.isoformat(),
                    document.eligible_from_utc.isoformat(),
                    document.content_sha256,
                    int(document.replay_eligible),
                    document.provider_id or document.source,
                    document.metric_id,
                    document.source_tier,
                    (document.first_seen_at_utc or document.retrieved_at_utc).isoformat(),
                    document.retrieval_method,
                    int(document.official_primary),
                    document.revision_status,
                    document.independence_key or document.provider_id or document.source,
                    document.near_duplicate_group,
                ),
            )
        return True

    @staticmethod
    def _fts_query(query: str) -> str:
        tokens = re.findall(r"[A-Za-z0-9]+", query)
        if not tokens:
            raise ValueError("Evidence query has no searchable tokens")
        return " OR ".join(f'"{token}"' for token in tokens[:20])

    def retrieve(
        self,
        query: str,
        *,
        event_type: str,
        as_of_utc: datetime,
        top_k: int = 5,
        replay_only: bool = False,
    ) -> tuple[EvidenceCitation, ...]:
        if as_of_utc.tzinfo is None or as_of_utc.utcoffset() is None:
            raise ValueError("as_of_utc must be timezone-aware")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not self.path.exists():
            return ()
        replay_clause = "AND d.replay_eligible = 1" if replay_only else ""
        with _sqlite_connection(self.path) as connection:
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(documents)").fetchall()
            }

            def field(name: str, fallback: str) -> str:
                return f"d.{name}" if name in columns else fallback

            sql = f"""
                SELECT d.doc_id, d.source, d.event_type, d.title, d.canonical_url,
                       d.published_at_utc, d.retrieved_at_utc,
                       d.eligible_from_utc, d.content_sha256,
                       bm25(documents_fts) AS rank,
                       {field('provider_id', 'd.source')},
                       {field('metric_id', 'NULL')},
                       {field('source_tier', "'A'")},
                       {field('first_seen_at_utc', 'd.retrieved_at_utc')},
                       {field('revision_status', "'UNKNOWN'")}
                FROM documents_fts
                JOIN documents d ON d.row_id = documents_fts.rowid
                WHERE documents_fts MATCH ?
                  AND d.event_type = ?
                  AND d.eligible_from_utc <= ?
                  {replay_clause}
                ORDER BY rank ASC, d.eligible_from_utc DESC
                LIMIT ?
            """
            rows = connection.execute(
                sql,
                (self._fts_query(query), event_type.upper(), as_of_utc.isoformat(), top_k),
            ).fetchall()
        return tuple(
            EvidenceCitation(
                doc_id=row[0],
                provider_id=row[10] or row[1],
                metric_id=row[11],
                source_tier=row[12],
                event_type=row[2],
                title=row[3],
                canonical_url=row[4],
                published_at_utc=(
                    None if row[5] is None else datetime.fromisoformat(row[5])
                ),
                retrieved_at_utc=datetime.fromisoformat(row[6]),
                first_seen_at_utc=datetime.fromisoformat(row[13] or row[6]),
                eligible_from_utc=datetime.fromisoformat(row[7]),
                content_sha256=row[8],
                score=float(-row[9]),
                revision_status=row[14],
            )
            for row in rows
        )

    def document_count(self) -> int:
        if not self.path.exists():
            return 0
        with _sqlite_connection(self.path) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def ingest_observations(self, observations: tuple[MacroObservation, ...]) -> int:
        """Insert immutable structured observations and reject id/content collisions."""

        self.initialize()
        inserted = 0
        with _sqlite_connection(self.path) as connection:
            for observation in observations:
                existing = connection.execute(
                    """
                    SELECT series_id, observation_date, value, realtime_start,
                           realtime_end, source_url, source_payload_sha256,
                           initial_release_only, provider_id, metric_id,
                           source_tier, published_at_utc, retrieval_method,
                           official_primary, revision_status, unit,
                           independence_key
                    FROM observations WHERE observation_id = ?
                    """,
                    (observation.observation_id,),
                ).fetchone()
                identity = (
                    observation.series_id,
                    observation.observation_date.isoformat(),
                    observation.value,
                    observation.realtime_start.isoformat(),
                    observation.realtime_end.isoformat(),
                    observation.source_url,
                    observation.source_payload_sha256,
                    int(observation.initial_release_only),
                    observation.provider_id,
                    observation.metric_id,
                    observation.source_tier,
                    (
                        None
                        if observation.published_at_utc is None
                        else observation.published_at_utc.isoformat()
                    ),
                    observation.retrieval_method,
                    int(observation.official_primary),
                    observation.revision_status,
                    observation.unit,
                    observation.independence_key or observation.provider_id,
                )
                if existing is not None:
                    if tuple(existing) != identity:
                        raise ValueError(
                            "Immutable macro observation id was reused for different content"
                        )
                    # Re-fetching identical provider content must preserve the first
                    # legally knowable timestamp already sealed in the active store.
                    continue
                connection.execute(
                    """
                    INSERT INTO observations (
                        observation_id, series_id, observation_date, value,
                        realtime_start, realtime_end, eligible_from_utc,
                        retrieved_at_utc, source_url, source_payload_sha256,
                        initial_release_only, provider_id, metric_id, source_tier,
                        published_at_utc, first_seen_at_utc, retrieval_method,
                        official_primary, revision_status, unit, independence_key
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation.observation_id,
                        observation.series_id,
                        observation.observation_date.isoformat(),
                        observation.value,
                        observation.realtime_start.isoformat(),
                        observation.realtime_end.isoformat(),
                        observation.eligible_from_utc.isoformat(),
                        observation.retrieved_at_utc.isoformat(),
                        observation.source_url,
                        observation.source_payload_sha256,
                        int(observation.initial_release_only),
                        observation.provider_id,
                        observation.metric_id,
                        observation.source_tier,
                        (
                            None
                            if observation.published_at_utc is None
                            else observation.published_at_utc.isoformat()
                        ),
                        (observation.first_seen_at_utc or observation.retrieved_at_utc).isoformat(),
                        observation.retrieval_method,
                        int(observation.official_primary),
                        observation.revision_status,
                        observation.unit,
                        observation.independence_key or observation.provider_id,
                    ),
                )
                inserted += 1
        return inserted

    def observations_as_of(
        self,
        *,
        series_ids: tuple[str, ...],
        as_of_utc: datetime,
    ) -> tuple[MacroObservation, ...]:
        """Return only the latest vintage legally knowable at the exact cutoff."""

        if as_of_utc.tzinfo is None or as_of_utc.utcoffset() is None:
            raise ValueError("as_of_utc must be timezone-aware")
        normalized = tuple(dict.fromkeys(value.strip().upper() for value in series_ids))
        if not normalized or any(not value for value in normalized):
            raise ValueError("At least one non-empty provider series or metric id is required")
        if not self.path.exists():
            return ()
        placeholders = ",".join("?" for _ in normalized)
        with _sqlite_connection(self.path) as connection:
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(observations)").fetchall()
            }

            def field(name: str, fallback: str) -> str:
                return name if name in columns else f"{fallback} AS {name}"

            sql = f"""
                WITH eligible AS (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY series_id, observation_date
                        ORDER BY eligible_from_utc DESC, realtime_start DESC, row_id DESC
                    ) AS vintage_rank
                    FROM observations
                    WHERE series_id IN ({placeholders})
                      AND observation_date <= ?
                      AND eligible_from_utc <= ?
                )
                SELECT observation_id, series_id, observation_date, value,
                       realtime_start, realtime_end, eligible_from_utc,
                       retrieved_at_utc, source_url, source_payload_sha256,
                       initial_release_only,
                       {field('provider_id', "'fred_alfred'")},
                       {field('metric_id', 'series_id')},
                       {field('source_tier', "'A'")},
                       {field('published_at_utc', 'NULL')},
                       {field('first_seen_at_utc', 'retrieved_at_utc')},
                       {field('retrieval_method', "'API'")},
                       {field('official_primary', '1')},
                       {field('revision_status', "'UNKNOWN'")},
                       {field('unit', 'NULL')},
                       {field('independence_key', "'fred_alfred'")}
                FROM eligible
                WHERE vintage_rank = 1
                ORDER BY series_id, observation_date
            """
            rows = connection.execute(
                sql,
                (*normalized, as_of_utc.date().isoformat(), as_of_utc.isoformat()),
            ).fetchall()
        return tuple(
            MacroObservation(
                observation_id=row[0],
                series_id=row[1],
                observation_date=row[2],
                value=row[3],
                realtime_start=row[4],
                realtime_end=row[5],
                eligible_from_utc=datetime.fromisoformat(row[6]),
                retrieved_at_utc=datetime.fromisoformat(row[7]),
                source_url=row[8],
                source_payload_sha256=row[9],
                initial_release_only=bool(row[10]),
                provider_id=row[11],
                metric_id=row[12],
                source_tier=row[13],
                published_at_utc=(
                    None if row[14] is None else datetime.fromisoformat(row[14])
                ),
                first_seen_at_utc=datetime.fromisoformat(row[15] or row[7]),
                retrieval_method=row[16],
                official_primary=bool(row[17]),
                revision_status=row[18],
                unit=row[19],
                independence_key=row[20],
            )
            for row in rows
        )

    def latest_observations_as_of(
        self,
        *,
        series_ids: tuple[str, ...],
        as_of_utc: datetime,
    ) -> tuple[MacroObservation, ...]:
        """Return one latest legally knowable initial-release value per series."""

        observations = self.observations_as_of(series_ids=series_ids, as_of_utc=as_of_utc)
        latest: dict[str, MacroObservation] = {}
        for observation in observations:
            existing = latest.get(observation.series_id)
            if existing is None or (
                observation.observation_date,
                observation.eligible_from_utc,
            ) > (
                existing.observation_date,
                existing.eligible_from_utc,
            ):
                latest[observation.series_id] = observation
        return tuple(latest[key] for key in sorted(latest))

    def observation_count(self) -> int:
        if not self.path.exists():
            return 0
        with _sqlite_connection(self.path) as connection:
            return int(connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0])

    def ingest_claims(self, claims: tuple[MacroClaim, ...]) -> int:
        self.initialize()
        inserted = 0
        with _sqlite_connection(self.path) as connection:
            for claim in claims:
                existing = connection.execute(
                    "SELECT content_sha256, normalized_value FROM claims WHERE claim_id = ?",
                    (claim.claim_id,),
                ).fetchone()
                if existing is not None:
                    if tuple(existing) != (claim.content_sha256, claim.normalized_value):
                        raise ValueError("Immutable claim_id was reused for different content")
                    continue
                connection.execute(
                    """
                    INSERT INTO claims (
                        claim_id, slot, event_type, claim_type, normalized_value,
                        display_text, value_json, unit, reference_period,
                        observed_at_utc, source_record_id, provider_id, source_tier,
                        canonical_url, published_at_utc, first_seen_at_utc,
                        retrieved_at_utc, eligible_from_utc, content_sha256,
                        retrieval_method, official_primary, revision_status,
                        independence_key, near_duplicate_group
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claim.claim_id,
                        claim.slot,
                        claim.event_type,
                        claim.claim_type,
                        claim.normalized_value,
                        claim.display_text,
                        (
                            None
                            if claim.value is None
                            else json.dumps(claim.value, ensure_ascii=False, sort_keys=True)
                        ),
                        claim.unit,
                        claim.reference_period,
                        (
                            None
                            if claim.observed_at_utc is None
                            else claim.observed_at_utc.isoformat()
                        ),
                        claim.source_record_id,
                        claim.provider_id,
                        claim.source_tier,
                        claim.canonical_url,
                        (
                            None
                            if claim.published_at_utc is None
                            else claim.published_at_utc.isoformat()
                        ),
                        claim.first_seen_at_utc.isoformat(),
                        claim.retrieved_at_utc.isoformat(),
                        claim.eligible_from_utc.isoformat(),
                        claim.content_sha256,
                        claim.retrieval_method,
                        int(claim.official_primary),
                        claim.revision_status,
                        claim.independence_key,
                        claim.near_duplicate_group,
                    ),
                )
                inserted += 1
        return inserted

    def claims_as_of(
        self,
        *,
        as_of_utc: datetime,
        slots: tuple[str, ...] | None = None,
    ) -> tuple[MacroClaim, ...]:
        if as_of_utc.tzinfo is None or as_of_utc.utcoffset() is None:
            raise ValueError("as_of_utc must be timezone-aware")
        if not self.path.exists():
            return ()
        with _sqlite_connection(self.path) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='claims'"
            ).fetchone()
            if exists is None:
                return ()
            parameters: list[object] = [as_of_utc.isoformat(), as_of_utc.isoformat()]
            slot_clause = ""
            if slots:
                normalized = tuple(dict.fromkeys(slots))
                slot_clause = f"AND slot IN ({','.join('?' for _ in normalized)})"
                parameters.extend(normalized)
            rows = connection.execute(
                f"""
                SELECT claim_id, slot, event_type, claim_type, normalized_value,
                       display_text, value_json, unit, reference_period,
                       observed_at_utc, source_record_id, provider_id, source_tier,
                       canonical_url, published_at_utc, first_seen_at_utc,
                       retrieved_at_utc, eligible_from_utc, content_sha256,
                       retrieval_method, official_primary, revision_status,
                       independence_key, near_duplicate_group
                FROM claims
                WHERE eligible_from_utc <= ?
                  AND (published_at_utc IS NULL OR published_at_utc <= ?)
                  {slot_clause}
                ORDER BY slot, observed_at_utc DESC, eligible_from_utc DESC, claim_id
                """,
                parameters,
            ).fetchall()
        return tuple(
            MacroClaim(
                claim_id=row[0],
                slot=row[1],
                event_type=row[2],
                claim_type=row[3],
                normalized_value=row[4],
                display_text=row[5],
                value=None if row[6] is None else json.loads(row[6]),
                unit=row[7],
                reference_period=row[8],
                observed_at_utc=(
                    None if row[9] is None else datetime.fromisoformat(row[9])
                ),
                source_record_id=row[10],
                provider_id=row[11],
                source_tier=row[12],
                canonical_url=row[13],
                published_at_utc=(
                    None if row[14] is None else datetime.fromisoformat(row[14])
                ),
                first_seen_at_utc=datetime.fromisoformat(row[15]),
                retrieved_at_utc=datetime.fromisoformat(row[16]),
                eligible_from_utc=datetime.fromisoformat(row[17]),
                content_sha256=row[18],
                retrieval_method=row[19],
                official_primary=bool(row[20]),
                revision_status=row[21],
                independence_key=row[22],
                near_duplicate_group=row[23],
            )
            for row in rows
        )

    def claim_count(self) -> int:
        if not self.path.exists():
            return 0
        with _sqlite_connection(self.path) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='claims'"
            ).fetchone()
            if exists is None:
                return 0
            return int(connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0])
