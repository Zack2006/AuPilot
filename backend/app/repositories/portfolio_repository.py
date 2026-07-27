"""File repository for auditable portfolio facts. / 可审计持仓事实的文件仓储。

Purpose / 文件用途: isolate per-user paths and atomic JSON replacement.
Inputs / 输入: validated Pydantic-ready dictionaries.
Outputs / 输出: settings versions, append-only transactions, forecast records.
Business invariants / 业务约束: old settings and transactions are never overwritten.
Side effects / 副作用: writes UTF-8 JSON below ``storage/users`` and ``storage/forecasts``.
Fallback behavior / 降级: corrupt JSON raises DataCorruptionError; it is never treated as healthy empty data.
"""

from pathlib import Path
from threading import RLock

from backend.app.repositories.json_repository import JsonRepository


class PortfolioRepository:
    """Coordinate the three file collections owned by one demo user. / 协调单个演示用户的三类文件集合。"""

    def __init__(self, storage_dir: Path, user_id: str = "demo") -> None:
        self.user_id = user_id
        user_dir = storage_dir / "users" / user_id
        self.settings = JsonRepository(user_dir / "settings_history.json", [])
        self.transactions = JsonRepository(user_dir / "transactions.json", [])
        self.allocation_snapshots = JsonRepository(
            user_dir / "gold_weight_snapshots.json", []
        )
        self.composite_snapshots = JsonRepository(
            user_dir / "today_advice_composite_snapshots.json", []
        )
        self.today_advice_h1_logs = JsonRepository(
            user_dir / "today_advice_h1_logs.json", []
        )
        self.forecasts = JsonRepository(storage_dir / "forecasts" / user_id / "action_forecasts.json", [])
        self.legacy_path = storage_dir / "users" / "demo_portfolio.json"
        self._lock = RLock()

    def read_settings(self) -> list[dict]:
        return self.settings.read()

    def append_settings(self, record: dict) -> None:
        with self._lock:
            records = self.settings.read()
            records.append(record)
            self.settings.write(records)

    def read_transactions(self) -> list[dict]:
        return self.transactions.read()

    def append_transaction(self, record: dict) -> None:
        with self._lock:
            records = self.transactions.read()
            records.append(record)
            self.transactions.write(records)

    def read_allocation_snapshots(self) -> list[dict]:
        return self.allocation_snapshots.read()

    def append_allocation_snapshot(self, record: dict) -> bool:
        """Append one immutable user-reported allocation snapshot."""
        with self._lock:
            records = self.allocation_snapshots.read()
            if any(
                item.get("snapshot_id") == record.get("snapshot_id")
                for item in records
            ):
                return False
            records.append(record)
            self.allocation_snapshots.write(records)
            return True

    def read_composite_snapshots(self) -> list[dict]:
        return self.composite_snapshots.read()

    def append_composite_snapshot(self, record: dict) -> bool:
        """Append a deterministic page-composition audit once."""
        with self._lock:
            records = self.composite_snapshots.read()
            if any(
                item.get("composite_snapshot_id")
                == record.get("composite_snapshot_id")
                for item in records
            ):
                return False
            records.append(record)
            self.composite_snapshots.write(records)
            return True

    def read_today_advice_h1_logs(self) -> list[dict]:
        return self.today_advice_h1_logs.read()

    def append_today_advice_h1_log(self, record: dict) -> bool:
        """Append one immutable H1 Today Advice record per issuance."""
        with self._lock:
            records = self.today_advice_h1_logs.read()
            if any(
                item.get("technical_issuance_id")
                == record.get("technical_issuance_id")
                for item in records
            ):
                return False
            records.append(record)
            self.today_advice_h1_logs.write(records)
            return True

    def replace_transaction(self, transaction_id: str, record: dict) -> bool:
        with self._lock:
            records = self.transactions.read()
            for index, item in enumerate(records):
                if item.get("transaction_id") == transaction_id:
                    records[index] = record
                    self.transactions.write(records)
                    return True
        return False

    def append_forecast(self, record: dict) -> None:
        with self._lock:
            records = self.forecasts.read()
            records.append(record)
            self.forecasts.write(records[-100:])

