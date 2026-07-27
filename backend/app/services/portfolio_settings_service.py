"""Immutable portfolio-settings version management. / 不可变持仓设置版本管理。

Purpose / 文件用途: migrate legacy demo settings and append new versions.
Inputs / 输入: validated PortfolioSettingsCreate and optional version numbers.
Outputs / 输出: current or historical PortfolioSettingsVersion values.
Business invariants / 业务约束: updates append; version numbers strictly increase.
Side effects / 副作用: first access may create migration version 1; updates write JSON.
Fallback behavior / 降级: corrupt storage raises an explicit application error.
"""

from datetime import datetime, timezone
from uuid import uuid4

from backend.app.core.exceptions import NotFoundError
from backend.app.repositories.portfolio_repository import PortfolioRepository
from backend.app.schemas.portfolio_settings import PortfolioSettingsCreate, PortfolioSettingsVersion


class PortfolioSettingsService:
    def __init__(self, repository: PortfolioRepository) -> None:
        self.repository = repository

    def history(self) -> list[PortfolioSettingsVersion]:
        """Return all versions in effective order without mutation. / 按生效顺序返回全部版本且不写入。"""
        self.ensure_migrated()
        return sorted(
            [PortfolioSettingsVersion.model_validate(item) for item in self.repository.read_settings()],
            key=lambda item: (item.version, item.effective_from),
        )

    def current(self, version: int | None = None) -> PortfolioSettingsVersion:
        """Select an explicit version or the latest effective version. / 选择指定版本或最新有效版本。"""
        records = self.history()
        if version is None:
            now = datetime.now(timezone.utc)
            effective = [record for record in records if (record.effective_from if record.effective_from.tzinfo else record.effective_from.replace(tzinfo=timezone.utc)) <= now]
            if not effective:
                raise NotFoundError("No portfolio settings version is effective yet")
            return effective[-1]
        for record in records:
            if record.version == version:
                return record
        raise NotFoundError(f"Portfolio settings version not found: {version}")

    def create(self, payload: PortfolioSettingsCreate) -> PortfolioSettingsVersion:
        """Append one version; never update an older JSON object. / 追加新版本，绝不覆盖旧对象。"""
        now = datetime.now(timezone.utc)
        next_version = self.current().version + 1
        record = PortfolioSettingsVersion(
            **payload.model_dump(), settings_id=f"settings_{uuid4().hex[:16]}",
            user_id=self.repository.user_id, version=next_version, effective_from=now, created_at=now,
        )
        self.repository.append_settings(record.model_dump(mode="json"))
        return record

    def ensure_migrated(self) -> None:
        """Create version 1 from the legacy file only when history is empty. / 仅在历史为空时由旧文件创建版本1。"""
        if self.repository.read_settings():
            return
        legacy = self._legacy()
        now = datetime.now(timezone.utc)
        payload = PortfolioSettingsCreate(
            product_type=legacy.get("product_type", "COMEX Gold Futures (Databento GC.v.0)"),
            instrument_type="FUTURES", currency="USD", quantity_unit="contract", contract_multiplier=100.0,
            min_core_ratio=legacy.get("min_core_ratio", 0.7), max_tactical_ratio=legacy.get("max_tactical_ratio", 0.3),
            max_single_adjustment=legacy.get("max_single_adjustment", 0.1),
            transaction_cost_rate=legacy.get("transaction_cost", 0.0002),
            acceptable_drawdown=legacy.get("acceptable_drawdown", 0.1),
            risk_profile=legacy.get("risk_profile", "BALANCED"),
            note="Migrated from demo_portfolio.json; version 1 preserves the original guardrails.",
        )
        record = PortfolioSettingsVersion(
            **payload.model_dump(), settings_id="settings_demo_v1", user_id=self.repository.user_id,
            version=1, effective_from=now, created_at=now,
        )
        self.repository.append_settings(record.model_dump(mode="json"))

    def _legacy(self) -> dict:
        import json
        if not self.repository.legacy_path.exists():
            return {}
        return json.loads(self.repository.legacy_path.read_text(encoding="utf-8"))

