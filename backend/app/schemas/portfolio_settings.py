"""Versioned portfolio guardrails. / 可版本化的持仓护栏模型。"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.core.constants import RiskProfile


class PortfolioSettingsCreate(BaseModel):
    """Editable settings only; market and holdings are deliberately excluded. / 仅包含可编辑设置。"""

    model_config = ConfigDict(extra="forbid")
    product_type: str = Field(min_length=1, max_length=100)
    instrument_type: str = Field(default="FUTURES", min_length=1, max_length=50)
    currency: str = Field(default="USD", min_length=3, max_length=8)
    quantity_unit: str = Field(default="contract", min_length=1, max_length=30)
    contract_multiplier: float = Field(default=100.0, gt=0)
    min_core_ratio: float = Field(default=0.7, ge=0, le=1)
    max_tactical_ratio: float = Field(default=0.3, ge=0, le=1)
    max_single_adjustment: float = Field(default=0.1, ge=0, le=1)
    transaction_cost_rate: float = Field(
        default=0.0002,
        ge=0,
        le=0.1,
    )
    acceptable_drawdown: float = Field(default=0.1, gt=0, le=1)
    risk_profile: RiskProfile = RiskProfile.BALANCED
    note: str = Field(default="", max_length=500)

    @model_validator(mode="after")
    def validate_guardrails(self) -> "PortfolioSettingsCreate":
        if self.max_single_adjustment > self.max_tactical_ratio:
            raise ValueError("max_single_adjustment cannot exceed max_tactical_ratio")
        if self.min_core_ratio + self.max_tactical_ratio > 1.000001:
            raise ValueError("min_core_ratio plus max_tactical_ratio cannot exceed 1")
        return self


class PortfolioSettingsVersion(PortfolioSettingsCreate):
    settings_id: str
    user_id: str
    version: int = Field(ge=1)
    effective_from: datetime
    created_at: datetime
