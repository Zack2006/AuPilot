from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.core.constants import RiskProfile


class PortfolioInput(BaseModel):
    """User constraints used by the decision layer; ratios use decimal form."""

    model_config = ConfigDict(extra="forbid")

    product_type: str = Field(default="COMEX Gold Futures (Databento GC.v.0)", min_length=1, max_length=100)
    current_price: float = Field(gt=0)
    average_cost: float = Field(gt=0)
    total_position: float = Field(gt=0)
    min_core_ratio: float = Field(ge=0, le=1)
    max_tactical_ratio: float = Field(ge=0, le=1)
    max_single_adjustment: float = Field(ge=0, le=1)
    transaction_cost: float = Field(default=0.0002, ge=0, le=0.1)
    acceptable_drawdown: float = Field(default=0.1, gt=0, le=1)
    risk_profile: RiskProfile = RiskProfile.BALANCED

    @model_validator(mode="after")
    def validate_allocation_constraints(self) -> "PortfolioInput":
        if self.max_single_adjustment > self.max_tactical_ratio:
            raise ValueError("max_single_adjustment cannot exceed max_tactical_ratio")
        if self.min_core_ratio + self.max_tactical_ratio > 1.000001:
            raise ValueError("min_core_ratio plus max_tactical_ratio cannot exceed 1")
        if self.min_core_ratio <= 0 and self.max_tactical_ratio <= 0:
            raise ValueError("at least one position bucket must be available")
        return self
