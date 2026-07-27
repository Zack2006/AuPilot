from datetime import datetime

from pydantic import BaseModel, Field

from backend.app.core.constants import MarketRegime


class MarketSnapshot(BaseModel):
    ts_event_utc: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    market_regime: MarketRegime | None = None
    source_id: str = "databento"
    dataset: str
    schema_name: str
    symbol: str
    retrieved_at_utc: datetime
    content_sha256: str | None = Field(default=None, pattern=r"^[A-Fa-f0-9]{64}$")
    is_formal_data: bool = True
    quality_status: str = "VALID"


class MarketSeriesMetadata(BaseModel):
    source_id: str = "databento"
    dataset: str
    schema_name: str
    symbol: str
    stype_in: str
    retrieved_at_utc: datetime
    data_from_utc: datetime
    data_until_utc: datetime
    row_count: int = Field(ge=1)
    content_sha256: str
    is_formal_data: bool = True
    quality_status: str = "VALID"


class GoldDailyBar(BaseModel):
    ts_event_utc: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)


class GoldDailySeriesResponse(BaseModel):
    metadata: MarketSeriesMetadata
    items: list[dict]
