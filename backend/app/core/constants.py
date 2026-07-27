from enum import StrEnum


class ActionType(StrEnum):
    HOLD_CORE = "HOLD_CORE"
    REDUCE_TACTICAL = "REDUCE_TACTICAL"
    REBUY_TACTICAL = "REBUY_TACTICAL"
    NO_ACTION = "NO_ACTION"


class DirectionalBias(StrEnum):
    BULLISH = "BULLISH"
    SLIGHTLY_BULLISH = "SLIGHTLY_BULLISH"
    NEUTRAL = "NEUTRAL"
    SLIGHTLY_BEARISH = "SLIGHTLY_BEARISH"
    BEARISH = "BEARISH"


class MarketRegime(StrEnum):
    UPTREND = "UPTREND"
    RANGE_BOUND = "RANGE_BOUND"
    DOWNTREND = "DOWNTREND"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"


class RiskProfile(StrEnum):
    CONSERVATIVE = "CONSERVATIVE"
    BALANCED = "BALANCED"
    AGGRESSIVE = "AGGRESSIVE"


class CycleStatus(StrEnum):
    OPEN = "OPEN"
    PARTIALLY_REBOUGHT = "PARTIALLY_REBOUGHT"
    COMPLETED = "COMPLETED"
    INVALIDATED = "INVALIDATED"


class TransactionEventType(StrEnum):
    TRADE = "TRADE"
    OPENING_BALANCE = "OPENING_BALANCE"
    ADJUSTMENT = "ADJUSTMENT"


class TransactionSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class PositionBucket(StrEnum):
    CORE = "CORE"
    TACTICAL = "TACTICAL"


class TransactionSource(StrEnum):
    MANUAL = "MANUAL"
    IMPORT = "IMPORT"
    MIGRATION = "MIGRATION"
