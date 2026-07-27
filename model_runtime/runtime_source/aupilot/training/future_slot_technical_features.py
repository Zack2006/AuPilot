from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

TECHNICAL_EXHAUSTION_FEATURE_COLUMNS = (
    "source_log_return_5",
    "source_close_sma_distance_20",
    "source_rsi_14_centered",
    "source_close_location_value",
)


@dataclass(frozen=True)
class TechnicalFeatureResult:
    frame: pd.DataFrame
    audit: dict[str, Any]


def _daily_technical_state(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "open", "high", "low", "close"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"R30 daily input missing: {sorted(missing)}")
    state = daily.loc[:, sorted(required)].copy()
    timestamp = pd.to_datetime(state["trade_date"], errors="coerce", utc=True)
    if (
        timestamp.isna().any()
        or not timestamp.eq(timestamp.dt.normalize()).all()
        or timestamp.duplicated().any()
        or not timestamp.is_monotonic_increasing
    ):
        raise ValueError("R30 trade_date must be unique ordered UTC midnight")
    state["feature_anchor_bucket"] = timestamp.dt.date
    for column in ("open", "high", "low", "close"):
        state[column] = pd.to_numeric(state[column], errors="coerce")
    values = state[["open", "high", "low", "close"]].to_numpy(float)
    if (
        not np.isfinite(values).all()
        or (values <= 0.0).any()
        or not state["high"].ge(state[["open", "close"]].max(axis=1)).all()
        or not state["low"].le(state[["open", "close"]].min(axis=1)).all()
        or not state["high"].ge(state["low"]).all()
    ):
        raise ValueError("R30 daily OHLC is invalid")
    close = state["close"]
    state["source_log_return_5"] = np.log(close / close.shift(5))
    sma20 = close.rolling(20, min_periods=20).mean()
    state["source_close_sma_distance_20"] = close / sma20 - 1.0
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=14).mean()
    relative_strength = gain / loss
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    rsi = rsi.mask(loss.eq(0.0) & gain.gt(0.0), 100.0)
    rsi = rsi.mask(loss.eq(0.0) & gain.eq(0.0), 50.0)
    state["source_rsi_14_centered"] = (rsi - 50.0) / 50.0
    daily_range = state["high"] - state["low"]
    state["source_close_location_value"] = np.where(
        daily_range.gt(0.0),
        (
            2.0 * state["close"] - state["high"] - state["low"]
        )
        / daily_range,
        0.0,
    )
    return state[
        ["feature_anchor_bucket", *TECHNICAL_EXHAUSTION_FEATURE_COLUMNS]
    ]


def build_technical_exhaustion_feature_table(
    daily: pd.DataFrame,
    base_features: pd.DataFrame,
) -> TechnicalFeatureResult:
    """Add four source-day-only technical states without changing rows."""

    if "feature_anchor_bucket" not in base_features.columns:
        raise ValueError("R30 base features lack feature_anchor_bucket")
    state = _daily_technical_state(daily)
    warmup_counts = {
        column: int(state[column].isna().sum())
        for column in TECHNICAL_EXHAUSTION_FEATURE_COLUMNS
    }
    state.loc[:, TECHNICAL_EXHAUSTION_FEATURE_COLUMNS] = state.loc[
        :,
        TECHNICAL_EXHAUSTION_FEATURE_COLUMNS,
    ].fillna(0.0)
    base = base_features.copy().reset_index(drop=True)
    anchor = pd.to_datetime(
        base["feature_anchor_bucket"],
        errors="coerce",
        utc=True,
    )
    if anchor.isna().any() or not anchor.eq(anchor.dt.normalize()).all():
        raise ValueError("R30 feature anchors must be UTC midnight")
    base["feature_anchor_bucket"] = anchor.dt.date
    output = base.merge(
        state,
        on="feature_anchor_bucket",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if len(output) != len(base):
        raise AssertionError("R30 feature merge changed row count")
    array = output.loc[
        :,
        TECHNICAL_EXHAUSTION_FEATURE_COLUMNS,
    ].to_numpy(float)
    if not np.isfinite(array).all():
        raise ValueError("R30 technical features contain non-finite values")
    if not output["issuance_id"].equals(base["issuance_id"]):
        raise AssertionError("R30 feature merge changed issuance order")
    return TechnicalFeatureResult(
        frame=output,
        audit={
            "rows": len(output),
            "issuances": int(output["issuance_id"].nunique()),
            "daily_rows": len(state),
            "feature_columns": list(
                TECHNICAL_EXHAUSTION_FEATURE_COLUMNS
            ),
            "warmup_zero_imputation_counts": warmup_counts,
            "warmup_policy": "SOURCE_PREFIX_UNAVAILABLE_TO_ZERO",
            "future_prices_used": False,
            "intraday_inputs_used": False,
            "row_identity_preserved": True,
        },
    )
