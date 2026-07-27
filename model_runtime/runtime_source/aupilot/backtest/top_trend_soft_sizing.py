"""Causal TOP soft-sizing overlay for an already-issued action tape."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aupilot.backtest.pivot_baseline import validate_daily_ohlc
from aupilot.training.future_slot_side_regime_gate import (
    _causal_regime_features,
)

_REQUIRED_ACTION_COLUMNS = {
    "trade_date",
    "signed_delta",
    "side_code",
    "action_id",
}


def attach_registered_top_feature_anchors(
    *,
    actions: pd.DataFrame,
    boundary_tape: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the original issuance anchor to TOP actions by target bucket."""

    if missing := _REQUIRED_ACTION_COLUMNS - set(actions):
        raise ValueError(f"actions lack columns: {sorted(missing)}")
    required_tape = {
        "target_bucket",
        "feature_anchor_bucket",
        "horizon_index",
    }
    if missing := required_tape - set(boundary_tape):
        raise ValueError(f"boundary_tape lacks columns: {sorted(missing)}")
    result = actions.copy().reset_index(drop=True)
    result["trade_date"] = pd.to_datetime(
        result["trade_date"], errors="raise"
    ).dt.date
    top = result["side_code"].eq(-1)
    lookup = boundary_tape.loc[
        :,
        ["target_bucket", "feature_anchor_bucket", "horizon_index"],
    ].copy()
    lookup["target_bucket"] = pd.to_datetime(
        lookup["target_bucket"], errors="raise"
    ).dt.date
    lookup["feature_anchor_bucket"] = pd.to_datetime(
        lookup["feature_anchor_bucket"], errors="raise"
    ).dt.date
    if lookup["target_bucket"].duplicated().any():
        raise ValueError("boundary target buckets are not unique")
    top_lookup = (
        result.loc[top, ["trade_date", "boundary_horizon_index"]]
        .merge(
            lookup,
            left_on="trade_date",
            right_on="target_bucket",
            how="left",
            validate="one_to_one",
        )
        .set_index(result.index[top])
    )
    if top_lookup["feature_anchor_bucket"].isna().any():
        missing_dates = top_lookup.loc[
            top_lookup["feature_anchor_bucket"].isna(), "trade_date"
        ].tolist()
        raise ValueError(f"TOP actions lack registered anchors: {missing_dates}")
    if not top_lookup["boundary_horizon_index"].astype(int).eq(
        top_lookup["horizon_index"].astype(int)
    ).all():
        raise ValueError("TOP action horizon differs from issuance tape")
    result["feature_anchor_bucket"] = pd.NaT
    result["registered_anchor_horizon_index"] = pd.NA
    result.loc[top, "feature_anchor_bucket"] = top_lookup[
        "feature_anchor_bucket"
    ].to_numpy()
    result.loc[top, "registered_anchor_horizon_index"] = top_lookup[
        "horizon_index"
    ].astype(int).to_numpy()
    return result


def apply_top_trend_continuation_soft_sizing(
    *,
    actions: pd.DataFrame,
    daily: pd.DataFrame,
    attenuation_factor: float,
) -> pd.DataFrame:
    """Shrink, but never delete, TOP requests in causal trend continuation.

    The continuation condition is the frozen R54 condition evaluated at the
    registered feature anchor:

    * five-bucket return is positive;
    * five-bucket return exceeds the preceding five-bucket return; and
    * five-bucket volatility exceeds twenty-bucket volatility.

    BOTTOM rows and all action dates, identifiers and probabilities are
    preserved exactly.
    """

    required_actions = _REQUIRED_ACTION_COLUMNS | {"feature_anchor_bucket"}
    if missing := required_actions - set(actions):
        raise ValueError(f"actions lack columns: {sorted(missing)}")
    factor = float(attenuation_factor)
    if not np.isfinite(factor) or not 0.0 < factor < 1.0:
        raise ValueError("attenuation_factor must be finite and in (0, 1)")

    result = actions.copy().reset_index(drop=True)
    result["trade_date"] = pd.to_datetime(
        result["trade_date"], errors="raise"
    ).dt.date
    result["feature_anchor_bucket"] = pd.to_datetime(
        result["feature_anchor_bucket"], errors="raise"
    ).dt.date
    if result["trade_date"].duplicated().any():
        raise ValueError("action trade dates are not unique")
    if result["action_id"].astype(str).duplicated().any():
        raise ValueError("action IDs are not unique")
    side = pd.to_numeric(result["side_code"], errors="raise").astype(int)
    delta = pd.to_numeric(result["signed_delta"], errors="raise").astype(
        float
    )
    if not set(side).issubset({-1, 1}):
        raise ValueError("side_code must be -1 or 1")
    if not (
        delta.loc[side.eq(-1)].lt(0.0).all()
        and delta.loc[side.eq(1)].gt(0.0).all()
    ):
        raise ValueError("signed_delta direction differs from side_code")

    prices = validate_daily_ohlc(daily).reset_index(drop=True)
    features = _causal_regime_features(prices).set_index("trade_date")
    top_mask = side.eq(-1)
    top_anchors = result.loc[top_mask, "feature_anchor_bucket"]
    missing_anchors = sorted(set(top_anchors) - set(features.index))
    if missing_anchors:
        raise ValueError(
            "TOP feature anchors are outside the daily calendar: "
            f"{missing_anchors[:10]}"
        )
    top_features = features.loc[
        top_anchors,
        [
            "source_return_5",
            "source_prior_return_5",
            "source_volatility_5",
            "source_volatility_20",
        ],
    ].reset_index(drop=True)
    top_features.index = result.index[top_mask]
    for column in top_features:
        result[column] = np.nan
        result.loc[top_mask, column] = top_features[column]

    feature_columns = [
        "source_return_5",
        "source_prior_return_5",
        "source_volatility_5",
        "source_volatility_20",
    ]
    complete = result.loc[:, feature_columns].notna().all(axis=1)
    incomplete_top = top_mask & ~complete
    if incomplete_top.any():
        dates = result.loc[incomplete_top, "feature_anchor_bucket"].tolist()
        raise ValueError(f"TOP causal features are incomplete: {dates[:10]}")
    continuation = (
        top_mask
        & complete
        & result["source_return_5"].gt(0.0)
        & result["source_return_5"].gt(result["source_prior_return_5"])
        & result["source_volatility_5"].gt(
            result["source_volatility_20"]
        )
    )
    result["original_signed_delta"] = delta
    result["top_trend_continuation"] = continuation
    result["top_soft_sizing_applied"] = continuation
    result["top_soft_sizing_factor"] = np.where(
        continuation,
        factor,
        1.0,
    )
    result["signed_delta"] = (
        result["original_signed_delta"]
        * result["top_soft_sizing_factor"]
    )

    bottom_mask = side.eq(1)
    if not result.loc[bottom_mask, "signed_delta"].equals(
        result.loc[bottom_mask, "original_signed_delta"]
    ):
        raise AssertionError("BOTTOM sizing changed")
    if not (
        result.loc[top_mask, "signed_delta"].lt(0.0).all()
        and result.loc[bottom_mask, "signed_delta"].gt(0.0).all()
    ):
        raise AssertionError("soft sizing changed action direction")
    return result
