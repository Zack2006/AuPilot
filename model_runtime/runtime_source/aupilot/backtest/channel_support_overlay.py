"""Causal support/resistance sizing overlays for a frozen action tape."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aupilot.backtest.pivot_baseline import validate_daily_ohlc

TOP_RESTORE = "TOP_FALSE_BREAKOUT_RESTORE"
BOTTOM_POSITION = "BOTTOM_SUPPORT_POSITION_SIZING"
COMBINED = "TOP_RESTORE_PLUS_BOTTOM_POSITION"
CHANNEL_SUPPORT_CANDIDATES = (TOP_RESTORE, BOTTOM_POSITION, COMBINED)


@dataclass(frozen=True)
class ChannelSupportOverlayResult:
    actions: pd.DataFrame
    audit: dict[str, float | int | bool | str]


def _state(daily: pd.DataFrame) -> pd.DataFrame:
    prices = validate_daily_ohlc(daily).reset_index(drop=True)
    previous_close = prices["close"].shift(1)
    true_range = pd.concat(
        [
            prices["high"] - prices["low"],
            (prices["high"] - previous_close).abs(),
            (prices["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_20 = true_range.rolling(20, min_periods=20).mean()
    high_20 = prices["high"].rolling(20, min_periods=20).max()
    low_20 = prices["low"].rolling(20, min_periods=20).min()
    width_20 = high_20 - low_20
    prior_high_63 = (
        prices["high"].shift(1).rolling(63, min_periods=63).max()
    )
    position_20 = (prices["close"] - low_20) / width_20.where(
        width_20.gt(0.0)
    )
    output = pd.DataFrame(
        {
            "feature_anchor_bucket": prices["trade_date"],
            "source_support_position_20": position_20.clip(0.0, 1.0),
            "source_distance_to_high_atr_20": (
                high_20 - prices["close"]
            )
            / atr_20,
            "source_distance_to_low_atr_20": (
                prices["close"] - low_20
            )
            / atr_20,
            "source_upside_false_breakout_63": (
                prices["high"].gt(prior_high_63)
                & prices["close"].lt(prior_high_63)
            ),
        }
    )
    return output


def apply_channel_support_overlay(
    *,
    actions: pd.DataFrame,
    daily: pd.DataFrame,
    bottom_boundary_tape: pd.DataFrame,
    candidate_id: str,
) -> ChannelSupportOverlayResult:
    """Apply one registered support/resistance overlay configuration."""

    if candidate_id not in CHANNEL_SUPPORT_CANDIDATES:
        raise ValueError(f"unknown channel-support candidate: {candidate_id}")
    required_actions = {
        "trade_date",
        "feature_anchor_bucket",
        "boundary_horizon_index",
        "signed_delta",
        "original_signed_delta",
        "side_code",
        "action_id",
    }
    if missing := required_actions - set(actions):
        raise ValueError(f"actions lack columns: {sorted(missing)}")
    required_tape = {
        "target_bucket",
        "feature_anchor_bucket",
        "horizon_index",
    }
    if missing := required_tape - set(bottom_boundary_tape):
        raise ValueError(f"bottom_boundary_tape lacks: {sorted(missing)}")

    result = actions.copy().reset_index(drop=True)
    result["trade_date"] = pd.to_datetime(
        result["trade_date"], errors="raise"
    ).dt.date
    if result["trade_date"].duplicated().any():
        raise ValueError("action dates are not unique")
    if result["action_id"].astype(str).duplicated().any():
        raise ValueError("action IDs are not unique")
    side = pd.to_numeric(result["side_code"], errors="raise").astype(int)
    if not set(side).issubset({-1, 1}):
        raise ValueError("side_code must be -1 or 1")
    top = side.eq(-1)
    bottom = side.eq(1)

    top_anchor = pd.to_datetime(
        result["feature_anchor_bucket"], errors="coerce"
    ).dt.date
    if top_anchor.loc[top].isna().any():
        raise ValueError("a TOP action lacks its registered anchor")
    lookup = bottom_boundary_tape.loc[
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
        raise ValueError("bottom boundary target buckets are not unique")
    bottom_lookup = (
        result.loc[bottom, ["trade_date", "boundary_horizon_index"]]
        .merge(
            lookup,
            left_on="trade_date",
            right_on="target_bucket",
            how="left",
            validate="one_to_one",
        )
        .set_index(result.index[bottom])
    )
    if bottom_lookup["feature_anchor_bucket"].isna().any():
        raise ValueError("a BOTTOM action lacks its registered anchor")
    if not bottom_lookup["boundary_horizon_index"].astype(int).eq(
        bottom_lookup["horizon_index"].astype(int)
    ).all():
        raise ValueError("BOTTOM action horizon differs from issuance tape")

    anchor = top_anchor.copy()
    anchor.loc[bottom] = bottom_lookup[
        "feature_anchor_bucket"
    ].to_numpy()
    state = _state(daily).set_index("feature_anchor_bucket")
    missing_anchors = sorted(set(anchor) - set(state.index))
    if missing_anchors:
        raise ValueError(
            "action anchors are outside channel-support state: "
            f"{missing_anchors[:10]}"
        )
    joined = state.loc[anchor].reset_index(drop=True)
    joined.index = result.index
    for column in joined:
        if column != "feature_anchor_bucket":
            result[column] = joined[column]
    if result.loc[
        bottom,
        [
            "source_support_position_20",
            "source_distance_to_high_atr_20",
            "source_distance_to_low_atr_20",
        ],
    ].isna().any().any():
        raise ValueError("a BOTTOM action lacks 20-bucket support history")
    if result.loc[
        top, "source_upside_false_breakout_63"
    ].isna().any():
        raise ValueError("a TOP action lacks 63-bucket resistance history")

    parent_delta = pd.to_numeric(
        result["signed_delta"], errors="raise"
    ).astype(float)
    original_delta = pd.to_numeric(
        result["original_signed_delta"], errors="raise"
    ).astype(float)
    result["mn12_parent_signed_delta"] = parent_delta
    result["top_false_breakout_restore_applied"] = False
    result["bottom_support_position_multiplier"] = 1.0
    result["bottom_support_position_applied"] = False

    use_top = candidate_id in {TOP_RESTORE, COMBINED}
    use_bottom = candidate_id in {BOTTOM_POSITION, COMBINED}
    if use_top:
        restore = (
            top
            & result["source_upside_false_breakout_63"].astype(bool)
            & original_delta.lt(parent_delta)
        )
        result.loc[restore, "signed_delta"] = original_delta.loc[restore]
        result.loc[restore, "top_false_breakout_restore_applied"] = True
    if use_bottom:
        multiplier = (
            2.0 - result.loc[bottom, "source_support_position_20"]
        ).clip(1.0, 2.0)
        result.loc[
            bottom, "bottom_support_position_multiplier"
        ] = multiplier
        result.loc[bottom, "signed_delta"] = (
            parent_delta.loc[bottom] * multiplier
        )
        result.loc[
            bottom, "bottom_support_position_applied"
        ] = multiplier.gt(1.0)

    final_delta = pd.to_numeric(
        result["signed_delta"], errors="raise"
    ).astype(float)
    if not (
        final_delta.loc[top].lt(0.0).all()
        and final_delta.loc[bottom].gt(0.0).all()
    ):
        raise AssertionError("channel-support overlay changed direction")
    return ChannelSupportOverlayResult(
        actions=result,
        audit={
            "candidate_id": candidate_id,
            "top_restore_enabled": use_top,
            "bottom_position_sizing_enabled": use_bottom,
            "top_false_breakout_restore_rows": int(
                result["top_false_breakout_restore_applied"].sum()
            ),
            "bottom_support_position_rows": int(
                result["bottom_support_position_applied"].sum()
            ),
            "action_rows_preserved": len(result) == len(actions),
            "action_dates_preserved": result["trade_date"].equals(
                pd.to_datetime(actions["trade_date"]).dt.date
            ),
            "action_deletion_performed": False,
            "future_prices_used": False,
        },
    )
