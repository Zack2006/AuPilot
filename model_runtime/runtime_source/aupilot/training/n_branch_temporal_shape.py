"""Causal 20-bucket temporal-shape features for the AuPilot N branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aupilot.training import n_branch_hurdle_lightgbm as hurdle
from aupilot.training.future_slot_features import FUTURE_SLOT_FEATURE_COLUMNS
from aupilot.training.future_slot_technical_features import (
    TECHNICAL_EXHAUSTION_FEATURE_COLUMNS,
)
from aupilot.training.graded_future_slot import (
    PROBABILITY_COLUMNS,
    GradedProbeResult,
)
from aupilot.training.n_branch_boundary_parity_model import (
    fit_boundary_parity_graded_ordinal_probe,
)
from aupilot.training.n_branch_market_geometry import _validate_daily
from aupilot.training.n_branch_top_false_breakout import (
    MN14_FALSE_BREAKOUT_COMPLETE,
    MN14_H1_FEATURE_COLUMNS,
    augment_issuance_with_top_false_breakout,
)

MN18_MODEL_ID = "MN18_GN01_TEMPORAL_SHAPE_LIGHTGBM"
MN18_CANDIDATE_ID = "SQRT_EXPOSURE_TOP_FALSE_BREAKOUT_TEMPORAL_SHAPE"
MN18_TEMPORAL_SHAPE_COLUMNS = (
    "source_signed_path_efficiency_20",
    "source_return_acceleration_z_20",
    "source_log_trend_r2_20",
    "source_channel_z_20",
    "source_log_volatility_ratio_5_to_14",
    "source_log_range_ratio_5_to_15",
    "source_return_sign_flip_rate_20",
)
MN18_TEMPORAL_SHAPE_COMPLETE = "source_temporal_shape_complete"
MN18_H1_FEATURE_COLUMNS = (
    *MN14_H1_FEATURE_COLUMNS,
    *MN18_TEMPORAL_SHAPE_COLUMNS,
)


@dataclass(frozen=True)
class TemporalShapeResult:
    frame: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class TemporalShapeIssuanceResult:
    frame: pd.DataFrame
    source_features: pd.DataFrame
    audit: dict[str, Any]


def _safe_log_ratio(numerator: float, denominator: float) -> float:
    epsilon = 1.0e-12
    return float(np.log((numerator + epsilon) / (denominator + epsilon)))


def build_temporal_shape_features(daily: pd.DataFrame) -> TemporalShapeResult:
    """Build a fixed, scale-invariant description of the latest 20-bucket path."""

    frame = _validate_daily(daily)
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    log_close = np.log(close)
    previous_close = np.concatenate(([close[0]], close[:-1]))
    true_range_fraction = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - previous_close),
            np.abs(low - previous_close),
        ),
    ) / previous_close

    values = {
        column: np.full(len(frame), np.nan, dtype=float)
        for column in MN18_TEMPORAL_SHAPE_COLUMNS
    }
    x = np.arange(20, dtype=float)
    centered_x = x - x.mean()
    denominator = float(np.square(centered_x).sum())
    for position in range(19, len(frame)):
        path = log_close[position - 19 : position + 1]
        returns = np.diff(path)
        absolute_path = float(np.abs(returns).sum())
        net_return = float(path[-1] - path[0])
        values["source_signed_path_efficiency_20"][position] = (
            net_return / absolute_path if absolute_path > 0.0 else 0.0
        )

        return_scale = float(np.std(returns, ddof=1))
        recent_rate = float(returns[-5:].mean())
        prior_rate = float(returns[:-5].mean())
        values["source_return_acceleration_z_20"][position] = (
            (recent_rate - prior_rate) / return_scale
            if return_scale > 0.0
            else 0.0
        )

        path_mean = float(path.mean())
        centered_path = path - path_mean
        slope = float(np.dot(centered_x, centered_path) / denominator)
        fitted = path_mean + slope * centered_x
        residual = path - fitted
        total_sum_squares = float(np.square(centered_path).sum())
        residual_sum_squares = float(np.square(residual).sum())
        residual_scale = float(np.std(residual, ddof=1))
        values["source_log_trend_r2_20"][position] = (
            1.0 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0.0
            else 0.0
        )
        values["source_channel_z_20"][position] = (
            float(residual[-1]) / residual_scale
            if residual_scale > 0.0
            else 0.0
        )

        recent_volatility = float(np.std(returns[-5:], ddof=1))
        prior_volatility = float(np.std(returns[:-5], ddof=1))
        values["source_log_volatility_ratio_5_to_14"][position] = (
            _safe_log_ratio(recent_volatility, prior_volatility)
        )

        range_window = true_range_fraction[position - 19 : position + 1]
        values["source_log_range_ratio_5_to_15"][position] = _safe_log_ratio(
            float(range_window[-5:].mean()),
            float(range_window[:-5].mean()),
        )

        signs = np.sign(returns)
        values["source_return_sign_flip_rate_20"][position] = float(
            np.not_equal(signs[1:], signs[:-1]).mean()
        )

    output = pd.DataFrame({"trade_date": frame["trade_date"], **values})
    matrix = output.loc[:, list(MN18_TEMPORAL_SHAPE_COLUMNS)].to_numpy(float)
    complete = np.isfinite(matrix).all(axis=1)
    output[MN18_TEMPORAL_SHAPE_COMPLETE] = complete
    if int(complete.sum()) != max(len(output) - 19, 0):
        raise AssertionError("MN18 temporal-shape warm-up identity changed")
    if complete.any():
        complete_values = output.loc[
            complete, list(MN18_TEMPORAL_SHAPE_COLUMNS)
        ]
        if (
            not np.isfinite(complete_values.to_numpy(float)).all()
            or not complete_values["source_signed_path_efficiency_20"]
            .between(-1.0 - 1.0e-12, 1.0 + 1.0e-12)
            .all()
            or not complete_values["source_log_trend_r2_20"]
            .between(0.0 - 1.0e-12, 1.0 + 1.0e-12)
            .all()
            or not complete_values["source_return_sign_flip_rate_20"]
            .between(0.0, 1.0)
            .all()
        ):
            raise AssertionError("MN18 temporal-shape features are invalid")
    return TemporalShapeResult(
        frame=output,
        audit={
            "daily_rows": len(output),
            "complete_rows": int(complete.sum()),
            "warmup_rows": int((~complete).sum()),
            "feature_columns": list(MN18_TEMPORAL_SHAPE_COLUMNS),
            "feature_count": len(MN18_TEMPORAL_SHAPE_COLUMNS),
            "fixed_window": 20,
            "m06_inner_selected_20_bucket_asset_used": True,
            "future_rows_used": False,
            "future_label_fields_used": False,
            "absolute_price_features_used": False,
            "scale_invariant_by_construction": True,
            "input_contract": "CANONICAL_UTC_DAILY_OHLC_ONLY",
        },
    )


def augment_issuance_with_mn18_temporal_shape(
    daily: pd.DataFrame,
    issuance_rows: pd.DataFrame,
) -> TemporalShapeIssuanceResult:
    """Attach MN14 false-breakout and MN18 temporal-shape source features."""

    false_breakout = augment_issuance_with_top_false_breakout(
        daily, issuance_rows
    )
    rows = false_breakout.frame.copy().reset_index(drop=True)
    rows["_row_order"] = np.arange(len(rows), dtype=int)
    rows["feature_anchor_bucket"] = pd.to_datetime(
        rows["feature_anchor_bucket"], errors="coerce", utc=True
    ).dt.date
    temporal = build_temporal_shape_features(daily)
    lookup = temporal.frame.copy()
    lookup["feature_anchor_position"] = np.arange(len(lookup), dtype=int)
    lookup = lookup.rename(columns={"trade_date": "feature_anchor_bucket"})
    joined = rows.merge(
        lookup.loc[
            :,
            [
                "feature_anchor_bucket",
                "feature_anchor_position",
                MN18_TEMPORAL_SHAPE_COMPLETE,
                *MN18_TEMPORAL_SHAPE_COLUMNS,
            ],
        ],
        on=["feature_anchor_bucket", "feature_anchor_position"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if joined[MN18_TEMPORAL_SHAPE_COMPLETE].isna().any():
        raise ValueError("MN18 issuance lacks its temporal-shape source row")
    joined = (
        joined.sort_values("_row_order", kind="stable")
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )
    if len(joined) != len(rows):
        raise AssertionError("MN18 feature join changed issuance row count")
    if not joined.groupby("issuance_id", sort=False).size().eq(21).all():
        raise AssertionError("MN18 feature join split an issuance")
    source = false_breakout.source_features.merge(
        temporal.frame,
        on="trade_date",
        how="inner",
        validate="one_to_one",
    )
    return TemporalShapeIssuanceResult(
        frame=joined,
        source_features=source,
        audit={
            "false_breakout": false_breakout.audit,
            "temporal_shape": temporal.audit,
            "issuance_rows": len(joined),
            "issuances": int(joined["issuance_id"].nunique()),
            "complete_issuances": int(
                joined.loc[
                    joined[MN14_FALSE_BREAKOUT_COMPLETE].astype(bool)
                    & joined[MN18_TEMPORAL_SHAPE_COMPLETE].astype(bool),
                    "issuance_id",
                ].nunique()
            ),
            "issuance_all_in_all_out": True,
            "row_order_preserved": True,
        },
    )


def fit_mn18_h1_hurdle_ordinal(
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    """Fit the frozen hurdle architecture with the MN18 temporal shape."""

    complete_train = train.loc[
        train[MN14_FALSE_BREAKOUT_COMPLETE].astype(bool)
        & train[MN18_TEMPORAL_SHAPE_COMPLETE].astype(bool)
    ].reset_index(drop=True)
    complete_test = test.loc[
        test[MN14_FALSE_BREAKOUT_COMPLETE].astype(bool)
        & test[MN18_TEMPORAL_SHAPE_COMPLETE].astype(bool)
    ].reset_index(drop=True)
    if len(complete_test) != len(test):
        raise ValueError("MN18 outer test contains warm-up rows")
    if complete_train.empty:
        raise ValueError("MN18 complete training rows are empty")
    if MN18_CANDIDATE_ID in hurdle.MN02_CANDIDATE_FEATURES:
        raise RuntimeError("MN18 candidate ID already registered")
    hurdle.MN02_CANDIDATE_FEATURES[MN18_CANDIDATE_ID] = (
        MN18_H1_FEATURE_COLUMNS
    )
    try:
        result = hurdle.fit_mn02_h1_hurdle_ordinal(
            complete_train,
            complete_test,
            candidate_id=MN18_CANDIDATE_ID,
        )
    finally:
        hurdle.MN02_CANDIDATE_FEATURES.pop(MN18_CANDIDATE_ID, None)
    return type(result)(
        predictions=result.predictions,
        audit={
            **result.audit,
            "model_id": MN18_MODEL_ID,
            "candidate_id": MN18_CANDIDATE_ID,
            "temporal_shape_feature_only_challenge": True,
            "complete_train_rows": len(complete_train),
            "complete_test_rows": len(complete_test),
        },
    )


def fit_mn18_full_horizon(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> GradedProbeResult:
    """Keep the frozen display model and replace only automatic h1."""

    all_features = (
        *FUTURE_SLOT_FEATURE_COLUMNS,
        *TECHNICAL_EXHAUSTION_FEATURE_COLUMNS,
    )
    base = fit_boundary_parity_graded_ordinal_probe(
        train,
        test,
        model_id="FIXED_SMALL_LGBM_ORDINAL",
        feature_columns=all_features,
        near_horizon_maximum=5,
    )
    h1 = fit_mn18_h1_hurdle_ordinal(train, test)
    output = base.predictions.copy()
    h1_mask = output["horizon_index"].eq(1)
    replacement = [
        *PROBABILITY_COLUMNS,
        "display_class",
        "train_prior_top_action",
        "train_prior_bottom_action",
    ]
    output.loc[h1_mask, replacement] = h1.predictions.loc[
        :, replacement
    ].to_numpy()
    return GradedProbeResult(
        predictions=output,
        audit={
            **base.audit,
            "h1_model": h1.audit,
            "h1_only_replacement": True,
            "far_display_contract_unchanged": True,
        },
    )

