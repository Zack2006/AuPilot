"""Causal TOP false-breakout features and the MN14 h1 model adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aupilot.training import n_branch_hurdle_lightgbm as hurdle
from aupilot.training.graded_future_slot import (
    PROBABILITY_COLUMNS,
    GradedProbeResult,
)
from aupilot.training.graded_hurdle_lightgbm import M01_MODEL_ID
from aupilot.training.n_branch_boundary_parity_model import (
    fit_boundary_parity_graded_ordinal_probe,
)
from aupilot.training.n_branch_hurdle_lightgbm import (
    MN02_BASELINE_FEATURE_COLUMNS,
)
from aupilot.training.n_branch_market_geometry import _validate_daily

MN14_MODEL_ID = "MN14_GN01_TOP_FALSE_BREAKOUT_LIGHTGBM"
MN14_CANDIDATE_ID = "SQRT_EXPOSURE_TOP_FALSE_BREAKOUT"
MN14_FALSE_BREAKOUT_COLUMNS = (
    "source_upside_false_breakout_21",
    "source_upside_false_breakout_63",
    "source_upside_breach_atr_63",
    "source_upside_rejection_atr_63",
)
MN14_FALSE_BREAKOUT_COMPLETE = "source_top_false_breakout_complete"
MN14_H1_FEATURE_COLUMNS = (
    *MN02_BASELINE_FEATURE_COLUMNS,
    *MN14_FALSE_BREAKOUT_COLUMNS,
)


@dataclass(frozen=True)
class TopFalseBreakoutResult:
    frame: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class TopFalseBreakoutIssuanceResult:
    frame: pd.DataFrame
    source_features: pd.DataFrame
    audit: dict[str, Any]


def build_top_false_breakout_features(
    daily: pd.DataFrame,
) -> TopFalseBreakoutResult:
    """Build fixed 21/63-bucket upside break-and-reject features."""

    frame = _validate_daily(daily)
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_21 = true_range.rolling(21, min_periods=21).mean()
    prior_high_21 = (
        frame["high"].shift(1).rolling(21, min_periods=21).max()
    )
    prior_high_63 = (
        frame["high"].shift(1).rolling(63, min_periods=63).max()
    )
    broke_21 = frame["high"].gt(prior_high_21)
    broke_63 = frame["high"].gt(prior_high_63)
    false_21 = broke_21 & frame["close"].lt(prior_high_21)
    false_63 = broke_63 & frame["close"].lt(prior_high_63)
    safe_atr = atr_21.where(atr_21.gt(0.0))
    breach_63 = (
        (frame["high"] - prior_high_63).clip(lower=0.0) / safe_atr
    ).where(false_63, 0.0)
    rejection_63 = (
        (prior_high_63 - frame["close"]).clip(lower=0.0) / safe_atr
    ).where(false_63, 0.0)
    complete = prior_high_63.notna() & safe_atr.notna()
    output = pd.DataFrame(
        {
            "trade_date": frame["trade_date"],
            "source_upside_false_breakout_21": false_21.astype(float),
            "source_upside_false_breakout_63": false_63.astype(float),
            "source_upside_breach_atr_63": breach_63,
            "source_upside_rejection_atr_63": rejection_63,
            MN14_FALSE_BREAKOUT_COMPLETE: complete,
        }
    )
    output.loc[~complete, list(MN14_FALSE_BREAKOUT_COLUMNS)] = np.nan
    values = output.loc[
        complete, list(MN14_FALSE_BREAKOUT_COLUMNS)
    ].to_numpy(dtype=float)
    if (
        len(values)
        and (
            not np.isfinite(values).all()
            or (values < 0.0).any()
            or not output.loc[
                complete,
                [
                    "source_upside_false_breakout_21",
                    "source_upside_false_breakout_63",
                ],
            ]
            .isin([0.0, 1.0])
            .all()
            .all()
        )
    ):
        raise AssertionError("MN14 false-breakout features are invalid")
    if int(complete.sum()) != max(len(output) - 63, 0):
        raise AssertionError("MN14 false-breakout warm-up changed")
    return TopFalseBreakoutResult(
        frame=output,
        audit={
            "daily_rows": len(output),
            "complete_rows": int(complete.sum()),
            "warmup_rows": int((~complete).sum()),
            "feature_columns": list(MN14_FALSE_BREAKOUT_COLUMNS),
            "feature_count": len(MN14_FALSE_BREAKOUT_COLUMNS),
            "fixed_windows": [21, 63],
            "future_rows_used": False,
            "future_label_fields_used": False,
            "absolute_price_features_used": False,
            "scale_invariant_by_construction": True,
            "input_contract": "CANONICAL_UTC_DAILY_OHLC_ONLY",
        },
    )


def augment_issuance_with_top_false_breakout(
    daily: pd.DataFrame,
    issuance_rows: pd.DataFrame,
) -> TopFalseBreakoutIssuanceResult:
    """Attach one causal source-state row to every 21-slot issuance."""

    required = {
        "issuance_id",
        "feature_anchor_bucket",
        "feature_anchor_position",
        "horizon_index",
    }
    if missing := required - set(issuance_rows):
        raise ValueError(f"MN14 issuance lacks: {sorted(missing)}")
    rows = issuance_rows.copy().reset_index(drop=True)
    rows["_row_order"] = np.arange(len(rows), dtype=int)
    rows["feature_anchor_bucket"] = pd.to_datetime(
        rows["feature_anchor_bucket"], errors="coerce", utc=True
    ).dt.date
    source = build_top_false_breakout_features(daily)
    lookup = source.frame.copy()
    lookup["feature_anchor_position"] = np.arange(len(lookup), dtype=int)
    lookup = lookup.rename(columns={"trade_date": "feature_anchor_bucket"})
    joined = rows.merge(
        lookup.loc[
            :,
            [
                "feature_anchor_bucket",
                "feature_anchor_position",
                MN14_FALSE_BREAKOUT_COMPLETE,
                *MN14_FALSE_BREAKOUT_COLUMNS,
            ],
        ],
        on=["feature_anchor_bucket", "feature_anchor_position"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if joined[MN14_FALSE_BREAKOUT_COMPLETE].isna().any():
        raise ValueError("MN14 issuance lacks its source-state row")
    joined = (
        joined.sort_values("_row_order", kind="stable")
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )
    if len(joined) != len(rows):
        raise AssertionError("MN14 feature join changed row count")
    if not joined.groupby("issuance_id", sort=False).size().eq(21).all():
        raise AssertionError("MN14 feature join split an issuance")
    return TopFalseBreakoutIssuanceResult(
        frame=joined,
        source_features=source.frame,
        audit={
            **source.audit,
            "issuance_rows": len(joined),
            "issuances": int(joined["issuance_id"].nunique()),
            "complete_issuances": int(
                joined.loc[
                    joined[MN14_FALSE_BREAKOUT_COMPLETE].astype(bool),
                    "issuance_id",
                ].nunique()
            ),
            "issuance_all_in_all_out": True,
            "row_order_preserved": True,
        },
    )


def fit_mn14_h1_hurdle_ordinal(
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    """Fit the frozen hurdle architecture with TOP-breakout features."""

    if MN14_FALSE_BREAKOUT_COMPLETE not in train or (
        MN14_FALSE_BREAKOUT_COMPLETE not in test
    ):
        raise ValueError("MN14 train/test lacks completeness flag")
    complete_train = train.loc[
        train[MN14_FALSE_BREAKOUT_COMPLETE].astype(bool)
    ].reset_index(drop=True)
    complete_test = test.loc[
        test[MN14_FALSE_BREAKOUT_COMPLETE].astype(bool)
    ].reset_index(drop=True)
    if len(complete_test) != len(test):
        raise ValueError("MN14 outer test contains warm-up rows")
    if complete_train.empty:
        raise ValueError("MN14 complete training rows are empty")
    if MN14_CANDIDATE_ID in hurdle.MN02_CANDIDATE_FEATURES:
        raise RuntimeError("MN14 candidate ID already registered")
    hurdle.MN02_CANDIDATE_FEATURES[MN14_CANDIDATE_ID] = (
        MN14_H1_FEATURE_COLUMNS
    )
    try:
        result = hurdle.fit_mn02_h1_hurdle_ordinal(
            complete_train,
            complete_test,
            candidate_id=MN14_CANDIDATE_ID,
        )
    finally:
        hurdle.MN02_CANDIDATE_FEATURES.pop(MN14_CANDIDATE_ID, None)
    audit = {
        **result.audit,
        "model_id": MN14_MODEL_ID,
        "candidate_id": MN14_CANDIDATE_ID,
        "top_false_breakout_feature_only_challenge": True,
        "complete_train_rows": len(complete_train),
        "complete_test_rows": len(complete_test),
    }
    return type(result)(predictions=result.predictions, audit=audit)


def fit_mn14_full_horizon(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    all_feature_columns: tuple[str, ...],
) -> GradedProbeResult:
    """Keep the frozen display model and replace only automatic h1."""

    base = fit_boundary_parity_graded_ordinal_probe(
        train,
        test,
        model_id="FIXED_SMALL_LGBM_ORDINAL",
        feature_columns=all_feature_columns,
        near_horizon_maximum=5,
    )
    h1 = fit_mn14_h1_hurdle_ordinal(train, test)
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
            "model_id": MN14_MODEL_ID,
            "candidate_id": MN14_CANDIDATE_ID,
            "architecture_parent": M01_MODEL_ID,
            "h1": h1.audit,
            "h2_h21": base.audit,
            "automatic_action_horizon": 1,
            "outer_labels_used_for_fit_or_candidate_selection": False,
        },
    )
