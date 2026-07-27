"""Low-capacity causal market geometry for the AuPilot N branch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

N_BRANCH_MARKET_GEOMETRY_COLUMNS = (
    "source_log_slope_total_21",
    "source_log_slope_total_63",
    "source_log_slope_total_126",
    "source_log_trend_r2_63",
    "source_log_trend_r2_126",
    "source_channel_z_63",
    "source_distance_to_high_atr_63",
    "source_distance_to_low_atr_63",
)
N_BRANCH_MARKET_GEOMETRY_COMPLETE = "source_market_geometry_complete"
_PRICE_COLUMNS = ("open", "high", "low", "close")


@dataclass(frozen=True)
class NBranchMarketGeometryResult:
    frame: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class NBranchGeometryIssuanceResult:
    frame: pd.DataFrame
    source_features: pd.DataFrame
    audit: dict[str, Any]


def _validate_daily(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", *_PRICE_COLUMNS}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"N-branch geometry missing: {sorted(missing)}")
    frame = daily.loc[:, ["trade_date", *_PRICE_COLUMNS]].copy()
    parsed = pd.to_datetime(frame["trade_date"], errors="coerce", utc=True)
    if (
        parsed.isna().any()
        or not parsed.eq(parsed.dt.normalize()).all()
        or parsed.duplicated().any()
        or not parsed.is_monotonic_increasing
    ):
        raise ValueError("N-branch geometry requires ordered UTC daily buckets")
    frame["trade_date"] = parsed.dt.date
    for column in _PRICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    values = frame.loc[:, _PRICE_COLUMNS].to_numpy(dtype=float)
    if (
        not np.isfinite(values).all()
        or (values <= 0.0).any()
        or not frame["high"].ge(
            frame[["open", "close"]].max(axis=1)
        ).all()
        or not frame["low"].le(
            frame[["open", "close"]].min(axis=1)
        ).all()
    ):
        raise ValueError("N-branch geometry received invalid OHLC")
    return frame.reset_index(drop=True)


def _rolling_regression(
    values: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    slope_total = np.full(len(values), np.nan, dtype=float)
    r_squared = np.full(len(values), np.nan, dtype=float)
    channel_z = np.full(len(values), np.nan, dtype=float)
    x = np.arange(window, dtype=float)
    centered_x = x - x.mean()
    denominator = float(np.square(centered_x).sum())
    for position in range(window - 1, len(values)):
        y = values[position - window + 1 : position + 1]
        mean = float(y.mean())
        centered_y = y - mean
        slope = float(np.dot(centered_x, centered_y) / denominator)
        fitted = mean + slope * centered_x
        residual = y - fitted
        total_sum_squares = float(np.square(centered_y).sum())
        residual_sum_squares = float(np.square(residual).sum())
        residual_scale = float(np.std(residual, ddof=1))
        slope_total[position] = slope * window
        r_squared[position] = (
            1.0 - residual_sum_squares / total_sum_squares
            if total_sum_squares > 0.0
            else 0.0
        )
        channel_z[position] = (
            float(residual[-1]) / residual_scale
            if residual_scale > 0.0
            else 0.0
        )
    return slope_total, r_squared, channel_z


def build_n_branch_market_geometry(
    daily: pd.DataFrame,
) -> NBranchMarketGeometryResult:
    """Build fixed 21/63/126-bucket causal, scale-invariant geometry."""

    frame = _validate_daily(daily)
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    log_close = np.log(close)
    regressions = {
        window: _rolling_regression(log_close, window)
        for window in (21, 63, 126)
    }
    previous_close = np.concatenate(([close[0]], close[:-1]))
    true_range = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - previous_close),
            np.abs(low - previous_close),
        ),
    )
    atr_21 = (
        pd.Series(true_range)
        .rolling(21, min_periods=21)
        .mean()
        .to_numpy(dtype=float)
    )
    rolling_high_63 = (
        pd.Series(high)
        .rolling(63, min_periods=63)
        .max()
        .to_numpy(dtype=float)
    )
    rolling_low_63 = (
        pd.Series(low)
        .rolling(63, min_periods=63)
        .min()
        .to_numpy(dtype=float)
    )
    output = pd.DataFrame(
        {
            "trade_date": frame["trade_date"],
            "source_log_slope_total_21": regressions[21][0],
            "source_log_slope_total_63": regressions[63][0],
            "source_log_slope_total_126": regressions[126][0],
            "source_log_trend_r2_63": regressions[63][1],
            "source_log_trend_r2_126": regressions[126][1],
            "source_channel_z_63": regressions[63][2],
            "source_distance_to_high_atr_63": (
                rolling_high_63 - close
            )
            / atr_21,
            "source_distance_to_low_atr_63": (
                close - rolling_low_63
            )
            / atr_21,
        }
    )
    values = output.loc[
        :, list(N_BRANCH_MARKET_GEOMETRY_COLUMNS)
    ].to_numpy(
        dtype=float
    )
    complete = np.isfinite(values).all(axis=1)
    output[N_BRANCH_MARKET_GEOMETRY_COMPLETE] = complete
    if complete.sum() != max(len(output) - 125, 0):
        raise AssertionError("N-branch geometry warm-up identity changed")
    if complete.any():
        complete_values = values[complete]
        r_squared_values = output.loc[
            complete,
            ["source_log_trend_r2_63", "source_log_trend_r2_126"],
        ]
        if (
            not np.isfinite(complete_values).all()
            or (
                output.loc[
                    complete,
                    [
                        "source_distance_to_high_atr_63",
                        "source_distance_to_low_atr_63",
                    ],
                ]
                < -1.0e-12
            )
            .any()
            .any()
            or not (
                (r_squared_values >= 0.0)
                & (r_squared_values <= 1.0)
            ).all().all()
        ):
            raise AssertionError("N-branch market geometry is invalid")
    return NBranchMarketGeometryResult(
        frame=output,
        audit={
            "daily_rows": len(output),
            "complete_rows": int(complete.sum()),
            "warmup_rows": int((~complete).sum()),
            "feature_columns": list(N_BRANCH_MARKET_GEOMETRY_COLUMNS),
            "feature_count": len(N_BRANCH_MARKET_GEOMETRY_COLUMNS),
            "fixed_windows": [21, 63, 126],
            "future_rows_used": False,
            "future_label_fields_used": False,
            "absolute_price_features_used": False,
            "scale_invariant_by_construction": True,
            "input_contract": "CANONICAL_UTC_DAILY_OHLC_ONLY",
        },
    )


def augment_issuance_with_n_branch_market_geometry(
    daily: pd.DataFrame,
    issuance_rows: pd.DataFrame,
) -> NBranchGeometryIssuanceResult:
    required = {
        "issuance_id",
        "feature_anchor_bucket",
        "feature_anchor_position",
        "horizon_index",
    }
    missing = required - set(issuance_rows.columns)
    if missing:
        raise ValueError(
            f"N-branch geometry issuance missing: {sorted(missing)}"
        )
    rows = issuance_rows.copy().reset_index(drop=True)
    rows["_row_order"] = np.arange(len(rows), dtype=int)
    rows["feature_anchor_bucket"] = pd.to_datetime(
        rows["feature_anchor_bucket"],
        errors="coerce",
        utc=True,
    ).dt.date
    source = build_n_branch_market_geometry(daily)
    lookup = source.frame.copy()
    lookup["feature_anchor_position"] = np.arange(len(lookup), dtype=int)
    lookup = lookup.rename(columns={"trade_date": "feature_anchor_bucket"})
    joined = rows.merge(
        lookup.loc[
            :,
            [
                "feature_anchor_bucket",
                "feature_anchor_position",
                N_BRANCH_MARKET_GEOMETRY_COMPLETE,
                *N_BRANCH_MARKET_GEOMETRY_COLUMNS,
            ],
        ],
        on=["feature_anchor_bucket", "feature_anchor_position"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if joined[N_BRANCH_MARKET_GEOMETRY_COMPLETE].isna().any():
        raise ValueError("N-branch issuance lacks its source geometry row")
    joined = (
        joined.sort_values("_row_order", kind="stable")
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )
    if len(joined) != len(rows):
        raise AssertionError("N-branch geometry changed issuance row count")
    if not joined.groupby("issuance_id", sort=False).size().eq(21).all():
        raise AssertionError("N-branch geometry split an issuance")
    return NBranchGeometryIssuanceResult(
        frame=joined,
        source_features=source.frame,
        audit={
            **source.audit,
            "issuance_rows": len(joined),
            "issuances": int(joined["issuance_id"].nunique()),
            "complete_issuances": int(
                joined.loc[
                    joined[N_BRANCH_MARKET_GEOMETRY_COMPLETE].astype(bool),
                    "issuance_id",
                ].nunique()
            ),
            "issuance_all_in_all_out": True,
            "row_order_preserved": True,
        },
    )
