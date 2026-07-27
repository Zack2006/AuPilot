from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aupilot.labels.main_swing import MainSwingPolicy, scan_main_swing_path

FUTURE_SLOT_FEATURE_COLUMNS = (
    "source_active_trend_code",
    "source_reversal_progress",
    "source_vol_adjusted_oriented_oc_reversal",
    "horizon_scaled_0_1",
)
DAILY_UNIT_ID = "CANONICAL_GC_UTC_DAILY_BUCKET_V1"
PRIOR_VOLATILITY_RETURN_COUNT = 20
MINIMUM_SOURCE_HISTORY_BUCKETS = 22
HORIZON_BUCKETS = 21
_TREND_CODE = {"UP": 1.0, "DOWN": -1.0}
_REVERSAL_ORIENTATION = {"UP": -1.0, "DOWN": 1.0}


@dataclass(frozen=True)
class FutureSlotFeatureResult:
    frame: pd.DataFrame
    source_features: pd.DataFrame
    excluded_issuances: pd.DataFrame
    audit: dict[str, Any]


def _validate_daily(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "open", "high", "low", "close"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"Missing future-slot daily columns: {sorted(missing)}")
    frame = daily.copy().reset_index(drop=True)
    frame["trade_date"] = pd.to_datetime(
        frame["trade_date"],
        errors="coerce",
        utc=True,
    )
    if (
        frame["trade_date"].isna().any()
        or not frame["trade_date"].eq(frame["trade_date"].dt.normalize()).all()
        or frame["trade_date"].duplicated().any()
        or not frame["trade_date"].is_monotonic_increasing
    ):
        raise ValueError(
            "Future-slot daily dates must be unique ordered UTC-midnight buckets"
        )
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    values = frame.loc[:, ["open", "high", "low", "close"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("Future-slot OHLC must be finite and positive")
    if (
        frame["high"] < frame[["open", "close"]].max(axis=1)
    ).any() or (
        frame["low"] > frame[["open", "close"]].min(axis=1)
    ).any():
        raise ValueError("Future-slot OHLC ordering is invalid")
    return frame


def _validate_issuance(rows: pd.DataFrame) -> pd.DataFrame:
    required = {
        "issuance_id",
        "feature_anchor_bucket",
        "feature_anchor_position",
        "horizon_index",
        "target_bucket",
        "target_event_group_id",
        "target_label",
        "target_label_available_at_utc",
    }
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(
            f"Missing future-slot issuance columns: {sorted(missing)}"
        )
    if {"open", "high", "low", "close"} & set(rows.columns):
        raise ValueError("Future-slot issuance rows contain target OHLC")
    frame = rows.copy().reset_index(drop=True)
    for column in ("feature_anchor_bucket", "target_bucket"):
        parsed = pd.to_datetime(frame[column], errors="coerce", utc=True)
        if (
            parsed.isna().any()
            or not parsed.eq(parsed.dt.normalize()).all()
        ):
            raise ValueError(f"{column} must be a UTC-midnight bucket")
        frame[column] = parsed.dt.date
    frame["target_label_available_at_utc"] = pd.to_datetime(
        frame["target_label_available_at_utc"],
        errors="coerce",
        utc=True,
    )
    if frame["target_label_available_at_utc"].isna().any():
        raise ValueError("Future-slot issuance label maturity is missing")
    sizes = frame.groupby("issuance_id", sort=False).size()
    if not sizes.eq(HORIZON_BUCKETS).all():
        raise ValueError("Every future-slot issuance must contain 21 rows")
    expected = tuple(range(1, HORIZON_BUCKETS + 1))
    horizons = frame.groupby("issuance_id", sort=False)[
        "horizon_index"
    ].apply(tuple)
    if not horizons.map(lambda value: value == expected).all():
        raise ValueError("Future-slot issuance horizons must be exactly 1...21")
    anchor_counts = frame.groupby("issuance_id", sort=False)[
        "feature_anchor_bucket"
    ].nunique()
    if not anchor_counts.eq(1).all():
        raise ValueError("An issuance contains multiple feature anchors")
    return frame


def _state_identity_audit(
    daily: pd.DataFrame,
    reconstructed: pd.DataFrame,
) -> dict[str, int]:
    required = {
        "active_trend",
        "candidate_type",
        "candidate_at",
        "candidate_price",
    }
    if not required <= set(daily.columns):
        return {
            "registered_state_columns_present": 0,
            "active_trend_mismatches": 0,
            "candidate_type_mismatches": 0,
            "candidate_at_mismatches": 0,
            "candidate_price_mismatches": 0,
        }
    observed = daily.loc[:, list(required)].copy()
    rebuilt = reconstructed.loc[:, list(required)].copy()
    observed_candidate_at = pd.to_datetime(
        observed["candidate_at"],
        errors="coerce",
    ).dt.date
    rebuilt_candidate_at = pd.to_datetime(
        rebuilt["candidate_at"],
        errors="coerce",
    ).dt.date
    candidate_at_equal = (
        observed_candidate_at.eq(rebuilt_candidate_at)
        | (observed_candidate_at.isna() & rebuilt_candidate_at.isna())
    )
    observed_price = pd.to_numeric(
        observed["candidate_price"],
        errors="coerce",
    )
    rebuilt_price = pd.to_numeric(
        rebuilt["candidate_price"],
        errors="coerce",
    )
    price_equal = np.isclose(
        observed_price.fillna(0.0).to_numpy(dtype=float),
        rebuilt_price.fillna(0.0).to_numpy(dtype=float),
        rtol=0.0,
        atol=1.0e-10,
    ) & (observed_price.isna().to_numpy() == rebuilt_price.isna().to_numpy())
    return {
        "registered_state_columns_present": 1,
        "active_trend_mismatches": int(
            observed["active_trend"]
            .astype(str)
            .ne(rebuilt["active_trend"].astype(str))
            .sum()
        ),
        "candidate_type_mismatches": int(
            observed["candidate_type"]
            .astype("string")
            .fillna("<NA>")
            .ne(
                rebuilt["candidate_type"]
                .astype("string")
                .fillna("<NA>")
            )
            .sum()
        ),
        "candidate_at_mismatches": int((~candidate_at_equal).sum()),
        "candidate_price_mismatches": int((~price_equal).sum()),
    }


def build_future_slot_feature_table(
    daily: pd.DataFrame,
    issuance_rows: pd.DataFrame,
) -> FutureSlotFeatureResult:
    """Build the frozen causal R26 source-state feature table."""

    source = _validate_daily(daily)
    issuance = _validate_issuance(issuance_rows)
    path_input = source.loc[:, ["trade_date", "open"]].rename(
        columns={"open": "close"}
    )
    path_result = scan_main_swing_path(
        path_input,
        MainSwingPolicy(
            reversal_threshold=0.05,
            threshold_mode="arithmetic",
        ),
    )
    state = path_result.state_tape.copy().reset_index(drop=True)
    state_identity = _state_identity_audit(source, state)
    if sum(
        value
        for key, value in state_identity.items()
        if key.endswith("_mismatches")
    ):
        raise ValueError("Rebuilt R04 causal state differs from registered state")

    features = pd.DataFrame(
        {
            "feature_anchor_bucket": source["trade_date"].dt.date,
            "feature_anchor_position": np.arange(len(source), dtype=int),
            "source_open": source["open"].to_numpy(dtype=float),
            "source_close": source["close"].to_numpy(dtype=float),
            "source_active_trend": state["active_trend"].astype(str),
            "source_candidate_at": state["candidate_at"],
            "source_candidate_price": pd.to_numeric(
                state["candidate_price"],
                errors="coerce",
            ),
        }
    )
    features["source_active_trend_code"] = features[
        "source_active_trend"
    ].map(_TREND_CODE)
    features["_orientation"] = features["source_active_trend"].map(
        _REVERSAL_ORIENTATION
    )
    features["_source_log_oc_return"] = np.log(
        features["source_close"] / features["source_open"]
    )
    features["_source_oriented_oc_reversal"] = (
        features["_orientation"] * features["_source_log_oc_return"]
    )

    log_return = np.log(source["close"].astype(float)).diff()
    features["_prior_volatility_20"] = (
        log_return.shift(1)
        .rolling(
            PRIOR_VOLATILITY_RETURN_COUNT,
            min_periods=PRIOR_VOLATILITY_RETURN_COUNT,
        )
        .std(ddof=0)
        .to_numpy()
    )
    denominator = features["_prior_volatility_20"].where(
        features["_prior_volatility_20"] > 0.0
    )
    features["source_vol_adjusted_oriented_oc_reversal"] = (
        features["_source_oriented_oc_reversal"] / denominator
    )

    up = features["source_active_trend"].eq("UP")
    down = features["source_active_trend"].eq("DOWN")
    features["source_reversal_progress"] = np.nan
    features.loc[up, "source_reversal_progress"] = (
        1.0
        - features.loc[up, "source_open"]
        / features.loc[up, "source_candidate_price"]
    )
    features.loc[down, "source_reversal_progress"] = (
        features.loc[down, "source_open"]
        / features.loc[down, "source_candidate_price"]
        - 1.0
    )
    features["source_state_known"] = up | down
    features["source_feature_complete"] = (
        features.loc[:, list(FUTURE_SLOT_FEATURE_COLUMNS[:-1])]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .all(axis=1)
        & features["source_state_known"]
    )
    complete_values = features.loc[
        features["source_feature_complete"],
        list(FUTURE_SLOT_FEATURE_COLUMNS[:-1]),
    ].to_numpy(dtype=float)
    if not np.isfinite(complete_values).all():
        raise AssertionError("Future-slot source features are non-finite")
    progress = features.loc[
        features["source_feature_complete"],
        "source_reversal_progress",
    ]
    if (progress < -1.0e-12).any() or (progress >= 0.05 + 1.0e-10).any():
        raise AssertionError("R26 reversal progress is outside causal bounds")

    model = issuance.merge(
        features.loc[
            :,
            [
                "feature_anchor_bucket",
                "feature_anchor_position",
                "source_active_trend",
                "source_active_trend_code",
                "source_reversal_progress",
                "source_vol_adjusted_oriented_oc_reversal",
                "source_state_known",
                "source_feature_complete",
            ],
        ],
        on=["feature_anchor_bucket", "feature_anchor_position"],
        how="left",
        validate="many_to_one",
    )
    if model["source_feature_complete"].isna().any():
        raise ValueError("An R26 issuance lacks its source feature anchor")
    excluded_ids = model.loc[
        ~model["source_feature_complete"].astype(bool),
        "issuance_id",
    ].drop_duplicates()
    excluded = (
        model.loc[model["issuance_id"].isin(excluded_ids)]
        .groupby("issuance_id", sort=False)
        .agg(
            feature_anchor_bucket=("feature_anchor_bucket", "first"),
            source_active_trend=("source_active_trend", "first"),
            rows=("horizon_index", "size"),
        )
        .reset_index()
    )
    model = model.loc[
        ~model["issuance_id"].isin(excluded_ids)
    ].reset_index(drop=True)
    model["horizon_scaled_0_1"] = (
        pd.to_numeric(model["horizon_index"], errors="raise") - 1.0
    ) / float(HORIZON_BUCKETS - 1)
    values = model.loc[:, list(FUTURE_SLOT_FEATURE_COLUMNS)].to_numpy(
        dtype=float
    )
    if not np.isfinite(values).all():
        raise AssertionError("R26 model feature table contains non-finite values")
    if not model.groupby("issuance_id", sort=False).size().eq(
        HORIZON_BUCKETS
    ).all():
        raise AssertionError("R26 feature filtering split an issuance")

    audit = {
        "daily_unit_id": DAILY_UNIT_ID,
        "daily_rows": len(source),
        "input_issuance_rows": len(issuance),
        "input_issuances": int(issuance["issuance_id"].nunique()),
        "model_ready_rows": len(model),
        "model_ready_issuances": int(model["issuance_id"].nunique()),
        "excluded_source_state_rows": int(
            features["source_feature_complete"].eq(False).sum()
        ),
        "excluded_issuance_rows": int(
            len(issuance) - len(model)
        ),
        "excluded_issuances": len(excluded),
        "unknown_source_state_rows": int(
            features["source_active_trend"].eq("UNKNOWN").sum()
        ),
        "model_source_trend_counts": {
            str(key): int(value)
            for key, value in (
                model[
                    ["issuance_id", "source_active_trend"]
                ]
                .drop_duplicates()["source_active_trend"]
                .value_counts()
                .items()
            )
        },
        "feature_columns": list(FUTURE_SLOT_FEATURE_COLUMNS),
        "prior_volatility_return_count": PRIOR_VOLATILITY_RETURN_COUNT,
        "minimum_source_history_buckets": MINIMUM_SOURCE_HISTORY_BUCKETS,
        "horizon_buckets": HORIZON_BUCKETS,
        "target_ohlc_used_as_features": False,
        "intraday_inputs_used": False,
        "future_prices_used_as_features": False,
        "registered_state_identity": state_identity,
    }
    return FutureSlotFeatureResult(
        frame=model,
        source_features=features.drop(
            columns=[
                "_orientation",
                "_source_log_oc_return",
                "_source_oriented_oc_reversal",
                "_prior_volatility_20",
            ]
        ).reset_index(drop=True),
        excluded_issuances=excluded,
        audit=audit,
    )
