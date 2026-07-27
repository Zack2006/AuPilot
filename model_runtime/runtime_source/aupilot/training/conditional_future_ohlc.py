"""Conditional 21-slot OHLC targets and low-capacity regression models.

The product supplies only completed canonical UTC daily OHLC history.  A
scenario label is an internal query (TOP/BOTTOM/NORMAL), not an observed
future feature.  Targets are scale-free candle coordinates and are rebuilt
into a legal OHLC envelope after prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from aupilot.backtest.pivot_baseline import validate_daily_ohlc
from aupilot.training.future_slot_features import (
    FUTURE_SLOT_FEATURE_COLUMNS,
    build_future_slot_feature_table,
)
from aupilot.training.future_slot_technical_features import (
    TECHNICAL_EXHAUSTION_FEATURE_COLUMNS,
    build_technical_exhaustion_feature_table,
)

SCENARIO_LABELS = (
    "TOP_ACTION_ZONE",
    "BOTTOM_ACTION_ZONE",
    "NORMAL",
)
ACTION_SCENARIO_LABELS = SCENARIO_LABELS[:2]
SCENARIO_INDEX = {
    value: index for index, value in enumerate(SCENARIO_LABELS)
}
OHLC_FIELDS = ("open", "high", "low", "close")
TARGET_TRANSFORM_COLUMNS = (
    "target_open_log_return",
    "target_close_log_return",
    "target_upper_wick_log",
    "target_lower_wick_log",
)
VOLATILITY_FEATURE_COLUMNS = (
    "source_log_return_20",
    "source_realized_volatility_20",
    "source_atr_fraction_14",
    "source_range_fraction",
)
SCENARIO_FEATURE_COLUMNS = (
    "scenario_is_top",
    "scenario_is_bottom",
)
MODEL_FEATURE_COLUMNS = (
    *FUTURE_SLOT_FEATURE_COLUMNS,
    *TECHNICAL_EXHAUSTION_FEATURE_COLUMNS,
    *VOLATILITY_FEATURE_COLUMNS,
    *SCENARIO_FEATURE_COLUMNS,
)
MODEL_ID = "TASK3_5_CONDITIONAL_21_SLOT_OHLC_LIGHTGBM_V1"
BASELINE_ID = "TASK3_5_CONDITIONAL_HORIZON_MEDIAN_V1"
LIGHTGBM_PARAMETERS: dict[str, Any] = {
    "objective": "regression_l1",
    "n_estimators": 120,
    "learning_rate": 0.03,
    "num_leaves": 3,
    "max_depth": 2,
    "min_data_in_leaf": 50,
    "reg_alpha": 0.0,
    "reg_lambda": 5.0,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "max_bin": 31,
    "random_state": 20260725,
    "n_jobs": 4,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


@dataclass(frozen=True)
class ConditionalOhlcFrame:
    frame: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class ConditionalMedianBundle:
    medians: dict[tuple[str, int], tuple[float, ...]]
    class_fallbacks: dict[str, tuple[float, ...]]
    global_fallback: tuple[float, ...]


@dataclass(frozen=True)
class ConditionalLightgbmResult:
    predictions: pd.DataFrame
    model_strings: dict[str, str]
    audit: dict[str, Any]


def _volatility_state(daily: pd.DataFrame) -> pd.DataFrame:
    frame = validate_daily_ohlc(daily)
    close = frame["close"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    open_value = frame["open"].to_numpy(dtype=float)
    previous_close = np.roll(close, 1)
    previous_close[0] = np.nan
    log_close = pd.Series(np.log(close))
    true_range = np.maximum.reduce(
        [
            high - low,
            np.abs(high - previous_close),
            np.abs(low - previous_close),
        ]
    )
    output = pd.DataFrame(
        {
            "feature_anchor_bucket": frame["trade_date"],
            "source_log_return_20": (
                log_close - log_close.shift(20)
            ).to_numpy(),
            "source_realized_volatility_20": log_close.diff()
            .rolling(20, min_periods=20)
            .std(ddof=0)
            .to_numpy(),
            "source_atr_fraction_14": pd.Series(true_range)
            .rolling(14, min_periods=14)
            .mean()
            .to_numpy()
            / close,
            "source_range_fraction": (high - low) / open_value,
        }
    )
    return output


def _join_prices(
    daily: pd.DataFrame,
    rows: pd.DataFrame,
) -> pd.DataFrame:
    frame = validate_daily_ohlc(daily)
    source = frame.loc[:, ["trade_date", "close"]].rename(
        columns={
            "trade_date": "feature_anchor_bucket",
            "close": "source_close",
        }
    )
    target = frame.loc[:, ["trade_date", *OHLC_FIELDS]].rename(
        columns={
            "trade_date": "target_bucket",
            **{field: f"target_{field}" for field in OHLC_FIELDS},
        }
    )
    output = rows.merge(
        source,
        on="feature_anchor_bucket",
        how="left",
        validate="many_to_one",
        sort=False,
    ).merge(
        target,
        on="target_bucket",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    price_columns = [
        "source_close",
        *(f"target_{field}" for field in OHLC_FIELDS),
    ]
    prices = output.loc[:, price_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if (
        prices.isna().any().any()
        or not np.isfinite(prices.to_numpy(dtype=float)).all()
        or (prices <= 0.0).any().any()
    ):
        raise ValueError("Conditional OHLC price join is incomplete")
    output.loc[:, price_columns] = prices
    return output


def build_conditional_ohlc_frame(
    daily: pd.DataFrame,
    issuance_features: pd.DataFrame,
) -> ConditionalOhlcFrame:
    """Attach causal source states and future OHLC training targets."""

    required = {
        "issuance_id",
        "feature_anchor_bucket",
        "horizon_index",
        "target_bucket",
        "target_label",
        *FUTURE_SLOT_FEATURE_COLUMNS,
    }
    missing = required - set(issuance_features.columns)
    if missing:
        raise ValueError(
            f"Conditional OHLC issuance fields missing: {sorted(missing)}"
        )
    if {f"target_{field}" for field in OHLC_FIELDS} & set(
        issuance_features.columns
    ):
        raise ValueError("Issuance feature input already contains target OHLC")
    rows = issuance_features.copy().reset_index(drop=True)
    for column in ("feature_anchor_bucket", "target_bucket"):
        rows[column] = pd.to_datetime(
            rows[column],
            errors="raise",
            utc=True,
        ).dt.date
    if not rows["target_label"].isin(SCENARIO_LABELS).all():
        raise ValueError("Conditional OHLC target label is invalid")
    technical = build_technical_exhaustion_feature_table(daily, rows)
    volatility = _volatility_state(daily)
    volatility["feature_anchor_bucket"] = pd.to_datetime(
        volatility["feature_anchor_bucket"],
        errors="raise",
    ).dt.date
    output = technical.frame.merge(
        volatility,
        on="feature_anchor_bucket",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if len(output) != len(rows):
        raise AssertionError("Conditional OHLC source join changed rows")
    warmup = {
        column: int(output[column].isna().sum())
        for column in VOLATILITY_FEATURE_COLUMNS
    }
    output.loc[:, VOLATILITY_FEATURE_COLUMNS] = output.loc[
        :, VOLATILITY_FEATURE_COLUMNS
    ].fillna(0.0)
    output = _join_prices(daily, output)

    source_close = output["source_close"].to_numpy(dtype=float)
    target_open = output["target_open"].to_numpy(dtype=float)
    target_high = output["target_high"].to_numpy(dtype=float)
    target_low = output["target_low"].to_numpy(dtype=float)
    target_close = output["target_close"].to_numpy(dtype=float)
    output["target_open_log_return"] = np.log(
        target_open / source_close
    )
    output["target_close_log_return"] = np.log(
        target_close / source_close
    )
    output["target_upper_wick_log"] = np.log(
        target_high / np.maximum(target_open, target_close)
    )
    output["target_lower_wick_log"] = np.log(
        np.minimum(target_open, target_close) / target_low
    )
    output["scenario_is_top"] = output["target_label"].eq(
        "TOP_ACTION_ZONE"
    ).astype(float)
    output["scenario_is_bottom"] = output["target_label"].eq(
        "BOTTOM_ACTION_ZONE"
    ).astype(float)
    output["target_class_index"] = output["target_label"].map(
        SCENARIO_INDEX
    ).astype(int)
    values = output.loc[
        :,
        [*MODEL_FEATURE_COLUMNS, *TARGET_TRANSFORM_COLUMNS],
    ].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Conditional OHLC frame contains non-finite values")
    if (
        output["target_upper_wick_log"].lt(-1.0e-12).any()
        or output["target_lower_wick_log"].lt(-1.0e-12).any()
    ):
        raise AssertionError("Conditional OHLC wick target is negative")
    if not output.groupby("issuance_id", sort=False).size().eq(21).all():
        raise AssertionError("Conditional OHLC frame split an issuance")
    return ConditionalOhlcFrame(
        frame=output,
        audit={
            "rows": len(output),
            "issuances": int(output["issuance_id"].nunique()),
            "target_buckets": int(output["target_bucket"].nunique()),
            "action_rows": int(
                output["target_label"].isin(ACTION_SCENARIO_LABELS).sum()
            ),
            "action_target_buckets": int(
                output.loc[
                    output["target_label"].isin(ACTION_SCENARIO_LABELS),
                    "target_bucket",
                ].nunique()
            ),
            "event_groups": int(
                output["target_event_group_id"].nunique()
            ),
            "model_features": list(MODEL_FEATURE_COLUMNS),
            "targets": list(TARGET_TRANSFORM_COLUMNS),
            "volatility_warmup_zero_counts": warmup,
            "target_ohlc_used_as_feature": False,
            "future_label_used_as_observed_feature": False,
            "scenario_label_is_internal_conditional_query": True,
            "intraday_inputs_used": False,
            "rag_or_macro_inputs_used": False,
        },
    )


def build_online_conditional_query(
    daily: pd.DataFrame,
    future_buckets: tuple[object, ...],
) -> pd.DataFrame:
    """Build 21 source-only rows for an as-of production forecast."""

    source = validate_daily_ohlc(daily)
    if len(future_buckets) != 21:
        raise ValueError("Conditional OHLC online schedule must have 21 slots")
    parsed = pd.to_datetime(
        pd.Series(future_buckets),
        errors="coerce",
        utc=True,
    )
    if (
        parsed.isna().any()
        or not parsed.eq(parsed.dt.normalize()).all()
        or parsed.duplicated().any()
        or not parsed.is_monotonic_increasing
    ):
        raise ValueError(
            "Conditional OHLC online schedule must be unique ordered UTC buckets"
        )
    target_dates = tuple(parsed.dt.date)
    latest = source.iloc[-1]["trade_date"]
    if target_dates[0] <= latest:
        raise ValueError("Conditional OHLC online schedule is not future-only")
    anchor_position = len(source) - 1
    issuance = pd.DataFrame(
        {
            "issuance_id": ["ONLINE-CONDITIONAL-OHLC"] * 21,
            "feature_anchor_bucket": [latest] * 21,
            "feature_anchor_position": [anchor_position] * 21,
            "horizon_index": np.arange(1, 22, dtype=int),
            "target_bucket": target_dates,
            "target_event_group_id": ["UNOBSERVED-FUTURE"] * 21,
            "target_label": ["NORMAL"] * 21,
            "target_label_available_at_utc": parsed
            + pd.Timedelta(1, unit="D"),
        }
    )
    base = build_future_slot_feature_table(source, issuance).frame
    if len(base) != 21:
        raise RuntimeError("Conditional OHLC online source state is unavailable")
    technical = build_technical_exhaustion_feature_table(source, base).frame
    volatility = _volatility_state(source)
    volatility["feature_anchor_bucket"] = pd.to_datetime(
        volatility["feature_anchor_bucket"],
        errors="raise",
    ).dt.date
    output = technical.merge(
        volatility,
        on="feature_anchor_bucket",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    output.loc[:, VOLATILITY_FEATURE_COLUMNS] = output.loc[
        :, VOLATILITY_FEATURE_COLUMNS
    ].fillna(0.0)
    output["source_close"] = float(source.iloc[-1]["close"])
    values = output.loc[
        :,
        [
            *FUTURE_SLOT_FEATURE_COLUMNS,
            *TECHNICAL_EXHAUSTION_FEATURE_COLUMNS,
            *VOLATILITY_FEATURE_COLUMNS,
            "source_close",
        ],
    ].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(
            "Conditional OHLC online features are non-finite"
        )
    return output


def with_scenario(
    frame: pd.DataFrame,
    scenario_label: str,
) -> pd.DataFrame:
    """Return a query copy with one explicit conditional scenario."""

    if scenario_label not in SCENARIO_LABELS:
        raise ValueError("Conditional OHLC scenario is invalid")
    output = frame.copy().reset_index(drop=True)
    output["scenario_label"] = scenario_label
    output["scenario_is_top"] = float(
        scenario_label == "TOP_ACTION_ZONE"
    )
    output["scenario_is_bottom"] = float(
        scenario_label == "BOTTOM_ACTION_ZONE"
    )
    return output


def predict_conditional_lightgbm_strings(
    model_strings: dict[str, str],
    query: pd.DataFrame,
) -> pd.DataFrame:
    """Predict queries from the four frozen native LightGBM strings."""

    if set(model_strings) != set(TARGET_TRANSFORM_COLUMNS):
        raise ValueError("Conditional LightGBM model string roles changed")
    matrix = _scenario_matrix(query)
    prediction = np.column_stack(
        [
            lgb.Booster(model_str=model_strings[column]).predict(matrix)
            for column in TARGET_TRANSFORM_COLUMNS
        ]
    ).astype(float)
    if prediction.shape != (
        len(query),
        len(TARGET_TRANSFORM_COLUMNS),
    ) or not np.isfinite(prediction).all():
        raise ValueError("Conditional LightGBM restored output is invalid")
    candle, audit = reconstruct_conditional_ohlc(
        query["source_close"].to_numpy(dtype=float),
        prediction,
    )
    output = query.copy().reset_index(drop=True)
    for index, target_column in enumerate(TARGET_TRANSFORM_COLUMNS):
        output[
            f"predicted_{target_column.removeprefix('target_')}"
        ] = prediction[:, index]
    output = pd.concat([output, candle], axis=1)
    output["price_model_id"] = MODEL_ID
    output["candle_legal"] = bool(audit["candle_legal"])
    return output


def reconstruct_conditional_ohlc(
    source_close: np.ndarray,
    transformed: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Map four scale-free coordinates to a positive legal candle."""

    origin = np.asarray(source_close, dtype=float).reshape(-1)
    values = np.asarray(transformed, dtype=float)
    if (
        values.shape != (len(origin), len(TARGET_TRANSFORM_COLUMNS))
        or not np.isfinite(origin).all()
        or not np.isfinite(values).all()
        or (origin <= 0.0).any()
    ):
        raise ValueError("Conditional OHLC reconstruction input is invalid")
    wick = values[:, 2:].copy()
    clipped = np.maximum(wick, 0.0)
    open_value = origin * np.exp(values[:, 0])
    close = origin * np.exp(values[:, 1])
    high = np.maximum(open_value, close) * np.exp(clipped[:, 0])
    low = np.minimum(open_value, close) * np.exp(-clipped[:, 1])
    output = pd.DataFrame(
        {
            "predicted_open": open_value,
            "predicted_high": high,
            "predicted_low": low,
            "predicted_close": close,
        }
    )
    legal = (
        output["predicted_high"]
        .ge(output[["predicted_open", "predicted_close"]].max(axis=1))
        .all()
        and output["predicted_low"]
        .le(output[["predicted_open", "predicted_close"]].min(axis=1))
        .all()
        and output["predicted_high"].ge(output["predicted_low"]).all()
        and (output > 0.0).all().all()
    )
    if not legal:
        raise AssertionError("Conditional OHLC reconstruction is illegal")
    return output, {
        "rows": len(output),
        "negative_wick_predictions_clipped": int((wick < 0.0).sum()),
        "candle_legal": bool(legal),
    }


def _scenario_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(MODEL_FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Conditional OHLC model features missing: {sorted(missing)}"
        )
    values = frame.loc[:, MODEL_FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Conditional OHLC model features are non-finite")
    return values


def fit_conditional_median(
    train: pd.DataFrame,
) -> ConditionalMedianBundle:
    """Fit a fold-local class-by-horizon transformed-candle baseline."""

    if train.empty:
        raise ValueError("Conditional median training is empty")
    grouped = train.groupby(
        ["target_label", "horizon_index"],
        sort=True,
    )
    medians = {
        (str(label), int(horizon)): tuple(
            group.loc[:, TARGET_TRANSFORM_COLUMNS]
            .median()
            .to_numpy(dtype=float)
        )
        for (label, horizon), group in grouped
    }
    class_fallbacks = {
        str(label): tuple(
            group.loc[:, TARGET_TRANSFORM_COLUMNS]
            .median()
            .to_numpy(dtype=float)
        )
        for label, group in train.groupby("target_label", sort=True)
    }
    global_fallback = tuple(
        train.loc[:, TARGET_TRANSFORM_COLUMNS]
        .median()
        .to_numpy(dtype=float)
    )
    return ConditionalMedianBundle(
        medians=medians,
        class_fallbacks=class_fallbacks,
        global_fallback=global_fallback,
    )


def predict_conditional_median(
    bundle: ConditionalMedianBundle,
    query: pd.DataFrame,
) -> pd.DataFrame:
    """Predict and reconstruct one or more explicit scenario rows."""

    required = {"scenario_label", "horizon_index", "source_close"}
    missing = required - set(query.columns)
    if missing:
        raise ValueError(
            f"Conditional median query missing: {sorted(missing)}"
        )
    transformed = np.asarray(
        [
            bundle.medians.get(
                (str(row.scenario_label), int(row.horizon_index)),
                bundle.class_fallbacks.get(
                    str(row.scenario_label),
                    bundle.global_fallback,
                ),
            )
            for row in query.itertuples(index=False)
        ],
        dtype=float,
    )
    candle, audit = reconstruct_conditional_ohlc(
        query["source_close"].to_numpy(dtype=float),
        transformed,
    )
    output = query.copy().reset_index(drop=True)
    for index, column in enumerate(TARGET_TRANSFORM_COLUMNS):
        output[f"predicted_{column.removeprefix('target_')}"] = transformed[
            :, index
        ]
    output = pd.concat([output, candle], axis=1)
    output["price_model_id"] = BASELINE_ID
    output["candle_legal"] = bool(audit["candle_legal"])
    return output


def _fold_class_weights(train: pd.DataFrame) -> np.ndarray:
    counts = train["target_label"].value_counts()
    if set(counts.index) != set(SCENARIO_LABELS):
        raise ValueError("Conditional OHLC training lacks a target class")
    largest = float(counts.max())
    mapping = {
        label: min(5.0, max(1.0, np.sqrt(largest / float(count))))
        for label, count in counts.items()
    }
    return train["target_label"].map(mapping).to_numpy(dtype=float)


def fit_conditional_lightgbm(
    train: pd.DataFrame,
    query: pd.DataFrame,
) -> ConditionalLightgbmResult:
    """Fit four fixed regressors and predict explicit scenario queries."""

    if train.empty or query.empty:
        raise ValueError("Conditional LightGBM train/query is empty")
    x_train = _scenario_matrix(train)
    x_query = _scenario_matrix(query)
    weight = _fold_class_weights(train)
    predictions = np.empty(
        (len(query), len(TARGET_TRANSFORM_COLUMNS)),
        dtype=float,
    )
    strings: dict[str, str] = {}
    target_audit: dict[str, Any] = {}
    for index, target_column in enumerate(TARGET_TRANSFORM_COLUMNS):
        target = pd.to_numeric(
            train[target_column],
            errors="coerce",
        ).to_numpy(dtype=float)
        if not np.isfinite(target).all():
            raise ValueError(
                f"Conditional LightGBM target invalid: {target_column}"
            )
        model = lgb.LGBMRegressor(**LIGHTGBM_PARAMETERS)
        model.fit(x_train, target, sample_weight=weight)
        predicted = np.asarray(model.predict(x_query), dtype=float)
        if predicted.shape != (len(query),) or not np.isfinite(
            predicted
        ).all():
            raise AssertionError(
                f"Conditional LightGBM prediction invalid: {target_column}"
            )
        model_string = model.booster_.model_to_string()
        restored = np.asarray(
            lgb.Booster(model_str=model_string).predict(x_query),
            dtype=float,
        )
        error = float(np.max(np.abs(predicted - restored)))
        if error != 0.0:
            raise AssertionError(
                "Conditional LightGBM serialization changed predictions"
            )
        predictions[:, index] = predicted
        strings[target_column] = model_string
        target_audit[target_column] = {
            "train_mean": float(np.mean(target)),
            "train_std": float(np.std(target, ddof=0)),
            "serialization_max_abs_error": error,
            "trees": int(model.booster_.num_trees()),
        }
    candle, candle_audit = reconstruct_conditional_ohlc(
        query["source_close"].to_numpy(dtype=float),
        predictions,
    )
    output = query.copy().reset_index(drop=True)
    for index, target_column in enumerate(TARGET_TRANSFORM_COLUMNS):
        output[
            f"predicted_{target_column.removeprefix('target_')}"
        ] = predictions[:, index]
    output = pd.concat([output, candle], axis=1)
    output["price_model_id"] = MODEL_ID
    output["candle_legal"] = bool(candle_audit["candle_legal"])
    return ConditionalLightgbmResult(
        predictions=output,
        model_strings=strings,
        audit={
            "model_id": MODEL_ID,
            "parameters": LIGHTGBM_PARAMETERS,
            "feature_columns": list(MODEL_FEATURE_COLUMNS),
            "target_columns": list(TARGET_TRANSFORM_COLUMNS),
            "train_rows": len(train),
            "query_rows": len(query),
            "class_counts": train["target_label"]
            .value_counts()
            .to_dict(),
            "sample_weight_policy": (
                "SQRT_INVERSE_CLASS_FREQUENCY_CLIPPED_1_TO_5"
            ),
            "targets": target_audit,
            "candle": candle_audit,
            "hyperparameter_selection": False,
            "feature_selection": False,
            "scenario_label_is_internal_query": True,
        },
    )


def conditional_ohlc_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    """Evaluate the scenario matching each observed action-zone target."""

    required = {
        "scenario_label",
        "target_label",
        "target_bucket",
        "target_event_group_id",
        "horizon_index",
        "source_close",
        *(f"target_{field}" for field in OHLC_FIELDS),
        *(f"predicted_{field}" for field in OHLC_FIELDS),
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(
            f"Conditional OHLC metrics missing: {sorted(missing)}"
        )
    matched = predictions.loc[
        predictions["target_label"].isin(ACTION_SCENARIO_LABELS)
        & predictions["scenario_label"].eq(predictions["target_label"])
    ].copy()
    if matched.empty:
        raise ValueError("Conditional OHLC metrics have no action rows")
    source = matched["source_close"].to_numpy(dtype=float)
    per_field: dict[str, Any] = {}
    relative_errors: list[np.ndarray] = []
    for field in OHLC_FIELDS:
        actual = matched[f"target_{field}"].to_numpy(dtype=float)
        predicted = matched[f"predicted_{field}"].to_numpy(dtype=float)
        absolute = np.abs(actual - predicted)
        relative = absolute / source
        relative_errors.append(relative)
        per_field[field] = {
            "mae_usd_per_oz": float(np.mean(absolute)),
            "mae_fraction_of_source_close": float(np.mean(relative)),
            "median_absolute_error_usd_per_oz": float(
                np.median(absolute)
            ),
        }
    row_error = np.mean(np.column_stack(relative_errors), axis=1)
    matched["relative_candle_mae"] = row_error
    segment = pd.cut(
        matched["horizon_index"],
        bins=[0, 5, 10, 21],
        labels=["h01_h05", "h06_h10", "h11_h21"],
    )
    return {
        "action_rows": len(matched),
        "unique_target_buckets": int(matched["target_bucket"].nunique()),
        "event_groups": int(
            matched["target_event_group_id"].nunique()
        ),
        "mean_relative_candle_mae": float(np.mean(row_error)),
        "median_relative_candle_mae": float(np.median(row_error)),
        "per_field": per_field,
        "per_side": {
            str(label): {
                "rows": len(group),
                "unique_target_buckets": int(
                    group["target_bucket"].nunique()
                ),
                "mean_relative_candle_mae": float(
                    group["relative_candle_mae"].mean()
                ),
            }
            for label, group in matched.groupby(
                "target_label",
                sort=True,
            )
        },
        "per_segment": {
            str(label): {
                "rows": int(mask.sum()),
                "mean_relative_candle_mae": float(
                    matched.loc[mask, "relative_candle_mae"].mean()
                ),
            }
            for label in segment.cat.categories
            for mask in [segment.eq(label)]
        },
        "candle_legal_rate": float(
            matched["candle_legal"].astype(bool).mean()
        )
        if "candle_legal" in matched
        else None,
    }
