"""Low-capacity causal residual LightGBM for conditional OHLC.

The fold-local raw conditional median remains the anchor.  Four small trees
only predict transformed-candle residuals from causal source state, explicit
scenario and horizon.  A shrinkage value of zero reproduces the anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from aupilot.training.conditional_future_ohlc import (
    MODEL_FEATURE_COLUMNS,
    SCENARIO_LABELS,
    TARGET_TRANSFORM_COLUMNS,
    ConditionalMedianBundle,
    predict_conditional_median,
    reconstruct_conditional_ohlc,
)
from aupilot.training.n_branch_temporal_shape import (
    MN18_TEMPORAL_SHAPE_COLUMNS,
    MN18_TEMPORAL_SHAPE_COMPLETE,
    augment_issuance_with_mn18_temporal_shape,
)
from aupilot.training.n_branch_top_false_breakout import (
    MN14_FALSE_BREAKOUT_COLUMNS,
    MN14_FALSE_BREAKOUT_COMPLETE,
)

MODEL_ID = "PN02_RAW_MEDIAN_CAUSAL_GEOMETRY_RESIDUAL_LIGHTGBM_V1"
BASELINE_TRANSFORM_COLUMNS = tuple(
    f"baseline_{column.removeprefix('target_')}"
    for column in TARGET_TRANSFORM_COLUMNS
)
RESIDUAL_COLUMNS = tuple(
    f"residual_{column.removeprefix('target_')}"
    for column in TARGET_TRANSFORM_COLUMNS
)
COMPLETENESS_COLUMNS = (
    MN14_FALSE_BREAKOUT_COMPLETE,
    MN18_TEMPORAL_SHAPE_COMPLETE,
)
RESIDUAL_FEATURE_COLUMNS = (
    *MODEL_FEATURE_COLUMNS,
    *MN14_FALSE_BREAKOUT_COLUMNS,
    *MN18_TEMPORAL_SHAPE_COLUMNS,
    *COMPLETENESS_COLUMNS,
    "source_combined_volatility_scale",
    *BASELINE_TRANSFORM_COLUMNS,
)
LIGHTGBM_PARAMETERS: dict[str, Any] = {
    "objective": "regression_l1",
    "n_estimators": 100,
    "learning_rate": 0.025,
    "num_leaves": 3,
    "max_depth": 2,
    "min_data_in_leaf": 80,
    "reg_alpha": 0.0,
    "reg_lambda": 10.0,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "max_bin": 31,
    "random_state": 20260727,
    "n_jobs": 4,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


@dataclass(frozen=True)
class ResidualLightgbmResult:
    predictions: pd.DataFrame
    model_strings: dict[str, str]
    audit: dict[str, Any]


def augment_conditional_frame_with_mn18(
    daily: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the frozen causal MN18 geometry asset without dropping rows."""

    result = augment_issuance_with_mn18_temporal_shape(daily, frame)
    output = result.frame.copy()
    for column in COMPLETENESS_COLUMNS:
        output[column] = output[column].astype(float)
    causal = (
        *MN14_FALSE_BREAKOUT_COLUMNS,
        *MN18_TEMPORAL_SHAPE_COLUMNS,
    )
    output.loc[:, causal] = output.loc[:, causal].fillna(0.0)
    output["source_combined_volatility_scale"] = 0.5 * (
        output["source_realized_volatility_20"]
        + output["source_atr_fraction_14"]
    )
    values = output.loc[
        :,
        [*causal, *COMPLETENESS_COLUMNS, "source_combined_volatility_scale"],
    ].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("PN02 causal geometry contains non-finite values")
    if len(output) != len(frame):
        raise AssertionError("PN02 causal geometry changed row count")
    if not output.groupby("issuance_id", sort=False).size().eq(21).all():
        raise AssertionError("PN02 causal geometry split an issuance")
    return output


def _training_query(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy().reset_index(drop=True)
    output["scenario_label"] = output["target_label"].astype(str)
    return output


def _with_anchor_features(
    frame: pd.DataFrame,
    median_bundle: ConditionalMedianBundle,
) -> tuple[pd.DataFrame, np.ndarray]:
    baseline = predict_conditional_median(median_bundle, frame)
    output = baseline.copy()
    transformed = []
    for target_column, baseline_column in zip(
        TARGET_TRANSFORM_COLUMNS,
        BASELINE_TRANSFORM_COLUMNS,
        strict=True,
    ):
        predicted_column = (
            f"predicted_{target_column.removeprefix('target_')}"
        )
        output[baseline_column] = output[predicted_column]
        transformed.append(output[predicted_column].to_numpy(dtype=float))
    return output, np.column_stack(transformed)


def _matrix(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(RESIDUAL_FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"PN02 residual features missing: {sorted(missing)}")
    matrix = frame.loc[:, RESIDUAL_FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("PN02 residual feature matrix is non-finite")
    return matrix


def _class_weights(train: pd.DataFrame) -> np.ndarray:
    counts = train["target_label"].value_counts()
    if set(counts.index) != set(SCENARIO_LABELS):
        raise ValueError("PN02 training fold lacks a scenario class")
    largest = float(counts.max())
    mapping = {
        label: min(5.0, max(1.0, np.sqrt(largest / float(count))))
        for label, count in counts.items()
    }
    return train["target_label"].map(mapping).to_numpy(dtype=float)


def apply_residual_shrinkage(
    prediction: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    """Apply one preregistered shared residual shrinkage and rebuild candles."""

    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("PN02 residual shrinkage must be in [0, 1]")
    baseline = prediction.loc[
        :, BASELINE_TRANSFORM_COLUMNS
    ].to_numpy(dtype=float)
    residual = prediction.loc[:, RESIDUAL_COLUMNS].to_numpy(dtype=float)
    transformed = baseline + float(alpha) * residual
    candle, audit = reconstruct_conditional_ohlc(
        prediction["source_close"].to_numpy(dtype=float),
        transformed,
    )
    output = prediction.copy().reset_index(drop=True)
    for index, target_column in enumerate(TARGET_TRANSFORM_COLUMNS):
        output[
            f"predicted_{target_column.removeprefix('target_')}"
        ] = transformed[:, index]
    for column in (
        "predicted_open",
        "predicted_high",
        "predicted_low",
        "predicted_close",
    ):
        output[column] = candle[column].to_numpy(dtype=float)
    output["price_model_id"] = MODEL_ID
    output["residual_shrinkage_alpha"] = float(alpha)
    output["candle_legal"] = bool(audit["candle_legal"])
    return output


def fit_conditional_residual_lightgbm(
    train: pd.DataFrame,
    query: pd.DataFrame,
    *,
    median_bundle: ConditionalMedianBundle,
    alpha: float = 1.0,
) -> ResidualLightgbmResult:
    """Fit four fixed low-capacity residual trees and predict queries."""

    if train.empty or query.empty:
        raise ValueError("PN02 train/query is empty")
    train_anchor, baseline_train = _with_anchor_features(
        _training_query(train),
        median_bundle,
    )
    query_anchor, _ = _with_anchor_features(query, median_bundle)
    x_train = _matrix(train_anchor)
    x_query = _matrix(query_anchor)
    target = train.loc[
        :, TARGET_TRANSFORM_COLUMNS
    ].to_numpy(dtype=float)
    residual_target = target - baseline_train
    weights = _class_weights(train)
    model_strings: dict[str, str] = {}
    residual_prediction = []
    target_audits: dict[str, Any] = {}
    for index, target_column in enumerate(TARGET_TRANSFORM_COLUMNS):
        model = lgb.LGBMRegressor(**LIGHTGBM_PARAMETERS)
        model.fit(
            x_train,
            residual_target[:, index],
            sample_weight=weights,
        )
        predicted = model.predict(x_query).astype(float)
        model_string = model.booster_.model_to_string()
        restored = lgb.Booster(model_str=model_string).predict(
            x_query
        ).astype(float)
        maximum_error = float(
            np.max(np.abs(predicted - restored), initial=0.0)
        )
        if maximum_error != 0.0:
            raise AssertionError("PN02 LightGBM serialization changed output")
        model_strings[target_column] = model_string
        residual_prediction.append(predicted)
        target_audits[target_column] = {
            "training_rows": len(train),
            "target_residual_median_abs": float(
                np.median(np.abs(residual_target[:, index]))
            ),
            "serialization_max_abs_error": maximum_error,
            "num_trees": int(model.booster_.num_trees()),
        }
    output = query_anchor.copy().reset_index(drop=True)
    for index, residual_column in enumerate(RESIDUAL_COLUMNS):
        output[residual_column] = residual_prediction[index]
    output = apply_residual_shrinkage(output, alpha)
    return ResidualLightgbmResult(
        predictions=output,
        model_strings=model_strings,
        audit={
            "model_id": MODEL_ID,
            "training_rows": len(train),
            "query_rows": len(query),
            "feature_columns": list(RESIDUAL_FEATURE_COLUMNS),
            "feature_count": len(RESIDUAL_FEATURE_COLUMNS),
            "parameters": dict(LIGHTGBM_PARAMETERS),
            "alpha": float(alpha),
            "targets": target_audits,
            "direct_target_model": False,
            "raw_conditional_median_anchor": True,
            "future_features_used": False,
            "external_technical_inputs_used": False,
        },
    )
