from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from aupilot.training.future_slot_logistic import (
    CLASS_LABELS,
    PROBABILITY_COLUMNS,
    FutureSlotPredictionResult,
    fit_constant_prior,
    group_normalized_weights,
)
from aupilot.training.future_slot_shrinkage import (
    blend_near_with_constant,
)

BASE_MODEL_ID = "FIXED_SMALL_LIGHTGBM_NEAR_FAR_PRIOR_V1"
SHRUNK_MODEL_ID = (
    "INNER_SELECTED_SHRUNK_FIXED_SMALL_LIGHTGBM_NEAR_FAR_PRIOR_V1"
)
FROZEN_PARAMETERS: dict[str, Any] = {
    "objective": "multiclass",
    "num_class": 3,
    "n_estimators": 80,
    "learning_rate": 0.03,
    "num_leaves": 7,
    "max_depth": 3,
    "min_data_in_leaf": 180,
    "reg_alpha": 0.0,
    "reg_lambda": 2.0,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "max_bin": 63,
    "random_state": 20260723,
    "n_jobs": 4,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


def _validate_features(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    if not feature_columns or len(set(feature_columns)) != len(
        feature_columns
    ):
        raise ValueError("R29 features must be non-empty and unique")
    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"R29 missing features: {sorted(missing)}")
    values = frame.loc[:, feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("R29 features contain non-finite values")
    return values


def _probability_output(
    frame: pd.DataFrame,
    probability: np.ndarray,
) -> pd.DataFrame:
    if (
        probability.shape != (len(frame), len(CLASS_LABELS))
        or not np.isfinite(probability).all()
        or (probability < 0.0).any()
        or (probability > 1.0).any()
        or not np.allclose(
            probability.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1.0e-10,
        )
    ):
        raise ValueError("R29 probabilities are invalid")
    output = frame.copy().reset_index(drop=True)
    for index, column in enumerate(PROBABILITY_COLUMNS):
        output[column] = probability[:, index]
    output["display_class"] = np.asarray(CLASS_LABELS, dtype=object)[
        np.argmax(probability, axis=1)
    ]
    return output


def fit_fixed_small_lightgbm_far_prior(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
) -> FutureSlotPredictionResult:
    """Fit one frozen h1-h5 LightGBM; emit a fold prior for h6-h21."""

    train_horizon = pd.to_numeric(train["horizon_index"], errors="coerce")
    test_horizon = pd.to_numeric(test["horizon_index"], errors="coerce")
    if (
        train_horizon.isna().any()
        or test_horizon.isna().any()
        or not train_horizon.between(1, 21).all()
        or not test_horizon.between(1, 21).all()
    ):
        raise ValueError("R29 horizon_index must be within 1...21")
    near_train = train.loc[train_horizon.between(1, 5)].reset_index(
        drop=True
    )
    ordered_test = test.copy().reset_index(drop=True)
    ordered_test["_r29_row_order"] = range(len(ordered_test))
    near_test = ordered_test.loc[
        test_horizon.between(1, 5).to_numpy()
    ].reset_index(drop=True)
    far_test = ordered_test.loc[
        test_horizon.between(6, 21).to_numpy()
    ].reset_index(drop=True)
    if (
        near_train.empty
        or near_test.empty
        or far_test.empty
        or len(near_test) + len(far_test) != len(test)
    ):
        raise ValueError("R29 near/far partition is incomplete")
    labels = near_train["target_label"].astype(str)
    if set(labels) != set(CLASS_LABELS):
        raise ValueError("R29 training requires all three classes")
    unknown = set(test["target_label"].astype(str)) - set(CLASS_LABELS)
    if unknown:
        raise ValueError(f"R29 unsupported test labels: {sorted(unknown)}")
    x_train = _validate_features(near_train, feature_columns)
    x_test = _validate_features(near_test, feature_columns)
    weights = group_normalized_weights(near_train)
    model = lgb.LGBMClassifier(**FROZEN_PARAMETERS)
    model.fit(x_train, labels, sample_weight=weights)
    raw = np.asarray(model.predict_proba(x_test), dtype=float)
    probability = np.column_stack(
        [
            raw[:, int(np.flatnonzero(model.classes_ == label)[0])]
            for label in CLASS_LABELS
        ]
    )
    near_predictions = _probability_output(near_test, probability)
    far = fit_constant_prior(train, far_test)
    predictions = (
        pd.concat(
            [near_predictions, far.predictions],
            ignore_index=True,
        )
        .sort_values("_r29_row_order", kind="stable")
        .drop(columns="_r29_row_order")
        .reset_index(drop=True)
    )
    if predictions["horizon_index"].tolist() != test[
        "horizon_index"
    ].reset_index(drop=True).tolist():
        raise AssertionError("R29 prediction row order changed")
    booster = model.booster_
    importance_gain = booster.feature_importance(importance_type="gain")
    importance_split = booster.feature_importance(importance_type="split")
    feature_importance = {
        feature: {
            "gain": float(importance_gain[index]),
            "split": int(importance_split[index]),
        }
        for index, feature in enumerate(feature_columns)
    }
    audit = {
        "model_id": BASE_MODEL_ID,
        "train_rows": len(train),
        "near_train_rows": len(near_train),
        "test_rows": len(test),
        "near_test_rows": len(near_test),
        "far_test_rows": len(far_test),
        "train_event_groups": int(
            near_train["target_event_group_id"].nunique()
        ),
        "feature_columns": list(feature_columns),
        "parameters": FROZEN_PARAMETERS,
        "trees": int(booster.num_trees()),
        "feature_importance": feature_importance,
        "class_weight": None,
        "resampling": False,
        "threshold_selected": False,
        "far_model_audit": far.audit,
    }
    return FutureSlotPredictionResult(
        predictions=predictions,
        audit=audit,
        artifact={
            "model_id": BASE_MODEL_ID,
            "parameters": FROZEN_PARAMETERS,
            "class_labels": list(model.classes_),
            "feature_columns": list(feature_columns),
            "feature_importance": feature_importance,
            "far_prior": far.artifact,
        },
    )


def fit_shrunk_fixed_small_lightgbm_far_prior(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    alpha: float,
) -> FutureSlotPredictionResult:
    """Apply an already inner-selected alpha to the fixed R29 tree."""

    base = fit_fixed_small_lightgbm_far_prior(
        train,
        test,
        feature_columns=feature_columns,
    )
    constant = fit_constant_prior(train, test)
    predictions = blend_near_with_constant(
        base.predictions,
        constant.predictions,
        alpha=alpha,
    )
    return FutureSlotPredictionResult(
        predictions=predictions,
        audit={
            "model_id": SHRUNK_MODEL_ID,
            "selected_alpha": float(alpha),
            "base_model_audit": base.audit,
            "constant_audit": constant.audit,
            "calibration": "INNER_SELECTED_LINEAR_SHRINKAGE_TO_FOLD_PRIOR",
            "threshold_selected": False,
        },
        artifact={
            "model_id": SHRUNK_MODEL_ID,
            "selected_alpha": float(alpha),
            "base_model": base.artifact,
            "constant_prior": constant.artifact,
        },
    )
