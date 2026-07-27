"""M01 hierarchical LightGBM for G01 graded daily pivot probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from aupilot.training.graded_future_slot import (
    DIRECTION_LABELS,
    GRADED_LABELS,
    PROBABILITY_COLUMNS,
    GradedProbeResult,
    fit_graded_ordinal_probe,
    group_normalized_weights,
)

M01_MODEL_ID = "M01_G01_H1_HURDLE_ORDINAL_LIGHTGBM"
M01_H1_FEATURE_COLUMNS = (
    "source_active_trend_code",
    "source_reversal_progress",
    "source_vol_adjusted_oriented_oc_reversal",
    "source_log_return_5",
    "source_close_sma_distance_20",
    "source_rsi_14_centered",
    "source_close_location_value",
)
M01_LIGHTGBM_PARAMETERS: dict[str, Any] = {
    "n_estimators": 160,
    "learning_rate": 0.03,
    "num_leaves": 5,
    "max_depth": 3,
    "min_data_in_leaf": 12,
    "reg_alpha": 0.0,
    "reg_lambda": 10.0,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "max_bin": 31,
    "class_weight": None,
    "random_state": 20260725,
    "n_jobs": 4,
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


@dataclass(frozen=True)
class M01PredictionResult:
    predictions: pd.DataFrame
    audit: dict[str, Any]


def _features(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    if not feature_columns or len(feature_columns) != len(set(feature_columns)):
        raise ValueError("M01 feature columns must be unique and nonempty")
    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"M01 missing features: {sorted(missing)}")
    values = frame.loc[:, feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("M01 features are non-finite")
    return values


def _labels(frame: pd.DataFrame) -> pd.Series:
    labels = frame["target_label"].astype(str)
    invalid = sorted(set(labels) - set(GRADED_LABELS))
    if invalid:
        raise ValueError(f"M01 unsupported labels: {invalid}")
    return labels


def _fit_binary(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    target = np.asarray(y_train, dtype=int)
    classes = np.unique(target)
    if set(classes) != {0, 1}:
        raise ValueError(f"M01 binary head lacks a class: {classes.tolist()}")
    model = lgb.LGBMClassifier(
        **M01_LIGHTGBM_PARAMETERS,
        objective="binary",
    )
    model.fit(x_train, target, sample_weight=weights)
    raw = np.asarray(model.predict_proba(x_test), dtype=float)
    positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
    probability = raw[:, positive_index]
    if (
        probability.shape != (len(x_test),)
        or not np.isfinite(probability).all()
        or (probability < 0.0).any()
        or (probability > 1.0).any()
    ):
        raise AssertionError("M01 binary probabilities are invalid")
    booster = model.booster_
    return probability, {
        "parameters": M01_LIGHTGBM_PARAMETERS,
        "positive_rows": int(target.sum()),
        "negative_rows": int((target == 0).sum()),
        "trees": int(booster.num_trees()),
    }


def _project_ordinal(
    q2: np.ndarray,
    q3: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    violation = q3 > q2
    midpoint = (q2 + q3) / 2.0
    return (
        np.where(violation, midpoint, q2),
        np.where(violation, midpoint, q3),
    )


def fit_m01_h1_hurdle_ordinal(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] = M01_H1_FEATURE_COLUMNS,
) -> M01PredictionResult:
    """Fit action, conditional direction, and conditional strength heads for h1."""

    train_h1 = train.loc[train["horizon_index"].eq(1)].reset_index(drop=True)
    test_h1 = test.loc[test["horizon_index"].eq(1)].reset_index(drop=True)
    if train_h1.empty or test_h1.empty:
        raise ValueError("M01 h1 train/test must be nonempty")
    train_labels = _labels(train_h1)
    _labels(test_h1)
    x_train = _features(train_h1, feature_columns)
    x_test = _features(test_h1, feature_columns)
    weights = group_normalized_weights(train_h1)

    action_truth = train_labels.ne("NORMAL").astype(int).to_numpy()
    p_action, action_audit = _fit_binary(
        x_train,
        action_truth,
        weights,
        x_test,
    )
    action_rows = train_h1.loc[action_truth.astype(bool)].reset_index(drop=True)
    action_labels = action_rows["target_label"].astype(str)
    action_x = _features(action_rows, feature_columns)
    action_weights = group_normalized_weights(action_rows)

    top_truth = action_labels.str.startswith("TOP_").astype(int).to_numpy()
    p_top_given_action, direction_audit = _fit_binary(
        action_x,
        top_truth,
        action_weights,
        x_test,
    )

    levels = (
        action_labels.str.extract(r"_L([123])$", expand=False)
        .astype(int)
        .to_numpy()
    )
    side_top = top_truth.astype(float)
    strength_x = np.column_stack([action_x, side_top])
    top_test_x = np.column_stack([x_test, np.ones(len(x_test))])
    bottom_test_x = np.column_stack([x_test, np.zeros(len(x_test))])
    ordinal: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    ordinal_audits: dict[str, Any] = {}
    for threshold in (2, 3):
        truth = (levels >= threshold).astype(int)
        p_top, audit = _fit_binary(
            strength_x,
            truth,
            action_weights,
            top_test_x,
        )
        p_bottom, _ = _fit_binary(
            strength_x,
            truth,
            action_weights,
            bottom_test_x,
        )
        ordinal[threshold] = (p_top, p_bottom)
        ordinal_audits[f"at_least_l{threshold}"] = audit

    top_q2, bottom_q2 = ordinal[2]
    top_q3, bottom_q3 = ordinal[3]
    top_q2, top_q3 = _project_ordinal(top_q2, top_q3)
    bottom_q2, bottom_q3 = _project_ordinal(bottom_q2, bottom_q3)
    top_strength = np.column_stack(
        [1.0 - top_q2, top_q2 - top_q3, top_q3]
    )
    bottom_strength = np.column_stack(
        [1.0 - bottom_q2, bottom_q2 - bottom_q3, bottom_q3]
    )
    p_top = p_action * p_top_given_action
    p_bottom = p_action * (1.0 - p_top_given_action)
    probability = np.column_stack(
        [
            1.0 - p_action,
            p_top[:, None] * top_strength,
            p_bottom[:, None] * bottom_strength,
        ]
    )
    if (
        not np.isfinite(probability).all()
        or (probability < -1.0e-12).any()
        or (probability > 1.0 + 1.0e-12).any()
        or not np.allclose(probability.sum(axis=1), 1.0, atol=1.0e-10)
    ):
        raise AssertionError("M01 seven-class probabilities are invalid")

    output = test_h1.copy()
    for index, column in enumerate(PROBABILITY_COLUMNS):
        output[column] = probability[:, index]
    output["display_class"] = np.asarray(GRADED_LABELS, dtype=object)[
        np.argmax(probability, axis=1)
    ]
    total_weight = weights.sum()
    p_top_prior = float(
        weights[train_labels.str.startswith("TOP_").to_numpy()].sum()
        / total_weight
    )
    p_bottom_prior = float(
        weights[train_labels.str.startswith("BOTTOM_").to_numpy()].sum()
        / total_weight
    )
    if p_top_prior <= 0.0 or p_bottom_prior <= 0.0:
        raise ValueError("M01 h1 training priors lack a direction")
    output["train_prior_top_action"] = p_top_prior
    output["train_prior_bottom_action"] = p_bottom_prior
    return M01PredictionResult(
        predictions=output,
        audit={
            "model_id": M01_MODEL_ID,
            "feature_columns": list(feature_columns),
            "train_rows": len(train_h1),
            "test_rows": len(test_h1),
            "train_action_rows": len(action_rows),
            "train_direction_counts": {
                label: int((np.where(top_truth == 1, "TOP", "BOTTOM") == label).sum())
                for label in DIRECTION_LABELS[:2]
            },
            "train_prior_top_action": p_top_prior,
            "train_prior_bottom_action": p_bottom_prior,
            "action_head": action_audit,
            "direction_head": direction_audit,
            "ordinal_heads": ordinal_audits,
            "outer_labels_used_for_fit_or_threshold": False,
        },
    )


def fit_m01_full_horizon(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    all_feature_columns: tuple[str, ...],
    h1_feature_columns: tuple[str, ...] = M01_H1_FEATURE_COLUMNS,
) -> GradedProbeResult:
    """Use M01 at h1, the frozen G05 tree at h2-h5, and priors at h6-h21."""

    base = fit_graded_ordinal_probe(
        train,
        test,
        model_id="FIXED_SMALL_LGBM_ORDINAL",
        feature_columns=all_feature_columns,
        near_horizon_maximum=5,
    )
    h1 = fit_m01_h1_hurdle_ordinal(
        train,
        test,
        feature_columns=h1_feature_columns,
    )
    output = base.predictions.copy()
    h1_mask = output["horizon_index"].eq(1)
    replacement_columns = [
        *PROBABILITY_COLUMNS,
        "display_class",
        "train_prior_top_action",
        "train_prior_bottom_action",
    ]
    output.loc[h1_mask, replacement_columns] = h1.predictions.loc[
        :,
        replacement_columns,
    ].to_numpy()
    return GradedProbeResult(
        predictions=output,
        audit={
            "model_id": M01_MODEL_ID,
            "h1": h1.audit,
            "h2_h5": base.audit,
            "h6_h21": "TRAIN_FOLD_EVENT_GROUP_NORMALIZED_SEVEN_CLASS_PRIOR",
            "outer_labels_used_for_fit_or_threshold": False,
        },
    )
