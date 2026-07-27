"""Low-capacity hierarchical probes for graded 21-slot pivot labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from aupilot.training.graded_issuance import GRADED_LABELS

DIRECTION_LABELS = ("TOP", "BOTTOM", "NORMAL")
PROBABILITY_COLUMNS = (
    "p_normal",
    "p_top_l1",
    "p_top_l2",
    "p_top_l3",
    "p_bottom_l1",
    "p_bottom_l2",
    "p_bottom_l3",
)
PROBE_MODELS = ("REGULARIZED_LOGISTIC_ORDINAL", "FIXED_SMALL_LGBM_ORDINAL")
LGBM_PARAMETERS: dict[str, Any] = {
    "n_estimators": 100,
    "learning_rate": 0.03,
    "num_leaves": 3,
    "max_depth": 2,
    "min_data_in_leaf": 15,
    "reg_alpha": 0.0,
    "reg_lambda": 5.0,
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
class GradedProbeResult:
    predictions: pd.DataFrame
    audit: dict[str, Any]


def group_normalized_weights(frame: pd.DataFrame) -> np.ndarray:
    if "target_event_group_id" not in frame.columns:
        raise ValueError("Graded probe lacks target_event_group_id")
    groups = frame["target_event_group_id"].astype(str)
    counts = groups.map(groups.value_counts())
    weights = 1.0 / counts.to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("Graded group weights are invalid")
    return weights


def _numeric_features(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    if not feature_columns or len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Graded feature columns must be unique and nonempty")
    missing = set(feature_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Graded probe missing features: {sorted(missing)}")
    values = frame.loc[:, feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Graded probe features are non-finite")
    return values


def _direction(labels: pd.Series) -> np.ndarray:
    values = labels.astype(str)
    invalid = sorted(set(values) - set(GRADED_LABELS))
    if invalid:
        raise ValueError(f"Unsupported graded labels: {invalid}")
    return np.where(
        values.str.startswith("TOP_"),
        "TOP",
        np.where(values.str.startswith("BOTTOM_"), "BOTTOM", "NORMAL"),
    )


def _level(labels: pd.Series) -> np.ndarray:
    values = labels.astype(str)
    return (
        values.str.extract(r"_L([123])$", expand=False)
        .fillna("0")
        .astype(int)
        .to_numpy()
    )


def _weighted_prior(
    labels: pd.Series,
    weights: np.ndarray,
    classes: tuple[str, ...],
) -> np.ndarray:
    values = labels.astype(str).to_numpy()
    prior = np.asarray(
        [weights[values == label].sum() for label in classes],
        dtype=float,
    )
    prior /= prior.sum()
    if not np.isfinite(prior).all() or (prior <= 0.0).any():
        raise ValueError("Graded training prior lacks a class")
    return prior


def _fit_classifier(
    *,
    model_id: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...], dict[str, Any]]:
    classes = tuple(sorted(set(map(str, y_train))))
    if len(classes) < 2:
        raise ValueError("Graded classifier training lacks two classes")
    if model_id == "REGULARIZED_LOGISTIC_ORDINAL":
        means = np.average(x_train, axis=0, weights=weights)
        variance = np.average(
            np.square(x_train - means),
            axis=0,
            weights=weights,
        )
        scale = np.sqrt(variance)
        scale = np.where(scale > 0.0, scale, 1.0)
        model = LogisticRegression(
            C=0.5,
            solver="lbfgs",
            max_iter=2000,
            tol=1.0e-9,
        )
        model.fit(
            (x_train - means) / scale,
            y_train,
            sample_weight=weights,
        )
        probability = model.predict_proba((x_test - means) / scale)
        audit = {
            "kind": "L2_REGULARIZED_LOGISTIC",
            "C": 0.5,
            "iterations": int(model.n_iter_.max()),
        }
    elif model_id == "FIXED_SMALL_LGBM_ORDINAL":
        objective = "binary" if len(classes) == 2 else "multiclass"
        parameters = {
            **LGBM_PARAMETERS,
            "objective": objective,
        }
        if objective == "multiclass":
            parameters["num_class"] = len(classes)
        model = lgb.LGBMClassifier(**parameters)
        model.fit(x_train, y_train, sample_weight=weights)
        probability = model.predict_proba(x_test)
        audit = {
            "kind": "FIXED_SMALL_LIGHTGBM",
            "parameters": parameters,
            "trees": int(model.booster_.num_trees()),
        }
    else:
        raise ValueError(f"Unsupported graded probe model: {model_id}")
    model_classes = tuple(map(str, model.classes_))
    probability = np.asarray(probability, dtype=float)
    if (
        probability.shape != (len(x_test), len(model_classes))
        or not np.isfinite(probability).all()
        or (probability < 0.0).any()
        or (probability > 1.0).any()
        or not np.allclose(probability.sum(axis=1), 1.0, atol=1.0e-10)
    ):
        raise AssertionError("Graded classifier probabilities are invalid")
    return probability, model_classes, audit


def _ordered_probability(
    probability: np.ndarray,
    classes: tuple[str, ...],
    order: tuple[str, ...],
) -> np.ndarray:
    return np.column_stack(
        [probability[:, classes.index(label)] for label in order]
    )


def _positive_probability(
    probability: np.ndarray,
    classes: tuple[str, ...],
) -> np.ndarray:
    return probability[:, classes.index("1")]


def _project_ordinal(q2: np.ndarray, q3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    violation = q3 > q2
    midpoint = (q2 + q3) / 2.0
    projected_q2 = np.where(violation, midpoint, q2)
    projected_q3 = np.where(violation, midpoint, q3)
    return projected_q2, projected_q3


def fit_graded_ordinal_probe(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    model_id: str,
    feature_columns: tuple[str, ...],
    near_horizon_maximum: int = 5,
) -> GradedProbeResult:
    """Fit a direction head and two conditional ordinal-strength heads."""

    if model_id not in PROBE_MODELS:
        raise ValueError(f"Unknown graded probe model: {model_id}")
    train_horizon = pd.to_numeric(train["horizon_index"], errors="coerce")
    test_horizon = pd.to_numeric(test["horizon_index"], errors="coerce")
    if (
        train_horizon.isna().any()
        or test_horizon.isna().any()
        or not train_horizon.between(1, 21).all()
        or not test_horizon.between(1, 21).all()
    ):
        raise ValueError("Graded horizons must be within 1...21")
    near_train = train.loc[
        train_horizon.between(1, near_horizon_maximum)
    ].reset_index(drop=True)
    ordered_test = test.copy().reset_index(drop=True)
    ordered_test["_graded_row_order"] = np.arange(len(ordered_test))
    near_test = ordered_test.loc[
        test_horizon.between(1, near_horizon_maximum).to_numpy()
    ].reset_index(drop=True)
    far_test = ordered_test.loc[
        ~test_horizon.between(1, near_horizon_maximum).to_numpy()
    ].reset_index(drop=True)
    if near_train.empty or near_test.empty or far_test.empty:
        raise ValueError("Graded near/far split is incomplete")

    x_train = _numeric_features(near_train, feature_columns)
    x_near = _numeric_features(near_test, feature_columns)
    weights = group_normalized_weights(near_train)
    direction_train = _direction(near_train["target_label"])
    if set(direction_train) != set(DIRECTION_LABELS):
        raise ValueError("Graded direction training lacks a class")
    direction_probability, direction_classes, direction_audit = (
        _fit_classifier(
            model_id=model_id,
            x_train=x_train,
            y_train=direction_train,
            weights=weights,
            x_test=x_near,
        )
    )
    direction_probability = _ordered_probability(
        direction_probability,
        direction_classes,
        DIRECTION_LABELS,
    )
    direction_prior = _weighted_prior(
        pd.Series(direction_train),
        weights,
        DIRECTION_LABELS,
    )

    action_mask = direction_train != "NORMAL"
    action_train = near_train.loc[action_mask].reset_index(drop=True)
    action_x = x_train[action_mask]
    action_weights = group_normalized_weights(action_train)
    levels = _level(action_train["target_label"])
    side_top = action_train["target_label"].astype(str).str.startswith(
        "TOP_"
    ).astype(float).to_numpy()
    action_x = np.column_stack([action_x, side_top])
    top_test_x = np.column_stack([x_near, np.ones(len(x_near))])
    bottom_test_x = np.column_stack([x_near, np.zeros(len(x_near))])

    ordinal_probabilities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ordinal_audits: dict[str, Any] = {}
    for threshold in (2, 3):
        target = (levels >= threshold).astype(int).astype(str)
        top_probability, top_classes, audit = _fit_classifier(
            model_id=model_id,
            x_train=action_x,
            y_train=target,
            weights=action_weights,
            x_test=top_test_x,
        )
        bottom_probability, bottom_classes, _ = _fit_classifier(
            model_id=model_id,
            x_train=action_x,
            y_train=target,
            weights=action_weights,
            x_test=bottom_test_x,
        )
        ordinal_probabilities[f"q{threshold}"] = (
            _positive_probability(top_probability, top_classes),
            _positive_probability(bottom_probability, bottom_classes),
        )
        ordinal_audits[f"at_least_l{threshold}"] = audit

    top_q2, bottom_q2 = ordinal_probabilities["q2"]
    top_q3, bottom_q3 = ordinal_probabilities["q3"]
    top_q2, top_q3 = _project_ordinal(top_q2, top_q3)
    bottom_q2, bottom_q3 = _project_ordinal(bottom_q2, bottom_q3)
    top_conditional = np.column_stack(
        [1.0 - top_q2, top_q2 - top_q3, top_q3]
    )
    bottom_conditional = np.column_stack(
        [1.0 - bottom_q2, bottom_q2 - bottom_q3, bottom_q3]
    )
    near_flat = np.column_stack(
        [
            direction_probability[:, 2],
            direction_probability[:, [0]] * top_conditional,
            direction_probability[:, [1]] * bottom_conditional,
        ]
    )

    full_weights = group_normalized_weights(train)
    flat_prior = _weighted_prior(
        train["target_label"],
        full_weights,
        GRADED_LABELS,
    )
    far_flat = np.repeat(flat_prior.reshape(1, -1), len(far_test), axis=0)
    flat = np.vstack([near_flat, far_flat])
    combined = pd.concat([near_test, far_test], ignore_index=True)
    for index, column in enumerate(PROBABILITY_COLUMNS):
        combined[column] = flat[:, index]
    combined["display_class"] = np.asarray(GRADED_LABELS, dtype=object)[
        np.argmax(flat, axis=1)
    ]
    combined["train_prior_top_action"] = direction_prior[0]
    combined["train_prior_bottom_action"] = direction_prior[1]
    combined = (
        combined.sort_values("_graded_row_order", kind="stable")
        .drop(columns="_graded_row_order")
        .reset_index(drop=True)
    )
    probability_values = combined.loc[
        :,
        PROBABILITY_COLUMNS,
    ].to_numpy(dtype=float)
    if (
        not np.isfinite(probability_values).all()
        or (probability_values < -1.0e-12).any()
        or (probability_values > 1.0 + 1.0e-12).any()
        or not np.allclose(probability_values.sum(axis=1), 1.0, atol=1.0e-10)
    ):
        raise AssertionError("Graded seven-class probabilities are invalid")
    return GradedProbeResult(
        predictions=combined,
        audit={
            "model_id": model_id,
            "feature_columns": list(feature_columns),
            "near_horizon_maximum": near_horizon_maximum,
            "near_train_rows": len(near_train),
            "near_train_event_groups": int(
                near_train["target_event_group_id"].nunique()
            ),
            "near_action_rows": len(action_train),
            "near_action_event_groups": int(
                action_train["target_event_group_id"].nunique()
            ),
            "direction_prior": dict(
                zip(DIRECTION_LABELS, direction_prior, strict=True)
            ),
            "flat_prior": dict(zip(GRADED_LABELS, flat_prior, strict=True)),
            "direction_head": direction_audit,
            "ordinal_heads": ordinal_audits,
            "outer_labels_used_for_fit_or_threshold": False,
            "threshold_selected": False,
        },
    )


def evaluate_graded_probabilities(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    labels = predictions["target_label"].astype(str)
    invalid = sorted(set(labels) - set(GRADED_LABELS))
    if invalid:
        raise ValueError(f"Unsupported evaluation labels: {invalid}")
    probability = predictions.loc[
        :,
        PROBABILITY_COLUMNS,
    ].to_numpy(dtype=float)
    weights = group_normalized_weights(predictions)
    normalized = weights / weights.sum()
    truth = np.column_stack(
        [(labels == label).astype(float) for label in GRADED_LABELS]
    )
    clipped = np.clip(probability, 1.0e-15, 1.0)
    output: dict[str, Any] = {
        "rows": len(predictions),
        "event_groups": int(predictions["target_event_group_id"].nunique()),
        "multiclass_log_loss": float(
            np.sum(normalized * -np.sum(truth * np.log(clipped), axis=1))
        ),
        "multiclass_brier": float(
            np.sum(normalized * np.sum(np.square(probability - truth), axis=1))
        ),
        "argmax_accuracy": float(
            np.sum(
                normalized
                * (
                    np.asarray(GRADED_LABELS, dtype=object)[
                        np.argmax(probability, axis=1)
                    ]
                    == labels.to_numpy()
                )
            )
        ),
        "binary": {},
    }
    binary_targets = {
        "ACTION": labels.ne("NORMAL").to_numpy(dtype=int),
        "TOP": labels.str.startswith("TOP_").to_numpy(dtype=int),
        "BOTTOM": labels.str.startswith("BOTTOM_").to_numpy(dtype=int),
        "AT_LEAST_L2": (_level(labels) >= 2).astype(int),
        "AT_LEAST_L3": (_level(labels) >= 3).astype(int),
    }
    binary_probabilities = {
        "ACTION": 1.0 - probability[:, 0],
        "TOP": probability[:, 1:4].sum(axis=1),
        "BOTTOM": probability[:, 4:7].sum(axis=1),
        "AT_LEAST_L2": probability[:, [2, 3, 5, 6]].sum(axis=1),
        "AT_LEAST_L3": probability[:, [3, 6]].sum(axis=1),
    }
    for name, target in binary_targets.items():
        prevalence = float(np.sum(normalized * target))
        metrics: dict[str, float | None] = {
            "prevalence": prevalence,
            "average_precision": None,
            "roc_auc": None,
        }
        if len(np.unique(target)) == 2:
            metrics["average_precision"] = float(
                average_precision_score(
                    target,
                    binary_probabilities[name],
                    sample_weight=weights,
                )
            )
            metrics["roc_auc"] = float(
                roc_auc_score(
                    target,
                    binary_probabilities[name],
                    sample_weight=weights,
                )
            )
        output["binary"][name] = metrics
    action = labels.ne("NORMAL").to_numpy()
    if action.any():
        true_level = _level(labels)[action]
        top = labels.str.startswith("TOP_").to_numpy()[action]
        conditional = np.where(
            top[:, None],
            probability[action, 1:4]
            / np.maximum(probability[action, 1:4].sum(axis=1, keepdims=True), 1e-15),
            probability[action, 4:7]
            / np.maximum(probability[action, 4:7].sum(axis=1, keepdims=True), 1e-15),
        )
        expected_level = conditional @ np.asarray([1.0, 2.0, 3.0])
        action_weights = weights[action]
        output["conditional_strength_mae"] = float(
            np.average(
                np.abs(expected_level - true_level),
                weights=action_weights,
            )
        )
    else:
        output["conditional_strength_mae"] = None
    return output


def boundary_predictions_to_signals(
    predictions: pd.DataFrame,
    *,
    lift_threshold: float = 1.5,
) -> pd.DataFrame:
    """Convert the latest causal boundary forecast per target date to B01 signals."""

    if lift_threshold <= 0.0:
        raise ValueError("lift_threshold must be positive")
    required = {
        "feature_anchor_bucket",
        "target_bucket",
        "boundary_action_eligible",
        "train_prior_top_action",
        "train_prior_bottom_action",
        *PROBABILITY_COLUMNS,
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Boundary predictions missing: {sorted(missing)}")
    frame = predictions.loc[
        predictions["boundary_action_eligible"].astype(bool)
    ].copy()
    frame["feature_anchor_bucket"] = pd.to_datetime(
        frame["feature_anchor_bucket"],
        errors="coerce",
    ).dt.date
    frame["target_bucket"] = pd.to_datetime(
        frame["target_bucket"],
        errors="coerce",
    ).dt.date
    frame = (
        frame.sort_values(
            ["target_bucket", "feature_anchor_bucket", "horizon_index"],
            kind="stable",
        )
        .drop_duplicates("target_bucket", keep="last")
        .reset_index(drop=True)
    )
    p_top = frame.loc[:, ["p_top_l1", "p_top_l2", "p_top_l3"]].sum(axis=1)
    p_bottom = frame.loc[
        :,
        ["p_bottom_l1", "p_bottom_l2", "p_bottom_l3"],
    ].sum(axis=1)
    top_prior = pd.to_numeric(
        frame["train_prior_top_action"],
        errors="coerce",
    )
    bottom_prior = pd.to_numeric(
        frame["train_prior_bottom_action"],
        errors="coerce",
    )
    if (
        top_prior.isna().any()
        or bottom_prior.isna().any()
        or (top_prior <= 0.0).any()
        or (bottom_prior <= 0.0).any()
    ):
        raise ValueError("Boundary action priors are invalid")
    top_lift = p_top / top_prior
    bottom_lift = p_bottom / bottom_prior
    choose_top = top_lift >= bottom_lift
    chosen_lift = np.where(choose_top, top_lift, bottom_lift)
    accepted = frame.loc[chosen_lift >= lift_threshold].copy()
    accepted_choose_top = choose_top.to_numpy()[
        chosen_lift >= lift_threshold
    ]
    if accepted.empty:
        return pd.DataFrame(
            columns=[
                "signal_id",
                "signal_date",
                "signal_label",
                "event_group_id",
                "model_probability",
                "probability_lift",
                "expected_delta_pp",
            ]
        )
    top_grade = accepted.loc[
        :,
        ["p_top_l1", "p_top_l2", "p_top_l3"],
    ].to_numpy(dtype=float)
    bottom_grade = accepted.loc[
        :,
        ["p_bottom_l1", "p_bottom_l2", "p_bottom_l3"],
    ].to_numpy(dtype=float)
    selected_grade = np.where(
        accepted_choose_top[:, None],
        top_grade,
        bottom_grade,
    )
    selected_side_probability = selected_grade.sum(axis=1)
    conditional = selected_grade / np.maximum(
        selected_side_probability[:, None],
        1.0e-15,
    )
    expected_delta = conditional @ np.asarray([10.0, 20.0, 40.0])
    level = np.where(expected_delta < 15.0, 1, np.where(expected_delta < 30.0, 2, 3))
    side = np.where(accepted_choose_top, "TOP", "BOTTOM")
    signals = pd.DataFrame(
        {
            "signal_date": accepted["target_bucket"].to_numpy(),
            "signal_label": [
                f"{event_side}_L{event_level}"
                for event_side, event_level in zip(side, level, strict=True)
            ],
            "model_probability": selected_side_probability,
            "probability_lift": chosen_lift[chosen_lift >= lift_threshold],
            "expected_delta_pp": expected_delta,
        }
    )
    signals.insert(
        0,
        "signal_id",
        [
            f"G05-PRED-{index:04d}"
            for index in range(1, len(signals) + 1)
        ],
    )
    signals["event_group_id"] = [
        f"G05-PREDICTED-DATE-{value.isoformat()}"
        for value in signals["signal_date"]
    ]
    return signals.loc[
        :,
        [
            "signal_id",
            "signal_date",
            "signal_label",
            "event_group_id",
            "model_probability",
            "probability_lift",
            "expected_delta_pp",
        ],
    ]
