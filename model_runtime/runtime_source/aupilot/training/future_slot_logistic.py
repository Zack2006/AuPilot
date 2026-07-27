from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

CLASS_LABELS = (
    "TOP_ACTION_ZONE",
    "BOTTOM_ACTION_ZONE",
    "NORMAL",
)
PROBABILITY_COLUMNS = (
    "p_top_action_zone",
    "p_bottom_action_zone",
    "p_normal",
)


@dataclass(frozen=True)
class FutureSlotPredictionResult:
    predictions: pd.DataFrame
    audit: dict[str, Any]
    artifact: dict[str, Any]


def group_normalized_weights(frame: pd.DataFrame) -> np.ndarray:
    if "target_event_group_id" not in frame.columns:
        raise ValueError("Missing target_event_group_id for R26 weights")
    groups = frame["target_event_group_id"].astype(str)
    counts = groups.map(groups.value_counts())
    weights = 1.0 / counts.to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("R26 group-normalized weights are invalid")
    group_sums = pd.Series(weights).groupby(groups.reset_index(drop=True)).sum()
    if not np.allclose(
        group_sums.to_numpy(dtype=float),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("R26 event-group weights do not sum to one")
    return weights


def _validate_labels(frame: pd.DataFrame) -> np.ndarray:
    if "target_label" not in frame.columns:
        raise ValueError("Missing R26 target_label")
    labels = frame["target_label"].astype(str).to_numpy()
    unknown = sorted(set(labels) - set(CLASS_LABELS))
    if unknown:
        raise ValueError(f"Unsupported R26 labels: {unknown}")
    return labels


def _probability_frame(
    frame: pd.DataFrame,
    probability: np.ndarray,
) -> pd.DataFrame:
    if probability.shape != (len(frame), len(CLASS_LABELS)):
        raise ValueError("R26 probability shape is invalid")
    if (
        not np.isfinite(probability).all()
        or (probability < 0.0).any()
        or (probability > 1.0).any()
        or not np.allclose(
            probability.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1.0e-10,
        )
    ):
        raise ValueError("R26 probabilities are invalid")
    output = frame.copy().reset_index(drop=True)
    for index, column in enumerate(PROBABILITY_COLUMNS):
        output[column] = probability[:, index]
    output["display_class"] = np.asarray(CLASS_LABELS, dtype=object)[
        np.argmax(probability, axis=1)
    ]
    return output


def fit_constant_prior(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> FutureSlotPredictionResult:
    labels = _validate_labels(train)
    _validate_labels(test)
    weights = group_normalized_weights(train)
    class_weight = {
        label: float(weights[labels == label].sum())
        for label in CLASS_LABELS
    }
    total = float(weights.sum())
    prior = np.asarray(
        [class_weight[label] / total for label in CLASS_LABELS],
        dtype=float,
    )
    if (prior <= 0.0).any() or not np.isclose(prior.sum(), 1.0):
        raise ValueError("R26 constant prior lacks a class")
    probability = np.repeat(prior.reshape(1, -1), len(test), axis=0)
    return FutureSlotPredictionResult(
        predictions=_probability_frame(test, probability),
        audit={
            "model_id": "CONSTANT_EVENT_GROUP_PRIOR",
            "train_rows": len(train),
            "test_rows": len(test),
            "train_event_groups": int(
                train["target_event_group_id"].nunique()
            ),
            "class_prior": {
                label: float(prior[index])
                for index, label in enumerate(CLASS_LABELS)
            },
            "event_group_total_weight": 1.0,
            "class_weight": None,
            "resampling": False,
            "calibrated": False,
        },
        artifact={
            "model_id": "CONSTANT_EVENT_GROUP_PRIOR",
            "class_labels": list(CLASS_LABELS),
            "class_prior": prior.tolist(),
        },
    )


def fit_multinomial_logistic(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    max_iter: int = 2000,
    tolerance: float = 1.0e-10,
) -> FutureSlotPredictionResult:
    if not feature_columns or len(set(feature_columns)) != len(
        feature_columns
    ):
        raise ValueError("R26 feature columns must be non-empty and unique")
    missing_train = set(feature_columns) - set(train.columns)
    missing_test = set(feature_columns) - set(test.columns)
    if missing_train or missing_test:
        raise ValueError(
            "Missing R26 model features: "
            f"train={sorted(missing_train)}, test={sorted(missing_test)}"
        )
    labels = _validate_labels(train)
    _validate_labels(test)
    if set(labels) != set(CLASS_LABELS):
        raise ValueError("R26 logistic training requires all three classes")
    x_train = train.loc[:, feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    x_test = test.loc[:, feature_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("R26 logistic features contain non-finite values")
    weights = group_normalized_weights(train)
    total_weight = float(weights.sum())
    means = np.average(x_train, axis=0, weights=weights)
    variances = np.average(
        np.square(x_train - means),
        axis=0,
        weights=weights,
    )
    standard_deviations = np.sqrt(variances)
    if (
        not np.isfinite(means).all()
        or not np.isfinite(standard_deviations).all()
        or (standard_deviations <= 0.0).any()
    ):
        raise ValueError("R26 logistic feature has zero weighted variance")
    z_train = (x_train - means) / standard_deviations
    z_test = (x_test - means) / standard_deviations
    model = LogisticRegression(
        C=np.inf,
        solver="lbfgs",
        fit_intercept=True,
        class_weight=None,
        max_iter=max_iter,
        tol=tolerance,
    )
    model.fit(z_train, labels, sample_weight=weights)
    if int(model.n_iter_[0]) >= max_iter:
        raise RuntimeError("R26 multinomial Logistic did not converge")
    raw_probability = model.predict_proba(z_test)
    probability = np.column_stack(
        [
            raw_probability[:, int(np.flatnonzero(model.classes_ == label)[0])]
            for label in CLASS_LABELS
        ]
    )
    coefficients = {
        str(class_label): {
            feature: float(model.coef_[class_index, feature_index])
            for feature_index, feature in enumerate(feature_columns)
        }
        for class_index, class_label in enumerate(model.classes_)
    }
    intercepts = {
        str(class_label): float(model.intercept_[class_index])
        for class_index, class_label in enumerate(model.classes_)
    }
    return FutureSlotPredictionResult(
        predictions=_probability_frame(test, probability),
        audit={
            "model_id": "MULTINOMIAL_LOGISTIC_CORE_V1",
            "train_rows": len(train),
            "test_rows": len(test),
            "train_event_groups": int(
                train["target_event_group_id"].nunique()
            ),
            "train_total_weight": total_weight,
            "feature_columns": list(feature_columns),
            "feature_means": {
                feature: float(means[index])
                for index, feature in enumerate(feature_columns)
            },
            "feature_standard_deviations": {
                feature: float(standard_deviations[index])
                for index, feature in enumerate(feature_columns)
            },
            "coefficients": coefficients,
            "intercepts": intercepts,
            "iterations": int(model.n_iter_[0]),
            "effective_parameter_count": (
                (len(CLASS_LABELS) - 1) * (len(feature_columns) + 1)
            ),
            "regularization": None,
            "class_weight": None,
            "resampling": False,
            "calibrated": False,
        },
        artifact={
            "model_id": "MULTINOMIAL_LOGISTIC_CORE_V1",
            "class_labels": list(model.classes_),
            "feature_columns": list(feature_columns),
            "feature_means": means.tolist(),
            "feature_standard_deviations": standard_deviations.tolist(),
            "coefficients": model.coef_.tolist(),
            "intercepts": model.intercept_.tolist(),
        },
    )


def evaluate_future_slot_probabilities(
    predictions: pd.DataFrame,
    *,
    probability_columns: tuple[str, ...] = PROBABILITY_COLUMNS,
    group_normalized: bool,
) -> dict[str, Any]:
    labels = _validate_labels(predictions)
    probability = predictions.loc[:, probability_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    if (
        probability.shape != (len(predictions), len(CLASS_LABELS))
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
        raise ValueError("R26 evaluation probabilities are invalid")
    weights = (
        group_normalized_weights(predictions)
        if group_normalized
        else np.ones(len(predictions), dtype=float)
    )
    normalized_weights = weights / float(weights.sum())
    truth = np.column_stack(
        [(labels == label).astype(float) for label in CLASS_LABELS]
    )
    clipped = np.clip(probability, 1.0e-15, 1.0)
    row_log_loss = -np.sum(truth * np.log(clipped), axis=1)
    row_brier = np.sum(np.square(probability - truth), axis=1)
    output: dict[str, Any] = {
        "rows": len(predictions),
        "event_groups": int(
            predictions["target_event_group_id"].nunique()
        ),
        "group_normalized": group_normalized,
        "multiclass_log_loss": float(
            np.sum(normalized_weights * row_log_loss)
        ),
        "multiclass_brier": float(
            np.sum(normalized_weights * row_brier)
        ),
        "argmax_accuracy": float(
            np.sum(
                normalized_weights
                * (
                    np.asarray(CLASS_LABELS, dtype=object)[
                        np.argmax(probability, axis=1)
                    ]
                    == labels
                )
            )
        ),
        "classes": {},
    }
    for index, label in enumerate(CLASS_LABELS):
        binary = (labels == label).astype(int)
        prevalence = float(np.sum(normalized_weights * binary))
        class_metrics: dict[str, float | None] = {
            "prevalence": prevalence,
            "roc_auc": None,
            "average_precision": None,
            "ap_over_prevalence": None,
        }
        if len(np.unique(binary)) == 2:
            auc = float(
                roc_auc_score(
                    binary,
                    probability[:, index],
                    sample_weight=weights,
                )
            )
            ap = float(
                average_precision_score(
                    binary,
                    probability[:, index],
                    sample_weight=weights,
                )
            )
            class_metrics.update(
                {
                    "roc_auc": auc,
                    "average_precision": ap,
                    "ap_over_prevalence": (
                        ap / prevalence if prevalence > 0.0 else None
                    ),
                }
            )
        output["classes"][label] = class_metrics
    return output
