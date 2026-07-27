"""Versioned full-horizon adapter for MN02 boundary-domain parity.

The historical graded model intentionally rejects any training-fold prior
that lacks one of the seven exact classes. Small early inner folds can still
fit the direction and ordinal heads while lacking one exact strength class.
This module preserves the historical implementation byte-for-byte and gives
only the far-horizon empirical prior an explicit zero for unobserved classes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from aupilot.training.graded_future_slot import (
    DIRECTION_LABELS,
    GRADED_LABELS,
    PROBABILITY_COLUMNS,
    PROBE_MODELS,
    GradedProbeResult,
    _direction,
    _fit_classifier,
    _level,
    _numeric_features,
    _ordered_probability,
    _positive_probability,
    _project_ordinal,
    _weighted_prior,
    group_normalized_weights,
)
from aupilot.training.graded_hurdle_lightgbm import M01_MODEL_ID
from aupilot.training.n_branch_hurdle_lightgbm import (
    MN02_MODEL_ID,
    fit_mn02_h1_hurdle_ordinal,
)


def _zero_support_empirical_prior(
    labels: pd.Series,
    weights: np.ndarray,
) -> np.ndarray:
    values = labels.astype(str).to_numpy()
    invalid = sorted(set(values) - set(GRADED_LABELS))
    if invalid:
        raise ValueError(
            f"Unsupported boundary-parity labels: {invalid}"
        )
    prior = np.asarray(
        [weights[values == label].sum() for label in GRADED_LABELS],
        dtype=float,
    )
    total = float(prior.sum())
    if (
        not np.isfinite(prior).all()
        or (prior < 0.0).any()
        or total <= 0.0
    ):
        raise ValueError("Boundary-parity far prior is invalid")
    return prior / total


def fit_boundary_parity_graded_ordinal_probe(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    model_id: str,
    feature_columns: tuple[str, ...],
    near_horizon_maximum: int = 5,
) -> GradedProbeResult:
    """Fit the frozen graded heads with an exact-zero far prior."""

    if model_id not in PROBE_MODELS:
        raise ValueError(f"Unknown graded probe model: {model_id}")
    train_horizon = pd.to_numeric(
        train["horizon_index"],
        errors="coerce",
    )
    test_horizon = pd.to_numeric(
        test["horizon_index"],
        errors="coerce",
    )
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
    side_top = (
        action_train["target_label"]
        .astype(str)
        .str.startswith("TOP_")
        .astype(float)
        .to_numpy()
    )
    action_x = np.column_stack([action_x, side_top])
    top_test_x = np.column_stack([x_near, np.ones(len(x_near))])
    bottom_test_x = np.column_stack([x_near, np.zeros(len(x_near))])

    ordinal_probabilities: dict[
        str,
        tuple[np.ndarray, np.ndarray],
    ] = {}
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
    flat_prior = _zero_support_empirical_prior(
        train["target_label"],
        full_weights,
    )
    far_flat = np.repeat(
        flat_prior.reshape(1, -1),
        len(far_test),
        axis=0,
    )
    flat = np.vstack([near_flat, far_flat])
    combined = pd.concat([near_test, far_test], ignore_index=True)
    for index, column in enumerate(PROBABILITY_COLUMNS):
        combined[column] = flat[:, index]
    combined["display_class"] = np.asarray(
        GRADED_LABELS,
        dtype=object,
    )[np.argmax(flat, axis=1)]
    combined["train_prior_top_action"] = direction_prior[0]
    combined["train_prior_bottom_action"] = direction_prior[1]
    combined = (
        combined.sort_values("_graded_row_order", kind="stable")
        .drop(columns="_graded_row_order")
        .reset_index(drop=True)
    )
    probability = combined.loc[
        :,
        PROBABILITY_COLUMNS,
    ].to_numpy(dtype=float)
    if (
        not np.isfinite(probability).all()
        or (probability < -1.0e-12).any()
        or (probability > 1.0 + 1.0e-12).any()
        or not np.allclose(probability.sum(axis=1), 1.0, atol=1.0e-10)
    ):
        raise AssertionError("Graded seven-class probabilities are invalid")
    zero_support = [
        label
        for label, value in zip(
            GRADED_LABELS,
            flat_prior,
            strict=True,
        )
        if value == 0.0
    ]
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
            "flat_prior": dict(
                zip(GRADED_LABELS, flat_prior, strict=True)
            ),
            "zero_support_far_prior_allowed": True,
            "zero_support_far_prior_classes": zero_support,
            "zero_support_far_prior_policy": (
                "EXACT_ZERO_NO_SMOOTHING_NO_PSEUDO_LABELS"
            ),
            "direction_head": direction_audit,
            "ordinal_heads": ordinal_audits,
            "outer_labels_used_for_fit_or_threshold": False,
            "threshold_selected": False,
        },
    )


def fit_mn02_boundary_parity_full_horizon(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    candidate_id: str,
    all_feature_columns: tuple[str, ...],
) -> GradedProbeResult:
    """Generate all 21 slots while preserving the frozen MN02 h1 head."""

    base = fit_boundary_parity_graded_ordinal_probe(
        train,
        test,
        model_id="FIXED_SMALL_LGBM_ORDINAL",
        feature_columns=all_feature_columns,
        near_horizon_maximum=5,
    )
    h1 = fit_mn02_h1_hurdle_ordinal(
        train,
        test,
        candidate_id=candidate_id,
    )
    output = base.predictions.copy()
    h1_mask = output["horizon_index"].eq(1)
    replacement = [
        *PROBABILITY_COLUMNS,
        "display_class",
        "train_prior_top_action",
        "train_prior_bottom_action",
    ]
    output.loc[h1_mask, replacement] = h1.predictions.loc[
        :,
        replacement,
    ].to_numpy()
    return GradedProbeResult(
        predictions=output,
        audit={
            "model_id": MN02_MODEL_ID,
            "candidate_id": candidate_id,
            "architecture_parent": M01_MODEL_ID,
            "h1": h1.audit,
            "h2_h21": {
                "model_id": M01_MODEL_ID,
                "h2_h5": base.audit,
                "h6_h21": (
                    "TRAIN_FOLD_EVENT_GROUP_NORMALIZED_SEVEN_CLASS_PRIOR"
                ),
            },
            "automatic_action_horizon": 1,
            "zero_support_far_prior_allowed": True,
            "outer_labels_used_for_fit_or_candidate_selection": False,
        },
    )
