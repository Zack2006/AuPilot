"""MN02 low-capacity hurdle model with separated exposure weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aupilot.training.graded_future_slot import (
    DIRECTION_LABELS,
    GRADED_LABELS,
    PROBABILITY_COLUMNS,
    GradedProbeResult,
    group_normalized_weights,
)
from aupilot.training.graded_hurdle_lightgbm import (
    M01_H1_FEATURE_COLUMNS,
    M01_MODEL_ID,
    M01PredictionResult,
    _features,
    _fit_binary,
    _project_ordinal,
    fit_m01_full_horizon,
)
from aupilot.training.n_branch_market_geometry import (
    N_BRANCH_MARKET_GEOMETRY_COLUMNS,
    N_BRANCH_MARKET_GEOMETRY_COMPLETE,
)

MN02_MODEL_ID = "MN02_GN01_SQRT_EXPOSURE_HURDLE_ORDINAL_LIGHTGBM"
MN02_BASELINE_FEATURE_COLUMNS = M01_H1_FEATURE_COLUMNS
MN02_GEOMETRY_FEATURE_COLUMNS = (
    *M01_H1_FEATURE_COLUMNS,
    *N_BRANCH_MARKET_GEOMETRY_COLUMNS,
)
MN02_CANDIDATE_FEATURES = {
    "SQRT_EXPOSURE_BASELINE": MN02_BASELINE_FEATURE_COLUMNS,
    "SQRT_EXPOSURE_MARKET_GEOMETRY": MN02_GEOMETRY_FEATURE_COLUMNS,
}


@dataclass(frozen=True)
class MN02WeightAudit:
    rows: int
    groups: int
    effective_rows: float
    minimum_group_total_weight: float
    median_group_total_weight: float
    maximum_group_total_weight: float


def sqrt_event_exposure_weights(frame: pd.DataFrame) -> np.ndarray:
    """Give an event group total weight of sqrt(number of exposed rows)."""

    if "target_event_group_id" not in frame.columns:
        raise ValueError("MN02 weights lack target_event_group_id")
    groups = frame["target_event_group_id"].astype(str)
    counts = groups.map(groups.value_counts()).to_numpy(dtype=float)
    weights = 1.0 / np.sqrt(counts)
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("MN02 exposure weights are invalid")
    return weights


def _weight_audit(
    frame: pd.DataFrame,
    weights: np.ndarray,
) -> MN02WeightAudit:
    groups = frame["target_event_group_id"].astype(str)
    totals = pd.Series(weights).groupby(groups, sort=False).sum()
    total = float(weights.sum())
    effective = total * total / float(np.square(weights).sum())
    return MN02WeightAudit(
        rows=len(frame),
        groups=int(groups.nunique()),
        effective_rows=effective,
        minimum_group_total_weight=float(totals.min()),
        median_group_total_weight=float(totals.median()),
        maximum_group_total_weight=float(totals.max()),
    )


def _labels(frame: pd.DataFrame) -> pd.Series:
    labels = frame["target_label"].astype(str)
    invalid = sorted(set(labels) - set(GRADED_LABELS))
    if invalid:
        raise ValueError(f"MN02 unsupported labels: {invalid}")
    return labels


def _complete_training_rows(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> pd.DataFrame:
    uses_geometry = any(
        column in N_BRANCH_MARKET_GEOMETRY_COLUMNS
        for column in feature_columns
    )
    if not uses_geometry:
        return frame.reset_index(drop=True)
    if N_BRANCH_MARKET_GEOMETRY_COMPLETE not in frame:
        raise ValueError("MN02 geometry candidate lacks completeness flag")
    return frame.loc[
        frame[N_BRANCH_MARKET_GEOMETRY_COMPLETE].astype(bool)
    ].reset_index(drop=True)


def fit_mn02_h1_hurdle_ordinal(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    candidate_id: str,
) -> M01PredictionResult:
    """Fit one preregistered MN02 h1 candidate without outer selection."""

    if candidate_id not in MN02_CANDIDATE_FEATURES:
        raise ValueError(f"Unknown MN02 candidate: {candidate_id}")
    feature_columns = MN02_CANDIDATE_FEATURES[candidate_id]
    train_h1 = train.loc[train["horizon_index"].eq(1)].reset_index(drop=True)
    test_h1 = test.loc[test["horizon_index"].eq(1)].reset_index(drop=True)
    train_h1 = _complete_training_rows(train_h1, feature_columns)
    if train_h1.empty or test_h1.empty:
        raise ValueError("MN02 h1 train/test must be nonempty")
    if any(
        column in N_BRANCH_MARKET_GEOMETRY_COLUMNS
        for column in feature_columns
    ) and not test_h1[N_BRANCH_MARKET_GEOMETRY_COMPLETE].astype(bool).all():
        raise ValueError("MN02 geometry outer test contains warm-up rows")
    train_labels = _labels(train_h1)
    _labels(test_h1)
    x_train = _features(train_h1, feature_columns)
    x_test = _features(test_h1, feature_columns)

    action_weights = sqrt_event_exposure_weights(train_h1)
    action_truth = train_labels.ne("NORMAL").astype(int).to_numpy()
    p_action, action_audit = _fit_binary(
        x_train,
        action_truth,
        action_weights,
        x_test,
    )

    action_rows = train_h1.loc[action_truth.astype(bool)].reset_index(drop=True)
    action_labels = action_rows["target_label"].astype(str)
    action_x = _features(action_rows, feature_columns)
    event_equal_action_weights = group_normalized_weights(action_rows)
    top_truth = action_labels.str.startswith("TOP_").astype(int).to_numpy()
    p_top_given_action, direction_audit = _fit_binary(
        action_x,
        top_truth,
        event_equal_action_weights,
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
            event_equal_action_weights,
            top_test_x,
        )
        p_bottom, _ = _fit_binary(
            strength_x,
            truth,
            event_equal_action_weights,
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
        raise AssertionError("MN02 seven-class probabilities are invalid")

    output = test_h1.copy()
    for index, column in enumerate(PROBABILITY_COLUMNS):
        output[column] = probability[:, index]
    output["display_class"] = np.asarray(GRADED_LABELS, dtype=object)[
        np.argmax(probability, axis=1)
    ]
    total_weight = float(action_weights.sum())
    p_top_prior = float(
        action_weights[
            train_labels.str.startswith("TOP_").to_numpy()
        ].sum()
        / total_weight
    )
    p_bottom_prior = float(
        action_weights[
            train_labels.str.startswith("BOTTOM_").to_numpy()
        ].sum()
        / total_weight
    )
    if p_top_prior <= 0.0 or p_bottom_prior <= 0.0:
        raise ValueError("MN02 h1 training priors lack a direction")
    output["train_prior_top_action"] = p_top_prior
    output["train_prior_bottom_action"] = p_bottom_prior
    weight_audit = _weight_audit(train_h1, action_weights)
    return M01PredictionResult(
        predictions=output,
        audit={
            "model_id": MN02_MODEL_ID,
            "candidate_id": candidate_id,
            "architecture_parent": M01_MODEL_ID,
            "feature_columns": list(feature_columns),
            "train_rows": len(train_h1),
            "test_rows": len(test_h1),
            "train_action_rows": len(action_rows),
            "train_direction_counts": {
                label: int(
                    (
                        np.where(top_truth == 1, "TOP", "BOTTOM") == label
                    ).sum()
                )
                for label in DIRECTION_LABELS[:2]
            },
            "train_prior_top_action": p_top_prior,
            "train_prior_bottom_action": p_bottom_prior,
            "action_head_weighting": "EVENT_GROUP_TOTAL_SQRT_EXPOSURE",
            "direction_and_strength_weighting": (
                "EVENT_GROUP_TOTAL_ONE_WITHIN_ACTION_ROWS"
            ),
            "action_weight_audit": weight_audit.__dict__,
            "action_head": action_audit,
            "direction_head": direction_audit,
            "ordinal_heads": ordinal_audits,
            "outer_labels_used_for_fit_or_candidate_selection": False,
        },
    )


def fit_mn02_full_horizon(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    candidate_id: str,
    all_feature_columns: tuple[str, ...],
) -> GradedProbeResult:
    """Keep the MN01 display model and replace only the automatic h1 head."""

    base = fit_m01_full_horizon(
        train,
        test,
        all_feature_columns=all_feature_columns,
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
        :, replacement
    ].to_numpy()
    return GradedProbeResult(
        predictions=output,
        audit={
            "model_id": MN02_MODEL_ID,
            "candidate_id": candidate_id,
            "architecture_parent": M01_MODEL_ID,
            "h1": h1.audit,
            "h2_h21": base.audit,
            "automatic_action_horizon": 1,
            "outer_labels_used_for_fit_or_candidate_selection": False,
        },
    )
