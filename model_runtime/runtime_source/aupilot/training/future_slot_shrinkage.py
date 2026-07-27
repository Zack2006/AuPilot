from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from aupilot.training.future_slot_hybrid import (
    fit_near_logistic_far_prior,
)
from aupilot.training.future_slot_logistic import (
    CLASS_LABELS,
    PROBABILITY_COLUMNS,
    FutureSlotPredictionResult,
    evaluate_future_slot_probabilities,
    fit_constant_prior,
)

MODEL_ID = "INNER_SELECTED_SHRUNK_NEAR_LOGISTIC_FAR_PRIOR_V1"
NEAR_HORIZON_MINIMUM = 1
NEAR_HORIZON_MAXIMUM = 5
FAR_HORIZON_MINIMUM = 6
FAR_HORIZON_MAXIMUM = 21
IDENTITY_COLUMNS = (
    "issuance_id",
    "horizon_index",
    "target_bucket",
    "target_event_group_id",
    "target_label",
)


@dataclass(frozen=True)
class ShrinkageSelection:
    selected_alpha: float
    metrics: pd.DataFrame
    audit: dict[str, Any]


def _validated_alpha_grid(values: Iterable[float]) -> tuple[float, ...]:
    grid = tuple(float(value) for value in values)
    if (
        not grid
        or tuple(sorted(set(grid))) != grid
        or not np.isfinite(np.asarray(grid, dtype=float)).all()
        or grid[0] < 0.0
        or grid[-1] > 1.0
    ):
        raise ValueError(
            "R28 alpha grid must be unique, ordered, finite, and within [0,1]"
        )
    return grid


def _validate_prediction_pair(
    hybrid: pd.DataFrame,
    constant: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(hybrid) != len(constant) or hybrid.empty:
        raise ValueError("R28 prediction pair has different or zero rows")
    missing = (
        set(IDENTITY_COLUMNS + PROBABILITY_COLUMNS)
        - set(hybrid.columns)
    ) | (
        set(IDENTITY_COLUMNS + PROBABILITY_COLUMNS)
        - set(constant.columns)
    )
    if missing:
        raise ValueError(f"R28 prediction pair is missing: {sorted(missing)}")
    left = hybrid.reset_index(drop=True)
    right = constant.reset_index(drop=True)
    if not left.loc[:, IDENTITY_COLUMNS].equals(
        right.loc[:, IDENTITY_COLUMNS]
    ):
        raise ValueError("R28 prediction identities differ")
    for frame in (left, right):
        horizon = pd.to_numeric(frame["horizon_index"], errors="coerce")
        probability = frame.loc[:, PROBABILITY_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        ).to_numpy(dtype=float)
        if (
            horizon.isna().any()
            or not horizon.between(1, 21).all()
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
            raise ValueError("R28 prediction pair is invalid")
    return left, right


def blend_near_with_constant(
    hybrid: pd.DataFrame,
    constant: pd.DataFrame,
    *,
    alpha: float,
) -> pd.DataFrame:
    """Shrink h1-h5 probabilities to a fold prior; keep h6-h21 prior-only."""

    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("R28 alpha must be finite and within [0,1]")
    left, right = _validate_prediction_pair(hybrid, constant)
    near_mask = left["horizon_index"].between(
        NEAR_HORIZON_MINIMUM,
        NEAR_HORIZON_MAXIMUM,
    ).to_numpy()
    far_mask = left["horizon_index"].between(
        FAR_HORIZON_MINIMUM,
        FAR_HORIZON_MAXIMUM,
    ).to_numpy()
    if not np.logical_xor(near_mask, far_mask).all():
        raise ValueError("R28 near/far partition is incomplete")
    hybrid_probability = left.loc[:, PROBABILITY_COLUMNS].to_numpy(
        dtype=float
    )
    constant_probability = right.loc[:, PROBABILITY_COLUMNS].to_numpy(
        dtype=float
    )
    probability = constant_probability.copy()
    probability[near_mask] = (
        alpha * hybrid_probability[near_mask]
        + (1.0 - alpha) * constant_probability[near_mask]
    )
    if not np.allclose(
        probability[far_mask],
        constant_probability[far_mask],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("R28 far probabilities changed")
    output = left.copy()
    output.loc[:, PROBABILITY_COLUMNS] = probability
    output["display_class"] = np.asarray(CLASS_LABELS, dtype=object)[
        np.argmax(probability, axis=1)
    ]
    return output


def select_inner_shrinkage_alpha(
    hybrid_oof: pd.DataFrame,
    constant_oof: pd.DataFrame,
    *,
    alpha_grid: Iterable[float],
    tie_tolerance: float = 1.0e-12,
) -> ShrinkageSelection:
    """Select one scalar only from inner OOF h1-h5 group log loss."""

    grid = _validated_alpha_grid(alpha_grid)
    if tie_tolerance < 0.0 or not np.isfinite(tie_tolerance):
        raise ValueError("R28 tie tolerance must be finite and non-negative")
    left, right = _validate_prediction_pair(hybrid_oof, constant_oof)
    if not left["horizon_index"].between(1, 5).all():
        raise ValueError("R28 alpha selection accepts only h1-h5 OOF rows")
    records: list[dict[str, Any]] = []
    for alpha in grid:
        blended = blend_near_with_constant(
            left,
            right,
            alpha=alpha,
        )
        metric = evaluate_future_slot_probabilities(
            blended,
            group_normalized=True,
        )
        records.append(
            {
                "alpha": alpha,
                "multiclass_log_loss": metric["multiclass_log_loss"],
                "multiclass_brier": metric["multiclass_brier"],
                "top_ap_over_prevalence": metric["classes"][
                    "TOP_ACTION_ZONE"
                ]["ap_over_prevalence"],
                "bottom_ap_over_prevalence": metric["classes"][
                    "BOTTOM_ACTION_ZONE"
                ]["ap_over_prevalence"],
                "rows": len(blended),
                "event_groups": int(
                    blended["target_event_group_id"].nunique()
                ),
            }
        )
    metrics = pd.DataFrame.from_records(records).sort_values(
        "alpha",
        kind="stable",
    )
    minimum = float(metrics["multiclass_log_loss"].min())
    eligible = metrics.loc[
        metrics["multiclass_log_loss"].le(minimum + tie_tolerance)
    ]
    selected_alpha = float(eligible["alpha"].min())
    metrics["selected"] = metrics["alpha"].eq(selected_alpha)
    return ShrinkageSelection(
        selected_alpha=selected_alpha,
        metrics=metrics.reset_index(drop=True),
        audit={
            "selection_scope": "INNER_OOF_H01_H05_ONLY",
            "selection_metric": "GROUP_NORMALIZED_MULTICLASS_LOG_LOSS",
            "alpha_grid": list(grid),
            "tie_tolerance": tie_tolerance,
            "tie_break": "LOWEST_ALPHA",
            "selected_alpha": selected_alpha,
            "rows": len(left),
            "event_groups": int(
                left["target_event_group_id"].nunique()
            ),
            "outer_labels_used_for_selection": False,
        },
    )


def fit_shrunk_near_logistic_far_prior(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    alpha: float,
    max_iter: int = 2000,
    tolerance: float = 1.0e-10,
) -> FutureSlotPredictionResult:
    """Fit the R27 base model, then apply an already-selected scalar."""

    hybrid = fit_near_logistic_far_prior(
        train,
        test,
        feature_columns=feature_columns,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    constant = fit_constant_prior(train, test)
    predictions = blend_near_with_constant(
        hybrid.predictions,
        constant.predictions,
        alpha=alpha,
    )
    return FutureSlotPredictionResult(
        predictions=predictions,
        audit={
            "model_id": MODEL_ID,
            "selected_alpha": float(alpha),
            "train_rows": len(train),
            "test_rows": len(test),
            "feature_columns": list(feature_columns),
            "base_hybrid_audit": hybrid.audit,
            "constant_audit": constant.audit,
            "calibration": "INNER_SELECTED_LINEAR_SHRINKAGE_TO_FOLD_PRIOR",
            "class_weight": None,
            "resampling": False,
            "threshold_selected": False,
        },
        artifact={
            "model_id": MODEL_ID,
            "selected_alpha": float(alpha),
            "base_hybrid": hybrid.artifact,
            "constant_prior": constant.artifact,
            "near_horizons": [1, 5],
            "far_horizons": [6, 21],
        },
    )
