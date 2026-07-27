from __future__ import annotations

from typing import Any

import pandas as pd

from aupilot.training.future_slot_logistic import (
    FutureSlotPredictionResult,
    fit_constant_prior,
    fit_multinomial_logistic,
)

MODEL_ID = "NEAR_H01_H05_LOGISTIC_FAR_EVENT_GROUP_PRIOR_V1"
NEAR_HORIZON_MINIMUM = 1
NEAR_HORIZON_MAXIMUM = 5
FAR_HORIZON_MINIMUM = 6
FAR_HORIZON_MAXIMUM = 21


def fit_near_logistic_far_prior(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    max_iter: int = 2000,
    tolerance: float = 1.0e-10,
) -> FutureSlotPredictionResult:
    """Fit a frozen near-horizon Logistic and far-horizon fold prior."""

    if "horizon_index" not in train.columns or "horizon_index" not in test:
        raise ValueError("R27 requires horizon_index")
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
        raise ValueError("R27 horizon_index must be exactly within 1...21")

    near_train = train.loc[
        train_horizon.between(
            NEAR_HORIZON_MINIMUM,
            NEAR_HORIZON_MAXIMUM,
        )
    ].reset_index(drop=True)
    ordered_test = test.copy().reset_index(drop=True)
    ordered_test["_r27_row_order"] = range(len(ordered_test))
    near_test = ordered_test.loc[
        test_horizon.between(
            NEAR_HORIZON_MINIMUM,
            NEAR_HORIZON_MAXIMUM,
        ).to_numpy()
    ].reset_index(drop=True)
    far_test = ordered_test.loc[
        test_horizon.between(
            FAR_HORIZON_MINIMUM,
            FAR_HORIZON_MAXIMUM,
        ).to_numpy()
    ].reset_index(drop=True)
    if (
        near_train.empty
        or near_test.empty
        or far_test.empty
        or len(near_test) + len(far_test) != len(test)
    ):
        raise ValueError("R27 near/far horizon partition is incomplete")

    near = fit_multinomial_logistic(
        near_train,
        near_test,
        feature_columns=feature_columns,
        max_iter=max_iter,
        tolerance=tolerance,
    )
    far = fit_constant_prior(train, far_test)
    predictions = (
        pd.concat(
            [near.predictions, far.predictions],
            ignore_index=True,
        )
        .sort_values("_r27_row_order", kind="stable")
        .drop(columns="_r27_row_order")
        .reset_index(drop=True)
    )
    if not predictions["horizon_index"].tolist() == test[
        "horizon_index"
    ].reset_index(drop=True).tolist():
        raise AssertionError("R27 prediction row order changed")

    audit: dict[str, Any] = {
        "model_id": MODEL_ID,
        "train_rows": len(train),
        "test_rows": len(test),
        "near_train_rows": len(near_train),
        "near_test_rows": len(near_test),
        "far_test_rows": len(far_test),
        "near_horizon_minimum": NEAR_HORIZON_MINIMUM,
        "near_horizon_maximum": NEAR_HORIZON_MAXIMUM,
        "far_horizon_minimum": FAR_HORIZON_MINIMUM,
        "far_horizon_maximum": FAR_HORIZON_MAXIMUM,
        "near_model_audit": near.audit,
        "far_model_audit": far.audit,
        "feature_columns": list(feature_columns),
        "class_weight": None,
        "resampling": False,
        "calibrated": False,
        "threshold_selected": False,
    }
    artifact = {
        "model_id": MODEL_ID,
        "near_model": near.artifact,
        "far_model": far.artifact,
        "near_horizons": [
            NEAR_HORIZON_MINIMUM,
            NEAR_HORIZON_MAXIMUM,
        ],
        "far_horizons": [
            FAR_HORIZON_MINIMUM,
            FAR_HORIZON_MAXIMUM,
        ],
    }
    return FutureSlotPredictionResult(
        predictions=predictions,
        audit=audit,
        artifact=artifact,
    )

