"""Direct probability calibration for the MN02 paired tactical policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

N_BRANCH_POLICY_HEAD_COLUMNS = (
    "top_l1_plus",
    "top_l2_plus",
    "top_l3",
    "bottom_l1_plus",
    "bottom_l2_plus",
    "bottom_l3",
)

# BN02 selected Sobol candidate 2726 in every causal policy block.  MN02
# freezes its six quantile locations and six tactical deltas, while
# recalibrating the raw probability thresholds inside each outer fold's
# inner-OOS tape.  No training prior or outer label enters this operation.
BN02_SELECTED_SOBOL_CANDIDATE = 2726
BN02_FROZEN_THRESHOLD_QUANTILES = np.asarray(
    [
        0.9784845730499365,
        0.9778323860210367,
        0.997303195798304,
        0.8784478970151395,
        0.9037410206347704,
        0.9258178230235353,
    ],
    dtype=float,
)
BN02_FROZEN_DELTAS = np.asarray(
    [-0.025, -0.075, -0.40, 0.05, 0.05, 0.20],
    dtype=float,
)


@dataclass(frozen=True)
class NBranchDirectPolicyCalibration:
    thresholds: np.ndarray
    quantiles: np.ndarray
    fit_rows: int
    audit: dict[str, Any]


def build_n_branch_boundary_tape(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Create one causal six-head probability record per target bucket."""

    required = {
        "boundary_action_eligible",
        "feature_anchor_bucket",
        "target_bucket",
        "horizon_index",
        "p_top_l1",
        "p_top_l2",
        "p_top_l3",
        "p_bottom_l1",
        "p_bottom_l2",
        "p_bottom_l3",
    }
    if missing := required - set(predictions.columns):
        raise ValueError(
            f"N-branch boundary tape missing: {sorted(missing)}"
        )
    frame = predictions.loc[
        predictions["boundary_action_eligible"].astype(bool)
    ].copy()
    if frame.empty:
        raise ValueError("N-branch boundary tape is empty")
    frame["feature_anchor_bucket"] = pd.to_datetime(
        frame["feature_anchor_bucket"],
        errors="raise",
    ).dt.date
    frame["target_bucket"] = pd.to_datetime(
        frame["target_bucket"],
        errors="raise",
    ).dt.date
    frame = (
        frame.sort_values(
            ["target_bucket", "feature_anchor_bucket", "horizon_index"],
            kind="stable",
        )
        .drop_duplicates("target_bucket", keep="last")
        .reset_index(drop=True)
    )
    frame["top_l1_plus"] = frame[
        ["p_top_l1", "p_top_l2", "p_top_l3"]
    ].sum(axis=1)
    frame["top_l2_plus"] = frame[["p_top_l2", "p_top_l3"]].sum(axis=1)
    frame["top_l3"] = frame["p_top_l3"]
    frame["bottom_l1_plus"] = frame[
        ["p_bottom_l1", "p_bottom_l2", "p_bottom_l3"]
    ].sum(axis=1)
    frame["bottom_l2_plus"] = frame[
        ["p_bottom_l2", "p_bottom_l3"]
    ].sum(axis=1)
    frame["bottom_l3"] = frame["p_bottom_l3"]
    probability = frame.loc[
        :, list(N_BRANCH_POLICY_HEAD_COLUMNS)
    ].to_numpy(dtype=float)
    if (
        not np.isfinite(probability).all()
        or (probability < -1.0e-12).any()
        or (probability > 1.0 + 1.0e-12).any()
    ):
        raise ValueError("N-branch boundary probabilities are invalid")
    if frame["target_bucket"].duplicated().any():
        raise AssertionError("N-branch boundary target buckets repeat")
    return frame


def calibrate_n_branch_direct_thresholds(
    tape: pd.DataFrame,
) -> NBranchDirectPolicyCalibration:
    """Map the frozen BN02 quantiles onto inner-OOS raw probabilities."""

    if missing := set(N_BRANCH_POLICY_HEAD_COLUMNS) - set(tape.columns):
        raise ValueError(
            f"N-branch calibration missing heads: {sorted(missing)}"
        )
    probability = tape.loc[
        :, list(N_BRANCH_POLICY_HEAD_COLUMNS)
    ].to_numpy(dtype=float)
    if len(probability) < 2 or not np.isfinite(probability).all():
        raise ValueError("N-branch calibration tape is invalid")
    thresholds = np.asarray(
        [
            np.quantile(
                probability[:, index],
                BN02_FROZEN_THRESHOLD_QUANTILES[index],
            )
            for index in range(len(N_BRANCH_POLICY_HEAD_COLUMNS))
        ],
        dtype=float,
    )
    if (
        not np.isfinite(thresholds).all()
        or (thresholds <= 0.0).any()
        or (thresholds > 1.0 + 1.0e-12).any()
    ):
        raise ValueError("N-branch direct thresholds are invalid")
    return NBranchDirectPolicyCalibration(
        thresholds=thresholds,
        quantiles=BN02_FROZEN_THRESHOLD_QUANTILES.copy(),
        fit_rows=len(tape),
        audit={
            "source_policy": "BN02_PAIRED_TACTICAL_INVENTORY",
            "source_sobol_candidate": BN02_SELECTED_SOBOL_CANDIDATE,
            "head_columns": list(N_BRANCH_POLICY_HEAD_COLUMNS),
            "threshold_quantiles": (
                BN02_FROZEN_THRESHOLD_QUANTILES.tolist()
            ),
            "deltas_percentage_points": (
                np.abs(BN02_FROZEN_DELTAS * 100.0).tolist()
            ),
            "fit_rows": len(tape),
            "training_priors_used": False,
            "outer_labels_used": False,
        },
    )


def n_branch_direct_actions(
    tape: pd.DataFrame,
    thresholds: np.ndarray,
    *,
    block_id: str,
) -> pd.DataFrame:
    """Convert six direct probability heads to at most one action per day."""

    probability = tape.loc[
        :, list(N_BRANCH_POLICY_HEAD_COLUMNS)
    ].to_numpy(dtype=float)
    boundary = np.asarray(thresholds, dtype=float)
    if boundary.shape != (6,) or not np.isfinite(boundary).all():
        raise ValueError("N-branch action thresholds must have shape (6,)")

    top_pass = probability[:, :3] >= boundary[:3]
    bottom_pass = probability[:, 3:] >= boundary[3:]
    top_level = np.where(
        top_pass[:, 2],
        3,
        np.where(top_pass[:, 1], 2, np.where(top_pass[:, 0], 1, 0)),
    ).astype(int)
    bottom_level = np.where(
        bottom_pass[:, 2],
        3,
        np.where(
            bottom_pass[:, 1],
            2,
            np.where(bottom_pass[:, 0], 1, 0),
        ),
    ).astype(int)
    row = np.arange(len(tape))
    top_index = np.maximum(top_level - 1, 0)
    bottom_index = np.maximum(bottom_level - 1, 0)
    top_probability = probability[row, top_index]
    bottom_probability = probability[row, 3 + bottom_index]
    top_threshold = boundary[top_index]
    bottom_threshold = boundary[3 + bottom_index]
    top_margin = np.where(
        top_level > 0,
        top_probability / np.maximum(top_threshold, 1.0e-15),
        -np.inf,
    )
    bottom_margin = np.where(
        bottom_level > 0,
        bottom_probability / np.maximum(bottom_threshold, 1.0e-15),
        -np.inf,
    )
    choose_top = (top_level > 0) & (top_margin > bottom_margin)
    choose_bottom = (bottom_level > 0) & ~choose_top
    signed_delta = np.zeros(len(tape), dtype=float)
    signed_delta[choose_top] = BN02_FROZEN_DELTAS[
        top_index[choose_top]
    ]
    signed_delta[choose_bottom] = BN02_FROZEN_DELTAS[
        3 + bottom_index[choose_bottom]
    ]
    side_code = np.where(choose_top, -1, np.where(choose_bottom, 1, 0))
    chosen_level = np.where(
        choose_top,
        top_level,
        np.where(choose_bottom, bottom_level, 0),
    )
    chosen_probability = np.where(
        choose_top,
        top_probability,
        np.where(choose_bottom, bottom_probability, np.nan),
    )
    chosen_threshold = np.where(
        choose_top,
        top_threshold,
        np.where(choose_bottom, bottom_threshold, np.nan),
    )
    chosen_margin = np.where(
        choose_top,
        top_margin,
        np.where(choose_bottom, bottom_margin, np.nan),
    )
    output = pd.DataFrame(
        {
            "internal_block_id": str(block_id),
            "trade_date": tape["target_bucket"].to_numpy(),
            "signed_delta": signed_delta,
            "side_code": side_code,
            "strength_level": chosen_level,
            "chosen_probability": chosen_probability,
            "chosen_threshold": chosen_threshold,
            "threshold_margin": chosen_margin,
        }
    )
    output = output.loc[output["side_code"].ne(0)].reset_index(drop=True)
    output["action_id"] = [
        f"MN02:{block_id}:{value.isoformat()}"
        for value in output["trade_date"]
    ]
    if output["trade_date"].duplicated().any():
        raise AssertionError("N-branch policy emitted duplicate action dates")
    return output
