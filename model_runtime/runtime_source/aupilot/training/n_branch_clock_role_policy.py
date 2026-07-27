"""Causal clock-role routing for the MN03 paired tactical policy."""

from __future__ import annotations

import pandas as pd

from aupilot.training.n_branch_direct_policy import (
    n_branch_direct_actions,
)

CONTROL_ALL_CLOCK_BOTH_SIDES = "CONTROL_ALL_CLOCK_BOTH_SIDES"
DUAL_CLOCK_H1_TOP_H2_BOTTOM = "DUAL_CLOCK_H1_TOP_H2_BOTTOM"
MN03_POLICY_CANDIDATES = (
    CONTROL_ALL_CLOCK_BOTH_SIDES,
    DUAL_CLOCK_H1_TOP_H2_BOTTOM,
)


def n_branch_clock_role_actions(
    tape: pd.DataFrame,
    thresholds,
    *,
    block_id: str,
    policy_candidate_id: str,
) -> pd.DataFrame:
    """Generate direct actions and apply one preregistered clock-role route.

    The router uses only the causal boundary horizon and predicted side:

    * control: retain both sides at h1 and h2;
    * paired role: retain h1 TOP sells and h2 BOTTOM buybacks.

    It never reads labels, prices, returns, quarters, or portfolio state.
    Portfolio feasibility remains the responsibility of the frozen BN02
    paired-inventory replay.
    """

    if policy_candidate_id not in MN03_POLICY_CANDIDATES:
        raise ValueError(
            "Unknown MN03 policy candidate: "
            f"{policy_candidate_id!r}"
        )
    required = {"target_bucket", "horizon_index"}
    if missing := required - set(tape.columns):
        raise ValueError(
            f"MN03 boundary tape missing: {sorted(missing)}"
        )

    boundary = tape.loc[
        :, ["target_bucket", "horizon_index"]
    ].copy()
    boundary = boundary.rename(
        columns={
            "target_bucket": "trade_date",
            "horizon_index": "boundary_horizon_index",
        }
    )
    boundary["trade_date"] = pd.to_datetime(
        boundary["trade_date"],
        errors="raise",
    ).dt.date
    boundary["boundary_horizon_index"] = boundary[
        "boundary_horizon_index"
    ].astype(int)
    if boundary["trade_date"].duplicated().any():
        raise AssertionError("MN03 boundary dates repeat")
    if not set(boundary["boundary_horizon_index"]).issubset({1, 2}):
        raise AssertionError("MN03 boundary contains a horizon beyond h1/h2")

    actions = n_branch_direct_actions(
        tape,
        thresholds,
        block_id=block_id,
    )
    actions["trade_date"] = pd.to_datetime(
        actions["trade_date"],
        errors="raise",
    ).dt.date
    actions = actions.merge(
        boundary,
        on="trade_date",
        how="left",
        validate="one_to_one",
    )
    if actions["boundary_horizon_index"].isna().any():
        raise AssertionError("MN03 action lacks a boundary horizon")
    actions["boundary_horizon_index"] = actions[
        "boundary_horizon_index"
    ].astype(int)

    if policy_candidate_id == DUAL_CLOCK_H1_TOP_H2_BOTTOM:
        keep = (
            actions["boundary_horizon_index"].eq(1)
            & actions["side_code"].eq(-1)
        ) | (
            actions["boundary_horizon_index"].eq(2)
            & actions["side_code"].eq(1)
        )
        actions = actions.loc[keep].copy()

    actions["policy_candidate_id"] = policy_candidate_id
    actions["action_id"] = [
        f"MN03:{block_id}:{value.isoformat()}"
        for value in actions["trade_date"]
    ]
    actions = actions.reset_index(drop=True)
    if actions["trade_date"].duplicated().any():
        raise AssertionError("MN03 routed actions duplicate dates")
    return actions

