from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import pandas as pd

from aupilot.backtest.pivot_baseline import validate_daily_ohlc
from aupilot.training.future_slot_logistic import PROBABILITY_COLUMNS

FIXED_SIDE_REGIME_GATE_ID = "FIXED_SIDE_SPECIFIC_CAUSAL_REGIME_GATE_V1"
GATED_EVENT_COLUMNS = (
    "event_id",
    "sequence_id",
    "event_type",
    "signal_date",
    "confirmation_date",
    "execution_date",
    "execution_open",
    "fold_id",
    "issuance_id",
    "target_label",
    "p_top_action_zone",
    "p_bottom_action_zone",
    "p_normal",
    "top_lift",
    "bottom_lift",
    "selected_lift_multiplier",
)


@dataclass(frozen=True)
class FixedSideRegimeGateResult:
    accepted_events: pd.DataFrame
    decision_tape: pd.DataFrame
    metrics: dict[str, Any]


def _causal_regime_features(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["close"].astype(float)
    returns = close.pct_change()
    output = pd.DataFrame({"trade_date": prices["trade_date"]})
    output["source_return_5"] = close / close.shift(5) - 1.0
    output["source_prior_return_5"] = (
        close.shift(5) / close.shift(10) - 1.0
    )
    output["source_volatility_5"] = returns.rolling(5).std(ddof=1)
    output["source_volatility_20"] = returns.rolling(20).std(ddof=1)
    return output


def apply_fixed_side_specific_regime_gate(
    decisions: pd.DataFrame,
    daily: pd.DataFrame,
) -> FixedSideRegimeGateResult:
    """Gate frozen R38 event predictions using source-only OHLC regimes."""

    required = {
        "outer_fold_id",
        "fold_id",
        "issuance_id",
        "feature_anchor_bucket",
        "feature_anchor_bucket_end_utc",
        "target_bucket",
        "target_bucket_start_utc",
        "execution_bucket",
        "first_executable_timestamp",
        "target_label",
        "predicted_label",
        "top_lift",
        "bottom_lift",
        "selected_lift_multiplier",
        *PROBABILITY_COLUMNS,
    }
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(
            "Fixed side regime gate decision columns missing: "
            f"{sorted(missing)}"
        )
    frame = decisions.copy().reset_index(drop=True)
    for column in (
        "feature_anchor_bucket",
        "target_bucket",
        "execution_bucket",
    ):
        frame[column] = pd.to_datetime(
            frame[column],
            errors="coerce",
            utc=True,
        ).dt.date
    for column in (
        "feature_anchor_bucket_end_utc",
        "target_bucket_start_utc",
        "first_executable_timestamp",
    ):
        frame[column] = pd.to_datetime(
            frame[column],
            errors="coerce",
            utc=True,
        )
    if (
        frame[
            [
                "feature_anchor_bucket",
                "target_bucket",
                "execution_bucket",
            ]
        ]
        .isna()
        .any()
        .any()
        or frame[
            [
                "feature_anchor_bucket_end_utc",
                "target_bucket_start_utc",
                "first_executable_timestamp",
            ]
        ]
        .isna()
        .any()
        .any()
    ):
        raise ValueError("Fixed side regime gate has invalid timestamps")
    if (
        frame["target_bucket"].duplicated().any()
        or not frame["execution_bucket"].eq(frame["target_bucket"]).all()
        or not (
            frame["feature_anchor_bucket_end_utc"]
            < frame["target_bucket_start_utc"]
        ).all()
        or not (
            frame["feature_anchor_bucket_end_utc"]
            <= frame["first_executable_timestamp"]
        ).all()
    ):
        raise ValueError("Fixed side regime gate clock is invalid")
    labels = frame["predicted_label"].astype(str)
    if not set(labels).issubset({"TOP", "BOTTOM", "NORMAL"}):
        raise ValueError("Fixed side regime gate has an unsupported prediction")
    probability = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    if (
        not np.isfinite(probability).all()
        or (probability < 0.0).any()
        or (probability > 1.0).any()
        or not np.allclose(probability.sum(axis=1), 1.0, atol=1.0e-10)
    ):
        raise ValueError("Fixed side regime gate probabilities are invalid")

    prices = validate_daily_ohlc(daily).reset_index(drop=True)
    if prices["trade_date"].duplicated().any():
        raise ValueError("Fixed side regime daily schedule is not unique")
    feature_table = _causal_regime_features(prices).set_index("trade_date")
    price_by_date = prices.set_index("trade_date")
    available_dates = set(price_by_date.index)
    missing_source = sorted(
        set(frame["feature_anchor_bucket"]) - available_dates
    )
    missing_target = sorted(set(frame["target_bucket"]) - available_dates)
    if missing_source or missing_target:
        raise ValueError(
            "Fixed side regime gate dates are outside the daily schedule: "
            f"source={missing_source[:5]} target={missing_target[:5]}"
        )

    state = "FULL"
    sequence_id = 0
    events: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    frame = frame.sort_values(
        ["target_bucket", "issuance_id"],
        kind="stable",
    ).reset_index(drop=True)
    for row in frame.itertuples(index=False):
        predicted = str(row.predicted_label)
        source = feature_table.loc[row.feature_anchor_bucket]
        source_return_5 = float(source["source_return_5"])
        source_prior_return_5 = float(
            source["source_prior_return_5"]
        )
        source_volatility_5 = float(source["source_volatility_5"])
        source_volatility_20 = float(source["source_volatility_20"])
        source_features_complete = bool(
            np.isfinite(
                [
                    source_return_5,
                    source_prior_return_5,
                    source_volatility_5,
                    source_volatility_20,
                ]
            ).all()
        )
        top_trend_continuation = bool(
            source_features_complete
            and source_return_5 > 0.0
            and source_return_5 > source_prior_return_5
            and source_volatility_5 > source_volatility_20
        )
        bottom_selloff_confirmed = bool(
            source_features_complete
            and source_return_5 < 0.0
            and source_return_5 < source_prior_return_5
        )
        gate_applicable = predicted in {"TOP", "BOTTOM"}
        gate_passed = bool(
            (predicted == "TOP" and not top_trend_continuation)
            or (predicted == "BOTTOM" and bottom_selloff_confirmed)
        )
        target_price = price_by_date.loc[row.target_bucket]
        execution_open = float(target_price["open"])
        state_before = state
        accepted = False
        if predicted == "NORMAL":
            reason = "NORMAL_NO_ACTION"
        elif not source_features_complete:
            reason = "SOURCE_REGIME_FEATURES_INCOMPLETE"
        elif predicted == "TOP" and top_trend_continuation:
            reason = "TOP_TREND_CONTINUATION_VETO"
        elif predicted == "BOTTOM" and not bottom_selloff_confirmed:
            reason = "BOTTOM_SELLOFF_EVIDENCE_REJECTED"
        elif predicted == "TOP" and state == "FULL":
            reason = "ACCEPT_REGIME_GATED_REDUCE"
            accepted = True
            state = "REDUCED"
        elif predicted == "BOTTOM" and state == "REDUCED":
            reason = "ACCEPT_REGIME_GATED_RESTORE"
            accepted = True
            state = "FULL"
        elif predicted == "TOP":
            reason = "STATE_BLOCKED_ALREADY_REDUCED"
        else:
            reason = "STATE_BLOCKED_ALREADY_FULL"

        event_id = None
        if accepted:
            sequence_id += 1
            event_id = f"R51-REGIME-GATED-ACTION-{sequence_id:04d}"
            events.append(
                {
                    "event_id": event_id,
                    "sequence_id": sequence_id,
                    "event_type": predicted,
                    "signal_date": row.feature_anchor_bucket,
                    "confirmation_date": row.target_bucket,
                    "execution_date": row.target_bucket,
                    "execution_open": execution_open,
                    "fold_id": str(row.fold_id),
                    "issuance_id": str(row.issuance_id),
                    "target_label": str(row.target_label),
                    "p_top_action_zone": float(row.p_top_action_zone),
                    "p_bottom_action_zone": float(
                        row.p_bottom_action_zone
                    ),
                    "p_normal": float(row.p_normal),
                    "top_lift": float(row.top_lift),
                    "bottom_lift": float(row.bottom_lift),
                    "selected_lift_multiplier": float(
                        row.selected_lift_multiplier
                    ),
                }
            )
        audit_rows.append(
            {
                **row._asdict(),
                "regime_gate_id": FIXED_SIDE_REGIME_GATE_ID,
                "source_return_5": source_return_5,
                "source_prior_return_5": source_prior_return_5,
                "source_volatility_5": source_volatility_5,
                "source_volatility_20": source_volatility_20,
                "source_features_complete": source_features_complete,
                "top_trend_continuation": top_trend_continuation,
                "bottom_selloff_confirmed": bottom_selloff_confirmed,
                "gate_applicable": gate_applicable,
                "gate_passed": gate_passed,
                "state_before_regime_gate": state_before,
                "action_accepted": accepted,
                "regime_gate_decision_reason": reason,
                "accepted_event_id": event_id,
                "state_after_regime_gate": state,
            }
        )

    accepted_events = pd.DataFrame.from_records(
        events,
        columns=GATED_EVENT_COLUMNS,
    )
    accepted_types = accepted_events["event_type"].astype(str).tolist()
    if any(left == right for left, right in pairwise(accepted_types)):
        raise AssertionError("Regime-gated actions do not alternate")
    if accepted_events["execution_date"].duplicated().any():
        raise AssertionError(
            "Regime-gated actions collide on an execution bucket"
        )
    decision_tape = pd.DataFrame.from_records(audit_rows)
    reasons = decision_tape["regime_gate_decision_reason"].astype(str)
    metrics = {
        "decision_rows": len(decision_tape),
        "raw_top_predictions": int(labels.eq("TOP").sum()),
        "raw_bottom_predictions": int(labels.eq("BOTTOM").sum()),
        "raw_event_predictions": int(
            labels.isin({"TOP", "BOTTOM"}).sum()
        ),
        "gate_applicable_rows": int(
            decision_tape["gate_applicable"].sum()
        ),
        "gate_passed_rows": int(decision_tape["gate_passed"].sum()),
        "gate_rejected_rows": int(
            (
                decision_tape["gate_applicable"]
                & ~decision_tape["gate_passed"]
            ).sum()
        ),
        "top_trend_continuation_vetoes": int(
            reasons.eq("TOP_TREND_CONTINUATION_VETO").sum()
        ),
        "bottom_selloff_evidence_rejections": int(
            reasons.eq("BOTTOM_SELLOFF_EVIDENCE_REJECTED").sum()
        ),
        "source_feature_incomplete_rejections": int(
            reasons.eq("SOURCE_REGIME_FEATURES_INCOMPLETE").sum()
        ),
        "accepted_actions": len(accepted_events),
        "accepted_top_actions": int(
            accepted_events["event_type"].eq("TOP").sum()
        ),
        "accepted_bottom_actions": int(
            accepted_events["event_type"].eq("BOTTOM").sum()
        ),
        "state_blocked_predictions": int(
            reasons.str.startswith("STATE_BLOCKED").sum()
        ),
        "accepted_actions_alternate": True,
        "execution_buckets_unique": True,
        "final_planned_state": state,
        "source_ohlc_only": True,
        "target_ohlc_used_for_decision": False,
        "future_buckets_used_for_decision": False,
        "target_labels_used_for_decision": False,
        "execution_at_registered_target_open": True,
    }
    return FixedSideRegimeGateResult(
        accepted_events=accepted_events,
        decision_tape=decision_tape,
        metrics=metrics,
    )
