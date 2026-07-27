"""Serializable M01 G01 graded release candidate.

The bundle consumes only completed canonical UTC daily OHLC history.  It
derives all technical features internally, produces seven-class
probabilities for 21 future valid UTC daily buckets, and permits an automatic
B01 recommendation from the strict h1 row only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from aupilot.backtest.graded_signal_pyramid import GradedPyramidPolicy
from aupilot.core.config import resolve_project_path
from aupilot.core.hashing import sha256_file
from aupilot.core.manifest import write_json_atomic
from aupilot.deployment.task3_release_candidate import (
    REQUIRED_HISTORY_START,
    _validated_completed_history,
    build_online_feature_rows,
    future_utc_bucket_schedule,
)
from aupilot.training.future_slot_features import (
    FUTURE_SLOT_FEATURE_COLUMNS,
    HORIZON_BUCKETS,
)
from aupilot.training.future_slot_technical_features import (
    TECHNICAL_EXHAUSTION_FEATURE_COLUMNS,
)
from aupilot.training.graded_future_slot import (
    DIRECTION_LABELS,
    GRADED_LABELS,
    LGBM_PARAMETERS,
    PROBABILITY_COLUMNS,
    group_normalized_weights,
)
from aupilot.training.graded_hurdle_lightgbm import (
    M01_H1_FEATURE_COLUMNS,
    M01_LIGHTGBM_PARAMETERS,
)

RELEASE_CANDIDATE_ID = "M01_G01_H1_GRADED_FULL_DEVELOPMENT_CANDIDATE_V1"
ARTIFACT_VERSION = "aupilot-m01-g01-graded-release-candidate-v1"
ALL_FEATURE_COLUMNS = (
    *FUTURE_SLOT_FEATURE_COLUMNS,
    *TECHNICAL_EXHAUSTION_FEATURE_COLUMNS,
)
NATIVE_MODEL_ROLES = (
    "h1_action",
    "h1_direction",
    "h1_strength_l2",
    "h1_strength_l3",
    "near_direction",
    "near_strength_l2",
    "near_strength_l3",
)


@dataclass(frozen=True)
class M01H1Heads:
    action: lgb.LGBMClassifier
    direction: lgb.LGBMClassifier
    strength_l2: lgb.LGBMClassifier
    strength_l3: lgb.LGBMClassifier
    top_action_prior: float
    bottom_action_prior: float


@dataclass(frozen=True)
class M01NearHeads:
    direction: lgb.LGBMClassifier
    strength_l2: lgb.LGBMClassifier
    strength_l3: lgb.LGBMClassifier
    top_action_prior: float
    bottom_action_prior: float


@dataclass(frozen=True)
class M01GradedReleaseCandidateBundle:
    model_version: str
    h1_heads: M01H1Heads
    near_heads: M01NearHeads
    far_prior: np.ndarray
    selected_lift_threshold: float
    development_cutoff_utc_exclusive: str
    training_rows: int
    training_issuances: int
    training_anchor_start: str
    training_anchor_end: str
    required_history_start: date | None = REQUIRED_HISTORY_START

    def predict_feature_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = _validate_prediction_rows(frame)
        horizon = pd.to_numeric(rows["horizon_index"], errors="raise")
        h1_mask = horizon.eq(1).to_numpy()
        near_mask = horizon.between(2, 5).to_numpy()
        far_mask = horizon.between(6, HORIZON_BUCKETS).to_numpy()
        probability = np.empty((len(rows), len(GRADED_LABELS)), dtype=float)
        top_prior = np.empty(len(rows), dtype=float)
        bottom_prior = np.empty(len(rows), dtype=float)

        probability[h1_mask] = _predict_h1(
            self.h1_heads,
            rows.loc[h1_mask, M01_H1_FEATURE_COLUMNS],
        )
        top_prior[h1_mask] = self.h1_heads.top_action_prior
        bottom_prior[h1_mask] = self.h1_heads.bottom_action_prior

        probability[near_mask] = _predict_near(
            self.near_heads,
            rows.loc[near_mask, ALL_FEATURE_COLUMNS],
        )
        top_prior[near_mask] = self.near_heads.top_action_prior
        bottom_prior[near_mask] = self.near_heads.bottom_action_prior

        probability[far_mask] = np.repeat(
            np.asarray(self.far_prior, dtype=float).reshape(1, -1),
            int(far_mask.sum()),
            axis=0,
        )
        top_prior[far_mask] = self.near_heads.top_action_prior
        bottom_prior[far_mask] = self.near_heads.bottom_action_prior
        _validate_probability(probability)

        output = rows.copy()
        for index, column in enumerate(PROBABILITY_COLUMNS):
            output[column] = probability[:, index]
        output["p_top"] = probability[:, 1:4].sum(axis=1)
        output["p_bottom"] = probability[:, 4:7].sum(axis=1)
        output["display_class"] = np.asarray(
            GRADED_LABELS,
            dtype=object,
        )[np.argmax(probability, axis=1)]
        output["train_prior_top_action"] = top_prior
        output["train_prior_bottom_action"] = bottom_prior
        output["model_id"] = RELEASE_CANDIDATE_ID
        output["controls_trading"] = False
        output.loc[h1_mask, "controls_trading"] = output.loc[
            h1_mask,
            "boundary_action_eligible",
        ].astype(bool)
        return output

    def action_decision(
        self,
        prediction: pd.DataFrame,
        *,
        current_gold_weight: float,
    ) -> dict[str, Any]:
        weight = float(current_gold_weight)
        if not np.isfinite(weight) or not 0.5 <= weight <= 1.0:
            raise ValueError("current_gold_weight must be within [0.5, 1.0]")
        h1 = prediction.loc[
            pd.to_numeric(prediction["horizon_index"], errors="coerce").eq(1)
        ]
        if len(h1) != 1:
            raise ValueError("Prediction must contain exactly one h1 row")
        row = h1.iloc[0]
        base: dict[str, Any] = {
            "action": "HOLD",
            "reason_code": "H1_NOT_FIRST_STRICTLY_LATER_BUCKET",
            "actionable_horizon": 1,
            "h2_or_later_automatic_action_enabled": False,
            "current_target_gold_weight": weight,
            "recommended_target_gold_weight": weight,
            "selected_lift_threshold": float(
                self.selected_lift_threshold
            ),
            "automatic_execution": False,
            "repeated_same_side_signal_allowed": True,
            "requires_actual_databento_bucket_open": True,
        }
        if not bool(row["boundary_action_eligible"]):
            return base

        top_probability = float(row["p_top"])
        bottom_probability = float(row["p_bottom"])
        top_lift = top_probability / self.h1_heads.top_action_prior
        bottom_lift = (
            bottom_probability / self.h1_heads.bottom_action_prior
        )
        choose_top = top_lift >= bottom_lift
        chosen_lift = top_lift if choose_top else bottom_lift
        side = "TOP" if choose_top else "BOTTOM"
        side_columns = (
            ("p_top_l1", "p_top_l2", "p_top_l3")
            if choose_top
            else ("p_bottom_l1", "p_bottom_l2", "p_bottom_l3")
        )
        side_probability = np.asarray(
            [float(row[column]) for column in side_columns],
            dtype=float,
        )
        conditional = side_probability / max(
            float(side_probability.sum()),
            1.0e-15,
        )
        expected_delta_pp = float(
            conditional @ np.asarray([10.0, 20.0, 40.0])
        )
        level = 1 if expected_delta_pp < 15.0 else (
            2 if expected_delta_pp < 30.0 else 3
        )
        signal_label = f"{side}_L{level}"
        base.update(
            {
                "predicted_action_class": signal_label,
                "predicted_side": side,
                "model_probability": (
                    top_probability if choose_top else bottom_probability
                ),
                "top_lift": float(top_lift),
                "bottom_lift": float(bottom_lift),
                "probability_lift": float(chosen_lift),
                "expected_delta_pp": expected_delta_pp,
                "expected_execution_bucket": (
                    row["execution_bucket"].isoformat()
                    if hasattr(row["execution_bucket"], "isoformat")
                    else str(row["execution_bucket"])
                ),
                "reason_code": "H1_BELOW_REGISTERED_LIFT_THRESHOLD",
            }
        )
        if chosen_lift < self.selected_lift_threshold:
            return base

        policy = GradedPyramidPolicy()
        next_weight = float(policy.next_weight(weight, signal_label))
        base["recommended_target_gold_weight"] = next_weight
        if np.isclose(next_weight, weight, rtol=0.0, atol=1.0e-12):
            base["reason_code"] = "QUALIFIED_H1_SIGNAL_AT_POSITION_BOUND"
            return base
        base["action"] = (
            "REDUCE_GOLD_WEIGHT" if side == "TOP" else "INCREASE_GOLD_WEIGHT"
        )
        base["reason_code"] = f"QUALIFIED_H1_{signal_label}"
        return base

    def predict_from_history(
        self,
        daily_history: pd.DataFrame,
        *,
        current_gold_weight: float,
        as_of_utc: datetime,
    ) -> dict[str, Any]:
        history = _validated_completed_history(
            daily_history,
            as_of_utc=as_of_utc,
            required_history_start=self.required_history_start,
        )
        source_bucket = history.iloc[-1]["trade_date"]
        schedule = future_utc_bucket_schedule(
            source_bucket,
            horizon=HORIZON_BUCKETS,
        )
        features = build_online_feature_rows(
            history,
            schedule.target_buckets,
        )
        prediction = self.predict_feature_rows(features)
        action = self.action_decision(
            prediction,
            current_gold_weight=current_gold_weight,
        )
        probability_rows: list[dict[str, Any]] = []
        for row in prediction.itertuples(index=False):
            probability_rows.append(
                {
                    "horizon_index": int(row.horizon_index),
                    "target_bucket": row.target_bucket.isoformat(),
                    "p_normal": float(row.p_normal),
                    "p_top_l1": float(row.p_top_l1),
                    "p_top_l2": float(row.p_top_l2),
                    "p_top_l3": float(row.p_top_l3),
                    "p_bottom_l1": float(row.p_bottom_l1),
                    "p_bottom_l2": float(row.p_bottom_l2),
                    "p_bottom_l3": float(row.p_bottom_l3),
                    "p_top": float(row.p_top),
                    "p_bottom": float(row.p_bottom),
                    "display_class": str(row.display_class),
                    "boundary_action_eligible": bool(
                        row.boundary_action_eligible
                    ),
                    "controls_trading": bool(row.controls_trading),
                }
            )
        return {
            "project": "AuPilot",
            "model_version": self.model_version,
            "model_id": RELEASE_CANDIDATE_ID,
            "model_status": (
                "DEVELOPMENT_CANDIDATE_REQUIRES_FORWARD_SHADOW"
            ),
            "as_of_utc": as_of_utc.astimezone(UTC).isoformat(),
            "input_contract": {
                "provider": "Databento",
                "dataset": "GLBX.MDP3",
                "symbol": "GC.v.0",
                "stype_in": "continuous",
                "schema": "ohlcv-1d",
                "bucket": "UTC_00_00_TO_24_00_ELECTRONIC_TRADES",
                "external_model_input": (
                    "COMPLETE_CANONICAL_UTC_DAILY_OHLC_HISTORY_ONLY"
                ),
                "technical_indicators_are_internal": True,
                "intraday_inputs_used": False,
                "rag_or_macro_inputs_used": False,
            },
            "history": {
                "rows": len(history),
                "start_bucket": history.iloc[0]["trade_date"].isoformat(),
                "source_bucket": source_bucket.isoformat(),
                "source_bucket_complete": True,
            },
            "forecast_contract": {
                "rows": HORIZON_BUCKETS,
                "semantics": "FUTURE_VALID_CANONICAL_UTC_DAILY_BUCKETS",
                "not_comex_sessions": True,
                "calendar_adapter": schedule.audit,
            },
            "probability_rows": probability_rows,
            "action": action,
            "price_outlook": None,
            "warnings": [
                "DEVELOPMENT_EVIDENCE_NOT_INDEPENDENT_FINAL",
                "FORWARD_SHADOW_VALIDATION_REQUIRED",
                "FINAL_HOLDOUT_NOT_USED",
                "UTC_DAILY_BUCKET_NOT_COMEX_SESSION_OR_SETTLEMENT",
                "AUTOMATIC_ACTION_STRICTLY_H1_ONLY",
                "H2_TO_H21_DISPLAY_ONLY",
                "ADVISORY_ONLY_NO_BROKER_EXECUTION",
            ],
        }


def _numeric(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
) -> np.ndarray:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"M01 release features missing: {sorted(missing)}")
    values = frame.loc[:, columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("M01 release features are non-finite")
    return values


def _labels(frame: pd.DataFrame) -> pd.Series:
    labels = frame["target_label"].astype(str)
    invalid = sorted(set(labels) - set(GRADED_LABELS))
    if invalid:
        raise ValueError(f"M01 release labels invalid: {invalid}")
    return labels


def _fit_binary_model(
    x: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    parameters: dict[str, Any],
) -> lgb.LGBMClassifier:
    values = np.asarray(target, dtype=int)
    if set(np.unique(values)) != {0, 1}:
        raise ValueError("M01 release binary head lacks a class")
    model = lgb.LGBMClassifier(**parameters, objective="binary")
    model.fit(x, values, sample_weight=weights)
    return model


def _positive_probability(
    model: lgb.LGBMClassifier,
    x: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    index = int(np.flatnonzero(model.classes_ == 1)[0])
    return raw[:, index]


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


def _conditional_strength(
    l2_model: lgb.LGBMClassifier,
    l3_model: lgb.LGBMClassifier,
    x: np.ndarray,
    *,
    side_top: float,
) -> np.ndarray:
    side = np.full((len(x), 1), side_top, dtype=float)
    values = np.column_stack([x, side])
    q2 = _positive_probability(l2_model, values)
    q3 = _positive_probability(l3_model, values)
    q2, q3 = _project_ordinal(q2, q3)
    return np.column_stack([1.0 - q2, q2 - q3, q3])


def _direction_probability(
    model: lgb.LGBMClassifier,
    x: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(model.predict_proba(x), dtype=float)
    return np.column_stack(
        [
            raw[:, int(np.flatnonzero(model.classes_ == label)[0])]
            for label in DIRECTION_LABELS
        ]
    )


def _predict_h1(
    heads: M01H1Heads,
    features: pd.DataFrame,
) -> np.ndarray:
    x = _numeric(features, M01_H1_FEATURE_COLUMNS)
    p_action = _positive_probability(heads.action, x)
    p_top_given_action = _positive_probability(heads.direction, x)
    top_strength = _conditional_strength(
        heads.strength_l2,
        heads.strength_l3,
        x,
        side_top=1.0,
    )
    bottom_strength = _conditional_strength(
        heads.strength_l2,
        heads.strength_l3,
        x,
        side_top=0.0,
    )
    p_top = p_action * p_top_given_action
    p_bottom = p_action * (1.0 - p_top_given_action)
    return np.column_stack(
        [
            1.0 - p_action,
            p_top[:, None] * top_strength,
            p_bottom[:, None] * bottom_strength,
        ]
    )


def _predict_near(
    heads: M01NearHeads,
    features: pd.DataFrame,
) -> np.ndarray:
    x = _numeric(features, ALL_FEATURE_COLUMNS)
    direction = _direction_probability(heads.direction, x)
    top_strength = _conditional_strength(
        heads.strength_l2,
        heads.strength_l3,
        x,
        side_top=1.0,
    )
    bottom_strength = _conditional_strength(
        heads.strength_l2,
        heads.strength_l3,
        x,
        side_top=0.0,
    )
    return np.column_stack(
        [
            direction[:, 2],
            direction[:, [0]] * top_strength,
            direction[:, [1]] * bottom_strength,
        ]
    )


def _direction(labels: pd.Series) -> np.ndarray:
    values = labels.astype(str)
    return np.where(
        values.str.startswith("TOP_"),
        "TOP",
        np.where(values.str.startswith("BOTTOM_"), "BOTTOM", "NORMAL"),
    )


def _levels(labels: pd.Series) -> np.ndarray:
    return (
        labels.astype(str)
        .str.extract(r"_L([123])$", expand=False)
        .fillna("0")
        .astype(int)
        .to_numpy()
    )


def _weighted_prior(
    labels: pd.Series,
    weights: np.ndarray,
) -> np.ndarray:
    values = labels.astype(str).to_numpy()
    prior = np.asarray(
        [weights[values == label].sum() for label in GRADED_LABELS],
        dtype=float,
    )
    prior /= prior.sum()
    if not np.isfinite(prior).all() or (prior <= 0.0).any():
        raise ValueError("M01 release far prior lacks a class")
    return prior


def _direction_priors(
    direction: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    total = float(weights.sum())
    top = float(weights[direction == "TOP"].sum() / total)
    bottom = float(weights[direction == "BOTTOM"].sum() / total)
    if top <= 0.0 or bottom <= 0.0:
        raise ValueError("M01 release action prior lacks a direction")
    return top, bottom


def _fit_h1(rows: pd.DataFrame) -> M01H1Heads:
    labels = _labels(rows)
    x = _numeric(rows, M01_H1_FEATURE_COLUMNS)
    weights = group_normalized_weights(rows)
    action = labels.ne("NORMAL").to_numpy(dtype=int)
    action_rows = rows.loc[action.astype(bool)].reset_index(drop=True)
    action_labels = action_rows["target_label"].astype(str)
    action_x = _numeric(action_rows, M01_H1_FEATURE_COLUMNS)
    action_weights = group_normalized_weights(action_rows)
    top = action_labels.str.startswith("TOP_").to_numpy(dtype=int)
    levels = _levels(action_labels)
    strength_x = np.column_stack([action_x, top.astype(float)])
    direction = _direction(labels)
    top_prior, bottom_prior = _direction_priors(direction, weights)
    return M01H1Heads(
        action=_fit_binary_model(
            x,
            action,
            weights,
            parameters=M01_LIGHTGBM_PARAMETERS,
        ),
        direction=_fit_binary_model(
            action_x,
            top,
            action_weights,
            parameters=M01_LIGHTGBM_PARAMETERS,
        ),
        strength_l2=_fit_binary_model(
            strength_x,
            (levels >= 2).astype(int),
            action_weights,
            parameters=M01_LIGHTGBM_PARAMETERS,
        ),
        strength_l3=_fit_binary_model(
            strength_x,
            (levels >= 3).astype(int),
            action_weights,
            parameters=M01_LIGHTGBM_PARAMETERS,
        ),
        top_action_prior=top_prior,
        bottom_action_prior=bottom_prior,
    )


def _fit_near(rows: pd.DataFrame) -> M01NearHeads:
    labels = _labels(rows)
    x = _numeric(rows, ALL_FEATURE_COLUMNS)
    weights = group_normalized_weights(rows)
    direction = _direction(labels)
    if set(direction) != set(DIRECTION_LABELS):
        raise ValueError("M01 release near direction lacks a class")
    parameters = {
        **LGBM_PARAMETERS,
        "objective": "multiclass",
        "num_class": len(DIRECTION_LABELS),
    }
    direction_model = lgb.LGBMClassifier(**parameters)
    direction_model.fit(x, direction, sample_weight=weights)
    action = direction != "NORMAL"
    action_rows = rows.loc[action].reset_index(drop=True)
    action_labels = action_rows["target_label"].astype(str)
    action_x = _numeric(action_rows, ALL_FEATURE_COLUMNS)
    action_weights = group_normalized_weights(action_rows)
    side_top = action_labels.str.startswith("TOP_").to_numpy(dtype=float)
    strength_x = np.column_stack([action_x, side_top])
    levels = _levels(action_labels)
    top_prior, bottom_prior = _direction_priors(direction, weights)
    return M01NearHeads(
        direction=direction_model,
        strength_l2=_fit_binary_model(
            strength_x,
            (levels >= 2).astype(int),
            action_weights,
            parameters=LGBM_PARAMETERS,
        ),
        strength_l3=_fit_binary_model(
            strength_x,
            (levels >= 3).astype(int),
            action_weights,
            parameters=LGBM_PARAMETERS,
        ),
        top_action_prior=top_prior,
        bottom_action_prior=bottom_prior,
    )


def _validate_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "issuance_id",
        "feature_anchor_bucket",
        "horizon_index",
        "target_event_group_id",
        "target_label",
        *ALL_FEATURE_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"M01 release training rows missing: {sorted(missing)}")
    output = frame.copy().reset_index(drop=True)
    horizon = pd.to_numeric(output["horizon_index"], errors="coerce")
    if (
        horizon.isna().any()
        or not horizon.between(1, HORIZON_BUCKETS).all()
        or not output.groupby("issuance_id", sort=False).size().eq(
            HORIZON_BUCKETS
        ).all()
        or set(output["target_label"].astype(str)) != set(GRADED_LABELS)
    ):
        raise ValueError("M01 release training identity is invalid")
    _numeric(output, ALL_FEATURE_COLUMNS)
    return output


def _validate_prediction_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "horizon_index",
        "boundary_action_eligible",
        *ALL_FEATURE_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"M01 release prediction rows missing: {sorted(missing)}"
        )
    output = frame.copy().reset_index(drop=True)
    horizon = pd.to_numeric(output["horizon_index"], errors="coerce")
    if (
        len(output) != HORIZON_BUCKETS
        or horizon.tolist() != list(range(1, HORIZON_BUCKETS + 1))
    ):
        raise ValueError("M01 release prediction requires ordered h1-h21")
    _numeric(output, ALL_FEATURE_COLUMNS)
    return output


def _validate_probability(probability: np.ndarray) -> None:
    if (
        probability.ndim != 2
        or probability.shape[1] != len(GRADED_LABELS)
        or not np.isfinite(probability).all()
        or (probability < -1.0e-12).any()
        or (probability > 1.0 + 1.0e-12).any()
        or not np.allclose(
            probability.sum(axis=1),
            1.0,
            rtol=0.0,
            atol=1.0e-10,
        )
    ):
        raise ValueError("M01 release probabilities are invalid")


def fit_m01_graded_release_candidate(
    training_rows: pd.DataFrame,
    *,
    selected_lift_threshold: float,
    development_cutoff_utc_exclusive: str,
    model_version: str,
    required_history_start: date | None = REQUIRED_HISTORY_START,
) -> M01GradedReleaseCandidateBundle:
    rows = _validate_training_rows(training_rows)
    threshold = float(selected_lift_threshold)
    if not np.isfinite(threshold) or threshold < 1.0:
        raise ValueError("A finite release lift threshold >= 1 is required")
    horizon = pd.to_numeric(rows["horizon_index"], errors="raise")
    h1 = rows.loc[horizon.eq(1)].reset_index(drop=True)
    near = rows.loc[horizon.between(1, 5)].reset_index(drop=True)
    anchor = pd.to_datetime(
        rows["feature_anchor_bucket"],
        errors="raise",
    )
    return M01GradedReleaseCandidateBundle(
        model_version=model_version,
        h1_heads=_fit_h1(h1),
        near_heads=_fit_near(near),
        far_prior=_weighted_prior(
            rows["target_label"],
            group_normalized_weights(rows),
        ),
        selected_lift_threshold=threshold,
        development_cutoff_utc_exclusive=development_cutoff_utc_exclusive,
        training_rows=len(rows),
        training_issuances=int(rows["issuance_id"].nunique()),
        training_anchor_start=anchor.min().date().isoformat(),
        training_anchor_end=anchor.max().date().isoformat(),
        required_history_start=required_history_start,
    )


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _record(path: Path, root: Path, role: str) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": role,
        "path": _relative(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    return record


def _native_models(
    bundle: M01GradedReleaseCandidateBundle,
) -> dict[str, lgb.LGBMClassifier]:
    return {
        "h1_action": bundle.h1_heads.action,
        "h1_direction": bundle.h1_heads.direction,
        "h1_strength_l2": bundle.h1_heads.strength_l2,
        "h1_strength_l3": bundle.h1_heads.strength_l3,
        "near_direction": bundle.near_heads.direction,
        "near_strength_l2": bundle.near_heads.strength_l2,
        "near_strength_l3": bundle.near_heads.strength_l3,
    }


def persist_m01_graded_release_candidate(
    bundle: M01GradedReleaseCandidateBundle,
    destination: str | Path,
    *,
    root: Path,
    evidence_identity: str = (
        "FULL_DEVELOPMENT_FIT_NOT_NEW_OOS_OR_FINAL_EVIDENCE"
    ),
    final_holdout_used: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    if not isinstance(evidence_identity, str) or not evidence_identity.strip():
        raise ValueError("A non-empty M01 evidence identity is required")
    if not isinstance(final_holdout_used, bool):
        raise TypeError("final_holdout_used must be boolean")
    target = resolve_project_path(destination, root=root)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite M01 bundle: {target}")
    target.mkdir(parents=True, exist_ok=False)
    bundle_path = target / "m01_graded_release_candidate.joblib"
    metadata_path = target / "model_metadata.json"
    joblib.dump(bundle, bundle_path, compress=3)
    files = [_record(bundle_path, root, "VERIFIED_LOCAL_JOBLIB_BUNDLE")]
    for role, model in _native_models(bundle).items():
        path = target / f"{role}_lightgbm.txt"
        model.booster_.save_model(str(path))
        files.append(_record(path, root, f"NATIVE_{role.upper()}"))
    write_json_atomic(
        metadata_path,
        {
            "project": "AuPilot",
            "artifact_version": ARTIFACT_VERSION,
            "model_id": RELEASE_CANDIDATE_ID,
            "model_version": bundle.model_version,
            "evidence_identity": evidence_identity,
            "development_cutoff_utc_exclusive": (
                bundle.development_cutoff_utc_exclusive
            ),
            "selected_lift_threshold": bundle.selected_lift_threshold,
            "training_rows": bundle.training_rows,
            "training_issuances": bundle.training_issuances,
            "far_prior": bundle.far_prior.tolist(),
            "routes": {
                "h1": "M01_HURDLE_DIRECTION_ORDINAL",
                "h2_h5": "G05_FIXED_SMALL_LIGHTGBM_ORDINAL",
                "h6_h21": "FULL_DEVELOPMENT_EVENT_GROUP_PRIOR",
            },
            "external_model_input": (
                "COMPLETE_DATABENTO_NATIVE_UTC_DAILY_OHLC_HISTORY_ONLY"
            ),
            "automatic_action_horizon": 1,
            "h2_or_later_controls_trading": False,
            "technical_indicators_are_internal": True,
            "repeated_same_side_signal_allowed": True,
            "b01_weight_bounds": [0.5, 1.0],
            "b01_delta_percentage_points": [10, 20, 40],
            "price_outlook": None,
            "rag_controls_trading": False,
            "final_holdout_used": final_holdout_used,
            "forward_shadow_required": True,
        },
    )
    files.append(_record(metadata_path, root, "MODEL_METADATA"))
    manifest = {
        "project": "AuPilot",
        "operation": "M01_FULL_DEVELOPMENT_RELEASE_CANDIDATE_ARTIFACT",
        "artifact_version": ARTIFACT_VERSION,
        "model_id": RELEASE_CANDIDATE_ID,
        "model_version": bundle.model_version,
        "evidence_identity": evidence_identity,
        "final_holdout_used": final_holdout_used,
        "trusted_joblib_requires_hash_verification": True,
        "files": files,
    }
    manifest_path = target / "bundle_manifest.json"
    write_json_atomic(manifest_path, manifest)
    return {
        **manifest,
        "manifest": _record(
            manifest_path,
            root,
            "BUNDLE_MANIFEST",
        ),
    }


def load_verified_m01_graded_release_candidate(
    manifest_path: str | Path,
    *,
    root: Path,
) -> M01GradedReleaseCandidateBundle:
    root = root.resolve()
    source = resolve_project_path(manifest_path, root=root)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if (
        manifest.get("project") != "AuPilot"
        or manifest.get("artifact_version") != ARTIFACT_VERSION
        or manifest.get("model_id") != RELEASE_CANDIDATE_ID
    ):
        raise ValueError("Unsupported M01 release manifest")
    bundle_path: Path | None = None
    for record in manifest["files"]:
        path = resolve_project_path(record["path"], root=root)
        if (
            path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(
                f"M01 release artifact verification failed: "
                f"{record['role']}"
            )
        if record["role"] == "VERIFIED_LOCAL_JOBLIB_BUNDLE":
            bundle_path = path
    if bundle_path is None:
        raise ValueError("M01 release manifest lacks its joblib bundle")
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, M01GradedReleaseCandidateBundle):
        raise TypeError("Unexpected M01 release bundle type")
    if (
        bundle.model_version != manifest["model_version"]
        or RELEASE_CANDIDATE_ID != manifest["model_id"]
    ):
        raise RuntimeError("M01 release identity differs from manifest")
    return bundle
