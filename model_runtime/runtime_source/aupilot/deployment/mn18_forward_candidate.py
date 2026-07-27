"""Serializable runtime implementation for the frozen MN18 candidate."""

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

from aupilot.backtest.channel_support_overlay import (
    TOP_RESTORE,
    apply_channel_support_overlay,
)
from aupilot.backtest.top_trend_soft_sizing import (
    apply_top_trend_continuation_soft_sizing,
    attach_registered_top_feature_anchors,
)
from aupilot.core.hashing import sha256_file
from aupilot.core.manifest import write_json_atomic
from aupilot.deployment.m01_graded_release_candidate import (
    ALL_FEATURE_COLUMNS,
    M01NearHeads,
    _direction_priors,
    _fit_binary_model,
    _fit_near,
    _levels,
    _numeric,
    _predict_near,
    _validate_prediction_rows,
    _validate_probability,
    _validate_training_rows,
    _weighted_prior,
)
from aupilot.deployment.mn03_fixed_dual_clock_candidate import (
    MN03H1Heads,
    _predict_mn03_h1,
    _validated_inventory,
)
from aupilot.deployment.task3_release_candidate import (
    REQUIRED_HISTORY_START,
    _validated_completed_history,
    build_online_feature_rows,
    future_utc_bucket_schedule,
)
from aupilot.training.future_slot_features import HORIZON_BUCKETS
from aupilot.training.graded_future_slot import (
    GRADED_LABELS,
    PROBABILITY_COLUMNS,
    group_normalized_weights,
)
from aupilot.training.graded_hurdle_lightgbm import (
    M01_LIGHTGBM_PARAMETERS,
)
from aupilot.training.n_branch_clock_role_policy import (
    DUAL_CLOCK_H1_TOP_H2_BOTTOM,
    n_branch_clock_role_actions,
)
from aupilot.training.n_branch_direct_policy import (
    N_BRANCH_POLICY_HEAD_COLUMNS,
    build_n_branch_boundary_tape,
)
from aupilot.training.n_branch_hurdle_lightgbm import (
    MN02_BASELINE_FEATURE_COLUMNS,
    MN02_GEOMETRY_FEATURE_COLUMNS,
    sqrt_event_exposure_weights,
)
from aupilot.training.n_branch_market_geometry import (
    N_BRANCH_MARKET_GEOMETRY_COMPLETE,
    augment_issuance_with_n_branch_market_geometry,
)
from aupilot.training.n_branch_temporal_shape import (
    MN18_H1_FEATURE_COLUMNS,
    MN18_TEMPORAL_SHAPE_COMPLETE,
    augment_issuance_with_mn18_temporal_shape,
)
from aupilot.training.n_branch_top_false_breakout import (
    MN14_FALSE_BREAKOUT_COMPLETE,
    MN14_H1_FEATURE_COLUMNS,
)

RELEASE_CANDIDATE_ID = "MN18_THREE_TOP_EXPERT_FORWARD_SHADOW_CANDIDATE_V1"
ARTIFACT_VERSION = "aupilot-mn18-three-top-expert-forward-candidate-v1"
EVIDENCE_IDENTITY = (
    "EXPOSED_FULL_DEVELOPMENT_REFIT",
    "PENDING_NEW_FORWARD_SHADOW_EVIDENCE",
    "NOT_HISTORICALLY_INDEPENDENT",
)
COMPONENT_IDS = (
    "PARENT_BASELINE",
    "BOTTOM_GEOMETRY",
    "TOP_FALSE_BREAKOUT",
    "TOP_TEMPORAL_SHAPE",
)
TOP_COMPONENT_IDS = (
    "PARENT_BASELINE",
    "TOP_FALSE_BREAKOUT",
    "TOP_TEMPORAL_SHAPE",
)
COMPONENT_FEATURE_COLUMNS = {
    "PARENT_BASELINE": tuple(MN02_BASELINE_FEATURE_COLUMNS),
    "BOTTOM_GEOMETRY": tuple(MN02_GEOMETRY_FEATURE_COLUMNS),
    "TOP_FALSE_BREAKOUT": tuple(MN14_H1_FEATURE_COLUMNS),
    "TOP_TEMPORAL_SHAPE": tuple(MN18_H1_FEATURE_COLUMNS),
}
COMPONENT_COMPLETENESS_COLUMNS = {
    "PARENT_BASELINE": (),
    "BOTTOM_GEOMETRY": (N_BRANCH_MARKET_GEOMETRY_COMPLETE,),
    "TOP_FALSE_BREAKOUT": (MN14_FALSE_BREAKOUT_COMPLETE,),
    "TOP_TEMPORAL_SHAPE": (
        MN14_FALSE_BREAKOUT_COMPLETE,
        MN18_TEMPORAL_SHAPE_COMPLETE,
    ),
}
TOP_EXPERT_WEIGHT = 1.0 / 3.0
TOP_CONTINUATION_ATTENUATION = 0.5


@dataclass(frozen=True)
class MN18Expert:
    component_id: str
    h1_heads: MN03H1Heads
    raw_probability_thresholds: np.ndarray
    threshold_fit_rows: int


@dataclass(frozen=True)
class MN18ForwardCandidateBundle:
    model_version: str
    experts: dict[str, MN18Expert]
    near_heads: M01NearHeads
    far_prior: np.ndarray
    development_cutoff_utc_exclusive: str
    training_rows: int
    training_issuances: int
    training_anchor_start: str
    training_anchor_end: str
    required_history_start: date | None = REQUIRED_HISTORY_START

    def _predict_component(
        self,
        component_id: str,
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        rows = _validate_prediction_rows(frame)
        expert = self.experts[component_id]
        horizon = pd.to_numeric(rows["horizon_index"], errors="raise")
        h1_mask = horizon.eq(1).to_numpy()
        near_mask = horizon.between(2, 5).to_numpy()
        far_mask = horizon.between(6, HORIZON_BUCKETS).to_numpy()
        probability = np.empty((len(rows), len(GRADED_LABELS)), dtype=float)
        top_prior = np.empty(len(rows), dtype=float)
        bottom_prior = np.empty(len(rows), dtype=float)

        probability[h1_mask] = _predict_mn03_h1(
            expert.h1_heads,
            rows.loc[h1_mask],
        )
        top_prior[h1_mask] = expert.h1_heads.top_action_prior
        bottom_prior[h1_mask] = expert.h1_heads.bottom_action_prior
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
        output["display_class"] = np.asarray(
            GRADED_LABELS,
            dtype=object,
        )[np.argmax(probability, axis=1)]
        output["train_prior_top_action"] = top_prior
        output["train_prior_bottom_action"] = bottom_prior
        output["component_id"] = component_id
        output["model_id"] = f"{RELEASE_CANDIDATE_ID}:{component_id}"
        return output

    def predict_feature_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        component = {
            component_id: self._predict_component(component_id, frame)
            for component_id in COMPONENT_IDS
        }
        parent = component["PARENT_BASELINE"]
        aligned_columns = [
            "issuance_id",
            "feature_anchor_bucket",
            "target_bucket",
            "horizon_index",
        ]
        for component_id in COMPONENT_IDS[1:]:
            if not parent.loc[:, aligned_columns].equals(
                component[component_id].loc[:, aligned_columns]
            ):
                raise AssertionError("MN18 component prediction keys differ")

        probability = np.mean(
            np.stack(
                [
                    component[component_id]
                    .loc[:, PROBABILITY_COLUMNS]
                    .to_numpy(dtype=float)
                    for component_id in TOP_COMPONENT_IDS
                ],
                axis=0,
            ),
            axis=0,
        )
        _validate_probability(probability)
        output = parent.copy()
        for index, column in enumerate(PROBABILITY_COLUMNS):
            output[column] = probability[:, index]
        output["p_top"] = probability[:, 1:4].sum(axis=1)
        output["p_bottom"] = probability[:, 4:7].sum(axis=1)
        output["display_class"] = np.asarray(
            GRADED_LABELS,
            dtype=object,
        )[np.argmax(probability, axis=1)]
        output["model_id"] = RELEASE_CANDIDATE_ID
        output["component_id"] = "FIXED_EQUAL_THREE_TOP_EXPERT_BLEND"
        horizon = pd.to_numeric(output["horizon_index"], errors="raise")
        output["controls_trading"] = horizon.isin((1, 2))
        output["fixed_role_permission"] = "DISPLAY_ONLY"
        output.loc[horizon.eq(1), "fixed_role_permission"] = "TOP_ONLY"
        output.loc[horizon.eq(2), "fixed_role_permission"] = "BOTTOM_ONLY"
        return output

    def component_predictions(
        self,
        frame: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        return {
            component_id: self._predict_component(component_id, frame)
            for component_id in COMPONENT_IDS
        }

    def scheduled_action_requests(
        self,
        *,
        features: pd.DataFrame,
        daily_history: pd.DataFrame,
        block_id: str,
    ) -> list[dict[str, Any]]:
        predictions = self.component_predictions(features)
        geometry_h2 = _forced_boundary_row(
            predictions["BOTTOM_GEOMETRY"],
            horizon=2,
        )
        geometry_tape = build_n_branch_boundary_tape(geometry_h2)
        top_deltas: dict[str, float] = {}
        top_details: dict[str, dict[str, Any]] = {}
        for component_id in TOP_COMPONENT_IDS:
            top = _top_request(
                prediction=predictions[component_id],
                thresholds=self.experts[
                    component_id
                ].raw_probability_thresholds,
                daily_history=daily_history,
                bottom_boundary_tape=geometry_tape,
                block_id=f"{block_id}:{component_id}",
            )
            if top is not None:
                top_deltas[component_id] = float(top["signed_delta"])
                top_details[component_id] = top
            else:
                top_deltas[component_id] = 0.0

        requests: list[dict[str, Any]] = []
        blended_top = float(
            sum(top_deltas.values()) * TOP_EXPERT_WEIGHT
        )
        if abs(blended_top) > 1.0e-12:
            h1 = predictions["PARENT_BASELINE"].loc[
                predictions["PARENT_BASELINE"]["horizon_index"].eq(1)
            ].iloc[0]
            requests.append(
                {
                    "request_id": f"MN18:{block_id}:H1_TOP",
                    "target_bucket": h1["target_bucket"].isoformat(),
                    "horizon_index": 1,
                    "side": "TOP",
                    "requested_delta_pp": blended_top * 100.0,
                    "expert_weight": TOP_EXPERT_WEIGHT,
                    "expert_signed_delta_pp": {
                        key: value * 100.0
                        for key, value in top_deltas.items()
                    },
                    "expert_details": top_details,
                    "execution_status": "SCHEDULED_FOR_TARGET_BUCKET_OPEN",
                }
            )

        bottom = n_branch_clock_role_actions(
            geometry_tape,
            self.experts[
                "BOTTOM_GEOMETRY"
            ].raw_probability_thresholds,
            block_id=f"{block_id}:BOTTOM_GEOMETRY",
            policy_candidate_id=DUAL_CLOCK_H1_TOP_H2_BOTTOM,
        )
        if not bottom.empty:
            row = bottom.iloc[0]
            requests.append(
                {
                    "request_id": f"MN18:{block_id}:H2_BOTTOM",
                    "target_bucket": row["trade_date"].isoformat(),
                    "horizon_index": 2,
                    "side": "BOTTOM",
                    "requested_delta_pp": float(row["signed_delta"]) * 100.0,
                    "chosen_probability": float(row["chosen_probability"]),
                    "chosen_threshold": float(row["chosen_threshold"]),
                    "strength_level": int(row["strength_level"]),
                    "execution_status": (
                        "SCHEDULED_CONDITIONAL_ON_OUTSTANDING_TOP_INVENTORY"
                    ),
                }
            )
        return requests

    def predict_from_history(
        self,
        daily_history: pd.DataFrame,
        *,
        current_gold_weight: float,
        outstanding_top_inventory_pp: float,
        as_of_utc: datetime,
    ) -> dict[str, Any]:
        weight, inventory = _validated_inventory(
            current_gold_weight,
            outstanding_top_inventory_pp,
        )
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
        base = build_online_feature_rows(history, schedule.target_buckets)
        geometry = augment_issuance_with_n_branch_market_geometry(
            history,
            base,
        ).frame
        features = augment_issuance_with_mn18_temporal_shape(
            history,
            geometry,
        ).frame
        prediction = self.predict_feature_rows(features)
        requests = self.scheduled_action_requests(
            features=features,
            daily_history=history,
            block_id=f"FORWARD:{source_bucket.isoformat()}",
        )
        immediate = _immediate_action(
            requests=requests,
            current_gold_weight=weight,
            outstanding_top_inventory_pp=inventory,
            first_target_bucket=schedule.target_buckets[0],
        )
        probability_rows = [
            {
                "horizon_index": int(row.horizon_index),
                "target_bucket": row.target_bucket.isoformat(),
                **{
                    column: float(getattr(row, column))
                    for column in PROBABILITY_COLUMNS
                },
                "p_top": float(row.p_top),
                "p_bottom": float(row.p_bottom),
                "display_class": str(row.display_class),
                "controls_trading": bool(row.controls_trading),
                "fixed_role_permission": str(row.fixed_role_permission),
            }
            for row in prediction.itertuples(index=False)
        ]
        return {
            "project": "AuPilot",
            "model_version": self.model_version,
            "model_id": RELEASE_CANDIDATE_ID,
            "model_status": list(EVIDENCE_IDENTITY),
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
            "action": immediate,
            "scheduled_action_requests": requests,
            "price_outlook": None,
            "warnings": [
                "NOT_HISTORICALLY_INDEPENDENT",
                "PENDING_NEW_FORWARD_SHADOW_EVIDENCE",
                "H1_TOP_AND_H2_BOTTOM_REQUESTS_REQUIRE_TARGET_DATE_LEDGER",
                "H3_TO_H21_DISPLAY_ONLY",
                "UTC_DAILY_BUCKET_NOT_COMEX_SESSION_OR_SETTLEMENT",
                "ADVISORY_ONLY_NO_BROKER_EXECUTION",
            ],
        }


def _fit_h1_heads(
    rows: pd.DataFrame,
    *,
    component_id: str,
) -> MN03H1Heads:
    feature_columns = COMPONENT_FEATURE_COLUMNS[component_id]
    horizon = pd.to_numeric(rows["horizon_index"], errors="raise")
    h1 = rows.loc[horizon.eq(1)].reset_index(drop=True)
    for column in COMPONENT_COMPLETENESS_COLUMNS[component_id]:
        h1 = h1.loc[h1[column].astype(bool)].reset_index(drop=True)
    if h1.empty:
        raise ValueError(f"{component_id} h1 training rows are empty")
    labels = h1["target_label"].astype(str)
    invalid = sorted(set(labels) - set(GRADED_LABELS))
    if invalid:
        raise ValueError(f"{component_id} labels invalid: {invalid}")
    x = _numeric(h1, feature_columns)
    action_weights = sqrt_event_exposure_weights(h1)
    action = labels.ne("NORMAL").to_numpy(dtype=int)
    action_rows = h1.loc[action.astype(bool)].reset_index(drop=True)
    action_labels = action_rows["target_label"].astype(str)
    action_x = _numeric(action_rows, feature_columns)
    event_equal_weights = group_normalized_weights(action_rows)
    top = action_labels.str.startswith("TOP_").to_numpy(dtype=int)
    levels = _levels(action_labels)
    strength_x = np.column_stack([action_x, top.astype(float)])
    direction = np.where(
        labels.str.startswith("TOP_"),
        "TOP",
        np.where(labels.str.startswith("BOTTOM_"), "BOTTOM", "NORMAL"),
    )
    top_prior, bottom_prior = _direction_priors(
        direction,
        action_weights,
    )
    return MN03H1Heads(
        action=_fit_binary_model(
            x,
            action,
            action_weights,
            parameters=M01_LIGHTGBM_PARAMETERS,
        ),
        direction=_fit_binary_model(
            action_x,
            top,
            event_equal_weights,
            parameters=M01_LIGHTGBM_PARAMETERS,
        ),
        strength_l2=_fit_binary_model(
            strength_x,
            (levels >= 2).astype(int),
            event_equal_weights,
            parameters=M01_LIGHTGBM_PARAMETERS,
        ),
        strength_l3=_fit_binary_model(
            strength_x,
            (levels >= 3).astype(int),
            event_equal_weights,
            parameters=M01_LIGHTGBM_PARAMETERS,
        ),
        top_action_prior=top_prior,
        bottom_action_prior=bottom_prior,
        candidate_id=component_id,
        feature_columns=feature_columns,
    )


def fit_mn18_forward_candidate(
    training_rows: pd.DataFrame,
    *,
    raw_probability_thresholds: dict[str, np.ndarray],
    threshold_fit_rows: dict[str, int],
    development_cutoff_utc_exclusive: str,
    model_version: str,
    required_history_start: date | None = REQUIRED_HISTORY_START,
) -> MN18ForwardCandidateBundle:
    rows = _validate_training_rows(training_rows)
    if set(raw_probability_thresholds) != set(COMPONENT_IDS):
        raise ValueError("MN18 threshold component identity differs")
    if set(threshold_fit_rows) != set(COMPONENT_IDS):
        raise ValueError("MN18 threshold row component identity differs")
    experts: dict[str, MN18Expert] = {}
    for component_id in COMPONENT_IDS:
        thresholds = np.asarray(
            raw_probability_thresholds[component_id],
            dtype=float,
        )
        if (
            thresholds.shape != (len(N_BRANCH_POLICY_HEAD_COLUMNS),)
            or not np.isfinite(thresholds).all()
            or (thresholds <= 0.0).any()
            or (thresholds > 1.0 + 1.0e-12).any()
            or int(threshold_fit_rows[component_id]) < 2
        ):
            raise ValueError(f"{component_id} threshold calibration invalid")
        experts[component_id] = MN18Expert(
            component_id=component_id,
            h1_heads=_fit_h1_heads(rows, component_id=component_id),
            raw_probability_thresholds=thresholds.copy(),
            threshold_fit_rows=int(threshold_fit_rows[component_id]),
        )
    horizon = pd.to_numeric(rows["horizon_index"], errors="raise")
    near = rows.loc[horizon.between(1, 5)].reset_index(drop=True)
    anchor = pd.to_datetime(rows["feature_anchor_bucket"], errors="raise")
    return MN18ForwardCandidateBundle(
        model_version=model_version,
        experts=experts,
        near_heads=_fit_near(near),
        far_prior=_weighted_prior(
            rows["target_label"],
            group_normalized_weights(rows),
        ),
        development_cutoff_utc_exclusive=development_cutoff_utc_exclusive,
        training_rows=len(rows),
        training_issuances=int(rows["issuance_id"].nunique()),
        training_anchor_start=anchor.min().date().isoformat(),
        training_anchor_end=anchor.max().date().isoformat(),
        required_history_start=required_history_start,
    )


def _forced_boundary_row(
    prediction: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    row = prediction.loc[
        prediction["horizon_index"].eq(horizon)
    ].copy().reset_index(drop=True)
    if len(row) != 1:
        raise ValueError(f"MN18 prediction lacks unique h{horizon}")
    row["boundary_action_eligible"] = True
    row["execution_bucket"] = row["target_bucket"]
    return row


def _top_request(
    *,
    prediction: pd.DataFrame,
    thresholds: np.ndarray,
    daily_history: pd.DataFrame,
    bottom_boundary_tape: pd.DataFrame,
    block_id: str,
) -> dict[str, Any] | None:
    boundary = build_n_branch_boundary_tape(
        _forced_boundary_row(prediction, horizon=1)
    )
    action = n_branch_clock_role_actions(
        boundary,
        thresholds,
        block_id=block_id,
        policy_candidate_id=DUAL_CLOCK_H1_TOP_H2_BOTTOM,
    )
    if action.empty:
        return None
    action = attach_registered_top_feature_anchors(
        actions=action,
        boundary_tape=boundary,
    )
    action = apply_top_trend_continuation_soft_sizing(
        actions=action,
        daily=daily_history,
        attenuation_factor=TOP_CONTINUATION_ATTENUATION,
    )
    action = apply_channel_support_overlay(
        actions=action,
        daily=daily_history,
        bottom_boundary_tape=bottom_boundary_tape,
        candidate_id=TOP_RESTORE,
    ).actions
    return action.iloc[0].to_dict()


def _immediate_action(
    *,
    requests: list[dict[str, Any]],
    current_gold_weight: float,
    outstanding_top_inventory_pp: float,
    first_target_bucket: date,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "action": "HOLD",
        "reason_code": "NO_QUALIFIED_H1_TOP_REQUEST",
        "current_target_gold_weight": current_gold_weight,
        "recommended_target_gold_weight": current_gold_weight,
        "outstanding_top_inventory_pp_before": (
            outstanding_top_inventory_pp
        ),
        "outstanding_top_inventory_pp_after": (
            outstanding_top_inventory_pp
        ),
        "requested_delta_pp": 0.0,
        "executed_delta_pp": 0.0,
        "automatic_execution": False,
        "requires_actual_databento_bucket_open": True,
    }
    eligible = [
        request
        for request in requests
        if request["horizon_index"] == 1
        and request["side"] == "TOP"
        and request["target_bucket"] == first_target_bucket.isoformat()
    ]
    if not eligible:
        return base
    request = eligible[0]
    requested = float(request["requested_delta_pp"])
    proposed = max(
        0.5,
        current_gold_weight + requested / 100.0,
    )
    executed = (proposed - current_gold_weight) * 100.0
    inventory_after = outstanding_top_inventory_pp - executed
    base.update(
        {
            "action": (
                "REDUCE_GOLD_WEIGHT"
                if executed < -1.0e-12
                else "HOLD"
            ),
            "reason_code": (
                "QUALIFIED_MN18_H1_TOP"
                if executed < -1.0e-12
                else "QUALIFIED_TOP_AT_MINIMUM_WEIGHT"
            ),
            "recommended_target_gold_weight": proposed,
            "outstanding_top_inventory_pp_after": inventory_after,
            "requested_delta_pp": requested,
            "executed_delta_pp": executed,
            "expected_execution_bucket": first_target_bucket.isoformat(),
            "request_id": request["request_id"],
        }
    )
    return base


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _record(path: Path, root: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": _relative(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _native_models(
    bundle: MN18ForwardCandidateBundle,
) -> dict[str, lgb.LGBMClassifier]:
    models: dict[str, lgb.LGBMClassifier] = {}
    for component_id, expert in bundle.experts.items():
        prefix = component_id.lower()
        models[f"{prefix}_action"] = expert.h1_heads.action
        models[f"{prefix}_direction"] = expert.h1_heads.direction
        models[f"{prefix}_strength_l2"] = expert.h1_heads.strength_l2
        models[f"{prefix}_strength_l3"] = expert.h1_heads.strength_l3
    models["near_direction"] = bundle.near_heads.direction
    models["near_strength_l2"] = bundle.near_heads.strength_l2
    models["near_strength_l3"] = bundle.near_heads.strength_l3
    return models


def persist_mn18_forward_candidate(
    bundle: MN18ForwardCandidateBundle,
    destination: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    directory = Path(destination)
    if not directory.is_absolute():
        directory = root_path / directory
    directory = directory.resolve()
    directory.relative_to(root_path)
    directory.mkdir(parents=True, exist_ok=False)

    model_path = directory / "mn18_forward_candidate.joblib"
    joblib.dump(bundle, model_path)
    files = {
        "bundle": _record(model_path, root_path, "SERIALIZED_BUNDLE"),
    }
    for role, model in _native_models(bundle).items():
        path = directory / f"{role}_lightgbm.txt"
        model.booster_.save_model(path)
        files[role] = _record(path, root_path, "LIGHTGBM_NATIVE_MODEL")
    metadata_path = directory / "model_metadata.json"
    write_json_atomic(
        metadata_path,
        {
            "artifact_version": ARTIFACT_VERSION,
            "model_id": RELEASE_CANDIDATE_ID,
            "model_version": bundle.model_version,
            "evidence_identity": list(EVIDENCE_IDENTITY),
            "historical_performance_claim": False,
            "component_ids": list(COMPONENT_IDS),
            "top_component_ids": list(TOP_COMPONENT_IDS),
            "top_expert_weight": TOP_EXPERT_WEIGHT,
            "top_continuation_attenuation": (
                TOP_CONTINUATION_ATTENUATION
            ),
            "routes": {
                "h1": "THREE_TOP_EXPERT_FIXED_EQUAL_BLEND",
                "h2": "BOTTOM_GEOMETRY",
                "h3_h21": "DISPLAY_ONLY",
            },
            "input": "COMPLETE_CANONICAL_UTC_DAILY_OHLC_HISTORY_ONLY",
            "price_outlook": None,
        },
    )
    files["metadata"] = _record(
        metadata_path,
        root_path,
        "MODEL_METADATA",
    )
    manifest_path = directory / "bundle_manifest.json"
    write_json_atomic(
        manifest_path,
        {
            "artifact_version": ARTIFACT_VERSION,
            "model_id": RELEASE_CANDIDATE_ID,
            "model_version": bundle.model_version,
            "files": files,
        },
    )
    return {
        "model_id": RELEASE_CANDIDATE_ID,
        "model_version": bundle.model_version,
        "manifest": _record(
            manifest_path,
            root_path,
            "BUNDLE_MANIFEST",
        ),
        "files": files,
    }


def load_verified_mn18_forward_candidate(
    manifest_path: str | Path,
    *,
    root: str | Path,
) -> MN18ForwardCandidateBundle:
    root_path = Path(root).resolve()
    path = Path(manifest_path)
    if not path.is_absolute():
        path = root_path / path
    path = path.resolve()
    path.relative_to(root_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_version") != ARTIFACT_VERSION
        or manifest.get("model_id") != RELEASE_CANDIDATE_ID
    ):
        raise ValueError("MN18 bundle manifest identity differs")
    for record in manifest["files"].values():
        artifact_path = (root_path / record["path"]).resolve()
        artifact_path.relative_to(root_path)
        if (
            not artifact_path.is_file()
            or artifact_path.stat().st_size != int(record["bytes"])
            or sha256_file(artifact_path) != record["sha256"]
        ):
            raise ValueError(
                f"MN18 bundle artifact differs: {record['path']}"
            )
    bundle_record = manifest["files"]["bundle"]
    bundle = joblib.load(root_path / bundle_record["path"])
    if (
        not isinstance(bundle, MN18ForwardCandidateBundle)
        or bundle.model_version != manifest["model_version"]
        or set(bundle.experts) != set(COMPONENT_IDS)
    ):
        raise ValueError("MN18 serialized bundle identity differs")
    return bundle

