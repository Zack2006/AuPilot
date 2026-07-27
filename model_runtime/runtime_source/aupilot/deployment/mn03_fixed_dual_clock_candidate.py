"""Serializable fixed-role MN03 forward-shadow candidate.

This module packages the already exposed MN03 h1-TOP/h2-BOTTOM hypothesis
without creating a new historical performance claim.  It preserves the
GN01 seven-class probability contract, MN02 h1 weighting, the mixed h1/h2
OOF thresholds, and the BN02 paired-inventory deltas.
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

from aupilot.core.config import resolve_project_path
from aupilot.core.hashing import sha256_file
from aupilot.core.manifest import write_json_atomic
from aupilot.deployment.m01_graded_release_candidate import (
    ALL_FEATURE_COLUMNS,
    M01NearHeads,
    _conditional_strength,
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
    BN02_FROZEN_DELTAS,
    BN02_FROZEN_THRESHOLD_QUANTILES,
    N_BRANCH_POLICY_HEAD_COLUMNS,
    build_n_branch_boundary_tape,
    n_branch_direct_actions,
)
from aupilot.training.n_branch_hurdle_lightgbm import (
    MN02_CANDIDATE_FEATURES,
    sqrt_event_exposure_weights,
)
from aupilot.training.n_branch_market_geometry import (
    N_BRANCH_MARKET_GEOMETRY_COLUMNS,
    N_BRANCH_MARKET_GEOMETRY_COMPLETE,
    augment_issuance_with_n_branch_market_geometry,
)

RELEASE_CANDIDATE_ID = (
    "MN03_GN01_FIXED_H1_TOP_H2_BOTTOM_FORWARD_CANDIDATE_V1"
)
ARTIFACT_VERSION = "aupilot-mn03-fixed-dual-clock-forward-candidate-v1"
EVIDENCE_IDENTITY = (
    "DEVELOPMENT_SELECTED_FIXED_DUAL_CLOCK_ROLE_HYPOTHESIS",
    "PENDING_NEW_FORWARD_SHADOW_EVIDENCE",
    "NOT_HISTORICALLY_INDEPENDENT",
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
class MN03H1Heads:
    action: lgb.LGBMClassifier
    direction: lgb.LGBMClassifier
    strength_l2: lgb.LGBMClassifier
    strength_l3: lgb.LGBMClassifier
    top_action_prior: float
    bottom_action_prior: float
    candidate_id: str
    feature_columns: tuple[str, ...]


@dataclass(frozen=True)
class MN03FixedDualClockCandidateBundle:
    model_version: str
    h1_heads: MN03H1Heads
    near_heads: M01NearHeads
    far_prior: np.ndarray
    raw_probability_thresholds: np.ndarray
    threshold_fit_rows: int
    candidate_selection_reason: str
    development_cutoff_utc_exclusive: str
    training_rows: int
    training_issuances: int
    training_anchor_start: str
    training_anchor_end: str
    required_history_start: date | None = REQUIRED_HISTORY_START

    @property
    def selected_candidate_id(self) -> str:
        return self.h1_heads.candidate_id

    def predict_feature_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = _validate_prediction_rows(frame)
        horizon = pd.to_numeric(rows["horizon_index"], errors="raise")
        h1_mask = horizon.eq(1).to_numpy()
        near_mask = horizon.between(2, 5).to_numpy()
        far_mask = horizon.between(6, HORIZON_BUCKETS).to_numpy()
        probability = np.empty((len(rows), len(GRADED_LABELS)), dtype=float)
        top_prior = np.empty(len(rows), dtype=float)
        bottom_prior = np.empty(len(rows), dtype=float)

        probability[h1_mask] = _predict_mn03_h1(
            self.h1_heads,
            rows.loc[h1_mask],
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
        output["controls_trading"] = output[
            "boundary_action_eligible"
        ].astype(bool) & horizon.isin((1, 2))
        output["fixed_role_permission"] = "DISPLAY_ONLY"
        output.loc[
            output["controls_trading"] & horizon.eq(1),
            "fixed_role_permission",
        ] = "TOP_ONLY"
        output.loc[
            output["controls_trading"] & horizon.eq(2),
            "fixed_role_permission",
        ] = "BOTTOM_ONLY"
        return output

    def action_decision(
        self,
        prediction: pd.DataFrame,
        *,
        current_gold_weight: float,
        outstanding_top_inventory_pp: float,
        block_id: str,
    ) -> dict[str, Any]:
        return fixed_dual_clock_action_decision(
            prediction,
            self.raw_probability_thresholds,
            current_gold_weight=current_gold_weight,
            outstanding_top_inventory_pp=outstanding_top_inventory_pp,
            block_id=block_id,
        )

    def predict_from_history(
        self,
        daily_history: pd.DataFrame,
        *,
        current_gold_weight: float,
        outstanding_top_inventory_pp: float,
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
        base_features = build_online_feature_rows(
            history,
            schedule.target_buckets,
        )
        features = augment_issuance_with_n_branch_market_geometry(
            history,
            base_features,
        ).frame
        prediction = self.predict_feature_rows(features)
        action = self.action_decision(
            prediction,
            current_gold_weight=current_gold_weight,
            outstanding_top_inventory_pp=outstanding_top_inventory_pp,
            block_id=f"FORWARD:{source_bucket.isoformat()}",
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
                    "fixed_role_permission": str(
                        row.fixed_role_permission
                    ),
                }
            )
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
                "market_geometry_is_internal": True,
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
            "probability_model": {
                "selected_h1_candidate": self.selected_candidate_id,
                "candidate_selection_reason": (
                    self.candidate_selection_reason
                ),
                "threshold_fit_rows": int(self.threshold_fit_rows),
                "threshold_head_order": list(
                    N_BRANCH_POLICY_HEAD_COLUMNS
                ),
                "raw_probability_thresholds": (
                    self.raw_probability_thresholds.tolist()
                ),
                "threshold_quantiles": (
                    BN02_FROZEN_THRESHOLD_QUANTILES.tolist()
                ),
            },
            "probability_rows": probability_rows,
            "action": action,
            "price_outlook": None,
            "warnings": [
                "NOT_HISTORICALLY_INDEPENDENT",
                "PENDING_NEW_FORWARD_SHADOW_EVIDENCE",
                "UTC_DAILY_BUCKET_NOT_COMEX_SESSION_OR_SETTLEMENT",
                "FIXED_H1_TOP_H2_BOTTOM_ROLE",
                "H3_TO_H21_DISPLAY_ONLY",
                "ADVISORY_ONLY_NO_BROKER_EXECUTION",
            ],
        }


def _fit_mn03_h1(
    rows: pd.DataFrame,
    *,
    candidate_id: str,
) -> MN03H1Heads:
    if candidate_id not in MN02_CANDIDATE_FEATURES:
        raise ValueError(f"Unknown MN03 h1 candidate: {candidate_id}")
    feature_columns = tuple(MN02_CANDIDATE_FEATURES[candidate_id])
    horizon = pd.to_numeric(rows["horizon_index"], errors="raise")
    h1 = rows.loc[horizon.eq(1)].reset_index(drop=True)
    if any(
        column in N_BRANCH_MARKET_GEOMETRY_COLUMNS
        for column in feature_columns
    ):
        if N_BRANCH_MARKET_GEOMETRY_COMPLETE not in h1:
            raise ValueError("MN03 geometry h1 lacks completeness")
        h1 = h1.loc[
            h1[N_BRANCH_MARKET_GEOMETRY_COMPLETE].astype(bool)
        ].reset_index(drop=True)
    if h1.empty:
        raise ValueError("MN03 h1 training rows are empty")
    labels = h1["target_label"].astype(str)
    invalid = sorted(set(labels) - set(GRADED_LABELS))
    if invalid:
        raise ValueError(f"MN03 h1 labels invalid: {invalid}")
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
        candidate_id=candidate_id,
        feature_columns=feature_columns,
    )


def _predict_mn03_h1(
    heads: MN03H1Heads,
    rows: pd.DataFrame,
) -> np.ndarray:
    x = _numeric(rows, heads.feature_columns)
    action_raw = np.asarray(heads.action.predict_proba(x), dtype=float)
    direction_raw = np.asarray(
        heads.direction.predict_proba(x),
        dtype=float,
    )
    action_index = int(np.flatnonzero(heads.action.classes_ == 1)[0])
    direction_index = int(
        np.flatnonzero(heads.direction.classes_ == 1)[0]
    )
    p_action = action_raw[:, action_index]
    p_top_given_action = direction_raw[:, direction_index]
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


def _validated_inventory(
    current_gold_weight: float,
    outstanding_top_inventory_pp: float,
) -> tuple[float, float]:
    weight = float(current_gold_weight)
    inventory = float(outstanding_top_inventory_pp)
    if (
        not np.isfinite(weight)
        or not np.isfinite(inventory)
        or not 0.5 <= weight <= 1.0
        or not 0.0 <= inventory <= 50.0
    ):
        raise ValueError("MN03 tactical inventory is outside its bounds")
    expected = (1.0 - weight) * 100.0
    if not np.isclose(inventory, expected, atol=1.0e-8):
        raise ValueError(
            "MN03 outstanding TOP inventory does not reconcile to weight"
        )
    return weight, inventory


def fixed_dual_clock_action_decision(
    prediction: pd.DataFrame,
    thresholds: np.ndarray,
    *,
    current_gold_weight: float,
    outstanding_top_inventory_pp: float,
    block_id: str,
) -> dict[str, Any]:
    weight, inventory = _validated_inventory(
        current_gold_weight,
        outstanding_top_inventory_pp,
    )
    boundary = build_n_branch_boundary_tape(prediction)
    if len(boundary) != 1:
        raise ValueError(
            "Online MN03 prediction must have one executable boundary"
        )
    horizon = int(boundary.iloc[0]["horizon_index"])
    base: dict[str, Any] = {
        "action": "HOLD",
        "reason_code": "NO_QUALIFIED_FIXED_ROLE_SIGNAL",
        "boundary_horizon_index": horizon,
        "fixed_role": DUAL_CLOCK_H1_TOP_H2_BOTTOM,
        "role_permission": "TOP_ONLY" if horizon == 1 else "BOTTOM_ONLY",
        "current_target_gold_weight": weight,
        "recommended_target_gold_weight": weight,
        "outstanding_top_inventory_pp_before": inventory,
        "outstanding_top_inventory_pp_after": inventory,
        "requested_delta_pp": 0.0,
        "executed_delta_pp": 0.0,
        "automatic_execution": False,
        "requires_actual_databento_bucket_open": True,
        "repeated_same_side_signal_allowed": True,
        "bottom_requires_outstanding_top_inventory": True,
        "fifo_pairing_required_by_execution_ledger": True,
    }
    direct = n_branch_direct_actions(
        boundary,
        thresholds,
        block_id=block_id,
    )
    if direct.empty:
        return base
    direct_row = direct.iloc[0]
    direct_side = "TOP" if int(direct_row["side_code"]) == -1 else "BOTTOM"
    routed = n_branch_clock_role_actions(
        boundary,
        thresholds,
        block_id=block_id,
        policy_candidate_id=DUAL_CLOCK_H1_TOP_H2_BOTTOM,
    )
    if routed.empty:
        base["reason_code"] = (
            f"FIXED_ROLE_REJECTED_H{horizon}_{direct_side}"
        )
        base["rejected_predicted_side"] = direct_side
        return base
    if len(routed) != 1:
        raise AssertionError("MN03 online router emitted multiple actions")
    action = routed.iloc[0]
    requested = float(action["signed_delta"]) * 100.0
    if requested < 0.0:
        proposed = max(0.5, weight + requested / 100.0)
    else:
        buyback_pp = min(
            requested,
            inventory,
            (1.0 - weight) * 100.0,
        )
        proposed = min(1.0, weight + buyback_pp / 100.0)
    executed = (proposed - weight) * 100.0
    inventory_after = inventory - executed
    if (
        inventory_after < -1.0e-8
        or not np.isclose(
            inventory_after,
            (1.0 - proposed) * 100.0,
            atol=1.0e-8,
        )
    ):
        raise AssertionError("MN03 action broke tactical inventory parity")
    base.update(
        {
            "predicted_side": direct_side,
            "predicted_strength_level": int(
                action["strength_level"]
            ),
            "chosen_probability": float(
                action["chosen_probability"]
            ),
            "chosen_threshold": float(action["chosen_threshold"]),
            "threshold_margin": float(action["threshold_margin"]),
            "requested_delta_pp": requested,
            "executed_delta_pp": executed,
            "recommended_target_gold_weight": proposed,
            "outstanding_top_inventory_pp_after": max(
                inventory_after,
                0.0,
            ),
            "expected_execution_bucket": (
                action["trade_date"].isoformat()
            ),
        }
    )
    if abs(executed) <= 1.0e-12:
        base["reason_code"] = (
            "QUALIFIED_TOP_AT_MINIMUM_WEIGHT"
            if requested < 0.0
            else "QUALIFIED_BOTTOM_WITHOUT_OUTSTANDING_TOP_INVENTORY"
        )
        return base
    base["action"] = (
        "REDUCE_GOLD_WEIGHT"
        if executed < 0.0
        else "INCREASE_GOLD_WEIGHT"
    )
    base["reason_code"] = (
        f"QUALIFIED_H{horizon}_{direct_side}_L"
        f"{int(action['strength_level'])}"
    )
    return base


def fit_mn03_fixed_dual_clock_candidate(
    training_rows: pd.DataFrame,
    *,
    selected_candidate_id: str,
    raw_probability_thresholds: np.ndarray,
    threshold_fit_rows: int,
    candidate_selection_reason: str,
    development_cutoff_utc_exclusive: str,
    model_version: str,
    required_history_start: date | None = REQUIRED_HISTORY_START,
) -> MN03FixedDualClockCandidateBundle:
    rows = _validate_training_rows(training_rows)
    thresholds = np.asarray(raw_probability_thresholds, dtype=float)
    if (
        thresholds.shape != (len(N_BRANCH_POLICY_HEAD_COLUMNS),)
        or not np.isfinite(thresholds).all()
        or (thresholds <= 0.0).any()
        or (thresholds > 1.0 + 1.0e-12).any()
    ):
        raise ValueError("MN03 raw probability thresholds are invalid")
    if int(threshold_fit_rows) < 2:
        raise ValueError("MN03 threshold calibration lacks OOF rows")
    horizon = pd.to_numeric(rows["horizon_index"], errors="raise")
    near = rows.loc[horizon.between(1, 5)].reset_index(drop=True)
    anchor = pd.to_datetime(
        rows["feature_anchor_bucket"],
        errors="raise",
    )
    return MN03FixedDualClockCandidateBundle(
        model_version=model_version,
        h1_heads=_fit_mn03_h1(
            rows,
            candidate_id=selected_candidate_id,
        ),
        near_heads=_fit_near(near),
        far_prior=_weighted_prior(
            rows["target_label"],
            group_normalized_weights(rows),
        ),
        raw_probability_thresholds=thresholds.copy(),
        threshold_fit_rows=int(threshold_fit_rows),
        candidate_selection_reason=str(candidate_selection_reason),
        development_cutoff_utc_exclusive=(
            development_cutoff_utc_exclusive
        ),
        training_rows=len(rows),
        training_issuances=int(rows["issuance_id"].nunique()),
        training_anchor_start=anchor.min().date().isoformat(),
        training_anchor_end=anchor.max().date().isoformat(),
        required_history_start=required_history_start,
    )


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
    bundle: MN03FixedDualClockCandidateBundle,
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


def persist_mn03_fixed_dual_clock_candidate(
    bundle: MN03FixedDualClockCandidateBundle,
    destination: str | Path,
    *,
    root: Path,
) -> dict[str, Any]:
    root = root.resolve()
    target = resolve_project_path(destination, root=root)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite MN03 bundle: {target}")
    target.mkdir(parents=True, exist_ok=False)
    bundle_path = target / "mn03_fixed_dual_clock_candidate.joblib"
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
            "evidence_identity": list(EVIDENCE_IDENTITY),
            "development_cutoff_utc_exclusive": (
                bundle.development_cutoff_utc_exclusive
            ),
            "selected_h1_candidate": bundle.selected_candidate_id,
            "candidate_selection_reason": (
                bundle.candidate_selection_reason
            ),
            "threshold_head_order": list(
                N_BRANCH_POLICY_HEAD_COLUMNS
            ),
            "raw_probability_thresholds": (
                bundle.raw_probability_thresholds.tolist()
            ),
            "threshold_quantiles": (
                BN02_FROZEN_THRESHOLD_QUANTILES.tolist()
            ),
            "threshold_fit_rows": bundle.threshold_fit_rows,
            "training_rows": bundle.training_rows,
            "training_issuances": bundle.training_issuances,
            "routes": {
                "h1_probability": (
                    "MN02_SQRT_EXPOSURE_HURDLE_DIRECTION_ORDINAL"
                ),
                "h2_h5_probability": (
                    "G01_FIXED_SMALL_LIGHTGBM_DIRECTION_ORDINAL"
                ),
                "h6_h21_probability": (
                    "FULL_DATA_EVENT_GROUP_NORMALIZED_SEVEN_CLASS_PRIOR"
                ),
                "action": "FIXED_H1_TOP_H2_BOTTOM",
            },
            "bn02_deltas_percentage_points": (
                BN02_FROZEN_DELTAS * 100.0
            ).tolist(),
            "gold_weight_bounds": [0.5, 1.0],
            "bottom_requires_outstanding_top_inventory": True,
            "pairing": "FIFO_BY_SOLD_WEIGHT_PP",
            "external_model_input": (
                "COMPLETE_DATABENTO_NATIVE_UTC_DAILY_OHLC_HISTORY_ONLY"
            ),
            "technical_indicators_are_internal": True,
            "price_outlook": None,
            "rag_controls_trading": False,
            "historical_performance_claim": False,
            "forward_shadow_required": True,
        },
    )
    files.append(_record(metadata_path, root, "MODEL_METADATA"))
    manifest = {
        "project": "AuPilot",
        "operation": (
            "MN03_FIXED_DUAL_CLOCK_FORWARD_CANDIDATE_ARTIFACT"
        ),
        "artifact_version": ARTIFACT_VERSION,
        "model_id": RELEASE_CANDIDATE_ID,
        "model_version": bundle.model_version,
        "evidence_identity": list(EVIDENCE_IDENTITY),
        "trusted_joblib_requires_hash_verification": True,
        "historical_performance_claim": False,
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


def load_verified_mn03_fixed_dual_clock_candidate(
    manifest_path: str | Path,
    *,
    root: Path,
) -> MN03FixedDualClockCandidateBundle:
    root = root.resolve()
    source = resolve_project_path(manifest_path, root=root)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if (
        manifest.get("project") != "AuPilot"
        or manifest.get("artifact_version") != ARTIFACT_VERSION
        or manifest.get("model_id") != RELEASE_CANDIDATE_ID
        or tuple(manifest.get("evidence_identity", ()))
        != EVIDENCE_IDENTITY
        or manifest.get("historical_performance_claim") is not False
    ):
        raise ValueError("Unsupported MN03 fixed-role manifest")
    bundle_path: Path | None = None
    for record in manifest["files"]:
        path = resolve_project_path(record["path"], root=root)
        if (
            not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(
                "MN03 release artifact verification failed: "
                f"{record['role']}"
            )
        if record["role"] == "VERIFIED_LOCAL_JOBLIB_BUNDLE":
            bundle_path = path
    if bundle_path is None:
        raise ValueError("MN03 release manifest lacks its joblib bundle")
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, MN03FixedDualClockCandidateBundle):
        raise TypeError("Unexpected MN03 release bundle type")
    if (
        bundle.model_version != manifest["model_version"]
        or RELEASE_CANDIDATE_ID != manifest["model_id"]
    ):
        raise RuntimeError("MN03 release bundle identity differs")
    return bundle
