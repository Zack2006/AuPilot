from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from importlib.metadata import version
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import exchange_calendars as xcals
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from aupilot.backtest.pivot_baseline import validate_daily_ohlc
from aupilot.core.config import resolve_project_path
from aupilot.core.hashing import sha256_file
from aupilot.core.manifest import write_json_atomic
from aupilot.training.future_slot_features import (
    FUTURE_SLOT_FEATURE_COLUMNS,
    HORIZON_BUCKETS,
    build_future_slot_feature_table,
)
from aupilot.training.future_slot_lightgbm import (
    FROZEN_PARAMETERS,
)
from aupilot.training.future_slot_logistic import (
    CLASS_LABELS,
    PROBABILITY_COLUMNS,
    group_normalized_weights,
)
from aupilot.training.future_slot_technical_features import (
    TECHNICAL_EXHAUSTION_FEATURE_COLUMNS,
    build_technical_exhaustion_feature_table,
)
from aupilot.training.probability_decision import (
    threshold_event_lift_labels,
)

RELEASE_CANDIDATE_ID = "TASK3_UTC21_R33_R38_FULL_DEVELOPMENT_CANDIDATE_V1"
ARTIFACT_VERSION = "aupilot-task3-utc21-release-candidate-v1"
CALENDAR_ADAPTER_ID = "EXCHANGE_CALENDARS_COMEX_TO_UTC_BUCKETS_V1"
REQUIRED_HISTORY_START = date(2010, 6, 7)
R29_FEATURE_COLUMNS = FUTURE_SLOT_FEATURE_COLUMNS
R30_FEATURE_COLUMNS = (
    *FUTURE_SLOT_FEATURE_COLUMNS,
    *TECHNICAL_EXHAUSTION_FEATURE_COLUMNS,
)
R31_H1_FEATURE_COLUMNS = FUTURE_SLOT_FEATURE_COLUMNS
TacticalState = Literal["FULL", "REDUCED"]
TARGET_WEIGHT = {"FULL": 1.0, "REDUCED": 0.5}


@dataclass(frozen=True)
class CalendarScheduleResult:
    target_buckets: tuple[date, ...]
    audit: dict[str, Any]


@dataclass(frozen=True)
class Task3ReleaseCandidateBundle:
    model_version: str
    h1_model: lgb.LGBMClassifier
    h2_model: lgb.LGBMClassifier
    h3_h5_model: lgb.LGBMClassifier
    far_prior: np.ndarray
    action_prior: np.ndarray
    selected_lift_multiplier: float
    development_cutoff_utc_exclusive: str
    training_rows: int
    training_issuances: int
    training_anchor_start: str
    training_anchor_end: str
    required_history_start: date | None = REQUIRED_HISTORY_START

    def predict_feature_rows(self, frame: pd.DataFrame) -> pd.DataFrame:
        rows = _validate_prediction_rows(frame)
        probability = np.empty((len(rows), len(CLASS_LABELS)), dtype=float)
        horizon = pd.to_numeric(rows["horizon_index"], errors="raise")
        masks = {
            "h1": horizon.eq(1).to_numpy(),
            "h2": horizon.eq(2).to_numpy(),
            "h3_h5": horizon.between(3, 5).to_numpy(),
            "far": horizon.between(6, 21).to_numpy(),
        }
        probability[masks["h1"]] = _ordered_probability(
            self.h1_model,
            rows.loc[masks["h1"], R31_H1_FEATURE_COLUMNS],
        )
        probability[masks["h2"]] = _ordered_probability(
            self.h2_model,
            rows.loc[masks["h2"], R30_FEATURE_COLUMNS],
        )
        probability[masks["h3_h5"]] = _ordered_probability(
            self.h3_h5_model,
            rows.loc[masks["h3_h5"], R29_FEATURE_COLUMNS],
        )
        probability[masks["far"]] = np.repeat(
            np.asarray(self.far_prior, dtype=float).reshape(1, -1),
            int(masks["far"].sum()),
            axis=0,
        )
        _validate_probability(probability)
        output = rows.copy()
        for index, column in enumerate(PROBABILITY_COLUMNS):
            output[column] = probability[:, index]
        output["display_class"] = np.asarray(
            CLASS_LABELS,
            dtype=object,
        )[np.argmax(probability, axis=1)]
        output["model_id"] = RELEASE_CANDIDATE_ID
        return output

    def predict_from_history(
        self,
        daily_history: pd.DataFrame,
        *,
        tactical_state: TacticalState,
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
            tactical_state=tactical_state,
        )
        rows = []
        for row in prediction.itertuples(index=False):
            rows.append(
                {
                    "horizon_index": int(row.horizon_index),
                    "target_bucket": row.target_bucket.isoformat(),
                    "p_top_action_zone": float(row.p_top_action_zone),
                    "p_bottom_action_zone": float(
                        row.p_bottom_action_zone
                    ),
                    "p_normal": float(row.p_normal),
                    "display_class": str(row.display_class),
                    "boundary_action_eligible": bool(
                        row.boundary_action_eligible
                    ),
                    "controls_trading": False,
                }
            )
        return {
            "project": "AuPilot",
            "model_version": self.model_version,
            "model_id": RELEASE_CANDIDATE_ID,
            "model_status": (
                "EXPOSED_DEVELOPMENT_CANDIDATE_REQUIRES_FORWARD_SHADOW"
            ),
            "as_of_utc": as_of_utc.astimezone(UTC).isoformat(),
            "input_contract": {
                "provider": "Databento",
                "dataset": "GLBX.MDP3",
                "symbol": "GC.v.0",
                "stype_in": "continuous",
                "schema": "ohlcv-1d",
                "bucket": "UTC_00_00_TO_24_00_ELECTRONIC_TRADES",
                "technical_indicators_are_internal": True,
                "intraday_inputs_used": False,
                "rag_or_macro_inputs_used": False,
            },
            "history": {
                "rows": len(history),
                "start_bucket": history.iloc[0][
                    "trade_date"
                ].isoformat(),
                "source_bucket": source_bucket.isoformat(),
                "source_bucket_complete": True,
            },
            "forecast_contract": {
                "rows": HORIZON_BUCKETS,
                "semantics": "FUTURE_VALID_CANONICAL_UTC_DAILY_BUCKETS",
                "not_comex_sessions": True,
                "calendar_adapter": schedule.audit,
            },
            "probability_rows": rows,
            "action": action,
            "warnings": [
                "DEVELOPMENT_EVIDENCE_NOT_INDEPENDENT_FINAL",
                "FORWARD_SHADOW_VALIDATION_REQUIRED",
                "UTC_DAILY_BUCKET_NOT_COMEX_SESSION_OR_SETTLEMENT",
                (
                    "CALENDAR_DATES_ARE_EXPECTED_BUCKETS; EXECUTION_MUST_WAIT_"
                    "FOR_AN_ACTUAL_DATABENTO_VALID_BUCKET"
                ),
                "ADVISORY_ONLY_NO_BROKER_EXECUTION",
            ],
        }

    def action_decision(
        self,
        prediction: pd.DataFrame,
        *,
        tactical_state: TacticalState,
    ) -> dict[str, Any]:
        if tactical_state not in TARGET_WEIGHT:
            raise ValueError("tactical_state must be FULL or REDUCED")
        h1 = prediction.loc[
            pd.to_numeric(
                prediction["horizon_index"],
                errors="coerce",
            ).eq(1)
        ]
        if len(h1) != 1:
            raise ValueError("Prediction must contain exactly one h1 row")
        row = h1.iloc[0]
        base = {
            "action": "HOLD",
            "reason_code": "H1_NOT_FIRST_STRICTLY_LATER_BUCKET",
            "tactical_state_before": tactical_state,
            "tactical_state_after_fill": tactical_state,
            "current_target_gold_weight": TARGET_WEIGHT[tactical_state],
            "recommended_target_gold_weight": TARGET_WEIGHT[
                tactical_state
            ],
            "selected_lift_multiplier": float(
                self.selected_lift_multiplier
            ),
            "actionable_horizon": 1,
            "h2_automatic_action_enabled": False,
            "automatic_execution": False,
            "requires_actual_databento_bucket_open": True,
        }
        if not bool(row["boundary_action_eligible"]):
            return base
        lift_frame = pd.DataFrame(
            {
                "top_lift": [
                    float(row["p_top_action_zone"])
                    / float(self.action_prior[0])
                ],
                "bottom_lift": [
                    float(row["p_bottom_action_zone"])
                    / float(self.action_prior[1])
                ],
            }
        )
        label = str(
            threshold_event_lift_labels(
                lift_frame,
                multiplier=float(self.selected_lift_multiplier),
            ).iloc[0]
        )
        base.update(
            {
                "reason_code": "H1_BELOW_REGISTERED_LIFT_THRESHOLD",
                "predicted_action_class": label,
                "top_lift": float(lift_frame.iloc[0]["top_lift"]),
                "bottom_lift": float(
                    lift_frame.iloc[0]["bottom_lift"]
                ),
                "expected_execution_bucket": row[
                    "execution_bucket"
                ].isoformat(),
            }
        )
        if label == "TOP" and tactical_state == "FULL":
            base.update(
                {
                    "action": "REDUCE_TACTICAL",
                    "reason_code": "QUALIFIED_H1_TOP_ACTION",
                    "tactical_state_after_fill": "REDUCED",
                    "recommended_target_gold_weight": TARGET_WEIGHT[
                        "REDUCED"
                    ],
                }
            )
        elif label == "BOTTOM" and tactical_state == "REDUCED":
            base.update(
                {
                    "action": "REBUY_TACTICAL",
                    "reason_code": "QUALIFIED_H1_BOTTOM_ACTION",
                    "tactical_state_after_fill": "FULL",
                    "recommended_target_gold_weight": TARGET_WEIGHT[
                        "FULL"
                    ],
                }
            )
        elif label == "TOP":
            base["reason_code"] = "STATE_BLOCKED_ALREADY_REDUCED"
        elif label == "BOTTOM":
            base["reason_code"] = "STATE_BLOCKED_ALREADY_FULL"
        return base


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("as_of_utc must be timezone-aware")
    return value.astimezone(UTC)


def _validated_completed_history(
    daily: pd.DataFrame,
    *,
    as_of_utc: datetime,
    required_history_start: date | None,
) -> pd.DataFrame:
    as_of = _as_utc(as_of_utc)
    history = validate_daily_ohlc(daily)
    if len(history) < 22:
        raise ValueError("At least 22 complete UTC daily buckets are required")
    if (
        required_history_start is not None
        and history.iloc[0]["trade_date"] != required_history_start
    ):
        raise ValueError(
            "History must start at "
            f"{required_history_start.isoformat()}"
        )
    last = history.iloc[-1]["trade_date"]
    available_at = datetime.combine(
        last + timedelta(days=1),
        datetime.min.time(),
        UTC,
    )
    if as_of < available_at:
        raise ValueError("Latest UTC daily bucket is incomplete at as_of_utc")
    return history


def future_utc_bucket_schedule(
    source_bucket: date,
    *,
    horizon: int = HORIZON_BUCKETS,
) -> CalendarScheduleResult:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    calendar = xcals.get_calendar("COMEX")
    query_days = max(45, horizon * 3)
    expected = expected_utc_buckets_between(
        source_bucket + timedelta(days=1),
        source_bucket + timedelta(days=query_days),
    )
    ordered = tuple(expected[:horizon])
    if len(ordered) != horizon:
        raise RuntimeError(
            f"Calendar adapter produced {len(ordered)} of {horizon} buckets"
        )
    if any(right <= left for left, right in pairwise(ordered)):
        raise AssertionError("Calendar adapter output is not increasing")
    return CalendarScheduleResult(
        target_buckets=ordered,
        audit={
            "adapter_id": CALENDAR_ADAPTER_ID,
            "calendar_name": calendar.name,
            "calendar_timezone": str(calendar.tz),
            "exchange_calendars_version": version(
                "exchange_calendars"
            ),
            "source_bucket": source_bucket.isoformat(),
            "first_target_bucket": ordered[0].isoformat(),
            "last_target_bucket": ordered[-1].isoformat(),
            "target_buckets": len(ordered),
            "schedule_is_ex_ante": True,
            "future_actual_bars_inspected": False,
        },
    )


def expected_utc_buckets_between(
    start_bucket: date,
    end_bucket: date,
) -> tuple[date, ...]:
    """Map published COMEX session intervals to UTC dates with any trading time."""

    if end_bucket < start_bucket:
        raise ValueError("end_bucket must not precede start_bucket")
    calendar = xcals.get_calendar("COMEX")
    start = pd.Timestamp(start_bucket) - pd.Timedelta("2D")
    end = pd.Timestamp(end_bucket) + pd.Timedelta("2D")
    sessions = calendar.sessions_in_range(start, end)
    schedule = calendar.schedule.loc[sessions]
    buckets: set[date] = set()
    for row in schedule.itertuples(index=False):
        opened = pd.Timestamp(row.open).tz_convert("UTC")
        closed = pd.Timestamp(row.close).tz_convert("UTC")
        current = opened.normalize()
        final = (closed - pd.Timedelta("1ns")).normalize()
        while current <= final:
            bucket = current.date()
            if start_bucket <= bucket <= end_bucket:
                buckets.add(bucket)
            current += pd.Timedelta("1D")
    ordered = tuple(sorted(buckets))
    if any(right <= left for left, right in pairwise(ordered)):
        raise AssertionError("Calendar adapter output is not increasing")
    return ordered


def build_online_feature_rows(
    daily_history: pd.DataFrame,
    target_buckets: Sequence[date],
) -> pd.DataFrame:
    history = validate_daily_ohlc(daily_history)
    targets = tuple(target_buckets)
    if (
        len(targets) != HORIZON_BUCKETS
        or len(set(targets)) != HORIZON_BUCKETS
        or tuple(sorted(targets)) != targets
    ):
        raise ValueError("Online schedule must have 21 ordered unique buckets")
    anchor = history.iloc[-1]["trade_date"]
    if targets[0] <= anchor:
        raise ValueError("Online target buckets must follow the source bucket")
    issuance_id = (
        "ONLINE-"
        + sha256(
            (
                anchor.isoformat()
                + "|"
                + "|".join(value.isoformat() for value in targets)
            ).encode("utf-8")
        ).hexdigest()[:24].upper()
    )
    issuance = pd.DataFrame(
        {
            "issuance_id": [issuance_id] * HORIZON_BUCKETS,
            "feature_anchor_bucket": [anchor] * HORIZON_BUCKETS,
            "feature_anchor_position": [
                len(history) - 1
            ]
            * HORIZON_BUCKETS,
            "horizon_index": range(1, HORIZON_BUCKETS + 1),
            "target_bucket": targets,
            "target_event_group_id": [
                f"UNOBSERVED-{value.isoformat()}" for value in targets
            ],
            "target_label": ["NORMAL"] * HORIZON_BUCKETS,
            "target_label_available_at_utc": [
                pd.Timestamp(value, tz="UTC") + pd.Timedelta("1D")
                for value in targets
            ],
        }
    )
    base = build_future_slot_feature_table(history, issuance).frame
    if base.empty:
        raise ValueError(
            "Latest source path state is unknown; no 21-slot issuance is valid"
        )
    technical = build_technical_exhaustion_feature_table(
        history,
        base,
    ).frame
    anchor_end = pd.Timestamp(anchor, tz="UTC") + pd.Timedelta("1D")
    target_start = pd.to_datetime(
        technical["target_bucket"],
        utc=True,
    )
    strictly_later = target_start.gt(anchor_end)
    eligible = pd.Series(False, index=technical.index)
    if strictly_later.any():
        eligible.loc[strictly_later.idxmax()] = True
    technical["feature_anchor_bucket_end_utc"] = anchor_end
    technical["target_bucket_start_utc"] = target_start
    technical["target_bucket_end_utc"] = (
        target_start + pd.Timedelta("1D")
    )
    technical["boundary_action_eligible"] = eligible
    technical["first_executable_timestamp"] = pd.Series(
        pd.NaT,
        index=technical.index,
        dtype="datetime64[ns, UTC]",
    )
    technical.loc[
        eligible,
        "first_executable_timestamp",
    ] = target_start.loc[eligible]
    technical["execution_bucket"] = None
    technical.loc[eligible, "execution_bucket"] = technical.loc[
        eligible,
        "target_bucket",
    ]
    technical["execution_clock_status"] = (
        "NOT_FIRST_STRICTLY_LATER_TARGET_OPEN"
    )
    technical.loc[
        eligible,
        "execution_clock_status",
    ] = "FIRST_STRICTLY_LATER_TARGET_OPEN"
    return technical


def _validate_training_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "issuance_id",
        "horizon_index",
        "target_event_group_id",
        "target_label",
        "feature_anchor_bucket",
        *R30_FEATURE_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"Release training rows missing: {sorted(missing)}"
        )
    output = frame.copy().reset_index(drop=True)
    horizons = pd.to_numeric(output["horizon_index"], errors="coerce")
    if (
        horizons.isna().any()
        or not horizons.between(1, HORIZON_BUCKETS).all()
        or not output.groupby("issuance_id", sort=False).size().eq(
            HORIZON_BUCKETS
        ).all()
        or set(output["target_label"].astype(str)) != set(CLASS_LABELS)
    ):
        raise ValueError("Release training identity is invalid")
    values = output.loc[:, R30_FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Release training features are non-finite")
    return output


def _validate_prediction_rows(frame: pd.DataFrame) -> pd.DataFrame:
    missing = {"horizon_index", *R30_FEATURE_COLUMNS} - set(frame.columns)
    if missing:
        raise ValueError(
            f"Release prediction rows missing: {sorted(missing)}"
        )
    output = frame.copy().reset_index(drop=True)
    horizons = pd.to_numeric(output["horizon_index"], errors="coerce")
    if (
        len(output) != HORIZON_BUCKETS
        or horizons.tolist() != list(range(1, HORIZON_BUCKETS + 1))
    ):
        raise ValueError("Release prediction requires ordered h1-h21")
    values = output.loc[:, R30_FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Release prediction features are non-finite")
    return output


def _fit_model(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> lgb.LGBMClassifier:
    labels = frame["target_label"].astype(str)
    if set(labels) != set(CLASS_LABELS):
        raise ValueError("Each release head requires all three classes")
    model = lgb.LGBMClassifier(**FROZEN_PARAMETERS)
    model.fit(
        frame.loc[:, feature_columns],
        labels,
        sample_weight=group_normalized_weights(frame),
    )
    return model


def _class_prior(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["target_label"].astype(str).to_numpy()
    weights = group_normalized_weights(frame)
    prior = np.asarray(
        [
            float(weights[labels == label].sum())
            for label in CLASS_LABELS
        ],
        dtype=float,
    )
    prior /= float(prior.sum())
    if (prior <= 0.0).any() or not np.isclose(prior.sum(), 1.0):
        raise ValueError("Release prior lacks a class")
    return prior


def fit_task3_release_candidate(
    training_rows: pd.DataFrame,
    *,
    selected_lift_multiplier: float,
    development_cutoff_utc_exclusive: str,
    model_version: str,
    required_history_start: date | None = REQUIRED_HISTORY_START,
) -> Task3ReleaseCandidateBundle:
    rows = _validate_training_rows(training_rows)
    if (
        not np.isfinite(selected_lift_multiplier)
        or selected_lift_multiplier < 1.0
    ):
        raise ValueError("A finite release lift multiplier is required")
    horizon = pd.to_numeric(rows["horizon_index"])
    h1 = rows.loc[horizon.eq(1)].reset_index(drop=True)
    near = rows.loc[horizon.between(1, 5)].reset_index(drop=True)
    anchor = pd.to_datetime(
        rows["feature_anchor_bucket"],
        errors="raise",
    )
    return Task3ReleaseCandidateBundle(
        model_version=model_version,
        h1_model=_fit_model(h1, R31_H1_FEATURE_COLUMNS),
        h2_model=_fit_model(near, R30_FEATURE_COLUMNS),
        h3_h5_model=_fit_model(near, R29_FEATURE_COLUMNS),
        far_prior=_class_prior(rows),
        action_prior=_class_prior(rows),
        selected_lift_multiplier=float(selected_lift_multiplier),
        development_cutoff_utc_exclusive=(
            development_cutoff_utc_exclusive
        ),
        training_rows=len(rows),
        training_issuances=int(rows["issuance_id"].nunique()),
        training_anchor_start=anchor.min().date().isoformat(),
        training_anchor_end=anchor.max().date().isoformat(),
        required_history_start=required_history_start,
    )


def _ordered_probability(
    model: lgb.LGBMClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    raw = np.asarray(model.predict_proba(features), dtype=float)
    return np.column_stack(
        [
            raw[:, int(np.flatnonzero(model.classes_ == label)[0])]
            for label in CLASS_LABELS
        ]
    )


def _validate_probability(probability: np.ndarray) -> None:
    if (
        probability.ndim != 2
        or probability.shape[1] != len(CLASS_LABELS)
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
        raise ValueError("Release probabilities are invalid")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _record(path: Path, root: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": _relative(path, root),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def persist_task3_release_candidate(
    bundle: Task3ReleaseCandidateBundle,
    destination: str | Path,
    *,
    root: Path,
) -> dict[str, Any]:
    root = root.resolve()
    target = resolve_project_path(destination, root=root)
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite release candidate: {target}"
        )
    target.mkdir(parents=True, exist_ok=False)
    bundle_path = target / "task3_release_candidate.joblib"
    h1_path = target / "h1_r31_lightgbm.txt"
    h2_path = target / "h2_r30_lightgbm.txt"
    h3_h5_path = target / "h3_h5_r29_lightgbm.txt"
    metadata_path = target / "model_metadata.json"
    joblib.dump(bundle, bundle_path, compress=3)
    bundle.h1_model.booster_.save_model(str(h1_path))
    bundle.h2_model.booster_.save_model(str(h2_path))
    bundle.h3_h5_model.booster_.save_model(str(h3_h5_path))
    write_json_atomic(
        metadata_path,
        {
            "project": "AuPilot",
            "artifact_version": ARTIFACT_VERSION,
            "model_id": RELEASE_CANDIDATE_ID,
            "model_version": bundle.model_version,
            "development_cutoff_utc_exclusive": (
                bundle.development_cutoff_utc_exclusive
            ),
            "selected_lift_multiplier": (
                bundle.selected_lift_multiplier
            ),
            "far_prior": bundle.far_prior.tolist(),
            "action_prior": bundle.action_prior.tolist(),
            "routes": {
                "h1": "R31_BASE_FEATURE_H1_HEAD",
                "h2": "R30_TECHNICAL_NEAR_HEAD",
                "h3_h5": "R29_BASE_NEAR_HEAD",
                "h6_h21": "FULL_DEVELOPMENT_EVENT_GROUP_PRIOR",
            },
            "external_model_input": (
                "COMPLETE_DATABENTO_NATIVE_UTC_DAILY_OHLC_HISTORY_ONLY"
            ),
            "technical_indicators_are_internal": True,
            "price_tree_controls_trading": False,
            "rag_controls_trading": False,
            "final_holdout_used": False,
            "forward_shadow_required": True,
        },
    )
    files = [
        _record(bundle_path, root, "VERIFIED_LOCAL_JOBLIB_BUNDLE"),
        _record(h1_path, root, "H1_R31_LIGHTGBM_NATIVE"),
        _record(h2_path, root, "H2_R30_LIGHTGBM_NATIVE"),
        _record(h3_h5_path, root, "H3_H5_R29_LIGHTGBM_NATIVE"),
        _record(metadata_path, root, "MODEL_METADATA"),
    ]
    manifest = {
        "project": "AuPilot",
        "operation": "TASK3_FULL_DEVELOPMENT_RELEASE_CANDIDATE_ARTIFACT",
        "artifact_version": ARTIFACT_VERSION,
        "model_id": RELEASE_CANDIDATE_ID,
        "model_version": bundle.model_version,
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


def load_verified_task3_release_candidate(
    manifest_path: str | Path,
    *,
    root: Path,
) -> Task3ReleaseCandidateBundle:
    root = root.resolve()
    source = resolve_project_path(manifest_path, root=root)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if (
        manifest.get("project") != "AuPilot"
        or manifest.get("artifact_version") != ARTIFACT_VERSION
        or manifest.get("model_id") != RELEASE_CANDIDATE_ID
    ):
        raise ValueError("Unsupported task3 release manifest")
    bundle_path: Path | None = None
    for record in manifest["files"]:
        path = resolve_project_path(record["path"], root=root)
        if (
            path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(
                f"Release artifact verification failed: {record['role']}"
            )
        if record["role"] == "VERIFIED_LOCAL_JOBLIB_BUNDLE":
            bundle_path = path
    if bundle_path is None:
        raise ValueError("Release manifest lacks its joblib bundle")
    bundle = joblib.load(bundle_path)
    if not isinstance(bundle, Task3ReleaseCandidateBundle):
        raise TypeError("Unexpected task3 release bundle type")
    if (
        bundle.model_version != manifest["model_version"]
        or RELEASE_CANDIDATE_ID != manifest["model_id"]
    ):
        raise RuntimeError("Release bundle identity differs from manifest")
    return bundle
