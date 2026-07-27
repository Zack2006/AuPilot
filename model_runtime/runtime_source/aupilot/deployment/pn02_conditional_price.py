"""Deployable PN02 conditional 21-slot OHLC price-outlook bundle."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from aupilot.backtest.pivot_baseline import validate_daily_ohlc
from aupilot.training.conditional_future_ohlc import (
    ACTION_SCENARIO_LABELS,
    OHLC_FIELDS,
    ConditionalMedianBundle,
    build_online_conditional_query,
    with_scenario,
)
from aupilot.training.conditional_ohlc_residual_lightgbm import (
    MODEL_ID,
    RESIDUAL_COLUMNS,
    _matrix,
    _with_anchor_features,
    apply_residual_shrinkage,
    augment_conditional_frame_with_mn18,
)

REQUIRED_HISTORY_START = date(2010, 6, 7)
SEGMENT = {
    **{value: "h01_h05" for value in range(1, 6)},
    **{value: "h06_h10" for value in range(6, 11)},
    **{value: "h11_h21" for value in range(11, 22)},
}


@dataclass(frozen=True)
class PN02ConditionalPriceBundle:
    """PN02 full-development refit for advisory price expectations."""

    model_version: str
    training_start: str
    training_end: str
    development_cutoff_utc_exclusive: str
    evidence_role: tuple[str, ...]
    median_bundle: ConditionalMedianBundle
    residual_model_strings: dict[str, str]
    residual_shrinkage_alpha: float
    residual_relative_bands: dict[str, tuple[float, float]]
    required_history_start: date | None = REQUIRED_HISTORY_START
    model_id: str = MODEL_ID

    def __post_init__(self) -> None:
        if self.model_id != MODEL_ID:
            raise ValueError("PN02 deployable model identity changed")
        if set(self.residual_model_strings) != {
            "target_open_log_return",
            "target_close_log_return",
            "target_upper_wick_log",
            "target_lower_wick_log",
        }:
            raise ValueError("PN02 residual target models are incomplete")
        if not 0.0 <= self.residual_shrinkage_alpha <= 1.0:
            raise ValueError("PN02 residual alpha is invalid")

    def _predict_frame(
        self,
        daily_history: pd.DataFrame,
        *,
        future_buckets: tuple[object, ...],
        as_of_utc: datetime,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        history = _validated_completed_history(
            daily_history,
            as_of_utc=as_of_utc,
            required_history_start=self.required_history_start,
        )
        base = build_online_conditional_query(history, future_buckets)
        base = augment_conditional_frame_with_mn18(history, base)
        query = pd.concat(
            [
                with_scenario(base, scenario)
                for scenario in ACTION_SCENARIO_LABELS
            ],
            ignore_index=True,
        )
        anchored, _ = _with_anchor_features(query, self.median_bundle)
        matrix = _matrix(anchored)
        output = anchored.copy().reset_index(drop=True)
        for target_column, residual_column in zip(
            self.residual_model_strings,
            RESIDUAL_COLUMNS,
            strict=True,
        ):
            booster = lgb.Booster(
                model_str=self.residual_model_strings[target_column]
            )
            output[residual_column] = booster.predict(matrix).astype(float)
        output = apply_residual_shrinkage(
            output,
            self.residual_shrinkage_alpha,
        )
        output["price_model_id"] = self.model_id
        return history, output

    def predict(
        self,
        daily_history: pd.DataFrame,
        *,
        future_buckets: tuple[object, ...],
        as_of_utc: datetime,
    ) -> dict[str, Any]:
        history, predicted = self._predict_frame(
            daily_history,
            future_buckets=future_buckets,
            as_of_utc=as_of_utc,
        )
        slots = []
        for horizon in range(1, 22):
            rows = predicted.loc[
                predicted["horizon_index"].eq(horizon)
            ]
            if len(rows) != 2:
                raise AssertionError("PN02 scenario/horizon row count changed")
            target_bucket = str(rows.iloc[0]["target_bucket"])
            slots.append(
                {
                    "horizon_index": horizon,
                    "target_bucket": target_bucket,
                    "top_conditional": self._scenario_payload(
                        rows,
                        scenario="TOP_ACTION_ZONE",
                    ),
                    "bottom_conditional": self._scenario_payload(
                        rows,
                        scenario="BOTTOM_ACTION_ZONE",
                    ),
                }
            )
        return {
            "project": "AuPilot",
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_status": list(self.evidence_role),
            "as_of_utc": as_of_utc.astimezone(UTC).isoformat(),
            "source_bucket": history.iloc[-1]["trade_date"].isoformat(),
            "daily_unit_id": "CANONICAL_GC_UTC_DAILY_BUCKET_V1",
            "slot_count": len(slots),
            "slots": slots,
            "controls_trading": False,
            "automatic_execution": False,
            "advisory_only": True,
            "input_contract": {
                "provider": "Databento",
                "dataset": "GLBX.MDP3",
                "symbol": "GC.v.0",
                "stype_in": "continuous",
                "schema": "ohlcv-1d",
                "history_ohlc_only": True,
                "history_start_required": (
                    self.required_history_start.isoformat()
                    if self.required_history_start
                    else None
                ),
                "latest_bucket_must_be_complete": True,
                "technical_features_internal": True,
                "intraday_inputs_used": False,
                "rag_or_macro_inputs_used": False,
            },
            "turning_integration_contract": {
                "compatible_turning_model": (
                    "MN18_THREE_TOP_EXPERT_FORWARD_SHADOW_CANDIDATE_V1"
                ),
                "join_keys": ["horizon_index", "target_bucket"],
                "top_price_source": "top_conditional",
                "bottom_price_source": "bottom_conditional",
                "normal_slot_price_semantics": "DISPLAY_BOTH_OR_HIDE",
                "price_model_must_not_change_action": True,
            },
        }

    def predict_from_turning_output(
        self,
        daily_history: pd.DataFrame,
        *,
        turning_output: dict[str, Any],
    ) -> dict[str, Any]:
        rows = turning_output.get("probability_rows")
        if not isinstance(rows, list) or len(rows) != 21:
            raise ValueError("MN18 output must contain 21 probability rows")
        ordered = sorted(rows, key=lambda row: int(row["horizon_index"]))
        horizons = [int(row["horizon_index"]) for row in ordered]
        if horizons != list(range(1, 22)):
            raise ValueError("MN18 horizon order is invalid")
        future_buckets = tuple(row["target_bucket"] for row in ordered)
        as_of_utc = datetime.fromisoformat(str(turning_output["as_of_utc"]))
        result = self.predict(
            daily_history,
            future_buckets=future_buckets,
            as_of_utc=as_of_utc,
        )
        for turning_row, price_row in zip(
            ordered,
            result["slots"],
            strict=True,
        ):
            if (
                int(turning_row["horizon_index"])
                != int(price_row["horizon_index"])
                or str(turning_row["target_bucket"])
                != str(price_row["target_bucket"])
            ):
                raise AssertionError("MN18/PN02 slot identities differ")
        result["turning_model_id"] = turning_output.get("model_id")
        return result

    def attach_to_turning_output(
        self,
        daily_history: pd.DataFrame,
        *,
        turning_output: dict[str, Any],
    ) -> dict[str, Any]:
        output = copy.deepcopy(turning_output)
        outlook = self.predict_from_turning_output(
            daily_history,
            turning_output=turning_output,
        )
        output["price_outlook"] = outlook
        return output

    def _scenario_payload(
        self,
        rows: pd.DataFrame,
        *,
        scenario: str,
    ) -> dict[str, Any]:
        match = rows.loc[rows["scenario_label"].eq(scenario)]
        if len(match) != 1:
            raise AssertionError("PN02 scenario row changed")
        row = match.iloc[0]
        source_close = float(row["source_close"])
        segment = SEGMENT[int(row["horizon_index"])]
        point = {
            field: float(row[f"predicted_{field}"])
            for field in OHLC_FIELDS
        }
        intervals: dict[str, dict[str, float] | None] = {}
        for field in OHLC_FIELDS:
            band = self.residual_relative_bands.get(
                f"{scenario}|{segment}|{field}"
            )
            intervals[field] = (
                None
                if band is None
                else {
                    "lower": max(
                        np.finfo(float).tiny,
                        point[field] + source_close * float(band[0]),
                    ),
                    "upper": max(
                        np.finfo(float).tiny,
                        point[field] + source_close * float(band[1]),
                    ),
                }
            )
        return {
            "scenario_label": scenario,
            "point": point,
            "marginal_80pct_intervals": intervals,
            "controls_trading": False,
        }


def _validated_completed_history(
    daily_history: pd.DataFrame,
    *,
    as_of_utc: datetime,
    required_history_start: date | None,
) -> pd.DataFrame:
    if as_of_utc.tzinfo is None or as_of_utc.utcoffset() is None:
        raise ValueError("as_of_utc must be timezone-aware")
    history = validate_daily_ohlc(daily_history)
    if len(history) < 60:
        raise ValueError("At least 60 complete UTC daily buckets are required")
    if (
        required_history_start is not None
        and history.iloc[0]["trade_date"] != required_history_start
    ):
        raise ValueError(
            f"History must start at {required_history_start.isoformat()}"
        )
    latest = history.iloc[-1]["trade_date"]
    complete_at = datetime.combine(
        latest + timedelta(days=1),
        datetime.min.time(),
        UTC,
    )
    if as_of_utc.astimezone(UTC) < complete_at:
        raise ValueError("Latest UTC daily bucket is incomplete")
    return history


def save_pn02_price_bundle(
    bundle: PN02ConditionalPriceBundle,
    path: str | Any,
) -> None:
    joblib.dump(bundle, path)


def load_pn02_price_bundle(path: str | Any) -> PN02ConditionalPriceBundle:
    value = joblib.load(path)
    if not isinstance(value, PN02ConditionalPriceBundle):
        raise TypeError("File does not contain a PN02 price bundle")
    return value
