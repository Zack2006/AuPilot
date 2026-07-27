"""Unified runtime adapter for the frozen MN18 and PN02 model bundles.

MN18 remains the sole owner of turning probabilities and tactical actions.
PN02 only attaches conditional OHLC expectations to the same 21 UTC slots.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from aupilot.deployment.mn18_forward_candidate import (
    RELEASE_CANDIDATE_ID,
    MN18ForwardCandidateBundle,
    load_verified_mn18_forward_candidate,
)
from aupilot.deployment.pn02_conditional_price import (
    PN02ConditionalPriceBundle,
    load_pn02_price_bundle,
)
from aupilot.training.conditional_ohlc_residual_lightgbm import (
    MODEL_ID as PN02_MODEL_ID,
)


@dataclass(frozen=True)
class MN18PN02DualModel:
    """Run MN18 first and attach advisory PN02 price expectations."""

    turning_bundle: MN18ForwardCandidateBundle
    price_bundle: PN02ConditionalPriceBundle

    def predict_from_history(
        self,
        daily_history: pd.DataFrame,
        *,
        current_gold_weight: float,
        outstanding_top_inventory_pp: float,
        as_of_utc: datetime,
    ) -> dict[str, Any]:
        turning = self.turning_bundle.predict_from_history(
            daily_history,
            current_gold_weight=current_gold_weight,
            outstanding_top_inventory_pp=outstanding_top_inventory_pp,
            as_of_utc=as_of_utc,
        )
        before = copy.deepcopy(turning)
        combined = self.price_bundle.attach_to_turning_output(
            daily_history,
            turning_output=turning,
        )
        validate_dual_output(before, combined)
        return combined


def validate_dual_output(
    turning_output: dict[str, Any],
    combined_output: dict[str, Any],
) -> None:
    """Prove that PN02 only filled ``price_outlook``."""

    turning_without_price = copy.deepcopy(turning_output)
    turning_without_price["price_outlook"] = None
    combined_without_price = copy.deepcopy(combined_output)
    outlook = combined_without_price.get("price_outlook")
    combined_without_price["price_outlook"] = None
    if combined_without_price != turning_without_price:
        raise AssertionError("PN02 changed the MN18 output outside price_outlook")
    if not isinstance(outlook, dict):
        raise AssertionError("PN02 price_outlook is missing")
    if outlook.get("model_id") != PN02_MODEL_ID:
        raise AssertionError("PN02 price model identity changed")
    if outlook.get("controls_trading") is not False:
        raise AssertionError("PN02 must not control trading")
    turning_rows = turning_output.get("probability_rows")
    price_rows = outlook.get("slots")
    if not isinstance(turning_rows, list) or len(turning_rows) != 21:
        raise AssertionError("MN18 must expose exactly 21 probability rows")
    if not isinstance(price_rows, list) or len(price_rows) != 21:
        raise AssertionError("PN02 must expose exactly 21 price slots")
    for turning, price in zip(turning_rows, price_rows, strict=True):
        if (
            int(turning["horizon_index"]) != int(price["horizon_index"])
            or str(turning["target_bucket"]) != str(price["target_bucket"])
        ):
            raise AssertionError("MN18 and PN02 slot identities differ")
        for scenario in ("top_conditional", "bottom_conditional"):
            payload = price.get(scenario)
            if not isinstance(payload, dict):
                raise AssertionError(f"PN02 scenario missing: {scenario}")
            if payload.get("controls_trading") is not False:
                raise AssertionError("PN02 scenario must remain advisory")


def load_verified_mn18_pn02_dual_model(
    *,
    turning_manifest_path: str | Path,
    price_bundle_path: str | Path,
    root: str | Path = ".",
) -> MN18PN02DualModel:
    """Load both frozen bundles without weakening MN18 manifest checks."""

    turning = load_verified_mn18_forward_candidate(
        turning_manifest_path,
        root=root,
    )
    price = load_pn02_price_bundle(price_bundle_path)
    if turning.model_version == "":
        raise ValueError("MN18 model version is empty")
    if price.model_id != PN02_MODEL_ID:
        raise ValueError("PN02 model identity changed")
    return MN18PN02DualModel(
        turning_bundle=turning,
        price_bundle=price,
    )


DUAL_MODEL_IDS = {
    "turning_model_id": RELEASE_CANDIDATE_ID,
    "price_model_id": PN02_MODEL_ID,
}

