"""Frozen five-level official macro-event risk policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .enums import MacroRiskLevel


MONITORING_HOURS = 48
CAUTION_HOURS = 24
HOLD_HOURS = 6
POST_RELEASE_CANCEL_HOURS = 1


@dataclass(frozen=True)
class MacroRiskBandPolicy:
    risk_level: MacroRiskLevel
    risk_score: int
    timing: str
    lower_bound_hours: int
    upper_bound_hours: int | None
    lower_bound_inclusive: bool
    upper_bound_inclusive: bool
    assessment_unavailable_also: bool = False
    event_factor_adjustment_allowed: bool = False


@dataclass(frozen=True)
class MacroEventFactorPolicy:
    """Auditable pre-release impact potential and rate-chain relevance."""

    impact_strength_score: int
    rate_relevance_score: int


MACRO_EVENT_FACTOR_POLICY = {
    "FOMC": MacroEventFactorPolicy(impact_strength_score=5, rate_relevance_score=5),
    "CPI": MacroEventFactorPolicy(impact_strength_score=5, rate_relevance_score=5),
    "PCE": MacroEventFactorPolicy(impact_strength_score=5, rate_relevance_score=5),
    "NFP": MacroEventFactorPolicy(impact_strength_score=4, rate_relevance_score=4),
}


FROZEN_MACRO_RISK_SCALE = (
    MacroRiskBandPolicy(
        MacroRiskLevel.APPROVED, 1, "OUTSIDE_MONITORING_WINDOW",
        MONITORING_HOURS, None, False, False,
    ),
    MacroRiskBandPolicy(
        MacroRiskLevel.CLEARED, 2, "BEFORE_RELEASE",
        CAUTION_HOURS, MONITORING_HOURS, False, True,
        event_factor_adjustment_allowed=True,
    ),
    MacroRiskBandPolicy(
        MacroRiskLevel.CAUTION, 3, "BEFORE_RELEASE",
        HOLD_HOURS, CAUTION_HOURS, False, True,
        assessment_unavailable_also=True,
        event_factor_adjustment_allowed=True,
    ),
    MacroRiskBandPolicy(
        MacroRiskLevel.HOLD, 4, "BEFORE_RELEASE",
        0, HOLD_HOURS, True, True,
        event_factor_adjustment_allowed=True,
    ),
    MacroRiskBandPolicy(
        MacroRiskLevel.CANCEL, 5, "AFTER_RELEASE",
        0, POST_RELEASE_CANCEL_HOURS, True, True,
    ),
)


def validate_frozen_calendar_windows(windows: Mapping[str, object]) -> None:
    expected = {
        "monitoring_hours": MONITORING_HOURS,
        "caution_hours": CAUTION_HOURS,
        "hold_hours": HOLD_HOURS,
        "post_release_cancel_hours": POST_RELEASE_CANCEL_HOURS,
    }
    actual = {key: int(windows[key]) for key in expected}
    if actual != expected:
        raise ValueError(f"calendar_windows must match the frozen five-level policy: {expected}")


def validate_event_factor_policy(policy: Mapping[str, object]) -> None:
    expected = {
        event_type: {
            "impact_strength_score": values.impact_strength_score,
            "rate_relevance_score": values.rate_relevance_score,
        }
        for event_type, values in MACRO_EVENT_FACTOR_POLICY.items()
    }
    actual = {
        str(event_type).upper(): {
            "impact_strength_score": int(values["impact_strength_score"]),
            "rate_relevance_score": int(values["rate_relevance_score"]),
        }
        for event_type, values in policy.items()
        if isinstance(values, Mapping)
    }
    if actual != expected:
        raise ValueError(f"event_factor_policy must match the audited event policy: {expected}")
