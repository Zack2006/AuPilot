from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from aupilot.core.enums import MacroRiskLevel
from aupilot.core.macro_policy import (
    CAUTION_HOURS,
    HOLD_HOURS,
    MACRO_EVENT_FACTOR_POLICY,
    MONITORING_HOURS,
    POST_RELEASE_CANCEL_HOURS,
)


@dataclass(frozen=True)
class MacroEvent:
    event_id: str
    event_type: str
    scheduled_release_at_utc: datetime
    actual_release_at_utc: datetime | None = None
    high_impact: bool = True


@dataclass(frozen=True)
class MacroGateResult:
    risk_level: MacroRiskLevel
    reason_codes: tuple[str, ...]
    relevant_event_ids: tuple[str, ...] = ()
    news_summary: tuple[str, ...] = ()
    proximity_score: int = 1
    impact_strength_score: int = 1
    rate_relevance_score: int = 1


def _event_factor_scores(event_type: str) -> tuple[int, int]:
    policy = MACRO_EVENT_FACTOR_POLICY.get(event_type.upper())
    if policy is None:
        return 1, 1
    return policy.impact_strength_score, policy.rate_relevance_score


def _adjust_pre_release_score(
    proximity_score: int,
    *,
    impact_strength_score: int,
    rate_relevance_score: int,
) -> int:
    # Only an event class with maximum potential impact and direct rate-chain
    # relevance may escalate one band. Pre-release risk never becomes Cancel.
    adjustment = int(impact_strength_score == 5 and rate_relevance_score == 5)
    return min(MacroRiskLevel.HOLD.score, proximity_score + adjustment)


def evaluate_macro_calendar(
    *,
    decision_as_of_utc: datetime,
    events: tuple[MacroEvent, ...],
    calendar_is_fresh: bool,
    fetch_succeeded: bool,
    pre_release_window: timedelta = timedelta(hours=MONITORING_HOURS),
    caution_window: timedelta = timedelta(hours=CAUTION_HOURS),
    hold_window: timedelta = timedelta(hours=HOLD_HOURS),
    post_release_cooldown: timedelta = timedelta(hours=POST_RELEASE_CANCEL_HOURS),
) -> MacroGateResult:
    if decision_as_of_utc.tzinfo is None or decision_as_of_utc.utcoffset() is None:
        raise ValueError("decision_as_of_utc must be timezone-aware")
    if not (
        timedelta(0) < hold_window <= caution_window <= pre_release_window
        and post_release_cooldown >= timedelta(0)
    ):
        raise ValueError("Macro calendar risk windows are inconsistent")
    if not fetch_succeeded:
        return MacroGateResult(
            MacroRiskLevel.CAUTION,
            ("MACRO_FETCH_FAILED", "ASSESSMENT_UNAVAILABLE"),
            news_summary=("官方宏观事件日历获取失败, 当前外部风险无法可靠评估。",),
        )
    if not calendar_is_fresh:
        return MacroGateResult(
            MacroRiskLevel.CAUTION,
            ("MACRO_CALENDAR_STALE", "ASSESSMENT_UNAVAILABLE"),
            news_summary=("官方宏观事件日历已过期, 当前外部风险无法可靠评估。",),
        )

    upcoming: list[tuple[MacroEvent, timedelta]] = []
    cooling: list[tuple[MacroEvent, timedelta]] = []
    for event in events:
        scheduled = event.scheduled_release_at_utc
        if scheduled.tzinfo is None or scheduled.utcoffset() is None:
            raise ValueError("Macro event timestamps must be timezone-aware")
        if not event.high_impact:
            continue
        until_release = scheduled - decision_as_of_utc
        if timedelta(0) <= until_release <= pre_release_window:
            upcoming.append((event, until_release))
        released = event.actual_release_at_utc
        if released is not None:
            if released.tzinfo is None or released.utcoffset() is None:
                raise ValueError("actual_release_at_utc must be timezone-aware")
            since_release = decision_as_of_utc - released
            if timedelta(0) <= since_release <= post_release_cooldown:
                cooling.append((event, since_release))

    if cooling:
        selected_event, _ = max(
            cooling,
            key=lambda value: (*_event_factor_scores(value[0].event_type), -value[1].total_seconds()),
        )
        impact_score, relevance_score = _event_factor_scores(selected_event.event_type)
        return MacroGateResult(
            MacroRiskLevel.CANCEL,
            ("HIGH_IMPACT_EVENT_COOLDOWN",),
            tuple(sorted({event.event_id for event, _ in cooling})),
            tuple(
                f"{event.event_type} 已于 {event.actual_release_at_utc.isoformat()} 发布, "
                "仍处于发布后高波动冷却窗口。"
                for event, _ in sorted(cooling, key=lambda value: value[1])
            ),
            proximity_score=MacroRiskLevel.CANCEL.score,
            impact_strength_score=impact_score,
            rate_relevance_score=relevance_score,
        )
    if upcoming:
        evaluated = []
        for event, delta in upcoming:
            if delta <= hold_window:
                proximity_score = MacroRiskLevel.HOLD.score
                reason = "HIGH_IMPACT_EVENT_IMMINENT"
            elif delta <= caution_window:
                proximity_score = MacroRiskLevel.CAUTION.score
                reason = "HIGH_IMPACT_EVENT_NEAR"
            else:
                proximity_score = MacroRiskLevel.CLEARED.score
                reason = "HIGH_IMPACT_EVENT_IN_48H_WINDOW"
            impact_score, relevance_score = _event_factor_scores(event.event_type)
            adjusted_score = _adjust_pre_release_score(
                proximity_score,
                impact_strength_score=impact_score,
                rate_relevance_score=relevance_score,
            )
            evaluated.append(
                (adjusted_score, proximity_score, impact_score, relevance_score, -delta.total_seconds(), event, reason)
            )
        selected = max(evaluated, key=lambda value: value[:5])
        adjusted_score, proximity_score, impact_score, relevance_score, _, _, reason = selected
        level = next(item for item in MacroRiskLevel if item.score == adjusted_score)
        reasons = [reason]
        if adjusted_score > proximity_score:
            reasons.append("EVENT_IMPACT_RELEVANCE_ESCALATION")
        return MacroGateResult(
            level,
            tuple(reasons),
            tuple(sorted({event.event_id for event, _ in upcoming})),
            tuple(
                f"{event.event_type} 计划于 {event.scheduled_release_at_utc.isoformat()} 发布; "
                f"事件类别潜在影响 { _event_factor_scores(event.event_type)[0] }/5, "
                f"利率链相关性 { _event_factor_scores(event.event_type)[1] }/5。"
                for event, _ in sorted(upcoming, key=lambda value: value[1])
            ),
            proximity_score=proximity_score,
            impact_strength_score=impact_score,
            rate_relevance_score=relevance_score,
        )
    return MacroGateResult(
        MacroRiskLevel.APPROVED,
        ("NO_HIGH_IMPACT_EVENT_IN_48H_WINDOW",),
        news_summary=("未来 48 小时官方日历内未发现白名单高影响事件。",),
    )
