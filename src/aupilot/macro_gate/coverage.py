from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from aupilot.core.enums import MacroRiskLevel

from .calendar_gate import MacroGateResult
from .schemas import (
    MacroClaim,
    MacroCoverageResult,
    MacroScoreComponents,
    MacroSummaryFact,
)


@dataclass(frozen=True)
class CoverageEvaluation:
    results: tuple[MacroCoverageResult, ...]
    accepted_claims: tuple[MacroClaim, ...]
    summary_facts: tuple[MacroSummaryFact, ...]
    assessment_supported: bool
    source_degraded: bool
    evidence_quality_score: int


@dataclass(frozen=True)
class RiskEvaluation:
    risk_level: MacroRiskLevel
    components: MacroScoreComponents
    market_probability_supported: bool


def flatten_coverage_config(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    flattened: dict[str, dict[str, Any]] = {}
    for event_type, children in config.items():
        if not isinstance(children, dict):
            raise ValueError(f"Coverage group {event_type} must be a mapping")
        for name, raw in children.items():
            if isinstance(raw, bool):
                policy = {"required": raw}
            elif isinstance(raw, dict):
                policy = dict(raw)
            else:
                raise ValueError(f"Coverage policy {event_type}.{name} is invalid")
            policy.setdefault("required", True)
            policy.setdefault("freshness_hours", 24 * 365)
            flattened[f"{event_type}.{name}"] = policy
    return flattened


def _freshness_basis(claim: MacroClaim) -> datetime:
    return claim.observed_at_utc or claim.published_at_utc or claim.first_seen_at_utc


def _latest_claims(claims: list[MacroClaim]) -> list[MacroClaim]:
    latest_by_source: dict[str, MacroClaim] = {}
    for claim in claims:
        current = latest_by_source.get(claim.independence_key)
        key = (
            claim.eligible_from_utc,
            _freshness_basis(claim),
            claim.reference_period or "",
            claim.claim_id,
        )
        if current is None or key > (
            current.eligible_from_utc,
            _freshness_basis(current),
            current.reference_period or "",
            current.claim_id,
        ):
            latest_by_source[claim.independence_key] = claim
    return list(latest_by_source.values())


def _independent_claims(claims: list[MacroClaim]) -> list[MacroClaim]:
    unique: dict[str, MacroClaim] = {}
    for claim in sorted(claims, key=lambda item: item.claim_id):
        key = claim.near_duplicate_group or claim.independence_key
        unique.setdefault(key, claim)
    return list(unique.values())


def evaluate_coverage(
    *,
    claims: tuple[MacroClaim, ...],
    required_coverage: dict[str, Any],
    decision_as_of_utc: datetime,
) -> CoverageEvaluation:
    if decision_as_of_utc.tzinfo is None or decision_as_of_utc.utcoffset() is None:
        raise ValueError("decision_as_of_utc must be timezone-aware")
    policies = flatten_coverage_config(required_coverage)
    by_slot: dict[str, list[MacroClaim]] = defaultdict(list)
    for claim in claims:
        if claim.eligible_from_utc > decision_as_of_utc:
            continue
        if claim.published_at_utc and claim.published_at_utc > decision_as_of_utc:
            continue
        if claim.revision_status == "REVISED" and claim.slot.endswith(
            "latest_initial_release"
        ):
            continue
        if claim.slot in policies:
            by_slot[claim.slot].append(claim)

    results: list[MacroCoverageResult] = []
    accepted: list[MacroClaim] = []
    facts: list[MacroSummaryFact] = []
    for slot, policy in policies.items():
        required = bool(policy["required"])
        candidates = by_slot.get(slot, [])
        freshness = timedelta(hours=float(policy["freshness_hours"]))
        fresh = [
            claim
            for claim in candidates
            if _freshness_basis(claim) > decision_as_of_utc
            or decision_as_of_utc - _freshness_basis(claim) <= freshness
        ]
        if not fresh:
            status = "STALE" if candidates else "MISSING"
            reason = f"{slot.upper().replace('.', '_')}_{status}"
            results.append(
                MacroCoverageResult(
                    slot=slot,
                    required=required,
                    status=status,
                    reason_codes=(reason,),
                )
            )
            continue

        latest = _latest_claims(fresh)
        official = [claim for claim in latest if claim.source_tier == "A"]
        selected: list[MacroClaim] = []
        status = "MISSING"
        reason_codes: tuple[str, ...]
        if official:
            values = {claim.normalized_value for claim in official}
            if len(values) != 1:
                status = "CONFLICTING"
                reason_codes = (f"{slot.upper().replace('.', '_')}_OFFICIAL_CONFLICT",)
            else:
                status = "COVERED"
                selected = _independent_claims(official)
                reason_codes = ("TIER_A_PRIMARY",)
        else:
            secondary = _independent_claims(
                [claim for claim in latest if claim.source_tier == "B"]
            )
            groups: dict[str, list[MacroClaim]] = defaultdict(list)
            for claim in secondary:
                groups[claim.normalized_value].append(claim)
            if len(groups) > 1:
                status = "CONFLICTING"
                reason_codes = (f"{slot.upper().replace('.', '_')}_SECONDARY_CONFLICT",)
            elif groups:
                selected = next(iter(groups.values()))
                if len(selected) >= 2:
                    status = "DEGRADED"
                    reason_codes = ("TIER_B_INDEPENDENT_QUORUM",)
                else:
                    status = "MISSING"
                    selected = []
                    reason_codes = ("TIER_B_QUORUM_INSUFFICIENT",)
            else:
                reason_codes = (f"{slot.upper().replace('.', '_')}_MISSING",)

        latest_at = max((_freshness_basis(claim) for claim in latest), default=None)
        results.append(
            MacroCoverageResult(
                slot=slot,
                required=required,
                status=status,
                supporting_claim_ids=tuple(claim.claim_id for claim in selected),
                latest_observation_at_utc=latest_at,
                reason_codes=reason_codes,
            )
        )
        accepted.extend(selected)
        if selected:
            facts.append(
                MacroSummaryFact(
                    text=selected[0].display_text,
                    claim_ids=tuple(claim.claim_id for claim in selected),
                )
            )

    bad_required = any(
        item.required and item.status not in {"COVERED", "DEGRADED"}
        for item in results
    )
    degraded_required = any(
        item.required and item.status == "DEGRADED" for item in results
    )
    optional_degraded = any(
        not item.required and item.status != "COVERED" for item in results
    )
    unique_accepted = {claim.claim_id: claim for claim in accepted}
    eligible_evidence_available = bool(unique_accepted)
    if not eligible_evidence_available:
        quality_score = 3
    elif degraded_required and not bad_required:
        quality_score = 3
    else:
        quality_score = 1
    return CoverageEvaluation(
        results=tuple(results),
        accepted_claims=tuple(unique_accepted.values()),
        summary_facts=tuple(facts),
        # Required-slot gaps remain visible diagnostics, but one eligible
        # accepted source is enough to support the partial official assessment.
        assessment_supported=eligible_evidence_available,
        source_degraded=bad_required or degraded_required or optional_degraded,
        evidence_quality_score=quality_score,
    )


def evaluate_risk(
    *,
    calendar_result: MacroGateResult,
    coverage: CoverageEvaluation,
    decision_as_of_utc: datetime,
    expectation_config: dict[str, Any],
    release_cooldown_hours: int,
) -> RiskEvaluation:
    release_cutoff = timedelta(hours=release_cooldown_hours)
    release_score = 1
    for claim in coverage.accepted_claims:
        if claim.published_at_utc is None:
            continue
        since_release = decision_as_of_utc - claim.published_at_utc
        if timedelta(0) <= since_release <= release_cutoff and (
            claim.slot.endswith("latest_initial_release")
            or claim.slot == "FOMC.latest_official_decision"
        ):
            release_score = 5
            break

    probabilities: list[float] = []
    for claim in coverage.accepted_claims:
        if claim.slot != "FOMC.market_expectation":
            continue
        if isinstance(claim.value, dict):
            value = claim.value.get("max_outcome_probability")
        else:
            value = claim.value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if 0 <= numeric <= 1:
                probabilities.append(numeric)
    probability_supported = bool(probabilities)
    if not probabilities:
        expectation_required = any(
            item.slot == "FOMC.market_expectation" and item.required
            for item in coverage.results
        )
        expectation_score = 3 if expectation_required else 1
    else:
        maximum = max(probabilities)
        if maximum >= float(expectation_config["high_confidence_probability"]):
            expectation_score = 1
        elif maximum >= float(expectation_config["medium_confidence_probability"]):
            expectation_score = 2
        else:
            expectation_score = 3

    components = MacroScoreComponents(
        calendar_score=calendar_result.risk_level.score,
        proximity_score=calendar_result.proximity_score,
        impact_strength_score=calendar_result.impact_strength_score,
        rate_relevance_score=calendar_result.rate_relevance_score,
        release_cooldown_score=release_score,
        expectation_uncertainty_score=expectation_score,
        evidence_quality_score=coverage.evidence_quality_score,
    )
    final_score = max(
        components.calendar_score,
        components.release_cooldown_score,
        components.expectation_uncertainty_score,
        components.evidence_quality_score,
    )
    if not coverage.assessment_supported or "ASSESSMENT_UNAVAILABLE" in calendar_result.reason_codes:
        final_score = max(final_score, MacroRiskLevel.CAUTION.score)
    level = next(item for item in MacroRiskLevel if item.score == final_score)
    return RiskEvaluation(
        risk_level=level,
        components=components,
        market_probability_supported=probability_supported,
    )
