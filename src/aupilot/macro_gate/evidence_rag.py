from __future__ import annotations

from datetime import datetime

from aupilot.core.enums import MacroRiskLevel

from .calendar_gate import MacroGateResult
from .coverage import evaluate_coverage, evaluate_risk
from .evidence_store import MacroEvidenceStore
from .schemas import EvidenceCitation, MacroAssessment, MacroClaim

DEFAULT_QUERIES = {
    "FOMC": "FOMC meeting monetary policy federal funds rate statement calendar",
    "RATES": "effective federal funds rate reference rates monetary policy",
    "CPI": "consumer price index inflation release schedule",
    "PCE": "personal consumption expenditures price index release schedule",
    "NFP": "employment situation nonfarm payrolls release schedule",
}

DEFAULT_RATE_SERIES = (
    "DFEDTARL",
    "DFEDTARU",
    "DFF",
    "DGS2",
    "DGS10",
    "DFII10",
    "T10YIE",
)


class EvidenceRAG:
    def __init__(
        self,
        store: MacroEvidenceStore,
        *,
        top_k: int = 5,
        rate_series_ids: tuple[str, ...] = DEFAULT_RATE_SERIES,
        minimum_rate_series: int = 5,
        maximum_rate_staleness_days: int = 10,
        require_rate_observations: bool = False,
        required_coverage: dict | None = None,
        expectation_config: dict | None = None,
        release_cooldown_hours: int = 6,
        allowed_provider_ids: frozenset[str] | None = None,
        read_stored_evidence: bool = True,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        normalized_series = tuple(
            dict.fromkeys(value.strip().upper() for value in rate_series_ids)
        )
        if not normalized_series or any(not value for value in normalized_series):
            raise ValueError("At least one rate series is required")
        if not 1 <= minimum_rate_series <= len(normalized_series):
            raise ValueError("minimum_rate_series must fit inside the rate-series allowlist")
        if maximum_rate_staleness_days < 0:
            raise ValueError("maximum_rate_staleness_days cannot be negative")
        self.store = store
        self.top_k = top_k
        self.rate_series_ids = normalized_series
        self.minimum_rate_series = minimum_rate_series
        self.maximum_rate_staleness_days = maximum_rate_staleness_days
        self.require_rate_observations = require_rate_observations
        self.required_coverage = required_coverage
        self.expectation_config = expectation_config or {
            "high_confidence_probability": 0.8,
            "medium_confidence_probability": 0.6,
        }
        self.release_cooldown_hours = release_cooldown_hours
        self.allowed_provider_ids = allowed_provider_ids
        self.read_stored_evidence = read_stored_evidence

    def _assess_claims(
        self,
        *,
        decision_as_of_utc: datetime,
        calendar_result: MacroGateResult,
        calendar_claims: tuple[MacroClaim, ...],
    ) -> MacroAssessment:
        try:
            stored = (
                self.store.claims_as_of(as_of_utc=decision_as_of_utc)
                if self.read_stored_evidence
                else ()
            )
            if self.allowed_provider_ids is not None:
                stored = tuple(
                    claim
                    for claim in stored
                    if claim.provider_id in self.allowed_provider_ids
                )
            all_claims = (*stored, *calendar_claims)
            coverage = evaluate_coverage(
                claims=all_claims,
                required_coverage=self.required_coverage or {},
                decision_as_of_utc=decision_as_of_utc,
            )
            risk = evaluate_risk(
                calendar_result=calendar_result,
                coverage=coverage,
                decision_as_of_utc=decision_as_of_utc,
                expectation_config=self.expectation_config,
                release_cooldown_hours=self.release_cooldown_hours,
            )
        except Exception as error:
            return MacroAssessment(
                risk_level=MacroRiskLevel.CAUTION,
                risk_score=MacroRiskLevel.CAUTION.score,
                decision_as_of_utc=decision_as_of_utc,
                reason_codes=("EVIDENCE_RETRIEVAL_FAILED", type(error).__name__),
                news_summary=("Official macro claim retrieval failed; reliable assessment is unavailable.",),
                assessment_supported=False,
            )

        citations = tuple(
            EvidenceCitation(
                doc_id=claim.source_record_id,
                provider_id=claim.provider_id,
                source_tier=claim.source_tier,
                event_type=claim.event_type,
                title=claim.display_text,
                canonical_url=claim.canonical_url,
                published_at_utc=claim.published_at_utc,
                retrieved_at_utc=claim.retrieved_at_utc,
                first_seen_at_utc=claim.first_seen_at_utc,
                eligible_from_utc=claim.eligible_from_utc,
                content_sha256=claim.content_sha256,
                revision_status=claim.revision_status,
                score=1.0,
                claim_ids=(claim.claim_id,),
            )
            for claim in coverage.accepted_claims
        )
        calendar_unavailable = "ASSESSMENT_UNAVAILABLE" in calendar_result.reason_codes
        reason_codes = [
            code
            for code in calendar_result.reason_codes
            if code != "ASSESSMENT_UNAVAILABLE" or not coverage.assessment_supported
        ]
        for item in coverage.results:
            reason_codes.extend(item.reason_codes)
        if coverage.source_degraded or calendar_unavailable:
            reason_codes.append("SOURCE_DEGRADED")
        if coverage.assessment_supported and (coverage.source_degraded or calendar_unavailable):
            reason_codes.append("PARTIAL_OFFICIAL_SOURCE_COVERAGE")
        if calendar_unavailable and coverage.assessment_supported:
            reason_codes.append("EVENT_CALENDAR_UNAVAILABLE")
        if not coverage.assessment_supported:
            reason_codes.append("ALL_OFFICIAL_SOURCES_UNAVAILABLE")
            reason_codes.append("ASSESSMENT_UNAVAILABLE")
        summaries = [*calendar_result.news_summary]
        summaries.extend(fact.text for fact in coverage.summary_facts)
        if coverage.assessment_supported and (coverage.source_degraded or calendar_unavailable):
            summaries.append(
                "Some official macro sources or coverage slots are unavailable; "
                "this assessment uses only the eligible first-party evidence cited below."
            )
        elif not coverage.assessment_supported:
            summaries.append(
                "No eligible first-party official macro evidence is currently available."
            )
        return MacroAssessment(
            risk_level=risk.risk_level,
            risk_score=risk.risk_level.score,
            decision_as_of_utc=decision_as_of_utc,
            reason_codes=tuple(dict.fromkeys(reason_codes)),
            news_summary=tuple(dict.fromkeys(summaries)),
            citations=citations,
            claims=coverage.accepted_claims,
            coverage=coverage.results,
            summary_facts=coverage.summary_facts,
            score_components=risk.components,
            source_degraded=coverage.source_degraded or calendar_unavailable,
            market_probability_supported=risk.market_probability_supported,
            assessment_supported=coverage.assessment_supported,
        )

    def _rate_observation_citations(
        self,
        decision_as_of_utc: datetime,
    ) -> tuple[EvidenceCitation, ...]:
        observations = self.store.latest_observations_as_of(
            series_ids=self.rate_series_ids,
            as_of_utc=decision_as_of_utc,
        )
        citations = []
        for observation in observations:
            age_days = (decision_as_of_utc.date() - observation.observation_date).days
            if (
                age_days < 0
                or age_days > self.maximum_rate_staleness_days
                or not observation.initial_release_only
            ):
                continue
            citations.append(
                    EvidenceCitation(
                        doc_id=observation.observation_id,
                        provider_id=observation.provider_id,
                        metric_id=observation.metric_id or observation.series_id,
                        source_tier=observation.source_tier,
                        event_type="RATES",
                    title=(
                        f"FRED {observation.series_id} initial release "
                        f"for {observation.observation_date.isoformat()}"
                    ),
                    canonical_url=observation.source_url,
                        eligible_from_utc=observation.eligible_from_utc,
                        content_sha256=observation.source_payload_sha256,
                        retrieved_at_utc=observation.retrieved_at_utc,
                        first_seen_at_utc=observation.retrieved_at_utc,
                        revision_status=(
                            "INITIAL" if observation.initial_release_only else "UNKNOWN"
                        ),
                    score=1.0 / (1.0 + age_days),
                    evidence_kind="OBSERVATION",
                    series_id=observation.series_id,
                    observation_date=observation.observation_date,
                    observation_value=observation.value,
                )
            )
        return tuple(citations)

    def assess(
        self,
        *,
        decision_as_of_utc: datetime,
        calendar_result: MacroGateResult,
        required_event_types: tuple[str, ...] = ("FOMC", "RATES", "CPI", "PCE", "NFP"),
        replay_only: bool = False,
        calendar_claims: tuple[MacroClaim, ...] = (),
    ) -> MacroAssessment:
        if self.required_coverage is not None:
            return self._assess_claims(
                decision_as_of_utc=decision_as_of_utc,
                calendar_result=calendar_result,
                calendar_claims=calendar_claims,
            )
        citations = []
        missing = []
        rate_observations_insufficient = False
        try:
            for event_type in required_event_types:
                results = self.store.retrieve(
                    DEFAULT_QUERIES[event_type],
                    event_type=event_type,
                    as_of_utc=decision_as_of_utc,
                    top_k=self.top_k,
                    replay_only=replay_only,
                )
                if event_type == "RATES":
                    rate_results = self._rate_observation_citations(decision_as_of_utc)
                    rate_ready = len(rate_results) >= self.minimum_rate_series
                    if self.require_rate_observations and not rate_ready:
                        rate_observations_insufficient = True
                        missing.append(event_type)
                    elif rate_ready:
                        results = (*results, *rate_results)
                if not results:
                    missing.append(event_type)
                citations.extend(results)
        except Exception as error:
            return MacroAssessment(
                risk_level=MacroRiskLevel.CAUTION,
                risk_score=MacroRiskLevel.CAUTION.score,
                decision_as_of_utc=decision_as_of_utc,
                reason_codes=("EVIDENCE_RETRIEVAL_FAILED", type(error).__name__),
                news_summary=("官方宏观资料检索失败, 当前外部风险无法可靠评估。",),
                assessment_supported=False,
            )
        summaries = list(calendar_result.news_summary)
        for citation in citations:
            if citation.evidence_kind == "OBSERVATION":
                summaries.append(
                    f"{citation.series_id} 于 {citation.observation_date} 的官方初值为 "
                    f"{citation.observation_value}。"
                )
            else:
                summaries.append(f"{citation.event_type}: {citation.title}")
        summaries = list(dict.fromkeys(summaries))
        if missing:
            reason_codes = ["OFFICIAL_EVIDENCE_MISSING", *sorted(set(missing))]
            if rate_observations_insufficient:
                reason_codes.append("RATE_OBSERVATIONS_INSUFFICIENT_OR_STALE")
            summaries.append(
                "缺少可在当前时点引用的官方资料: "
                + "、".join(sorted(set(missing)))
                + "。"
            )
            return MacroAssessment(
                risk_level=MacroRiskLevel.CAUTION,
                risk_score=MacroRiskLevel.CAUTION.score,
                decision_as_of_utc=decision_as_of_utc,
                reason_codes=tuple(reason_codes),
                news_summary=tuple(summaries),
                citations=tuple(citations),
                assessment_supported=False,
            )
        return MacroAssessment(
            risk_level=calendar_result.risk_level,
            risk_score=calendar_result.risk_level.score,
            decision_as_of_utc=decision_as_of_utc,
            reason_codes=(
                "OFFICIAL_EVIDENCE_COMPLETE",
                *calendar_result.reason_codes,
            ),
            news_summary=tuple(summaries),
            citations=tuple(citations),
            assessment_supported=(
                "ASSESSMENT_UNAVAILABLE" not in calendar_result.reason_codes
            ),
        )
