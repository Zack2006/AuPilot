"""Independent official macro-rate risk page."""

import streamlit as st

from frontend.api_client import AurumPilotAPIError, api
from frontend.business_theme import apply_business_theme
from frontend.components.risk_badge import risk_badge, risk_scale_legend
from frontend.i18n import (
    localize,
    localize_content,
    localize_macro_claim,
    localize_macro_component,
    localize_macro_datetime,
    localize_macro_interpretation,
    localize_macro_interpretation_status,
    localize_macro_interpretation_title,
    localize_macro_official_fact,
    localize_macro_provider,
    localize_macro_reason,
    localize_macro_slot,
    page_header,
    section_header,
    tr,
)


def _claim_ids_caption(claim_ids: list[str]) -> None:
    rendered = ", ".join(f"`{claim_id}`" for claim_id in claim_ids)
    st.caption(f"{tr('macro_claim_ids')}: {rendered}")


apply_business_theme("macro-radar")
page_header("macro_title", "macro_body")
try:
    with st.spinner(tr("loading_macro"), show_time=False):
        assessment = api.get("macro/assessment/latest")
        calendar = api.get("macro/events")
        source_status = api.get("macro/status")
except AurumPilotAPIError:
    st.error(tr("connection_error"), icon=":material/cloud_off:")
    st.stop()

status_label = tr("macro_supported") if assessment["assessment_supported"] else tr("macro_unsupported")
status_icon = ":material/check_circle:" if assessment["assessment_supported"] else ":material/block:"
with st.container(border=True, key="macro_risk_summary"):
    risk_summary, source_summary = st.columns([1, 1.35], gap="large")
    with risk_summary:
        st.caption(tr("macro_current_level"))
        risk_badge(
            assessment["risk_level"],
            assessment["risk_score"],
            assessment_supported=assessment["assessment_supported"],
        )
        st.status(
            status_label,
            state="complete" if assessment["assessment_supported"] else "error",
            expanded=False,
        )
    with source_summary:
        st.caption(tr("macro_source_health"))
        st.markdown(f"**{localize(source_status['status'])}**")
        st.caption(tr("macro_informational_disclaimer"))
        if assessment["risk_level"] == "Cancel":
            st.caption(tr("macro_cancel_supported"))
        elif not assessment["assessment_supported"]:
            st.caption(tr("macro_caution_unsupported"))

    risk_scale_legend(
        assessment["risk_scale"],
        current_level=assessment["risk_level"],
        assessment_supported=assessment["assessment_supported"],
    )

    coverage_column, components_column = st.columns(2, gap="medium")
    with coverage_column:
        with st.expander(tr("macro_coverage"), expanded=False):
            for provider in source_status.get("providers", []):
                label = localize_macro_provider(provider["provider_id"])
                if not provider.get("required"):
                    label = f"{label} ({tr('macro_optional_source')})"
                st.caption(f"{label}: {localize(provider['status'])}")
            for slot in source_status.get("coverage", []):
                requirement = tr("macro_required_slot") if slot.get("required", True) else tr("macro_optional_slot")
                st.caption(f"{localize_macro_slot(slot['slot'])} | {requirement}: {localize(slot['status'])}")
                if slot.get("supporting_claim_ids"):
                    _claim_ids_caption(slot["supporting_claim_ids"])

    components = assessment.get("score_components")
    with components_column:
        if components:
            with st.expander(tr("macro_score_components"), expanded=False):
                for name, value in components.items():
                    st.caption(f"{localize_macro_component(name)}: {value}")

claims_by_id = {claim["claim_id"]: claim for claim in assessment.get("claims", [])}

left, right = st.columns([1.25, 1], gap="large")
with left:
    section_header(tr("summary"))
    st.caption(tr("macro_interpretation_disclaimer"))
    interpretations = assessment.get("interpretations", [])
    facts = assessment.get("summary_facts", [])
    if interpretations:
        for item in interpretations:
            with st.container(border=True):
                st.markdown(
                    f"**{localize(item['event_type'])} · "
                    f"{localize_macro_interpretation_title(item['topic_key'])}**"
                )
                st.caption(localize_macro_interpretation_status(item["interpretation_status"]))
                st.markdown(f"**{tr('macro_official_fact')}**")
                st.write(localize_macro_official_fact(item, claims_by_id))
                seen_urls = set()
                for claim_id in item["claim_ids"]:
                    claim = claims_by_id.get(claim_id)
                    canonical_url = None if not claim else claim.get("canonical_url")
                    if not claim or not canonical_url or canonical_url in seen_urls:
                        continue
                    seen_urls.add(canonical_url)
                    source_name = localize_macro_provider(claim.get("provider_id", tr("source")))
                    st.markdown(
                        f"[{tr('macro_official_source')} · {source_name}]"
                        f"({canonical_url})"
                    )
                st.markdown(f"**{tr('macro_assisted_interpretation')}**")
                st.write(localize_macro_interpretation(item, claims_by_id))
                st.caption(
                    f"{tr('macro_interpretation_audit')} · {item['method']}"
                )
                _claim_ids_caption(item["claim_ids"])
    elif facts:
        for fact in facts:
            claim = next((claims_by_id.get(claim_id) for claim_id in fact["claim_ids"] if claim_id in claims_by_id), None)
            st.write(localize_macro_claim(claim, fact["text"]))
            _claim_ids_caption(fact["claim_ids"])
    elif assessment["assessment_supported"]:
        for summary in assessment["news_summary"]:
            st.write(localize_content(summary))
    else:
        st.write(tr("macro_summary_unavailable"))
    section_header(tr("macro_citations"))
    if not assessment["citations"]:
        st.caption(tr("macro_no_citations"))
    for citation in assessment["citations"]:
        with st.container(border=True):
            claim = next(
                (claims_by_id.get(claim_id) for claim_id in citation.get("claim_ids", []) if claim_id in claims_by_id),
                None,
            )
            title = localize_macro_claim(claim, citation["title"])
            st.markdown(f"**[{title}]({citation['canonical_url']})**")
            st.caption(f"{localize(citation['event_type'])} | {citation['doc_id']}")
            st.caption(f"{tr('macro_available_from')}: {citation['eligible_from_utc']}")
            st.caption(f"{tr('macro_source_tier')}: {citation.get('source_tier', 'A')}")
            if citation.get("published_at_utc"):
                st.caption(f"{tr('macro_published_at')}: {citation['published_at_utc']}")
            if citation.get("retrieved_at_utc"):
                st.caption(f"{tr('macro_retrieved_at')}: {citation['retrieved_at_utc']}")
            if citation.get("claim_ids"):
                _claim_ids_caption(citation["claim_ids"])
            if citation.get("evidence_kind") == "OBSERVATION":
                st.caption(
                    f"{citation['series_id']} | {citation['observation_date']} | "
                    f"{citation['observation_value']}"
                )
with right:
    section_header(tr("macro_calendar"))
    calendar_status = localize("supported") if calendar["fetch_succeeded"] else localize("unsupported")
    st.caption(f"{tr('macro_calendar_status')}: {calendar_status}")
    for event in calendar["items"][:5]:
        with st.container(border=True):
            st.markdown(f"**{localize(event['event_type'])}**")
            st.caption(localize_macro_datetime(event["scheduled_release_at_utc"]))
            if event.get("source_url"):
                st.markdown(f"[{tr('source')}]({event['source_url']})")
    if calendar.get("fresh_until_utc"):
        st.caption(f"{tr('macro_available_until')}: {calendar['fresh_until_utc']}")

section_header(tr("macro_reason_codes"))
st.caption(" | ".join(localize_macro_reason(code) for code in assessment["reason_codes"]))
st.info(tr("macro_informational_disclaimer"), icon=status_icon)
