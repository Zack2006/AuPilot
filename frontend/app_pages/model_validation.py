"""Two-part model trust page: live chain health and OOS/forward evidence."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend.api_client import AurumPilotAPIError, api
from frontend.business_theme import apply_business_theme
from frontend.components.gold_daily_chart import render_gold_daily_chart
from frontend.components.metric_cards import metric_row
from frontend.i18n import page_header, section_header, tr
from frontend.market_data import load_gold_cache


def _utc(value: str | None) -> str:
    if not value:
        return "-"
    return str(value).replace("T", " ").replace("+00:00", " UTC")


def _percent(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}%}"


def _date_only(value: str | None) -> str:
    return str(value).split("T", 1)[0] if value else "-"


def _pipeline_state(
    status: dict[str, Any],
    timeline: dict[str, Any],
) -> tuple[str, str]:
    ready = bool(
        status.get("action_probability_model_loaded")
        and timeline.get("hash_verification", {}).get("passed")
        and timeline.get("status")
        == "CURRENT_WITHIN_ONE_COMPLETE_BUCKET"
    )
    if ready:
        return (
            tr("validation_pipeline_current"),
            tr("validation_pipeline_current_help"),
        )
    return (
        tr("validation_pipeline_waiting"),
        tr("validation_pipeline_waiting_help"),
    )


apply_business_theme("model-validation")
page_header("validation_title", "validation_trust_body")

try:
    technical_status = api.get("technical/status")
    timeline = api.get("model-validation/mn18-timeline")
    history_payload, _ = load_gold_cache()
except (AurumPilotAPIError, KeyError, TypeError, ValueError):
    st.error(tr("validation_data_unavailable"), icon=":material/cloud_off:")
    st.info(tr("no_mock_fallback"), icon=":material/verified_user:")
    st.stop()

pipeline_value, pipeline_help = _pipeline_state(
    technical_status, timeline
)
model_loaded = bool(
    technical_status.get("action_probability_model_loaded")
)
model_value = (
    tr("validation_model_verified_candidate")
    if model_loaded
    else tr("validation_model_unavailable")
)
hash_state = timeline["hash_verification"]
unseen_daily = timeline.get("post_training_unseen_daily", {})

with st.container(border=True, key="validation_status_ribbon"):
    section_header(
        tr("validation_runtime_title"),
        tr("validation_runtime_help"),
    )
    metric_row(
        [
            (
                tr("validation_pipeline"),
                pipeline_value,
                pipeline_help,
            ),
            (
                tr("validation_latest_market_bucket"),
                _date_only(timeline.get("latest_complete_bucket")),
                tr("validation_latest_market_bucket_help"),
            ),
            (
                tr("validation_dual_model"),
                model_value,
                tr("validation_model_candidate_help"),
            ),
            (
                tr("validation_unseen_daily_record_count"),
                (
                    f"{int(unseen_daily.get('record_count', 0))} "
                    f"{tr('validation_days')} / "
                    f"{int(unseen_daily.get('action_count', 0))} "
                    f"{tr('validation_rebalances')}"
                ),
                tr("validation_unseen_daily_record_count_help"),
            ),
        ]
    )
    st.caption(
        " · ".join(
            [
                (
                    f"{tr('validation_unseen_latest_source')}: "
                    f"{unseen_daily.get('source_until') or '-'}"
                ),
                (
                    f"{tr('validation_timeline_hash')}: "
                    f"{hash_state['combined_timeline_sha256'][:12]}…"
                ),
                tr("validation_one_bucket_lag"),
                tr("validation_no_auto_execution"),
            ]
        )
    )

title_column, refresh_column = st.columns(
    [0.82, 0.18], vertical_alignment="center"
)
with title_column:
    section_header(
        tr("validation_oos_title"),
        tr("validation_oos_help"),
    )
with refresh_column:
    refresh_requested = st.button(
        tr("validation_refresh"),
        icon=":material/sync:",
        width="stretch",
        key="validation_refresh_timeline",
    )

if refresh_requested:
    try:
        with st.spinner(tr("validation_refreshing")):
            timeline = api.post(
                "model-validation/mn18-timeline/refresh",
                timeout=900,
            )
            history_payload, _ = load_gold_cache()
        refresh = timeline["refresh"]
        st.success(
            tr("validation_refresh_complete").format(
                runs=refresh["model_runs_created"],
                records=refresh["forward_records_registered"],
                unseen=refresh[
                    "post_training_unseen_daily_records_created"
                ],
                market=refresh["market_complete_buckets_added"],
                signals=refresh[
                    "post_training_unseen_daily_qualified_signals_created"
                ],
                actions=refresh[
                    "post_training_unseen_daily_actions_created"
                ],
                pending=timeline.get(
                    "post_training_unseen_daily", {}
                ).get("pending_action_count", 0),
                realized=refresh[
                    "portfolio_rebalances_realized_total"
                ],
                latest=refresh["latest_complete_bucket_after"],
                buckets=(
                    ", ".join(refresh["source_buckets_processed"])
                    or tr("validation_no_new_bucket")
                ),
            ),
            icon=":material/check_circle:",
        )
    except (AurumPilotAPIError, KeyError, TypeError, ValueError):
        st.error(
            tr("validation_refresh_failed"),
            icon=":material/sync_problem:",
        )

oos = timeline["oos"]
summary = timeline.get("combined_summary", oos["summary"])
st.badge(
    tr("validation_combined_evidence_badge"),
    icon=":material/science:",
    color="orange",
)
metric_row(
    [
        (
            tr("validation_model_return"),
            _percent(summary["strategy_total_return"]),
            tr("validation_model_return_help"),
        ),
        (
            tr("validation_buy_hold_return"),
            _percent(summary["benchmark_total_return"]),
            tr("validation_buy_hold_return_help"),
        ),
        (
            tr("validation_return_lift"),
            (
                f"{float(summary['absolute_return_lift_vs_buy_hold']):+.2%}"
            ),
            tr("validation_return_lift_help"),
        ),
        (
            tr("validation_max_drawdown"),
            (
                f"{_percent(summary['strategy_max_drawdown'])} / "
                f"{_percent(summary['benchmark_max_drawdown'])}"
            ),
            tr("validation_max_drawdown_help"),
        ),
        (
            tr("validation_signal_fill_count"),
            (
                f"{int(summary['signal_rows'])} / "
                f"{int(summary['filled_trades'])}"
            ),
            tr("validation_signal_fill_count_help"),
        ),
    ]
)

with st.container(border=True, key="validation_chart_panel"):
    marker_column, boundary_column, note_column = st.columns(
        [0.24, 0.28, 0.48],
        vertical_alignment="center",
    )
    with marker_column:
        show_markers = st.toggle(
            tr("validation_show_historical_signals"),
            value=True,
            key="validation_show_historical_signals",
        )
    with boundary_column:
        show_position_bound_signals = st.toggle(
            tr("validation_show_position_bound_signals"),
            value=True,
            disabled=not show_markers,
            key="validation_show_position_bound_signals",
        )
    with note_column:
        st.caption(tr("validation_marker_toggle_help"))
    all_markers = [
        *timeline["historical_markers"],
        *timeline.get(
            "post_training_unseen_daily_markers",
            [],
        ),
        *timeline["forward_markers"],
    ]
    markers = []
    if show_markers:
        markers = (
            all_markers
            if show_position_bound_signals
            else [
                marker
                for marker in all_markers
                if marker.get("action") != "HOLD"
            ]
        )
    render_gold_daily_chart(
        history_payload,
        key_prefix="model_validation",
        historical_signals=markers,
        comparison_curve=timeline["comparison_curve"],
        validation_boundary={
            "date": timeline["oos_cutoff"],
            "label": tr("validation_oos_boundary_label"),
        },
        initial_bars=620,
        initial_end_date=timeline["latest_complete_bucket"],
        show_indicators=False,
    )
    st.caption(
        tr("validation_chart_contract").format(
            start=oos["date_from"],
            end=oos["date_until"],
            cost=oos["transaction_cost_bps_per_side"],
        )
    )
    st.caption(tr("validation_cost_policy"))
    st.caption(
        tr("validation_phase_contract").format(
            cutoff=timeline["oos_cutoff"],
            unseen_start=(
                unseen_daily.get("source_from")
                or tr("validation_not_started")
            ),
            unseen_end=(
                unseen_daily.get("source_until")
                or tr("validation_not_started")
            ),
        )
    )

st.caption(tr("disclaimer"))
