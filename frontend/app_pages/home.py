"""Databento-only gold market page with shared UTC daily technical charts."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.api_client import AurumPilotAPIError, api
from frontend.business_theme import apply_business_theme
from frontend.components.gold_daily_chart import market_frame, render_gold_daily_chart
from frontend.components.metric_cards import metric_row
from frontend.i18n import localize, page_header, section_header, tr
from frontend.market_data import load_gold_cache, refresh_gold_cache


def _rsi_state(value: float) -> str:
    if value >= 70:
        return tr("overbought")
    if value <= 30:
        return tr("oversold")
    return tr("neutral_zone")


def _market_unavailable_copy() -> str:
    """Distinguish a missing cache from a missing Databento credential."""
    try:
        settings = api.get("settings/sources")
        databento = next(
            source for source in settings["sources"] if source["source_id"] == "databento"
        )
    except (AurumPilotAPIError, KeyError, StopIteration, TypeError):
        return tr("databento_setup_required")
    if databento.get("credential_status") == "CONFIGURED":
        return tr("databento_refresh_required")
    return tr("databento_setup_required")


apply_business_theme("gold-market")
page_header("overview_title", "overview_body")

try:
    with st.spinner(tr("loading_market"), show_time=False):
        history_payload, latest = load_gold_cache(include_latest=True)
        if latest is None:
            raise ValueError("Latest Databento bar is unavailable")
        frame = market_frame(history_payload)
except (AurumPilotAPIError, KeyError, TypeError, ValueError):
    st.error(tr("formal_market_unavailable"), icon=":material/cloud_off:")
    st.info(_market_unavailable_copy(), icon=":material/key:")
    if st.button(
        tr("refresh_market_data"),
        icon=":material/refresh:",
        key="refresh_databento_without_cache",
    ):
        try:
            with st.spinner(tr("refreshing_market_data"), show_time=True):
                refresh_gold_cache()
            st.session_state["databento_refresh_completed"] = True
            st.rerun()
        except (AurumPilotAPIError, KeyError, TypeError, ValueError):
            st.error(tr("market_refresh_failed"), icon=":material/cloud_off:")
    st.stop()

try:
    technical_forecast = api.get("technical/forecast/21-slots")
except AurumPilotAPIError:
    technical_forecast = None

if st.session_state.pop("databento_refresh_completed", False):
    st.success(tr("market_data_refreshed"), icon=":material/check_circle:")

recent_30 = frame.tail(min(30, len(frame)))
previous_close = frame["close"].iloc[-2] if len(frame) > 1 else frame["close"].iloc[-1]
daily_change = frame["close"].iloc[-1] / previous_close - 1
return_30 = recent_30["close"].iloc[-1] / recent_30["close"].iloc[0] - 1 if len(recent_30) > 1 else 0.0
latest_indicators = frame.iloc[-1]

with st.container(key="market_kpi_ribbon"):
    metric_row([
        (tr("current_price"), f"${latest['close']:,.2f}"),
        (tr("daily_change"), f"{daily_change:+.2%}"),
        (tr("market_status"), localize(latest["market_regime"])),
        ("RSI(14)", f"{latest_indicators['rsi_14']:.1f} · {_rsi_state(latest_indicators['rsi_14'])}"),
    ])
    metric_row([
        (tr("return_30"), f"{return_30:+.2%}"),
        (tr("high_30"), f"${recent_30['high'].max():,.2f}"),
        (tr("low_30"), f"${recent_30['low'].min():,.2f}"),
        (tr("volatility_20"), f"{latest_indicators['volatility_20']:.2%}"),
        (tr("max_drawdown"), f"{frame['drawdown_from_high'].min():.2%}"),
    ])

with st.container(border=True, key="market_chart_panel"):
    chart_title, chart_action = st.columns([3, 1], vertical_alignment="bottom")
    with chart_title:
        section_header(tr("daily_candlestick"), tr("price_trend_help"))
    with chart_action:
        refresh_requested = st.button(
            tr("refresh_market_data"),
            icon=":material/refresh:",
            key="refresh_databento_gold_market",
            width="stretch",
        )

    if refresh_requested:
        try:
            with st.spinner(tr("refreshing_market_data"), show_time=True):
                refresh_gold_cache()
            st.session_state["databento_refresh_completed"] = True
            st.rerun()
        except (AurumPilotAPIError, KeyError, TypeError, ValueError):
            st.error(tr("market_refresh_failed"), icon=":material/cloud_off:")

    period_label = st.segmented_control(
        tr("initial_chart_window"),
        [tr("days_30"), tr("days_90"), tr("days_180"), tr("days_365")],
        default=tr("days_180"),
        key="home_period",
    )
    initial_bars = {
        tr("days_30"): 30,
        tr("days_90"): 90,
        tr("days_180"): 180,
        tr("days_365"): 365,
    }[period_label]
    render_gold_daily_chart(
        history_payload,
        key_prefix="gold_market",
        technical_forecast=technical_forecast,
        initial_bars=initial_bars,
        show_indicators=True,
    )

with st.container(key="market_technical_panel"):
    section_header(tr("latest_technical_snapshot"))
    metric_row([
        (
            f"BOLL (20, 2) · {tr('boll_lower_mid_upper')} · USD/oz",
            f"{latest_indicators['boll_lower']:,.2f} · "
            f"{latest_indicators['boll_mid']:,.2f} · {latest_indicators['boll_upper']:,.2f}",
        ),
        (
            "MACD (12, 26, 9)",
            f"DIF {latest_indicators['macd_dif']:+.3f} · "
            f"DEA {latest_indicators['macd_dea']:+.3f} · "
            f"HIST {latest_indicators['macd_hist']:+.3f}",
        ),
        ("RSI (14)", f"{latest_indicators['rsi_14']:.1f} · {_rsi_state(latest_indicators['rsi_14'])}"),
    ])
    metric_row([
        (
            "KDJ (9, 3, 3)",
            f"K {latest_indicators['kdj_k']:.1f} · D {latest_indicators['kdj_d']:.1f} · "
            f"J {latest_indicators['kdj_j']:.1f}",
        ),
        ("ATR (14)", f"${latest_indicators['atr_14']:,.2f}"),
        (tr("volatility_20"), f"{latest_indicators['volatility_20']:.2%}"),
        (tr("drawdown_chart"), f"{latest_indicators['drawdown_from_high']:.2%}"),
    ])
    st.caption(tr("technical_snapshot_note"))

metadata = history_payload["metadata"]
as_of = pd.Timestamp(metadata["data_until_utc"]).strftime("%Y-%m-%d UTC")
retrieved = pd.Timestamp(metadata["retrieved_at_utc"]).strftime("%Y-%m-%d %H:%M UTC")
st.caption(
    f"{tr('data_source')}: Databento · {metadata['dataset']} · {metadata['symbol']} · "
    f"{metadata['schema_name']} · {tr('market_as_of')}: {as_of} · {tr('retrieved_at')}: {retrieved} · "
    f"SHA256: {metadata['content_sha256'][:12]}"
)
st.info(tr("disclaimer"), icon=":material/shield:")
