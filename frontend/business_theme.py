"""Scoped visual system for the four AuPilot business pages.

The homepage intentionally never imports this module.  Every selector is also
guarded by the page marker so the established homepage composition remains
byte-for-byte and visually isolated from business-page iteration.
"""

from __future__ import annotations

from html import escape

import streamlit as st


_BUSINESS_THEME_CSS = r"""
<style>
:root {
  --ap-ink: #171816;
  --ap-ink-soft: #3f423d;
  --ap-muted: #747871;
  --ap-faint: #a5a9a1;
  --ap-canvas: #f6f7f5;
  --ap-surface: rgba(255, 255, 255, .92);
  --ap-surface-solid: #ffffff;
  --ap-line: rgba(32, 35, 31, .09);
  --ap-line-strong: rgba(32, 35, 31, .16);
  --ap-gold: #b38a3e;
  --ap-gold-deep: #755821;
  --ap-gold-soft: #f5eddd;
  --ap-shadow-sm: 0 1px 2px rgba(20, 22, 19, .025), 0 8px 24px rgba(20, 22, 19, .035);
  --ap-shadow-md: 0 2px 4px rgba(20, 22, 19, .035), 0 18px 48px rgba(20, 22, 19, .07);
  --ap-ease: cubic-bezier(.22, .8, .32, 1);
}

.ap-business-page {
  width: 0;
  height: 0;
  margin: 0;
  padding: 0;
  overflow: hidden;
}

.stApp:has(.ap-business-page) [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 82% -12%, rgba(196, 160, 87, .10), transparent 31rem),
    radial-gradient(circle at -8% 26%, rgba(113, 137, 155, .07), transparent 30rem),
    linear-gradient(180deg, #fbfcfa 0, var(--ap-canvas) 28rem, #f8f8f6 100%);
  color: var(--ap-ink);
}

.stApp:has(.ap-business-page) header[data-testid="stHeader"] {
  background: rgba(250, 251, 249, .78);
  border-bottom: 1px solid rgba(32, 35, 31, .06);
  backdrop-filter: blur(18px) saturate(135%);
  -webkit-backdrop-filter: blur(18px) saturate(135%);
}

.stApp:has(.ap-business-page) [data-testid="stTopNav"] {
  background: transparent;
}

.stApp:has(.ap-business-page) [data-testid="stTopNav"] a {
  border-radius: 999px;
  color: #60645e;
  transition: color .18s ease, background-color .18s ease, transform .18s var(--ap-ease);
}

.stApp:has(.ap-business-page) [data-testid="stTopNav"] a:hover {
  color: var(--ap-ink);
  background: rgba(179, 138, 62, .08);
  transform: translateY(-1px);
}

.stApp:has(.ap-business-page) [data-testid="stTopNav"] a[aria-current="page"] {
  color: var(--ap-gold-deep);
  background: var(--ap-gold-soft);
  box-shadow: inset 0 0 0 1px rgba(179, 138, 62, .16);
}

.stMainBlockContainer:has(.ap-business-page) {
  max-width: 1360px;
  padding-top: 2.25rem;
  padding-bottom: 5rem;
}

.stMainBlockContainer:has(.ap-business-page) > div[data-testid="stVerticalBlock"] {
  gap: 1.05rem;
}

.stMainBlockContainer:has(.ap-business-page) h1 {
  position: relative;
  margin: 0;
  padding-top: .88rem;
  color: var(--ap-ink);
  font-size: clamp(2.1rem, 4vw, 3.35rem);
  font-weight: 620;
  letter-spacing: -.052em;
  line-height: 1.02;
}

.stMainBlockContainer:has(.ap-business-page) h1::before {
  content: "";
  position: absolute;
  top: 0;
  left: .08rem;
  width: 2.7rem;
  height: 3px;
  border-radius: 999px;
  background: linear-gradient(90deg, #8f6b29, #d7bb7a);
  box-shadow: 0 0 0 4px rgba(179, 138, 62, .055);
}

.stMainBlockContainer:has(.ap-business-page) h2,
.stMainBlockContainer:has(.ap-business-page) h3 {
  color: var(--ap-ink);
  letter-spacing: -.027em;
}

.stMainBlockContainer:has(.ap-business-page) h2 {
  margin-top: 1.3rem;
  font-size: clamp(1.35rem, 2.2vw, 1.82rem);
  font-weight: 610;
}

.stMainBlockContainer:has(.ap-business-page) h3 {
  font-weight: 610;
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stCaptionContainer"] {
  color: var(--ap-muted);
  font-size: .81rem;
  line-height: 1.65;
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stWidgetLabel"] {
  display: flex;
  align-items: center;
  min-height: 1.5rem;
  margin: 0 0 .3rem;
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stWidgetLabel"] p {
  margin: 0;
  line-height: 1.35;
}

.stMainBlockContainer:has(.ap-business-page) h1 + div [data-testid="stCaptionContainer"],
.stMainBlockContainer:has(.ap-business-page) h1 + [data-testid="stCaptionContainer"] {
  max-width: 76rem;
  color: #6b6f68;
  font-size: .94rem;
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stMetric"] {
  min-width: 10.75rem;
  padding: 1rem 1.1rem 1.05rem;
  border: 1px solid var(--ap-line) !important;
  border-radius: 16px;
  background:
    linear-gradient(145deg, rgba(255,255,255,.97), rgba(252,252,249,.90));
  box-shadow: var(--ap-shadow-sm);
  transition:
    transform .22s var(--ap-ease),
    box-shadow .22s ease,
    border-color .22s ease;
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stMetric"]:hover {
  transform: translateY(-3px);
  border-color: rgba(179, 138, 62, .25) !important;
  box-shadow: var(--ap-shadow-md);
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stMetricLabel"] {
  color: var(--ap-muted);
  font-size: .76rem;
  letter-spacing: .015em;
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stMetricValue"] {
  color: var(--ap-ink);
  font-size: clamp(1.15rem, 2vw, 1.55rem);
  font-weight: 610;
  letter-spacing: -.025em;
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stVerticalBlockBorderWrapper"] {
  overflow: hidden;
  border: 1px solid var(--ap-line) !important;
  border-radius: 19px !important;
  background: var(--ap-surface);
  box-shadow: var(--ap-shadow-sm);
  transition: border-color .22s ease, box-shadow .22s ease, transform .22s var(--ap-ease);
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: rgba(179, 138, 62, .18) !important;
  box-shadow: var(--ap-shadow-md);
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stAlert"] {
  border: 1px solid rgba(89, 91, 85, .09);
  border-radius: 14px;
  background: rgba(255, 255, 255, .86) !important;
  box-shadow: 0 8px 24px rgba(20, 22, 19, .025);
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stAlertContainer"] {
  color: #4f544d;
  background: rgba(255, 255, 255, .88) !important;
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stAlert"] p {
  line-height: 1.55;
}

.stMainBlockContainer:has(.ap-business-page) button {
  border-radius: 11px;
  border-color: rgba(44, 46, 42, .15);
  font-weight: 580;
  transition:
    transform .18s var(--ap-ease),
    border-color .18s ease,
    background-color .18s ease,
    box-shadow .18s ease;
}

.stMainBlockContainer:has(.ap-business-page) button:hover {
  transform: translateY(-1px);
  border-color: rgba(179, 138, 62, .42);
  box-shadow: 0 8px 22px rgba(35, 31, 21, .08);
}

.stMainBlockContainer:has(.ap-business-page) button[kind="primary"],
.stMainBlockContainer:has(.ap-business-page) [data-testid="stFormSubmitButton"] button {
  color: #fff;
  border-color: #7b5d25;
  background: linear-gradient(135deg, #8a682a, #b38a3e);
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stSegmentedControl"] {
  padding: .22rem;
  border: 1px solid var(--ap-line);
  border-radius: 13px;
  background: rgba(234, 235, 231, .62);
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stButtonGroup"] {
  padding: 0;
  border: 0;
  border-radius: 0;
  background: transparent;
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stButtonGroup"] [role="radiogroup"] {
  width: fit-content;
  padding: .22rem;
  border: 1px solid var(--ap-line);
  border-radius: 13px;
  background: rgba(234, 235, 231, .62);
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stSegmentedControl"] button,
.stMainBlockContainer:has(.ap-business-page) [data-testid="stButtonGroup"] [role="radio"] {
  border: 0;
  border-radius: 9px;
  background: transparent !important;
  box-shadow: none;
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
  color: var(--ap-gold-deep);
  background: #fff;
  box-shadow: 0 1px 5px rgba(26, 28, 24, .09);
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] {
  color: var(--ap-gold-deep);
  background: #fff !important;
  box-shadow: 0 1px 5px rgba(26, 28, 24, .09);
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stButtonGroup"] [role="radio"][aria-checked="true"] {
  color: var(--ap-gold-deep);
  background: #fff !important;
  box-shadow: 0 1px 5px rgba(26, 28, 24, .09);
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stExpander"] {
  overflow: hidden;
  border: 1px solid var(--ap-line);
  border-radius: 14px;
  background: rgba(255, 255, 255, .72);
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stDataFrame"],
.stMainBlockContainer:has(.ap-business-page) iframe {
  overflow: hidden;
  border-radius: 16px;
  box-shadow: 0 12px 36px rgba(20, 22, 19, .055);
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stTextInput"] input,
.stMainBlockContainer:has(.ap-business-page) [data-testid="stNumberInput"] input,
.stMainBlockContainer:has(.ap-business-page) [data-testid="stDateInput"] input,
.stMainBlockContainer:has(.ap-business-page) [data-baseweb="select"] > div {
  border-color: var(--ap-line-strong);
  border-radius: 11px;
  background: rgba(255, 255, 255, .9);
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stMultiSelect"] input {
  min-width: 2.75rem;
  margin: 0;
  padding: 0 .2rem;
  border: 0 !important;
  border-radius: 0;
  outline: 0;
  background: transparent !important;
  box-shadow: none !important;
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stMultiSelect"] [data-baseweb="tag"] {
  flex: 0 0 auto;
  min-height: 1.75rem;
  margin: .2rem .25rem .2rem 0;
  border-radius: 999px;
  line-height: 1.15;
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stMultiSelect"] [data-baseweb="tag"] > span,
.stMainBlockContainer:has(.ap-business-page) [data-testid="stMultiSelect"] [data-baseweb="tag"] > div {
  display: inline-flex;
  align-items: center;
  line-height: 1.15;
}

.stMainBlockContainer:has(.ap-business-page) a {
  color: #7c5c22;
  text-decoration-color: rgba(124, 92, 34, .32);
  text-underline-offset: .18em;
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stStatusWidget"] {
  border-radius: 14px;
  background: rgba(255, 255, 255, .82);
  border-color: var(--ap-line);
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stBadge"] {
  border-radius: 999px;
  font-weight: 620;
}

@keyframes ap-page-arrive {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}

@keyframes ap-card-arrive {
  from { opacity: 0; transform: translateY(5px) scale(.995); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}

.stMainBlockContainer:has(.ap-business-page) h1 {
  animation: ap-page-arrive .48s var(--ap-ease) both;
}

.stMainBlockContainer:has(.ap-business-page) [data-testid="stMetric"],
.stMainBlockContainer:has(.ap-business-page) [data-testid="stVerticalBlockBorderWrapper"],
.stMainBlockContainer:has(.ap-business-page) [data-testid="stAlert"] {
  animation: ap-card-arrive .38s var(--ap-ease) both;
}

.stMainBlockContainer:has(.ap-page-gold-market) [data-testid="stMetricValue"] {
  font-variant-numeric: tabular-nums;
}

.stMainBlockContainer:has(.ap-page-gold-market) .st-key-market_kpi_ribbon {
  position: relative;
  padding-top: .35rem;
}

.stMainBlockContainer:has(.ap-page-gold-market) .st-key-market_chart_panel {
  margin-top: .65rem;
  border-color: rgba(179, 138, 62, .14) !important;
  background: rgba(255,255,255,.95);
}

.stMainBlockContainer:has(.ap-page-gold-market) .st-key-market_technical_panel {
  padding: .4rem 0 0;
}

.stMainBlockContainer:has(.ap-page-decision-assistant) [data-testid="stVerticalBlockBorderWrapper"]:has(.probability-overview) {
  border-color: rgba(179, 138, 62, .20) !important;
}

.stMainBlockContainer:has(.ap-page-decision-assistant) .st-key-decision_today_card {
  margin-top: .65rem;
  border-color: rgba(179, 138, 62, .14) !important;
  background: rgba(255,255,255,.95);
}

.stMainBlockContainer:has(.ap-page-decision-assistant) .st-key-decision_macro_summary {
  border-color: rgba(32, 35, 31, .07) !important;
  background: linear-gradient(115deg, rgba(247,249,246,.95), rgba(255,255,255,.95));
}

.stMainBlockContainer:has(.ap-page-macro-radar) .macro-risk-scale {
  margin-top: .55rem;
}

.stMainBlockContainer:has(.ap-page-macro-radar) .st-key-macro_risk_summary {
  position: relative;
  border-color: rgba(179, 138, 62, .16) !important;
  background:
    radial-gradient(circle at 92% -35%, rgba(179,138,62,.09), transparent 26rem),
    rgba(255,255,255,.94);
}

.stMainBlockContainer:has(.ap-page-macro-radar) .st-key-macro_risk_summary::after {
  content: "";
  position: absolute;
  top: 1.2rem;
  right: 1.35rem;
  width: 6px;
  height: 6px;
  border-radius: 999px;
  background: #b38a3e;
  box-shadow: 0 0 0 5px rgba(179,138,62,.08);
}

.stMainBlockContainer:has(.ap-page-model-validation) [data-testid="stDataFrame"] {
  border: 1px solid var(--ap-line);
}

.stMainBlockContainer:has(.ap-page-model-validation) .st-key-validation_status_ribbon {
  padding-top: .3rem;
}

.stMainBlockContainer:has(.ap-page-model-validation) .st-key-validation_chart_panel {
  border-color: rgba(179, 138, 62, .13) !important;
  background: rgba(255,255,255,.95);
}

.stMainBlockContainer:has(.ap-page-model-validation) .st-key-validation_performance_panel {
  background: linear-gradient(120deg, rgba(255,255,255,.96), rgba(248,249,247,.92));
}

.stMainBlockContainer:has(.ap-page-model-validation) .st-key-validation_issuance_card {
  border-left: 3px solid rgba(179, 138, 62, .56) !important;
}

@media (max-width: 760px) {
  .stMainBlockContainer:has(.ap-business-page) {
    padding-top: 1.6rem;
    padding-left: 1rem;
    padding-right: 1rem;
    overflow-x: clip;
  }

  .stMainBlockContainer:has(.ap-business-page) h1 {
    font-size: 2.15rem;
    overflow-wrap: anywhere;
  }

  .stMainBlockContainer:has(.ap-business-page) [data-testid="stHorizontalBlock"],
  .stMainBlockContainer:has(.ap-business-page) [data-testid="stColumn"],
  .stMainBlockContainer:has(.ap-business-page) [data-testid="stVerticalBlock"],
  .stMainBlockContainer:has(.ap-business-page) [data-testid="stVerticalBlockBorderWrapper"] {
    min-width: 0;
    max-width: 100%;
  }

  .stMainBlockContainer:has(.ap-business-page) [data-testid="stMetric"] {
    min-width: min(100%, 9rem);
    max-width: 100%;
    padding: .85rem .9rem;
  }

  .stMainBlockContainer:has(.ap-business-page) [data-testid="stMetricValue"],
  .stMainBlockContainer:has(.ap-business-page) [data-testid="stMetricLabel"],
  .stMainBlockContainer:has(.ap-business-page) [data-testid="stAlert"] p,
  .stMainBlockContainer:has(.ap-business-page) [data-testid="stCaptionContainer"] {
    max-width: 100%;
    overflow-wrap: anywhere;
  }

  .stMainBlockContainer:has(.ap-business-page) [data-testid="stSegmentedControl"],
  .stMainBlockContainer:has(.ap-business-page) [data-testid="stButtonGroup"] [role="radiogroup"] {
    max-width: 100%;
    overflow-x: auto;
    overscroll-behavior-inline: contain;
    scrollbar-width: thin;
  }

  .stMainBlockContainer:has(.ap-business-page) [data-testid="stSegmentedControl"] button,
  .stMainBlockContainer:has(.ap-business-page) [data-testid="stButtonGroup"] [role="radio"] {
    min-height: 42px;
    white-space: nowrap;
  }

  .stMainBlockContainer:has(.ap-business-page) [data-testid="stDataFrame"],
  .stMainBlockContainer:has(.ap-business-page) [data-testid="stPlotlyChart"],
  .stMainBlockContainer:has(.ap-business-page) iframe {
    max-width: 100%;
  }

  .stMainBlockContainer:has(.ap-business-page) button,
  .stMainBlockContainer:has(.ap-business-page) [data-testid="stPageLink"] a {
    min-height: 42px;
  }
}

@media (max-width: 420px) {
  .stMainBlockContainer:has(.ap-business-page) h1 {
    font-size: 1.9rem;
  }
}

@media (prefers-reduced-motion: reduce) {
  .stApp:has(.ap-business-page) *,
  .stApp:has(.ap-business-page) *::before,
  .stApp:has(.ap-business-page) *::after {
    animation-duration: .001ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: .001ms !important;
  }
}
</style>
"""


def apply_business_theme(page_id: str) -> None:
    """Apply the isolated business-page visual system."""

    safe_page_id = "".join(
        character for character in page_id.lower() if character.isalnum() or character == "-"
    )
    st.html(
        _BUSINESS_THEME_CSS
        + (
            '<div class="ap-business-page '
            f'ap-page-{escape(safe_page_id)}" aria-hidden="true"></div>'
        )
    )
