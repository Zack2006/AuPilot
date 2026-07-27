"""Static AuPilot product homepage and local navigation hub."""

from __future__ import annotations

from html import escape

import streamlit as st

from frontend.i18n import tr


PRODUCTS = (
    ("01", "overview_nav", "homepage_gold_description", "homepage_open_gold", "app_pages/home.py", ":material/candlestick_chart:", "gold"),
    ("02", "decision_nav", "homepage_advice_description", "homepage_open_advice", "app_pages/decision.py", ":material/strategy:", "advice"),
    ("03", "macro_nav", "homepage_macro_description", "homepage_open_macro", "app_pages/macro.py", ":material/radar:", "macro"),
    ("04", "validation_nav", "homepage_validation_description", "homepage_open_validation", "app_pages/model_validation.py", ":material/fact_check:", "validation"),
)

TECHNICAL_STEPS = (
    ("01", "homepage_flow_data_title", "homepage_flow_data_body", "yellow"),
    ("02", "homepage_flow_state_title", "homepage_flow_state_body", "blue"),
    ("03", "homepage_flow_issuance_title", "homepage_flow_issuance_body", "pink"),
    ("04", "homepage_flow_action_title", "homepage_flow_action_body", "yellow"),
    ("05", "homepage_flow_outlook_title", "homepage_flow_outlook_body", "blue"),
    ("06", "homepage_flow_validation_title", "homepage_flow_validation_body", "pink"),
)

MACRO_STEPS = (
    ("01", "homepage_macro_source_title", "homepage_macro_source_body"),
    ("02", "homepage_macro_audit_title", "homepage_macro_audit_body"),
    ("03", "homepage_macro_risk_title", "homepage_macro_risk_body"),
)

GUARDRAILS = (
    ("01", "homepage_guardrail_causal_title", "homepage_guardrail_causal_body", "yellow"),
    ("02", "homepage_guardrail_issuance_title", "homepage_guardrail_issuance_body", "blue"),
    ("03", "homepage_guardrail_isolation_title", "homepage_guardrail_isolation_body", "pink"),
    ("04", "homepage_guardrail_execution_title", "homepage_guardrail_execution_body", "green"),
)


def _copy(key: str) -> str:
    return escape(tr(key))


HOMEPAGE_CSS = r"""
<style>
:root {
    --ap-ink: #101014;
    --ap-paper: #f4f4f1;
    --ap-white: #ffffff;
    --ap-muted: #666a72;
    --ap-line: #cfd1d5;
    --ap-yellow: #f3c536;
    --ap-blue: #3155f5;
    --ap-pink: #e54375;
    --ap-green: #0b8a67;
    --ap-font-sans: "Source Han Sans SC", "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", "WenQuanYi Micro Hei", sans-serif;
    --ap-font-mono: ui-monospace, SFMono-Regular, Consolas, "Source Han Sans SC", "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", monospace;
}

[data-testid="stAppViewContainer"] { background: var(--ap-paper); font-family: var(--ap-font-sans); }
[data-testid="stMainBlockContainer"] { max-width: 1240px; padding-top: 4.5rem; }

.ap-homepage-heading,
.ap-homepage-heading *,
.ap-hero-copy,
.ap-hero-copy *,
.ap-window-scene,
.ap-window-scene *,
.ap-technical-stage,
.ap-technical-stage *,
.ap-macro-stage,
.ap-macro-stage *,
.ap-guardrail-grid,
.ap-guardrail-grid *,
.ap-home-footer,
.ap-home-footer * {
    box-sizing: border-box;
    letter-spacing: 0;
}

.st-key-ap_home_hero {
    position: relative;
    margin: 0 0 26px;
    padding: 34px 34px 28px;
    overflow: hidden;
    color: var(--ap-ink);
    border-bottom: 1px solid var(--ap-ink);
}
.st-key-ap_home_hero > div { position: relative; z-index: 3; }
.ap-hero-layout {
    display: grid;
    grid-template-columns: minmax(0, 1.08fr) minmax(420px, .92fr);
    gap: clamp(28px, 4.5vw, 68px);
    align-items: center;
    min-height: 498px;
}
.ap-hero-copy {
    position: relative;
    z-index: 4;
    min-width: 0;
    max-width: 690px;
    padding: 42px 0 18px;
}
.ap-hero-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px 14px;
    margin-bottom: 28px;
}
.ap-hero-brand {
    margin: 0;
    color: var(--ap-ink);
    font-size: 20px;
    font-weight: 800;
}
.ap-hero-index {
    display: inline-flex;
    align-items: center;
    min-height: 26px;
    margin: 0;
    padding: 3px 9px;
    color: var(--ap-ink);
    background: var(--ap-yellow);
    border: 1px solid var(--ap-ink);
    border-radius: 3px;
    font-family: var(--ap-font-mono);
    font-size: 11px;
    font-weight: 750;
}
.ap-hero-copy h1 {
    max-width: 100%;
    margin: 0;
    color: var(--ap-ink);
    font-size: clamp(48px, 4.7vw, 64px);
    line-height: 1.03;
    font-weight: 780;
    overflow-wrap: normal;
    word-break: normal;
    text-wrap: balance;
}
.ap-hero-copy h1::after {
    display: block;
    width: 176px;
    height: 13px;
    margin-top: 20px;
    content: "";
    background: var(--ap-yellow);
    border: 1px solid var(--ap-ink);
}
.ap-hero-subtitle {
    max-width: 680px;
    margin: 22px 0 0;
    color: #383a40;
    font-size: 18px;
    line-height: 1.65;
}
.ap-hero-tags { display: flex; flex-wrap: wrap; gap: 8px; max-width: 100%; margin-top: 22px; }
.ap-hero-tags span {
    display: inline-flex;
    align-items: center;
    max-width: 100%;
    min-height: 30px;
    padding: 5px 10px;
    color: var(--ap-ink);
    background: rgba(255,255,255,.88);
    border: 1px solid var(--ap-ink);
    border-radius: 3px;
    font-size: 12px;
    font-weight: 700;
    line-height: 1.35;
    white-space: normal;
}
.ap-hero-tags span:nth-child(2) { background: #e8ebff; }
.ap-hero-tags span:nth-child(3) { background: #fde4ec; }

.ap-window-scene {
    position: relative;
    z-index: 1;
    min-width: 0;
    height: 440px;
    pointer-events: none;
}
.ap-window {
    position: absolute;
    overflow: hidden;
    background: rgba(255,255,255,.94);
    border: 1px solid var(--ap-ink);
    border-radius: 6px;
    box-shadow: 12px 12px 0 rgba(16,16,20,.10);
}
.ap-window.back { top: 42px; right: -2%; width: 92%; height: 330px; transform: translate(18px, -18px); opacity: .78; }
.ap-window.front { top: 86px; right: 4%; width: 96%; height: 322px; }
.ap-window-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 36px;
    padding: 0 12px;
    color: #474950;
    background: #eceef1;
    border-bottom: 1px solid var(--ap-ink);
    font-family: var(--ap-font-mono);
    font-size: 10px;
    font-weight: 700;
}
.ap-window-controls { display: flex; gap: 5px; }
.ap-window-controls i { display: block; width: 10px; height: 10px; border: 1px solid var(--ap-ink); }
.ap-window-controls i:nth-child(1) { background: var(--ap-pink); }
.ap-window-controls i:nth-child(2) { background: var(--ap-yellow); }
.ap-window-controls i:nth-child(3) { background: var(--ap-blue); }
.ap-window-body { position: relative; height: calc(100% - 36px); padding: 18px; }
.ap-window-grid {
    position: absolute;
    inset: 0;
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    opacity: .42;
}
.ap-window-grid i { border-left: 1px solid #c9cbd0; }
.ap-window-grid::before,
.ap-window-grid::after { position: absolute; right: 0; left: 0; content: ""; border-top: 1px solid #c9cbd0; }
.ap-window-grid::before { top: 34%; }
.ap-window-grid::after { top: 68%; }
.ap-window-copy { position: relative; z-index: 2; max-width: 72%; }
.ap-window-label { margin: 0; color: var(--ap-blue); font-family: var(--ap-font-mono); font-size: 10px; font-weight: 800; }
.ap-window-copy strong { display: block; margin-top: 9px; color: var(--ap-ink); font-size: 22px; line-height: 1.12; }
.ap-window-copy strong mark { color: var(--ap-ink); background: var(--ap-yellow); }
.ap-window-slots { position: absolute; right: 18px; bottom: 20px; left: 18px; z-index: 2; display: grid; grid-template-columns: repeat(21, 1fr); gap: 3px; align-items: end; height: 92px; }
.ap-window-slots i { display: block; height: var(--slot); background: var(--ap-blue); border: 1px solid var(--ap-ink); }
.ap-window-slots i:nth-child(4n) { background: var(--ap-yellow); }
.ap-window-slots i:nth-child(7n) { background: var(--ap-pink); }
.ap-scene-ticket {
    position: absolute;
    right: -5%;
    bottom: 28px;
    z-index: 3;
    width: 190px;
    padding: 12px 14px;
    color: var(--ap-white);
    background: var(--ap-ink);
    border: 1px solid var(--ap-ink);
    font-family: var(--ap-font-mono);
    font-size: 10px;
    line-height: 1.5;
}

.st-key-ap_home_hero [data-testid="stPageLink"] a {
    min-height: 46px;
    justify-content: center;
    color: var(--ap-white);
    font-weight: 750;
    background: var(--ap-ink);
    border: 1px solid var(--ap-ink);
    border-radius: 4px;
}
.st-key-ap_home_hero [data-testid="stPageLink"] a p,
.st-key-ap_home_hero [data-testid="stPageLink"] a span { color: var(--ap-white) !important; }
.st-key-ap_home_hero [data-testid="stPageLink"] a[href="home"] {
    color: var(--ap-ink);
    background: var(--ap-yellow);
}
.st-key-ap_home_hero [data-testid="stPageLink"] a[href="home"] p,
.st-key-ap_home_hero [data-testid="stPageLink"] a[href="home"] span { color: var(--ap-ink) !important; }
.st-key-ap_home_hero [data-testid="stPageLink"] a:hover,
.st-key-ap_home_hero [data-testid="stPageLink"] a:focus-visible {
    transform: translateY(-2px);
    box-shadow: 5px 5px 0 var(--ap-blue);
    outline: 2px solid var(--ap-ink);
    outline-offset: 2px;
}

.ap-homepage-heading { max-width: 800px; margin: 0 0 26px; }
.ap-homepage-kicker {
    margin: 0 0 10px;
    color: var(--ap-blue);
    font-family: var(--ap-font-mono);
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
}
.ap-homepage-heading h2 { margin: 0; color: var(--ap-ink); font-size: 38px; line-height: 1.14; }
.ap-homepage-heading p:last-child { margin: 12px 0 0; color: var(--ap-muted); font-size: 15px; line-height: 1.7; }

.ap-product-section { padding: 32px 0 18px; }
.st-key-ap_home_product_gold,
.st-key-ap_home_product_advice,
.st-key-ap_home_product_macro,
.st-key-ap_home_product_validation {
    position: relative;
    min-height: 220px;
    padding: 20px 20px 14px;
    overflow: hidden;
    background: var(--ap-white);
    border: 1px solid var(--ap-ink) !important;
    border-radius: 6px !important;
    box-shadow: 8px 8px 0 #dfe1e5;
}
.st-key-ap_home_product_gold { border-top: 7px solid var(--ap-yellow) !important; }
.st-key-ap_home_product_advice { border-top: 7px solid var(--ap-blue) !important; }
.st-key-ap_home_product_macro { border-top: 7px solid var(--ap-pink) !important; }
.st-key-ap_home_product_validation { border-top: 7px solid var(--ap-green) !important; }
.ap-product-head { display: flex; justify-content: space-between; align-items: center; gap: 16px; }
.ap-product-index { margin: 0; color: var(--ap-muted); font-family: var(--ap-font-mono); font-size: 11px; font-weight: 800; }
.ap-product-status { width: 54px; height: 10px; background: var(--ap-line); border: 1px solid var(--ap-ink); }
.ap-product-copy h3 { margin: 22px 0 0; color: var(--ap-ink); font-size: 24px; line-height: 1.2; }
.ap-product-copy p { min-height: 68px; margin: 12px 0 14px; color: #555961; font-size: 14px; line-height: 1.62; }
.st-key-ap_home_product_gold [data-testid="stPageLink"] a,
.st-key-ap_home_product_advice [data-testid="stPageLink"] a,
.st-key-ap_home_product_macro [data-testid="stPageLink"] a,
.st-key-ap_home_product_validation [data-testid="stPageLink"] a {
    min-height: 40px;
    color: var(--ap-ink);
    background: #f4f4f1;
    border: 1px solid var(--ap-ink);
    border-radius: 3px;
    font-weight: 700;
}

.ap-technical-stage { margin: 74px 0 0; padding: 42px 38px 38px; color: #f5f5f2; background: var(--ap-ink); border-radius: 6px; }
.ap-technical-stage .ap-homepage-kicker { color: var(--ap-yellow); }
.ap-technical-stage .ap-homepage-heading h2 { color: #ffffff; }
.ap-technical-stage .ap-homepage-heading p:last-child { color: #b7bac1; }
.ap-type-rail { display: flex; gap: 12px; margin: 4px -38px 30px; padding: 10px 38px; overflow: hidden; color: var(--ap-ink); background: var(--ap-yellow); border-top: 1px solid #fff; border-bottom: 1px solid #fff; white-space: nowrap; }
.ap-type-rail span { font-family: var(--ap-font-mono); font-size: 11px; font-weight: 800; }
.ap-type-rail b { color: var(--ap-blue); }
.ap-flow-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1px; background: #3b3d45; border: 1px solid #3b3d45; }
.ap-flow-node { position: relative; min-height: 188px; padding: 20px; background: #16171c; }
.ap-flow-node > span { display: inline-block; padding: 3px 7px; color: var(--ap-ink); background: var(--accent); font-family: var(--ap-font-mono); font-size: 10px; font-weight: 800; }
.ap-flow-node.yellow { --accent: var(--ap-yellow); }
.ap-flow-node.blue { --accent: #7289ff; }
.ap-flow-node.pink { --accent: #f06c94; }
.ap-flow-node h3 { margin: 22px 0 10px; color: #fff; font-size: 18px; line-height: 1.25; }
.ap-flow-node p { margin: 0; color: #adb0b8; font-size: 13px; line-height: 1.62; }
.ap-flow-node::after { position: absolute; right: 16px; bottom: 12px; content: "\2192"; color: var(--accent); font-size: 20px; }
.ap-flow-responsibility { margin: 20px 0 0; padding: 14px 16px; color: #e3e4e8; border: 1px solid #555861; border-left: 5px solid var(--ap-yellow); font-size: 13px; line-height: 1.65; }

.ap-macro-stage { margin: 0 0 58px; padding: 32px 38px 36px; color: #fff; background: #1a1c24; border-top: 7px solid var(--ap-blue); border-radius: 0 0 6px 6px; }
.ap-macro-stage .ap-lane-label { color: #9aabff; font-family: var(--ap-font-mono); font-size: 11px; font-weight: 800; text-transform: uppercase; }
.ap-macro-stage h2 { max-width: 780px; margin: 10px 0 0; color: #fff; font-size: 29px; line-height: 1.2; }
.ap-macro-steps { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 24px; margin-top: 28px; }
.ap-macro-step { padding-top: 14px; border-top: 1px solid #666a78; }
.ap-macro-step > span { color: var(--ap-pink); font-family: var(--ap-font-mono); font-size: 10px; font-weight: 800; }
.ap-macro-step h3 { margin: 14px 0 8px; color: #fff; font-size: 17px; }
.ap-macro-step p { margin: 0; color: #b9bbc4; font-size: 13px; line-height: 1.62; }
.ap-macro-isolation { margin: 24px 0 0; padding: 13px 15px; color: #fff; background: #242733; border: 1px solid #72788c; border-left: 5px solid var(--ap-pink); font-size: 13px; font-weight: 650; line-height: 1.6; }

.ap-guardrail-section { margin-top: 42px; }
.ap-guardrail-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 18px; }
.ap-guardrail { min-height: 208px; padding: 18px 16px; background: var(--ap-white); border: 1px solid var(--ap-ink); border-top: 7px solid var(--accent); border-radius: 5px; }
.ap-guardrail.yellow { --accent: var(--ap-yellow); }
.ap-guardrail.blue { --accent: var(--ap-blue); }
.ap-guardrail.pink { --accent: var(--ap-pink); }
.ap-guardrail.green { --accent: var(--ap-green); }
.ap-guardrail > span { color: var(--ap-muted); font-family: var(--ap-font-mono); font-size: 10px; font-weight: 800; }
.ap-guardrail h3 { margin: 24px 0 10px; color: var(--ap-ink); font-size: 18px; line-height: 1.25; }
.ap-guardrail p { margin: 0; color: #5b5e66; font-size: 13px; line-height: 1.65; }

.ap-home-footer { margin: 68px 0 14px; padding: 30px 0 20px; border-top: 1px solid var(--ap-ink); }
.ap-home-footer-head { display: flex; align-items: baseline; justify-content: space-between; gap: 24px; }
.ap-home-footer strong { color: var(--ap-ink); font-size: 28px; }
.ap-home-footer code { color: var(--ap-blue); font-family: var(--ap-font-mono); font-size: 11px; }
.ap-home-footer p { max-width: 840px; margin: 8px 0 0; color: var(--ap-muted); font-size: 13px; line-height: 1.65; }

@media (hover: hover) {
    .st-key-ap_home_product_gold:hover,
    .st-key-ap_home_product_advice:hover,
    .st-key-ap_home_product_macro:hover,
    .st-key-ap_home_product_validation:hover { transform: translate(-3px, -3px); box-shadow: 11px 11px 0 #cfd2d8; }
}
@media (max-width: 900px) {
    [data-testid="stMainBlockContainer"] { padding-top: 4rem; }
    .st-key-ap_home_hero { padding: 22px 20px 30px; }
    .ap-hero-layout { grid-template-columns: minmax(0, 1fr); gap: 12px; min-height: 0; }
    .ap-hero-copy { max-width: 760px; padding-bottom: 4px; }
    .ap-hero-copy h1 { max-width: 720px; font-size: clamp(46px, 7.2vw, 56px); }
    .ap-window-scene { height: 344px; margin-top: 4px; opacity: .72; }
    .ap-window.back { height: 300px; }
    .ap-window.front { top: 64px; height: 276px; }
    .ap-flow-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .ap-guardrail-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
    [data-testid="stMainBlockContainer"] { padding-right: 1rem; padding-left: 1rem; }
    .st-key-ap_home_hero { padding: 16px 4px 28px; }
    .ap-hero-layout { gap: 10px; }
    .ap-hero-meta { margin-bottom: 22px; }
    .ap-hero-copy { padding: 26px 0 0; }
    .ap-hero-copy h1 { font-size: clamp(34px, 10.8vw, 42px); line-height: 1.06; }
    .ap-hero-copy h1::after { width: 128px; height: 10px; margin-top: 16px; }
    .ap-hero-subtitle { max-width: 100%; font-size: 16px; line-height: 1.55; }
    .ap-hero-tags { gap: 6px; }
    .ap-window-scene { height: 206px; margin: 4px -6px 0; opacity: .78; }
    .ap-window.back, .ap-scene-ticket { display: none; }
    .ap-window.front { top: 34px; right: 2%; width: 96%; height: 156px; }
    .ap-window-body { padding: 11px 12px; }
    .ap-window-copy { max-width: 54%; }
    .ap-window-copy strong { margin-top: 5px; font-size: 14px; }
    .ap-window-label { font-size: 8px; }
    .ap-window-slots { right: 10px; bottom: 9px; left: 58%; height: 52px; gap: 1px; }
    .ap-homepage-heading h2 { font-size: 31px; }
    .ap-product-copy p { min-height: 0; }
    .ap-technical-stage { margin-top: 54px; padding: 32px 18px 28px; }
    .ap-type-rail { margin-right: -18px; margin-left: -18px; padding-right: 18px; padding-left: 18px; }
    .ap-flow-grid, .ap-macro-steps, .ap-guardrail-grid { grid-template-columns: 1fr; }
    .ap-flow-node, .ap-guardrail { min-height: 0; }
    .ap-macro-stage { padding: 28px 18px 30px; }
    .ap-home-footer-head { display: block; }
    .ap-home-footer code { display: block; margin-top: 8px; }
}
@media (prefers-reduced-motion: reduce) {
    .st-key-ap_home_hero *,
    .st-key-ap_home_product_gold,
    .st-key-ap_home_product_advice,
    .st-key-ap_home_product_macro,
    .st-key-ap_home_product_validation { transition: none !important; transform: none !important; }
}
</style>
"""


def _render_hero() -> None:
    slots = "".join(
        f"<i style='--slot:{height}px'></i>"
        for height in (24, 34, 29, 46, 38, 55, 43, 62, 52, 70, 47, 66, 58, 78, 61, 72, 64, 84, 68, 80, 74)
    )
    with st.container(key="ap_home_hero"):
        st.markdown(
            f"""
            <div class="ap-hero-layout">
                <div class="ap-hero-copy">
                    <div class="ap-hero-meta">
                        <p class="ap-hero-brand">{_copy("homepage_brand")}</p>
                        <p class="ap-hero-index">01 / {_copy("homepage_hero_serial")}</p>
                    </div>
                    <h1>{_copy("homepage_title")}</h1>
                    <p class="ap-hero-subtitle">{_copy("homepage_subtitle")}</p>
                    <div class="ap-hero-tags">
                        <span>{_copy("homepage_tag_long_term")}</span>
                        <span>{_copy("homepage_tag_traceable")}</span>
                        <span>{_copy("homepage_tag_no_auto")}</span>
                    </div>
                </div>
                <div class="ap-window-scene" aria-hidden="true">
                    <div class="ap-window back"><div class="ap-window-bar">AUPILOT / {_copy("homepage_window_audit")} <span>{_copy("homepage_window_daily")}</span></div></div>
                    <div class="ap-window front">
                        <div class="ap-window-bar">
                            <span>GC.v.0 / {_copy("homepage_window_slots")}</span>
                            <span class="ap-window-controls"><i></i><i></i><i></i></span>
                        </div>
                        <div class="ap-window-body">
                            <div class="ap-window-grid"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
                            <div class="ap-window-copy">
                                <p class="ap-window-label">{_copy("homepage_window_label")}</p>
                                <strong>{_copy("homepage_flow_data_title")}<br><mark>{_copy("homepage_flow_issuance_title")}</mark></strong>
                            </div>
                            <div class="ap-window-slots">{slots}</div>
                        </div>
                    </div>
                    <div class="ap-scene-ticket">CANONICAL_GC_UTC_DAILY_BUCKET_V1<br>{_copy("homepage_window_no_execution")}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _section_heading(kicker_key: str, title_key: str, body_key: str) -> None:
    st.markdown(
        f"""
        <div class="ap-homepage-heading">
            <p class="ap-homepage-kicker">{_copy(kicker_key)}</p>
            <h2>{_copy(title_key)}</h2>
            <p>{_copy(body_key)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_product_entries() -> None:
    st.markdown('<section class="ap-product-section">', unsafe_allow_html=True)
    _section_heading("homepage_products_kicker", "homepage_products_title", "homepage_products_body")
    for row_start in range(0, len(PRODUCTS), 2):
        columns = st.columns(2, gap="large")
        for column, (index, title_key, body_key, action_key, page, icon, slug) in zip(columns, PRODUCTS[row_start : row_start + 2]):
            with column:
                with st.container(border=True, key=f"ap_home_product_{slug}"):
                    st.markdown(
                        f"""
                        <div class="ap-product-copy">
                            <div class="ap-product-head"><p class="ap-product-index">{_copy("homepage_view_prefix")} / {index}</p><i class="ap-product-status"></i></div>
                            <h3>{_copy(title_key)}</h3>
                            <p>{_copy(body_key)}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.page_link(page, label=tr(action_key), icon=icon, width="stretch")
    st.markdown("</section>", unsafe_allow_html=True)


def _render_system_map() -> None:
    nodes = "".join(
        f"<article class='ap-flow-node {accent}'><span>{index}</span><h3>{_copy(title)}</h3><p>{_copy(body)}</p></article>"
        for index, title, body, accent in TECHNICAL_STEPS
    )
    st.markdown(
        f"""
        <section class="ap-technical-stage">
            <div class="ap-homepage-heading">
                <p class="ap-homepage-kicker">{_copy("homepage_system_kicker")}</p>
                <h2>{_copy("homepage_system_title")}</h2>
                <p>{_copy("homepage_system_body")}</p>
            </div>
            <div class="ap-type-rail" aria-hidden="true">
                <span>Databento OHLC</span><b>/</b><span>{_copy("homepage_flow_action_title")}</span><b>/</b><span>{_copy("homepage_flow_outlook_title")}</span><b>/</b><span>PN02 OHLC</span><b>/</b><span>{_copy("homepage_flow_issuance_title")}</span>
            </div>
            <div class="ap-flow-grid">{nodes}</div>
            <p class="ap-flow-responsibility">{_copy("homepage_flow_responsibility")}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    macro_nodes = "".join(
        f"<article class='ap-macro-step'><span>{index}</span><h3>{_copy(title)}</h3><p>{_copy(body)}</p></article>"
        for index, title, body in MACRO_STEPS
    )
    st.markdown(
        f"""
        <section class="ap-macro-stage">
            <span class="ap-lane-label">{_copy("homepage_macro_lane")}</span>
            <h2>{_copy("homepage_macro_lane_title")}</h2>
            <div class="ap-macro-steps">{macro_nodes}</div>
            <p class="ap-macro-isolation">{_copy("homepage_macro_isolation")}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_guardrails() -> None:
    cards = "".join(
        f"<article class='ap-guardrail {accent}'><span>{_copy('homepage_rule_prefix')} / {index}</span><h3>{_copy(title)}</h3><p>{_copy(body)}</p></article>"
        for index, title, body, accent in GUARDRAILS
    )
    st.markdown(
        f"""
        <section class="ap-guardrail-section">
            <div class="ap-homepage-heading">
                <p class="ap-homepage-kicker">{_copy("homepage_guardrails_kicker")}</p>
                <h2>{_copy("homepage_guardrails_title")}</h2>
                <p>{_copy("homepage_guardrails_body")}</p>
            </div>
            <div class="ap-guardrail-grid">{cards}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_footer() -> None:
    st.markdown(
        f"""
        <footer class="ap-home-footer">
            <div class="ap-home-footer-head"><strong>{_copy("homepage_brand")}</strong><code>{_copy("homepage_footer_meta")}</code></div>
            <p>{_copy("homepage_footer_disclaimer")}</p>
            <p>{_copy("homepage_footer_performance")}</p>
        </footer>
        """,
        unsafe_allow_html=True,
    )


st.markdown(HOMEPAGE_CSS, unsafe_allow_html=True)
_render_hero()
_render_product_entries()
_render_system_map()
_render_guardrails()
_render_footer()
