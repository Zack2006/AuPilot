"""Shared AuPilot brand, navigation and global settings entry."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from frontend.app_settings import show_settings_dialog
from frontend.i18n import tr


BRAND_LOGO_PATH = Path(__file__).resolve().parent / "assets" / "brand" / "aupilot-logo.svg"
GLOBAL_SHELL_CSS = r"""
<style>
:root {
    --ap-font-sans: "Source Han Sans SC", "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", "WenQuanYi Micro Hei", sans-serif;
    --ap-font-mono: ui-monospace, SFMono-Regular, Consolas, "Source Han Sans SC", "Noto Sans CJK SC", "Microsoft YaHei", "PingFang SC", "Hiragino Sans GB", monospace;
}

html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"],
button,
input,
textarea,
select {
    font-family: var(--ap-font-sans);
    letter-spacing: 0;
}

html,
body,
[data-testid="stAppViewContainer"] {
    max-width: 100%;
    overflow-x: clip;
}

[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stAlert"] p,
[data-testid="stCaptionContainer"] {
    overflow-wrap: anywhere;
}

[data-testid="stIconMaterial"] {
    font-family: "Material Symbols Rounded" !important;
}

.st-key-ap_global_settings_slot {
    position: fixed;
    z-index: 1000000;
    top: 10px;
    right: 126px;
    width: 40px;
    height: 40px;
    margin: 0;
}
.st-key-ap_global_settings_slot [data-testid="stVerticalBlock"] { gap: 0; }
.st-key-open_app_settings_global button {
    width: 40px;
    min-width: 40px;
    height: 40px;
    padding: 0;
    color: #383a36;
    background: rgba(255, 255, 255, 0.94);
    border: 1px solid rgba(32, 35, 31, 0.13);
    border-radius: 12px;
    box-shadow: 0 7px 22px rgba(25, 27, 24, 0.07);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    transition:
        color 180ms ease,
        border-color 180ms ease,
        background-color 180ms ease,
        transform 180ms cubic-bezier(.22, .8, .32, 1),
        box-shadow 180ms ease;
}
.st-key-open_app_settings_global button:hover,
.st-key-open_app_settings_global button:focus-visible {
    color: #755821;
    background: #ffffff;
    border-color: rgba(179, 138, 62, 0.48);
    transform: translateY(-1px);
    box-shadow: 0 10px 28px rgba(25, 27, 24, 0.11);
    outline: 2px solid rgba(179, 138, 62, 0.20);
    outline-offset: 2px;
}

/* The overlay is deliberately neutral. Gold remains a small accent inside. */
.stDialog {
    background: rgba(18, 20, 18, 0.22) !important;
    backdrop-filter: blur(8px) saturate(90%);
    -webkit-backdrop-filter: blur(8px) saturate(90%);
}
.stDialog div[role="dialog"] {
    width: min(560px, calc(100vw - 32px)) !important;
    max-width: 560px !important;
    overflow: hidden;
    color: #1b1c1a;
    background:
        linear-gradient(180deg, rgba(255,255,255,.995), rgba(249,250,248,.985));
    border: 1px solid rgba(255, 255, 255, 0.78);
    border-radius: 22px;
    box-shadow:
        0 2px 5px rgba(10, 12, 10, 0.08),
        0 28px 90px rgba(10, 12, 10, 0.24);
}
.stDialog div[role="dialog"]::before {
    content: "";
    position: absolute;
    top: 0;
    left: 2rem;
    width: 2.7rem;
    height: 3px;
    border-radius: 0 0 999px 999px;
    background: linear-gradient(90deg, #8f6b29, #d7bb7a);
}
.stDialog div[role="dialog"] h2 {
    color: #171816;
    font-weight: 630;
    letter-spacing: -0.035em;
}
.stDialog div[role="dialog"] [data-baseweb="tab-list"] {
    gap: 1.2rem;
    border-bottom-color: rgba(32, 35, 31, 0.08);
}
.stDialog div[role="dialog"] [data-baseweb="tab"] {
    color: #71756e;
}
.stDialog div[role="dialog"] [aria-selected="true"] {
    color: #755821;
}
.stDialog div[role="dialog"] [data-testid="stExpander"] {
    overflow: hidden;
    border-color: rgba(32, 35, 31, 0.09);
    border-radius: 13px;
    background: rgba(255,255,255,.78);
}
.stDialog div[role="dialog"] [data-testid="stAlert"] {
    border-radius: 12px;
}
.stDialog div[role="dialog"] [data-testid="stAlertContainer"] {
    color: #50554e;
    background: #f6f7f5 !important;
}
.stDialog div[role="dialog"] [data-testid="stSegmentedControl"],
.stDialog div[role="dialog"] [data-testid="stButtonGroup"] {
    padding: 3px;
    border: 1px solid rgba(32, 35, 31, .10);
    border-radius: 12px;
    background: #f1f3f0;
}
.stDialog div[role="dialog"] [data-testid="stSegmentedControl"] button,
.stDialog div[role="dialog"] [data-testid="stButtonGroup"] [role="radio"] {
    border: 0;
    background: #ffffff !important;
    box-shadow: none;
}
.stDialog div[role="dialog"] [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
    color: #755821;
    background: #ffffff;
    box-shadow: 0 1px 5px rgba(24, 27, 23, .09);
}
.stDialog div[role="dialog"] [data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] {
    color: #755821;
    background: #ffffff !important;
    box-shadow: 0 1px 5px rgba(24, 27, 23, .09);
}
.stDialog div[role="dialog"] [data-testid="stButtonGroup"] [role="radio"][aria-checked="true"] {
    color: #755821;
    background: #ffffff !important;
    box-shadow: 0 1px 5px rgba(24, 27, 23, .09);
}
.stDialog div[role="dialog"] button {
    border-radius: 10px;
    transition:
        transform 180ms cubic-bezier(.22, .8, .32, 1),
        border-color 180ms ease,
        box-shadow 180ms ease;
}
.stDialog div[role="dialog"] button:hover {
    transform: translateY(-1px);
    border-color: rgba(179, 138, 62, .42);
}

@media (max-width: 820px) {
    header[data-testid="stHeader"] {
        min-height: 58px;
    }
    .st-key-ap_global_settings_slot {
        top: 9px;
        right: 54px;
    }
    .stDialog div[role="dialog"] {
        width: calc(100vw - 18px) !important;
        max-height: min(88dvh, 760px);
        overflow-x: hidden;
        overflow-y: auto;
        border-radius: 18px;
    }
    .stDialog div[role="dialog"] [data-baseweb="tab-list"] {
        max-width: 100%;
        overflow-x: auto;
    }
    .stDialog div[role="dialog"] button,
    .stDialog div[role="dialog"] [role="radio"] {
        min-height: 42px;
        white-space: normal;
    }
}

@media (prefers-reduced-motion: reduce) {
    .st-key-open_app_settings_global button,
    .stDialog div[role="dialog"] button {
        transition-duration: .001ms !important;
    }
}
</style>
"""


def render_global_shell() -> None:
    """Render the logo and the single Dock-level settings entry."""

    st.logo(
        str(BRAND_LOGO_PATH),
        size="large",
    )
    st.markdown(GLOBAL_SHELL_CSS, unsafe_allow_html=True)
    with st.container(key="ap_global_settings_slot"):
        if st.button(
            "",
            icon=":material/settings:",
            help=tr("settings_help"),
            key="open_app_settings_global",
        ):
            show_settings_dialog()
