"""Semantic macro-risk badge. / 语义化宏观风险徽章。"""

from html import escape

import streamlit as st

from frontend.i18n import is_chinese, localize, localize_macro_risk_rule, tr


def risk_badge(level: str, score: int | None = None, *, assessment_supported: bool = True) -> None:
    """Render risk with both text and color for accessibility.

    同时使用文字与颜色表达风险等级，避免仅依赖颜色传达含义。
    """
    colors = {"Approved": "green", "Cleared": "blue", "Caution": "orange", "Hold": "red", "Cancel": "gray"}
    icons = {
        "Approved": ":material/check_circle:", "Cleared": ":material/check_circle:",
        "Caution": ":material/warning:", "Hold": ":material/error:", "Cancel": ":material/block:",
    }
    label = localize(level)
    if level == "Caution" and not assessment_supported:
        label += f"（{tr('macro_data_missing_suffix')}）" if is_chinese() else f" ({tr('macro_data_missing_suffix').lower()})"
    if score is not None:
        label += f" · {score}/5"
    st.badge(label, icon=icons.get(level, ":material/info:"), color=colors.get(level, "gray"))


def risk_scale_legend(
    bands: list[dict],
    *,
    current_level: str,
    assessment_supported: bool = True,
) -> None:
    """Render the backend-owned five-level scale and highlight the current band."""

    st.subheader(tr("macro_risk_scale"))
    st.caption(tr("macro_risk_scale_help"))
    cards = []
    for band in bands:
        level = band.get("level", band.get("risk_level"))
        is_current = level == current_level
        marker = (
            f"{escape(tr('macro_current_level'))} · {int(band['risk_score'])}/5"
            if is_current
            else f"{int(band['risk_score'])}/5"
        )
        uppercase_label = level.upper()
        if is_current and level == "Caution" and not assessment_supported:
            uppercase_label += (
                f"（{tr('macro_data_missing_suffix')}）"
                if is_chinese()
                else f" ({tr('macro_data_missing_suffix')})"
            )
        localized_label = localize(level) if is_chinese() else ""
        aria_current = ' aria-current="true"' if is_current else ""
        cards.append(
            '<article class="macro-risk-band macro-risk-'
            f"{escape(level.lower())}{' macro-risk-current' if is_current else ''}"
            f' role="listitem"{aria_current}>'
            f'<span class="macro-risk-marker">{marker}</span>'
            f'<strong>{escape(uppercase_label)}</strong>'
            f'<span class="macro-risk-localized">{escape(localized_label)}</span>'
            f'<p>{escape(localize_macro_risk_rule(band))}</p>'
            "</article>"
        )
    st.html(
        """
        <style>
        .macro-risk-scale {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 0.7rem;
            width: 100%;
            margin: 0.35rem 0 1.2rem;
        }
        .macro-risk-band {
            position: relative;
            min-width: 0;
            min-height: 10.25rem;
            padding: 1rem;
            border: 1px solid rgba(32, 35, 31, 0.09);
            border-radius: 16px;
            background:
                linear-gradient(155deg, rgba(255,255,255,.98), rgba(248,249,247,.88));
            box-shadow: 0 2px 3px rgba(20,22,19,.02), 0 10px 28px rgba(20,22,19,.04);
            overflow-wrap: anywhere;
            overflow: hidden;
            transition:
                transform .22s cubic-bezier(.22,.8,.32,1),
                border-color .22s ease,
                box-shadow .22s ease;
        }
        .macro-risk-band::before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 3px;
            background: var(--macro-band-color);
            opacity: .72;
        }
        .macro-risk-band:hover {
            transform: translateY(-3px);
            border-color: color-mix(in srgb, var(--macro-band-color) 28%, transparent);
            box-shadow: 0 3px 6px rgba(20,22,19,.03), 0 18px 42px rgba(20,22,19,.075);
        }
        .macro-risk-band strong,
        .macro-risk-localized,
        .macro-risk-band p,
        .macro-risk-marker {
            display: block;
            letter-spacing: 0;
        }
        .macro-risk-band strong {
            margin-top: 0.75rem;
            color: #252824;
            font-size: 1.03rem;
            line-height: 1.35;
        }
        .macro-risk-band p {
            margin: 0.65rem 0 0;
            color: #747971;
            font-size: 0.79rem;
            line-height: 1.52;
        }
        .macro-risk-localized {
            margin-top: 0.15rem;
            color: #777c74;
            font-size: 0.8rem;
        }
        .macro-risk-marker {
            font-size: 0.78rem;
            line-height: 1.4;
            color: var(--macro-band-color);
            font-weight: 650;
        }
        .macro-risk-current {
            transform: translateY(-4px);
            border-color: color-mix(in srgb, var(--macro-band-color) 34%, transparent);
            box-shadow:
                0 0 0 3px color-mix(in srgb, var(--macro-band-color) 9%, transparent),
                0 22px 48px rgba(20,22,19,.11);
        }
        .macro-risk-current .macro-risk-marker {
            width: fit-content;
            padding: 0.2rem 0.48rem;
            border-radius: 999px;
            color: white;
            background: var(--macro-band-color);
        }
        .macro-risk-approved { --macro-band-color: #2f7d5a; }
        .macro-risk-cleared { --macro-band-color: #326aa8; }
        .macro-risk-caution { --macro-band-color: #a86812; }
        .macro-risk-hold { --macro-band-color: #c2410c; }
        .macro-risk-cancel { --macro-band-color: #a63838; }
        @media (prefers-reduced-motion: reduce) {
            .macro-risk-band { transition: none; }
        }
        @media (max-width: 800px) {
            .macro-risk-scale { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 480px) {
            .macro-risk-scale { grid-template-columns: minmax(0, 1fr); }
        }
        </style>
        <div class="macro-risk-scale" role="list" aria-label="five-level macro risk scale">
        """
        + "".join(cards)
        + "</div>"
    )
