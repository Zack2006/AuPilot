"""Localized primary recommendation card. / 本地化核心建议卡。"""

import streamlit as st

from frontend.i18n import localize, tr


ACTION_COLORS = {"HOLD_CORE": "green", "REDUCE_TACTICAL": "orange", "REBUY_TACTICAL": "blue", "NO_ACTION": "gray"}


def decision_card(result: dict) -> None:
    """Show action, adjustment and confidence without exposing untranslated enums. / 展示动作、调整与置信度且不暴露未翻译枚举。"""
    with st.container(border=True):
        left, right = st.columns([1.7, 1], vertical_alignment="center")
        with left:
            st.caption(tr("decision_title"))
            st.subheader(localize(result["action"]))
            st.badge(localize(result["action"]), icon=":material/verified_user:", color=ACTION_COLORS.get(result["action"], "gray"))
        with right:
            st.metric(tr("adjustment"), f"{result['tactical_adjustment_ratio']:.0%}")
            st.metric(tr("confidence"), f"{result['confidence']:.0%}")
