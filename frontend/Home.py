"""AurumPilot Streamlit composition root. / AurumPilot Streamlit 组合根。

Purpose / 文件用途: configure the browser session, language and modern multipage navigation.
Inputs / 输入: per-tab language stored in Streamlit session state.
Outputs / 输出: one localized navigation shell and the selected direct-script page.
Business invariants / 业务约束: system text is never displayed bilingually; legacy ``pages/`` discovery is unused.
Side effects / 副作用: initializes session state only.
Fallback behavior / 降级: defaults to Chinese without affecting persisted user records.
"""

import streamlit as st

from frontend.i18n import initialize_app, tr
from frontend.ui_shell import render_global_shell


initialize_app()

navigation = st.navigation(
    [
        st.Page(
            "app_pages/homepage.py",
            title=tr("homepage_nav"),
            icon=":material/space_dashboard:",
            default=True,
            visibility="hidden",
        ),
        st.Page("app_pages/home.py", title=tr("overview_nav"), icon=":material/candlestick_chart:"),
        st.Page("app_pages/decision.py", title=tr("decision_nav"), icon=":material/strategy:"),
        st.Page("app_pages/macro.py", title=tr("macro_nav"), icon=":material/radar:"),
        st.Page("app_pages/model_validation.py", title=tr("validation_nav"), icon=":material/fact_check:"),
    ],
    position="top",
)
render_global_shell()
navigation.run()

