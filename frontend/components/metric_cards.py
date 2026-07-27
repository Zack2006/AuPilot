"""Responsive metric-card helpers. / 响应式指标卡辅助组件。"""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st


MetricItem = tuple[str, str] | tuple[str, str, str]


def metric_row(items: Sequence[MetricItem]) -> None:
    """Render metrics in a wrapping row with optional help text.

    将指标渲染为可自动换行的卡片行；第三个元组元素可提供简短帮助说明。
    """
    # English: A horizontal container wraps naturally on narrow screens, unlike
    # a rigid six-column grid that becomes unreadable on laptops.
    # 中文：水平容器可在窄屏自然换行，避免固定六列在笔记本界面上变得拥挤。
    with st.container(horizontal=True, gap="small"):
        for item in items:
            label, value = item[:2]
            help_text = item[2] if len(item) == 3 else None
            st.metric(label, value, help=help_text, border=True)
