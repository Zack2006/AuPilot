"""Display-only conditional OHLC chart for non-NORMAL MN18 slots."""

from __future__ import annotations

from html import escape
from math import isfinite

import streamlit as st

from frontend.i18n import tr


_COLORS = {
    ("TOP", "L1"): "#E89A9A",
    ("TOP", "L2"): "#C84B4B",
    ("TOP", "L3"): "#8F2323",
    ("BOTTOM", "L1"): "#9BCDB4",
    ("BOTTOM", "L2"): "#238B57",
    ("BOTTOM", "L3"): "#11683D",
}


def render_conditional_price_chart(slots: list[dict]) -> None:
    """Render only same-direction PN02 candles selected by MN18."""

    if len(slots) != 21:
        raise ValueError("Conditional price chart requires exactly 21 slots")
    selected = [
        slot
        for slot in slots
        if isinstance(slot.get("conditional_price_outlook"), dict)
    ]
    if not selected:
        st.info(
            tr("no_conditional_signals_21"),
            icon=":material/hide_source:",
        )
        return

    values: list[float] = []
    for slot in selected:
        outlook = slot["conditional_price_outlook"]
        point = outlook.get("point") or {}
        values.extend(float(point[field]) for field in ("open", "high", "low", "close"))
        intervals = outlook.get("marginal_80pct_intervals") or {}
        for field in ("open", "high", "low", "close"):
            interval = intervals.get(field)
            if isinstance(interval, dict):
                values.extend(
                    (float(interval["lower"]), float(interval["upper"]))
                )
    if not values or not all(isfinite(value) and value > 0 for value in values):
        raise ValueError("Conditional price chart received an invalid price")

    lower, upper = min(values), max(values)
    padding = max((upper - lower) * 0.08, upper * 0.002)
    lower -= padding
    upper += padding
    chart_height = 260
    baseline = 18

    def y(value: float) -> float:
        return baseline + (upper - value) / (upper - lower) * chart_height

    slot_width = 52
    left = 58
    width = left + slot_width * 21 + 18
    elements = [
        (
            f"<text x='4' y='{baseline + 8}' class='axis'>"
            f"${upper:,.0f}</text>"
        ),
        (
            f"<text x='4' y='{baseline + chart_height}' class='axis'>"
            f"${lower:,.0f}</text>"
        ),
    ]
    for index, slot in enumerate(slots):
        x = left + index * slot_width + slot_width / 2
        target = str(slot["target_bucket"])
        signal = slot.get("display_signal") or {}
        side = str(signal.get("side") or "NORMAL")
        strength = signal.get("strength")
        outlook = slot.get("conditional_price_outlook")
        elements.append(
            f"<text x='{x:.1f}' y='{baseline + chart_height + 20}' "
            f"class='date' transform='rotate(45 {x:.1f} "
            f"{baseline + chart_height + 20})'>{escape(target[5:])}</text>"
        )
        if not isinstance(outlook, dict):
            elements.append(
                f"<circle cx='{x:.1f}' cy='{baseline + chart_height - 2}' "
                "r='2.2' fill='#8A939A'><title>"
                f"{escape(target)} | NORMAL | "
                f"{escape(tr('normal_no_price'))}</title></circle>"
            )
            continue
        point = outlook["point"]
        color = _COLORS.get((side, strength), "#68717A")
        open_y, close_y = y(float(point["open"])), y(float(point["close"]))
        high_y, low_y = y(float(point["high"])), y(float(point["low"]))
        body_top = min(open_y, close_y)
        body_height = max(abs(open_y - close_y), 2.0)
        close_interval = (
            outlook.get("marginal_80pct_intervals") or {}
        ).get("close")
        interval_svg = ""
        interval_copy = tr("interval_unavailable")
        if isinstance(close_interval, dict):
            interval_y1 = y(float(close_interval["upper"]))
            interval_y2 = y(float(close_interval["lower"]))
            interval_svg = (
                f"<line x1='{x:.1f}' y1='{interval_y1:.1f}' "
                f"x2='{x:.1f}' y2='{interval_y2:.1f}' "
                f"stroke='{color}' stroke-width='5' opacity='.18' "
                "stroke-dasharray='3 3'/>"
            )
            interval_copy = (
                f"close 80% marginal "
                f"[{float(close_interval['lower']):.2f}, "
                f"{float(close_interval['upper']):.2f}]"
            )
        probabilities = slot["probabilities"]
        probability_copy = " | ".join(
            f"{label} {float(value):.2%}"
            for label, value in probabilities.items()
        )
        title = escape(
            f"{target} | {side} {strength} | {probability_copy} | "
            f"O {point['open']:.2f} H {point['high']:.2f} "
            f"L {point['low']:.2f} C {point['close']:.2f} | "
            f"{interval_copy} | {tr('conditional_price_advisory')}"
        )
        elements.extend(
            [
                interval_svg,
                (
                    f"<g><title>{title}</title>"
                    f"<line x1='{x:.1f}' y1='{high_y:.1f}' x2='{x:.1f}' "
                    f"y2='{low_y:.1f}' stroke='{color}' stroke-width='2' "
                    "stroke-dasharray='4 2'/>"
                    f"<rect x='{x - 7:.1f}' y='{body_top:.1f}' width='14' "
                    f"height='{body_height:.1f}' fill='{color}' "
                    f"fill-opacity='.45' stroke='{color}' "
                    "stroke-width='1.5' stroke-dasharray='3 2'/></g>"
                ),
                (
                    f"<text x='{x:.1f}' y='{max(high_y - 5, 10):.1f}' "
                    f"class='signal' fill='{color}'>{side[0]}{strength[-1]}</text>"
                ),
            ]
        )

    st.html(
        (
            "<style>.ap-price-shell{width:100%;overflow-x:auto;padding:4px 0 12px}"
            ".ap-price-svg{min-width:1168px;width:100%;height:326px;"
            "background:rgba(245,243,238,.55);border:1px dashed #B9C0C6;"
            "border-radius:6px}.ap-price-svg .axis{font:10px Inter,"
            "\"Segoe UI\",sans-serif;fill:#68717A}.ap-price-svg .date{"
            "font:10px Inter,\"Segoe UI\",sans-serif;fill:#68717A;"
            "text-anchor:start}.ap-price-svg .signal{font:700 10px Inter,"
            "\"Segoe UI\",sans-serif;text-anchor:middle}</style>"
            f"<div class='ap-price-shell'><svg class='ap-price-svg' "
            f"viewBox='0 0 {width} 326' role='img' "
            f"aria-label='{escape(tr('conditional_price_view'))}'>"
            + "".join(elements)
            + "</svg></div>"
        )
    )
    st.caption(tr("marginal_interval_notice"))
