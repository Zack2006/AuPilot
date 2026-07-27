"""Decision Assistant: one persisted Today Advice card and 21-slot calendar."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from html import escape

import streamlit as st

from frontend.api_client import AurumPilotAPIError, api
from frontend.business_theme import apply_business_theme
from frontend.components.metric_cards import metric_row
from frontend.components.risk_badge import risk_badge
from frontend.i18n import (
    localize,
    localize_reason,
    page_header,
    section_header,
    tr,
)


DISPLAY_TIMEZONE = "Asia/Shanghai"
ACTION_COLORS = {"HOLD": "blue", "REDUCE": "orange", "ADD": "green"}
def _utc(value: str | None) -> str:
    if not value:
        return "-"
    return (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        .astimezone(UTC)
        .strftime("%Y-%m-%d %H:%M UTC")
    )


def _fmt_pp(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) < 0.005:
        return "0 pp"
    return f"{value:+.1f} pp"


def _fmt_pct(value: float | None) -> str:
    return tr("allocation_required") if value is None else f"{value:.1f}%"


def _probability_overview_html(probabilities: dict[str, float]) -> str:
    def value(label: str) -> str:
        return f"{float(probabilities.get(label, 0.0)):.2%}"

    def strength_cells(side: str) -> str:
        return "".join(
            (
                f'<div class="probability-strength level-{level}">'
                f"<span>{side}_L{level}</span>"
                f"<strong>{value(f'{side}_L{level}')}</strong>"
                "</div>"
            )
            for level in range(1, 4)
        )

    return f"""
<style>
.probability-overview {{
  display:grid;grid-template-columns:minmax(135px,.72fr) repeat(2,minmax(230px,1fr));
  gap:12px;margin:8px 0 18px;
}}
.probability-overview-group {{
  border:1px solid rgba(32,35,31,.09);border-radius:17px;padding:15px 16px;
  background:rgba(255,255,255,.94);
  box-shadow:0 2px 3px rgba(20,22,19,.025),0 12px 34px rgba(20,22,19,.05);
  transition:transform .22s cubic-bezier(.22,.8,.32,1),box-shadow .22s ease,
             border-color .22s ease;
}}
.probability-overview-group:hover {{
  transform:translateY(-3px);border-color:rgba(179,138,62,.20);
  box-shadow:0 3px 6px rgba(20,22,19,.035),0 18px 44px rgba(20,22,19,.075);
}}
.probability-overview-title {{
  display:flex;align-items:center;justify-content:space-between;gap:10px;
  margin-bottom:11px;color:#3f423d;font-size:.75rem;font-weight:700;
  letter-spacing:.07em;text-transform:uppercase;
}}
.probability-overview-title small {{color:#92978f;font-size:.66rem;font-weight:500}}
.probability-normal {{
  display:flex;flex-direction:column;justify-content:center;align-items:center;
  min-height:92px;text-align:center;
  background:
    radial-gradient(circle at 50% -12%,rgba(179,138,62,.11),transparent 66%),
    linear-gradient(145deg,#ffffff,#f6f7f4);
}}
.probability-normal span {{color:#7a7f77;font-size:.72rem;letter-spacing:.08em}}
.probability-normal strong {{font-size:1.65rem;margin-top:7px;color:#171816;letter-spacing:-.035em}}
.probability-top {{
  background:linear-gradient(145deg,#fff,#fff8f8);
}}
.probability-bottom {{
  background:linear-gradient(145deg,#fff,#f7fcf9);
}}
.probability-strengths {{display:grid;grid-template-columns:repeat(3,1fr);gap:7px}}
.probability-strength {{
  display:flex;flex-direction:column;gap:5px;padding:9px 8px;border-radius:10px;
  background:rgba(248,249,247,.78);border:1px solid rgba(32,35,31,.055);
}}
.probability-strength span {{color:#7b8079;font-size:.64rem}}
.probability-strength strong {{font-size:.9rem;color:#282b27;font-variant-numeric:tabular-nums}}
.probability-top .level-1 {{box-shadow:inset 0 2px #b36b74}}
.probability-top .level-2 {{box-shadow:inset 0 2px #c84f5e}}
.probability-top .level-3 {{box-shadow:inset 0 2px #e23950}}
.probability-bottom .level-1 {{box-shadow:inset 0 2px #5b9a80}}
.probability-bottom .level-2 {{box-shadow:inset 0 2px #2f8e67}}
.probability-bottom .level-3 {{box-shadow:inset 0 2px #13a86f}}
@media (max-width:760px) {{
  .probability-overview {{grid-template-columns:1fr}}
  .probability-normal {{min-height:74px}}
}}
</style>
<div class="probability-overview" role="group"
     aria-label="{escape(tr('seven_class_probabilities'))}">
  <div class="probability-overview-group probability-normal">
    <span>NORMAL</span><strong>{value("NORMAL")}</strong>
  </div>
  <div class="probability-overview-group probability-top">
    <div class="probability-overview-title">
      <span>TOP</span><small>{escape(tr("three_strength_levels"))}</small>
    </div>
    <div class="probability-strengths">{strength_cells("TOP")}</div>
  </div>
  <div class="probability-overview-group probability-bottom">
    <div class="probability-overview-title">
      <span>BOTTOM</span><small>{escape(tr("three_strength_levels"))}</small>
    </div>
    <div class="probability-strengths">{strength_cells("BOTTOM")}</div>
  </div>
</div>
"""


def _price_rows(price: dict | None) -> list[tuple[str, str]]:
    if not isinstance(price, dict):
        return []
    rows = []
    for field in ("open", "high", "low", "close"):
        item = price.get(field) or {}
        point = item.get("point")
        lower = item.get("p10")
        upper = item.get("p90")
        if point is None:
            value = tr("interval_unavailable")
        elif lower is None or upper is None:
            value = f"${point:,.2f} · {tr('interval_unavailable')}"
        else:
            value = f"${point:,.2f} · ${lower:,.2f}–${upper:,.2f}"
        rows.append((tr(field), value))
    return rows


def _slot_popover_html(
    slot: dict,
    *,
    is_h1: bool,
    freshness_status: str,
    advice: dict,
) -> str:
    probability = slot["probabilities"]
    aggregate = slot["aggregate_probabilities"]
    rows = [
        (
            "Horizon",
            f"H{slot['horizon_index']}"
            + (
                f" · {tr('current_action_slot')}"
                if is_h1
                else f" · {tr('forecast_only')}"
            ),
        ),
        (
            tr("utc_bucket"),
            f"{_utc(slot['target_bucket_start_utc'])} → "
            f"{_utc(slot['target_bucket_end_utc'])}",
        ),
        (
            tr("local_mapping"),
            f"{slot['local_date']} · {escape(slot['display_timezone'])}",
        ),
        (tr("predicted_label"), localize(slot["predicted_label"])),
        (
            tr("aggregate_probability"),
            (
                f"N {aggregate['NORMAL']:.2%} · "
                f"TOP {aggregate['TOP']:.2%} · "
                f"BOTTOM {aggregate['BOTTOM']:.2%}"
            ),
        ),
    ]
    rows_html = "".join(
        (
            '<div class="slot-detail-row">'
            f"<span>{escape(str(label))}</span>"
            f"<strong>{escape(str(value))}</strong>"
            "</div>"
        )
        for label, value in rows
    )

    probability_html = (
        '<div class="slot-section-title">'
        f"{escape(tr('seven_class_probabilities'))}</div>"
        '<div class="probability-list">'
        + "".join(
            (
                '<div class="probability-item">'
                f"<span>{escape(label)}</span>"
                f"<b>{value:.2%}</b>"
                "</div>"
            )
            for label, value in probability.items()
        )
        + "</div>"
    )

    action_html = ""
    if is_h1:
        title = (
            tr("today_advice_current")
            if freshness_status == "CURRENT"
            else tr("recent_available_advice")
        )
        action_html = (
            '<div class="slot-section-title">'
            f"{escape(title)}</div>"
            '<div class="slot-detail-row">'
            f"<span>{escape(tr('recommended_action'))}</span>"
            f"<strong>{escape(localize(advice['recommended_action']))}</strong>"
            "</div>"
            '<div class="slot-detail-row">'
            f"<span>{escape(tr('requested_delta_pp'))}</span>"
            f"<strong>{escape(_fmt_pp(advice['requested_delta_pp']))}</strong>"
            "</div>"
            '<div class="slot-detail-row">'
            f"<span>{escape(tr('model_tactical_state'))}</span>"
            f"<strong>{advice['model_tactical_weight_before_pct']:.1f}%"
            f" → {advice['model_tactical_weight_after_pct']:.1f}%</strong>"
            "</div>"
            '<p class="slot-note">'
            f"{escape(localize_reason(advice['action_reason_code']))}</p>"
            '<p class="slot-note advisory">'
            f"{escape(tr('advisory_only_no_execution'))}</p>"
        )

    price = slot.get("conditional_price")
    if slot["predicted_label"] == "NORMAL":
        price_html = (
            '<div class="normal-note">'
            f"{escape(tr('normal_lightweight_detail'))}</div>"
        )
    elif not isinstance(price, dict):
        price_html = (
            '<div class="normal-note">'
            f"{escape(tr('conditional_price_unavailable'))}</div>"
        )
    else:
        price_rows = "".join(
            (
                '<div class="slot-detail-row price">'
                f"<span>{escape(label)}</span><strong>{escape(value)}</strong>"
                "</div>"
            )
            for label, value in _price_rows(price)
        )
        price_html = (
            '<div class="slot-section-title">'
            f"{escape(tr('conditional_price_scenario'))} · "
            f"{escape(price['side'])}</div>"
            f"{price_rows}"
            '<p class="slot-note">'
            f"{escape(tr('marginal_interval_notice'))}</p>"
            '<p class="slot-note advisory">'
            f"{escape(tr('conditional_price_role_strict'))}</p>"
        )

    footer = (
        ""
        if is_h1
        else (
            '<p class="slot-note forecast-note">'
            f"{escape(tr('future_not_current_action'))}</p>"
        )
    )
    return rows_html + probability_html + action_html + price_html + footer


def _calendar_html(
    slots: list[dict],
    freshness_status: str,
    advice: dict,
) -> str:
    by_date = {
        date.fromisoformat(slot["local_date"]): slot for slot in slots
    }
    if len(by_date) != 21:
        raise ValueError("LOCAL_SLOT_DATE_COLLISION")
    first_date = min(by_date)
    last_date = max(by_date)
    cursor = first_date - timedelta(days=first_date.weekday())
    final = last_date + timedelta(days=6 - last_date.weekday())
    weekdays = "".join(
        f'<div class="weekday">{escape(tr(f"weekday_{index}"))}</div>'
        for index in range(7)
    )
    cells = []
    while cursor <= final:
        item = by_date.get(cursor)
        if item is None:
            if cursor < first_date or cursor > last_date:
                cells.append(
                    '<div class="calendar-cell calendar-pad" '
                    'aria-hidden="true"></div>'
                )
            else:
                cells.append(
                    (
                        '<div class="calendar-cell market-gap" '
                        'aria-hidden="true"><span>'
                        f"{cursor.strftime('%m/%d')}</span>"
                        f"<small>{escape(tr('market_closed'))}</small></div>"
                    )
                )
        else:
            index = int(item["horizon_index"])
            is_h1 = index == 1
            label = item["predicted_label"]
            current_copy = ""
            if is_h1:
                current_copy = (
                    tr("current_slot")
                    if freshness_status == "CURRENT"
                    else tr("recent_h1")
                )
            popover = _slot_popover_html(
                item,
                is_h1=is_h1,
                freshness_status=freshness_status,
                advice=advice,
            )
            alignment = " align-right" if cursor.weekday() >= 4 else ""
            cells.append(
                (
                    f'<div class="calendar-cell slot-wrap{alignment}">'
                    f'<details name="forecast-slot" '
                    f'class="slot-tile label-{escape(label)}'
                    f'{" h1" if is_h1 else ""}">'
                    f'<summary aria-label="H{index} {escape(label)} '
                    f'{escape(cursor.isoformat())}">'
                    '<span class="slot-date">'
                    f"{cursor.strftime('%m/%d')}</span>"
                    f'<span class="slot-horizon">H{index}</span>'
                    f'<strong class="slot-label">{escape(label)}</strong>'
                    + (
                        f'<em class="current-chip">{escape(current_copy)}</em>'
                        if current_copy
                        else ""
                    )
                    + "</summary>"
                    "</details>"
                    '<div class="slot-popover" role="dialog">'
                    f"{popover}</div></div>"
                )
            )
        cursor += timedelta(days=1)

    return f"""
<style>
.calendar-wrap {{
  --normal:#eef1ef; --top1:#a75d68; --top2:#cb5060; --top3:#eb465e;
  --bottom1:#4f896f; --bottom2:#2d9a6e; --bottom3:#32c988;
  position:relative; overflow:visible; padding:4px 2px 220px;
}}
.forecast-grid {{
  display:grid; grid-template-columns:repeat(7,minmax(86px,1fr));
  gap:10px; position:relative; overflow:visible;
}}
.weekday {{
  color:#868b84; text-align:center; font-size:.72rem; letter-spacing:.08em;
  text-transform:uppercase; padding:0 0 5px;
}}
.calendar-cell {{
  min-height:102px; border-radius:13px; box-sizing:border-box;
}}
.slot-wrap {{position:relative;overflow:visible;z-index:1}}
.market-gap {{
  border:1px dashed rgba(83,90,82,.13); color:rgba(77,84,76,.34);
  padding:12px; display:flex; flex-direction:column; gap:8px;
}}
.calendar-pad {{border:1px solid transparent}}
.market-gap small {{font-size:.65rem;}}
.slot-tile {{
  min-height:102px;box-sizing:border-box;position:relative;
  border:1px solid rgba(32,35,31,.08);border-radius:14px;
  background:var(--normal); color:#333732;
  transition:transform .18s cubic-bezier(.2,.85,.3,1.25),
             border-color .18s ease,box-shadow .18s ease,filter .18s ease;
  will-change:transform;
}}
.slot-wrap:hover,.slot-wrap:focus-within {{z-index:20}}
.slot-wrap:hover .slot-tile,.slot-wrap:focus-within .slot-tile,.slot-tile[open] {{
  transform:translateY(-5px) scale(1.018); border-color:rgba(179,138,62,.38);
  box-shadow:0 18px 46px rgba(26,29,25,.16);filter:brightness(1.025);
}}
.slot-tile.h1 {{
  border:2px solid #b38a3e; box-shadow:0 0 0 3px rgba(179,138,62,.10);
}}
.label-TOP_L1 {{background:var(--top1)}} .label-TOP_L2 {{background:var(--top2)}}
.label-TOP_L3 {{background:var(--top3)}} .label-BOTTOM_L1 {{background:var(--bottom1)}}
.label-BOTTOM_L2 {{background:var(--bottom2)}} .label-BOTTOM_L3 {{background:var(--bottom3)}}
.label-TOP_L1,.label-TOP_L2,.label-TOP_L3,
.label-BOTTOM_L1,.label-BOTTOM_L2,.label-BOTTOM_L3 {{color:#fff}}
.label-BOTTOM_L3 {{color:#0f3828}}
.label-BOTTOM_L3 .slot-date,.label-BOTTOM_L3 .slot-horizon {{color:#1f513e}}
.slot-tile summary {{
  list-style:none; cursor:pointer; min-height:102px; padding:11px;
  display:grid; grid-template-columns:1fr auto; gap:6px; align-content:start;
  position:relative;overflow:hidden;border-radius:12px;
}}
.slot-tile summary::-webkit-details-marker {{display:none}}
.slot-tile summary::after {{
  content:"";position:absolute;inset:-35% auto -35% -45%;width:34%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.16),transparent);
  transform:skewX(-18deg);transition:left .35s ease;
}}
.slot-wrap:hover summary::after,.slot-wrap:focus-within summary::after,
.slot-tile[open] summary::after {{left:125%}}
.slot-date {{font-size:.78rem;color:#454a44}} .slot-horizon {{font-size:.72rem;color:#7c827a}}
.label-TOP_L1 .slot-date,.label-TOP_L1 .slot-horizon,
.label-TOP_L2 .slot-date,.label-TOP_L2 .slot-horizon,
.label-TOP_L3 .slot-date,.label-TOP_L3 .slot-horizon,
.label-BOTTOM_L1 .slot-date,.label-BOTTOM_L1 .slot-horizon,
.label-BOTTOM_L2 .slot-date,.label-BOTTOM_L2 .slot-horizon {{color:rgba(255,255,255,.82)}}
.slot-label {{grid-column:1/-1;font-size:.82rem;letter-spacing:.03em;margin-top:4px}}
.current-chip {{
  grid-column:1/-1; width:max-content; padding:2px 7px;border-radius:20px;
  background:#f4e8cc;color:#755821;font-size:.64rem;font-style:normal;font-weight:700;
  border:1px solid rgba(179,138,62,.22);
}}
.slot-popover {{
  display:none; position:absolute; top:calc(100% + 9px); left:0;
  width:min(440px,78vw); max-height:600px; overflow:auto; padding:18px;
  border:1px solid rgba(179,138,62,.24); border-radius:17px;
  background:rgba(255,255,255,.985); color:#242723;
  box-shadow:0 24px 70px rgba(22,25,21,.22);
  backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);
  z-index:200; transform-origin:top left;
}}
.align-right .slot-popover {{left:auto;right:0}}
.slot-wrap:hover .slot-popover,.slot-wrap:focus-within .slot-popover,
.slot-tile[open] + .slot-popover {{
  display:block;animation:slot-pop-in .16s cubic-bezier(.2,.8,.25,1) both;
}}
@keyframes slot-pop-in {{
  from {{opacity:0;transform:translateY(-7px) scale(.975)}}
  to {{opacity:1;transform:translateY(0) scale(1)}}
}}
.slot-detail-row {{
  display:grid;grid-template-columns:minmax(115px,.8fr) minmax(180px,1.5fr);
  gap:14px;padding:7px 0;border-bottom:1px solid rgba(32,35,31,.07);
  font-size:.75rem;
}}
.slot-detail-row span {{color:#777c75}} .slot-detail-row strong {{font-weight:600;text-align:right;color:#252824}}
.slot-section-title {{
  margin:15px 0 7px;color:#755821;font-size:.72rem;font-weight:700;
  letter-spacing:.06em;text-transform:uppercase;
}}
.probability-list {{display:grid;grid-template-columns:repeat(2,1fr);gap:5px 15px}}
.probability-item {{display:flex;justify-content:space-between;font-size:.72rem;color:#737870}}
.probability-item b {{color:#252824}}
.slot-note {{color:#737870;font-size:.7rem;line-height:1.5;margin:9px 0 0}}
.slot-note.advisory {{color:#755821}} .forecast-note {{border-top:1px solid rgba(32,35,31,.07);padding-top:9px}}
.normal-note {{margin-top:13px;padding:10px;border-radius:9px;background:#f3f5f2;font-size:.73rem;color:#555a53}}
.strength-legend {{
  display:grid;grid-template-columns:repeat(2,minmax(220px,1fr));gap:12px;
  margin-top:16px;
}}
.strength-legend-group {{
  display:grid;grid-template-columns:auto repeat(3,1fr);align-items:center;gap:7px;
  padding:10px 12px;border:1px solid rgba(32,35,31,.08);border-radius:12px;
  background:rgba(255,255,255,.86);color:#686d66;font-size:.69rem;
  box-shadow:0 6px 22px rgba(20,22,19,.035);
}}
.strength-legend-group strong {{color:#343833;margin-right:4px}}
.strength-swatch {{height:9px;border-radius:4px}}
.strength-swatch.top-1 {{background:var(--top1)}} .strength-swatch.top-2 {{background:var(--top2)}}
.strength-swatch.top-3 {{background:var(--top3)}} .strength-swatch.bottom-1 {{background:var(--bottom1)}}
.strength-swatch.bottom-2 {{background:var(--bottom2)}} .strength-swatch.bottom-3 {{background:var(--bottom3)}}
@media (max-width:720px) {{
  .forecast-grid {{grid-template-columns:repeat(7,minmax(44px,1fr));gap:5px}}
  .calendar-cell,.slot-tile summary {{min-height:86px}}
  .slot-tile summary {{padding:6px;display:flex;flex-direction:column;gap:3px}}
  .slot-label {{font-size:.58rem}} .slot-date,.slot-horizon {{font-size:.62rem}}
  .current-chip {{font-size:.52rem;padding:2px 4px}}
  .market-gap {{padding:6px;font-size:.58rem}} .market-gap small {{display:none}}
  .strength-legend {{grid-template-columns:1fr}}
  .slot-popover,.align-right .slot-popover {{
    position:fixed;left:5vw;right:5vw;top:15vh;width:auto;max-height:70vh;
  }}
}}
@media (prefers-reduced-motion:reduce) {{
  .slot-tile,.slot-tile summary::after {{transition:none!important}}
  .slot-popover {{animation:none!important}}
}}
</style>
<div class="calendar-wrap">
  <div class="forecast-grid">{weekdays}{''.join(cells)}</div>
  <div class="strength-legend" aria-label="{escape(tr('strength_color_legend'))}">
    <div class="strength-legend-group"><strong>TOP</strong>
      <span class="strength-swatch top-1" title="TOP_L1"></span>
      <span class="strength-swatch top-2" title="TOP_L2"></span>
      <span class="strength-swatch top-3" title="TOP_L3"></span>
    </div>
    <div class="strength-legend-group"><strong>BOTTOM</strong>
      <span class="strength-swatch bottom-1" title="BOTTOM_L1"></span>
      <span class="strength-swatch bottom-2" title="BOTTOM_L2"></span>
      <span class="strength-swatch bottom-3" title="BOTTOM_L3"></span>
    </div>
  </div>
</div>
"""


try:
    apply_business_theme("decision-assistant")
    page = api.get(
        "advice/today",
        params={"timezone": DISPLAY_TIMEZONE},
    )
except AurumPilotAPIError:
    page_header("decision_title", "today_advice_body")
    st.error(tr("technical_issuance_unavailable"), icon=":material/cloud_off:")
    st.info(tr("no_mock_fallback"), icon=":material/verified_user:")
    st.stop()

slots = page.get("slots") or []
advice = page.get("today_advice") or {}
issuance = page.get("technical_issuance") or {}
if page.get("slot_count") != 21 or len(slots) != 21:
    st.error(tr("technical_contract_invalid"), icon=":material/error:")
    st.stop()

freshness = page["freshness_status"]
page_header(
    "decision_title",
    "today_advice_body",
)
st.warning(
    tr("dual_model_forward_shadow_warning"),
    icon=":material/science:",
)
if issuance.get("issuance_kind") == "PRE_FORWARD_PRODUCT_PREVIEW":
    st.info(
        tr("pre_forward_preview_notice"),
        icon=":material/preview:",
    )
if freshness == "STALE":
    st.warning(
        f"{tr('technical_issuance_stale')}: "
        f"{localize_reason(page.get('stale_reason') or 'UNKNOWN')}",
        icon=":material/history:",
    )

aggregate = advice["aggregate_probabilities"]
with st.container(border=True, key="decision_today_card"):
    badge_columns = st.columns(2)
    with badge_columns[0]:
        st.badge(
            localize(advice["predicted_label"]),
            color=(
                "red"
                if advice["predicted_label"].startswith("TOP_")
                else "green"
                if advice["predicted_label"].startswith("BOTTOM_")
                else "gray"
            ),
            icon=":material/query_stats:",
        )
    with badge_columns[1]:
        st.badge(
            localize(advice["recommended_action"]),
            color=ACTION_COLORS.get(advice["recommended_action"], "gray"),
            icon=":material/strategy:",
        )
    section_header(
        tr("today_advice_current"),
        localize_reason(advice["action_reason_code"]),
    )
    macro = page["macro"]
    macro_level = str(macro["label"]).title()
    with st.container(border=True, key="decision_macro_summary"):
        macro_columns = st.columns([1, 1.8], gap="medium")
        with macro_columns[0]:
            st.caption(tr("current_macro_risk_level"))
            risk_badge(
                macro_level,
                {
                    "Approved": 1,
                    "Cleared": 2,
                    "Caution": 3,
                    "Hold": 4,
                    "Cancel": 5,
                }.get(macro_level),
                assessment_supported=bool(
                    macro.get("assessment_supported")
                ),
            )
        with macro_columns[1]:
            st.caption(tr("macro_data_updated_at"))
            st.markdown(f"**{_utc(macro.get('as_of_utc'))}**")

    metric_row(
        [
            (tr("top_probability"), f"{aggregate['TOP']:.2%}"),
            (tr("bottom_probability"), f"{aggregate['BOTTOM']:.2%}"),
        ]
    )
    metric_row(
        [
            (tr("normal_probability"), f"{aggregate['NORMAL']:.2%}"),
            (
                tr("recommended_action"),
                localize(advice["recommended_action"]),
            ),
        ]
    )
    st.caption(tr("seven_class_always_visible"))
    st.html(_probability_overview_html(advice["probabilities"]))

    section_header(tr("allocation_advice"), tr("pp_not_multiplier"))
    allocation = page.get("user_allocation")
    personalized = page["personalized_target"]
    metric_row(
        [
            (
                tr("model_tactical_state"),
                f"{advice['model_tactical_weight_before_pct']:.1f}%"
                f" → {advice['model_tactical_weight_after_pct']:.1f}%",
            ),
            (tr("requested_delta_pp"), _fmt_pp(advice["requested_delta_pp"])),
        ]
    )
    metric_row(
        [
            (
                tr("user_current_gold_weight"),
                _fmt_pct(
                    allocation["current_gold_weight_pct"]
                    if allocation
                    else None
                ),
            ),
            (
                tr("personalized_target_weight"),
                _fmt_pct(personalized["target_gold_weight_pct"]),
            ),
        ]
    )
    if personalized["was_clamped"]:
        st.warning(
            f"{tr('allocation_clamped')}: "
            f"{personalized['strategy_min_weight_pct']:.0f}%–"
            f"{personalized['strategy_max_weight_pct']:.0f}%",
            icon=":material/vertical_align_center:",
        )
    if allocation:
        allocation_copy = (
            f"{tr('allocation_updated_at')}: "
            f"{_utc(allocation['updated_at_utc'])}"
        )
        if allocation["is_stale"]:
            st.warning(
                f"{allocation_copy} · {tr('allocation_stale_confirm')}",
                icon=":material/update:",
            )
        else:
            st.caption(allocation_copy)
    else:
        st.info(tr("allocation_missing_notice"), icon=":material/person_edit:")
    st.info(tr("allocation_update_principle"), icon=":material/event_repeat:")

    with st.popover(
        tr("update_current_allocation"),
        icon=":material/edit:",
        width="stretch",
    ):
        st.caption(tr("allocation_input_help"))
        default_weight = (
            float(allocation["current_gold_weight_pct"])
            if allocation
            else 50.0
        )
        with st.form("user_gold_allocation_form"):
            gold_weight = st.number_input(
                tr("user_current_gold_weight"),
                min_value=0.0,
                max_value=100.0,
                value=default_weight,
                step=1.0,
                format="%.1f",
            )
            submitted = st.form_submit_button(
                tr("save_allocation_snapshot"),
                type="primary",
                width="stretch",
            )
        if submitted:
            try:
                api.post(
                    "user/portfolio-snapshots",
                    {
                        "gold_weight_pct": float(gold_weight),
                        "as_of_utc": datetime.now(UTC).isoformat(),
                        "source": "USER_REPORTED",
                    },
                )
            except AurumPilotAPIError:
                st.error(tr("allocation_save_failed"))
            else:
                st.success(tr("allocation_snapshot_saved"))
                st.rerun()

    section_header(
        tr("h1_conditional_price"),
        tr("conditional_price_role_strict"),
    )
    if advice["predicted_label"] == "NORMAL":
        st.info(tr("normal_no_price"), icon=":material/hide_source:")
    elif not isinstance(advice.get("conditional_price"), dict):
        st.warning(
            tr("conditional_price_unavailable"),
            icon=":material/cloud_off:",
        )
    else:
        price_rows = _price_rows(advice["conditional_price"])
        st.dataframe(
            [
                {tr("ohlc_field"): field, tr("point_and_interval"): value}
                for field, value in price_rows
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(tr("marginal_interval_notice"))

    metric_row(
        [
            (tr("source_bucket"), issuance["source_bucket_utc"]),
            (
                tr("expected_execution_bucket"),
                advice["target_bucket"],
            ),
        ]
    )
    metric_row(
        [
            (tr("issued_at"), _utc(issuance["issued_at_utc"])),
        ]
    )
    st.info(tr("advisory_only_no_execution"), icon=":material/shield:")

section_header(tr("forecast_calendar_21"))
try:
    st.html(_calendar_html(slots, freshness, advice))
except ValueError:
    st.error(tr("technical_contract_invalid"), icon=":material/error:")

with st.expander(tr("audit_identity"), icon=":material/fingerprint:"):
    st.caption(
        f"{tr('composite_snapshot_id')}: "
        f"{page['composite_snapshot_id']}\n\n"
        f"{tr('issuance_id')}: {issuance['issuance_id']} · "
        f"SHA256: {issuance['output_sha256']}\n\n"
        f"MN18 manifest: {issuance['model_manifest_sha256']}\n\n"
        f"MN18 joblib: {issuance['model_artifact_sha256']}\n\n"
        f"PN02: {page['price_issuance']['model_id']} · "
        f"{page['price_issuance']['model_artifact_sha256']}\n\n"
        f"{tr('page_get_no_model_run')}"
    )

st.info(tr("disclaimer"), icon=":material/shield:")
