"""Centralized localization and presentation helpers.

Visible product copy lives in reviewed UTF-8 JSON locale files.  Stable API
enum values remain English contracts and are localized only at presentation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import streamlit as st


_LOCALE_ROOT = Path(__file__).resolve().parent / "locales"


def _load_locale(language: str) -> dict[str, str]:
    path = _LOCALE_ROOT / f"{language}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in payload.items()
    ):
        raise RuntimeError(f"Invalid locale catalog: {path}")
    return payload


TEXT = {"en": _load_locale("en"), "zh": _load_locale("zh")}
if set(TEXT["en"]) != set(TEXT["zh"]):
    raise RuntimeError("English and Chinese locale keys do not match")

ENUM_ZH = {
    "BULLISH": "看多", "SLIGHTLY_BULLISH": "略偏多", "NEUTRAL": "中性", "SLIGHTLY_BEARISH": "略偏空", "BEARISH": "看空",
    "UP": "上涨", "DOWN": "下跌", "NONE": "无拐点", "UP_TURN": "向上拐点", "DOWN_TURN": "向下拐点",
    "Approved": "已批准", "Cleared": "已解除", "Caution": "谨慎", "Hold": "暂停", "Cancel": "取消评估",
    "supported": "可评估", "unsupported": "不可评估",
    "HOLD": "保持当前状态", "FULL": "完整战术权重", "REDUCED": "已降低战术权重",
    "TOP_ACTION_ZONE": "TOP 高位动作区", "BOTTOM_ACTION_ZONE": "BOTTOM 低位动作区", "NORMAL": "NORMAL 常态区",
    "REALIZED": "已有实际行情", "PENDING": "等待实际行情",
    "HOLD_CORE": "持有核心仓", "REDUCE_TACTICAL": "减持战术仓",
    "REBUY_TACTICAL": "回补战术仓", "NO_ACTION": "暂不操作", "CONSERVATIVE": "稳健",
    "BALANCED": "平衡", "AGGRESSIVE": "进取", "UPTREND": "上升趋势", "RANGE_BOUND": "区间震荡", "DOWNTREND": "下降趋势",
    "HIGH_VOLATILITY": "高波动", "BUY": "买入", "SELL": "卖出", "CORE": "核心仓", "TACTICAL": "战术仓",
    "TRADE": "成交", "OPENING_BALANCE": "期初余额", "ADJUSTMENT": "调整", "MANUAL": "手工记录", "IMPORT": "导入", "MIGRATION": "迁移",
    "OPEN": "进行中", "PARTIALLY_REBOUGHT": "部分回补", "COMPLETED": "已完成", "INVALIDATED": "已失效",
    "NEXT_SESSION_OPEN": "下一交易日开盘执行", "EVALUATED": "已评估",
    "MARKET_BAR_MISSING": "缺少实际行情", "PRIOR_BAR_MISSING": "缺少预测前行情",
    "PUBLISHED": "已发布", "MISSING_CARRIED_FORWARD": "预测缺失并沿用敞口",
    "EXECUTE_RECOMMENDATION": "执行建议", "CORE_ONLY": "只持有核心仓", "contract": "合约",
    "CPI Release": "CPI 发布", "FOMC Rate Decision": "FOMC 利率决议", "PCE Inflation": "PCE 通胀", "Nonfarm Payrolls": "非农就业",
    "FOMC policy path demonstration note": "FOMC 政策路径演示说明",
    "Consumer Price Index demonstration summary": "消费者价格指数演示摘要",
    "Employment situation demonstration summary": "就业形势演示摘要",
    "Personal consumption expenditures demonstration summary": "个人消费支出演示摘要",
    "Federal Reserve": "美联储", "U.S. Bureau of Labor Statistics": "美国劳工统计局",
    "U.S. Bureau of Economic Analysis": "美国经济分析局",
    "FOMC": "FOMC 利率决议", "RATES": "利率", "CPI": "CPI 通胀", "PCE": "PCE 通胀", "NFP": "非农就业",
    "OK": "正常", "DEGRADED": "降级（部分官方来源或证据槽不可用）", "AVAILABLE": "可用", "UNAVAILABLE": "不可用",
    "COVERED": "已覆盖", "MISSING": "缺失", "STALE": "已过期", "FAILED": "失败",
}

ENUM_ZH.update(
    {
        "HOLD": "维持当前权重",
        "REDUCE_GOLD_WEIGHT": "降低黄金权重",
        "INCREASE_GOLD_WEIGHT": "提高黄金权重",
        "ADD": "加仓",
        "REDUCE": "减仓",
        "NORMAL": "NORMAL 常态",
        "TOP_L1": "TOP L1",
        "TOP_L2": "TOP L2",
        "TOP_L3": "TOP L3",
        "BOTTOM_L1": "BOTTOM L1",
        "BOTTOM_L2": "BOTTOM L2",
        "BOTTOM_L3": "BOTTOM L3",
        "TOP_ACTION_ZONE": "TOP 条件情景",
        "BOTTOM_ACTION_ZONE": "BOTTOM 条件情景",
    }
)


MACRO_PROVIDER_ZH = {
    "official_documents": "官方文档",
    "federal_reserve_calendar": "美联储事件日历",
    "federal_reserve_fomc_release": "美联储 FOMC 声明",
    "new_york_fed_effr": "纽约联储有效联邦基金利率",
    "us_treasury_curve": "美国财政部收益率曲线",
    "bls_public_data_api": "美国劳工统计局数据",
    "bea_personal_income_outlays": "美国经济分析局个人收入与支出",
    "cme_authorized_expectations": "CME 市场概率",
    "optional_fred": "FRED/ALFRED 历史数据",
}

MACRO_SLOT_ZH = {
    "FOMC.calendar": "下一次 FOMC 时间",
    "FOMC.latest_official_decision": "最近一次 FOMC 官方决议",
    "FOMC.market_expectation": "FOMC 市场概率（可选）",
    "RATES.effr": "有效联邦基金利率",
    "RATES.nominal_2y": "2 年期美国国债收益率",
    "RATES.nominal_10y": "10 年期美国国债收益率",
    "RATES.real_10y": "10 年期美国国债实际收益率",
    "RATES.breakeven_proxy_10y": "10 年期通胀补偿代理值（可选）",
    "CPI.next_release_time": "下一次 CPI 发布时间",
    "CPI.latest_initial_release": "最近一次 CPI 初值",
    "PCE.next_release_time": "下一次 PCE 发布时间",
    "PCE.latest_initial_release": "最近一次 PCE 初值",
    "NFP.next_release_time": "下一次非农发布时间",
    "NFP.latest_initial_release": "最近一次非农初值",
}

MACRO_REASON_ZH = {
    "NO_HIGH_IMPACT_EVENT_IN_48H_WINDOW": "未来 48 小时内没有白名单高影响官方事件",
    "HIGH_IMPACT_EVENT_IN_48H_WINDOW": "高影响官方事件将在 24 至 48 小时内发布",
    "TIER_A_PRIMARY": "证据来自 A 级官方一手来源",
    "FOMC_MARKET_EXPECTATION_MISSING": "FOMC 市场概率证据缺失（可选）",
    "SOURCE_DEGRADED": "部分官方来源或证据槽当前不可用",
    "PARTIAL_OFFICIAL_SOURCE_COVERAGE": "当前仅有部分官方来源覆盖，但已有证据仍支持评估",
    "ALL_OFFICIAL_SOURCES_UNAVAILABLE": "所有候选官方来源均不可用或未产出合格证据",
    "ALL_PRIMARY_MACRO_SOURCES_DISABLED": "全部主要宏观官方来源均已关闭",
    "OPTIONAL_SOURCE_UNAVAILABLE_FRED": "可选 FRED/ALFRED 来源当前不可用",
    "OPTIONAL_SOURCE_DISABLED_FRED": "可选 FRED/ALFRED 来源已关闭",
    "EVENT_CALENDAR_UNAVAILABLE": "官方事件日历当前不可用，评估仅使用其他合格官方证据",
    "RATES_EFFR_STALE": "有效联邦基金利率（EFFR）官方证据已过期",
    "MACRO_CALENDAR_REFRESH_FAILED": "官方事件日历最近一次刷新失败",
    "MACRO_EVIDENCE_REFRESH_FAILED": "官方宏观证据最近一次刷新失败",
    "OFFICIAL_EVIDENCE_COMPLETE": "必需官方证据完整",
    "HIGH_IMPACT_EVENT_NEAR": "高影响官方事件临近",
    "EVENT_IMPACT_RELEVANCE_ESCALATION": "事件类别潜在影响和利率链相关性触发上调一档",
    "ASSESSMENT_UNAVAILABLE": "当前数据不足以支持可靠评估",
}

MACRO_COMPONENT_ZH = {
    "calendar_score": "事件日历风险分",
    "proximity_score": "事件时间距离基础分",
    "impact_strength_score": "事件类别潜在影响分",
    "rate_relevance_score": "利率决策链相关性分",
    "release_cooldown_score": "发布后冷却期风险分",
    "expectation_uncertainty_score": "市场预期不确定性分",
    "evidence_quality_score": "证据质量分",
}

MACRO_INTERPRETATION_TOPIC_ZH = {
    "FOMC_POLICY": "FOMC 政策利率",
    "EFFR_ALIGNMENT": "有效联邦基金利率",
    "TREASURY_YIELD_CURVE": "2年期与10年期国债收益率",
    "REAL_YIELD_10Y": "10年期实际利率",
    "BREAKEVEN_PROXY_10Y": "10年期通胀补偿代理值",
    "CPI_RELEASE": "CPI 官方发布",
    "PCE_RELEASE": "PCE 官方发布",
    "NFP_RELEASE": "非农就业官方发布",
}

MACRO_INTERPRETATION_TOPIC_EN = {
    "FOMC_POLICY": "FOMC policy rate",
    "EFFR_ALIGNMENT": "Effective federal funds rate",
    "TREASURY_YIELD_CURVE": "2-year and 10-year Treasury yields",
    "REAL_YIELD_10Y": "10-year real yield",
    "BREAKEVEN_PROXY_10Y": "10-year inflation-compensation proxy",
    "CPI_RELEASE": "Official CPI release",
    "PCE_RELEASE": "Official PCE release",
    "NFP_RELEASE": "Official nonfarm-payroll release",
}


def initialize_app() -> None:
    """Initialize page config and shared language state. / 初始化页面配置与共享语言状态。"""
    st.set_page_config(page_title="AuPilot", page_icon=":material/finance_mode:", layout="wide", initial_sidebar_state="collapsed")
    st.session_state.setdefault("language", "zh")
    st.markdown(
        """
        <style>
        div[role="dialog"] { width: min(430px, calc(100vw - 32px)); max-width: 430px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def tr(key: str) -> str:
    """Return current-language system copy. / 返回当前语言的系统文案。"""
    return TEXT[st.session_state.get("language", "zh")][key]


def is_chinese() -> bool:
    return st.session_state.get("language", "zh") == "zh"


def localize(value: str | None) -> str:
    if value is None:
        return "—"
    return ENUM_ZH.get(value, value) if is_chinese() else value.replace("_", " ").title()


def localize_content(value: str) -> str:
    """Keep user/backend prose intact; localize only stable technical prefixes. / 保留原始内容，仅转换稳定技术前缀。"""
    if not is_chinese():
        return value
    replacements = {
        "RSI(14)=": "RSI(14)=", "Price vs 20-day average=": "价格相对20日均线=",
        "Annualized 20-day volatility=": "20日年化波动率=", "Drawdown from 60-day high=": "距60日高点回撤=",
        "Market closed; no action is generated.": "休市，不生成建议动作。",
        "Federal Reserve policy path": "美联储政策路径", "Inflation persistence": "通胀持续性",
        "Employment momentum and real yields": "就业动能与实际利率",
    }
    for source, target in replacements.items():
        if value.startswith(source):
            return value.replace(source, target, 1)
    if value.startswith("Demonstration analysis:"):
        return "演示分析：当前宏观证据存在分歧，仍需技术面确认。"
    if value.startswith("Stop the tactical plan"):
        return "若价格收盘超出情景区间，应停止当前战术计划。"
    if value.startswith("Re-evaluate if event risk"):
        return "若事件风险升高或波动显著扩大，应重新评估。"
    if value.startswith("Never reduce the protected core"):
        return "不得降低受保护核心仓，不得做空或增加杠杆。"
    return value


def localize_reason(value: str) -> str:
    if not is_chinese():
        return value.replace("_", " ").title()
    return {
        "NO_QUALIFIED_H1_TOP_REQUEST": "H1 未形成达到冻结门槛的 TOP 减仓请求",
        "NO_QUALIFIED_REQUEST_FOR_TARGET_BUCKET": "该目标日桶没有达到冻结门槛的请求",
        "QUALIFIED_MN18_H1_TOP": "MN18 H1 TOP 减仓请求已达到冻结门槛",
        "QUALIFIED_MN18_H2_BOTTOM_FIFO_REENTER": "MN18 H2 BOTTOM 请求已按 FIFO 战术库存形成回补建议",
        "QUALIFIED_TOP_AT_MINIMUM_WEIGHT": "TOP 请求已达标，但当前权重已在策略下界",
        "REJECTED_NO_FIFO_INVENTORY": "BOTTOM 请求缺少可回补的 FIFO 战术库存",
        "REJECTED_POSITION_BOUND": "请求受策略仓位边界限制",
        "H1_NOT_FIRST_STRICTLY_LATER_BUCKET": "H1 不是输入日桶之后的第一个严格未来有效日桶",
        "H1_BELOW_REGISTERED_LIFT_THRESHOLD": "H1 概率提升未达到冻结阈值",
        "QUALIFIED_H1_SIGNAL_AT_POSITION_BOUND": "H1 信号达标，但当前权重已位于允许边界",
        "QUALIFIED_H1_TOP_L1": "H1 TOP L1 信号达到冻结动作门槛",
        "QUALIFIED_H1_TOP_L2": "H1 TOP L2 信号达到冻结动作门槛",
        "QUALIFIED_H1_TOP_L3": "H1 TOP L3 信号达到冻结动作门槛",
        "QUALIFIED_H1_BOTTOM_L1": "H1 BOTTOM L1 信号达到冻结动作门槛",
        "QUALIFIED_H1_BOTTOM_L2": "H1 BOTTOM L2 信号达到冻结动作门槛",
        "QUALIFIED_H1_BOTTOM_L3": "H1 BOTTOM L3 信号达到冻结动作门槛",
        "LATEST_MARKET_BUCKET_DIFFERS_FROM_ISSUANCE": "最新行情日桶与最近发行的输入日桶不一致",
        "TECHNICAL_ISSUANCE_NOT_READY": "尚无成功的技术模型发行记录",
        "MARKET_DATA_UNAVAILABLE": "Databento 正式行情当前不可用",
        "UNKNOWN": "未知原因",
        "DATABENTO_CREDENTIAL_REQUIRED": "尚未配置必需的 Databento API key",
        "DATABENTO_MARKET_CACHE_UNAVAILABLE": "尚无新鲜且通过校验的 Databento 日线缓存",
        "FORMAL_MODEL_MANIFEST_MISSING": "尚无通过校验的正式模型清单",
        "FORMAL_MODEL_MANIFEST_INVALID": "正式模型清单、模型路径或文件 SHA 校验失败",
        "FORMAL_PREDICTION_LOG_EMPTY": "尚无事前发布的正式预测记录",
        "FORMAL_PREDICTION_LOG_INVALID": "正式预测日志或其防篡改检查点校验失败",
        "FORMAL_MARKET_COVERAGE_INCOMPLETE": "正式预测对应的历史 Databento 交易日数据不完整",
        "FORMAL_PREDICTIONS_NOT_YET_REALIZED": "正式预测尚未到达可用实际行情进行验证的日期",
    }.get(value, value)


def localize_macro_datetime(value: str | None) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if not is_chinese():
        return parsed.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    beijing = parsed.astimezone(timezone(timedelta(hours=8)))
    return f"{beijing.year}年{beijing.month}月{beijing.day}日 {beijing:%H:%M}（北京时间）"


def _macro_period(value: str | None) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return f"{parsed.year}年{parsed.month}月"
    except ValueError:
        pass
    months = {
        "January": "1月", "February": "2月", "March": "3月", "April": "4月",
        "May": "5月", "June": "6月", "July": "7月", "August": "8月",
        "September": "9月", "October": "10月", "November": "11月", "December": "12月",
    }
    parts = value.split()
    if len(parts) == 2 and parts[0] in months and parts[1].isdigit():
        return f"{parts[1]}年{months[parts[0]]}"
    return value


def _macro_date(value: str | None) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return _macro_period(value)
    return f"{parsed.year}年{parsed.month}月{parsed.day}日"


def _macro_value(claim: dict) -> float | int | str | None:
    value = claim.get("value")
    if isinstance(value, dict):
        return None
    return value


def localize_macro_claim(claim: dict | None, fallback: str) -> str:
    """Render an official macro claim in the selected language without changing its meaning."""
    if not is_chinese():
        return fallback
    if not claim:
        return localize_content(fallback)

    slot = claim.get("slot")
    value = _macro_value(claim)
    period = _macro_period(claim.get("reference_period"))
    claim_value = claim.get("value")

    if slot == "FOMC.calendar":
        scheduled = claim_value.get("scheduled_release_at_utc") if isinstance(claim_value, dict) else None
        return f"美联储官方日历显示，下一次 FOMC 利率决议定于 {localize_macro_datetime(scheduled)}。"
    if slot == "FOMC.latest_official_decision":
        if isinstance(claim_value, dict):
            lower = claim_value.get("target_lower")
            upper = claim_value.get("target_upper")
            if lower is not None and upper is not None:
                return f"美联储最近一次已接受的 FOMC 决议将联邦基金目标区间设为 {lower}% 至 {upper}%。"
        return f"美联储于{_macro_date(claim.get('reference_period'))}发布了最近一次 FOMC 利率决议声明。"
    if slot == "RATES.effr":
        return f"纽约联储数据显示，{_macro_date(claim.get('reference_period'))}的有效联邦基金利率（EFFR）为 {value:g}%。"
    if slot == "RATES.nominal_2y":
        return f"美国财政部数据显示，{_macro_date(claim.get('reference_period'))}的 2 年期美国国债收益率为 {value:g}%。"
    if slot == "RATES.nominal_10y":
        return f"美国财政部数据显示，{_macro_date(claim.get('reference_period'))}的 10 年期美国国债收益率为 {value:g}%。"
    if slot == "RATES.real_10y":
        return f"美国财政部数据显示，{_macro_date(claim.get('reference_period'))}的 10 年期美国国债实际收益率为 {value:g}%。"
    if slot == "RATES.breakeven_proxy_10y":
        return f"{_macro_date(claim.get('reference_period'))}的 10 年期名义收益率减实际收益率代理值为 {value:g} 个百分点。"
    if slot == "CPI.next_release_time":
        scheduled = claim_value.get("scheduled_release_at_utc") if isinstance(claim_value, dict) else None
        return f"美国劳工统计局官方日历显示，下一次 CPI 定于 {localize_macro_datetime(scheduled)}发布。"
    if slot == "CPI.latest_initial_release":
        return f"美国劳工统计局公布，{period}的城市消费者价格指数（CPI-U）为 {value:g} 点。"
    if slot == "PCE.next_release_time":
        scheduled = claim_value.get("scheduled_release_at_utc") if isinstance(claim_value, dict) else None
        return f"美国经济分析局官方日历显示，下一次 PCE 报告定于 {localize_macro_datetime(scheduled)}发布。"
    if slot == "PCE.latest_initial_release":
        return f"美国经济分析局已发布{period}个人收入与支出报告。"
    if slot == "NFP.next_release_time":
        scheduled = claim_value.get("scheduled_release_at_utc") if isinstance(claim_value, dict) else None
        return f"美国劳工统计局官方日历显示，下一次非农就业报告定于 {localize_macro_datetime(scheduled)}发布。"
    if slot == "NFP.latest_initial_release":
        total_millions = float(value) / 1000
        total_hundred_millions = total_millions / 100
        return f"美国劳工统计局公布，{period}非农就业总人数约为 {total_hundred_millions:.5f} 亿人。"
    return localize_content(fallback)


def localize_macro_interpretation_title(topic_key: str) -> str:
    mapping = MACRO_INTERPRETATION_TOPIC_ZH if is_chinese() else MACRO_INTERPRETATION_TOPIC_EN
    return mapping.get(topic_key, topic_key.replace("_", " ").title())


def _macro_claim_for_slot(claims_by_id: dict[str, dict], claim_ids: list[str], slot: str) -> dict | None:
    return next(
        (
            claims_by_id[claim_id]
            for claim_id in claim_ids
            if claim_id in claims_by_id and claims_by_id[claim_id].get("slot") == slot
        ),
        None,
    )


def localize_macro_official_fact(item: dict, claims_by_id: dict[str, dict]) -> str:
    if not is_chinese():
        return item["official_fact"]
    claim_ids = item.get("claim_ids", [])
    topic = item.get("topic_key")
    if topic == "TREASURY_YIELD_CURVE":
        value_2y = _macro_claim_for_slot(claims_by_id, claim_ids, "RATES.nominal_2y")
        value_10y = _macro_claim_for_slot(claims_by_id, claim_ids, "RATES.nominal_10y")
        if value_2y and value_10y:
            return (
                f"美国财政部数据显示，{_macro_date(value_2y.get('reference_period'))}的2年期国债收益率为 "
                f"{float(value_2y['value']):g}%，10年期为 {float(value_10y['value']):g}%。"
            )
    claim = next((claims_by_id.get(claim_id) for claim_id in claim_ids if claim_id in claims_by_id), None)
    return localize_macro_claim(claim, item["official_fact"])


def localize_macro_interpretation(item: dict, claims_by_id: dict[str, dict]) -> str:
    if not is_chinese():
        return item["analysis"]
    claim_ids = item.get("claim_ids", [])
    topic = item.get("topic_key")
    if topic == "FOMC_POLICY":
        return "这说明当前政策利率设定。缺少上一期决议和可比较的声明措辞时，不能只凭这一项判断政策变得更宽松还是更限制。"
    if topic == "EFFR_ALIGNMENT":
        effr = _macro_claim_for_slot(claims_by_id, claim_ids, "RATES.effr")
        fomc = _macro_claim_for_slot(claims_by_id, claim_ids, "FOMC.latest_official_decision")
        target = fomc.get("value") if fomc and isinstance(fomc.get("value"), dict) else {}
        if effr and target.get("target_lower") is not None and target.get("target_upper") is not None:
            value = float(effr["value"])
            lower = float(target["target_lower"])
            upper = float(target["target_upper"])
            position = "位于" if lower <= value <= upper else "不在"
            return f"EFFR {position}已接受的 FOMC 目标区间 {lower:g}% 至 {upper:g}% 内。这反映当前利率设定的执行情况，本身不是一次新的加息或降息。"
        return "这是实际隔夜联邦基金利率。还需要经过验证的目标区间，才能判断它是否与当前政策设定一致。"
    if topic == "TREASURY_YIELD_CURVE":
        value_2y = _macro_claim_for_slot(claims_by_id, claim_ids, "RATES.nominal_2y")
        value_10y = _macro_claim_for_slot(claims_by_id, claim_ids, "RATES.nominal_10y")
        if value_2y and value_10y and value_2y.get("reference_period") == value_10y.get("reference_period"):
            spread_bps = round((float(value_10y["value"]) - float(value_2y["value"])) * 100)
            shape = "向上倾斜" if spread_bps >= 0 else "倒挂"
            return f"10年期减2年期利差为 {spread_bps:+d} 个基点，因此该时点收益率曲线{shape}。单个时点不能说明趋势，也不能直接推出黄金价格方向。"
        return "需要同一天、同口径的2年期和10年期收益率，才能可靠解释收益率曲线形态。"
    if topic == "REAL_YIELD_10Y":
        return "这是实际收益率的单点水平。必须有前一项同口径数据才能判断它上升还是下降；本卡不会把这个点位转换成黄金价格或交易结论。"
    if topic == "BREAKEVEN_PROXY_10Y":
        return "这是10年期名义收益率减实际收益率得到的长期通胀补偿代理值，不是 CPI 或 PCE 预测。只有一个点位时不能判断通胀预期是否正在上升。"
    if topic == "CPI_RELEASE":
        return "这是 CPI 指数点位，不是月度或年度通胀率。还需要上一期同口径指数以及官方环比或同比变化，才能判断通胀升温还是降温。"
    if topic == "PCE_RELEASE":
        return "这项事实只能确认官方 PCE 报告已经发布，但没有可比较的整体或核心 PCE 变化值，因此不能判断通胀变强还是变弱。"
    if topic == "NFP_RELEASE":
        claim = _macro_claim_for_slot(claims_by_id, claim_ids, "NFP.latest_initial_release")
        total = ""
        if claim and claim.get("unit") == "thousands":
            total = f"，约 {float(claim['value']) / 1000:g} 百万人"
        return f"这是非农就业总量{total}，不是当月新增就业人数。还需要上月总量、失业率和工资变化，才能判断就业市场增强还是减弱。"
    return localize_content(item["analysis"])


def localize_macro_interpretation_status(value: str) -> str:
    keys = {
        "EXPLAINED": "macro_interpretation_explained",
        "CONTEXT_ONLY": "macro_interpretation_context",
        "INSUFFICIENT_COMPARISON": "macro_interpretation_insufficient",
    }
    return tr(keys.get(value, "macro_interpretation_insufficient"))


def localize_macro_provider(value: str) -> str:
    return MACRO_PROVIDER_ZH.get(value, value) if is_chinese() else value


def localize_macro_slot(value: str) -> str:
    return MACRO_SLOT_ZH.get(value, value) if is_chinese() else value


def localize_macro_reason(value: str) -> str:
    return MACRO_REASON_ZH.get(value, value) if is_chinese() else value


def localize_macro_component(value: str) -> str:
    return MACRO_COMPONENT_ZH.get(value, value) if is_chinese() else value


def localize_macro_risk_rule(band: dict) -> str:
    level = str(band.get("level", band.get("risk_level")))
    lower = int(band["lower_bound_hours"])
    upper = band.get("upper_bound_hours")
    if is_chinese():
        if level == "Approved":
            return f"下一事件超过 {lower} 小时，或窗口内无事件"
        if level in {"Cleared", "Caution"}:
            rule = f"基准：距发布 {lower} 至 {int(upper)} 小时"
            if band.get("assessment_unavailable_also"):
                rule += "；数据缺失时也为此档"
            if band.get("event_factor_adjustment_allowed"):
                rule += "；影响和相关性可上调"
            return rule
        if level == "Hold":
            return f"基准：距发布 0 至 {int(upper)} 小时；也可由影响和相关性上调至此"
        return f"发布后 0 至 {int(upper)} 小时"
    if level == "Approved":
        return f"Next event is over {lower}h away, or none is in the window"
    if level in {"Cleared", "Caution"}:
        rule = f"Base: {lower}h to {int(upper)}h before release"
        if band.get("assessment_unavailable_also"):
            rule += "; also used when data is missing"
        if band.get("event_factor_adjustment_allowed"):
            rule += "; impact/relevance may escalate"
        return rule
    if level == "Hold":
        return f"Base: 0h to {int(upper)}h before release; impact/relevance may also escalate here"
    return f"0h to {int(upper)}h after release"


def page_header(title_key: str, body_key: str, *, badge_key: str | None = None) -> None:
    if badge_key:
        st.badge(tr(badge_key), icon=":material/info:", color="primary")
    st.title(tr(title_key))
    st.caption(tr(body_key))


def section_header(title: str, caption: str | None = None) -> None:
    st.subheader(title)
    if caption:
        st.caption(caption)


def month_label(value) -> str:
    return f"{value.year}年{value.month}月" if is_chinese() else f"{tr(f'month_{value.month}')} {value.year}"

# Locale catalogs are intentionally data-only. Add reviewed strings to both JSON files.
