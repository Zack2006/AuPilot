"""Deterministic explanations for accepted official macro claims.

This module is downstream of the frozen macro gate. It can explain an accepted
fact, but it cannot alter evidence coverage, the five-level score, or any
technical response.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from aupilot.macro_gate.schemas import MacroClaim, MacroSummaryFact

from backend.app.schemas.macro import MacroInterpretation


def _number(claim: MacroClaim | None) -> float | None:
    if claim is None or isinstance(claim.value, (dict, list)) or claim.value is None:
        return None
    try:
        return float(claim.value)
    except (TypeError, ValueError):
        return None


def _first(by_slot: dict[str, list[MacroClaim]], slot: str) -> MacroClaim | None:
    values = by_slot.get(slot, [])
    return values[0] if values else None


def _ids(*claims: MacroClaim | None) -> list[str]:
    return list(dict.fromkeys(claim.claim_id for claim in claims if claim is not None))


def _item(
    *,
    event_type: str,
    topic_key: str,
    official_fact: str,
    analysis: str,
    status: str,
    claims: tuple[MacroClaim | None, ...],
) -> MacroInterpretation:
    claim_ids = _ids(*claims)
    return MacroInterpretation(
        interpretation_id=f"interpretation:{topic_key.lower()}:{claim_ids[0]}",
        event_type=event_type,
        topic_key=topic_key,
        official_fact=official_fact,
        analysis=analysis,
        interpretation_status=status,
        claim_ids=claim_ids,
    )


def build_macro_interpretations(
    claims: Iterable[MacroClaim],
    summary_facts: Iterable[MacroSummaryFact],
) -> list[MacroInterpretation]:
    """Explain only claim IDs already accepted into the official summary."""

    accepted_ids = {
        claim_id
        for fact in summary_facts
        for claim_id in fact.claim_ids
    }
    by_slot: dict[str, list[MacroClaim]] = defaultdict(list)
    for claim in claims:
        if claim.claim_id in accepted_ids:
            by_slot[claim.slot].append(claim)

    result: list[MacroInterpretation] = []
    fomc = _first(by_slot, "FOMC.latest_official_decision")
    if fomc is not None:
        target = fomc.value if isinstance(fomc.value, dict) else {}
        lower = target.get("target_lower")
        upper = target.get("target_upper")
        fact = fomc.display_text
        if lower is not None and upper is not None:
            fact = f"The latest accepted FOMC decision set the federal funds target range at {lower}%-{upper}%."
        result.append(_item(
            event_type="FOMC",
            topic_key="FOMC_POLICY",
            official_fact=fact,
            analysis=(
                "This establishes the current policy-rate setting. Without the prior decision and comparable "
                "statement language, this fact alone cannot show whether policy became more or less restrictive."
            ),
            status="CONTEXT_ONLY",
            claims=(fomc,),
        ))

    effr = _first(by_slot, "RATES.effr")
    if effr is not None:
        effr_value = _number(effr)
        target = fomc.value if fomc is not None and isinstance(fomc.value, dict) else {}
        try:
            lower = float(target["target_lower"])
            upper = float(target["target_upper"])
        except (KeyError, TypeError, ValueError):
            lower = upper = None
        if effr_value is not None and lower is not None and upper is not None:
            position = "inside" if lower <= effr_value <= upper else "outside"
            analysis = (
                f"EFFR is {position} the accepted FOMC target range of {lower:g}%-{upper:g}%. "
                "This describes implementation of the current rate setting; it is not by itself a new hike or cut."
            )
            status = "EXPLAINED"
        else:
            analysis = (
                "This is the effective overnight federal funds rate. A verified target range is also required "
                "before the system can explain whether the operational rate is aligned with policy."
            )
            status = "INSUFFICIENT_COMPARISON"
        result.append(_item(
            event_type="RATES",
            topic_key="EFFR_ALIGNMENT",
            official_fact=effr.display_text,
            analysis=analysis,
            status=status,
            claims=(effr, fomc),
        ))

    nominal_2y = _first(by_slot, "RATES.nominal_2y")
    nominal_10y = _first(by_slot, "RATES.nominal_10y")
    if nominal_2y is not None and nominal_10y is not None:
        value_2y = _number(nominal_2y)
        value_10y = _number(nominal_10y)
        same_period = nominal_2y.reference_period == nominal_10y.reference_period
        fact = f"{nominal_2y.display_text} {nominal_10y.display_text}"
        if value_2y is not None and value_10y is not None and same_period:
            spread_bps = round((value_10y - value_2y) * 100)
            shape = "upward sloping" if spread_bps >= 0 else "inverted"
            analysis = (
                f"The 10-year minus 2-year spread is {spread_bps:+d} basis points, so the curve is {shape} "
                "at this observation. One point does not establish a trend or a gold-price direction."
            )
            status = "EXPLAINED"
        else:
            analysis = (
                "Comparable 2-year and 10-year observations from the same date are required before the curve "
                "shape can be explained reliably."
            )
            status = "INSUFFICIENT_COMPARISON"
        result.append(_item(
            event_type="RATES",
            topic_key="TREASURY_YIELD_CURVE",
            official_fact=fact,
            analysis=analysis,
            status=status,
            claims=(nominal_2y, nominal_10y),
        ))

    real_10y = _first(by_slot, "RATES.real_10y")
    if real_10y is not None:
        result.append(_item(
            event_type="RATES",
            topic_key="REAL_YIELD_10Y",
            official_fact=real_10y.display_text,
            analysis=(
                "This is a real-yield level. A prior comparable observation is needed to say whether it rose or "
                "fell; this single point is not converted into a gold-price or trading conclusion."
            ),
            status="INSUFFICIENT_COMPARISON",
            claims=(real_10y,),
        ))

    breakeven = _first(by_slot, "RATES.breakeven_proxy_10y")
    if breakeven is not None:
        result.append(_item(
            event_type="RATES",
            topic_key="BREAKEVEN_PROXY_10Y",
            official_fact=breakeven.display_text,
            analysis=(
                "This is a nominal-minus-real yield proxy for long-run inflation compensation. It is not a CPI "
                "or PCE forecast, and one observation cannot show whether inflation expectations are rising."
            ),
            status="INSUFFICIENT_COMPARISON",
            claims=(breakeven,),
        ))

    cpi = _first(by_slot, "CPI.latest_initial_release")
    if cpi is not None:
        result.append(_item(
            event_type="CPI",
            topic_key="CPI_RELEASE",
            official_fact=cpi.display_text,
            analysis=(
                "This is a CPI index level, not a monthly or annual inflation rate. A prior comparable index and "
                "the official month-over-month or year-over-year change are needed before calling inflation hotter "
                "or cooler."
            ),
            status="INSUFFICIENT_COMPARISON",
            claims=(cpi,),
        ))

    pce = _first(by_slot, "PCE.latest_initial_release")
    if pce is not None:
        result.append(_item(
            event_type="PCE",
            topic_key="PCE_RELEASE",
            official_fact=pce.display_text,
            analysis=(
                "This claim confirms that the official PCE report was published, but it does not contain a "
                "comparable headline or core PCE change. The system therefore cannot label inflation stronger or weaker."
            ),
            status="INSUFFICIENT_COMPARISON",
            claims=(pce,),
        ))

    nfp = _first(by_slot, "NFP.latest_initial_release")
    if nfp is not None:
        value = _number(nfp)
        total = f" about {value / 1000:g} million people" if value is not None and nfp.unit == "thousands" else ""
        result.append(_item(
            event_type="NFP",
            topic_key="NFP_RELEASE",
            official_fact=nfp.display_text,
            analysis=(
                f"This is the total nonfarm payroll employment level{total}, not jobs added during the month. "
                "The prior total, unemployment rate, and wage change are needed before describing the labor market "
                "as stronger or weaker."
            ),
            status="INSUFFICIENT_COMPARISON",
            claims=(nfp,),
        ))

    return result
