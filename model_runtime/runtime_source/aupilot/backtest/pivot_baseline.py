from __future__ import annotations

from dataclasses import dataclass
from html import escape
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd

from aupilot.backtest.metrics import maximum_drawdown


@dataclass(frozen=True)
class PivotReplayPolicy:
    initial_nav_usd: float = 1_000_000.0
    initial_gold_weight: float = 1.0
    weight_after_top: float = 0.5
    weight_after_bottom: float = 1.0
    cost_bps_per_side: float = 2.0

    def __post_init__(self) -> None:
        weights = (
            self.initial_gold_weight,
            self.weight_after_top,
            self.weight_after_bottom,
        )
        if self.initial_nav_usd <= 0.0:
            raise ValueError("initial_nav_usd must be positive")
        if any(not 0.0 <= value <= 1.0 for value in weights):
            raise ValueError("gold weights must be between zero and one")
        if self.weight_after_top > self.weight_after_bottom:
            raise ValueError("TOP weight cannot exceed BOTTOM weight")
        if self.initial_gold_weight != 1.0:
            raise ValueError("minimal baseline must start at 100% gold")
        if self.cost_bps_per_side < 0.0:
            raise ValueError("cost_bps_per_side must be non-negative")


@dataclass(frozen=True)
class PivotReplayResult:
    orders: pd.DataFrame
    fills: pd.DataFrame
    daily_portfolio: pd.DataFrame
    cycles: pd.DataFrame
    metrics: dict[str, float | int | str | bool]


def validate_daily_ohlc(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "open", "high", "low", "close"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"Missing daily OHLC columns: {sorted(missing)}")
    frame = daily.loc[:, ["trade_date", "open", "high", "low", "close"]].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.date
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values("trade_date", kind="stable").reset_index(drop=True)
    if frame.empty:
        raise ValueError("Daily OHLC input is empty")
    if frame["trade_date"].duplicated().any():
        raise ValueError("Duplicate trade_date")
    values = frame.loc[:, ["open", "high", "low", "close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError("Daily OHLC values must be finite and positive")
    body_low = frame[["open", "close"]].min(axis=1)
    body_high = frame[["open", "close"]].max(axis=1)
    if (frame["low"] > body_low).any() or (frame["high"] < body_high).any():
        raise ValueError("Invalid daily OHLC ordering")
    return frame


def enrich_pivot_events(
    daily: pd.DataFrame,
    raw_events: pd.DataFrame,
    *,
    reference_price_column: str = "close",
) -> pd.DataFrame:
    frame = validate_daily_ohlc(daily)
    if reference_price_column not in {"open", "close"}:
        raise ValueError("Pivot reference price must be open or close")
    columns = (
        "event_id",
        "sequence_id",
        "event_type",
        "pivot_date",
        "pivot_close",
        "pivot_open",
        "pivot_high",
        "pivot_low",
        "pivot_marker_price",
        "confirmation_date",
        "confirmation_close",
        "confirmation_delay_sessions",
        "execution_date",
        "execution_open",
        "leg_amplitude_fraction",
    )
    if raw_events.empty:
        return pd.DataFrame(columns=columns)
    required = {
        "sequence_id",
        "event_type",
        "event_at",
        "event_price",
        "confirmed_at",
        "confirmation_price",
        "confirmation_delay_sessions",
    }
    missing = required - set(raw_events.columns)
    if missing:
        raise ValueError(f"Missing raw pivot columns: {sorted(missing)}")

    positions = {value: index for index, value in enumerate(frame["trade_date"])}
    records: list[dict[str, object]] = []
    previous_reference_price: float | None = None
    for row in raw_events.sort_values("sequence_id", kind="stable").itertuples(index=False):
        pivot_date = pd.Timestamp(row.event_at).date()
        confirmation_date = pd.Timestamp(row.confirmed_at).date()
        if pivot_date not in positions or confirmation_date not in positions:
            raise ValueError("Pivot event date is outside the daily OHLC tape")
        pivot_index = positions[pivot_date]
        confirmation_index = positions[confirmation_date]
        if confirmation_index <= pivot_index:
            raise ValueError("A pivot must be confirmed after its extreme date")
        pivot = frame.iloc[pivot_index]
        confirmation = frame.iloc[confirmation_index]
        execution_index = confirmation_index + 1
        execution_date = (
            frame.iloc[execution_index]["trade_date"]
            if execution_index < len(frame)
            else None
        )
        execution_open = (
            float(frame.iloc[execution_index]["open"])
            if execution_index < len(frame)
            else np.nan
        )
        event_type = str(row.event_type)
        if event_type not in {"TOP", "BOTTOM"}:
            raise ValueError(f"Unsupported pivot event type: {event_type}")
        pivot_close = float(pivot["close"])
        pivot_reference_price = float(pivot[reference_price_column])
        if not np.isclose(
            pivot_reference_price,
            float(row.event_price),
            rtol=0.0,
            atol=1e-10,
        ):
            raise ValueError("Raw pivot event price does not match daily reference price")
        if previous_reference_price is None:
            leg_amplitude = np.nan
        elif event_type == "TOP":
            leg_amplitude = pivot_reference_price / previous_reference_price - 1.0
        else:
            leg_amplitude = previous_reference_price / pivot_reference_price - 1.0
        records.append(
            {
                "event_id": f"R03A-PIVOT-{int(row.sequence_id):04d}",
                "sequence_id": int(row.sequence_id),
                "event_type": event_type,
                "pivot_date": pivot_date,
                "pivot_close": pivot_close,
                "pivot_open": float(pivot["open"]),
                "pivot_high": float(pivot["high"]),
                "pivot_low": float(pivot["low"]),
                "pivot_marker_price": float(
                    pivot["high"] if event_type == "TOP" else pivot["low"]
                ),
                "confirmation_date": confirmation_date,
                "confirmation_close": float(confirmation["close"]),
                "confirmation_delay_sessions": int(row.confirmation_delay_sessions),
                "execution_date": execution_date,
                "execution_open": execution_open,
                "leg_amplitude_fraction": leg_amplitude,
            }
        )
        previous_reference_price = pivot_reference_price
    events = pd.DataFrame.from_records(records, columns=columns)
    event_types = events["event_type"].tolist()
    if any(left == right for left, right in pairwise(event_types)):
        raise AssertionError("Pivot events do not alternate")
    amplitudes = pd.to_numeric(events["leg_amplitude_fraction"], errors="coerce").dropna()
    if not amplitudes.ge(0.05 - 1e-12).all():
        raise AssertionError("Adjacent pivot amplitude is below the registered 5%")
    return events


STRUCTURAL_ORACLE_ROLE = "STRUCTURAL_ORACLE_LABEL_CAPACITY"


def make_structural_oracle_events(
    daily: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Place hindsight structural labels at the matching pivot-day Open.

    This transformation is deliberately non-causal. It measures whether a frozen
    mathematical label set has useful trading geometry before any model is fitted.
    It must never be presented as an executable strategy backtest.
    """
    frame = validate_daily_ohlc(daily)
    required = {
        "pivot_date",
        "pivot_open",
        "confirmation_date",
        "execution_date",
        "execution_open",
    }
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Missing structural Oracle event columns: {sorted(missing)}")
    oracle = events.copy()
    pivot_dates = pd.to_datetime(oracle["pivot_date"]).dt.date
    confirmation_dates = pd.to_datetime(oracle["confirmation_date"]).dt.date
    if pivot_dates.isna().any() or confirmation_dates.isna().any():
        raise ValueError("Structural Oracle dates must be present")
    if pivot_dates.duplicated().any():
        raise ValueError("Structural Oracle pivot dates must be unique")
    if any(
        confirmation_date <= pivot_date
        for pivot_date, confirmation_date in zip(
            pivot_dates,
            confirmation_dates,
            strict=True,
        )
    ):
        raise ValueError("Structural Oracle requires labels confirmed after pivot dates")

    open_by_date = frame.set_index("trade_date")["open"]
    missing_dates = sorted(set(pivot_dates) - set(open_by_date.index))
    if missing_dates:
        raise ValueError(f"Structural Oracle pivot dates are outside daily tape: {missing_dates}")
    actual_pivot_opens = pivot_dates.map(open_by_date).astype(float)
    recorded_pivot_opens = pd.to_numeric(oracle["pivot_open"], errors="coerce")
    if not np.isfinite(recorded_pivot_opens.to_numpy(dtype=float)).all() or not np.allclose(
        actual_pivot_opens.to_numpy(dtype=float),
        recorded_pivot_opens.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("Structural Oracle pivot Open differs from daily tape")

    oracle["causal_execution_date"] = oracle["execution_date"]
    oracle["causal_execution_open"] = oracle["execution_open"]
    oracle["execution_date"] = pivot_dates
    oracle["execution_open"] = actual_pivot_opens.to_numpy(dtype=float)
    oracle["replay_role"] = STRUCTURAL_ORACLE_ROLE
    oracle["tradable"] = False
    oracle["future_label_used"] = True
    return oracle


def build_daily_state_tape(
    daily: pd.DataFrame,
    state_tape: pd.DataFrame,
    events: pd.DataFrame,
    *,
    open_candidate_at: object | None,
) -> pd.DataFrame:
    frame = validate_daily_ohlc(daily)
    if len(frame) != len(state_tape):
        raise ValueError("State tape length does not match daily OHLC")
    output = frame.copy()
    output["active_trend"] = state_tape["active_trend"].astype(str).to_numpy()
    output["causal_episode_id"] = state_tape["causal_episode_id"].astype(int).to_numpy()
    output["event_group_id"] = output["causal_episode_id"].map(
        lambda value: f"r03-pivot-episode:{value:04d}"
    )
    output["candidate_type"] = state_tape["candidate_type"].to_numpy()
    output["candidate_at"] = state_tape["candidate_at"].to_numpy()
    output["candidate_price"] = state_tape["candidate_price"].to_numpy()
    output["structural_label"] = pd.Series([pd.NA] * len(output), dtype="string")
    output["label"] = pd.Series(["NORMAL"] * len(output), dtype="string")
    output["label_status"] = "MATURED_NONE"
    output["label_available_at"] = output["trade_date"].astype(object)
    unknown = output["active_trend"].eq("UNKNOWN")
    output.loc[unknown, "label"] = pd.NA
    output.loc[unknown, "label_status"] = "INIT_UNRESOLVED"
    output.loc[unknown, "label_available_at"] = pd.NaT

    dates = output["trade_date"].tolist()
    candidate_dates = [
        None if pd.isna(value) else pd.Timestamp(value).date()
        for value in output["candidate_at"]
    ]
    for index, trade_date in enumerate(dates):
        if unknown.iloc[index] or candidate_dates[index] != trade_date:
            continue
        resolution_index = next(
            (
                future_index
                for future_index in range(index + 1, len(output))
                if candidate_dates[future_index] != trade_date
            ),
            None,
        )
        if resolution_index is None:
            output.loc[index, "label"] = pd.NA
            output.loc[index, "label_status"] = "RIGHT_CENSORED"
            output.loc[index, "label_available_at"] = pd.NaT
        else:
            output.loc[index, "label_available_at"] = dates[resolution_index]

    for event in events.itertuples(index=False):
        mask = output["trade_date"].eq(event.pivot_date)
        if int(mask.sum()) != 1:
            raise AssertionError("Pivot date is not unique in state tape")
        output.loc[mask, "structural_label"] = event.event_type
        output.loc[mask, "label"] = event.event_type
        output.loc[mask, "label_status"] = "MATURED_PIVOT"
        output.loc[mask, "label_available_at"] = event.confirmation_date

    censored = output.loc[output["label_status"].eq("RIGHT_CENSORED")]
    if open_candidate_at is not None and not pd.isna(open_candidate_at):
        candidate_date = pd.Timestamp(open_candidate_at).date()
        if len(censored) != 1 or censored.iloc[0]["trade_date"] != candidate_date:
            raise AssertionError("Terminal right-censored candidate does not match path state")
    elif not censored.empty:
        raise AssertionError("State tape has a censored candidate but path state does not")
    matured = output["label_status"].isin({"MATURED_NONE", "MATURED_PIVOT"})
    if output.loc[matured, "label_available_at"].isna().any():
        raise AssertionError("A matured daily label is missing its availability date")
    return output


def _target_trade_notional(
    *,
    nav_before: float,
    gold_value_before: float,
    target_weight: float,
    cost_rate: float,
) -> float:
    raw = target_weight * nav_before - gold_value_before
    if abs(raw) <= 1e-10:
        return 0.0
    if raw > 0.0:
        return raw / (1.0 + target_weight * cost_rate)
    denominator = 1.0 - target_weight * cost_rate
    if denominator <= 0.0:
        raise ValueError("Cost rate is incompatible with target weight")
    return raw / denominator


def _build_cycles(events: pd.DataFrame, orders: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "cycle_id",
        "top_event_id",
        "bottom_event_id",
        "reduce_execution_date",
        "reduce_execution_open",
        "restore_execution_date",
        "restore_execution_open",
        "sessions_underweight",
        "execution_price_change",
        "completed",
    )
    if events.empty or orders.empty:
        return pd.DataFrame(columns=columns)
    filled_ids = set(orders.loc[orders["status"].eq("FILLED"), "event_id"])
    positions = {value: index for index, value in enumerate(daily["trade_date"])}
    pending_top: object | None = None
    records: list[dict[str, object]] = []
    for event in events.sort_values("sequence_id", kind="stable").itertuples(index=False):
        if event.event_type == "TOP" and event.event_id in filled_ids:
            if pending_top is not None:
                records.append(
                    {
                        "cycle_id": len(records) + 1,
                        "top_event_id": pending_top.event_id,
                        "bottom_event_id": None,
                        "reduce_execution_date": pending_top.execution_date,
                        "reduce_execution_open": pending_top.execution_open,
                        "restore_execution_date": None,
                        "restore_execution_open": np.nan,
                        "sessions_underweight": len(daily)
                        - positions[pending_top.execution_date],
                        "execution_price_change": np.nan,
                        "completed": False,
                    }
                )
            pending_top = event
            continue
        if (
            event.event_type == "BOTTOM"
            and event.event_id in filled_ids
            and pending_top is not None
        ):
            records.append(
                {
                    "cycle_id": len(records) + 1,
                    "top_event_id": pending_top.event_id,
                    "bottom_event_id": event.event_id,
                    "reduce_execution_date": pending_top.execution_date,
                    "reduce_execution_open": pending_top.execution_open,
                    "restore_execution_date": event.execution_date,
                    "restore_execution_open": event.execution_open,
                    "sessions_underweight": positions[event.execution_date]
                    - positions[pending_top.execution_date],
                    "execution_price_change": event.execution_open
                    / pending_top.execution_open
                    - 1.0,
                    "completed": True,
                }
            )
            pending_top = None
    if pending_top is not None:
        records.append(
            {
                "cycle_id": len(records) + 1,
                "top_event_id": pending_top.event_id,
                "bottom_event_id": None,
                "reduce_execution_date": pending_top.execution_date,
                "reduce_execution_open": pending_top.execution_open,
                "restore_execution_date": None,
                "restore_execution_open": np.nan,
                "sessions_underweight": len(daily) - positions[pending_top.execution_date],
                "execution_price_change": np.nan,
                "completed": False,
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def _run_pivot_replay(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    policy: PivotReplayPolicy | None = None,
    *,
    signal_date_column: str,
) -> PivotReplayResult:
    policy = PivotReplayPolicy() if policy is None else policy
    frame = validate_daily_ohlc(daily)
    if signal_date_column not in events.columns:
        raise ValueError(f"Missing replay signal date column: {signal_date_column}")
    schedule: dict[object, object] = {}
    for event in events.itertuples(index=False):
        if pd.isna(event.execution_date):
            continue
        if event.execution_date in schedule:
            raise ValueError("Multiple pivot executions share one daily Open")
        schedule[event.execution_date] = event

    first_open = float(frame.iloc[0]["open"])
    gold_quantity = policy.initial_nav_usd * policy.initial_gold_weight / first_open
    cash = policy.initial_nav_usd - gold_quantity * first_open
    benchmark_quantity = policy.initial_nav_usd / first_open
    current_target = policy.initial_gold_weight
    cost_rate = policy.cost_bps_per_side / 10_000.0
    order_records: list[dict[str, object]] = []
    fill_records: list[dict[str, object]] = []
    daily_records: list[dict[str, object]] = []

    for row in frame.itertuples(index=False):
        trade_date = row.trade_date
        open_price = float(row.open)
        nav_before = cash + gold_quantity * open_price
        executed_event_id: str | None = None
        event = schedule.get(trade_date)
        if event is not None:
            target_weight = (
                policy.weight_after_top
                if event.event_type == "TOP"
                else policy.weight_after_bottom
            )
            gold_value_before = gold_quantity * open_price
            trade_notional = _target_trade_notional(
                nav_before=nav_before,
                gold_value_before=gold_value_before,
                target_weight=target_weight,
                cost_rate=cost_rate,
            )
            fee = abs(trade_notional) * cost_rate
            side = "NONE"
            status = "NO_POSITION_CHANGE"
            if abs(trade_notional) > 1e-8:
                side = "BUY" if trade_notional > 0.0 else "SELL"
                gold_quantity += trade_notional / open_price
                cash -= trade_notional + fee
                status = "FILLED"
                executed_event_id = str(event.event_id)
                fill_records.append(
                    {
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "signal_date": getattr(event, signal_date_column),
                        "fill_date": trade_date,
                        "fill_price": open_price,
                        "side": side,
                        "trade_notional_usd": trade_notional,
                        "fee_usd": fee,
                        "target_weight": target_weight,
                    }
                )
            order_records.append(
                {
                    "event_id": event.event_id,
                    "sequence_id": int(event.sequence_id),
                    "event_type": event.event_type,
                    "signal_date": getattr(event, signal_date_column),
                    "execution_date": trade_date,
                    "previous_target_weight": current_target,
                    "target_weight": target_weight,
                    "side": side,
                    "trade_notional_usd": trade_notional,
                    "fee_usd": fee,
                    "status": status,
                }
            )
            current_target = target_weight
        close_price = float(row.close)
        nav = cash + gold_quantity * close_price
        benchmark_nav = benchmark_quantity * close_price
        if cash < -1e-6 or gold_quantity < -1e-10 or nav <= 0.0:
            raise AssertionError("Long-only replay violated cash, quantity, or NAV bounds")
        daily_records.append(
            {
                "trade_date": trade_date,
                "open": open_price,
                "close": close_price,
                "cash": cash,
                "gold_quantity": gold_quantity,
                "target_weight": current_target,
                "actual_gold_weight_close": gold_quantity * close_price / nav,
                "nav": nav,
                "benchmark_nav": benchmark_nav,
                "executed_event_id": executed_event_id,
            }
        )

    for event in events.loc[events["execution_date"].isna()].itertuples(index=False):
        target_weight = (
            policy.weight_after_top
            if event.event_type == "TOP"
            else policy.weight_after_bottom
        )
        order_records.append(
            {
                "event_id": event.event_id,
                "sequence_id": int(event.sequence_id),
                "event_type": event.event_type,
                "signal_date": getattr(event, signal_date_column),
                "execution_date": None,
                "previous_target_weight": np.nan,
                "target_weight": target_weight,
                "side": "NONE",
                "trade_notional_usd": 0.0,
                "fee_usd": 0.0,
                "status": "UNFILLED_RIGHT_EDGE",
            }
        )

    order_columns = (
        "event_id",
        "sequence_id",
        "event_type",
        "signal_date",
        "execution_date",
        "previous_target_weight",
        "target_weight",
        "side",
        "trade_notional_usd",
        "fee_usd",
        "status",
    )
    fill_columns = (
        "event_id",
        "event_type",
        "signal_date",
        "fill_date",
        "fill_price",
        "side",
        "trade_notional_usd",
        "fee_usd",
        "target_weight",
    )
    orders = pd.DataFrame.from_records(
        order_records,
        columns=order_columns,
    ).sort_values(
        "sequence_id", kind="stable", ignore_index=True
    )
    fills = pd.DataFrame.from_records(fill_records, columns=fill_columns)
    daily_portfolio = pd.DataFrame.from_records(daily_records)
    cycles = _build_cycles(events, orders, frame)
    nav_with_start = pd.concat(
        [pd.Series([policy.initial_nav_usd]), daily_portfolio["nav"]],
        ignore_index=True,
    )
    benchmark_with_start = pd.concat(
        [pd.Series([policy.initial_nav_usd]), daily_portfolio["benchmark_nav"]],
        ignore_index=True,
    )
    final_nav = float(daily_portfolio.iloc[-1]["nav"])
    final_benchmark = float(daily_portfolio.iloc[-1]["benchmark_nav"])
    metrics: dict[str, float | int | str | bool] = {
        "initial_nav_usd": policy.initial_nav_usd,
        "final_nav_usd": final_nav,
        "benchmark_final_nav_usd": final_benchmark,
        "strategy_total_return": final_nav / policy.initial_nav_usd - 1.0,
        "benchmark_total_return": final_benchmark / policy.initial_nav_usd - 1.0,
        "terminal_nav_ratio_vs_benchmark": final_nav / final_benchmark - 1.0,
        "strategy_max_drawdown": maximum_drawdown(nav_with_start),
        "benchmark_max_drawdown": maximum_drawdown(benchmark_with_start),
        "signal_events": len(events),
        "filled_trades": len(fills),
        "completed_cycles": int(cycles["completed"].sum()) if not cycles.empty else 0,
        "underweight_sessions": int(daily_portfolio["target_weight"].lt(1.0).sum()),
        "turnover_on_initial_nav": (
            float(fills["trade_notional_usd"].abs().sum()) / policy.initial_nav_usd
            if not fills.empty
            else 0.0
        ),
        "fees_paid_usd": float(fills["fee_usd"].sum()) if not fills.empty else 0.0,
    }
    return PivotReplayResult(
        orders=orders,
        fills=fills,
        daily_portfolio=daily_portfolio,
        cycles=cycles,
        metrics=metrics,
    )


def run_confirmed_pivot_replay(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    policy: PivotReplayPolicy | None = None,
) -> PivotReplayResult:
    return _run_pivot_replay(
        daily,
        events,
        policy,
        signal_date_column="confirmation_date",
    )


def run_structural_oracle_replay(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    policy: PivotReplayPolicy | None = None,
) -> PivotReplayResult:
    oracle_events = make_structural_oracle_events(daily, events)
    replay = _run_pivot_replay(
        daily,
        oracle_events,
        policy,
        signal_date_column="pivot_date",
    )
    flags = {
        "replay_role": STRUCTURAL_ORACLE_ROLE,
        "tradable": False,
        "future_label_used": True,
    }
    return PivotReplayResult(
        orders=replay.orders.assign(**flags),
        fills=replay.fills.assign(**flags),
        daily_portfolio=replay.daily_portfolio,
        cycles=replay.cycles.assign(**flags),
        metrics={**replay.metrics, **flags},
    )


def render_pivot_svg(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    destination: str | Path,
    *,
    width: int = 4800,
    height: int = 1800,
    chart_title: str = "AuPilot — 5% Daily-Close Pivot Baseline",
    chart_subtitle: str = (
        "Development only · confirmation at close · execution at next daily Open"
    ),
    chart_description: str = (
        "Development-only COMEX gold daily close with red TOP and green BOTTOM "
        "structural pivots, confirmation circles, and execution diamonds."
    ),
    show_causal_audit: bool = True,
) -> dict[str, int]:
    frame = validate_daily_ohlc(daily)
    if width < 1200 or height < 700:
        raise ValueError("SVG dimensions are too small for the full-history chart")
    left, right, top, bottom = 220.0, 100.0, 150.0, 170.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    prices_min = float(frame["low"].min())
    prices_max = float(frame["high"].max())
    padding = max((prices_max - prices_min) * 0.04, prices_max * 0.01)
    y_min = prices_min - padding
    y_max = prices_max + padding
    positions = {value: index for index, value in enumerate(frame["trade_date"])}

    def x_position(index: int) -> float:
        return left + index / max(len(frame) - 1, 1) * plot_width

    def y_position(price: float) -> float:
        return top + (y_max - price) / (y_max - y_min) * plot_height

    close_path = " ".join(
        f"{'M' if index == 0 else 'L'} {x_position(index):.2f} "
        f"{y_position(float(row.close)):.2f}"
        for index, row in enumerate(frame.itertuples(index=False))
    )
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(chart_title)}</title>',
        f'<desc id="desc">{escape(chart_description)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="220" y="62" font-family="Arial, sans-serif" font-size="38" '
        f'font-weight="600" fill="#111827">{escape(chart_title)}</text>',
        '<text x="220" y="105" font-family="Arial, sans-serif" font-size="24" '
        f'fill="#4b5563">{escape(chart_subtitle)}</text>',
    ]
    for value in np.linspace(y_min, y_max, 7):
        y_value = y_position(float(value))
        parts.append(
            f'<line x1="{left:.2f}" y1="{y_value:.2f}" x2="{width-right:.2f}" '
            'y2="{:.2f}" stroke="#e5e7eb" stroke-width="1"/>'.format(y_value)
        )
        parts.append(
            f'<text x="{left-24:.2f}" y="{y_value+9:.2f}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="22" fill="#4b5563">${value:,.0f}</text>'
        )
    first_index_by_year: dict[int, int] = {}
    for index, value in enumerate(frame["trade_date"]):
        first_index_by_year.setdefault(value.year, index)
    years = sorted(first_index_by_year)
    year_step = 1 if len(years) <= 10 else 2
    for year in years[::year_step]:
        x_value = x_position(first_index_by_year[year])
        parts.append(
            f'<line x1="{x_value:.2f}" y1="{top:.2f}" x2="{x_value:.2f}" '
            f'y2="{height-bottom:.2f}" stroke="#f3f4f6" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x_value:.2f}" y="{height-bottom+46:.2f}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="22" fill="#4b5563">{year}</text>'
        )
    parts.extend(
        [
            f'<path d="{close_path}" fill="none" stroke="#374151" stroke-width="3" '
            'stroke-linejoin="round" stroke-linecap="round"/>',
            f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" '
            f'y2="{height-bottom}" stroke="#6b7280" stroke-width="2"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" '
            'stroke="#6b7280" stroke-width="2"/>',
        ]
    )
    top_markers = 0
    bottom_markers = 0
    confirmation_markers = 0
    execution_markers = 0
    for event in events.itertuples(index=False):
        pivot_index = positions[event.pivot_date]
        confirmation_index = positions[event.confirmation_date]
        pivot_x = x_position(pivot_index)
        pivot_y = y_position(float(event.pivot_marker_price))
        confirmation_x = x_position(confirmation_index)
        confirmation_y = y_position(float(event.confirmation_close))
        color = "#c62828" if event.event_type == "TOP" else "#148b4a"
        if show_causal_audit:
            parts.append(
                f'<line x1="{pivot_x:.2f}" y1="{pivot_y:.2f}" '
                f'x2="{confirmation_x:.2f}" y2="{confirmation_y:.2f}" '
                f'stroke="{color}" stroke-width="1.5" '
                'stroke-dasharray="8 7" opacity="0.45"/>'
            )
        if event.event_type == "TOP":
            points = (
                f"{pivot_x-12:.2f},{pivot_y-16:.2f} "
                f"{pivot_x+12:.2f},{pivot_y-16:.2f} {pivot_x:.2f},{pivot_y+11:.2f}"
            )
            parts.append(
                f'<polygon class="pivot top" points="{points}" fill="{color}"><title>'
                f'TOP {escape(str(event.pivot_date))} · ${event.pivot_marker_price:,.2f}'
                '</title></polygon>'
            )
            top_markers += 1
        else:
            points = (
                f"{pivot_x-12:.2f},{pivot_y+16:.2f} "
                f"{pivot_x+12:.2f},{pivot_y+16:.2f} {pivot_x:.2f},{pivot_y-11:.2f}"
            )
            parts.append(
                f'<polygon class="pivot bottom" points="{points}" fill="{color}"><title>'
                f'BOTTOM {escape(str(event.pivot_date))} · ${event.pivot_marker_price:,.2f}'
                '</title></polygon>'
            )
            bottom_markers += 1
        if show_causal_audit:
            parts.append(
                f'<circle class="confirmation" cx="{confirmation_x:.2f}" '
                f'cy="{confirmation_y:.2f}" r="9" fill="#ffffff" '
                f'stroke="{color}" stroke-width="4"><title>Confirmed '
                f'{escape(str(event.confirmation_date))}</title></circle>'
            )
            confirmation_markers += 1
        if show_causal_audit and not pd.isna(event.execution_date):
            execution_index = positions[event.execution_date]
            execution_x = x_position(execution_index)
            execution_y = y_position(float(event.execution_open))
            diamond = (
                f"{execution_x:.2f},{execution_y-10:.2f} "
                f"{execution_x+10:.2f},{execution_y:.2f} "
                f"{execution_x:.2f},{execution_y+10:.2f} "
                f"{execution_x-10:.2f},{execution_y:.2f}"
            )
            parts.append(
                f'<polygon class="execution" points="{diamond}" fill="#1d4ed8"><title>'
                f'Executed {escape(str(event.execution_date))} Open '
                f'${event.execution_open:,.2f}</title></polygon>'
            )
            execution_markers += 1
    legend_y = height - 58
    parts.extend(
        [
            f'<polygon points="{left},{legend_y-8} {left+24},{legend_y-8} '
            f'{left+12},{legend_y+16}" fill="#c62828"/>',
            f'<text x="{left+38}" y="{legend_y+8}" font-family="Arial, sans-serif" '
            'font-size="22" fill="#111827">TOP pivot (red ▼)</text>',
            f'<polygon points="{left+330},{legend_y+14} {left+354},{legend_y+14} '
            f'{left+342},{legend_y-10}" fill="#148b4a"/>',
            f'<text x="{left+368}" y="{legend_y+8}" font-family="Arial, sans-serif" '
            'font-size="22" fill="#111827">BOTTOM pivot (green ▲)</text>',
        ]
    )
    if show_causal_audit:
        parts.extend(
            [
            f'<circle cx="{left+760}" cy="{legend_y+1}" r="9" fill="#ffffff" '
            'stroke="#6b7280" stroke-width="4"/>',
            f'<text x="{left+780}" y="{legend_y+8}" font-family="Arial, sans-serif" '
            'font-size="22" fill="#111827">confirmation</text>',
            f'<polygon points="{left+1035},{legend_y-10} {left+1045},{legend_y+1} '
            f'{left+1035},{legend_y+12} {left+1025},{legend_y+1}" fill="#1d4ed8"/>',
            f'<text x="{left+1058}" y="{legend_y+8}" font-family="Arial, sans-serif" '
            'font-size="22" fill="#111827">next-Open execution</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="{width/2:.2f}" y="{height/2:.2f}" text-anchor="middle" '
            'font-family="Arial, sans-serif" font-size="72" fill="#9ca3af" '
            'opacity="0.16" transform="rotate(-16 '
            f'{width/2:.2f} {height/2:.2f})">STRUCTURAL / ORACLE PIVOTS — '
            'NOT TRADEABLE AT PIVOT DATE</text>',
            '</svg>',
        ]
    )
    svg = "\n".join(parts) + "\n"
    expected_top = int(events["event_type"].eq("TOP").sum())
    expected_bottom = int(events["event_type"].eq("BOTTOM").sum())
    if svg.count('class="pivot top"') != expected_top:
        raise AssertionError("SVG TOP marker count differs from event table")
    if svg.count('class="pivot bottom"') != expected_bottom:
        raise AssertionError("SVG BOTTOM marker count differs from event table")
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8", newline="\n")
    return {
        "width": width,
        "height": height,
        "top_markers": top_markers,
        "bottom_markers": bottom_markers,
        "confirmation_markers": confirmation_markers,
        "execution_markers": execution_markers,
    }
