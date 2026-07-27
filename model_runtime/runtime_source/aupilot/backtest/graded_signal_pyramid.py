from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from aupilot.backtest.metrics import maximum_drawdown
from aupilot.backtest.pivot_baseline import (
    _target_trade_notional,
    validate_daily_ohlc,
)

GRADED_SIGNAL_LABELS = (
    "TOP_L1",
    "TOP_L2",
    "TOP_L3",
    "BOTTOM_L1",
    "BOTTOM_L2",
    "BOTTOM_L3",
)


@dataclass(frozen=True)
class GradedPyramidPolicy:
    """Deterministic long-only position sizing for graded daily signals."""

    initial_nav_usd: float = 1_000_000.0
    initial_gold_weight: float = 1.0
    minimum_gold_weight: float = 0.5
    maximum_gold_weight: float = 1.0
    level_1_delta: float = 0.1
    level_2_delta: float = 0.2
    level_3_delta: float = 0.4
    cost_bps_per_side: float = 2.0

    def __post_init__(self) -> None:
        if self.initial_nav_usd <= 0.0:
            raise ValueError("initial_nav_usd must be positive")
        weights = (
            self.minimum_gold_weight,
            self.initial_gold_weight,
            self.maximum_gold_weight,
        )
        if any(not 0.0 <= value <= 1.0 for value in weights):
            raise ValueError("gold weights must be in [0, 1]")
        if not (
            self.minimum_gold_weight
            <= self.initial_gold_weight
            <= self.maximum_gold_weight
        ):
            raise ValueError("initial weight must be inside the policy bounds")
        deltas = (
            self.level_1_delta,
            self.level_2_delta,
            self.level_3_delta,
        )
        if not 0.0 < deltas[0] < deltas[1] < deltas[2] <= 1.0:
            raise ValueError("graded deltas must be strictly increasing in (0, 1]")
        if self.cost_bps_per_side < 0.0:
            raise ValueError("cost_bps_per_side must be non-negative")

    def delta_for(self, label: str) -> float:
        level = int(label[-1])
        return {
            1: self.level_1_delta,
            2: self.level_2_delta,
            3: self.level_3_delta,
        }[level]

    def next_weight(self, current_weight: float, label: str) -> float:
        if label not in GRADED_SIGNAL_LABELS:
            raise ValueError(f"unsupported graded signal: {label}")
        delta = self.delta_for(label)
        if label.startswith("TOP_"):
            return max(self.minimum_gold_weight, current_weight - delta)
        return min(self.maximum_gold_weight, current_weight + delta)


@dataclass(frozen=True)
class GradedPyramidReplayResult:
    orders: pd.DataFrame
    fills: pd.DataFrame
    daily_portfolio: pd.DataFrame
    metrics: dict[str, Any]


def _calendar_cagr(
    start_value: float,
    final_value: float,
    start_date: object,
    end_date: object,
) -> float:
    elapsed_days = (
        pd.Timestamp(end_date) - pd.Timestamp(start_date)
    ).days
    if start_value <= 0.0 or final_value <= 0.0 or elapsed_days <= 0:
        return float("nan")
    years = elapsed_days / 365.2425
    return float((final_value / start_value) ** (1.0 / years) - 1.0)


def _validate_signals(signals: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    required = {"signal_id", "signal_date", "signal_label", "event_group_id"}
    missing = required - set(signals.columns)
    if missing:
        raise ValueError(f"missing graded signal columns: {sorted(missing)}")
    frame = signals.loc[:, list(required)].copy()
    frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.date
    frame["signal_label"] = frame["signal_label"].astype(str)
    frame["signal_id"] = frame["signal_id"].astype(str)
    frame["event_group_id"] = frame["event_group_id"].astype(str)
    frame = frame.sort_values(
        ["signal_date", "signal_id"], kind="stable"
    ).reset_index(drop=True)
    if frame.empty:
        raise ValueError("graded signal table is empty")
    if frame["signal_id"].duplicated().any():
        raise ValueError("duplicate signal_id")
    if frame["signal_date"].duplicated().any():
        raise ValueError("more than one graded signal shares a daily Open")
    invalid = sorted(set(frame["signal_label"]) - set(GRADED_SIGNAL_LABELS))
    if invalid:
        raise ValueError(f"unsupported graded signals: {invalid}")
    available_dates = set(daily["trade_date"])
    outside = sorted(set(frame["signal_date"]) - available_dates)
    if outside:
        raise ValueError(f"signal dates outside daily OHLC: {outside[:3]}")
    return frame


def run_graded_pyramid_replay(
    daily: pd.DataFrame,
    signals: pd.DataFrame,
    policy: GradedPyramidPolicy | None = None,
) -> GradedPyramidReplayResult:
    """Replay each daily graded signal, including repeated same-side signals.

    Event groups are retained for statistical identity only. They never block
    a daily action. Every signal changes the current target by 10/20/40
    percentage points according to its own label, subject only to the fixed
    50%-100% long-only bounds.
    """

    policy = GradedPyramidPolicy() if policy is None else policy
    prices = validate_daily_ohlc(daily)
    events = _validate_signals(signals, prices)
    schedule = events.set_index("signal_date").to_dict(orient="index")

    first_open = float(prices.iloc[0]["open"])
    gold_quantity = (
        policy.initial_nav_usd * policy.initial_gold_weight / first_open
    )
    cash = policy.initial_nav_usd - gold_quantity * first_open
    benchmark_quantity = policy.initial_nav_usd / first_open
    current_target = policy.initial_gold_weight
    cost_rate = policy.cost_bps_per_side / 10_000.0
    order_records: list[dict[str, object]] = []
    fill_records: list[dict[str, object]] = []
    daily_records: list[dict[str, object]] = []
    previous_signal_side: str | None = None
    repeated_same_side = 0

    for row in prices.itertuples(index=False):
        trade_date = row.trade_date
        open_price = float(row.open)
        nav_before = cash + gold_quantity * open_price
        executed_signal_id: str | None = None
        signal = schedule.get(trade_date)
        if signal is not None:
            label = str(signal["signal_label"])
            side = label.split("_", maxsplit=1)[0]
            if previous_signal_side == side:
                repeated_same_side += 1
            previous_signal_side = side
            target_weight = policy.next_weight(current_target, label)
            gold_value_before = gold_quantity * open_price
            if target_weight == current_target:
                trade_notional = 0.0
            else:
                trade_notional = _target_trade_notional(
                    nav_before=nav_before,
                    gold_value_before=gold_value_before,
                    target_weight=target_weight,
                    cost_rate=cost_rate,
                )
            fee = abs(trade_notional) * cost_rate
            trade_side = "NONE"
            status = "NO_POSITION_CHANGE_AT_BOUND"
            if abs(trade_notional) > 1.0e-8:
                trade_side = "BUY" if trade_notional > 0.0 else "SELL"
                gold_quantity += trade_notional / open_price
                cash -= trade_notional + fee
                status = "FILLED"
                executed_signal_id = str(signal["signal_id"])
                fill_records.append(
                    {
                        "signal_id": signal["signal_id"],
                        "signal_date": trade_date,
                        "signal_label": label,
                        "event_group_id": signal["event_group_id"],
                        "fill_price": open_price,
                        "side": trade_side,
                        "previous_target_weight": current_target,
                        "target_weight": target_weight,
                        "trade_notional_usd": trade_notional,
                        "fee_usd": fee,
                    }
                )
            order_records.append(
                {
                    "signal_id": signal["signal_id"],
                    "signal_date": trade_date,
                    "signal_label": label,
                    "event_group_id": signal["event_group_id"],
                    "previous_target_weight": current_target,
                    "target_weight": target_weight,
                    "side": trade_side,
                    "trade_notional_usd": trade_notional,
                    "fee_usd": fee,
                    "status": status,
                }
            )
            current_target = target_weight

        close_price = float(row.close)
        nav = cash + gold_quantity * close_price
        benchmark_nav = benchmark_quantity * close_price
        if cash < -1.0e-6 or gold_quantity < -1.0e-10 or nav <= 0.0:
            raise AssertionError("graded replay violated long-only portfolio bounds")
        actual_weight = gold_quantity * close_price / nav
        daily_records.append(
            {
                "trade_date": trade_date,
                "open": open_price,
                "close": close_price,
                "cash": cash,
                "gold_quantity": gold_quantity,
                "target_weight": current_target,
                "actual_gold_weight_close": actual_weight,
                "nav": nav,
                "benchmark_nav": benchmark_nav,
                "executed_signal_id": executed_signal_id,
            }
        )

    orders = pd.DataFrame.from_records(order_records)
    fills = pd.DataFrame.from_records(fill_records)
    portfolio = pd.DataFrame.from_records(daily_records)
    nav_with_start = pd.concat(
        [pd.Series([policy.initial_nav_usd]), portfolio["nav"]],
        ignore_index=True,
    )
    benchmark_with_start = pd.concat(
        [pd.Series([policy.initial_nav_usd]), portfolio["benchmark_nav"]],
        ignore_index=True,
    )
    final_nav = float(portfolio.iloc[-1]["nav"])
    final_benchmark = float(portfolio.iloc[-1]["benchmark_nav"])
    strategy_return = final_nav / policy.initial_nav_usd - 1.0
    benchmark_return = final_benchmark / policy.initial_nav_usd - 1.0
    strategy_cagr = _calendar_cagr(
        policy.initial_nav_usd,
        final_nav,
        prices.iloc[0]["trade_date"],
        prices.iloc[-1]["trade_date"],
    )
    benchmark_cagr = _calendar_cagr(
        policy.initial_nav_usd,
        final_benchmark,
        prices.iloc[0]["trade_date"],
        prices.iloc[-1]["trade_date"],
    )
    metrics: dict[str, Any] = {
        "initial_nav_usd": policy.initial_nav_usd,
        "final_nav_usd": final_nav,
        "benchmark_final_nav_usd": final_benchmark,
        "strategy_total_return": strategy_return,
        "benchmark_total_return": benchmark_return,
        "absolute_return_lift_vs_buy_hold": strategy_return - benchmark_return,
        "terminal_nav_ratio_vs_buy_hold": final_nav / final_benchmark - 1.0,
        "strategy_calendar_cagr": strategy_cagr,
        "benchmark_calendar_cagr": benchmark_cagr,
        "calendar_cagr_excess": strategy_cagr - benchmark_cagr,
        "strategy_max_drawdown": maximum_drawdown(nav_with_start),
        "benchmark_max_drawdown": maximum_drawdown(benchmark_with_start),
        "signal_rows": len(events),
        "independent_event_groups": int(events["event_group_id"].nunique()),
        "repeated_same_side_signal_rows": repeated_same_side,
        "filled_trades": len(fills),
        "no_position_change_at_bound": int(
            orders["status"].eq("NO_POSITION_CHANGE_AT_BOUND").sum()
        ),
        "underweight_daily_buckets": int(
            portfolio["target_weight"].lt(policy.maximum_gold_weight).sum()
        ),
        "average_target_gold_weight": float(
            portfolio["target_weight"].mean()
        ),
        "minimum_target_gold_weight_reached": float(
            portfolio["target_weight"].min()
        ),
        "maximum_target_gold_weight_reached": float(
            portfolio["target_weight"].max()
        ),
        "turnover_on_initial_nav": (
            float(fills["trade_notional_usd"].abs().sum())
            / policy.initial_nav_usd
            if not fills.empty
            else 0.0
        ),
        "fees_paid_usd": (
            float(fills["fee_usd"].sum()) if not fills.empty else 0.0
        ),
    }
    return GradedPyramidReplayResult(
        orders=orders,
        fills=fills,
        daily_portfolio=portfolio,
        metrics=metrics,
    )
