from __future__ import annotations

import math

import numpy as np
import pandas as pd


def maximum_drawdown(nav: pd.Series) -> float:
    values = nav.astype(float)
    if values.empty or (values <= 0).any():
        raise ValueError("NAV must be positive and non-empty")
    return float((values / values.cummax() - 1.0).min())


def annualized_return(nav: pd.Series, sessions_per_year: int = 252) -> float:
    if len(nav) < 2 or nav.iloc[0] <= 0 or nav.iloc[-1] <= 0:
        return math.nan
    years = (len(nav) - 1) / sessions_per_year
    return float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else math.nan


def sortino_ratio(returns: pd.Series, sessions_per_year: int = 252) -> float:
    values = returns.dropna().astype(float)
    downside = values[values < 0]
    if values.empty or downside.empty:
        return math.nan
    downside_deviation = float(np.sqrt(np.mean(downside.to_numpy() ** 2)))
    if downside_deviation == 0:
        return math.nan
    return float(values.mean() / downside_deviation * np.sqrt(sessions_per_year))


def calmar_ratio(nav: pd.Series) -> float:
    drawdown = maximum_drawdown(nav)
    cagr = annualized_return(nav)
    if drawdown == 0 or math.isnan(cagr):
        return math.nan
    return float(cagr / abs(drawdown))


def paired_block_bootstrap_uplift_ci(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    block_size: int = 10,
    samples: int = 2000,
    random_seed: int = 20260722,
) -> tuple[float, float]:
    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < block_size * 2:
        return math.nan, math.nan
    log_excess = np.log1p(aligned.iloc[:, 0].to_numpy()) - np.log1p(aligned.iloc[:, 1].to_numpy())
    length = len(log_excess)
    starts = np.arange(0, length - block_size + 1)
    blocks_needed = math.ceil(length / block_size)
    rng = np.random.default_rng(random_seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([log_excess[start : start + block_size] for start in selected])[
            :length
        ]
        estimates[index] = np.exp(sample.sum()) - 1.0
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def portfolio_metrics(
    daily: pd.DataFrame, cycles: pd.DataFrame, fills: pd.DataFrame
) -> dict[str, float]:
    if daily.empty:
        raise ValueError("Daily portfolio is empty")
    strategy_nav = daily["nav"].astype(float)
    benchmark_nav = daily["benchmark_nav"].astype(float)
    strategy_returns = strategy_nav.pct_change()
    benchmark_returns = benchmark_nav.pct_change()
    strategy_cagr = annualized_return(strategy_nav)
    benchmark_cagr = annualized_return(benchmark_nav)
    strategy_drawdown = maximum_drawdown(strategy_nav)
    benchmark_drawdown = maximum_drawdown(benchmark_nav)
    strategy_calmar = calmar_ratio(strategy_nav)
    benchmark_calmar = calmar_ratio(benchmark_nav)
    ci_lower, ci_upper = paired_block_bootstrap_uplift_ci(strategy_returns, benchmark_returns)
    negative_benchmark = benchmark_returns < 0
    downside_capture = (
        float(
            strategy_returns.loc[negative_benchmark].sum()
            / benchmark_returns.loc[negative_benchmark].sum()
        )
        if negative_benchmark.any() and benchmark_returns.loc[negative_benchmark].sum() != 0
        else math.nan
    )
    completed = cycles.loc[cycles["completed"]] if not cycles.empty else cycles
    positive_cycle_rate = (
        float(completed["net_saving"].gt(0).mean()) if not completed.empty else math.nan
    )
    initial_nav = float(strategy_nav.iloc[0])
    regret = (strategy_nav - benchmark_nav) / initial_nav
    fees = float(fills["fee"].sum()) if not fills.empty else 0.0
    net_saving = float(completed["net_saving"].sum()) if not completed.empty else 0.0
    years = max((len(daily) - 1) / 252.0, 1 / 252.0)
    first_row = daily.iloc[0]
    initial_gold_value = float(
        first_row.get(
            "initial_position_value",
            first_row["price"] * first_row["initial_gold_qty"],
        )
    )
    initial_gold_equivalent_oz = float(
        first_row.get("initial_gold_equivalent_oz", first_row["initial_gold_qty"])
    )
    if initial_gold_value <= 0 or initial_gold_equivalent_oz <= 0:
        raise ValueError("Initial position value and gold-equivalent ounces must be positive")
    final_24m_uplift = math.nan
    if len(daily) >= 505:
        strategy_24m = strategy_nav.iloc[-1] / strategy_nav.iloc[-505]
        benchmark_24m = benchmark_nav.iloc[-1] / benchmark_nav.iloc[-505]
        final_24m_uplift = float(strategy_24m / benchmark_24m - 1.0)
    return {
        "strategy_total_return": float(strategy_nav.iloc[-1] / strategy_nav.iloc[0] - 1.0),
        "benchmark_total_return": float(benchmark_nav.iloc[-1] / benchmark_nav.iloc[0] - 1.0),
        "uplift": float(strategy_nav.iloc[-1] / benchmark_nav.iloc[-1] - 1.0),
        "strategy_cagr": strategy_cagr,
        "benchmark_cagr": benchmark_cagr,
        "annualized_excess_cagr": strategy_cagr - benchmark_cagr,
        "strategy_max_drawdown": strategy_drawdown,
        "benchmark_max_drawdown": benchmark_drawdown,
        "max_drawdown_relative_improvement": (
            (abs(benchmark_drawdown) - abs(strategy_drawdown)) / abs(benchmark_drawdown)
            if benchmark_drawdown != 0
            else math.nan
        ),
        "strategy_sortino": sortino_ratio(strategy_returns),
        "benchmark_sortino": sortino_ratio(benchmark_returns),
        "strategy_calmar": strategy_calmar,
        "benchmark_calmar": benchmark_calmar,
        "calmar_relative_improvement": (
            (strategy_calmar - benchmark_calmar) / abs(benchmark_calmar)
            if math.isfinite(strategy_calmar)
            and math.isfinite(benchmark_calmar)
            and benchmark_calmar != 0
            else math.nan
        ),
        "final_24m_uplift": final_24m_uplift,
        "downside_capture": downside_capture,
        "max_relative_regret": float(abs(min(0.0, regret.min()))),
        "completed_cycles": len(completed),
        "positive_cycle_rate": positive_cycle_rate,
        "median_completed_cycle_net_saving": (
            float(completed["net_saving"].median()) if not completed.empty else math.nan
        ),
        "maximum_cycle_sessions": (
            float(completed["holding_gap_sessions"].max())
            if not completed.empty
            else math.nan
        ),
        "forced_rebuy_cycle_rate": (
            float(completed["forced"].mean()) if not completed.empty else math.nan
        ),
        "net_saving_total": net_saving,
        "basis_reduction_total_usd_per_oz": net_saving / initial_gold_equivalent_oz,
        "annualized_basis_reduction_fraction": net_saving / initial_gold_value / years,
        "fees_paid": fees,
        "paired_uplift_ci_95_lower": ci_lower,
        "paired_uplift_ci_95_upper": ci_upper,
    }
