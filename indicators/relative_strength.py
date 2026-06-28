"""Relative strength calculations vs benchmark."""

from __future__ import annotations

import pandas as pd

from utils.helpers import safe_pct


def period_return(df: pd.DataFrame, idx: int, days: int) -> float:
    """Return percentage over `days` completed candles ending at idx."""
    if idx < days:
        return 0.0
    start_close = df.iloc[idx - days]["close"]
    end_close = df.iloc[idx]["close"]
    return safe_pct(end_close, start_close)


def _aligned_benchmark_idx(stock_df: pd.DataFrame, benchmark_df: pd.DataFrame, idx: int) -> int | None:
    """Map stock evaluation index to benchmark index by trade_date."""
    if idx < 0 or idx >= len(stock_df) or benchmark_df.empty:
        return None

    trade_date = stock_df.iloc[idx]["trade_date"]
    matches = benchmark_df.index[benchmark_df["trade_date"] == trade_date].tolist()
    if matches:
        return int(matches[0])

    prior = benchmark_df[benchmark_df["trade_date"] <= trade_date]
    if prior.empty:
        return None
    return int(prior.index[-1])


def relative_strength_vs_benchmark(
    stock_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    idx: int,
    days: int,
) -> float:
    """Stock return minus benchmark return over period (percentage points)."""
    bench_idx = _aligned_benchmark_idx(stock_df, benchmark_df, idx)
    if bench_idx is None:
        return float("nan")
    stock_ret = period_return(stock_df, idx, days)
    bench_ret = period_return(benchmark_df, bench_idx, days)
    return stock_ret - bench_ret


def compute_cross_sectional_returns(
    candles_map: dict[str, pd.DataFrame],
    idx_map: dict[str, int],
    days: int,
) -> pd.Series:
    """Compute return for all symbols at their evaluation index."""
    returns: dict[str, float] = {}
    for symbol, df in candles_map.items():
        idx = idx_map.get(symbol, len(df) - 1)
        if idx >= days:
            returns[symbol] = period_return(df, idx, days)
    return pd.Series(returns)


def percentile_rank(series: pd.Series, value: float) -> float:
    """Return percentile rank (0-100) of value within series."""
    if series.empty:
        return 0.0
    return (series < value).sum() / len(series) * 100.0
