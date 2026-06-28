"""Backtest performance metrics."""

from __future__ import annotations

import math

import numpy as np

from backtest.models import StrategySummary, TradeLog
from strategies.strategy_engine import STRATEGIES

_STRATEGY_NAMES = {sid: name for sid, name, _ in STRATEGIES}


def summarize_trades(
    trades: list[TradeLog],
    strategy_id: int | None = None,
) -> StrategySummary:
    """Compute aggregate metrics for a strategy (or all if strategy_id is None)."""
    subset = trades if strategy_id is None else [t for t in trades if t.strategy_id == strategy_id]
    sid = strategy_id or (subset[0].strategy_id if subset else 0)
    name = _STRATEGY_NAMES.get(sid, subset[0].strategy_name if subset else "All")

    valid = [t for t in subset if t.net_return_pct is not None]
    if not valid:
        return StrategySummary(
            strategy_id=sid,
            strategy_name=name,
            signal_count=0,
            win_rate=0.0,
            avg_return=0.0,
            median_return=0.0,
            avg_alpha=0.0,
            expectancy=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            sharpe=0.0,
        )

    returns = np.array([t.net_return_pct for t in valid], dtype=float)
    alphas = np.array([t.alpha_pct or 0.0 for t in valid], dtype=float)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    win_rate = float((returns > 0).mean() * 100.0)
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) else 0.0
    loss_rate = 1.0 - win_rate / 100.0
    expectancy = (win_rate / 100.0) * avg_win - loss_rate * avg_loss

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    equity = np.cumsum(returns)
    peak = np.maximum.accumulate(equity)
    drawdowns = peak - equity
    max_drawdown = float(drawdowns.max()) if len(drawdowns) else 0.0

    std = float(returns.std())
    sharpe = float(returns.mean() / std * math.sqrt(252 / 5)) if std > 0 else 0.0

    return StrategySummary(
        strategy_id=sid,
        strategy_name=name,
        signal_count=len(valid),
        win_rate=round(win_rate, 2),
        avg_return=round(float(returns.mean()), 2),
        median_return=round(float(np.median(returns)), 2),
        avg_alpha=round(float(alphas.mean()), 2),
        expectancy=round(expectancy, 2),
        profit_factor=round(profit_factor, 2),
        max_drawdown=round(max_drawdown, 2),
        sharpe=round(sharpe, 2),
    )


def summarize_all_strategies(trades: list[TradeLog]) -> list[StrategySummary]:
    ids = sorted({t.strategy_id for t in trades})
    return [summarize_trades(trades, sid) for sid in ids]


def summaries_to_dataframe(summaries: list[StrategySummary]):
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "Strategy": f"S{s.strategy_id} {s.strategy_name}",
                "Signals": s.signal_count,
                "Win %": s.win_rate,
                "Avg Return %": s.avg_return,
                "Median %": s.median_return,
                "Alpha %": s.avg_alpha,
                "Expectancy": s.expectancy,
                "Profit Factor": s.profit_factor,
                "Max DD %": s.max_drawdown,
                "Sharpe": s.sharpe,
            }
            for s in summaries
        ]
    )
