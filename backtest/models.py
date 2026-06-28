"""Backtest data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class TradeLog:
    """Single backtest trade / signal outcome."""

    signal_date: date
    symbol: str
    strategy_id: int
    strategy_name: str
    entry_price: float
    exit_price: float | None
    exit_date: date | None
    hold_days: int
    return_pct: float | None
    alpha_pct: float | None
    score: float
    exit_reason: str = "forward_hold"
    cost_bps: float = 0.0
    net_return_pct: float | None = None
    simulated: bool = False
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.return_pct is not None and self.net_return_pct is None:
            self.net_return_pct = self.return_pct - self.cost_bps / 100.0


@dataclass
class StrategySummary:
    """Aggregated backtest stats for one strategy."""

    strategy_id: int
    strategy_name: str
    signal_count: int
    win_rate: float
    avg_return: float
    median_return: float
    avg_alpha: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    sharpe: float


@dataclass
class BacktestResult:
    """Full backtest output."""

    segment: str
    start_date: date
    end_date: date
    hold_days: int
    trades: list[TradeLog]
    summaries: list[StrategySummary]
    cost_bps: float = 0.0
    mode: str = "forward_return"
