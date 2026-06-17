"""Strategy result types and base protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd


@dataclass
class StrategySignal:
    """Output from a strategy evaluation."""

    strategy_id: int
    strategy_name: str
    symbol: str
    signal_date: Any
    score: float
    trigger_price: float
    suggested_action: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


class Strategy(Protocol):
    """Protocol for backtest-compatible strategies."""

    strategy_id: int
    name: str

    def evaluate(
        self,
        symbol: str,
        candles: pd.DataFrame,
        evaluation_index: int,
        context: dict[str, Any],
    ) -> StrategySignal | None:
        """Evaluate strategy at a specific candle index."""
        ...
