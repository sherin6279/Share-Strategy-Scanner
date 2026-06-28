"""F&O intraday strategy engine."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable

import pandas as pd

from fno.strategies import f1_opening_range as f1
from fno.strategies import f2_vwap_trend as f2
from fno.strategies import f3_session_breakout as f3
from fno.intraday_context import IntradayDayContext
from strategies.base import StrategySignal
from utils.logger import get_logger

logger = get_logger(__name__)

FNO_STRATEGIES: list[tuple[int, str, Callable]] = [
    (f1.STRATEGY_ID, f1.STRATEGY_NAME, f1.evaluate),
    (f2.STRATEGY_ID, f2.STRATEGY_NAME, f2.evaluate),
    (f3.STRATEGY_ID, f3.STRATEGY_NAME, f3.evaluate),
]


class FnoStrategyEngine:
    """Evaluate F&O intraday strategies on one trading day."""

    def evaluate_day(
        self,
        symbol: str,
        day_df: pd.DataFrame,
        prior_session_high: float | None = None,
        strategy_ids: list[int] | None = None,
    ) -> list[tuple[int, int, StrategySignal]]:
        """
        Returns list of (bar_idx, strategy_id, signal) for all signals in the day.
        """
        allowed = set(strategy_ids) if strategy_ids else None
        day_ctx = IntradayDayContext(day_df)
        results: list[tuple[int, int, StrategySignal]] = []

        for bar_idx in range(len(day_df)):
            bar_context = day_ctx.build_bar_context(bar_idx)
            bar_context["prior_session_high"] = prior_session_high

            for strategy_id, _name, evaluate_fn in FNO_STRATEGIES:
                if allowed and strategy_id not in allowed:
                    continue
                try:
                    sig = evaluate_fn(symbol, day_df, bar_idx, bar_context)
                    if sig is not None:
                        results.append((bar_idx, strategy_id, sig))
                except Exception as exc:
                    logger.warning("F&O strategy %d failed: %s", strategy_id, exc)

        return results

    @staticmethod
    def simulate_intraday_trade(
        day_df: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        stop_pct: float = 0.5,
        target_pct: float = 1.0,
    ) -> tuple[float, str, float]:
        """Bar-forward simulation until stop, target, or EOD."""
        stop_price = entry_price * (1.0 - stop_pct / 100.0)
        target_price = entry_price * (1.0 + target_pct / 100.0)

        for i in range(entry_idx + 1, len(day_df)):
            row = day_df.iloc[i]
            low = float(row["low"])
            high = float(row["high"])
            close = float(row["close"])

            if low <= stop_price:
                return (stop_price / entry_price - 1.0) * 100.0, "stop_loss", stop_price
            if high >= target_price:
                return (target_price / entry_price - 1.0) * 100.0, "target", target_price
            if i == len(day_df) - 1:
                return (close / entry_price - 1.0) * 100.0, "eod", close

        return 0.0, "no_data", entry_price
