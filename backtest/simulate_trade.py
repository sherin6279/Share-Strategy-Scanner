"""Equity trade simulation with stop, target, and calendar-based exits."""

from __future__ import annotations

from datetime import date

import pandas as pd

from backtest.date_index import DateIndex


def simulate_equity_trade(
    candles: pd.DataFrame,
    date_index: DateIndex,
    symbol: str,
    signal_date: date,
    entry_price: float,
    stop_pct: float = 5.0,
    target_pct: float = 10.0,
    max_hold_days: int = 5,
) -> tuple[float | None, date | None, str, float | None]:
    """
    Walk forward day-by-day after signal until stop, target, or max hold.

    Returns (return_pct, exit_date, exit_reason, exit_price).
    """
    stop_price = entry_price * (1.0 - stop_pct / 100.0)
    target_price = entry_price * (1.0 + target_pct / 100.0)

    for day_offset in range(1, max_hold_days + 1):
        exit_date = date_index.forward_date(signal_date, day_offset)
        if exit_date is None:
            break

        day_idx = date_index.symbol_date_idx.get(symbol, {}).get(exit_date)
        if day_idx is None:
            continue

        row = candles.iloc[day_idx]
        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])

        if low <= stop_price:
            ret = (stop_price / entry_price - 1.0) * 100.0
            return ret, exit_date, "stop_loss", stop_price

        if high >= target_price:
            ret = (target_price / entry_price - 1.0) * 100.0
            return ret, exit_date, "target", target_price

        if day_offset == max_hold_days:
            ret = (close / entry_price - 1.0) * 100.0
            return ret, exit_date, "max_hold", close

    return None, None, "no_data", None
