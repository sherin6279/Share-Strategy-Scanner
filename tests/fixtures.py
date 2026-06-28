"""Shared test fixtures."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

from indicators import enrich_candles


def make_daily_candles(
    symbol: str,
    start: date,
    days: int,
    start_price: float = 100.0,
    daily_return: float = 0.001,
) -> pd.DataFrame:
    rows = []
    price = start_price
    for i in range(days):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        open_p = price
        close_p = price * (1 + daily_return)
        high_p = max(open_p, close_p) * 1.005
        low_p = min(open_p, close_p) * 0.995
        rows.append(
            {
                "symbol": symbol,
                "trade_date": d,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": 1_000_000 + i * 1000,
            }
        )
        price = close_p
    return pd.DataFrame(rows)


def make_intraday_day(
    symbol: str,
    trade_date: date,
    bars: int = 30,
    base_price: float = 100.0,
) -> pd.DataFrame:
    """Generate one session of 5-minute bars."""
    rows = []
    start_dt = datetime.combine(trade_date, datetime.strptime("09:15", "%H:%M").time())
    price = base_price
    for i in range(bars):
        dt = start_dt + timedelta(minutes=5 * i)
        drift = 0.002 if i > 3 else 0.0
        open_p = price
        close_p = price * (1 + drift)
        high_p = max(open_p, close_p) * 1.001
        low_p = min(open_p, close_p) * 0.999
        rows.append(
            {
                "symbol": symbol,
                "interval": "5minute",
                "trade_datetime": dt,
                "open": open_p,
                "high": high_p,
                "low": low_p,
                "close": close_p,
                "volume": 50_000 + i * 500,
            }
        )
        price = close_p
    return pd.DataFrame(rows)
