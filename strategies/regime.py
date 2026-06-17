"""Market regime helpers shared by strategies 4–7."""

from __future__ import annotations

import pandas as pd

from indicators.relative_strength import _aligned_benchmark_idx


def get_market_regime(
    stock_candles: pd.DataFrame,
    nifty_candles: pd.DataFrame | None,
    evaluation_index: int,
) -> tuple[bool, float | None]:
    """
    Return (market_uptrend, nifty_sma200) for the stock's evaluation date.

    Aligns NIFTY 50 to the stock candle by trade_date so stocks with shorter
    history are not checked against the wrong market day.
    """
    if nifty_candles is None or nifty_candles.empty:
        return False, None

    bench_idx = _aligned_benchmark_idx(stock_candles, nifty_candles, evaluation_index)
    row = nifty_candles.iloc[bench_idx]
    nifty_sma200 = row.get("sma200")
    if pd.isna(nifty_sma200):
        return False, None

    market_uptrend = bool(row["close"] > nifty_sma200)
    return market_uptrend, float(nifty_sma200)
