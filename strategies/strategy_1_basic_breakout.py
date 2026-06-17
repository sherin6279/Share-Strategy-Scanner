"""Strategy 1 – Conservative Breakout."""

from __future__ import annotations

from typing import Any

import pandas as pd

from indicators import previous_high
from strategies.base import StrategySignal
from utils.helpers import metrics_to_dict


STRATEGY_ID = 1
STRATEGY_NAME = "Conservative Breakout"


def evaluate(
    symbol: str,
    candles: pd.DataFrame,
    evaluation_index: int,
    context: dict[str, Any] | None = None,
) -> StrategySignal | None:
    """
    Conservative breakout strategy.

    Conditions:
    - Close > previous 125-day high
    - Volume > 2x 125-day average volume
    - RSI between 55 and 75
    - Close > EMA20
    - EMA20 > SMA50
    - SMA50 > SMA200
    - Close > SMA200

    Score:
    Combines breakout strength, volume surge, and RSI quality.
    """

    idx = evaluation_index

    # Need enough history for SMA200
    if idx < 200 or len(candles) <= idx:
        return None

    row = candles.iloc[idx]

    prev_high_125 = previous_high(candles, idx, 125)

    avg_vol_125 = row.get("avg_vol_125")
    rsi_val = row.get("rsi14")

    ema20 = row.get("ema20")
    sma50 = row.get("sma50")
    sma200 = row.get("sma200")

    required_values = [
        prev_high_125,
        avg_vol_125,
        rsi_val,
        ema20,
        sma50,
        sma200,
    ]

    if any(pd.isna(v) for v in required_values):
        return None

    breakout_ok = row["close"] > prev_high_125

    volume_ok = row["volume"] > (2 * avg_vol_125)

    rsi_ok = 55 < rsi_val < 75

    trend_ok = (
        row["close"] > ema20
        and ema20 > sma50
        and sma50 > sma200
        and row["close"] > sma200
    )

    if not (
        breakout_ok
        and volume_ok
        and rsi_ok
        and trend_ok
    ):
        return None

    breakout_pct = (
        (row["close"] / prev_high_125) - 1.0
    ) * 100.0

    volume_ratio = row["volume"] / avg_vol_125

    score = (
        breakout_pct * 0.50
        + volume_ratio * 10.0
        + (rsi_val - 55.0) * 0.50
    )

    return StrategySignal(
        strategy_id=STRATEGY_ID,
        strategy_name=STRATEGY_NAME,
        symbol=symbol,
        signal_date=row["trade_date"],
        score=float(score),
        trigger_price=float(row["close"]),
        metrics=metrics_to_dict(
            {
                "breakout_pct": breakout_pct,
                "volume_ratio": volume_ratio,
                "rsi14": rsi_val,
                "ema20": ema20,
                "sma50": sma50,
                "sma200": sma200,
                "prev_high_125": prev_high_125,
            }
        ),
    )