"""Strategy 2 – Refined Breakout."""

from __future__ import annotations

from typing import Any

import pandas as pd

from indicators import previous_high
from strategies.base import StrategySignal
from utils.helpers import metrics_to_dict


STRATEGY_ID = 2
STRATEGY_NAME = "Refined Breakout"


def evaluate(
    symbol: str,
    candles: pd.DataFrame,
    evaluation_index: int,
    context: dict[str, Any] | None = None,
) -> StrategySignal | None:
    """
    Refined breakout strategy.

    Conditions:
    - Close > previous 250-day high
    - Volume > 1.5x 125-day average volume
    - RSI between 55 and 80
    - Close > EMA20
    - EMA20 > SMA50
    - SMA50 > SMA200
    - Close > SMA200

    Score:
    Combines breakout strength, volume expansion, and trend quality.
    """

    idx = evaluation_index

    # Need enough history for SMA200 and 250-day breakout
    if idx < 250 or len(candles) <= idx:
        return None

    row = candles.iloc[idx]

    prev_high_250 = previous_high(candles, idx, 250)

    avg_vol_125 = row.get("avg_vol_125")
    rsi_val = row.get("rsi14")

    ema20 = row.get("ema20")
    sma50 = row.get("sma50")
    sma200 = row.get("sma200")

    required_values = [
        prev_high_250,
        avg_vol_125,
        rsi_val,
        ema20,
        sma50,
        sma200,
    ]

    if any(pd.isna(v) for v in required_values):
        return None

    breakout_ok = row["close"] > prev_high_250

    volume_ok = row["volume"] > (1.5 * avg_vol_125)

    rsi_ok = 55 < rsi_val < 80

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
        (row["close"] / prev_high_250) - 1.0
    ) * 100.0

    volume_ratio = row["volume"] / avg_vol_125

    score = (
        breakout_pct * 0.60
        + volume_ratio * 8.0
        + (rsi_val - 55.0) * 0.30
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
                "prev_high_250": prev_high_250,
            }
        ),
    )