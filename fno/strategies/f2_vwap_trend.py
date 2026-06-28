"""F&O Strategy 102 – VWAP trend with volume confirmation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategies.base import StrategySignal

STRATEGY_ID = 102
STRATEGY_NAME = "F&O VWAP Trend"

MIN_BARS = 21
VOLUME_RATIO_MIN = 1.1


def evaluate(
    symbol: str,
    candles: pd.DataFrame,
    evaluation_index: int,
    context: dict[str, Any] | None = None,
) -> StrategySignal | None:
    context = context or {}
    if evaluation_index < MIN_BARS or evaluation_index >= len(candles):
        return None

    if context.get("is_eod_bar"):
        return None

    vwap = context.get("vwap")
    ema9 = context.get("ema9")
    ema21 = context.get("ema21")
    avg_vol = context.get("avg_vol_20")

    if any(pd.isna(v) or v is None for v in [vwap, ema9, ema21, avg_vol]):
        return None

    row = candles.iloc[evaluation_index]
    if not (row["close"] > vwap and ema9 > ema21):
        return None

    vol_ratio = row["volume"] / avg_vol if avg_vol > 0 else 0
    if vol_ratio < VOLUME_RATIO_MIN:
        return None

    score = min(10.0, (row["close"] / vwap - 1.0) * 200 + vol_ratio)

    return StrategySignal(
        strategy_id=STRATEGY_ID,
        strategy_name=STRATEGY_NAME,
        symbol=symbol,
        signal_date=row["trade_datetime"],
        score=round(score, 2),
        trigger_price=float(row["close"]),
        metrics={
            "vwap": round(float(vwap), 2),
            "volume_ratio": round(float(vol_ratio), 2),
        },
    )
