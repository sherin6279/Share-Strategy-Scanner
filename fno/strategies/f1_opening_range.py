"""F&O Strategy 101 – Opening Range Breakout on index/stock futures."""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategies.base import StrategySignal

STRATEGY_ID = 101
STRATEGY_NAME = "F&O Opening Range Breakout"

MIN_BARS = 4
VOLUME_RATIO_MIN = 1.2


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

    orb_high = context.get("orb_high")
    orb_low = context.get("orb_low")
    avg_vol = context.get("avg_vol_20")
    if any(pd.isna(v) or v is None for v in [orb_high, orb_low, avg_vol]):
        return None

    row = candles.iloc[evaluation_index]
    if row["close"] <= orb_high:
        return None

    vol_ratio = row["volume"] / avg_vol if avg_vol > 0 else 0
    if vol_ratio < VOLUME_RATIO_MIN:
        return None

    breakout_pct = (row["close"] / orb_high - 1.0) * 100.0
    score = min(10.0, breakout_pct * 2.0 + vol_ratio * 2.0)

    return StrategySignal(
        strategy_id=STRATEGY_ID,
        strategy_name=STRATEGY_NAME,
        symbol=symbol,
        signal_date=row["trade_datetime"],
        score=round(score, 2),
        trigger_price=float(row["close"]),
        metrics={
            "orb_high": round(float(orb_high), 2),
            "volume_ratio": round(float(vol_ratio), 2),
            "breakout_pct": round(float(breakout_pct), 2),
        },
    )
