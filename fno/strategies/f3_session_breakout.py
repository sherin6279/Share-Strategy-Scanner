"""F&O Strategy 103 – Intraday breakout above prior session high."""

from __future__ import annotations

from typing import Any

import pandas as pd

from strategies.base import StrategySignal

STRATEGY_ID = 103
STRATEGY_NAME = "F&O Session High Breakout"

MIN_BARS = 5
VOLUME_RATIO_MIN = 1.3


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

    prior_high = context.get("prior_session_high")
    avg_vol = context.get("avg_vol_20")
    if prior_high is None or pd.isna(prior_high) or pd.isna(avg_vol):
        return None

    row = candles.iloc[evaluation_index]
    if row["close"] <= prior_high:
        return None

    vol_ratio = row["volume"] / avg_vol if avg_vol > 0 else 0
    if vol_ratio < VOLUME_RATIO_MIN:
        return None

    breakout_pct = (row["close"] / prior_high - 1.0) * 100.0
    score = min(10.0, breakout_pct * 3.0 + vol_ratio * 2.0)

    return StrategySignal(
        strategy_id=STRATEGY_ID,
        strategy_name=STRATEGY_NAME,
        symbol=symbol,
        signal_date=row["trade_datetime"],
        score=round(score, 2),
        trigger_price=float(row["close"]),
        metrics={
            "prior_session_high": round(float(prior_high), 2),
            "volume_ratio": round(float(vol_ratio), 2),
        },
    )
