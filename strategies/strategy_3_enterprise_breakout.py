"""Strategy 3 – Enterprise Breakout."""

from __future__ import annotations

from typing import Any

import pandas as pd

from config.settings import MIN_TRADED_VALUE_CR_S1, NIFTY50_SYMBOL
from indicators import previous_high
from indicators.relative_strength import relative_strength_vs_benchmark
from strategies.base import StrategySignal
from utils.helpers import avg_traded_value_cr, candle_position_score, clamp, metrics_to_dict


STRATEGY_ID = 3
STRATEGY_NAME = "Enterprise Breakout"


def _score_component(value: float, low: float, high: float) -> float:
    """Normalize a metric to 0-10 scale."""
    if high <= low:
        return 0.0
    return clamp((value - low) / (high - low) * 10.0, 0.0, 10.0)


def evaluate(
    symbol: str,
    candles: pd.DataFrame,
    evaluation_index: int,
    context: dict[str, Any] | None = None,
) -> StrategySignal | None:
    """Enterprise-grade breakout with composite scoring."""
    context = context or {}
    idx = evaluation_index

    if symbol == NIFTY50_SYMBOL or idx < 250:
        return None

    # Market regime check handled at engine level; skip if flagged
    if not context.get("market_uptrend", True):
        return None

    nifty50 = context.get("nifty50_candles")
    if nifty50 is None or len(nifty50) <= idx:
        return None

    row = candles.iloc[idx]
    prev_high_250 = previous_high(candles, idx, 250)
    avg_vol_125 = row.get("avg_vol_125")

    sma200 = row.get("sma200")
    sma50 = row.get("sma50")
    ema20 = row.get("ema20")
    rsi_val = row.get("rsi14")
    atr14 = row.get("atr14")

    if any(pd.isna(v) for v in [prev_high_250, avg_vol_125, sma200, sma50, ema20, rsi_val, atr14]):
        return None

    rs_60 = relative_strength_vs_benchmark(candles, nifty50, idx, 60)
    atr_pct = atr14 / row["close"] * 100.0
    candle_pos = candle_position_score(row)
    traded_value_cr = avg_traded_value_cr(candles, 20, idx)

    breakout_threshold = prev_high_250 * 1.01

    if not (
        row["close"] > breakout_threshold
        and row["volume"] > 2 * avg_vol_125
        and 55 <= rsi_val <= 70
        and row["close"] > sma200
        and sma50 > sma200
        and row["close"] > ema20
        and ema20 > sma50 > sma200
        and rs_60 >= 10.0
        and 1.5 <= atr_pct <= 5.0
        and candle_pos >= 0.75
        and traded_value_cr > MIN_TRADED_VALUE_CR_S1
    ):
        return None

    breakout_pct = (row["close"] / prev_high_250 - 1.0) * 100.0
    volume_ratio = row["volume"] / avg_vol_125

    # Weighted score 0-10
    breakout_score = _score_component(breakout_pct, 1.0, 15.0)
    volume_score = _score_component(volume_ratio, 2.0, 5.0)
    rs_score = _score_component(rs_60, 10.0, 40.0)
    trend_score = _score_component(
        (row["close"] / sma200 - 1.0) * 100.0, 0.0, 30.0
    )
    rsi_score = _score_component(rsi_val, 55.0, 70.0)

    composite = (
        breakout_score * 0.30
        + volume_score * 0.25
        + rs_score * 0.20
        + trend_score * 0.15
        + rsi_score * 0.10
    )

    if composite < 6.0:
        return None

    action = "Strong Buy" if composite >= 8.0 else "Buy"

    return StrategySignal(
        strategy_id=STRATEGY_ID,
        strategy_name=STRATEGY_NAME,
        symbol=symbol,
        signal_date=row["trade_date"],
        score=round(composite, 2),
        trigger_price=float(row["close"]),
        suggested_action=action,
        metrics=metrics_to_dict(
            {
                "breakout_pct": breakout_pct,
                "volume_ratio": volume_ratio,
                "rs_60": rs_60,
                "atr_pct": atr_pct,
                "candle_position": candle_pos,
                "traded_value_cr": traded_value_cr,
                "rsi14": rsi_val,
                "suggested_action": action,
            }
        ),
    )
