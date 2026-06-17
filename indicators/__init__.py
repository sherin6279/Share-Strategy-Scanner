"""Enrich OHLCV data with all technical indicators."""

from __future__ import annotations

import pandas as pd

from indicators.atr import add_atr
from indicators.moving_averages import add_moving_averages
from indicators.rsi import add_rsi
from utils.helpers import ensure_ohlcv


def enrich_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators used by strategies."""
    if df.empty:
        return df
    out = ensure_ohlcv(df)
    out = add_moving_averages(out)
    out = add_rsi(out, 14)
    out = add_atr(out, 14)
    out["avg_vol_125"] = out["volume"].rolling(125, min_periods=125).mean()
    out["avg_vol_20"] = out["volume"].rolling(20, min_periods=20).mean()
    out["avg_vol_30"] = out["volume"].rolling(30, min_periods=30).mean()
    out["avg_vol_10"] = out["volume"].rolling(10, min_periods=10).mean()
    out["traded_value"] = out["close"] * out["volume"]
    out["avg_traded_value_20"] = out["traded_value"].rolling(20, min_periods=20).mean()
    return out


def previous_high(df: pd.DataFrame, idx: int, lookback: int) -> float:
    """Highest high of previous `lookback` candles (excludes current candle at idx)."""
    if idx < lookback:
        return float("nan")
    window = df.iloc[idx - lookback : idx]
    return float(window["high"].max())


def previous_low(df: pd.DataFrame, idx: int, lookback: int) -> float:
    """Lowest low of previous `lookback` candles (excludes current)."""
    if idx < lookback:
        return float("nan")
    window = df.iloc[idx - lookback : idx]
    return float(window["low"].min())


def atr_percentile(df: pd.DataFrame, idx: int, lookback: int = 250) -> float:
    """Percentile rank of current 20-day ATR% within last `lookback` days."""
    if idx < lookback or "atr_pct" not in df.columns:
        return 100.0

    current_atr_pct = df.iloc[idx]["atr_pct"]
    if pd.isna(current_atr_pct):
        return 100.0

    # 20-day average ATR%
    atr_20_series = df["atr_pct"].rolling(20, min_periods=20).mean()
    window = atr_20_series.iloc[idx - lookback + 1 : idx + 1].dropna()
    if window.empty:
        return 100.0

    return float((window < current_atr_pct).sum() / len(window) * 100.0)
