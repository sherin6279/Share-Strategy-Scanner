"""Bar-safe intraday indicators (no lookahead)."""

from __future__ import annotations

import pandas as pd


def vwap_up_to(df: pd.DataFrame, idx: int) -> float:
    """VWAP using bars 0..idx inclusive."""
    window = df.iloc[: idx + 1]
    vol = window["volume"].sum()
    if vol <= 0:
        return float(window.iloc[-1]["close"])
    tp = (window["close"] * window["volume"]).sum()
    return float(tp / vol)


def ema_last(series: pd.Series, period: int, idx: int) -> float:
    """EMA at idx using only data up to idx."""
    if idx < period - 1:
        return float("nan")
    return float(series.iloc[: idx + 1].ewm(span=period, adjust=False).mean().iloc[-1])


def opening_range(df: pd.DataFrame, orb_bars: int = 3) -> tuple[float, float]:
    """High/low of the first `orb_bars` bars in the session."""
    if len(df) < orb_bars:
        return float("nan"), float("nan")
    orb = df.iloc[:orb_bars]
    return float(orb["high"].max()), float(orb["low"].min())


def avg_volume(df: pd.DataFrame, period: int, idx: int) -> float:
    if idx < period - 1:
        return float("nan")
    return float(df.iloc[idx - period + 1 : idx + 1]["volume"].mean())
