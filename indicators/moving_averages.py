"""Moving average indicators."""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMA(20/50/200) and EMA(20) columns to OHLCV dataframe."""
    out = df.copy()
    out["sma20"] = sma(out["close"], 20)
    out["sma50"] = sma(out["close"], 50)
    out["sma200"] = sma(out["close"], 200)
    out["ema20"] = ema(out["close"], 20)
    return out


def sma_slope_positive(df: pd.DataFrame, period: int, idx: int, lookback: int = 5) -> bool:
    """Check if SMA slope is positive over lookback days ending at idx."""
    col = f"sma{period}"
    if col not in df.columns or idx < lookback:
        return False
    start_val = df.iloc[idx - lookback][col]
    end_val = df.iloc[idx][col]
    if pd.isna(start_val) or pd.isna(end_val):
        return False
    return end_val > start_val
