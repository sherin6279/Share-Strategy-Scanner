"""Shared helper utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize candle dataframe columns and sort by date."""
    if df.empty:
        return df

    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    if "date" in out.columns and "trade_date" not in out.columns:
        out = out.rename(columns={"date": "trade_date"})

    out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.date
    out = out.sort_values("trade_date").reset_index(drop=True)
    return out


def safe_pct(numerator: float, denominator: float) -> float:
    """Compute percentage change safely."""
    if denominator == 0 or np.isnan(denominator) or np.isnan(numerator):
        return 0.0
    return (numerator / denominator - 1.0) * 100.0


def candle_position_score(row: pd.Series) -> float:
    """(Close - Low) / (High - Low) for a single candle."""
    high_low = row["high"] - row["low"]
    if high_low <= 0:
        return 0.0
    return (row["close"] - row["low"]) / high_low


def avg_traded_value_cr(df: pd.DataFrame, window: int, idx: int) -> float:
    """Average daily traded value in ₹ crore over `window` days ending at idx."""
    if idx < window - 1:
        return 0.0
    subset = df.iloc[idx - window + 1 : idx + 1]
    avg_value = (subset["close"] * subset["volume"]).mean()
    return avg_value / 1e7  # convert to crore


def drawdown_from_high(df: pd.DataFrame, window: int, idx: int) -> float:
    """Percentage drawdown from rolling high."""
    if idx < window - 1:
        return 100.0
    window_df = df.iloc[idx - window + 1 : idx + 1]
    high = window_df["high"].max()
    close = df.iloc[idx]["close"]
    if high <= 0:
        return 100.0
    return (1.0 - close / high) * 100.0


def touches_level(row: pd.Series, level: float, tolerance_pct: float = 0.5) -> bool:
    """Check if candle low touches a price level within tolerance."""
    if level <= 0:
        return False
    tolerance = level * tolerance_pct / 100.0
    return row["low"] <= level + tolerance


def clamp(value: float, low: float, high: float) -> float:
    """Clamp value between bounds."""
    return max(low, min(high, value))


def metrics_to_dict(metrics: dict[str, Any]) -> dict[str, Any]:
    """Convert metrics to JSON-serializable dict."""
    result: dict[str, Any] = {}
    for key, val in metrics.items():
        if isinstance(val, (np.floating, np.integer)):
            result[key] = float(val)
        elif isinstance(val, (pd.Timestamp,)):
            result[key] = str(val)
        else:
            result[key] = val
    return result
