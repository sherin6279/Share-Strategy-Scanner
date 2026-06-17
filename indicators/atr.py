"""Average True Range indicator."""

from __future__ import annotations

import pandas as pd

try:
    import talib

    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        return pd.Series(
            talib.ATR(
                high.values.astype(float),
                low.values.astype(float),
                close.values.astype(float),
                timeperiod=period,
            ),
            index=high.index,
        )

except ImportError:
    import pandas_ta as ta

    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        result = ta.atr(high=high, low=low, close=close, length=period)
        return result


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add ATR and ATR% columns."""
    out = df.copy()
    out[f"atr{period}"] = atr(out["high"], out["low"], out["close"], period)
    out["atr_pct"] = out[f"atr{period}"] / out["close"] * 100.0
    return out
