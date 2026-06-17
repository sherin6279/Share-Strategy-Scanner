"""RSI indicator with TA-Lib fallback to pandas-ta."""

from __future__ import annotations

import pandas as pd

try:
    import talib

    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        return pd.Series(
            talib.RSI(series.values.astype(float), timeperiod=period),
            index=series.index,
        )

except ImportError:
    import pandas_ta as ta

    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        return ta.rsi(series, length=period)


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI column to dataframe."""
    out = df.copy()
    out[f"rsi{period}"] = rsi(out["close"], period)
    return out
