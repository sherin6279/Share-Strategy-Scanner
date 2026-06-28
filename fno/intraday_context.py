"""Per-day intraday context without lookahead."""

from __future__ import annotations

from datetime import time

import pandas as pd

from indicators.intraday import avg_volume, ema_last, opening_range, vwap_up_to

SESSION_END = time(15, 15)
ORB_BARS = 3


class IntradayDayContext:
    """Session context for one symbol on one trading day."""

    def __init__(self, day_df: pd.DataFrame, orb_bars: int = ORB_BARS) -> None:
        self.day_df = day_df.reset_index(drop=True)
        self.orb_bars = orb_bars
        self.orb_high, self.orb_low = opening_range(self.day_df, orb_bars)

    def build_bar_context(self, bar_idx: int) -> dict:
        """Context available at bar_idx — uses only past/current bars."""
        row = self.day_df.iloc[bar_idx]
        bar_time = pd.Timestamp(row["trade_datetime"]).time()
        return {
            "vwap": vwap_up_to(self.day_df, bar_idx),
            "ema9": ema_last(self.day_df["close"], 9, bar_idx),
            "ema21": ema_last(self.day_df["close"], 21, bar_idx),
            "avg_vol_20": avg_volume(self.day_df, 20, bar_idx),
            "orb_high": self.orb_high,
            "orb_low": self.orb_low,
            "is_eod_bar": bar_time >= time(15, 10),
            "bar_idx": bar_idx,
        }
