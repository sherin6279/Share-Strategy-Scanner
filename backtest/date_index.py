"""Pre-indexed date lookups for fast historical backtests."""

from __future__ import annotations

from datetime import date

import pandas as pd

from config.settings import NIFTY50_SYMBOL


class DateIndex:
    """
    Trading calendar from NIFTY 50 plus O(1) symbol/date → row index lookups.

    Built once per backtest run to avoid repeated dataframe filtering.
    """

    def __init__(
        self,
        calendar: list[date],
        symbol_date_idx: dict[str, dict[date, int]],
        close_lookup: dict[str, dict[date, float]],
    ) -> None:
        self.calendar = calendar
        self._date_pos = {d: i for i, d in enumerate(calendar)}
        self.symbol_date_idx = symbol_date_idx
        self.close_lookup = close_lookup

    @classmethod
    def from_enriched(
        cls,
        enriched_map: dict[str, pd.DataFrame],
        benchmark_symbol: str = NIFTY50_SYMBOL,
    ) -> DateIndex:
        nifty = enriched_map.get(benchmark_symbol)
        if nifty is None or nifty.empty:
            raise ValueError(f"Benchmark {benchmark_symbol} not found in enriched data")

        calendar = [d for d in nifty["trade_date"].tolist()]
        symbol_date_idx: dict[str, dict[date, int]] = {}
        close_lookup: dict[str, dict[date, float]] = {}

        for symbol, df in enriched_map.items():
            if df.empty:
                continue
            idx_map: dict[date, int] = {}
            close_map: dict[date, float] = {}
            for i, row in df.iterrows():
                td = row["trade_date"]
                idx_map[td] = int(i)
                close_map[td] = float(row["close"])
            symbol_date_idx[symbol] = idx_map
            close_lookup[symbol] = close_map

        return cls(calendar, symbol_date_idx, close_lookup)

    def idx_map_for_date(self, signal_date: date) -> dict[str, int]:
        """Return {symbol: row_index} for symbols with data on signal_date."""
        result: dict[str, int] = {}
        for symbol, date_map in self.symbol_date_idx.items():
            idx = date_map.get(signal_date)
            if idx is not None:
                result[symbol] = idx
        return result

    def forward_date(self, signal_date: date, hold_days: int) -> date | None:
        pos = self._date_pos.get(signal_date)
        if pos is None:
            return None
        target = pos + hold_days
        if target >= len(self.calendar):
            return None
        return self.calendar[target]

    def close_on(self, symbol: str, trade_date: date) -> float | None:
        return self.close_lookup.get(symbol, {}).get(trade_date)

    def iter_signal_dates(
        self,
        start: date | None = None,
        end: date | None = None,
        hold_days: int = 5,
        step_days: int = 1,
    ):
        """Yield dates usable as signal dates (leave room for forward hold)."""
        cal = self.calendar
        if not cal:
            return

        start_pos = 0 if start is None else max(0, self._date_pos.get(start, 0))
        end_pos = len(cal) - hold_days - 1
        if end is not None:
            end_pos = min(end_pos, self._date_pos.get(end, end_pos))

        pos = start_pos
        while pos <= end_pos:
            yield cal[pos]
            pos += step_days
