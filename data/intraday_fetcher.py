"""Fetch F&O intraday candles from Kite Connect."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from config.settings import REQUEST_DELAY_SEC
from data.kite_fetcher import KiteFetcher
from utils.logger import get_logger

logger = get_logger(__name__)

# Default liquid F&O universe (nearest month future tradingsymbols resolved at fetch time)
DEFAULT_FNO_UNDERLYINGS = ["NIFTY", "BANKNIFTY"]


class IntradayFetcher(KiteFetcher):
    """Extends KiteFetcher for NFO intraday candles."""

    def load_nfo_instruments(self) -> None:
        self._rate_limit()
        instruments = self.kite.instruments("NFO")
        self._instrument_map = {}
        for inst in instruments:
            ts = inst["tradingsymbol"]
            self._instrument_map[ts] = inst["instrument_token"]
        logger.info("Loaded %d NFO instruments", len(self._instrument_map))

    def nearest_future_symbol(self, underlying: str) -> str | None:
        """Find nearest expiry FUT tradingsymbol for an underlying."""
        if not self._instrument_map:
            self.load_nfo_instruments()

        candidates = [
            ts for ts in self._instrument_map
            if ts.startswith(underlying) and ts.endswith("FUT")
        ]
        if not candidates:
            return None
        return sorted(candidates)[0]

    def fetch_intraday(
        self,
        tradingsymbol: str,
        interval: str = "5minute",
        days: int = 30,
    ) -> pd.DataFrame:
        if not self._instrument_map:
            self.load_nfo_instruments()

        token = self._instrument_map.get(tradingsymbol)
        if token is None:
            logger.error("No NFO token for %s", tradingsymbol)
            return pd.DataFrame()

        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)

        for attempt in range(1, 4):
            try:
                self._rate_limit()
                records = self.kite.historical_data(
                    token, from_date, to_date, interval=interval
                )
                if not records:
                    return pd.DataFrame()

                df = pd.DataFrame(records)
                df = df.rename(columns={"date": "trade_datetime"})
                df.columns = [c.lower() for c in df.columns]
                df["trade_datetime"] = pd.to_datetime(df["trade_datetime"])
                df = df.sort_values("trade_datetime").reset_index(drop=True)
                df["symbol"] = tradingsymbol
                df["interval"] = interval
                return df[
                    ["symbol", "interval", "trade_datetime", "open", "high", "low", "close", "volume"]
                ]
            except Exception as exc:
                logger.warning(
                    "Intraday fetch failed for %s (attempt %d): %s",
                    tradingsymbol,
                    attempt,
                    exc,
                )

        return pd.DataFrame()

    def fetch_default_fno_universe(
        self,
        interval: str = "5minute",
        days: int = 30,
    ) -> list[pd.DataFrame]:
        results: list[pd.DataFrame] = []
        for underlying in DEFAULT_FNO_UNDERLYINGS:
            ts = self.nearest_future_symbol(underlying)
            if ts is None:
                logger.warning("No future found for %s", underlying)
                continue
            df = self.fetch_intraday(ts, interval=interval, days=days)
            if not df.empty:
                results.append(df)
        return results
