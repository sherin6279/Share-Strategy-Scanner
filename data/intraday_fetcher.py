"""Fetch F&O intraday candles from Kite Connect."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from config.settings import FNO_INDEX_UNDERLYINGS, REQUEST_DELAY_SEC
from data.kite_fetcher import KiteFetcher
from utils.logger import get_logger

logger = get_logger(__name__)


class IntradayFetcher(KiteFetcher):
    """Extends KiteFetcher for NFO intraday candles."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._nfo_instruments: list[dict[str, Any]] = []

    def load_nfo_instruments(self) -> None:
        self._rate_limit()
        instruments = self.kite.instruments("NFO")
        self._nfo_instruments = instruments
        self._instrument_map = {
            inst["tradingsymbol"]: inst["instrument_token"] for inst in instruments
        }
        logger.info("Loaded %d NFO instruments", len(self._instrument_map))

    def _live_futures(self) -> list[dict[str, Any]]:
        if not self._nfo_instruments:
            self.load_nfo_instruments()
        today = datetime.now().date()
        live: list[dict[str, Any]] = []
        for inst in self._nfo_instruments:
            if inst.get("instrument_type") != "FUT":
                continue
            expiry = inst.get("expiry")
            if expiry is None:
                continue
            exp_date = pd.Timestamp(expiry).date()
            if exp_date < today:
                continue
            live.append(inst)
        return live

    def list_live_fut_underlyings(self) -> set[str]:
        """Underlying names with at least one non-expired FUT contract."""
        return {inst["name"] for inst in self._live_futures()}

    def nearest_future_symbol(self, underlying: str) -> str | None:
        """Find nearest-expiry FUT tradingsymbol for an underlying."""
        candidates: list[tuple[Any, str]] = []
        for inst in self._live_futures():
            if inst.get("name") != underlying:
                continue
            expiry = pd.Timestamp(inst["expiry"]).date()
            candidates.append((expiry, inst["tradingsymbol"]))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

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
                    [
                        "symbol",
                        "interval",
                        "trade_datetime",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
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
        for underlying in FNO_INDEX_UNDERLYINGS:
            ts = self.nearest_future_symbol(underlying)
            if ts is None:
                logger.warning("No future found for %s", underlying)
                continue
            df = self.fetch_intraday(ts, interval=interval, days=days)
            if not df.empty:
                results.append(df)
        return results
