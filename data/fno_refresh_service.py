"""F&O intraday data refresh."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import pandas as pd

from data.intraday_fetcher import IntradayFetcher
from database.duckdb_manager import DuckDBManager
from utils.logger import get_logger

logger = get_logger(__name__)


class FnoRefreshService:
    """Download and store F&O intraday candles."""

    def __init__(
        self,
        db: DuckDBManager | None = None,
        fetcher: IntradayFetcher | None = None,
    ) -> None:
        self.db = db or DuckDBManager()
        self.fetcher = fetcher or IntradayFetcher()

    def refresh(
        self,
        interval: str = "5minute",
        days: int = 30,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        if not self.fetcher.is_authenticated():
            raise RuntimeError("Kite not authenticated.")

        if not self.fetcher.validate_token():
            raise RuntimeError("Kite access token invalid or expired.")

        underlyings = ["NIFTY", "BANKNIFTY"]
        total = len(underlyings)
        rows = 0
        fetched: list[str] = []
        failed: list[str] = []

        for i, underlying in enumerate(underlyings, start=1):
            ts = self.fetcher.nearest_future_symbol(underlying)
            if progress_callback:
                progress_callback(i, total, ts or underlying)

            if ts is None:
                failed.append(underlying)
                continue

            df = self.fetcher.fetch_intraday(ts, interval=interval, days=days)
            if df.empty:
                failed.append(ts)
                continue

            rows += self.db.upsert_intraday_candles(df)
            fetched.append(ts)

        ts_now = datetime.now().isoformat()
        self.db.set_metadata("last_fno_refresh_timestamp", ts_now)
        self.db.set_metadata("last_fno_refresh_symbols", ",".join(fetched))

        return {
            "timestamp": ts_now,
            "symbols_fetched": len(fetched),
            "symbols_failed": len(failed),
            "failed_symbols": failed,
            "rows_upserted": rows,
            "interval": interval,
        }
