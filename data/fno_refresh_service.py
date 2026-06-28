"""F&O intraday data refresh."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from config.settings import (
    FNO_INDEX_UNDERLYINGS,
    FNO_INTRADAY_DAYS,
    FNO_INTERVAL,
    FNO_STOCK_COUNT,
    FNO_STOCK_VOLUME_LOOKBACK,
)
from data.fno_universe import build_fno_refresh_targets
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
        interval: str = FNO_INTERVAL,
        days: int = FNO_INTRADAY_DAYS,
        stock_count: int = FNO_STOCK_COUNT,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        if not self.fetcher.is_authenticated():
            raise RuntimeError("Kite not authenticated.")

        if not self.fetcher.validate_token():
            raise RuntimeError("Kite access token invalid or expired.")

        targets = build_fno_refresh_targets(
            self.fetcher,
            self.db,
            index_underlyings=FNO_INDEX_UNDERLYINGS,
            stock_count=stock_count,
            volume_lookback_days=FNO_STOCK_VOLUME_LOOKBACK,
        )
        total = len(targets)
        rows = 0
        fetched: list[str] = []
        failed: list[str] = []
        index_count = 0
        stock_count_fetched = 0

        for i, target in enumerate(targets, start=1):
            ts = target["tradingsymbol"]
            label = f"{target['underlying']} → {ts}"
            if progress_callback:
                progress_callback(i, total, label)

            df = self.fetcher.fetch_intraday(ts, interval=interval, days=days)
            if df.empty:
                failed.append(ts)
                continue

            rows += self.db.upsert_intraday_candles(df)
            fetched.append(ts)
            if target["segment"] == "index":
                index_count += 1
            else:
                stock_count_fetched += 1

        ts_now = datetime.now().isoformat()
        self.db.set_metadata("last_fno_refresh_timestamp", ts_now)
        self.db.set_metadata("last_fno_refresh_symbols", ",".join(fetched))
        self.db.set_metadata("last_fno_refresh_index_count", str(index_count))
        self.db.set_metadata("last_fno_refresh_stock_count", str(stock_count_fetched))

        return {
            "timestamp": ts_now,
            "symbols_fetched": len(fetched),
            "index_fetched": index_count,
            "stock_fetched": stock_count_fetched,
            "symbols_failed": len(failed),
            "failed_symbols": failed,
            "rows_upserted": rows,
            "interval": interval,
            "targets_requested": total,
        }
