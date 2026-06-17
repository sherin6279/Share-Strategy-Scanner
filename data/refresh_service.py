"""Market data refresh orchestration."""

from __future__ import annotations

from datetime import datetime
from typing import Callable

import pandas as pd

from config.settings import CANDLE_DAYS
from data.kite_fetcher import KiteFetcher
from data.nifty500_loader import load_nifty500_symbols
from database.duckdb_manager import DuckDBManager
from utils.logger import get_logger

logger = get_logger(__name__)


class RefreshService:
    """Orchestrates downloading and storing market data."""

    def __init__(
        self,
        db: DuckDBManager | None = None,
        fetcher: KiteFetcher | None = None,
    ) -> None:
        self.db = db or DuckDBManager()
        self.fetcher = fetcher or KiteFetcher()

    def refresh(
        self,
        progress_callback: Callable[[int, int, str], None] | None = None,
        force_symbol_refresh: bool = False,
    ) -> dict:
        """
        Download NIFTY 50 + NIFTY 500 candles and upsert to DuckDB.

        Returns summary dict with counts and failures.
        """
        if not self.fetcher.is_authenticated():
            raise RuntimeError(
                "Kite not authenticated. Set KITE_ACCESS_TOKEN or complete login."
            )

        symbols = load_nifty500_symbols(force_refresh=force_symbol_refresh)
        logger.info("Starting refresh for %d NIFTY 500 symbols + NIFTY 50", len(symbols))

        dataframes, failed = self.fetcher.fetch_all_symbols(
            symbols,
            progress_callback=progress_callback,
            include_nifty50=True,
        )

        total_rows = 0
        for df in dataframes:
            total_rows += self.db.upsert_candles(df)

        timestamp = datetime.now().isoformat()
        self.db.set_metadata("last_refresh_timestamp", timestamp)
        self.db.set_metadata("last_refresh_symbol_count", str(len(dataframes)))
        self.db.set_metadata("last_refresh_failed_count", str(len(failed)))

        if failed:
            logger.warning("Refresh completed with %d failures: %s", len(failed), failed[:20])

        return {
            "timestamp": timestamp,
            "symbols_fetched": len(dataframes),
            "symbols_failed": len(failed),
            "failed_symbols": failed,
            "rows_upserted": total_rows,
            "candle_days": CANDLE_DAYS,
        }
