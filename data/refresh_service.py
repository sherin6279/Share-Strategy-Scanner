"""Market data refresh orchestration."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Callable

import pandas as pd

from config.settings import CANDLE_DAYS, NIFTY50_SYMBOL
from data.kite_fetcher import KiteFetcher
from data.nifty500_loader import load_nifty500_symbols
from database.duckdb_manager import DuckDBManager
from utils.logger import get_logger

logger = get_logger(__name__)

ProgressCallback = Callable[[int, int, str, str], None]
MARKET_DATE_PROBE_DAYS = 14


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
        progress_callback: ProgressCallback | None = None,
        force_symbol_refresh: bool = False,
    ) -> dict:
        """
        Incrementally download NIFTY 50 + NIFTY 500 candles and upsert to DuckDB.

        First run per symbol backfills ~CANDLE_DAYS trading days. Later runs fetch
        only missing sessions and skip symbols already up to date.

        Returns summary dict with counts and failures.
        """
        if not self.fetcher.is_authenticated():
            raise RuntimeError(
                "Kite not authenticated. Set KITE_ACCESS_TOKEN or complete login."
            )

        if not self.fetcher._instrument_map:
            self.fetcher.load_instruments()

        symbols = load_nifty500_symbols(force_refresh=force_symbol_refresh)
        all_symbols = list(symbols)
        if NIFTY50_SYMBOL not in all_symbols:
            all_symbols = [NIFTY50_SYMBOL] + all_symbols

        latest_dates = self.db.get_latest_candle_dates(all_symbols)
        latest_market_date = self._resolve_latest_market_date(latest_dates)

        logger.info(
            "Incremental refresh for %d symbols (latest market session: %s)",
            len(all_symbols),
            latest_market_date,
        )

        total = len(all_symbols)
        updated: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []
        total_rows = 0

        for i, symbol in enumerate(all_symbols, start=1):
            stored_date = latest_dates.get(symbol)
            if stored_date is not None and stored_date >= latest_market_date:
                skipped.append(symbol)
                if progress_callback:
                    progress_callback(i, total, symbol, "skipped")
                continue

            token = self.fetcher.get_token(symbol)
            if token is None:
                logger.error("No instrument token for %s", symbol)
                failed.append(symbol)
                if progress_callback:
                    progress_callback(i, total, symbol, "failed")
                continue

            status = "backfill" if stored_date is None else "updating"
            if progress_callback:
                progress_callback(i, total, symbol, status)

            df = self._fetch_symbol_candles(symbol, token, stored_date)
            if df.empty:
                if stored_date is None:
                    failed.append(symbol)
                    if i == 1 and not self.fetcher.validate_token():
                        remaining = all_symbols[i:]
                        logger.error(
                            "Access token is invalid or expired. Aborting fetch — "
                            "%d symbols skipped.",
                            len(remaining),
                        )
                        failed.extend(s for s in remaining if s not in failed)
                        break
                else:
                    skipped.append(symbol)
                continue

            total_rows += self.db.upsert_candles(df)
            updated.append(symbol)
            latest_dates[symbol] = pd.to_datetime(df["trade_date"]).max().date()

        timestamp = datetime.now().isoformat()
        self.db.set_metadata("last_refresh_timestamp", timestamp)
        self.db.set_metadata("last_refresh_symbol_count", str(len(updated)))
        self.db.set_metadata("last_refresh_skipped_count", str(len(skipped)))
        self.db.set_metadata("last_refresh_failed_count", str(len(failed)))

        if failed:
            logger.warning("Refresh completed with %d failures: %s", len(failed), failed[:20])

        return {
            "timestamp": timestamp,
            "latest_market_date": latest_market_date.isoformat(),
            "symbols_updated": len(updated),
            "symbols_skipped": len(skipped),
            "symbols_failed": len(failed),
            "symbols_fetched": len(updated),
            "updated_symbols": updated,
            "skipped_symbols": skipped,
            "failed_symbols": failed,
            "rows_upserted": total_rows,
            "initial_backfill_days": CANDLE_DAYS,
        }

    def _resolve_latest_market_date(self, latest_dates: dict[str, date]) -> date:
        """
        Probe NIFTY 50 for the latest available trading session.

        Does not write to the database; the main loop handles NIFTY 50 storage.
        """
        token = self.fetcher.get_token(NIFTY50_SYMBOL)
        if token is None:
            return self._fallback_market_date(latest_dates)

        to_date = datetime.now()
        from_date = to_date - timedelta(days=MARKET_DATE_PROBE_DAYS)
        df = self.fetcher.fetch_historical_range(
            NIFTY50_SYMBOL,
            token,
            from_date,
            to_date,
        )
        if df.empty:
            return self._fallback_market_date(latest_dates)

        return pd.to_datetime(df["trade_date"]).max().date()

    def _fallback_market_date(self, latest_dates: dict[str, date]) -> date:
        nifty_date = latest_dates.get(NIFTY50_SYMBOL)
        if nifty_date is not None:
            return nifty_date
        if latest_dates:
            return max(latest_dates.values())
        return date.today() - timedelta(days=1)

    def _fetch_symbol_candles(
        self,
        symbol: str,
        instrument_token: int,
        latest_stored: date | None,
    ) -> pd.DataFrame:
        if latest_stored is None:
            return self.fetcher.fetch_historical(symbol, instrument_token, days=CANDLE_DAYS)

        from_date = datetime.combine(latest_stored + timedelta(days=1), time.min)
        to_date = datetime.now()
        if from_date.date() > to_date.date():
            return pd.DataFrame()

        df = self.fetcher.fetch_historical_range(
            symbol,
            instrument_token,
            from_date,
            to_date,
        )
        if df.empty:
            return df

        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        return df[df["trade_date"] > latest_stored].reset_index(drop=True)
