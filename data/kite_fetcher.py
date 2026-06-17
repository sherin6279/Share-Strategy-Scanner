"""Kite Connect API client for historical data."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Callable

import pandas as pd
from kiteconnect import KiteConnect

from config.settings import (
    CANDLE_DAYS,
    KITE_ACCESS_TOKEN,
    KITE_API_KEY,
    KITE_API_SECRET,
    NIFTY50_SYMBOL,
    REQUEST_DELAY_SEC,
    RETRY_COUNT,
)
from utils.helpers import ensure_ohlcv
from utils.logger import get_logger

logger = get_logger(__name__)


class KiteFetcher:
    """Fetches historical candles from Zerodha Kite Connect."""

    def __init__(
        self,
        api_key: str | None = None,
        access_token: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        self.api_key = api_key or KITE_API_KEY
        self.api_secret = api_secret or KITE_API_SECRET
        self.access_token = access_token or KITE_ACCESS_TOKEN
        self._kite: KiteConnect | None = None
        self._instrument_map: dict[str, int] = {}
        self._last_request_time: float = 0.0

    @property
    def kite(self) -> KiteConnect:
        if self._kite is None:
            if not self.api_key:
                raise ValueError("KITE_API_KEY not set in environment")
            self._kite = KiteConnect(api_key=self.api_key)
            if self.access_token:
                self._kite.set_access_token(self.access_token)
        return self._kite

    def is_authenticated(self) -> bool:
        return bool(self.api_key and self.access_token)

    def validate_token(self) -> bool:
        """
        Verify the access token is valid by making a lightweight API call.

        Returns True if the token is accepted by Kite, False otherwise.
        Kite tokens expire daily; this catches stale tokens before starting
        a 505-symbol fetch loop that would otherwise waste ~15 minutes.
        """
        try:
            self.kite.profile()
            return True
        except Exception as exc:
            logger.warning("Token validation failed: %s", exc)
            return False

    def login_url(self) -> str:
        """Return Kite login URL for OAuth flow."""
        return self.kite.login_url()

    def generate_session(self, request_token: str) -> str:
        """Exchange request token for access token."""
        if not self.api_secret:
            raise ValueError("KITE_API_SECRET not set in environment")
        session = self.kite.generate_session(request_token, api_secret=self.api_secret)
        self.access_token = session["access_token"]
        self.kite.set_access_token(self.access_token)
        return self.access_token

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY_SEC:
            time.sleep(REQUEST_DELAY_SEC - elapsed)
        self._last_request_time = time.time()

    def load_instruments(self) -> None:
        """Build symbol -> instrument_token map for NSE equities and indices."""
        self._rate_limit()
        instruments = self.kite.instruments("NSE")
        self._instrument_map = {}
        for inst in instruments:
            ts = inst["tradingsymbol"]
            self._instrument_map[ts] = inst["instrument_token"]
        logger.info("Loaded %d NSE instruments", len(self._instrument_map))

    def get_token(self, symbol: str) -> int | None:
        """Get instrument token for NSE tradingsymbol."""
        return self._instrument_map.get(symbol)

    def fetch_historical(
        self,
        symbol: str,
        instrument_token: int,
        days: int = CANDLE_DAYS,
    ) -> pd.DataFrame:
        """Fetch daily candles with retry logic."""
        to_date = datetime.now()
        from_date = to_date - timedelta(days=int(days * 1.6))  # buffer for holidays

        for attempt in range(1, RETRY_COUNT + 1):
            try:
                self._rate_limit()
                records = self.kite.historical_data(
                    instrument_token,
                    from_date,
                    to_date,
                    interval="day",
                )
                if not records:
                    return pd.DataFrame()

                df = pd.DataFrame(records)
                df = df.rename(columns={"date": "trade_date"})
                df["symbol"] = symbol
                df = ensure_ohlcv(df)
                # Keep only last `days` candles
                if len(df) > days:
                    df = df.iloc[-days:].reset_index(drop=True)
                return df[["symbol", "trade_date", "open", "high", "low", "close", "volume"]]

            except Exception as exc:
                logger.warning(
                    "Fetch failed for %s (attempt %d/%d): %s",
                    symbol,
                    attempt,
                    RETRY_COUNT,
                    exc,
                )
                if attempt < RETRY_COUNT:
                    time.sleep(REQUEST_DELAY_SEC * attempt)

        return pd.DataFrame()

    def fetch_all_symbols(
        self,
        symbols: list[str],
        progress_callback: Callable[[int, int, str], None] | None = None,
        include_nifty50: bool = True,
    ) -> tuple[list[pd.DataFrame], list[str]]:
        """
        Fetch candles for all symbols.

        Returns (successful_dataframes, failed_symbols).
        """
        if not self._instrument_map:
            self.load_instruments()

        all_symbols = list(symbols)
        if include_nifty50 and NIFTY50_SYMBOL not in all_symbols:
            all_symbols = [NIFTY50_SYMBOL] + all_symbols

        total = len(all_symbols)
        results: list[pd.DataFrame] = []
        failed: list[str] = []
        _AUTH_ERRORS = ("incorrect", "invalid", "expired", "unauthorised", "unauthorized")

        for i, symbol in enumerate(all_symbols, start=1):
            if progress_callback:
                progress_callback(i, total, symbol)

            token = self.get_token(symbol)
            if token is None:
                logger.error("No instrument token for %s", symbol)
                failed.append(symbol)
                continue

            df = self.fetch_historical(symbol, token)
            if df.empty:
                failed.append(symbol)
                # Fast-fail: if the first real fetch fails, check whether the
                # token itself is the problem before burning time on all symbols.
                if i == 1:
                    last_log = logger.handlers  # already logged in fetch_historical
                    if not self.validate_token():
                        remaining = all_symbols[i:]
                        logger.error(
                            "Access token is invalid or expired. "
                            "Aborting fetch — %d symbols skipped. "
                            "Generate a fresh token via the Kite login flow.",
                            len(remaining),
                        )
                        failed.extend(remaining)
                        return results, failed
            else:
                results.append(df)

        return results, failed
