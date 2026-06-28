"""Tests for incremental market data refresh."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import patch

import pandas as pd
import pytest

from config.settings import CANDLE_DAYS, NIFTY50_SYMBOL
from data.refresh_service import MARKET_DATE_PROBE_DAYS, RefreshService
from database.duckdb_manager import DuckDBManager
from tests.fixtures import make_daily_candles


class MockKiteFetcher:
    """Minimal fetcher stub for refresh orchestration tests."""

    def __init__(self, market_date: date, symbol_prices: dict[str, float] | None = None) -> None:
        self.market_date = market_date
        self.symbol_prices = symbol_prices or {}
        self._instrument_map = {NIFTY50_SYMBOL: 1}
        self.calls: list[tuple[str, str]] = []

    def is_authenticated(self) -> bool:
        return True

    def load_instruments(self) -> None:
        return None

    def validate_token(self) -> bool:
        return True

    def get_token(self, symbol: str) -> int | None:
        if symbol not in self._instrument_map:
            self._instrument_map[symbol] = len(self._instrument_map) + 1
        return self._instrument_map[symbol]

    def fetch_historical_range(self, symbol, instrument_token, from_date, to_date):
        self.calls.append(("range", symbol))
        if symbol == NIFTY50_SYMBOL:
            return _candles(
                symbol,
                self.market_date - timedelta(days=MARKET_DATE_PROBE_DAYS),
                self.market_date,
                self.symbol_prices.get(symbol, 1000.0),
            )

        start = from_date.date() if isinstance(from_date, datetime) else from_date
        if start > self.market_date:
            return pd.DataFrame()
        return _candles(symbol, start, self.market_date, self.symbol_prices.get(symbol, 100.0))

    def fetch_historical(self, symbol, instrument_token, days=CANDLE_DAYS):
        self.calls.append(("backfill", symbol))
        start = self.market_date - timedelta(days=int(days * 1.4))
        return _candles(symbol, start, self.market_date, self.symbol_prices.get(symbol, 100.0))[-days:]


def _candles(symbol: str, start: date, end: date, price: float) -> pd.DataFrame:
    rows = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            rows.append(
                {
                    "symbol": symbol,
                    "trade_date": current,
                    "open": price,
                    "high": price + 1,
                    "low": price - 1,
                    "close": price,
                    "volume": 1000,
                }
            )
        current += timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.fixture
def refresh_db(tmp_path):
    db = DuckDBManager(db_path=tmp_path / "refresh.duckdb")
    yield db
    db.close()


def test_get_latest_candle_dates(refresh_db):
    start = date(2025, 1, 1)
    refresh_db.upsert_candles(make_daily_candles("AAA", start, 30, 100.0, 0.0))
    refresh_db.upsert_candles(make_daily_candles("BBB", start, 10, 200.0, 0.0))

    latest = refresh_db.get_latest_candle_dates(["AAA", "BBB", "MISSING"])
    assert "AAA" in latest
    assert "BBB" in latest
    assert "MISSING" not in latest
    assert latest["BBB"] < latest["AAA"]


@patch("data.refresh_service.load_nifty500_symbols", return_value=["AAA", "BBB"])
def test_initial_refresh_backfills_symbols(mock_symbols, refresh_db):
    market_date = date(2025, 6, 20)
    fetcher = MockKiteFetcher(market_date=market_date)
    service = RefreshService(db=refresh_db, fetcher=fetcher)

    summary = service.refresh()

    assert summary["symbols_updated"] == 3  # NIFTY 50 + AAA + BBB
    assert summary["symbols_skipped"] == 0
    assert summary["symbols_failed"] == 0
    assert refresh_db.get_candles("AAA").shape[0] == CANDLE_DAYS
    assert ("backfill", "AAA") in fetcher.calls
    assert ("range", NIFTY50_SYMBOL) in fetcher.calls


@patch("data.refresh_service.load_nifty500_symbols", return_value=["AAA"])
def test_incremental_refresh_skips_current_symbols(mock_symbols, refresh_db):
    market_date = date(2025, 6, 20)
    fetcher = MockKiteFetcher(market_date=market_date)

    existing = _candles("AAA", market_date - timedelta(days=30), market_date, 100.0)
    nifty = _candles(NIFTY50_SYMBOL, market_date - timedelta(days=30), market_date, 1000.0)
    refresh_db.upsert_candles(existing)
    refresh_db.upsert_candles(nifty)

    service = RefreshService(db=refresh_db, fetcher=fetcher)
    summary = service.refresh()

    assert summary["symbols_updated"] == 0
    assert summary["symbols_skipped"] == 2
    assert refresh_db.get_candles("AAA").shape[0] == len(existing)
    assert ("backfill", "AAA") not in fetcher.calls
    assert ("range", "AAA") not in fetcher.calls


@patch("data.refresh_service.load_nifty500_symbols", return_value=["AAA"])
def test_incremental_refresh_appends_only_missing_days(mock_symbols, refresh_db):
    market_date = date(2025, 6, 20)
    last_stored = market_date - timedelta(days=5)
    fetcher = MockKiteFetcher(market_date=market_date)

    existing = _candles("AAA", last_stored - timedelta(days=20), last_stored, 100.0)
    nifty = _candles(NIFTY50_SYMBOL, market_date - timedelta(days=5), market_date, 1000.0)
    refresh_db.upsert_candles(existing)
    refresh_db.upsert_candles(nifty)

    service = RefreshService(db=refresh_db, fetcher=fetcher)
    summary = service.refresh()

    assert summary["symbols_updated"] == 1
    assert summary["symbols_skipped"] == 1  # NIFTY 50 already current
    assert ("range", "AAA") in fetcher.calls
    assert ("backfill", "AAA") not in fetcher.calls

    all_rows = refresh_db.get_candles("AAA")
    assert all_rows["trade_date"].max() == market_date
    assert all_rows["trade_date"].min() == existing["trade_date"].min()
    assert len(all_rows) > len(existing)


@patch("data.refresh_service.load_nifty500_symbols", return_value=["AAA"])
def test_load_all_candles_returns_full_history_after_growth(mock_symbols, refresh_db):
    market_date = date(2025, 6, 20)
    fetcher = MockKiteFetcher(market_date=market_date)
    service = RefreshService(db=refresh_db, fetcher=fetcher)
    service.refresh()

    first_count = len(refresh_db.get_candles("AAA"))
    assert first_count == CANDLE_DAYS

    fetcher.market_date = market_date + timedelta(days=7)
    service.refresh()
    second_count = len(refresh_db.get_candles("AAA"))
    assert second_count > first_count

    candles_map = refresh_db.load_all_candles()
    assert len(candles_map["AAA"]) == second_count
