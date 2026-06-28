"""Tests for F&O universe builder."""

from datetime import date, datetime

import pytest

from data.fno_universe import build_fno_refresh_targets
from data.intraday_fetcher import IntradayFetcher
from database.duckdb_manager import DuckDBManager
from tests.fixtures import make_daily_candles


class MockIntradayFetcher(IntradayFetcher):
    """Fetcher with canned NFO instruments (no Kite API)."""

    def __init__(self) -> None:
        super().__init__()
        today = date.today()
        self._nfo_instruments = [
            {
                "tradingsymbol": "NIFTY26AUGFUT",
                "instrument_token": 1,
                "name": "NIFTY",
                "instrument_type": "FUT",
                "expiry": datetime(today.year, today.month + 1 if today.month < 12 else 12, 25),
            },
            {
                "tradingsymbol": "BANKNIFTY26AUGFUT",
                "instrument_token": 2,
                "name": "BANKNIFTY",
                "instrument_type": "FUT",
                "expiry": datetime(today.year, today.month + 1 if today.month < 12 else 12, 25),
            },
            {
                "tradingsymbol": "RELIANCE26AUGFUT",
                "instrument_token": 3,
                "name": "RELIANCE",
                "instrument_type": "FUT",
                "expiry": datetime(today.year, today.month + 1 if today.month < 12 else 12, 25),
            },
            {
                "tradingsymbol": "TCS26AUGFUT",
                "instrument_token": 4,
                "name": "TCS",
                "instrument_type": "FUT",
                "expiry": datetime(today.year, today.month + 1 if today.month < 12 else 12, 25),
            },
            {
                "tradingsymbol": "OLDCO25JANFUT",
                "instrument_token": 5,
                "name": "OLDCO",
                "instrument_type": "FUT",
                "expiry": datetime(2020, 1, 25),
            },
        ]
        self._instrument_map = {
            i["tradingsymbol"]: i["instrument_token"] for i in self._nfo_instruments
        }

    def load_nfo_instruments(self) -> None:
        pass


@pytest.fixture
def universe_db(tmp_path, monkeypatch):
    db = DuckDBManager(db_path=tmp_path / "fno_univ.duckdb")
    start = date(2025, 1, 1)
    for sym, vol in [("RELIANCE", 5_000_000), ("TCS", 2_000_000), ("INFY", 8_000_000)]:
        df = make_daily_candles(sym, start, 30, 1000.0, 0.001)
        df["volume"] = vol
        db.upsert_candles(df)

    monkeypatch.setattr(
        "data.fno_universe.load_nifty500_symbols",
        lambda force_refresh=False: ["RELIANCE", "TCS", "INFY", "NOTFNO"],
    )
    yield db
    db.close()


def test_nearest_future_skips_expired():
    fetcher = MockIntradayFetcher()
    assert fetcher.nearest_future_symbol("OLDCO") is None
    assert fetcher.nearest_future_symbol("NIFTY") == "NIFTY26AUGFUT"


def test_build_universe_index_plus_stocks(universe_db):
    fetcher = MockIntradayFetcher()
    targets = build_fno_refresh_targets(
        fetcher,
        universe_db,
        index_underlyings=["NIFTY", "BANKNIFTY"],
        stock_count=2,
    )
    segments = {t["segment"] for t in targets}
    assert "index" in segments
    assert "stock" in segments
    assert len(targets) == 4  # 2 index + 2 stocks (RELIANCE, TCS — INFY has no mock FUT)
    stock_underlyings = [t["underlying"] for t in targets if t["segment"] == "stock"]
    assert stock_underlyings[0] == "RELIANCE"  # higher volume than TCS


def test_list_live_fut_underlyings():
    fetcher = MockIntradayFetcher()
    names = fetcher.list_live_fut_underlyings()
    assert "NIFTY" in names
    assert "RELIANCE" in names
    assert "OLDCO" not in names
