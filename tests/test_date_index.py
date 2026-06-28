"""Tests for DateIndex."""

from datetime import date, timedelta

from backtest.date_index import DateIndex
from indicators import enrich_candles
from tests.fixtures import make_daily_candles


def test_date_index_lookup_and_forward():
    start = date(2025, 1, 1)
    nifty = make_daily_candles("NIFTY 50", start, 120, 100.0, 0.002)
    stock = make_daily_candles("RELIANCE", start, 120, 200.0, 0.003)

    enriched = {
        "NIFTY 50": enrich_candles(nifty),
        "RELIANCE": enrich_candles(stock),
    }
    idx = DateIndex.from_enriched(enriched)

    signal_date = idx.calendar[50]
    assert idx.idx_map_for_date(signal_date)["RELIANCE"] >= 0

    exit_date = idx.forward_date(signal_date, 5)
    assert exit_date is not None
    assert idx._date_pos[exit_date] == idx._date_pos[signal_date] + 5

    entry = idx.close_on("RELIANCE", signal_date)
    exit_p = idx.close_on("RELIANCE", exit_date)
    assert entry is not None and exit_p is not None
    assert exit_p > entry
