"""Tests for calendar forward returns."""

from datetime import date

from backtest.date_index import DateIndex
from backtest.forward_returns import forward_return_pct
from indicators import enrich_candles
from tests.fixtures import make_daily_candles


def test_forward_return_uses_calendar_not_index():
    start = date(2025, 1, 1)
    nifty = make_daily_candles("NIFTY 50", start, 100)
    # Stock missing some days — simulate gap
    stock = make_daily_candles("GAPSTOCK", start, 100)
    stock = stock.iloc[::2].reset_index(drop=True)  # every other day

    enriched = {
        "NIFTY 50": enrich_candles(nifty),
        "GAPSTOCK": enrich_candles(stock),
    }
    idx = DateIndex.from_enriched(enriched)
    signal_date = idx.calendar[20]
    exit_date = idx.forward_date(signal_date, 5)

    ret, _, entry, exit_p = forward_return_pct(idx, "GAPSTOCK", signal_date, 5)
    assert exit_date is not None
    if ret is not None:
        assert entry == idx.close_on("GAPSTOCK", signal_date)
        assert exit_p == idx.close_on("GAPSTOCK", exit_date)
