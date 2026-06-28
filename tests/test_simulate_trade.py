"""Tests for equity trade simulation."""

from datetime import date

from backtest.date_index import DateIndex
from backtest.simulate_trade import simulate_equity_trade
from indicators import enrich_candles
from tests.fixtures import make_daily_candles


def test_simulate_hits_target():
    start = date(2025, 1, 1)
    stock = make_daily_candles("TARGET", start, 30, 100.0, 0.02)
    enriched = enrich_candles(stock)
    idx_obj = DateIndex.from_enriched({"NIFTY 50": enriched, "TARGET": enriched})

    signal_date = idx_obj.calendar[10]
    entry = idx_obj.close_on("TARGET", signal_date)
    assert entry is not None

    ret, exit_date, reason, _ = simulate_equity_trade(
        enriched, idx_obj, "TARGET", signal_date, entry,
        stop_pct=5.0, target_pct=5.0, max_hold_days=5,
    )
    assert ret is not None
    assert reason in ("target", "max_hold", "stop_loss")
    assert exit_date is not None
