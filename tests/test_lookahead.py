"""Look-ahead bias validation."""

from datetime import date

from backtest.date_index import DateIndex
from config.settings import NIFTY50_SYMBOL
from indicators import enrich_candles
from strategies.strategy_engine import StrategyEngine
from tests.fixtures import make_daily_candles


def test_context_uses_historical_nifty_not_future():
    start = date(2024, 1, 1)
    days = 250
    nifty = make_daily_candles(NIFTY50_SYMBOL, start, days, 100.0, -0.002)
    stock = make_daily_candles("TESTCO", start, days, 500.0, 0.001)

    full_enriched = {
        NIFTY50_SYMBOL: enrich_candles(nifty),
        "TESTCO": enrich_candles(stock),
    }
    date_index = DateIndex.from_enriched(full_enriched)

    mid_date = date_index.calendar[100]
    idx_map = date_index.idx_map_for_date(mid_date)

    engine = StrategyEngine()
    ctx = engine._build_context(
        full_enriched,
        idx_map,
        full_enriched[NIFTY50_SYMBOL],
        nifty_eval_idx=idx_map.get(NIFTY50_SYMBOL),
    )

    nifty_idx = idx_map[NIFTY50_SYMBOL]
    row_at_signal = full_enriched[NIFTY50_SYMBOL].iloc[nifty_idx]
    latest_row = full_enriched[NIFTY50_SYMBOL].iloc[-1]

    assert row_at_signal["trade_date"] != latest_row["trade_date"]

    import pandas as pd

    sma200 = row_at_signal.get("sma200")
    expected_uptrend = (
        not pd.isna(sma200) and row_at_signal["close"] > sma200
    )
    assert ctx["market_uptrend"] == expected_uptrend

    truncated_nifty = full_enriched[NIFTY50_SYMBOL].iloc[: nifty_idx + 1]
    truncated_ctx = engine._build_context(
        full_enriched,
        idx_map,
        truncated_nifty,
        nifty_eval_idx=len(truncated_nifty) - 1,
    )
    assert truncated_ctx["market_uptrend"] == ctx["market_uptrend"]
