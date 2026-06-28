"""Integration test for equity backtest."""

from datetime import date

import pytest

from backtest.equity_backtest import EquityBacktester
from database.duckdb_manager import DuckDBManager
from indicators import enrich_candles
from tests.fixtures import make_daily_candles


@pytest.fixture
def temp_db(tmp_path):
    db = DuckDBManager(db_path=tmp_path / "test.duckdb")
    start = date(2024, 6, 1)
    for sym in ["NIFTY 50", "RELIANCE", "TCS"]:
        df = make_daily_candles(sym, start, 280, 100.0 + hash(sym) % 50, 0.001)
        db.upsert_candles(df)
    yield db
    db.close()


def test_equity_backtest_runs(temp_db):
    bt = EquityBacktester(db=temp_db)
    result = bt.run(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 6, 1),
        hold_days=5,
        step_days=10,
        strategy_ids=[1, 2],
        cost_bps=15,
    )
    assert result.segment == "equity"
    assert isinstance(result.trades, list)
    assert isinstance(result.summaries, list)
