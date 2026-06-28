"""Integration test for F&O intraday backtest."""

from datetime import date, timedelta

import pytest

from backtest.fno_backtest import FnoBacktester
from database.duckdb_manager import DuckDBManager
from tests.fixtures import make_intraday_day


@pytest.fixture
def temp_db_fno(tmp_path):
    db = DuckDBManager(db_path=tmp_path / "test_fno.duckdb")
    start = date(2025, 5, 1)
    frames = []
    for i in range(10):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        frames.append(make_intraday_day("NIFTY25MAYFUT", d, bars=40, base_price=24000 + i * 10))
    import pandas as pd

    combined = pd.concat(frames, ignore_index=True)
    db.upsert_intraday_candles(combined)
    yield db
    db.close()


def test_fno_backtest_runs(temp_db_fno):
    bt = FnoBacktester(db=temp_db_fno)
    result = bt.run(
        strategy_ids=[101],
        cost_bps=5,
    )
    assert result.segment == "fno"
    assert isinstance(result.trades, list)
