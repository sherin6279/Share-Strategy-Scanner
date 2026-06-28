"""Tests for paper trading P/L tracking."""

from datetime import date

import pytest

from database.duckdb_manager import DuckDBManager
from paper_trading.service import PaperTradingService
from tests.fixtures import make_daily_candles


@pytest.fixture
def paper_db(tmp_path):
    db = DuckDBManager(db_path=tmp_path / "paper.duckdb")
    start = date(2025, 1, 1)

    for sym, ret in [("RELIANCE", 0.01), ("TCS", -0.005), ("NIFTY 50", 0.002)]:
        df = make_daily_candles(sym, start, 120, 1000.0 if sym != "NIFTY 50" else 24000.0, ret)
        db.upsert_candles(df)

    scan_time = __import__("datetime").datetime(2025, 3, 1, 10, 0, 0)
    db.insert_scan_run("scan1", scan_time, "equity", 2, True, {1: 2})
    db.insert_scan_results(
        [
            {
                "scan_run_id": "scan1",
                "scan_timestamp": scan_time,
                "strategy_id": 1,
                "symbol": "RELIANCE",
                "signal_date": date(2025, 2, 28),
                "score": 80.0,
                "trigger_price": 1050.0,
                "metrics": {},
            },
            {
                "scan_run_id": "scan1",
                "scan_timestamp": scan_time,
                "strategy_id": 1,
                "symbol": "TCS",
                "signal_date": date(2025, 2, 28),
                "score": 75.0,
                "trigger_price": 980.0,
                "metrics": {},
            },
        ]
    )
    yield db
    db.close()


def test_record_from_scan_creates_batch(paper_db):
    svc = PaperTradingService(db=paper_db)
    result = svc.record_from_scan("scan1")
    assert result["position_count"] == 2
    assert result["already_recorded"] is False

    again = svc.record_from_scan("scan1")
    assert again["already_recorded"] is True


def test_compute_pl_uses_latest_close(paper_db):
    svc = PaperTradingService(db=paper_db)
    batch = svc.record_from_scan("scan1")
    pl = svc.compute_pl(batch["batch_id"], min_hold_days=0)

    assert len(pl) == 2
    assert pl["pl_pct"].notna().all()
    assert (pl["days_held"] >= 0).all()


def test_min_hold_days_filters(paper_db):
    svc = PaperTradingService(db=paper_db)
    batch = svc.record_from_scan("scan1")
    pl = svc.compute_pl(batch["batch_id"], min_hold_days=999)
    assert pl.empty


def test_summarize_pl(paper_db):
    svc = PaperTradingService(db=paper_db)
    batch = svc.record_from_scan("scan1")
    summary = svc.summarize_pl(batch["batch_id"])
    assert summary["position_count"] == 2
    assert summary["with_prices"] == 2
    assert summary["win_rate"] is not None
