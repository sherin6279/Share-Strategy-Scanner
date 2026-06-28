"""Tests for paper portfolio tracking."""

from datetime import date

import pytest

from database.duckdb_manager import DuckDBManager
from paper_trading.service import PaperTradingService
from tests.fixtures import make_daily_candles


@pytest.fixture
def paper_db(tmp_path):
    db = DuckDBManager(db_path=tmp_path / "paper.duckdb")
    start = date(2025, 1, 1)

    for sym, ret in [("RELIANCE", 0.01), ("TCS", -0.005), ("INFY", 0.008)]:
        df = make_daily_candles(sym, start, 120, 1000.0, ret)
        db.upsert_candles(df)

    scan_time = __import__("datetime").datetime(2025, 3, 1, 10, 0, 0)
    db.insert_scan_run("scan1", scan_time, "equity", 3, True, {1: 2, 6: 1})
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
                "strategy_id": 6,
                "symbol": "RELIANCE",
                "signal_date": date(2025, 2, 28),
                "score": 85.0,
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


def test_record_collapses_confluence(paper_db):
    svc = PaperTradingService(db=paper_db)
    result = svc.record_from_scan("scan1")
    assert result["holdings_added"] == 2
    assert result["confluence_count"] == 1

    portfolio = svc.get_portfolio()
    reliance = portfolio[portfolio["symbol"] == "RELIANCE"].iloc[0]
    assert reliance["source_type"] == "confluence"
    assert "Confluence" in reliance["source"]


def test_record_idempotent_per_scan(paper_db):
    svc = PaperTradingService(db=paper_db)
    svc.record_from_scan("scan1")
    again = svc.record_from_scan("scan1")
    assert again["already_recorded"] is True
    assert again["holdings_added"] == 0


def test_portfolio_pl_uses_latest_close(paper_db):
    svc = PaperTradingService(db=paper_db)
    svc.record_from_scan("scan1")
    df = svc.get_portfolio()
    assert len(df) == 2
    assert df["pl_pct"].notna().all()


def test_multiple_scans_accumulate(paper_db):
    scan_time2 = __import__("datetime").datetime(2025, 3, 8, 10, 0, 0)
    paper_db.insert_scan_run("scan2", scan_time2, "equity", 1, True, {1: 1})
    paper_db.insert_scan_results(
        [
            {
                "scan_run_id": "scan2",
                "scan_timestamp": scan_time2,
                "strategy_id": 1,
                "symbol": "INFY",
                "signal_date": date(2025, 3, 7),
                "score": 70.0,
                "trigger_price": 1500.0,
                "metrics": {},
            },
        ]
    )
    svc = PaperTradingService(db=paper_db)
    svc.record_from_scan("scan1")
    svc.record_from_scan("scan2")
    assert len(svc.get_portfolio()) == 3


def test_summarize_portfolio(paper_db):
    svc = PaperTradingService(db=paper_db)
    svc.record_from_scan("scan1")
    summary = svc.summarize_portfolio()
    assert summary["holding_count"] == 2
    assert summary["total_cost"] > 0
    assert summary["win_rate"] is not None
