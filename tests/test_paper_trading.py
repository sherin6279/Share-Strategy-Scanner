"""Tests for paper portfolio tracking."""

from datetime import date

import pytest

from database.duckdb_manager import DuckDBManager
from paper_trading.service import PaperTradingService
from tests.fixtures import make_daily_candles


def _add_scan(db, run_id, scan_time, results):
    db.insert_scan_run(run_id, scan_time, "equity", len(results), True, {1: len(results)})
    db.insert_scan_results(
        [
            {
                "scan_run_id": run_id,
                "scan_timestamp": scan_time,
                "strategy_id": r["strategy_id"],
                "symbol": r["symbol"],
                "signal_date": r["signal_date"],
                "score": r["score"],
                "trigger_price": r["trigger_price"],
                "metrics": {},
            }
            for r in results
        ]
    )


@pytest.fixture
def paper_db(tmp_path):
    db = DuckDBManager(db_path=tmp_path / "paper.duckdb")
    start = date(2025, 1, 1)

    for sym, ret in [("RELIANCE", 0.01), ("TCS", -0.005), ("INFY", 0.008)]:
        df = make_daily_candles(sym, start, 120, 1000.0, ret)
        db.upsert_candles(df)

    scan_time = __import__("datetime").datetime(2025, 3, 1, 10, 0, 0)
    _add_scan(
        db,
        "scan1",
        scan_time,
        [
            {
                "strategy_id": 1,
                "symbol": "RELIANCE",
                "signal_date": date(2025, 2, 28),
                "score": 80.0,
                "trigger_price": 1050.0,
            },
            {
                "strategy_id": 6,
                "symbol": "RELIANCE",
                "signal_date": date(2025, 2, 28),
                "score": 85.0,
                "trigger_price": 1050.0,
            },
            {
                "strategy_id": 1,
                "symbol": "TCS",
                "signal_date": date(2025, 2, 28),
                "score": 75.0,
                "trigger_price": 980.0,
            },
        ],
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


def test_same_day_second_scan_skips_duplicate_symbols(paper_db):
    svc = PaperTradingService(db=paper_db)
    svc.record_from_scan("scan1")

    scan_time2 = __import__("datetime").datetime(2025, 3, 1, 15, 0, 0)
    _add_scan(
        paper_db,
        "scan2",
        scan_time2,
        [
            {
                "strategy_id": 2,
                "symbol": "RELIANCE",
                "signal_date": date(2025, 2, 28),
                "score": 90.0,
                "trigger_price": 1050.0,
            },
            {
                "strategy_id": 2,
                "symbol": "INFY",
                "signal_date": date(2025, 2, 28),
                "score": 70.0,
                "trigger_price": 1500.0,
            },
        ],
    )
    result = svc.record_from_scan("scan2")
    assert result["holdings_added"] == 1
    assert result["holdings_skipped_duplicate"] == 1
    assert len(svc.get_portfolio()) == 3


def test_sync_imports_historical_scans(paper_db):
    scan_time2 = __import__("datetime").datetime(2025, 3, 8, 10, 0, 0)
    _add_scan(
        paper_db,
        "scan2",
        scan_time2,
        [
            {
                "strategy_id": 1,
                "symbol": "INFY",
                "signal_date": date(2025, 3, 7),
                "score": 70.0,
                "trigger_price": 1500.0,
            },
        ],
    )
    svc = PaperTradingService(db=paper_db)
    sync = svc.sync_portfolio_from_scans()
    assert sync["holdings_added"] == 3
    assert sync["scans_processed"] == 2
    assert len(svc.get_portfolio()) == 3

    again = svc.sync_portfolio_from_scans()
    assert again["holdings_added"] == 0
    assert again["scans_skipped"] == 2


def test_portfolio_pl_uses_latest_close(paper_db):
    svc = PaperTradingService(db=paper_db)
    svc.record_from_scan("scan1")
    df = svc.get_portfolio()
    assert len(df) == 2
    assert df["pl_pct"].notna().all()


def test_summarize_portfolio(paper_db):
    svc = PaperTradingService(db=paper_db)
    svc.record_from_scan("scan1")
    summary = svc.summarize_portfolio()
    assert summary["holding_count"] == 2
    assert summary["total_cost"] > 0
    assert summary["win_rate"] is not None


def test_summarize_by_date_and_strategy(paper_db):
    svc = PaperTradingService(db=paper_db)
    svc.record_from_scan("scan1")
    df = svc.get_portfolio()
    by_date = svc.summarize_by_date(df)
    by_strat = svc.summarize_by_strategy(df)
    assert len(by_date) == 1
    assert len(by_strat) >= 2
    assert "Confluence" in by_strat["strategy"].values
