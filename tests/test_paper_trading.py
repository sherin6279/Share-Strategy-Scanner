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
    assert paper_db.count_unprocessed_equity_scan_runs() == 2
    sync = svc.sync_portfolio_from_scans()
    assert sync["scans_pending"] == 2
    assert sync["holdings_added"] == 3
    assert sync["scans_processed"] == 2
    assert len(svc.get_portfolio()) == 3

    again = svc.sync_portfolio_from_scans()
    assert again["holdings_added"] == 0
    assert again["scans_processed"] == 0
    assert again["scans_pending"] == 0


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


def test_weighted_vs_average_pl_pct(tmp_path):
    """Expensive losers can make total P/L negative while avg % is positive."""
    db = DuckDBManager(db_path=tmp_path / "pl.duckdb")
    start = date(2025, 1, 1)
    for sym, price in [("CHEAP", 100.0), ("EXPENSIVE", 50000.0)]:
        df = make_daily_candles(sym, start, 30, price, 0.0)
        db.upsert_candles(df)
    purchase_date = db.get_candles("CHEAP")["trade_date"].iloc[-1]
    db.insert_portfolio_holdings(
        [
            {
                "holding_id": "h1",
                "scan_run_id": "s1",
                "symbol": "CHEAP",
                "source_type": "strategy",
                "source_label": "Basic Breakout",
                "strategy_ids": [1],
                "purchase_date": purchase_date,
                "purchase_price": 100.0,
                "quantity": 1,
                "score": 8.0,
                "created_at": __import__("datetime").datetime.now(),
            },
            {
                "holding_id": "h2",
                "scan_run_id": "s1",
                "symbol": "EXPENSIVE",
                "source_type": "strategy",
                "source_label": "Basic Breakout",
                "strategy_ids": [1],
                "purchase_date": purchase_date,
                "purchase_price": 50000.0,
                "quantity": 1,
                "score": 8.0,
                "created_at": __import__("datetime").datetime.now(),
            },
        ]
    )
    # Bump cheap +10%, expensive -5% in latest candles
    cheap = db.get_candles("CHEAP")
    cheap.iloc[-1, cheap.columns.get_loc("close")] = 110.0
    db.upsert_candles(cheap)
    exp = db.get_candles("EXPENSIVE")
    exp.iloc[-1, exp.columns.get_loc("close")] = 47500.0
    db.upsert_candles(exp)

    svc = PaperTradingService(db=db)
    summary = svc.summarize_portfolio()
    assert summary["total_pl_amount"] < 0
    assert summary["avg_holding_pl_pct"] > 0
    db.close()


def test_portfolio_only_top_five_rs_leaders(paper_db):
    scan_time = __import__("datetime").datetime(2025, 4, 1, 10, 0, 0)
    results = [
        {
            "strategy_id": 6,
            "symbol": f"RS{i}",
            "signal_date": date(2025, 3, 31),
            "score": float(i),
            "trigger_price": 1000.0 + i,
        }
        for i in range(10)
    ]
    paper_db.insert_scan_run("rs_scan", scan_time, "equity", 10, True, {6: 10})
    paper_db.insert_scan_results(
        [
            {
                "scan_run_id": "rs_scan",
                "scan_timestamp": scan_time,
                **r,
                "metrics": {},
            }
            for r in results
        ]
    )
    for sym in [f"RS{i}" for i in range(10)]:
        df = make_daily_candles(sym, date(2025, 1, 1), 90, 1000.0, 0.001)
        paper_db.upsert_candles(df)

    svc = PaperTradingService(db=paper_db)
    result = svc.record_from_scan("rs_scan")
    assert result["holdings_added"] == 5
    symbols = set(svc.get_portfolio()["symbol"])
    assert symbols == {f"RS{i}" for i in range(5, 10)}


def test_one_time_rs_cleanup_keeps_top_five_per_day(paper_db):
    scan_time = __import__("datetime").datetime(2025, 4, 1, 10, 0, 0)
    holdings = []
    for i in range(8):
        holdings.append(
            {
                "holding_id": f"h{i}",
                "scan_run_id": "old_rs",
                "symbol": f"RS{i}",
                "source_type": "strategy",
                "source_label": "RS Leaders",
                "strategy_ids": [6],
                "purchase_date": date(2025, 3, 31),
                "purchase_price": 100.0 + i,
                "quantity": 1,
                "score": float(i),
                "created_at": scan_time,
            }
        )
    paper_db.insert_portfolio_holdings(holdings)

    svc = PaperTradingService(db=paper_db)
    result = svc.run_one_time_rs_cleanup()
    assert result["removed"] == 3
    assert paper_db.count_portfolio_holdings() == 5

    again = svc.run_one_time_rs_cleanup()
    assert again["already_done"] is True
    assert again["removed"] == 0


def test_record_skips_expensive_shares(paper_db):
    scan_time = __import__("datetime").datetime(2025, 3, 3, 10, 0, 0)
    _add_scan(
        paper_db,
        "expensive_scan",
        scan_time,
        [
            {
                "strategy_id": 1,
                "symbol": "RELIANCE",
                "signal_date": date(2025, 2, 28),
                "score": 80.0,
                "trigger_price": 15_000.0,
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
    svc = PaperTradingService(db=paper_db)
    result = svc.record_from_scan("expensive_scan")
    assert result["holdings_added"] == 1
    assert set(svc.get_portfolio()["symbol"]) == {"TCS"}


def test_one_time_max_price_cleanup_removes_expensive_holdings(paper_db):
    scan_time = __import__("datetime").datetime(2025, 4, 1, 10, 0, 0)
    paper_db.insert_portfolio_holdings(
        [
            {
                "holding_id": "cheap",
                "scan_run_id": "s1",
                "symbol": "TCS",
                "source_type": "strategy",
                "source_label": "Basic Breakout",
                "strategy_ids": [1],
                "purchase_date": date(2025, 3, 31),
                "purchase_price": 5000.0,
                "quantity": 1,
                "score": 5.0,
                "created_at": scan_time,
            },
            {
                "holding_id": "dear",
                "scan_run_id": "s1",
                "symbol": "POWERINDIA",
                "source_type": "strategy",
                "source_label": "RS Leaders",
                "strategy_ids": [6],
                "purchase_date": date(2025, 3, 31),
                "purchase_price": 18_000.0,
                "quantity": 1,
                "score": 9.0,
                "created_at": scan_time,
            },
        ]
    )
    svc = PaperTradingService(db=paper_db)
    result = svc.run_one_time_max_price_cleanup()
    assert result["removed"] == 1
    assert paper_db.count_portfolio_holdings() == 1
    assert svc.get_portfolio().iloc[0]["symbol"] == "TCS"

    again = svc.run_one_time_max_price_cleanup()
    assert again["already_done"] is True
    assert again["removed"] == 0


def test_rs_cleanup_keeps_confluence(paper_db):
    scan_time = __import__("datetime").datetime(2025, 4, 1, 10, 0, 0)
    paper_db.insert_portfolio_holdings(
        [
            {
                "holding_id": "conf1",
                "scan_run_id": "s1",
                "symbol": "RELIANCE",
                "source_type": "confluence",
                "source_label": "Confluence",
                "strategy_ids": [1, 6],
                "purchase_date": date(2025, 3, 31),
                "purchase_price": 1000.0,
                "quantity": 1,
                "score": 9.0,
                "created_at": scan_time,
            },
            {
                "holding_id": "rs_low",
                "scan_run_id": "s1",
                "symbol": "LOWRS",
                "source_type": "strategy",
                "source_label": "RS Leaders",
                "strategy_ids": [6],
                "purchase_date": date(2025, 3, 31),
                "purchase_price": 100.0,
                "quantity": 1,
                "score": 1.0,
                "created_at": scan_time,
            },
        ]
    )
    svc = PaperTradingService(db=paper_db)
    # reset flag for test
    paper_db.set_metadata("portfolio_rs_top5_cleanup_done", "")
    result = svc.run_one_time_rs_cleanup()
    assert result["removed"] == 0
    assert paper_db.count_portfolio_holdings() == 2


def test_sync_skips_empty_scan_runs(paper_db):
    empty_time = __import__("datetime").datetime(2025, 3, 2, 9, 0, 0)
    paper_db.insert_scan_run("empty_scan", empty_time, "equity", 0, True, {})
    svc = PaperTradingService(db=paper_db)
    sync = svc.sync_portfolio_from_scans()
    assert sync["empty_scans"] >= 1
    assert paper_db.portfolio_scan_processed("empty_scan")


def test_portfolio_live_price_overrides(paper_db):
    svc = PaperTradingService(db=paper_db)
    svc.record_from_scan("scan1")
    today = date(2025, 6, 28)
    overrides = {"RELIANCE": (today, 1200.0), "TCS": (today, 1100.0)}
    df = svc.get_portfolio(price_overrides=overrides)
    reliance = df[df["symbol"] == "RELIANCE"].iloc[0]
    assert reliance["current_price"] == 1200.0
    assert reliance["price_source"] == "live"
    summary = svc.summarize_portfolio(df)
    assert summary["using_live_prices"] is True
    assert summary["total_pl_amount"] != 0


def test_summarize_by_date_and_strategy(paper_db):
    svc = PaperTradingService(db=paper_db)
    svc.record_from_scan("scan1")
    df = svc.get_portfolio()
    by_date = svc.summarize_by_date(df)
    by_strat = svc.summarize_by_strategy(df)
    assert len(by_date) == 1
    assert len(by_strat) >= 2
    assert "Confluence" in by_strat["strategy"].values
