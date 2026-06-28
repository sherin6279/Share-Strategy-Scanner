"""Build the F&O intraday fetch universe (indices + liquid NIFTY 500 stock futures)."""

from __future__ import annotations

from data.intraday_fetcher import IntradayFetcher
from data.nifty500_loader import load_nifty500_symbols
from database.duckdb_manager import DuckDBManager
from utils.logger import get_logger

logger = get_logger(__name__)


def build_fno_refresh_targets(
    fetcher: IntradayFetcher,
    db: DuckDBManager,
    index_underlyings: list[str],
    stock_count: int,
    volume_lookback_days: int = 20,
) -> list[dict[str, str]]:
    """
    Resolve tradingsymbols to fetch: index futures first, then top stock FUTs.

    Returns list of dicts: underlying, tradingsymbol, segment ('index'|'stock').
    """
    fetcher.load_nfo_instruments()
    stock_fno_names = fetcher.list_live_fut_underlyings()

    nifty500 = set(load_nifty500_symbols())
    eligible = sorted(stock_fno_names & nifty500)
    if not eligible:
        logger.warning("No NIFTY 500 symbols with live NFO futures found")

    volumes = db.get_avg_equity_volumes(eligible, lookback_days=volume_lookback_days)
    ranked = sorted(
        eligible,
        key=lambda sym: volumes.get(sym, 0.0),
        reverse=True,
    )
    stock_picks = ranked[: max(0, stock_count)]

    targets: list[dict[str, str]] = []
    seen_ts: set[str] = set()

    for underlying in index_underlyings:
        ts = fetcher.nearest_future_symbol(underlying)
        if ts and ts not in seen_ts:
            targets.append(
                {"underlying": underlying, "tradingsymbol": ts, "segment": "index"}
            )
            seen_ts.add(ts)

    for underlying in stock_picks:
        ts = fetcher.nearest_future_symbol(underlying)
        if ts and ts not in seen_ts:
            targets.append(
                {"underlying": underlying, "tradingsymbol": ts, "segment": "stock"}
            )
            seen_ts.add(ts)
        elif ts is None:
            logger.debug("No live FUT for stock underlying %s", underlying)

    logger.info(
        "F&O universe: %d index + %d stock futures (%d eligible in NIFTY 500)",
        sum(1 for t in targets if t["segment"] == "index"),
        sum(1 for t in targets if t["segment"] == "stock"),
        len(eligible),
    )
    return targets
