"""Calendar-based forward return calculations."""

from __future__ import annotations

from datetime import date

from backtest.date_index import DateIndex


def forward_return_pct(
    date_index: DateIndex,
    symbol: str,
    signal_date: date,
    hold_days: int = 5,
) -> tuple[float | None, date | None, float | None, float | None]:
    """
    Compute gross forward return using the NIFTY trading calendar.

    Returns (return_pct, exit_date, entry_price, exit_price).
    """
    entry = date_index.close_on(symbol, signal_date)
    exit_date = date_index.forward_date(signal_date, hold_days)
    if entry is None or exit_date is None:
        return None, exit_date, entry, None

    exit_price = date_index.close_on(symbol, exit_date)
    if exit_price is None or entry <= 0:
        return None, exit_date, entry, exit_price

    ret = (exit_price / entry - 1.0) * 100.0
    return ret, exit_date, entry, exit_price


def alpha_vs_benchmark(
    date_index: DateIndex,
    symbol: str,
    benchmark: str,
    signal_date: date,
    hold_days: int = 5,
) -> float | None:
    stock_ret, _, _, _ = forward_return_pct(date_index, symbol, signal_date, hold_days)
    bench_ret, _, _, _ = forward_return_pct(date_index, benchmark, signal_date, hold_days)
    if stock_ret is None or bench_ret is None:
        return None
    return stock_ret - bench_ret
