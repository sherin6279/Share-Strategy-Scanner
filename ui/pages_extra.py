"""Streamlit UI sections for F&O and Backtest."""

from __future__ import annotations

import uuid
from datetime import date, datetime

import pandas as pd
import streamlit as st

from backtest.equity_backtest import EquityBacktester
from backtest.fno_backtest import FnoBacktester
from backtest.metrics import summaries_to_dataframe
from data.fno_refresh_service import FnoRefreshService
from data.intraday_fetcher import IntradayFetcher
from database.duckdb_manager import DuckDBManager
from exports.export_service import prepare_display_df
from scanners.fno_scanner import FnoScanner


FNO_TABS = {101: "Opening Range", 102: "VWAP Trend", 103: "Session Breakout"}


def _fmt_ts(ts) -> str:
    if ts is None:
        return ""
    if isinstance(ts, str):
        return ts.replace("T", " ")[:19]
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def render_fno_page(db: DuckDBManager, get_fetcher) -> None:
    from config.settings import FNO_INDEX_UNDERLYINGS, FNO_STOCK_COUNT

    st.subheader("F&O Intraday Screener")
    st.caption(
        f"5-minute candles — {', '.join(FNO_INDEX_UNDERLYINGS)} index futures plus "
        f"top {FNO_STOCK_COUNT} liquid NIFTY 500 stock futures (by equity volume)."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh F&O Data", type="primary", use_container_width=True):
            _run_fno_refresh(db, get_fetcher)
    with col2:
        if st.button("Run F&O Scan", use_container_width=True):
            _run_fno_scan(db)

    symbols = db.get_intraday_symbols()
    index_n = db.get_metadata("last_fno_refresh_index_count", "—")
    stock_n = db.get_metadata("last_fno_refresh_stock_count", "—")
    st.caption(
        f"Intraday symbols loaded: **{len(symbols)}** "
        f"(last refresh: {index_n} index, {stock_n} stock)"
    )
    if symbols:
        preview = ", ".join(symbols[:8])
        suffix = f" … +{len(symbols) - 8} more" if len(symbols) > 8 else ""
        st.caption(f"{preview}{suffix}")

    runs = db.list_fno_scan_runs()
    scan_run_id = None
    scan_ts = None
    if runs:
        labels = [
            f"{_fmt_ts(r['scan_timestamp'])} ({r['signal_count']} signals)"
            for r in runs
        ]
        default_idx = 0
        selected_run_id = st.session_state.get("selected_fno_scan_run_id")
        if selected_run_id:
            for i, r in enumerate(runs):
                if r["run_id"] == selected_run_id:
                    default_idx = i
                    break
        elif st.session_state.get("selected_fno_scan_ts") is not None:
            legacy_ts = st.session_state.selected_fno_scan_ts
            for i, r in enumerate(runs):
                if str(r["scan_timestamp"]) == str(legacy_ts):
                    default_idx = i
                    break
        choice = st.selectbox("View F&O scan from", labels, index=default_idx)
        chosen = runs[labels.index(choice)]
        scan_run_id = chosen["run_id"]
        scan_ts = chosen["scan_timestamp"]
        st.session_state.selected_fno_scan_run_id = scan_run_id
        st.session_state.selected_fno_scan_ts = scan_ts
        st.caption(f"Showing results from scan at **{_fmt_ts(scan_ts)}**")

    results = db.get_fno_scan_results(
        scan_run_id=scan_run_id, scan_timestamp=scan_ts
    )
    if results.empty:
        st.info("No F&O scan results. Refresh data and run scan.")
        return

    tabs = st.tabs([FNO_TABS[sid] for sid in sorted(FNO_TABS)])
    for i, sid in enumerate(sorted(FNO_TABS)):
        with tabs[i]:
            sub = results[results["strategy_id"] == sid].sort_values("score", ascending=False)
            if sub.empty:
                st.info(f"No signals for {FNO_TABS[sid]}")
            else:
                display = prepare_display_df(sub.rename(columns={"signal_datetime": "signal_date"}))
                st.dataframe(display, use_container_width=True, hide_index=True)


def _run_fno_refresh(db: DuckDBManager, get_fetcher) -> None:
    fetcher = get_fetcher()
    if not isinstance(fetcher, IntradayFetcher):
        fetcher = IntradayFetcher(
            api_key=fetcher.api_key,
            access_token=fetcher.access_token,
            api_secret=fetcher.api_secret,
        )
    progress = st.progress(0, text="Resolving F&O universe...")
    status = st.empty()
    try:

        def callback(current: int, total: int, label: str) -> None:
            pct = current / total if total else 1.0
            progress.progress(pct, text=f"Fetching {current} / {total}")
            status.caption(label)

        with st.spinner("Fetching F&O intraday data..."):
            summary = FnoRefreshService(db=db, fetcher=fetcher).refresh(
                progress_callback=callback
            )
        progress.progress(1.0, text="F&O refresh complete")
        st.success(
            f"Fetched {summary['symbols_fetched']} symbols "
            f"({summary['index_fetched']} index, {summary['stock_fetched']} stock), "
            f"{summary['rows_upserted']:,} bars"
        )
        if summary["failed_symbols"]:
            with st.expander(f"Failed ({summary['symbols_failed']})"):
                st.write(summary["failed_symbols"])
    except Exception as exc:
        st.error(str(exc))


def _run_fno_scan(db: DuckDBManager) -> None:
    try:
        with st.spinner("Running F&O scan..."):
            summary = FnoScanner(db=db).run_scan()
        st.session_state.selected_fno_scan_run_id = summary.get("scan_run_id")
        st.session_state.selected_fno_scan_ts = datetime.fromisoformat(
            summary["scan_timestamp"]
        )
        st.success(
            f"F&O scan saved at {summary['scan_timestamp']}: "
            f"{summary['signal_count']} signals"
        )
        st.rerun()
    except Exception as exc:
        st.error(str(exc))


def render_backtest_page(db: DuckDBManager) -> None:
    st.subheader("Backtest")
    segment = st.radio("Segment", ["Equity", "F&O"], horizontal=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        hold_days = st.number_input("Hold days (equity)", 1, 20, 5)
        step_days = st.number_input("Step days (equity)", 1, 20, 5)
    with col2:
        cost_bps = st.number_input("Cost (bps)", 0.0, 100.0, 15.0 if segment == "Equity" else 5.0)
        simulate = st.checkbox("Simulate stop/target (equity)", value=False)
    with col3:
        stop_pct = st.number_input("Stop %", 0.1, 20.0, 5.0 if segment == "Equity" else 0.5)
        target_pct = st.number_input("Target %", 0.1, 50.0, 10.0 if segment == "Equity" else 1.0)

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                if segment == "Equity":
                    result = EquityBacktester(db=db).run(
                        hold_days=int(hold_days),
                        step_days=int(step_days),
                        cost_bps=float(cost_bps),
                        simulate=simulate,
                        stop_pct=float(stop_pct),
                        target_pct=float(target_pct),
                    )
                else:
                    result = FnoBacktester(db=db).run(
                        cost_bps=float(cost_bps),
                        stop_pct=float(stop_pct),
                        target_pct=float(target_pct),
                    )

                run_id = str(uuid.uuid4())[:8]
                summary_records = [
                    {
                        "strategy_id": s.strategy_id,
                        "signals": s.signal_count,
                        "win_rate": s.win_rate,
                        "avg_return": s.avg_return,
                    }
                    for s in result.summaries
                ]
                db.save_backtest_run(
                    run_id,
                    result.segment,
                    result.start_date,
                    result.end_date,
                    {"mode": result.mode, "cost_bps": result.cost_bps},
                    summary_records,
                )

                st.success(
                    f"{result.segment.upper()} backtest ({result.mode}): "
                    f"{len(result.trades)} trades"
                )
                st.caption(
                    "Signal-quality study — gross returns before costs unless cost bps applied."
                )

                summary_df = summaries_to_dataframe(result.summaries)
                if not summary_df.empty:
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Export Summary CSV",
                        data=summary_df.to_csv(index=False).encode(),
                        file_name=f"backtest_{result.segment}_{datetime.now():%Y%m%d}.csv",
                        mime="text/csv",
                    )

                if result.trades:
                    trades_df = pd.DataFrame(
                        [
                            {
                                "Date": t.signal_date,
                                "Symbol": t.symbol,
                                "Strategy": t.strategy_id,
                                "Return %": t.net_return_pct,
                                "Alpha %": t.alpha_pct,
                                "Exit": t.exit_reason,
                            }
                            for t in result.trades[:500]
                        ]
                    )
                    with st.expander(f"Trade log (first {min(500, len(result.trades))})"):
                        st.dataframe(trades_df, use_container_width=True, hide_index=True)

            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
