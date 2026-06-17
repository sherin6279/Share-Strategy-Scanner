"""
NIFTY 500 Momentum & Leadership Screener
Enterprise-grade local desktop application.
"""

from __future__ import annotations

import json
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from config.settings import KITE_ACCESS_TOKEN, KITE_API_KEY, KITE_API_SECRET, NIFTY50_SYMBOL
from data.kite_fetcher import KiteFetcher
from data.refresh_service import RefreshService
from database.duckdb_manager import DuckDBManager
from exports.export_service import export_csv, export_excel, prepare_display_df
from scanners.scanner import Scanner
from strategies.strategy_engine import STRATEGIES

# Page config
st.set_page_config(
    page_title="NIFTY 500 Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

STRATEGY_TABS = {
    1: "Basic Breakout",
    2: "Refined Breakout",
    3: "Enterprise Breakout",
    4: "Volatility Compression",
    5: "Trend Pullback",
    6: "RS Leaders",
    7: "Stage-1 Base Breakout",
}


def init_session_state() -> None:
    defaults = {
        "access_token": KITE_ACCESS_TOKEN,
        "refresh_summary": None,
        "scan_summary": None,
        "strategy_3_paused": False,
        "market_uptrend": True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_fetcher() -> KiteFetcher:
    return KiteFetcher(
        api_key=KITE_API_KEY,
        api_secret=KITE_API_SECRET,
        access_token=st.session_state.get("access_token", KITE_ACCESS_TOKEN),
    )


def render_kite_login() -> None:
    """Kite Connect OAuth login (no custom web auth)."""
    st.sidebar.header("Kite Connect Login")

    if not KITE_API_KEY:
        st.sidebar.error("Set KITE_API_KEY in your .env file")
        return

    fetcher = get_fetcher()

    if fetcher.is_authenticated():
        if st.sidebar.button("Verify Token"):
            with st.sidebar:
                with st.spinner("Checking token..."):
                    if fetcher.validate_token():
                        st.sidebar.success("Token is valid")
                    else:
                        st.sidebar.error("Token expired — generate a new one below")
                        st.session_state.access_token = ""
                        st.rerun()
        else:
            st.sidebar.success("Connected to Kite")
        if st.sidebar.button("Disconnect"):
            st.session_state.access_token = ""
            st.rerun()
    else:
        st.sidebar.warning("Not authenticated")
        login_url = fetcher.login_url()
        st.sidebar.markdown(f"[Open Kite Login]({login_url})")
        st.sidebar.caption(
            "Log in with your Zerodha credentials on Kite, then paste the request_token below."
        )

        request_token = st.sidebar.text_input("Request Token", type="password")
        if st.sidebar.button("Generate Access Token") and request_token:
            try:
                token = fetcher.generate_session(request_token.strip())
                st.session_state.access_token = token
                st.sidebar.success("Access token generated!")
                st.sidebar.info(
                    "Add this to your .env file:\n\n"
                    f"KITE_ACCESS_TOKEN={token}"
                )
                st.rerun()
            except Exception as exc:
                st.sidebar.error(f"Login failed: {exc}")


def render_header(db: DuckDBManager) -> None:
    col1, col2, col3 = st.columns([2, 1, 1])

    last_refresh = db.get_last_refresh_timestamp()
    last_scan = db.get_metadata("last_scan_timestamp")

    with col1:
        st.title("NIFTY 500 Momentum & Leadership Screener")
        if last_refresh:
            st.caption(f"Last refresh: {last_refresh}")
        else:
            st.caption("No market data loaded yet")
        if last_scan:
            st.caption(f"Last scan: {last_scan}")

    with col2:
        if st.button("Refresh Market Data", type="primary", use_container_width=True):
            _run_refresh(db)

    with col3:
        if st.button("Run Scan", type="secondary", use_container_width=True):
            _run_scan(db)


def _run_refresh(db: DuckDBManager) -> None:
    fetcher = get_fetcher()
    if not fetcher.is_authenticated():
        st.error("Authenticate with Kite first (see sidebar)")
        return

    progress = st.progress(0, text="Starting refresh...")
    status = st.empty()

    def callback(current: int, total: int, symbol: str) -> None:
        pct = current / total
        progress.progress(pct, text=f"Fetching {current} / {total} — {symbol}")
        status.caption(f"Current: {symbol}")

    try:
        service = RefreshService(db=db, fetcher=fetcher)
        start = time.time()
        summary = service.refresh(progress_callback=callback)
        elapsed = time.time() - start

        progress.progress(1.0, text="Refresh complete!")
        st.session_state.refresh_summary = summary

        fetched = summary["symbols_fetched"]
        n_failed = summary["symbols_failed"]
        elapsed_msg = f"{elapsed:.1f}s"

        if fetched == 0:
            st.error(
                f"Refresh failed — 0 symbols fetched ({n_failed} failed) in {elapsed_msg}. "
                "Your access token is likely expired. Use 'Verify Token' or re-login via the sidebar."
            )
        elif n_failed > 10:
            st.warning(
                f"Partial refresh: {fetched} symbols fetched, {n_failed} failed in {elapsed_msg}"
            )
        else:
            st.success(
                f"Fetched {fetched} symbols ({n_failed} failed) in {elapsed_msg}"
            )

        if summary["failed_symbols"]:
            with st.expander(f"Failed symbols ({n_failed})"):
                st.write(summary["failed_symbols"])
    except Exception as exc:
        st.error(f"Refresh failed: {exc}")


def _run_scan(db: DuckDBManager) -> None:
    with st.spinner("Running scan on local data..."):
        try:
            start = time.time()
            scanner = Scanner(db=db)
            summary = scanner.run_scan()
            elapsed = time.time() - start

            st.session_state.scan_summary = summary
            st.session_state.strategy_3_paused = summary.get("strategy_3_paused", False)
            st.session_state.market_uptrend = summary.get("market_uptrend", True)

            st.success(
                f"Scan complete: {summary['signal_count']} signals in {elapsed:.1f}s"
            )
            if summary.get("strategy_3_paused"):
                st.warning("Market in downtrend — Strategy 3 paused")
            if not summary.get("market_uptrend", True):
                st.warning(
                    "NIFTY 50 is below its 200-day SMA — Strategies 4–7 still run "
                    "but signals are tagged with a caution note"
                )
        except Exception as exc:
            st.error(f"Scan failed: {exc}")


def render_strategy_tab(scanner: Scanner, strategy_id: int, tab_name: str) -> None:
    if strategy_id == 3 and st.session_state.get("strategy_3_paused"):
        st.warning("Market in downtrend — Strategy 3 paused")

    if strategy_id in (4, 5, 6, 7) and not st.session_state.get("market_uptrend", True):
        st.warning(
            "NIFTY 50 is below its 200-day SMA — review these picks with extra caution"
        )

    df = scanner.get_results_by_strategy(strategy_id)
    if df.empty:
        st.info("No results. Run a scan after refreshing market data.")
        return

    _render_results_table(df, tab_name)


def render_confluence_tab(scanner: Scanner) -> None:
    df = scanner.get_confluence(min_strategies=2)
    if df.empty:
        st.info("No confluence signals (stocks in 2+ strategies).")
        return

    display = df.rename(
        columns={
            "symbol": "Symbol",
            "current_price": "Current Price",
            "strategy_count": "Strategy Count",
            "matching_strategies": "Matching Strategies",
            "highest_score": "Highest Score",
            "average_score": "Average Score",
        }
    )
    _render_dataframe_with_controls(display, "confluence")


def _render_results_table(df: pd.DataFrame, export_name: str) -> None:
    display = prepare_display_df(df)
    _render_dataframe_with_controls(display, export_name, export_source=df)


def _render_dataframe_with_controls(
    df: pd.DataFrame,
    export_name: str,
    export_source: pd.DataFrame | None = None,
) -> None:
    search = st.text_input("Search", key=f"search_{export_name}", placeholder="Filter symbols...")
    filtered = df.copy()
    if search:
        mask = filtered.astype(str).apply(
            lambda row: row.str.contains(search, case=False, na=False).any(), axis=1
        )
        filtered = filtered[mask]

    st.dataframe(
        filtered,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.NumberColumn(format="%.2f"),
            "Current Price": st.column_config.NumberColumn(format="₹%.2f"),
            "Highest Score": st.column_config.NumberColumn(format="%.2f"),
            "Average Score": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    export_df = export_source if export_source is not None else df
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Export CSV",
            data=export_csv(export_df),
            file_name=f"{export_name.lower().replace(' ', '_')}_{datetime.now():%Y%m%d}.csv",
            mime="text/csv",
            key=f"csv_{export_name}",
        )
    with col2:
        st.download_button(
            "Export Excel",
            data=export_excel(export_df),
            file_name=f"{export_name.lower().replace(' ', '_')}_{datetime.now():%Y%m%d}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"xlsx_{export_name}",
        )


def main() -> None:
    init_session_state()
    render_kite_login()

    db = DuckDBManager()
    render_header(db)

    scanner = Scanner(db=db)

    tabs = st.tabs(
        [STRATEGY_TABS[i] for i in range(1, 8)] + ["Confluence"]
    )

    for i, tab in enumerate(tabs[:7], start=1):
        with tab:
            render_strategy_tab(scanner, i, STRATEGY_TABS[i])

    with tabs[7]:
        render_confluence_tab(scanner)

    # Footer stats
    with st.sidebar.expander("Database Stats"):
        symbols = db.get_all_symbols()
        st.write(f"Symbols in DB: {len(symbols)}")
        st.write(f"NIFTY 50 loaded: {NIFTY50_SYMBOL in symbols}")
        if st.session_state.refresh_summary:
            st.json(st.session_state.refresh_summary)


if __name__ == "__main__":
    main()
