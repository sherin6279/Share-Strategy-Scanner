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
from ui.pages_extra import render_backtest_page, render_fno_page

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


@st.cache_resource
def get_db() -> DuckDBManager:
    """Single write connection for the Streamlit process."""
    return DuckDBManager()


def init_session_state() -> None:
    defaults = {
        "access_token": KITE_ACCESS_TOKEN,
        "refresh_summary": None,
        "scan_summary": None,
        "strategy_3_paused": False,
        "market_uptrend": True,
        "selected_equity_scan_ts": None,
        "selected_fno_scan_ts": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Hydrate scan state from DB so tabs show results after page refresh
    if st.session_state.get("scan_summary") is None:
        try:
            db = get_db()
        except RuntimeError as exc:
            st.warning(str(exc))
            return
        last_scan = db.get_metadata("last_scan_timestamp")
        if last_scan:
            uptrend_raw = db.get_metadata("last_scan_market_uptrend", "true")
            st.session_state.market_uptrend = uptrend_raw == "true"
            st.session_state.strategy_3_paused = not st.session_state.market_uptrend
            counts_raw = db.get_metadata("last_scan_strategy_counts", "{}")
            try:
                strategy_counts = {
                    int(k): v for k, v in json.loads(counts_raw).items()
                }
            except (json.JSONDecodeError, ValueError):
                strategy_counts = {}
            st.session_state.scan_summary = {
                "scan_timestamp": last_scan,
                "signal_count": sum(strategy_counts.values()) if strategy_counts else 0,
                "strategy_counts": strategy_counts,
                "market_uptrend": st.session_state.market_uptrend,
                "strategy_3_paused": st.session_state.strategy_3_paused,
            }


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


def _format_scan_timestamp(ts) -> str:
    if ts is None:
        return ""
    if isinstance(ts, str):
        return ts.replace("T", " ")[:19]
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _format_run_label(run: dict) -> str:
    ts = _format_scan_timestamp(run["scan_timestamp"])
    counts = run.get("strategy_counts") or {}
    if counts:
        breakdown = ", ".join(f"S{k}:{v}" for k, v in sorted(counts.items()) if v > 0)
        return f"{ts} ({run['signal_count']} signals — {breakdown})"
    return f"{ts} ({run['signal_count']} signals)"


def render_equity_scan_selector(db: DuckDBManager) -> dict | None:
    """Return the selected equity scan run metadata."""
    runs = db.list_scan_runs("equity")
    if not runs:
        return None

    labels = [_format_run_label(r) for r in runs]
    selected_run_id = st.session_state.get("selected_equity_scan_run_id")
    default_idx = 0
    if selected_run_id:
        for i, r in enumerate(runs):
            if r["run_id"] == selected_run_id:
                default_idx = i
                break
    elif st.session_state.get("selected_equity_scan_ts") is not None:
        legacy_ts = st.session_state.selected_equity_scan_ts
        for i, r in enumerate(runs):
            if str(r["scan_timestamp"]) == str(legacy_ts):
                default_idx = i
                break

    choice = st.selectbox("View scan from", labels, index=default_idx, key="equity_scan_picker")
    chosen = runs[labels.index(choice)]
    st.session_state.selected_equity_scan_run_id = chosen["run_id"]
    st.session_state.selected_equity_scan_ts = chosen["scan_timestamp"]
    return chosen


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
            st.session_state.selected_equity_scan_run_id = summary.get("scan_run_id")
            st.session_state.selected_equity_scan_ts = datetime.fromisoformat(
                summary["scan_timestamp"]
            )

            counts = summary.get("strategy_counts", {})
            breakdown = ", ".join(
                f"S{k}: {v}" for k, v in sorted(counts.items()) if v > 0
            )
            st.success(
                f"Scan complete: {summary['signal_count']} signals in {elapsed:.1f}s"
                + (f" ({breakdown})" if breakdown else "")
                + f" — saved at {summary['scan_timestamp']}"
            )
            if summary.get("strategy_3_paused"):
                st.warning("Market in downtrend — Strategy 3 paused")
            if not summary.get("market_uptrend", True):
                st.warning(
                    "NIFTY 50 is below its 200-day SMA — Strategies 4–7 still run "
                    "but signals are tagged with a caution note"
                )
            st.rerun()
        except Exception as exc:
            st.error(f"Scan failed: {exc}")


def render_strategy_tab(
    scanner: Scanner,
    strategy_id: int,
    tab_name: str,
    scan_run_id: str | None = None,
    scan_timestamp: datetime | None = None,
    market_uptrend: bool = True,
) -> None:
    if strategy_id == 3 and not market_uptrend:
        st.warning("Market in downtrend — Strategy 3 paused")

    if strategy_id in (4, 5, 6, 7) and not market_uptrend:
        st.warning(
            "NIFTY 50 is below its 200-day SMA — review these picks with extra caution"
        )

    df = scanner.get_results_by_strategy(
        strategy_id, scan_run_id=scan_run_id, scan_timestamp=scan_timestamp
    )
    if df.empty:
        all_results = scanner.db.get_scan_results(
            scan_run_id=scan_run_id, scan_timestamp=scan_timestamp
        )
        if all_results.empty:
            st.info("No results yet. Click **Run Scan** after refreshing market data.")
        else:
            counts = all_results.groupby("strategy_id").size().to_dict()
            active = ", ".join(
                f"S{k}:{v}" for k, v in sorted(counts.items()) if v > 0
            )
            st.info(
                f"No matches for **{tab_name}** in this scan."
                + (f" Other strategies did find signals ({active})." if active else "")
            )
        return

    _render_results_table(df, tab_name, scan_timestamp=scan_timestamp)


def render_confluence_tab(
    scanner: Scanner,
    scan_run_id: str | None = None,
    scan_timestamp: datetime | None = None,
) -> None:
    df = scanner.get_confluence(
        min_strategies=2, scan_run_id=scan_run_id, scan_timestamp=scan_timestamp
    )
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
    _render_dataframe_with_controls(display, "confluence", scan_timestamp=scan_timestamp)


def _render_results_table(
    df: pd.DataFrame,
    export_name: str,
    scan_timestamp: datetime | None = None,
) -> None:
    display = prepare_display_df(df)
    _render_dataframe_with_controls(
        display, export_name, export_source=df, scan_timestamp=scan_timestamp
    )


def _render_dataframe_with_controls(
    df: pd.DataFrame,
    export_name: str,
    export_source: pd.DataFrame | None = None,
    scan_timestamp: datetime | None = None,
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
    ts_suffix = ""
    if scan_timestamp is not None:
        ts_suffix = f"_{scan_timestamp:%Y%m%d_%H%M%S}"
    elif "Scan Time" in df.columns and not df.empty:
        ts_suffix = f"_{pd.Timestamp(df['Scan Time'].iloc[0]):%Y%m%d_%H%M%S}"

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Export CSV",
            data=export_csv(export_df),
            file_name=f"{export_name.lower().replace(' ', '_')}{ts_suffix}.csv",
            mime="text/csv",
            key=f"csv_{export_name}",
        )
    with col2:
        st.download_button(
            "Export Excel",
            data=export_excel(export_df),
            file_name=f"{export_name.lower().replace(' ', '_')}{ts_suffix}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"xlsx_{export_name}",
        )


def main() -> None:
    init_session_state()
    render_kite_login()

    try:
        db = get_db()
        db._initialize_schema()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    mode = st.sidebar.radio(
        "Segment",
        ["Equity", "F&O Intraday", "Backtest"],
        index=0,
    )

    if mode == "Equity":
        render_header(db)
        scanner = Scanner(db=db)
        selected_run = render_equity_scan_selector(db)
        scan_run_id = None
        scan_ts = None
        market_uptrend = st.session_state.get("market_uptrend", True)
        if selected_run:
            scan_run_id = selected_run["run_id"]
            scan_ts = selected_run["scan_timestamp"]
            market_uptrend = selected_run.get("market_uptrend", market_uptrend)
            st.caption(
                f"Showing results from scan at **{_format_scan_timestamp(scan_ts)}**"
            )
            if not market_uptrend:
                st.caption(
                    "NIFTY 50 is below its 200-day SMA — Strategy 3 is paused; "
                    "Strategies 4–7 may show fewer picks than in an uptrend."
                )
        tabs = st.tabs(
            [STRATEGY_TABS[i] for i in range(1, 8)] + ["Confluence"]
        )
        for i, tab in enumerate(tabs[:7], start=1):
            with tab:
                render_strategy_tab(
                    scanner,
                    i,
                    STRATEGY_TABS[i],
                    scan_run_id=scan_run_id,
                    scan_timestamp=scan_ts,
                    market_uptrend=market_uptrend,
                )
        with tabs[7]:
            render_confluence_tab(
                scanner, scan_run_id=scan_run_id, scan_timestamp=scan_ts
            )
    elif mode == "F&O Intraday":
        render_fno_page(db, get_fetcher)
    else:
        render_backtest_page(db)

    # Footer stats
    with st.sidebar.expander("Database Stats"):
        symbols = db.get_all_symbols()
        st.write(f"Equity symbols: {len(symbols)}")
        st.write(f"NIFTY 50 loaded: {NIFTY50_SYMBOL in symbols}")
        fno_syms = db.get_intraday_symbols()
        st.write(f"F&O intraday symbols: {len(fno_syms)}")
        if st.session_state.refresh_summary:
            st.json(st.session_state.refresh_summary)


if __name__ == "__main__":
    main()
