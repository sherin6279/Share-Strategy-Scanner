"""Streamlit UI for paper trading P/L tracking."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database.duckdb_manager import DuckDBManager
from paper_trading.service import PaperTradingService


def _fmt_ts(ts) -> str:
    if ts is None:
        return ""
    if isinstance(ts, str):
        return ts.replace("T", " ")[:19]
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_date(d) -> str:
    if d is None:
        return ""
    if isinstance(d, str):
        return d[:10]
    return d.strftime("%Y-%m-%d")


def render_paper_trading_page(db: DuckDBManager) -> None:
    st.subheader("Paper Trading")
    st.caption(
        "Each equity scan is saved here at the signal-day close price. "
        "After a few days, refresh market data and review actual P/L."
    )

    service = PaperTradingService(db=db)
    batches = db.list_paper_trade_batches()

    if not batches:
        st.info(
            "No paper trades yet. Run an **Equity** scan — picks are recorded automatically."
        )
        return

    labels = [
        f"{_fmt_ts(b['created_at'])} — {b['position_count']} picks (entry {_fmt_date(b['entry_date'])})"
        for b in batches
    ]
    choice = st.selectbox("Paper trade batch", labels, index=0)
    batch = batches[labels.index(choice)]
    batch_id = batch["batch_id"]

    col1, col2, col3 = st.columns(3)
    with col1:
        min_hold = st.number_input(
            "Min hold days",
            min_value=0,
            max_value=30,
            value=2,
            help="Show only positions held at least this many calendar days",
        )
    with col2:
        st.metric("Entry date", _fmt_date(batch["entry_date"]))
    with col3:
        if batch.get("scan_run_id"):
            st.caption(f"Scan run: `{batch['scan_run_id']}`")

    summary = service.summarize_pl(batch_id, min_hold_days=int(min_hold))
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Positions", summary["position_count"])
    with m2:
        wr = summary["win_rate"]
        st.metric("Win rate", f"{wr:.1f}%" if wr is not None else "—")
    with m3:
        avg = summary["avg_pl_pct"]
        st.metric("Avg P/L %", f"{avg:+.2f}%" if avg is not None else "—")
    with m4:
        st.metric(
            "Price as of",
            _fmt_date(summary.get("latest_price_date")),
        )

    if summary["with_prices"] == 0:
        st.warning(
            "No updated prices found. Click **Refresh Market Data** on the Equity tab first."
        )

    pl_df = service.compute_pl(batch_id, min_hold_days=int(min_hold))
    if pl_df.empty:
        st.info(f"No positions held for at least {min_hold} day(s) yet.")
        return

    display = pl_df.rename(
        columns={
            "symbol": "Symbol",
            "strategy_name": "Strategy",
            "entry_date": "Entry Date",
            "entry_price": "Entry Price",
            "current_date": "Price Date",
            "current_price": "Current Price",
            "days_held": "Days Held",
            "pl_amount": "P/L ₹",
            "pl_pct": "P/L %",
            "score": "Score",
        }
    )
    display["Entry Date"] = display["Entry Date"].apply(_fmt_date)
    display["Price Date"] = display["Price Date"].apply(_fmt_date)

    if "P/L %" in display.columns:
        display = display.sort_values("P/L %", ascending=False, na_position="last")

    st.dataframe(
        display[
            [
                "Symbol",
                "Strategy",
                "Entry Date",
                "Entry Price",
                "Price Date",
                "Current Price",
                "Days Held",
                "P/L ₹",
                "P/L %",
                "Score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Entry Price": st.column_config.NumberColumn(format="₹%.2f"),
            "Current Price": st.column_config.NumberColumn(format="₹%.2f"),
            "P/L ₹": st.column_config.NumberColumn(format="₹%+.2f"),
            "P/L %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Score": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    winners = pl_df[pl_df["pl_pct"].notna() & (pl_df["pl_pct"] > 0)]
    losers = pl_df[pl_df["pl_pct"].notna() & (pl_df["pl_pct"] < 0)]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top gainers**")
        if winners.empty:
            st.caption("None yet")
        else:
            top = winners.nlargest(5, "pl_pct")[["symbol", "pl_pct", "days_held"]]
            st.dataframe(
                top.rename(columns={"symbol": "Symbol", "pl_pct": "P/L %", "days_held": "Days"}),
                hide_index=True,
                use_container_width=True,
            )
    with c2:
        st.markdown("**Top losers**")
        if losers.empty:
            st.caption("None yet")
        else:
            bottom = losers.nsmallest(5, "pl_pct")[["symbol", "pl_pct", "days_held"]]
            st.dataframe(
                bottom.rename(columns={"symbol": "Symbol", "pl_pct": "P/L %", "days_held": "Days"}),
                hide_index=True,
                use_container_width=True,
            )
