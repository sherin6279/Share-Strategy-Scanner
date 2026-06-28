"""Portfolio UI — holdings and sidebar summary."""

from __future__ import annotations

import streamlit as st

from database.duckdb_manager import DuckDBManager
from paper_trading.service import PaperTradingService


def _fmt_date(d) -> str:
    if d is None:
        return "—"
    if isinstance(d, str):
        return d[:10]
    return d.strftime("%Y-%m-%d")


def render_portfolio_sidebar(db: DuckDBManager) -> None:
    """Compact portfolio summary always visible in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Portfolio")

    service = PaperTradingService(db=db)
    summary = service.summarize_portfolio()

    if summary["holding_count"] == 0:
        st.sidebar.caption("No holdings yet. Run a scan to add picks.")
        return

    st.sidebar.metric("Holdings", summary["holding_count"])
    pl = summary["total_pl_amount"]
    pl_pct = summary["total_pl_pct"]
    st.sidebar.metric(
        "Total P/L",
        f"₹{pl:+,.2f}" if pl is not None else "—",
        delta=f"{pl_pct:+.2f}%" if pl_pct is not None else None,
    )
    st.sidebar.caption(f"Prices as of {_fmt_date(summary['latest_price_date'])}")

    df = service.get_portfolio()
    if not df.empty and df["pl_pct"].notna().any():
        top = df.nlargest(3, "pl_pct")[["symbol", "pl_pct"]]
        st.sidebar.caption("Top movers")
        for _, row in top.iterrows():
            st.sidebar.write(f"{row['symbol']}: {row['pl_pct']:+.1f}%")


def render_paper_trading_page(db: DuckDBManager) -> None:
    st.subheader("Portfolio")
    st.caption(
        "Each scan adds 1 share per pick at that day's close. "
        "Confluence picks (2+ strategies) appear as a single holding. "
        "Refresh market data to update P/L."
    )

    service = PaperTradingService(db=db)
    summary = service.summarize_portfolio()

    if summary["holding_count"] == 0:
        st.info("No holdings yet. Run an **Equity** scan — picks are added automatically.")
        return

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Holdings", summary["holding_count"])
    with m2:
        st.metric("Invested", f"₹{summary['total_cost']:,.2f}")
    with m3:
        st.metric("Market value", f"₹{summary['total_market_value']:,.2f}")
    with m4:
        pl = summary["total_pl_amount"]
        st.metric("Total P/L", f"₹{pl:+,.2f}" if pl is not None else "—")
    with m5:
        wr = summary["win_rate"]
        st.metric("Win rate", f"{wr:.0f}%" if wr is not None else "—")

    st.caption(f"Prices as of **{_fmt_date(summary['latest_price_date'])}**")

    if summary["holding_count"] > 0 and summary["total_market_value"] == 0:
        st.warning("Refresh market data on the Equity tab to load current prices.")

    df = service.get_portfolio()
    display = df.rename(
        columns={
            "symbol": "Symbol",
            "source": "Bought From",
            "purchase_date": "Bought On",
            "purchase_price": "Buy Price",
            "quantity": "Qty",
            "current_price": "Current Price",
            "market_value": "Market Value",
            "pl_amount": "P/L ₹",
            "pl_pct": "P/L %",
            "days_held": "Days Held",
            "score": "Score",
        }
    )
    display["Bought On"] = display["Bought On"].apply(_fmt_date)
    display = display.sort_values(["Bought On", "Symbol"], ascending=[False, True])

    st.dataframe(
        display[
            [
                "Symbol",
                "Bought From",
                "Bought On",
                "Buy Price",
                "Qty",
                "Current Price",
                "Market Value",
                "P/L ₹",
                "P/L %",
                "Days Held",
                "Score",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Buy Price": st.column_config.NumberColumn(format="₹%.2f"),
            "Current Price": st.column_config.NumberColumn(format="₹%.2f"),
            "Market Value": st.column_config.NumberColumn(format="₹%.2f"),
            "P/L ₹": st.column_config.NumberColumn(format="₹%+.2f"),
            "P/L %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Score": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    winners = df[df["pl_amount"].notna() & (df["pl_amount"] > 0)]
    losers = df[df["pl_amount"].notna() & (df["pl_amount"] < 0)]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Best performers**")
        if winners.empty:
            st.caption("—")
        else:
            st.dataframe(
                winners.nlargest(5, "pl_pct")[
                    ["symbol", "source", "pl_pct", "days_held"]
                ].rename(
                    columns={
                        "symbol": "Symbol",
                        "source": "Bought From",
                        "pl_pct": "P/L %",
                        "days_held": "Days",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
    with c2:
        st.markdown("**Worst performers**")
        if losers.empty:
            st.caption("—")
        else:
            st.dataframe(
                losers.nsmallest(5, "pl_pct")[
                    ["symbol", "source", "pl_pct", "days_held"]
                ].rename(
                    columns={
                        "symbol": "Symbol",
                        "source": "Bought From",
                        "pl_pct": "P/L %",
                        "days_held": "Days",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
