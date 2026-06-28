"""Portfolio UI — date/strategy grouped holdings with charts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database.duckdb_manager import DuckDBManager
from paper_trading.service import PaperTradingService

CHART_COLORS = px.colors.qualitative.Set2
PL_GREEN = "#22c55e"
PL_RED = "#ef4444"


def _fmt_date(d) -> str:
    if d is None:
        return "—"
    if isinstance(d, str):
        return d[:10]
    return d.strftime("%Y-%m-%d")


def _fmt_date_long(d) -> str:
    if d is None:
        return "—"
    ts = pd.Timestamp(d)
    return ts.strftime("%A, %d %b %Y")


def _metric_pl(label: str, amount: float | None, pct: float | None = None) -> None:
    if amount is None:
        st.metric(label, "—")
        return
    delta = f"{pct:+.2f}%" if pct is not None else None
    st.metric(label, f"₹{amount:+,.2f}", delta=delta)


def _holdings_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.caption("No holdings")
        return
    display = df[
        [
            "symbol",
            "purchase_price",
            "current_price",
            "market_value",
            "pl_amount",
            "pl_pct",
            "days_held",
            "score",
        ]
    ].rename(
        columns={
            "symbol": "Symbol",
            "purchase_price": "Buy",
            "current_price": "Now",
            "market_value": "Value",
            "pl_amount": "P/L ₹",
            "pl_pct": "P/L %",
            "days_held": "Days",
            "score": "Score",
        }
    )
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Buy": st.column_config.NumberColumn(format="₹%.2f"),
            "Now": st.column_config.NumberColumn(format="₹%.2f"),
            "Value": st.column_config.NumberColumn(format="₹%.2f"),
            "P/L ₹": st.column_config.NumberColumn(format="₹%+.2f"),
            "P/L %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Score": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def _pie_by_date(date_df: pd.DataFrame) -> go.Figure:
    labels = [_fmt_date(d) for d in date_df["purchase_date"]]
    values = date_df["market_value"].fillna(date_df["invested"])
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(colors=CHART_COLORS),
                textinfo="label+percent",
                hovertemplate="%{label}<br>₹%{value:,.0f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Allocation by purchase date",
        margin=dict(t=40, b=20, l=20, r=20),
        height=340,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
    )
    return fig


def _pie_by_strategy(strat_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Pie(
                labels=strat_df["strategy"],
                values=strat_df["holdings"],
                hole=0.45,
                marker=dict(colors=CHART_COLORS),
                textinfo="label+value",
                hovertemplate="%{label}<br>%{value} holdings<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Holdings count by strategy",
        margin=dict(t=40, b=20, l=20, r=20),
        height=340,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    return fig


def _bar_strategy_pl(strat_df: pd.DataFrame) -> go.Figure:
    sorted_df = strat_df.sort_values("pl_amount", ascending=True)
    colors = [PL_GREEN if v >= 0 else PL_RED for v in sorted_df["pl_amount"]]
    fig = go.Figure(
        go.Bar(
            x=sorted_df["pl_amount"],
            y=sorted_df["strategy"],
            orientation="h",
            marker_color=colors,
            text=[f"₹{v:+,.0f}" for v in sorted_df["pl_amount"]],
            textposition="outside",
            hovertemplate="%{y}<br>P/L: ₹%{x:+,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="P/L by strategy (until today)",
        xaxis_title="P/L ₹",
        margin=dict(t=40, b=40, l=20, r=60),
        height=max(280, 44 * len(sorted_df)),
    )
    fig.add_vline(x=0, line_width=1, line_color="#94a3b8")
    return fig


def _bar_date_pl(date_df: pd.DataFrame) -> go.Figure:
    sorted_df = date_df.sort_values("purchase_date")
    colors = [PL_GREEN if v >= 0 else PL_RED for v in sorted_df["pl_amount"]]
    labels = [_fmt_date(d) for d in sorted_df["purchase_date"]]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=sorted_df["pl_amount"],
            marker_color=colors,
            text=[f"₹{v:+,.0f}" for v in sorted_df["pl_amount"]],
            textposition="outside",
            hovertemplate="%{x}<br>P/L: ₹%{y:+,.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="P/L by purchase date",
        yaxis_title="P/L ₹",
        margin=dict(t=40, b=60, l=40, r=20),
        height=320,
    )
    fig.add_hline(y=0, line_width=1, line_color="#94a3b8")
    return fig


def render_portfolio_sidebar(db: DuckDBManager) -> None:
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

    by_date = service.summarize_by_date()
    if not by_date.empty:
        st.sidebar.caption("By scan date")
        for _, row in by_date.head(4).iterrows():
            st.sidebar.write(
                f"**{_fmt_date(row['purchase_date'])}** · "
                f"{int(row['holdings'])} stk · "
                f"₹{row['pl_amount']:+,.0f}"
            )


def _render_overview_charts(
    service: PaperTradingService,
    df: pd.DataFrame,
) -> None:
    date_sum = service.summarize_by_date(df)
    strat_sum = service.summarize_by_strategy(df)

    st.markdown("#### Performance overview")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_pie_by_date(date_sum), use_container_width=True)
    with c2:
        st.plotly_chart(_pie_by_strategy(strat_sum), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(_bar_strategy_pl(strat_sum), use_container_width=True)
    with c4:
        st.plotly_chart(_bar_date_pl(date_sum), use_container_width=True)

    st.markdown("#### Strategy summary")
    strat_display = strat_sum.rename(
        columns={
            "strategy": "Strategy",
            "holdings": "Holdings",
            "invested": "Invested",
            "market_value": "Market value",
            "pl_amount": "P/L ₹",
            "pl_pct": "P/L %",
            "win_rate": "Win %",
        }
    )
    st.dataframe(
        strat_display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Invested": st.column_config.NumberColumn(format="₹%.2f"),
            "Market value": st.column_config.NumberColumn(format="₹%.2f"),
            "P/L ₹": st.column_config.NumberColumn(format="₹%+.2f"),
            "P/L %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Win %": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )


def _render_holdings_by_date(df: pd.DataFrame, date_filter: str | None) -> None:
    st.markdown("#### Holdings by date & strategy")

    dates = sorted(df["purchase_date"].unique(), reverse=True)
    if date_filter and date_filter != "All dates":
        dates = [d for d in dates if _fmt_date(d) == date_filter]

    for purchase_date in dates:
        day_df = df[df["purchase_date"] == purchase_date]
        day_cost = float(day_df["cost_basis"].sum())
        priced = day_df[day_df["pl_amount"].notna()]
        day_pl = float(priced["pl_amount"].sum()) if len(priced) else 0.0
        day_pl_pct = (day_pl / day_cost * 100.0) if day_cost else None
        pl_icon = "🟢" if day_pl >= 0 else "🔴"

        header = (
            f"{pl_icon} **{_fmt_date_long(purchase_date)}** · "
            f"{len(day_df)} holdings · "
            f"Invested ₹{day_cost:,.0f} · "
            f"P/L **₹{day_pl:+,.2f}**"
            + (f" ({day_pl_pct:+.2f}%)" if day_pl_pct is not None else "")
        )

        with st.expander(header, expanded=(purchase_date == dates[0])):
            strategies = sorted(
                day_df["strategy_group"].unique(),
                key=lambda s: (s == "Confluence", s),
            )
            for strategy in strategies:
                sub = day_df[day_df["strategy_group"] == strategy].sort_values(
                    "pl_pct", ascending=False, na_position="last"
                )
                sub_priced = sub[sub["pl_amount"].notna()]
                sub_pl = float(sub_priced["pl_amount"].sum()) if len(sub_priced) else 0.0
                sub_cost = float(sub["cost_basis"].sum())
                sub_pct = (sub_pl / sub_cost * 100.0) if sub_cost else None

                st.markdown(
                    f"**{strategy}** — {len(sub)} picks · "
                    f"P/L ₹{sub_pl:+,.2f}"
                    + (f" ({sub_pct:+.2f}%)" if sub_pct is not None else "")
                )
                _holdings_table(sub)
                st.divider()


def render_paper_trading_page(db: DuckDBManager) -> None:
    st.subheader("Portfolio")
    st.caption(
        "Scan picks grouped by purchase date and strategy. "
        "One share per symbol per day — duplicate scans are ignored. "
        "Refresh market data to update live P/L."
    )

    service = PaperTradingService(db=db)
    sync = service.sync_portfolio_from_scans()
    if sync["holdings_added"] > 0:
        st.success(
            f"Imported {sync['holdings_added']} holdings from "
            f"{sync['scans_processed']} past scan(s)."
        )
    if sync.get("duplicates_removed", 0) > 0:
        st.caption(f"Removed {sync['duplicates_removed']} duplicate holding(s).")

    summary = service.summarize_portfolio()

    if summary["holding_count"] == 0:
        st.info("No holdings yet. Run an **Equity** scan — picks are added automatically.")
        return

    if summary["holding_count"] > 0 and summary["total_market_value"] == 0:
        st.warning("Refresh market data on the Equity tab to load current prices.")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Holdings", summary["holding_count"])
    with m2:
        st.metric("Invested", f"₹{summary['total_cost']:,.2f}")
    with m3:
        st.metric("Market value", f"₹{summary['total_market_value']:,.2f}")
    with m4:
        _metric_pl("Total P/L", summary["total_pl_amount"], summary["total_pl_pct"])
    with m5:
        wr = summary["win_rate"]
        st.metric("Win rate", f"{wr:.0f}%" if wr is not None else "—")

    st.caption(f"Live prices as of **{_fmt_date(summary['latest_price_date'])}**")

    df = service.get_portfolio()
    date_options = ["All dates"] + [
        _fmt_date(d) for d in sorted(df["purchase_date"].unique(), reverse=True)
    ]

    tab_overview, tab_holdings = st.tabs(["Overview & charts", "Holdings"])

    with tab_overview:
        _render_overview_charts(service, df)

    with tab_holdings:
        selected_date = st.selectbox("Filter by purchase date", date_options, index=0)
        _render_holdings_by_date(
            df,
            None if selected_date == "All dates" else selected_date,
        )
