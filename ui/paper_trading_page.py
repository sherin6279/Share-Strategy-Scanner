"""Portfolio UI — date/strategy grouped holdings with charts."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import MAX_SHARE_PRICE_INR
from database.duckdb_manager import DuckDBManager
from paper_trading.service import PaperTradingService

CHART_COLORS = px.colors.qualitative.Set2
PL_GREEN = "#16a34a"
PL_RED = "#dc2626"
ACCENT = "#4f46e5"
MUTED = "#64748b"

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#334155"),
    title=dict(font=dict(size=14, color="#0f172a")),
)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .pf-hero {
            padding: 1.1rem 1.25rem;
            border-radius: 12px;
            background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
            border: 1px solid #e2e8f0;
            margin-bottom: 1rem;
        }
        .pf-hero h2 { margin: 0 0 0.35rem 0; font-size: 1.45rem; color: #0f172a; }
        .pf-hero p { margin: 0; color: #64748b; font-size: 0.92rem; line-height: 1.45; }
        .stat-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.95rem 1rem;
            min-height: 5.5rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        .stat-card.accent { border-top: 3px solid #4f46e5; }
        .stat-card.positive { border-top: 3px solid #16a34a; }
        .stat-card.negative { border-top: 3px solid #dc2626; }
        .stat-label {
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            color: #64748b;
            margin-bottom: 0.35rem;
        }
        .stat-value {
            font-size: 1.35rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.2;
            word-break: break-word;
        }
        .stat-sub { font-size: 0.82rem; color: #64748b; margin-top: 0.3rem; }
        .stat-sub.up { color: #16a34a; font-weight: 600; }
        .stat-sub.down { color: #dc2626; font-weight: 600; }
        .chip-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.15rem; }
        .chip {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
            border: 1px solid #e2e8f0;
            background: #f8fafc;
            color: #334155;
        }
        .chip.win { background: #ecfdf5; color: #166534; border-color: #bbf7d0; }
        .chip.loss { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
        .chip.flat { background: #f8fafc; color: #475569; }
        .section-title {
            font-size: 1rem;
            font-weight: 700;
            color: #0f172a;
            margin: 1.25rem 0 0.75rem 0;
        }
        div[data-testid="stPlotlyChart"] {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.35rem;
            background: #fff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def _fmt_inr(amount: float | None, signed: bool = False) -> str:
    if amount is None:
        return "—"
    if signed:
        return f"₹{amount:+,.0f}"
    return f"₹{amount:,.0f}"


def _stat_card(
    label: str,
    value: str,
    sub: str | None = None,
    tone: str = "accent",
    sub_tone: str | None = None,
) -> None:
    sub_class = f"stat-sub {sub_tone}" if sub_tone else "stat-sub"
    sub_html = f'<div class="{sub_class}">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="stat-card {tone}">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_summary_cards(summary: dict) -> None:
    pl = summary["total_pl_amount"]
    pl_pct = summary["total_pl_pct"]
    pl_tone = "positive" if pl is not None and pl >= 0 else "negative"
    pl_sub_tone = "up" if pl is not None and pl >= 0 else "down"

    c1, c2, c3 = st.columns(3)
    with c1:
        _stat_card("Invested", _fmt_inr(summary["total_cost"]), "Total buy value")
    with c2:
        _stat_card("Market value", _fmt_inr(summary["total_market_value"]), "As of latest prices")
    with c3:
        _stat_card(
            "Total P/L",
            _fmt_inr(pl, signed=True),
            f"{pl_pct:+.2f}% portfolio return" if pl_pct is not None else None,
            tone=pl_tone,
            sub_tone=pl_sub_tone,
        )

    c4, c5, c6, c7 = st.columns([1, 1, 1.4, 1.6])
    with c4:
        _stat_card("Holdings", str(summary["holding_count"]), f"{summary['symbols']} symbols")
    with c5:
        avg = summary.get("avg_holding_pl_pct")
        avg_tone = "positive" if avg is not None and avg >= 0 else "negative"
        _stat_card(
            "Avg holding P/L",
            f"{avg:+.2f}%" if avg is not None else "—",
            "Equal-weight across rows",
            tone=avg_tone,
            sub_tone="up" if avg_tone == "positive" else "down",
        )
    with c6:
        st.markdown(
            f"""
            <div class="stat-card accent">
                <div class="stat-label">Outcome mix</div>
                <div class="chip-row">
                    <span class="chip win">▲ {summary['winners']} win</span>
                    <span class="chip loss">▼ {summary['losers']} loss</span>
                    <span class="chip flat">■ {summary.get('flat', 0)} flat</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c7:
        wr = summary.get("win_rate")
        _stat_card(
            "Win rate",
            f"{wr:.0f}%" if wr is not None else "—",
            "Share of holdings in profit",
        )


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


def _apply_chart_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(**CHART_LAYOUT)
    return fig


def _pie_by_date(date_df: pd.DataFrame) -> go.Figure:
    labels = [_fmt_date(d) for d in date_df["purchase_date"]]
    values = date_df["market_value"].fillna(date_df["invested"])
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=CHART_COLORS, line=dict(color="#fff", width=2)),
                textinfo="percent",
                textposition="inside",
                insidetextorientation="horizontal",
                hovertemplate="%{label}<br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Capital by purchase date",
        margin=dict(t=48, b=16, l=16, r=16),
        height=360,
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.05, x=0),
    )
    return _apply_chart_style(fig)


def _pie_by_strategy(strat_df: pd.DataFrame) -> go.Figure:
    sorted_df = strat_df.sort_values("holdings", ascending=False)
    fig = go.Figure(
        data=[
            go.Pie(
                labels=sorted_df["strategy"],
                values=sorted_df["holdings"],
                hole=0.55,
                marker=dict(colors=CHART_COLORS, line=dict(color="#fff", width=2)),
                textinfo="label+value",
                textposition="outside",
                textfont=dict(size=11),
                hovertemplate="%{label}<br>%{value} holdings<br>%{percent}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title="Holdings by strategy",
        margin=dict(t=48, b=16, l=16, r=16),
        height=360,
        showlegend=False,
    )
    return _apply_chart_style(fig)


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
        title="P/L by strategy",
        xaxis_title="P/L ₹",
        margin=dict(t=48, b=32, l=16, r=72),
        height=max(300, 48 * len(sorted_df)),
    )
    fig.add_vline(x=0, line_width=1, line_color="#cbd5e1")
    return _apply_chart_style(fig)


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
        margin=dict(t=48, b=48, l=48, r=16),
        height=340,
    )
    fig.add_hline(y=0, line_width=1, line_color="#cbd5e1")
    return _apply_chart_style(fig)


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
        _fmt_inr(pl, signed=True),
        delta=f"{pl_pct:+.2f}%" if pl_pct is not None else None,
    )
    st.sidebar.caption(f"Prices as of {_fmt_date(summary['latest_price_date'])}")

    by_date = service.summarize_by_date()
    if not by_date.empty:
        st.sidebar.markdown("**Recent scan dates**")
        for _, row in by_date.head(4).iterrows():
            icon = "🟢" if row["pl_amount"] >= 0 else "🔴"
            st.sidebar.caption(
                f"{icon} {_fmt_date(row['purchase_date'])} · "
                f"{int(row['holdings'])} stk · {_fmt_inr(row['pl_amount'], signed=True)}"
            )


def _render_overview_charts(
    service: PaperTradingService,
    df: pd.DataFrame,
) -> None:
    date_sum = service.summarize_by_date(df)
    strat_sum = service.summarize_by_strategy(df)

    st.markdown('<div class="section-title">Allocation</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_pie_by_date(date_sum), use_container_width=True)
    with c2:
        st.plotly_chart(_pie_by_strategy(strat_sum), use_container_width=True)

    st.markdown('<div class="section-title">Performance</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(_bar_strategy_pl(strat_sum), use_container_width=True)
    with c4:
        st.plotly_chart(_bar_date_pl(date_sum), use_container_width=True)

    st.markdown('<div class="section-title">Strategy breakdown</div>', unsafe_allow_html=True)
    strat_display = strat_sum.rename(
        columns={
            "strategy": "Strategy",
            "holdings": "Holdings",
            "invested": "Invested",
            "market_value": "Market value",
            "pl_amount": "P/L ₹",
            "pl_pct": "P/L %",
            "avg_pl_pct": "Avg P/L %",
            "win_rate": "Win %",
        }
    )
    st.dataframe(
        strat_display[
            [
                "Strategy",
                "Holdings",
                "Invested",
                "Market value",
                "P/L ₹",
                "P/L %",
                "Avg P/L %",
                "Win %",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Invested": st.column_config.NumberColumn(format="₹%.0f"),
            "Market value": st.column_config.NumberColumn(format="₹%.0f"),
            "P/L ₹": st.column_config.NumberColumn(format="₹%+,.0f"),
            "P/L %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Avg P/L %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Win %": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )


def _render_holdings_by_date(df: pd.DataFrame, date_filter: str | None) -> None:
    dates = sorted(df["purchase_date"].unique(), reverse=True)
    if date_filter and date_filter != "All dates":
        dates = [d for d in dates if _fmt_date(d) == date_filter]

    for purchase_date in dates:
        day_df = df[df["purchase_date"] == purchase_date]
        day_cost = float(day_df["cost_basis"].sum())
        priced = day_df[day_df["pl_amount"].notna()]
        day_pl = float(priced["pl_amount"].sum()) if len(priced) else 0.0
        priced_cost = float(priced["cost_basis"].sum()) if len(priced) else 0.0
        day_pl_pct = (day_pl / priced_cost * 100.0) if priced_cost else None
        avg_day_pct = float(priced["pl_pct"].mean()) if len(priced) else None
        pl_icon = "🟢" if day_pl >= 0 else "🔴"

        header = (
            f"{pl_icon} **{_fmt_date_long(purchase_date)}** · "
            f"{len(day_df)} holdings · "
            f"Invested {_fmt_inr(day_cost)} · "
            f"P/L **{_fmt_inr(day_pl, signed=True)}**"
            + (f" ({day_pl_pct:+.2f}%)" if day_pl_pct is not None else "")
            + (f" · avg {avg_day_pct:+.1f}%" if avg_day_pct is not None else "")
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
                sub_cost = float(sub_priced["cost_basis"].sum()) if len(sub_priced) else 0.0
                sub_pct = (sub_pl / sub_cost * 100.0) if sub_cost else None
                badge = "🟢" if sub_pl >= 0 else "🔴"

                st.markdown(
                    f"##### {badge} {strategy}"
                    f"  \n"
                    f"<span style='color:{MUTED};font-size:0.9rem'>"
                    f"{len(sub)} picks · P/L {_fmt_inr(sub_pl, signed=True)}"
                    + (f" ({sub_pct:+.2f}%)" if sub_pct is not None else "")
                    + "</span>",
                    unsafe_allow_html=True,
                )
                _holdings_table(sub)


def render_paper_trading_page(db: DuckDBManager) -> None:
    _inject_styles()

    st.markdown(
        f"""
        <div class="pf-hero">
            <h2>Portfolio</h2>
            <p>Scan picks grouped by date and strategy. One share per symbol per day,
            max ₹{MAX_SHARE_PRICE_INR:,.0f} per share. Refresh market data on the Equity tab to update live prices.</p>
        </div>
        """,
        unsafe_allow_html=True,
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
    if sync.get("rs_cleanup_removed", 0) > 0:
        st.info(
            f"One-time cleanup: removed {sync['rs_cleanup_removed']} extra RS Leader "
            f"holdings (kept top 5 per day)."
        )
    if sync.get("max_price_cleanup_removed", 0) > 0:
        st.info(
            f"One-time cleanup: removed {sync['max_price_cleanup_removed']} holding(s) "
            f"above ₹{MAX_SHARE_PRICE_INR:,.0f} per share."
        )

    summary = service.summarize_portfolio()

    if summary["holding_count"] == 0:
        st.info("No holdings yet. Run an **Equity** scan — picks are added automatically.")
        return

    if summary["holding_count"] > 0 and summary["total_market_value"] == 0:
        st.warning("Refresh market data on the Equity tab to load current prices.")

    _render_summary_cards(summary)

    with st.expander("Why can total P/L differ from row-level P/L %?"):
        st.markdown(
            """
            **Total P/L ₹** adds rupee gains/losses — expensive stocks move it more than cheap ones.

            **Portfolio return %** is weighted by capital. **Avg holding P/L** treats every row equally.

            Example: two POWERINDIA lots at ~₹36k each losing ~6% can outweigh many small +20% winners.
            """
        )

    st.caption(f"Live prices as of **{_fmt_date(summary['latest_price_date'])}**")

    df = service.get_portfolio()
    date_options = ["All dates"] + [
        _fmt_date(d) for d in sorted(df["purchase_date"].unique(), reverse=True)
    ]

    tab_overview, tab_holdings = st.tabs(["Overview & charts", "Holdings by date"])

    with tab_overview:
        _render_overview_charts(service, df)

    with tab_holdings:
        selected_date = st.selectbox("Filter by purchase date", date_options, index=0)
        _render_holdings_by_date(
            df,
            None if selected_date == "All dates" else selected_date,
        )
