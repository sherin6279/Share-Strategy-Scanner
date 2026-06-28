"""Paper portfolio service — record scan picks and track live P/L."""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from typing import Any

import pandas as pd

from database.duckdb_manager import DuckDBManager
from utils.logger import get_logger

logger = get_logger(__name__)

STRATEGY_NAMES = {
    1: "Basic Breakout",
    2: "Refined Breakout",
    3: "Enterprise Breakout",
    4: "Volatility Compression",
    5: "Trend Pullback",
    6: "RS Leaders",
    7: "Stage-1 Base Breakout",
}

CONFLUENCE_TYPE = "confluence"
STRATEGY_TYPE = "strategy"


class PaperTradingService:
    """Manages a cumulative paper portfolio from equity scan picks."""

    def __init__(self, db: DuckDBManager | None = None) -> None:
        self.db = db or DuckDBManager()

    def record_from_scan(self, scan_run_id: str) -> dict[str, Any]:
        """
        Buy 1 share of each unique scan pick at the signal-day close.

        Multiple strategies for the same symbol in one scan → one Confluence row.
        """
        if self.db.portfolio_exists_for_scan(scan_run_id):
            holdings = self.db.list_portfolio_holdings()
            n = len(holdings[holdings["scan_run_id"] == scan_run_id])
            logger.info("Portfolio already recorded for scan %s", scan_run_id)
            return {
                "scan_run_id": scan_run_id,
                "holdings_added": 0,
                "holdings_count": n,
                "already_recorded": True,
            }

        scan_df = self.db.get_scan_results(scan_run_id=scan_run_id)
        if scan_df.empty:
            raise RuntimeError(f"No scan results found for run {scan_run_id}")

        created_at = datetime.now()
        holdings = self._build_holdings_from_scan(scan_df, scan_run_id, created_at)
        if not holdings:
            return {
                "scan_run_id": scan_run_id,
                "holdings_added": 0,
                "holdings_count": 0,
                "already_recorded": False,
            }

        self.db.insert_portfolio_holdings(holdings)
        logger.info(
            "Added %d holdings to portfolio from scan %s",
            len(holdings),
            scan_run_id,
        )
        return {
            "scan_run_id": scan_run_id,
            "holdings_added": len(holdings),
            "holdings_count": len(holdings),
            "confluence_count": sum(
                1 for h in holdings if h["source_type"] == CONFLUENCE_TYPE
            ),
            "already_recorded": False,
        }

    def _build_holdings_from_scan(
        self,
        scan_df: pd.DataFrame,
        scan_run_id: str,
        created_at: datetime,
    ) -> list[dict[str, Any]]:
        grouped = (
            scan_df.groupby("symbol", as_index=False)
            .agg(
                strategy_ids=("strategy_id", lambda s: sorted({int(x) for x in s})),
                purchase_price=("trigger_price", "first"),
                signal_date=("signal_date", "first"),
                score=("score", "max"),
            )
        )

        holdings: list[dict[str, Any]] = []
        for row in grouped.itertuples(index=False):
            strategy_ids: list[int] = list(row.strategy_ids)
            purchase_date = pd.to_datetime(row.signal_date).date()

            if len(strategy_ids) >= 2:
                source_type = CONFLUENCE_TYPE
                source_label = "Confluence"
            else:
                source_type = STRATEGY_TYPE
                source_label = STRATEGY_NAMES.get(strategy_ids[0], f"S{strategy_ids[0]}")

            holdings.append(
                {
                    "holding_id": uuid.uuid4().hex[:16],
                    "scan_run_id": scan_run_id,
                    "symbol": row.symbol,
                    "source_type": source_type,
                    "source_label": source_label,
                    "strategy_ids": strategy_ids,
                    "purchase_date": purchase_date,
                    "purchase_price": float(row.purchase_price),
                    "quantity": 1,
                    "score": float(row.score),
                    "created_at": created_at,
                }
            )
        return holdings

    def get_portfolio(self) -> pd.DataFrame:
        """All holdings marked to market with P/L as of latest stored prices."""
        holdings = self.db.list_portfolio_holdings()
        if holdings.empty:
            return pd.DataFrame()

        symbols = holdings["symbol"].unique().tolist()
        latest_closes = self.db.get_latest_closes(symbols)

        rows: list[dict[str, Any]] = []
        for _, h in holdings.iterrows():
            symbol = h["symbol"]
            purchase_date = pd.to_datetime(h["purchase_date"]).date()
            purchase_price = float(h["purchase_price"])
            qty = int(h["quantity"]) if pd.notna(h["quantity"]) else 1

            strategy_ids = h["strategy_ids"]
            if isinstance(strategy_ids, str):
                try:
                    strategy_ids = json.loads(strategy_ids)
                except json.JSONDecodeError:
                    strategy_ids = []

            current_date: date | None = None
            current_price: float | None = None
            market_value: float | None = None
            cost_basis = purchase_price * qty
            pl_amount: float | None = None
            pl_pct: float | None = None
            days_held = 0

            if symbol in latest_closes:
                current_date, current_price = latest_closes[symbol]
                days_held = max(0, (current_date - purchase_date).days)
                if current_date >= purchase_date and current_price is not None:
                    market_value = current_price * qty
                    pl_amount = market_value - cost_basis
                    pl_pct = (pl_amount / cost_basis) * 100.0 if cost_basis else None

            source_label = h["source_label"]
            if h["source_type"] == CONFLUENCE_TYPE and strategy_ids:
                names = [STRATEGY_NAMES.get(int(s), f"S{s}") for s in strategy_ids]
                source_label = f"Confluence ({', '.join(names)})"

            rows.append(
                {
                    "holding_id": h["holding_id"],
                    "symbol": symbol,
                    "source": source_label,
                    "source_type": h["source_type"],
                    "purchase_date": purchase_date,
                    "purchase_price": purchase_price,
                    "quantity": qty,
                    "cost_basis": cost_basis,
                    "current_date": current_date,
                    "current_price": current_price,
                    "market_value": market_value,
                    "days_held": days_held,
                    "pl_amount": pl_amount,
                    "pl_pct": pl_pct,
                    "score": float(h["score"]) if pd.notna(h["score"]) else None,
                    "scan_run_id": h["scan_run_id"],
                }
            )

        return pd.DataFrame(rows)

    def summarize_portfolio(self) -> dict[str, Any]:
        df = self.get_portfolio()
        if df.empty:
            return {
                "holding_count": 0,
                "total_cost": 0.0,
                "total_market_value": 0.0,
                "total_pl_amount": 0.0,
                "total_pl_pct": None,
                "winners": 0,
                "losers": 0,
                "win_rate": None,
                "latest_price_date": None,
                "symbols": 0,
            }

        priced = df[df["pl_amount"].notna()]
        total_cost = float(df["cost_basis"].sum())
        total_mv = float(priced["market_value"].sum()) if len(priced) else 0.0
        total_pl = float(priced["pl_amount"].sum()) if len(priced) else 0.0
        winners = int((priced["pl_amount"] > 0).sum())
        losers = int((priced["pl_amount"] < 0).sum())
        win_rate = (winners / len(priced) * 100.0) if len(priced) else None

        return {
            "holding_count": len(df),
            "total_cost": total_cost,
            "total_market_value": total_mv,
            "total_pl_amount": total_pl,
            "total_pl_pct": (total_pl / total_cost * 100.0) if total_cost else None,
            "winners": winners,
            "losers": losers,
            "win_rate": win_rate,
            "latest_price_date": priced["current_date"].max() if len(priced) else None,
            "symbols": df["symbol"].nunique(),
        }
