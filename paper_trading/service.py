"""Paper portfolio service — record scan picks and track live P/L."""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime
from typing import Any

import pandas as pd

from config.settings import MAX_SHARE_PRICE_INR, RS_LEADERS_MAX_PICKS
from database.duckdb_manager import DuckDBManager
from scanners.signal_filters import apply_scan_df_filters
from strategies.strategy_6_relative_strength_leaders import STRATEGY_ID as RS_LEADERS_ID
from utils.logger import get_logger

logger = get_logger(__name__)

RS_CLEANUP_METADATA_KEY = "portfolio_rs_top5_cleanup_done"
MAX_PRICE_CLEANUP_METADATA_KEY = "portfolio_max_share_price_cleanup_done"

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

    def run_one_time_rs_cleanup(self) -> dict[str, Any]:
        """
        One-time removal of RS-only holdings beyond top N per purchase date.

        Confluence picks are kept. Safe to call repeatedly — runs once only.
        """
        if self.db.get_metadata(RS_CLEANUP_METADATA_KEY) == "true":
            return {"removed": 0, "already_done": True}

        removed = self._cleanup_excess_rs_leader_holdings()
        self.db.set_metadata(RS_CLEANUP_METADATA_KEY, "true")
        logger.info("RS Leaders cleanup removed %d excess holdings", removed)
        return {"removed": removed, "already_done": False}

    def _cleanup_excess_rs_leader_holdings(self) -> int:
        holdings = self.db.list_portfolio_holdings()
        if holdings.empty:
            return 0

        rs_only = holdings[
            (holdings["source_type"] == STRATEGY_TYPE)
            & (holdings["source_label"] == STRATEGY_NAMES[RS_LEADERS_ID])
        ].copy()
        if rs_only.empty:
            return 0

        rs_only["purchase_date"] = pd.to_datetime(rs_only["purchase_date"]).dt.date
        to_delete: list[str] = []
        for _, grp in rs_only.groupby("purchase_date", sort=False):
            ranked = grp.sort_values("score", ascending=False, na_position="last")
            excess = ranked.iloc[RS_LEADERS_MAX_PICKS:]
            to_delete.extend(excess["holding_id"].astype(str).tolist())

        return self.db.delete_portfolio_holdings(to_delete)

    def run_one_time_max_price_cleanup(self) -> dict[str, Any]:
        """
        One-time removal of holdings bought above MAX_SHARE_PRICE_INR.

        Safe to call repeatedly — runs once only.
        """
        if self.db.get_metadata(MAX_PRICE_CLEANUP_METADATA_KEY) == "true":
            return {"removed": 0, "already_done": True}

        holdings = self.db.list_portfolio_holdings()
        if holdings.empty:
            self.db.set_metadata(MAX_PRICE_CLEANUP_METADATA_KEY, "true")
            return {"removed": 0, "already_done": False}

        expensive = holdings[holdings["purchase_price"] > MAX_SHARE_PRICE_INR]
        to_delete = expensive["holding_id"].astype(str).tolist()
        removed = self.db.delete_portfolio_holdings(to_delete)
        self.db.set_metadata(MAX_PRICE_CLEANUP_METADATA_KEY, "true")
        logger.info(
            "Max-price cleanup removed %d holdings above ₹%.0f",
            removed,
            MAX_SHARE_PRICE_INR,
        )
        return {"removed": removed, "already_done": False}

    def sync_portfolio_from_scans(self) -> dict[str, Any]:
        """
        Import all historical equity scans into the portfolio.

        Skips scans already processed and symbols already held on the same date.
        """
        rs_cleanup = self.run_one_time_rs_cleanup()
        price_cleanup = self.run_one_time_max_price_cleanup()
        removed = self.db.dedupe_portfolio_holdings()
        runs = self.db.list_equity_scan_runs_chronological()
        scans_processed = 0
        total_added = 0
        skipped_scans = 0
        empty_scans = 0

        for run in runs:
            run_id = run.get("run_id")
            scan_ts = run.get("scan_timestamp")
            if not run_id:
                if scan_ts is None:
                    continue
                run_id = f"legacy_{pd.Timestamp(scan_ts).strftime('%Y%m%d_%H%M%S')}"
            if self.db.portfolio_scan_processed(run_id):
                skipped_scans += 1
                continue
            result = self.record_from_scan(run_id, scan_timestamp=scan_ts)
            scans_processed += 1
            total_added += result.get("holdings_added", 0)
            if result.get("empty_scan"):
                empty_scans += 1

        return {
            "scans_processed": scans_processed,
            "scans_skipped": skipped_scans,
            "empty_scans": empty_scans,
            "holdings_added": total_added,
            "duplicates_removed": removed,
            "rs_cleanup_removed": rs_cleanup.get("removed", 0),
            "max_price_cleanup_removed": price_cleanup.get("removed", 0),
            "total_holdings": self.db.count_portfolio_holdings(),
        }

    def record_from_scan(
        self,
        scan_run_id: str,
        scan_timestamp: datetime | None = None,
    ) -> dict[str, Any]:
        """
        Buy 1 share of each unique scan pick at the signal-day close.

        - Multiple strategies for the same symbol in one scan → Confluence row.
        - Same symbol on the same purchase date is never added twice.
        - Re-running the same scan is a no-op (tracked in portfolio_scan_log).
        """
        if self.db.portfolio_scan_processed(scan_run_id):
            holdings = self.db.list_portfolio_holdings()
            n = len(holdings[holdings["scan_run_id"] == scan_run_id])
            logger.info("Scan %s already in portfolio log", scan_run_id)
            return {
                "scan_run_id": scan_run_id,
                "holdings_added": 0,
                "holdings_skipped_duplicate": 0,
                "holdings_count": n,
                "already_recorded": True,
            }

        scan_df = self.db.get_scan_results(
            scan_run_id=scan_run_id, scan_timestamp=scan_timestamp
        )
        scan_df = apply_scan_df_filters(scan_df)
        if scan_df.empty:
            logger.warning("No scan results for run %s — marking as processed", scan_run_id)
            self.db.log_portfolio_scan(scan_run_id, datetime.now(), 0)
            return {
                "scan_run_id": scan_run_id,
                "holdings_added": 0,
                "holdings_skipped_duplicate": 0,
                "holdings_count": 0,
                "already_recorded": False,
                "empty_scan": True,
            }

        created_at = datetime.now()
        candidates = self._build_holdings_from_scan(scan_df, scan_run_id, created_at)
        holdings, skipped = self._filter_duplicate_symbol_dates(candidates)

        added = 0
        if holdings:
            added = self.db.insert_portfolio_holdings(holdings)
            logger.info("Added %d holdings from scan %s (%d dupes skipped)", added, scan_run_id, skipped)

        self.db.log_portfolio_scan(scan_run_id, created_at, added)

        return {
            "scan_run_id": scan_run_id,
            "holdings_added": added,
            "holdings_skipped_duplicate": skipped,
            "holdings_count": added,
            "confluence_count": sum(
                1 for h in holdings if h["source_type"] == CONFLUENCE_TYPE
            ),
            "already_recorded": False,
        }

    def _filter_duplicate_symbol_dates(
        self,
        holdings: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        existing = self.db.get_portfolio_symbol_dates()
        kept: list[dict[str, Any]] = []
        skipped = 0
        for h in holdings:
            key = (h["symbol"], h["purchase_date"])
            if key in existing:
                skipped += 1
                continue
            kept.append(h)
            existing.add(key)
        return kept, skipped

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

    def get_strategy_group(self, source_type: str, source_label: str) -> str:
        if source_type == CONFLUENCE_TYPE:
            return "Confluence"
        return source_label

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
                    "strategy_group": self.get_strategy_group(
                        h["source_type"], h["source_label"]
                    ),
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
        unpriced_count = len(df) - len(priced)
        total_cost = float(df["cost_basis"].sum())
        priced_cost = float(priced["cost_basis"].sum()) if len(priced) else 0.0
        total_mv = float(priced["market_value"].sum()) if len(priced) else 0.0
        total_pl = float(priced["pl_amount"].sum()) if len(priced) else 0.0
        winners = int((priced["pl_amount"] > 0).sum())
        losers = int((priced["pl_amount"] < 0).sum())
        flat = int((priced["pl_amount"] == 0).sum())
        win_rate = (winners / len(priced) * 100.0) if len(priced) else None
        avg_pl_pct = float(priced["pl_pct"].mean()) if len(priced) else None

        return {
            "holding_count": len(df),
            "priced_count": len(priced),
            "unpriced_count": unpriced_count,
            "total_cost": total_cost,
            "priced_cost": priced_cost,
            "total_market_value": total_mv,
            "total_pl_amount": total_pl,
            "total_pl_pct": (total_pl / priced_cost * 100.0) if priced_cost else None,
            "avg_holding_pl_pct": avg_pl_pct,
            "winners": winners,
            "losers": losers,
            "flat": flat,
            "win_rate": win_rate,
            "latest_price_date": priced["current_date"].max() if len(priced) else None,
            "symbols": df["symbol"].nunique(),
        }

    def summarize_by_date(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        df = df if df is not None else self.get_portfolio()
        if df.empty:
            return pd.DataFrame()

        rows = []
        for purchase_date, grp in df.groupby("purchase_date", sort=False):
            priced = grp[grp["pl_amount"].notna()]
            cost = float(grp["cost_basis"].sum())
            pl = float(priced["pl_amount"].sum()) if len(priced) else 0.0
            rows.append(
                {
                    "purchase_date": purchase_date,
                    "holdings": len(grp),
                    "invested": cost,
                    "priced_cost": float(priced["cost_basis"].sum()) if len(priced) else 0.0,
                    "market_value": float(priced["market_value"].sum()) if len(priced) else 0.0,
                    "pl_amount": pl,
                    "pl_pct": (
                        (pl / float(priced["cost_basis"].sum()) * 100.0)
                        if len(priced) and priced["cost_basis"].sum()
                        else None
                    ),
                    "avg_pl_pct": float(priced["pl_pct"].mean()) if len(priced) else None,
                    "winners": int((priced["pl_amount"] > 0).sum()) if len(priced) else 0,
                    "losers": int((priced["pl_amount"] < 0).sum()) if len(priced) else 0,
                }
            )
        out = pd.DataFrame(rows)
        return out.sort_values("purchase_date", ascending=False).reset_index(drop=True)

    def summarize_by_strategy(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        df = df if df is not None else self.get_portfolio()
        if df.empty:
            return pd.DataFrame()

        rows = []
        for strategy, grp in df.groupby("strategy_group", sort=False):
            priced = grp[grp["pl_amount"].notna()]
            cost = float(grp["cost_basis"].sum())
            priced_cost = float(priced["cost_basis"].sum()) if len(priced) else 0.0
            pl = float(priced["pl_amount"].sum()) if len(priced) else 0.0
            rows.append(
                {
                    "strategy": strategy,
                    "holdings": len(grp),
                    "invested": cost,
                    "market_value": float(priced["market_value"].sum()) if len(priced) else 0.0,
                    "pl_amount": pl,
                    "pl_pct": (pl / priced_cost * 100.0) if priced_cost else None,
                    "avg_pl_pct": float(priced["pl_pct"].mean()) if len(priced) else None,
                    "win_rate": (
                        float((priced["pl_amount"] > 0).mean() * 100.0) if len(priced) else None
                    ),
                }
            )
        out = pd.DataFrame(rows)
        return out.sort_values("pl_amount", ascending=False).reset_index(drop=True)
