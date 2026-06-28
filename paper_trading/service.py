"""Paper trading service — record scan picks and mark-to-market P/L."""

from __future__ import annotations

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


class PaperTradingService:
    """Snapshots scan signals at entry prices and tracks actual P/L from local candles."""

    def __init__(self, db: DuckDBManager | None = None) -> None:
        self.db = db or DuckDBManager()

    def record_from_scan(self, scan_run_id: str) -> dict[str, Any]:
        """
        Save all signals from a scan run as paper-trade positions.

        Entry price is the signal-day close (trigger_price) — the actual market
        value on the scan date.
        """
        existing = self.db.get_paper_trade_batch_by_scan(scan_run_id)
        if existing:
            logger.info("Paper trade batch already exists for scan %s", scan_run_id)
            return {
                "batch_id": existing["batch_id"],
                "scan_run_id": scan_run_id,
                "position_count": existing["position_count"],
                "already_recorded": True,
            }

        scan_df = self.db.get_scan_results(scan_run_id=scan_run_id)
        if scan_df.empty:
            raise RuntimeError(f"No scan results found for run {scan_run_id}")

        batch_id = uuid.uuid4().hex[:16]
        created_at = datetime.now()
        entry_dates = pd.to_datetime(scan_df["signal_date"]).dt.date
        entry_date = entry_dates.max()

        positions: list[dict[str, Any]] = []
        for _, row in scan_df.iterrows():
            signal_date = pd.to_datetime(row["signal_date"]).date()
            positions.append(
                {
                    "position_id": uuid.uuid4().hex[:16],
                    "batch_id": batch_id,
                    "symbol": row["symbol"],
                    "strategy_id": int(row["strategy_id"]),
                    "entry_date": signal_date,
                    "entry_price": float(row["trigger_price"]),
                    "score": float(row["score"]),
                }
            )

        self.db.insert_paper_trade_batch(
            batch_id=batch_id,
            scan_run_id=scan_run_id,
            created_at=created_at,
            entry_date=entry_date,
            position_count=len(positions),
            notes=f"Auto-recorded from equity scan {scan_run_id}",
        )
        self.db.insert_paper_trade_positions(positions)

        logger.info(
            "Recorded paper trade batch %s with %d positions",
            batch_id,
            len(positions),
        )
        return {
            "batch_id": batch_id,
            "scan_run_id": scan_run_id,
            "entry_date": str(entry_date),
            "position_count": len(positions),
            "already_recorded": False,
        }

    def compute_pl(
        self,
        batch_id: str,
        min_hold_days: int = 0,
    ) -> pd.DataFrame:
        """
        Mark all positions to market using latest closes in DuckDB.

        Refresh market data first so prices are current.
        """
        batch = self.db.get_paper_trade_batch(batch_id)
        if batch is None:
            raise ValueError(f"Unknown paper trade batch: {batch_id}")

        positions = self.db.get_paper_trade_positions(batch_id)
        if positions.empty:
            return pd.DataFrame()

        symbols = positions["symbol"].unique().tolist()
        latest_closes = self.db.get_latest_closes(symbols)

        rows: list[dict[str, Any]] = []
        for _, pos in positions.iterrows():
            entry_date = pd.to_datetime(pos["entry_date"]).date()
            entry_price = float(pos["entry_price"])
            symbol = pos["symbol"]

            current_date: date | None = None
            current_price: float | None = None
            pl_amount: float | None = None
            pl_pct: float | None = None
            days_held = 0

            if symbol in latest_closes:
                current_date, current_price = latest_closes[symbol]
                days_held = (current_date - entry_date).days
                if current_date >= entry_date and current_price is not None:
                    pl_amount = current_price - entry_price
                    pl_pct = (pl_amount / entry_price) * 100.0 if entry_price else None

            strategy_id = int(pos["strategy_id"])
            rows.append(
                {
                    "position_id": pos["position_id"],
                    "symbol": symbol,
                    "strategy_id": strategy_id,
                    "strategy_name": STRATEGY_NAMES.get(strategy_id, str(strategy_id)),
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "current_date": current_date,
                    "current_price": current_price,
                    "days_held": days_held,
                    "pl_amount": pl_amount,
                    "pl_pct": pl_pct,
                    "score": float(pos["score"]),
                }
            )

        df = pd.DataFrame(rows)
        if min_hold_days > 0 and not df.empty:
            df = df[df["days_held"] >= min_hold_days].reset_index(drop=True)
        return df

    def summarize_pl(self, batch_id: str, min_hold_days: int = 0) -> dict[str, Any]:
        """Aggregate P/L stats for a paper trade batch."""
        df = self.compute_pl(batch_id, min_hold_days=min_hold_days)
        if df.empty:
            return {
                "batch_id": batch_id,
                "position_count": 0,
                "with_prices": 0,
                "winners": 0,
                "losers": 0,
                "flat": 0,
                "avg_pl_pct": None,
                "total_pl_amount": None,
                "win_rate": None,
            }

        priced = df[df["pl_pct"].notna()]
        winners = int((priced["pl_pct"] > 0).sum())
        losers = int((priced["pl_pct"] < 0).sum())
        flat = int((priced["pl_pct"] == 0).sum())
        win_rate = (winners / len(priced) * 100.0) if len(priced) else None

        return {
            "batch_id": batch_id,
            "position_count": len(df),
            "with_prices": len(priced),
            "winners": winners,
            "losers": losers,
            "flat": flat,
            "avg_pl_pct": float(priced["pl_pct"].mean()) if len(priced) else None,
            "total_pl_amount": float(priced["pl_amount"].sum()) if len(priced) else None,
            "win_rate": win_rate,
            "latest_price_date": (
                priced["current_date"].max() if len(priced) else None
            ),
        }
