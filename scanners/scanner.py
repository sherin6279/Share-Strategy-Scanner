"""Scan orchestration – loads local data and runs strategy engine."""

from __future__ import annotations

import json
import uuid
from datetime import datetime

import pandas as pd

from database.duckdb_manager import DuckDBManager
from scanners.signal_filters import apply_scan_df_filters, apply_signal_filters
from strategies.strategy_engine import StrategyEngine
from utils.logger import get_logger

logger = get_logger(__name__)


class Scanner:
    """Runs scan using local DuckDB data only (no API calls)."""

    def __init__(
        self,
        db: DuckDBManager | None = None,
        engine: StrategyEngine | None = None,
    ) -> None:
        self.db = db or DuckDBManager()
        self.engine = engine or StrategyEngine()

    def run_scan(self) -> dict:
        """
        Execute full scan pipeline.

        Returns summary with signals count and strategy 3 status.
        """
        candles_map = self.db.load_all_candles()
        if not candles_map:
            raise RuntimeError("No candle data in database. Run Refresh Market Data first.")

        signals, scan_time = self.engine.run(candles_map)
        signals = apply_signal_filters(signals)
        run_id = uuid.uuid4().hex[:16]
        records = StrategyEngine.signals_to_records(
            signals, scan_time, scan_run_id=run_id
        )

        inserted = self.db.insert_scan_results(records)

        strategy_counts: dict[int, int] = {}
        for sig in signals:
            strategy_counts[sig.strategy_id] = strategy_counts.get(sig.strategy_id, 0) + 1

        self.db.insert_scan_run(
            run_id,
            scan_time,
            "equity",
            len(signals),
            bool(self.engine.market_uptrend),
            strategy_counts,
        )
        self.db.set_metadata("last_scan_run_id", run_id)
        self.db.set_metadata("last_scan_timestamp", scan_time.isoformat())
        self.db.set_metadata(
            "last_scan_market_uptrend",
            "true" if self.engine.market_uptrend else "false",
        )

        self.db.set_metadata("last_scan_strategy_counts", json.dumps(strategy_counts))

        logger.info("Stored %d scan results", inserted)

        return {
            "scan_run_id": run_id,
            "scan_timestamp": scan_time.isoformat(),
            "signal_count": len(signals),
            "strategy_counts": strategy_counts,
            "strategy_3_paused": self.engine.strategy_3_paused,
            "market_uptrend": bool(self.engine.market_uptrend),
        }

    def get_results_by_strategy(
        self,
        strategy_id: int,
        scan_run_id: str | None = None,
        scan_timestamp: datetime | None = None,
    ) -> pd.DataFrame:
        """Get scan results for a strategy from a specific scan run."""
        df = self.db.get_scan_results(
            scan_run_id=scan_run_id, scan_timestamp=scan_timestamp
        )
        if df.empty:
            return df
        df = apply_scan_df_filters(df)
        result = df[df["strategy_id"] == strategy_id].sort_values(
            "score", ascending=False
        ).reset_index(drop=True)
        return result

    def get_confluence(
        self,
        min_strategies: int = 2,
        scan_run_id: str | None = None,
        scan_timestamp: datetime | None = None,
    ) -> pd.DataFrame:
        """Find stocks appearing in multiple strategies."""
        df = self.db.get_scan_results(
            scan_run_id=scan_run_id, scan_timestamp=scan_timestamp
        )
        if df.empty:
            return pd.DataFrame()

        df = apply_scan_df_filters(df)

        strategy_names = {
            1: "Basic Breakout",
            2: "Refined Breakout",
            3: "Enterprise Breakout",
            4: "Volatility Compression",
            5: "Trend Pullback",
            6: "RS Leaders",
            7: "Stage-1 Base Breakout",
        }

        grouped = df.groupby("symbol").agg(
            strategy_count=("strategy_id", "nunique"),
            highest_score=("score", "max"),
            average_score=("score", "mean"),
            current_price=("trigger_price", "first"),
            strategies=("strategy_id", lambda x: sorted(x.unique().tolist())),
        ).reset_index()

        grouped["matching_strategies"] = grouped["strategies"].apply(
            lambda ids: ", ".join(strategy_names.get(i, str(i)) for i in ids)
        )
        grouped = grouped[grouped["strategy_count"] >= min_strategies]
        grouped = grouped.sort_values(
            ["strategy_count", "highest_score"], ascending=[False, False]
        ).reset_index(drop=True)

        return grouped[
            [
                "symbol",
                "current_price",
                "strategy_count",
                "matching_strategies",
                "highest_score",
                "average_score",
            ]
        ]
