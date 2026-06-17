"""Scan orchestration – loads local data and runs strategy engine."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from database.duckdb_manager import DuckDBManager
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
        records = StrategyEngine.signals_to_records(signals, scan_time)

        self.db.clear_scan_results()
        inserted = self.db.insert_scan_results(records)
        self.db.set_metadata("last_scan_timestamp", scan_time.isoformat())

        logger.info("Stored %d scan results", inserted)

        return {
            "scan_timestamp": scan_time.isoformat(),
            "signal_count": len(signals),
            "strategy_3_paused": self.engine.strategy_3_paused,
            "market_uptrend": self.engine.market_uptrend,
        }

    def get_results_by_strategy(self, strategy_id: int) -> pd.DataFrame:
        """Get latest scan results for a specific strategy."""
        df = self.db.get_latest_scan_results()
        if df.empty:
            return df
        return df[df["strategy_id"] == strategy_id].sort_values(
            "score", ascending=False
        ).reset_index(drop=True)

    def get_confluence(self, min_strategies: int = 2) -> pd.DataFrame:
        """Find stocks appearing in multiple strategies."""
        df = self.db.get_latest_scan_results()
        if df.empty:
            return pd.DataFrame()

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
